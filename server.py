import grpc
from concurrent import futures
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import keystroke_pb2
import keystroke_pb2_grpc

from utils.data import flatten_attempts_press_wait_only, augment_with_noise
from utils.utils import save_model_and_scaler, count_models_for_user, delete_model_and_scaler

class KeystrokeServiceServicer(keystroke_pb2_grpc.KeystrokeServiceServicer):
    def __init__(self):
        self.scaler = None
        self.autoencoder = None

    def build_autoencoder(self, input_dim):
        model = Sequential([
            
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(input_dim, activation='linear')
        ])
        model.compile(optimizer='adam', loss=tf.keras.losses.Huber())
        return model

    def Train(self, request, context):
        X = flatten_attempts_press_wait_only(request.attempts)

        # Zamiana na macierz numpy
        X = np.array(X)
        X_augmented = augment_with_noise(X, noise_level=0.05, count=3)

        # Skalowanie
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_augmented)

        # Budowa i trenowanie autoenkodera
        self.autoencoder = self.build_autoencoder(X_scaled.shape[1])

        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=100,
            shuffle=True,
            verbose=0
        )

        # Statystyki strat
        losses = history.history['loss']
        final_loss = losses[-1]
        min_loss = min(losses)
        max_loss = max(losses)
        avg_loss = sum(losses) / len(losses)
        std_loss = np.std(losses)
        unique_id = save_model_and_scaler(request.email, self.autoencoder, self.scaler)

        return keystroke_pb2.TrainResponse(
            message="Model trained successfully.",
            stats=keystroke_pb2.TrainStats(
                samples=len(X_scaled),
                finalLoss=final_loss,
                minLoss=min_loss,
                maxLoss=max_loss,
                avgLoss=avg_loss,
                stdLoss=std_loss,
            ),
            id=unique_id
        )


    def GetModelCount(self, request, context):
        email = request.email
        count = count_models_for_user(email)
        return keystroke_pb2.ModelCountResponse(count=count)
    
    def DeleteModel(self, request, context):
        email = request.email
        model_id = request.modelName  # <-- upewnij się, że to pole to modelName

        success = delete_model_and_scaler(email, model_id)
        return keystroke_pb2.DeleteModelResponse(success=success, message="Model deleted successfully." if success else "Model not found or could not be deleted.")
    
    def Evaluate(self, request, context):
        all_press_durations = []
        all_wait_durations = []

        for attempt in request.attempts:
            for kp in attempt.keyPresses:
                all_press_durations.append(kp.pressDuration)
                all_wait_durations.append(kp.waitDuration)

        def compute_stats(data):
            arr = np.array(data)
            return {
                "avg": float(arr.mean()) if arr.size else 0.0,
                "std": float(arr.std()) if arr.size else 0.0
            }

        press_stats = compute_stats(all_press_durations)
        wait_stats = compute_stats(all_wait_durations)

        results = []
        overall_anomalies = []

        for attempt in request.attempts:
            local_anomalies = []
            for kp in attempt.keyPresses:
                if abs(kp.pressDuration - press_stats["avg"]) > 2 * press_stats["std"]:
                    local_anomalies.append(
                        f"Key '{kp.value}' has unusual press duration: {kp.pressDuration}ms"
                    )
                if abs(kp.waitDuration - wait_stats["avg"]) > 2 * wait_stats["std"]:
                    local_anomalies.append(
                        f"Key '{kp.value}' has unusual wait duration: {kp.waitDuration}ms"
                    )

            is_anomalous = len(local_anomalies) > 0
            score = 0.0 if is_anomalous else 1.0

            overall_anomalies.extend(local_anomalies)

            results.append(
                keystroke_pb2.EvaluationAttempt(
                    keyPresses=[kp for kp in attempt.keyPresses],
                    isAnomalous=is_anomalous,
                    score=score,
                    message="; ".join(local_anomalies) if local_anomalies else "Normal"
                )
            )

        return keystroke_pb2.EvaluateResponse(
            message="Evaluation complete.",
            stats=keystroke_pb2.EvaluateStats(
                samples=len(request.attempts),
                pressAvg=press_stats["avg"],
                pressStd=press_stats["std"],
                waitAvg=wait_stats["avg"],
                waitStd=wait_stats["std"]
            ),
            results=results,
            anomalies=overall_anomalies
        )



    def Predict(self, request, context):
        if self.autoencoder is None or self.scaler is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details('Model not trained yet')
            return keystroke_pb2.PredictResponse()

        threshold = request.threshold if request.threshold > 0 else 3.0
        attempts = request.attempts
        if not attempts:
            return keystroke_pb2.PredictResponse()

        X = np.array([attempt.features for attempt in attempts])
        X_scaled = self.scaler.transform(X)
        reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        max_mse = max(mse.max(), threshold * 2)
        confidence = 100 * (1 - np.log1p(mse) / np.log1p(max_mse))
        confidence = np.clip(confidence, 0, 100)
        is_yours = mse < threshold

        results = []
        for i in range(len(attempts)):
            result = keystroke_pb2.PredictResult(
                is_yours=bool(is_yours[i]),
                confidence=float(confidence[i]),
                mse=float(mse[i])
            )
            results.append(result)

        return keystroke_pb2.PredictResponse(results=results)
        
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    keystroke_pb2_grpc.add_KeystrokeServiceServicer_to_server(KeystrokeServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
