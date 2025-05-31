import grpc
from concurrent import futures
import time
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from urllib.parse import quote_plus
import os
import pickle

import keystroke_pb2
import keystroke_pb2_grpc

from utils.data import flatten_attempts_press_wait_only, augment_with_noise
from utils.utils import save_model_and_scaler, count_models_for_user, delete_model_and_scaler, safe_email_dir

class KeystrokeServiceServicer(keystroke_pb2_grpc.KeystrokeServiceServicer):
    def __init__(self):
        self.scaler = None
        self.autoencoder = None
        self.model_base_path = "models"

    def load_model_and_scaler(self, email: str, model_name: str):

        model_dir = safe_email_dir(email)
        scaler_path = os.path.join("models", model_dir, model_name + ".keras")
        model_path = os.path.join("models", model_dir, model_name + ".h5")

        print(scaler_path, model_path)
        if not os.path.exists(scaler_path) or not os.path.exists(model_path):
            raise FileNotFoundError("Model or scaler file not found")

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        self.autoencoder = load_model(model_path)

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

        print("CIPA")
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
        all_press_durations_global = []
        all_wait_durations_global = []

        if request.attempts:
            for attempt in request.attempts:
                for kp in attempt.keyPresses:
                    all_press_durations_global.append(kp.pressDuration)

                    all_wait_durations_global.append(kp.waitDuration)

        def compute_stats(data, label=""):
            arr = np.array(data)
            print(f"DEBUG compute_stats for '{label}': array={arr}, size={arr.size}")
            if not arr.size:
                return {"avg": 0.0, "std": 0.0, "samples": 0}
            
            return {
                "avg": float(arr.mean()),
                "std": float(arr.std()),
                "samples": int(arr.size)
            }

        global_press_stats = compute_stats(all_press_durations_global, "global_press")
        global_wait_stats = compute_stats(all_wait_durations_global, "global_wait")

        max_keypresses = 0
        if request.attempts:
            attempt_lengths = [len(attempt.keyPresses) for attempt in request.attempts if attempt.keyPresses]
            if attempt_lengths:
                max_keypresses = max(attempt_lengths)
        
        press_stats_by_position = []
        wait_stats_by_position = []

        if max_keypresses > 0:
            press_durations_by_position = [[] for _ in range(max_keypresses)]
            wait_durations_by_position = [[] for _ in range(max_keypresses)]

            for attempt in request.attempts:
                for i, kp in enumerate(attempt.keyPresses):
                    if i < max_keypresses:
                        press_durations_by_position[i].append(kp.pressDuration)
                        wait_durations_by_position[i].append(kp.waitDuration)
            
            for i, durations_at_pos in enumerate(press_durations_by_position):
                press_stats_by_position.append(compute_stats(durations_at_pos, f"press_pos_{i+1}"))

            for i, durations_at_pos in enumerate(wait_durations_by_position):
                wait_stats_by_position.append(compute_stats(durations_at_pos, f"wait_pos_{i+1}"))

        results = []
        overall_anomalies = []
        epsilon = 1e-6

        for attempt_idx, attempt in enumerate(request.attempts):
            local_anomalies = []
            
            if not press_stats_by_position or not attempt.keyPresses:
                message_text = "Normal"
                if not attempt.keyPresses:
                    message_text = "Normal (no keypresses in attempt)"
                elif not press_stats_by_position and len(request.attempts) <=1 :
                     message_text = "Normal (insufficient data for per-position anomaly detection; single attempt)"
                elif not press_stats_by_position :
                     message_text = "Normal (insufficient data for per-position anomaly detection)"


                is_anomalous = False # Domyślnie nie jest anomalią, jeśli nie można ocenić
                score = 1.0
            else:
                for kp_idx, kp in enumerate(attempt.keyPresses):
                    if kp_idx < len(press_stats_by_position) and kp_idx < len(wait_stats_by_position):
                        current_press_stats = press_stats_by_position[kp_idx]
                        current_wait_stats = wait_stats_by_position[kp_idx]
                        if current_press_stats["samples"] > 1 and current_press_stats["std"] > epsilon:
                            if abs(kp.pressDuration - current_press_stats["avg"]) > 2 * current_press_stats["std"]:
                                local_anomalies.append(
                                    f"Key '{kp.value}' (pos {kp_idx+1}) has unusual press duration: {kp.pressDuration}ms (avg: {current_press_stats['avg']:.1f}, std: {current_press_stats['std']:.1f})"
                                )
                        elif current_press_stats["samples"] > 0 and current_press_stats["std"] <= epsilon:
                            if abs(kp.pressDuration - current_press_stats["avg"]) > epsilon:
                                local_anomalies.append(
                                    f"Key '{kp.value}' (pos {kp_idx+1}) has unusual press duration (std~0): {kp.pressDuration}ms (avg: {current_press_stats['avg']:.1f})"
                                )
                        
                        if current_wait_stats["samples"] > 1 and current_wait_stats["std"] > epsilon:
                            if abs(kp.waitDuration - current_wait_stats["avg"]) > 2 * current_wait_stats["std"]:
                                local_anomalies.append(
                                    f"Key '{kp.value}' (pos {kp_idx+1}) has unusual wait duration: {kp.waitDuration}ms (avg: {current_wait_stats['avg']:.1f}, std: {current_wait_stats['std']:.1f})"
                                )
                        elif current_wait_stats["samples"] > 0 and current_wait_stats["std"] <= epsilon:
                            if abs(kp.waitDuration - current_wait_stats["avg"]) > epsilon:
                                local_anomalies.append(
                                    f"Key '{kp.value}' (pos {kp_idx+1}) has unusual wait duration (std~0): {kp.waitDuration}ms (avg: {current_wait_stats['avg']:.1f})"
                                )

                is_anomalous = len(local_anomalies) > 0
                score = 0.0 if is_anomalous else 1.0
                message_text = "; ".join(local_anomalies) if local_anomalies else "Normal"

            overall_anomalies.extend(local_anomalies)
            results.append(
                keystroke_pb2.EvaluationAttempt(
                    keyPresses=list(attempt.keyPresses),
                    isAnomalous=is_anomalous,
                    score=score,
                    message=message_text
                )
            )

        return keystroke_pb2.EvaluateResponse(
            message="Evaluation complete.",
            stats=keystroke_pb2.EvaluateStats(
                samples=len(request.attempts),
                pressAvg=global_press_stats["avg"],
                pressStd=global_press_stats["std"],
                waitAvg=global_wait_stats["avg"],
                waitStd=global_wait_stats["std"]
            ),
            results=results,
            anomalies=overall_anomalies
        )



    def Predict(self, request, context):
        try:
            self.load_model_and_scaler(request.email, request.modelName)
        except Exception as e:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(str(e))
            return keystroke_pb2.PredictResponse()
        
        # Tu używamy progu (threshold) z request, jeśli jest, inaczej domyślnie 3.0
        threshold = request.threshold if hasattr(request, 'threshold') and request.threshold > 0 else 3.0

        # W proto mamy pojedynczy attempt z keyPresses, więc pakujemy w listę, żeby dalej działać z pętlą
        attempts = [request.attempt] if request.HasField('attempt') else []
        if not attempts:
            return keystroke_pb2.PredictResponse()

        # Zakładamy, że masz funkcję ekstrakcji cech z attempt.keyPresses:
        # np. każdemu attempt przypisujemy tablicę cech np. length, avg_press, etc.
        features_list = []
        for attempt in attempts:
            features = self.extract_features_from_keypresses(attempt.keyPresses)
            features_list.append(features)

        X = np.array(features_list)
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
                confidence=float(confidence[i]),   # tutaj similarity w %
                mse=float(mse[i])                  # tutaj error
            )
            results.append(result)

        return keystroke_pb2.PredictResponse(results=results)

    # Przykładowa funkcja ekstrakcji cech (dopasuj do swojego przypadku)
    def extract_features_from_keypresses(self, keypresses):
        # Przykład: konwersja keyPresses do prostego wektora cech
        durations = [kp.pressDuration for kp in keypresses]
        waits = [kp.waitDuration for kp in keypresses]
        # Przykładowe cechy: średnia długość naciśnięcia i średni czas między naciśnięciami
        avg_press = np.mean(durations) if durations else 0
        avg_wait = np.mean(waits) if waits else 0
        return np.array([avg_press, avg_wait])

        
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
