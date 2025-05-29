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
from utils.utils import save_model_and_scaler, count_models_for_user

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
                final_loss=final_loss,
                min_loss=min_loss,
                max_loss=max_loss,
                avg_loss=avg_loss,
                std_loss=std_loss,
            ),
            id=unique_id
        )


    def GetModelCount(self, request, context):
        email = request.email
        count = count_models_for_user(email)
        return keystroke_pb2.ModelCountResponse(count=count)
    
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

class LoggingInterceptor(grpc.ServerInterceptor):
    def intercept_service(self, continuation, handler_call_details):
        def log_and_call(request, context):
            print("Interceptor got request:", request)
            return continuation(handler_call_details).unary_unary(request, context)
        return grpc.unary_unary_rpc_method_handler(log_and_call)
        
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
