import os
import joblib
import uuid

def safe_email_dir(email: str) -> str:
    return email.replace("@", "_at_").replace(".", "_dot_")

def save_model_and_scaler(email: str, model, scaler) -> None:
    """
    Zapisuje model i scaler w ../models/[safe_email]/model
    (czyli katalog models/ obok katalogu utils/)
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    user_model_dir = os.path.join(base_dir, safe_email_dir(email))
    os.makedirs(user_model_dir, exist_ok=True)

    unique_id = uuid.uuid4().hex[:8]
    model_filename = f"{unique_id}.keras"
    scaler_filename = f"{unique_id}.pkl"

    model_path = os.path.join(user_model_dir, model_filename)
    scaler_path = os.path.join(user_model_dir, scaler_filename)

    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    return unique_id

