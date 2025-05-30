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

def count_models_for_user(email: str, base_dir: str = "models") -> int:
    user_dir = os.path.join(base_dir, safe_email_dir(email))
    if not os.path.exists(user_dir):
        return 0
    model_files = [f for f in os.listdir(user_dir) if f.endswith('.keras')]
    return len(model_files)

def delete_model_and_scaler(email: str, model_id: str) -> bool:
    """
    Usuwa pliki modelu (.keras) i scalera (.pkl) na podstawie emaila i model_id.
    Zwraca True jeśli oba pliki zostały usunięte, False jeśli któregokolwiek nie znaleziono.
    """
    base_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    user_model_dir = os.path.join(base_dir, safe_email_dir(email))

    model_path = os.path.join(user_model_dir, f"{model_id}.keras")
    scaler_path = os.path.join(user_model_dir, f"{model_id}.pkl")

    model_exists = os.path.exists(model_path)
    scaler_exists = os.path.exists(scaler_path)

    if model_exists:
        os.remove(model_path)
    if scaler_exists:
        os.remove(scaler_path)

    return model_exists and scaler_exists
