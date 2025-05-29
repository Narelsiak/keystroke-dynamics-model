import numpy as np

def flatten_attempts_press_wait_only(attempts):
    """
    Zwraca spłaszczoną macierz NumPy z naprzemiennych pressDuration/waitDuration.
    
    Args:
        attempts: lista prób (z protobuf)

    Returns:
        np.ndarray: kształt [num_attempts, num_features]
    """
    X = []

    for attempt in attempts:
        flat_features = []
        for kp in attempt.keyPresses:
            flat_features.append(kp.pressDuration)
            flat_features.append(getattr(kp, "waitDuration", 0.0))  # domyślnie 0 jeśli brak
        X.append(flat_features)

    return np.array(X)

def augment_with_noise(data: np.ndarray, noise_level: float = 0.05, count: int = 2) -> np.ndarray:
    """
    Rozszerza dane przez dodanie dwóch kopii z lekkim szumem Gaussa.
    
    :param data: Numpy array o wymiarze (n_samples, n_features)
    :param noise_level: Procentowy poziom szumu (np. 0.05 to 5%)
    :return: Nowy numpy array z oryginalnymi i zaszumionymi danymi
    """
    augmented = [data]

    for _ in range(count):
        noise = np.random.normal(loc=0.0, scale=noise_level, size=data.shape) * data
        augmented_sample = data + noise
        augmented.append(augmented_sample)

    return np.vstack(augmented)
