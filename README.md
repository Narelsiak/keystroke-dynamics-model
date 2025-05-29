## Jak to uruchomić

### Wymagania
*   Python (3.8+)
*   `pip`

### Kroki
1.  **Pobierz kod:**
    ```bash
    git clone <URL_TWOJEGO_REPOZYTORIUM>
    cd <NAZWA_KATALOGU_PROJEKTU>
    ```

2.  **Przygotuj środowisko wirtualne:**
    ```bash
    # Utwórz
    python -m venv venv
    # Aktywuj (Windows)
    .\venv\Scripts\activate
    # Aktywuj (Linux/macOS)
    source venv/bin/activate
    ```

3.  **Zainstaluj zależności:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Skompiluj gRPC proto:**
    (Upewnij się, że `keystroke.proto` jest w głównym katalogu)
    ```bash
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. keystroke.proto
    ```

5.  **Uruchom serwer:**
    ```bash
    # Windows
    .\start.bat
    # Linux/macOS (lub bezpośrednio, np. python server.py)
    python <nazwa_głównego_pliku_serwera.py>
    ```
    Po uruchomieniu serwer nasłuchuje na porcie `50051`.