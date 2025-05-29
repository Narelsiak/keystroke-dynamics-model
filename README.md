create venv
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

compile grpc:
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. keystroke.proto

run server:
.\start.bat