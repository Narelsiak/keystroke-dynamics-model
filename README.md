compile grpc:
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. keystroke.proto