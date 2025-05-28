@echo off
REM dev_server.bat - A simple wrapper script to run your gRPC server with nodemon

REM Check if nodemon is installed
where nodemon >nul 2>nul
IF ERRORLEVEL 1 (
    echo nodemon is not installed. Installing nodemon...
    npm install -g nodemon
)

REM Run the server with nodemon
echo Starting gRPC Auth server with auto-reload...
nodemon --exec "python server.py" --ext py --ignore *.pyc --ignore tests\ --ignore __pycache__\
pause
