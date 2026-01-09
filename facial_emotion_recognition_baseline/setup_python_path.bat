@echo off
echo Setting up Python PATH permanently...

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python314
set PYTHON_SCRIPTS=%LOCALAPPDATA%\Programs\Python\Python314\Scripts

echo Adding Python to PATH...
setx PATH "%PATH%;%PYTHON_PATH%;%PYTHON_SCRIPTS%"

echo.
echo Python PATH setup complete!
echo.
echo Please close and reopen your PowerShell terminal for changes to take effect.
echo.
echo Then you can run:
echo   python --version
echo   pip --version
echo.
pause