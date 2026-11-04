@echo off
cd /d "%~dp0"
echo Checking requirements (the first run takes a few minutes)...
python -m pip install -q -r requirements.txt
if errorlevel 1 (
  echo.
  echo Could not install requirements. Install Python 3.10+ from python.org
  echo and tick "Add Python to PATH" during setup, then try again.
  pause
  exit /b 1
)
python -m vidtextswap %*
echo.
pause
