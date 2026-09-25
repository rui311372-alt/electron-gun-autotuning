@echo off
cd /d "%~dp0"
python -X utf8 run_offline.py
if errorlevel 1 (
  echo Run failed. Please check the error above and the README.
) else (
  echo Completed. Results are in the results folder.
)
pause
