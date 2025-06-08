@echo off
echo ====================================
echo   CHATTERBOX AUDIOBOOK STUDIO
echo   REFACTORED VERSION LAUNCHER
echo   Port: 7682 (Testing Alongside Original)
echo ====================================

REM Navigate to the project directory
cd /d "H:\CurserProjects\chatterbox-Audiobook"

echo.
echo Starting REFACTORED Chatterbox Audiobook Studio...
echo Original system can run on port 7860
echo Refactored system will run on port 7682
echo.

REM Run the refactored gradio app
python refactor\python app.py

echo.
echo Refactored Chatterbox Audiobook Studio has stopped.
pause 