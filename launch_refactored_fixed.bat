@echo off
title Chatterbox Audiobook Studio - Refactored (Fixed)
echo.
echo 🎧 Chatterbox Audiobook Studio - Refactored Edition (Fixed Imports)
echo =====================================================================
echo.
echo 🚀 Launching with fixed import structure...
echo 💡 This version supports both module and app.py execution
echo.

cd /d "%~dp0refactor"

if not exist "app.py" (
    echo ❌ Error: app.py not found in refactor directory
    pause
    exit /b 1
)

echo 🔄 Activating virtual environment...
if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
    echo ✅ Virtual environment activated
) else (
    echo ⚠️  Warning: Virtual environment not found, using system Python
)

echo.
echo 🎉 Starting Chatterbox Audiobook Studio (Refactored)...
echo 🌐 Will be available at: http://localhost:7860
echo.
echo 💡 Features available:
echo    🎤 Text-to-Speech with GPU acceleration
echo    📚 Voice Library management
echo    📖 Single Voice Audiobook generation ✨ NEW
echo    🎭 Multi-Voice character assignment
echo    🎬 Production Studio for editing
echo    🎧 Listen & Edit mode
echo.

python app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Application error (Exit code: %ERRORLEVEL%)
    echo.
    echo 🔧 Troubleshooting tips:
    echo    - Make sure virtual environment is set up correctly
    echo    - Check that all dependencies are installed
    echo    - Try running: pip install -r requirements.txt
    echo.
)

echo.
echo 🔚 Press any key to close...
pause >nul 