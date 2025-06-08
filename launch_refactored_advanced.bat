@echo off
title Chatterbox Audiobook Studio - Refactored Edition (Advanced)
echo.
echo =================================================================
echo 🎧 Chatterbox Audiobook Studio - Refactored Edition (Advanced)
echo =================================================================
echo.
echo Select launch option:
echo.
echo 1. 🚀 Standard Launch (localhost:7860)
echo 2. 🌐 Public Share Launch (shareable link)
echo 3. 🔧 Debug Mode Launch
echo 4. 🧪 Test Modules First
echo 5. 📊 Custom Port Launch
echo 6. ❌ Exit
echo.
set /p choice="Enter your choice (1-6): "

cd /d "%~dp0refactor"

if not exist "app.py" (
    echo ❌ Error: app.py not found in refactor directory
    pause
    exit /b 1
)

if "%choice%"=="1" (
    echo 🚀 Standard launch...
    python app.py
) else if "%choice%"=="2" (
    echo 🌐 Public share launch...
    python app.py --share
) else if "%choice%"=="3" (
    echo 🔧 Debug mode launch...
    python app.py --debug
) else if "%choice%"=="4" (
    echo 🧪 Testing modules first...
    python app.py --test-modules
    if %ERRORLEVEL% equ 0 (
        echo ✅ Tests passed! Launching app...
        python app.py
    ) else (
        echo ❌ Tests failed! Check the errors above.
        pause
        exit /b 1
    )
) else if "%choice%"=="5" (
    set /p port="Enter port number (default 7860): "
    if "%port%"=="" set port=7860
    echo 📊 Launching on port %port%...
    python app.py --port %port%
) else if "%choice%"=="6" (
    exit /b 0
) else (
    echo ❌ Invalid choice. Using standard launch...
    python app.py
)

if %ERRORLEVEL% neq 0 (
    echo.
    echo ❌ Application error (Exit code: %ERRORLEVEL%)
    echo 💡 Try the debug mode option for more information
    echo.
)

echo.
echo 🔚 Press any key to close...
pause >nul 