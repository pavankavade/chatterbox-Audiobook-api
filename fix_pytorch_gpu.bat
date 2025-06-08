@echo off
title Fix PyTorch GPU Installation
echo.
echo =================================================================
echo 🔧 PyTorch GPU Installation Fix
echo =================================================================
echo.
echo 🔍 Current Issue: PyTorch 2.6.0+cpu (CPU-only version installed)
echo 🎯 Solution: Install PyTorch with CUDA support for your RTX 3090
echo.
echo 📋 This will:
echo    ✅ Uninstall CPU-only PyTorch
echo    ✅ Install PyTorch with CUDA 12.1 support
echo    ✅ Fix xFormers compatibility
echo    ✅ Enable GPU acceleration for TTS
echo.
echo ⚠️  WARNING: This will modify your Python environment
echo 💡 Make sure you're in the correct virtual environment
echo.
set /p confirm="Continue with PyTorch GPU installation? (y/n): "

if /i "%confirm%" neq "y" (
    echo Installation cancelled.
    pause
    exit /b 0
)

echo.
echo 🔄 Starting PyTorch GPU installation...
echo.

echo 📂 Activating virtual environment...
call venv\Scripts\activate

echo.
echo 🗑️  Step 1: Uninstalling CPU-only PyTorch...
pip uninstall torch torchvision torchaudio -y

echo.
echo 📥 Step 2: Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 🔧 Step 3: Testing CUDA availability...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✅ PyTorch GPU installation completed successfully!
    echo 🚀 Your RTX 3090 should now be available for TTS generation
    echo.
    echo 🎯 Next steps:
    echo    1. Restart the refactored app
    echo    2. Look for "Selected TTS Device: cuda" in the output
    echo    3. TTS generation should now show "using GPU"
    echo.
) else (
    echo.
    echo ❌ Installation encountered an issue
    echo 💡 Try running this script as administrator
    echo.
)

echo 🔚 Press any key to continue...
pause >nul 