@echo off
echo ========================================
echo   Chatterbox TTS - Installation Setup
echo ========================================
echo.
echo This will install Chatterbox TTS in a virtual environment
echo to keep it isolated from other Python projects.
echo.
echo Requirements:
echo - Python 3.10 or higher
echo - NVIDIA GPU with CUDA support (recommended)
echo - Git (if you want to pull updates)
echo.
echo Current directory: %CD%
echo.
pause

echo.
echo [1/8] Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

echo.
echo [2/8] Checking if we're in the correct directory...
if not exist "pyproject.toml" (
    echo ERROR: pyproject.toml not found!
    echo Please make sure you're running this from the chatterbox repository root.
    echo Expected files: pyproject.toml, gradio_tts_app.py, src/chatterbox/
    pause
    exit /b 1
)

if not exist "src\chatterbox" (
    echo ERROR: src\chatterbox directory not found!
    echo Please make sure you're in the correct chatterbox repository.
    pause
    exit /b 1
)

echo Repository structure verified ✓

echo.
echo [3/8] Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q venv
)
python -m venv venv

echo.
echo [4/8] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [5/8] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [6/8] Installing compatible PyTorch with CUDA support...
echo This may take a while (downloading ~2.5GB)...
echo Installing PyTorch 2.4.1 + torchvision 0.19.1 (compatible versions)...
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

echo.
echo [7/8] Installing Chatterbox TTS and dependencies...
pip install -e .
pip install gradio
pip install numpy==1.26.0 --force-reinstall

echo.
echo [8/8] Testing installation...
echo Testing PyTorch and CUDA...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

if %errorlevel% neq 0 (
    echo WARNING: PyTorch test failed. Trying to fix torchvision compatibility...
    pip uninstall torchvision -y
    pip install torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
    echo Retesting...
    python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
)

echo.
echo Testing Chatterbox import...
python -c "from chatterbox.tts import ChatterboxTTS; print('Chatterbox TTS imported successfully!')"

if %errorlevel% neq 0 (
    echo WARNING: Chatterbox import failed. This might be a dependency issue.
    echo The installation will continue, but you may need to troubleshoot.
    echo Common fixes:
    echo 1. Run install.bat again
    echo 2. Check NVIDIA drivers are up to date
    echo 3. Restart your computer
)

echo.
echo ========================================
echo        Installation Complete!
echo ========================================
echo.
echo Virtual environment created at: %CD%\venv
echo.

echo Final system check...
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ========================================
echo           Ready for Audiobooks!
echo ========================================
echo.
echo To start Chatterbox TTS:
echo 1. Run launch_chatterbox.bat (recommended)
echo 2. Or manually: venv\Scripts\activate.bat then python gradio_tts_app.py
echo.
echo Perfect for:
echo - Voice cloning for audiobook narration
echo - Multiple character voices
echo - Consistent voice quality across chapters
echo - Professional audiobook production
echo.
echo Installation finished successfully!
pause 