Chatterbox TTS — API quick start

This document shows how to run the API server locally, load models, and test endpoints.

Prerequisites
- Python 3.10+ (or newer)
- Enough disk space for model files (hundreds of MB)

Quick setup (first time)

```bash
# from repository root
# create a local virtualenv if you don't already have one
python -m venv .venv
# activate the venv (Git Bash)
source .venv/Scripts/activate
python -V

# upgrade pip tooling and install dependencies (CPU default)
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
# install the package in editable mode so local changes are picked up
pip install -e .
```

Notes on CUDA vs CPU
- The default `requirements.txt` installs CPU-only PyTorch wheels. If you have an NVIDIA GPU and want CUDA acceleration, edit `requirements.txt` to uncomment the CUDA index and `+cu121` wheel pins (or use the appropriate CUDA version for your machine) before running `pip install -r requirements.txt`.


1) Activate virtualenv (Git Bash)

```bash
# from repository root
source .venv/Scripts/activate
python -V
```

2) Start the API server

```bash
# run from the src folder so imports resolve
cd src
python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
```

Keep this terminal open while the server runs.

3) Trigger background model load (in another terminal)

```bash
curl -X POST "http://127.0.0.1:8000/load_models"
```

4) Poll status until models are loaded

```bash
curl http://127.0.0.1:8000/status
# look for {"loaded": true, ...}
```

5) Test synth endpoint

```bash
# simple text-only synthesis
curl -X POST "http://127.0.0.1:8000/synthesize" -F "text=Hello from API" -o out.wav

# with reference wav
curl -X POST "http://127.0.0.1:8000/synthesize" -F "text=Read this" -F "ref_audio=@/path/to/ref.wav;type=audio/wav" -o out_ref.wav
```

6) Use the browser UI (optional)

```bash
# serve static UI
python -m http.server 8080 --directory chatterbox_UI
# open http://127.0.0.1:8080
```

Notes and troubleshooting
- If you see CORS errors in the browser, ensure the server is running and restart uvicorn after any CORS configuration changes.
- If you see ImportError while loading models (e.g., missing packages), install the missing package into the venv:
  python -m pip install <package-name>
- For production, restrict CORS origins in `src/api_server.py` and run uvicorn under a process manager (systemd, supervisor, or Windows service).

Contact
- If you need help debugging errors, copy/paste the uvicorn server terminal output and I can help interpret it.