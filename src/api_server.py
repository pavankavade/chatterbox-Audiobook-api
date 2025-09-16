from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import soundfile as sf
import tempfile
import threading
import time
import torch
from pathlib import Path
from pyngrok import ngrok
import uvicorn

app = FastAPI(title="Chatterbox TTS API")

# Allow local UI use (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    # For local development allow all origins. In production restrict this to your UI origin(s).
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model holder
MODEL = {
    "tts": None,
    "device": "cpu",
    "loaded": False,
    "last_requested_device": None,
    "note": None,
}


@app.on_event("startup")
def load_models():
    """Don't eagerly load heavy ML models during startup. The server will load models on-demand
    or when the `/load_models` endpoint is called. This keeps startup fast and avoids crashes if
    ML packages or checkpoints are not yet available.
    """
    MODEL["tts"] = None
    MODEL["device"] = "cpu"
    MODEL["loaded"] = False
    MODEL["loading"] = False
    MODEL["lock"] = threading.Lock()
    MODEL["last_requested_device"] = None
    MODEL["note"] = None
    print("Startup complete: models are not loaded. Call /load_models to start loading.")


def load_models_sync(device: str = "cpu"):
    """Synchronous model loader. This performs the heavy imports and model initialization.
    It's safe to call from a background thread.
    """
    ckpt_dir = Path("./ckpt")
    try:
        from chatterbox.tts import ChatterboxTTS

        print("Starting model load (this can take a while)...")
        if ckpt_dir.exists():
            tts = ChatterboxTTS.from_local(ckpt_dir, device)
        else:
            tts = ChatterboxTTS.from_pretrained(device)

        MODEL["tts"] = tts
        MODEL["device"] = device
        MODEL["loaded"] = True
        MODEL["loading"] = False
        print("✅ Models loaded successfully")
    except Exception as e:
        MODEL["tts"] = None
        MODEL["loaded"] = False
        MODEL["loading"] = False
        print("❌ Failed to load models:", e)
        raise


def start_background_load():
    with MODEL["lock"]:
        if MODEL.get("loaded"):
            return "already_loaded"
        if MODEL.get("loading"):
            return "already_loading"
        MODEL["loading"] = True

    def _worker():
        try:
            load_models_sync(MODEL.get("device", "cpu"))
        except Exception:
            pass

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return "started"


def ensure_model():
    if not MODEL.get("loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    ref_audio: UploadFile | None = File(None),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
):
    """Synthesize speech for `text`. Optionally provide `ref_audio` (wav) to condition voice."""
    ensure_model()
    tts = MODEL["tts"]

    # If ref_audio provided, save to temp and prepare conditionals
    conds = tts.conds
    if ref_audio is not None:
        contents = await ref_audio.read()
        try:
            # read into numpy via soundfile
            data, sr = sf.read(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

        # write to a temporary file for prepare_conditionals which expects a path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_name = tf.name
            tf.write(contents)

        try:
            conds = tts.prepare_conditionals(tmp_name, exaggeration=exaggeration)
        finally:
            try:
                Path(tmp_name).unlink()
            except Exception:
                pass

    with torch.no_grad():
        wav_tensor = tts.generate(
            text=text,
            conds=conds,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

    # wav_tensor is (1, n_samples)
    wav = wav_tensor.squeeze(0).cpu().numpy()

    buf = io.BytesIO()
    sf.write(buf, wav, tts.sr, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


@app.post("/load_models")
async def load_models_endpoint(background: bool = True, device: str | None = None, force: bool = False):
    """Trigger model loading. If called with background=true (default) it returns immediately
    while loading proceeds in a background thread. If background=false this call will block
    until loading completes (may take a long time).
    """
    # If a device is provided, set it before starting load
    if device is not None:
        desired = device.lower()
        MODEL["last_requested_device"] = desired
        note = None
        if desired == "cuda":
            if torch.cuda.is_available():
                MODEL["device"] = "cuda"
                note = "using cuda"
            else:
                MODEL["device"] = "cpu"
                note = "cuda requested but unavailable; using cpu"
        elif desired == "cpu":
            MODEL["device"] = "cpu"
            note = "cpu requested"
        else:
            note = f"unknown device '{desired}', keeping existing: {MODEL.get('device')}"
        MODEL["note"] = note

    if MODEL.get("loaded"):
        # If already loaded but force reload requested (e.g., to switch device), proceed
        if not force:
            return {"status": "already_loaded"}
        # mark as not loaded to allow reload
        MODEL["loaded"] = False

    if background:
        started = start_background_load()
        return {"status": started}
    else:
        # Synchronous load (blocking)
        try:
            load_models_sync(MODEL.get("device", "cpu"))
            return {"status": "loaded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    info = {
        "loaded": MODEL.get("loaded", False),
        "loading": MODEL.get("loading", False),
        "device": MODEL.get("device", "cpu"),
        "last_requested_device": MODEL.get("last_requested_device"),
        "note": MODEL.get("note"),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
        else:
            info["gpu_name"] = None
    except Exception:
        info["gpu_name"] = None
    return info

if __name__ == "__main__":
    port = 8000
    # Use the 'domain' parameter to specify your static ngrok domain
    public_url = ngrok.connect(port, "http", domain="umbrellaless-meghan-subovarian.ngrok-free.app").public_url
    print(f"Public URL: {public_url}")
    # It's good practice to bind to 0.0.0.0 to make it accessible within the container/VM
    uvicorn.run(app, host="0.0.0.0", port=port)