from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import soundfile as sf
import tempfile
import threading
import torch
from pathlib import Path
from pyngrok import ngrok
import uvicorn

app = FastAPI(title="Chatterbox TTS API")

# Allow local UI use (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production restrict to your UI origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model holder
MODEL = {
    "tts": None,
    "device": "cpu",
    "loaded": False,
    "loading": False,
    "lock": threading.Lock(),
    "last_requested_device": None,
    "note": None,
}


@app.on_event("startup")
def load_models_startup():
    """Try to load models automatically when the server starts."""
    if torch.cuda.is_available():
        MODEL["device"] = "cuda"
        MODEL["note"] = "GPU available, using CUDA"
    else:
        MODEL["device"] = "cpu"
        MODEL["note"] = "CUDA not available, using CPU"

    print(f"🚀 Starting server... Loading models in background on {MODEL['device']}.")
    start_background_load()



def load_models_sync(device: str = "cpu"):
    """Synchronous model loader. Performs heavy imports and model initialization."""
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
    """Ensure the model is loaded. If not, load it synchronously."""
    if not MODEL.get("loaded"):
        if torch.cuda.is_available():
            MODEL["device"] = "cuda"
        else:
            MODEL["device"] = "cpu"
        print(f"⚠️ Model not loaded. Loading synchronously on {MODEL['device']}...")
        try:
            load_models_sync(MODEL.get("device"))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model load failed: {e}")


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    ref_audio: UploadFile | None = File(None),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
):
    """Synthesize speech for `text`. Optionally provide `ref_audio` (wav/mp3).
    If not provided, fallback to /notebooklmsamplevoice.mp3.
    """
    ensure_model()
    tts = MODEL["tts"]

    conds = tts.conds  # default conditionals

    # Case 1: user uploaded reference audio
    if ref_audio is not None:
        contents = await ref_audio.read()
        try:
            data, sr = sf.read(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid audio file: {e}")

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

    # Case 2: fallback to hardcoded sample
    else:
        default_ref = Path(__file__).parent / "notebooklmsamplevoice.mp3"
        if not default_ref.exists():
            raise HTTPException(status_code=500, detail="Default reference file not found")
        try:
            conds = tts.prepare_conditionals(str(default_ref), exaggeration=exaggeration)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to use default reference: {e}")

    # Generate audio
    with torch.no_grad():
        wav_tensor = tts.generate(
            text=text,
            conds=conds,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

    wav = wav_tensor.squeeze(0).cpu().numpy()
    buf = io.BytesIO()
    sf.write(buf, wav, tts.sr, format="WAV")
    buf.seek(0)

    return StreamingResponse(buf, media_type="audio/wav")


@app.post("/load_models")
async def load_models_endpoint(background: bool = True, device: str | None = None, force: bool = False):
    """Trigger model loading manually if needed."""
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

    if MODEL.get("loaded") and not force:
        return {"status": "already_loaded"}

    if background:
        started = start_background_load()
        return {"status": started}
    else:
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
    public_url = ngrok.connect(
        port, "http", domain="variolous-londa-patiently.ngrok-free.dev"
    ).public_url
    print(f"Public URL: {public_url}")
    uvicorn.run(app, host="0.0.0.0", port=port)
