Chatterbox TTS - Simple UI

Open `index.html` in a browser (you can use a static file server or open the file directly). The UI will attempt to call the API at `http://127.0.0.1:8000/synthesize`.

If the browser blocks calls due to mixed content or CORS, run a simple static server (Python):

```bash
# from repository root
python -m http.server 8080 --directory chatterbox_UI
# then open http://127.0.0.1:8080 in your browser
```

Features
- Paste text to synthesize
- Optionally upload a reference WAV to condition the voice
- Adjust exaggeration, cfg weight and temperature sliders
- Play and download the returned WAV

Notes
- CORS: If your API is on a different host/port, enable CORS on the FastAPI server (I can add this) or run the UI from the same origin.
- Models: The API must be running and the models loaded (or accessible) for synth requests to succeed.
 
Model loading
- To speed up initial behaviour, the API now loads models on-demand. Before making synth requests you can trigger loading with:

```bash
curl -X POST "http://127.0.0.1:8000/load_models"
```

The endpoint returns immediately while loading happens in the background. Check progress with:

```bash
curl http://127.0.0.1:8000/status
```

Wait until `loaded: true` before calling `/synthesize` for best results.
