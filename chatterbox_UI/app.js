const API_URL = "https://umbrellaless-meghan-subovarian.ngrok-free.app";

const textEl = document.getElementById('text');
const refEl = document.getElementById('ref_audio');
const synthBtn = document.getElementById('synthesize');
const statusEl = document.getElementById('status');
const player = document.getElementById('player');
const downloadBtn = document.getElementById('download');
const loadBtn = document.getElementById('load_models');
const exaggerationEl = document.getElementById('exaggeration');
const cfgEl = document.getElementById('cfg_weight');
const tempEl = document.getElementById('temperature');
const useGpuEl = document.getElementById('use_gpu');
const deviceInfoEl = document.getElementById('device_info');

let lastBlobUrl = null;

// Define the header once to avoid repetition
const ngrokHeaders = {
  'ngrok-skip-browser-warning': 'true'
};

function setStatus(msg) {
  statusEl.textContent = msg;
}

synthBtn.addEventListener('click', async () => {
  const text = textEl.value.trim();
  if (!text) {
    setStatus('Please enter text to synthesize');
    return;
  }
  setStatus('Preparing...');
  synthBtn.disabled = true;
  downloadBtn.disabled = true;

  const loaded = await ensureModelsLoaded();
  if (!loaded) {
    setStatus('Models not available. Try clicking Load Models.');
    synthBtn.disabled = false;
    return;
  }

  setStatus('Sending request...');

  const fd = new FormData();
  fd.append('text', text);
  fd.append('exaggeration', exaggerationEl.value);
  fd.append('cfg_weight', cfgEl.value);
  fd.append('temperature', tempEl.value);

  if (refEl.files.length > 0) {
    fd.append('ref_audio', refEl.files[0]);
  }

  try {
    // NOTE: FormData requests don't let you set a Content-Type header manually,
    // but custom headers like ngrok's are fine.
    const res = await fetch(`${API_URL}/synthesize`, {
      method: 'POST',
      headers: ngrokHeaders, // <-- FIX ADDED HERE
      body: fd
    });
    if (!res.ok) {
      const txt = await res.text();
      setStatus('Error: ' + res.status + ' ' + txt);
      synthBtn.disabled = false;
      return;
    }

    const blob = await res.blob();
    if (lastBlobUrl) {
      URL.revokeObjectURL(lastBlobUrl);
    }
    const url = URL.createObjectURL(blob);
    lastBlobUrl = url;
    player.src = url;
    player.play().catch(() => {});
    downloadBtn.disabled = false;
    downloadBtn.onclick = () => {
      const a = document.createElement('a');
      a.href = url;
      a.download = 'chatterbox_output.wav';
      a.click();
    };
    setStatus('Playback ready');
  } catch (err) {
    setStatus('Network or server error: ' + err.message);
  } finally {
    synthBtn.disabled = false;
  }
});


async function ensureModelsLoaded() {
  try {
    const r = await fetch(`${API_URL}/status`, { headers: ngrokHeaders }); // <-- FIX ADDED HERE
    if (!r.ok) { return false; } // Handle errors if the server isn't ok
    const j = await r.json();
    if (deviceInfoEl) {
      const parts = [];
      if (j.device) parts.push(`device: ${j.device}`);
      if (typeof j.cuda_available !== 'undefined') parts.push(`cuda_available: ${j.cuda_available}`);
      if (j.gpu_name) parts.push(`gpu: ${j.gpu_name}`);
      if (j.note) parts.push(j.note);
      deviceInfoEl.textContent = `(${parts.join(' · ')})`;
    }
    if (j.loaded) return true;

    setStatus('Models not loaded, starting background load...');
    const desiredDevice = useGpuEl && useGpuEl.checked ? 'cuda' : 'cpu';
    await fetch(`${API_URL}/load_models?device=${encodeURIComponent(desiredDevice)}&force=true`, { method: 'POST', headers: ngrokHeaders }); // <-- FIX ADDED HERE
    
    for (let i = 0; i < 600; i++) {
      await new Promise(res => setTimeout(res, 1000));
      const s = await fetch(`${API_URL}/status`, { headers: ngrokHeaders }); // <-- FIX ADDED HERE
      if (s.ok) {
        const js = await s.json();
        if (deviceInfoEl) {
          const parts = [];
          if (js.device) parts.push(`device: ${js.device}`);
          if (typeof js.cuda_available !== 'undefined') parts.push(`cuda_available: ${js.cuda_available}`);
          if (js.gpu_name) parts.push(`gpu: ${js.gpu_name}`);
          if (js.note) parts.push(js.note);
          deviceInfoEl.textContent = `(${parts.join(' · ')})`;
        }
        if (js.loaded) {
          setStatus('Models loaded.');
          return true;
        }
        setStatus('Loading models... (' + (i + 1) + 's)');
      }
    }
    setStatus('Model loading timed out.');
    return false;
  } catch (e) {
    console.warn(e);
    setStatus('An error occurred while checking model status: ' + e.message);
  }
  return false;
}

loadBtn.addEventListener('click', async () => {
  setStatus('Starting background model load...');
  try {
    const desiredDevice = useGpuEl && useGpuEl.checked ? 'cuda' : 'cpu';
    await fetch(`${API_URL}/load_models?device=${encodeURIComponent(desiredDevice)}&force=true`, { method: 'POST', headers: ngrokHeaders }); // <-- FIX ADDED HERE
    
    for (let i = 0; i < 10; i++) {
      await new Promise(r => setTimeout(r, 500));
      const s = await fetch(`${API_URL}/status`, { headers: ngrokHeaders }); // <-- FIX ADDED HERE
      if (s.ok) {
        const js = await s.json();
        setStatus('Loading models: loading=' + js.loading + ' loaded=' + js.loaded);
        if (deviceInfoEl && js.device) deviceInfoEl.textContent = `(server device: ${js.device})`;
        if (js.loaded) {
          setStatus('Models loaded');
          return;
        }
      }
    }
    setStatus('Background load started. Check status again in a moment.');
  } catch (e) {
    setStatus('Failed to start loading: ' + e.message);
  }
});

(async function ping() {
  try {
    const r = await fetch(`${API_URL}/docs`, { headers: ngrokHeaders }); // <-- FIX ADDED HERE
    if (r.ok) setStatus('API reachable. Please load models.');
  } catch (e) {
    setStatus(`API not reachable at ${API_URL}`);
  }
})();