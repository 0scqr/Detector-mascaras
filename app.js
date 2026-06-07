/**
 * app.js — Detector de Máscaras Faciales
 * Inferencia 100% en el navegador:
 *  · BlazeFace  →  detección de rostros
 *  · TFLite.js  →  clasificación con/sin máscara (carga mask_detector_trained.tflite)
 */

// ============================================================
// Estado Global
// ============================================================
const state = {
  stream:          null,
  cameraActive:    false,
  continuousMode:  false,
  scanInterval:    null,
  scanIntervalMs:  600,
  isProcessing:    false,
  modelLoaded:     false,
  faceModelLoaded: false,

  maskModel:    null,   // TFLite classifier
  faceDetector: null,   // BlazeFace
};

// ============================================================
// Referencias del DOM
// ============================================================
const video          = document.getElementById('webcam-video');
const canvas         = document.getElementById('detection-canvas');
const ctx            = canvas.getContext('2d');
const placeholder    = document.getElementById('video-placeholder');
const loadingOverlay = document.getElementById('loading-overlay');
const btnToggle      = document.getElementById('btn-toggle-camera');
const btnSingle      = document.getElementById('btn-single-capture');
const btnCameraText  = document.getElementById('btn-camera-text');
const switchCont     = document.getElementById('switch-continuous');
const statusBadge    = document.getElementById('system-status-badge');
const noResults      = document.getElementById('no-results-placeholder');
const resultsList    = document.getElementById('results-list');
const statPeople     = document.getElementById('stat-people-count');
const statLatency    = document.getElementById('stat-latency');
const statModel      = document.getElementById('stat-model-status');

// ============================================================
// Tema (Claro / Oscuro)
// ============================================================
const htmlEl     = document.documentElement;
const themeBtn   = document.getElementById('theme-toggle');
const themeLabel = document.getElementById('theme-label');

function setTheme(theme) {
  htmlEl.setAttribute('data-theme', theme);
  themeLabel.textContent = theme === 'dark' ? 'Oscuro' : 'Claro';
  localStorage.setItem('mask-detector-theme', theme);
}

function initTheme() {
  const saved = localStorage.getItem('mask-detector-theme');
  const sys   = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  setTheme(saved || sys);
  window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
    if (!localStorage.getItem('mask-detector-theme')) setTheme(e.matches ? 'light' : 'dark');
  });
}

// ============================================================
// Carga de Modelos
// ============================================================
async function loadModels() {
  setStatus('processing', 'Cargando modelos...');
  statModel.textContent = 'Cargando...';
  statModel.className   = 'stat-value text-warning';

  const errors = [];

  // ── 1. BlazeFace (detector de rostros) ──────────────────────
  try {
    setStatus('processing', 'Cargando detector facial...');
    state.faceDetector    = await blazeface.load();
    state.faceModelLoaded = true;
    console.log('[TF.js] BlazeFace ✓');
  } catch (e) {
    console.error('[TF.js] BlazeFace ERROR:', e);
    errors.push('BlazeFace: ' + e.message);
  }

  // ── 2. Modelo TFLite de máscaras ────────────────────────────
  try {
    setStatus('processing', 'Cargando clasificador TFLite...');

    // tflite está expuesto por @tensorflow/tfjs-tflite como window.tflite
    if (typeof tflite === 'undefined') {
      throw new Error('tfjs-tflite no cargó correctamente desde el CDN.');
    }

    // Configurar la ruta del WASM de TFLite
    tflite.setWasmPath(
      'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/'
    );

    state.maskModel    = await tflite.loadTFLiteModel('./mask_detector_trained.tflite');
    state.modelLoaded  = true;
    console.log('[TF.js] Modelo TFLite ✓');

    // Warmup
    const dummy = tf.zeros([1, 224, 224, 3]);
    state.maskModel.predict(dummy);
    dummy.dispose();

  } catch (e) {
    console.error('[TFLite] ERROR:', e);
    errors.push('TFLite: ' + e.message);
  }

  // ── Resultado final ──────────────────────────────────────────
  if (state.modelLoaded && state.faceModelLoaded) {
    statModel.textContent = 'Listo ✓';
    statModel.className   = 'stat-value text-success';
    setStatus('online', 'Modelos listos');
  } else {
    const msg = errors.length ? errors.join(' | ') : 'Error desconocido';
    statModel.textContent = 'Error';
    statModel.className   = 'stat-value text-danger';
    setStatus('offline', 'Error de modelo');
    noResults.innerHTML   = `
      <div style="color:var(--danger);font-size:.9rem;line-height:1.5;">
        <strong>⚠️ Error cargando modelos</strong><br>
        <code style="font-size:.75rem;opacity:.8;">${msg}</code><br><br>
        <span style="opacity:.7;font-size:.8rem;">
          Asegúrate de que <code>mask_detector_trained.tflite</code>
          está en la raíz del proyecto y se subió a Vercel.
        </span>
      </div>`;
  }
}

// ============================================================
// Inicialización
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  themeBtn.addEventListener('click',  () => setTheme(htmlEl.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'));
  btnToggle.addEventListener('click', toggleCamera);
  btnSingle.addEventListener('click', () => captureAndDetect());
  switchCont.addEventListener('change', toggleContinuousMode);

  loadModels();
});

// ============================================================
// Control de Cámara
// ============================================================
async function toggleCamera() {
  state.cameraActive ? stopCamera() : await startCamera();
}

async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
      audio: false
    });
    video.srcObject = state.stream;
    await video.play();

    state.cameraActive = true;
    placeholder.style.display = 'none';
    btnCameraText.textContent = 'Detener Cámara';
    btnToggle.querySelector('.btn-icon').textContent = '⏹️';

    const ready = state.modelLoaded && state.faceModelLoaded;
    btnSingle.disabled  = !ready;
    switchCont.disabled = !ready;
    setStatus('online', ready ? 'Cámara Activa' : 'Cargando modelos...');
  } catch (err) {
    alert('No se pudo acceder a la cámara: ' + err.message);
  }
}

function stopCamera() {
  state.stream?.getTracks().forEach(t => t.stop());
  state.stream      = null;
  video.srcObject   = null;
  state.cameraActive = false;
  placeholder.style.display = 'flex';
  btnCameraText.textContent = 'Iniciar Cámara';
  btnToggle.querySelector('.btn-icon').textContent = '📹';
  btnSingle.disabled  = true;
  switchCont.disabled = true;

  if (state.continuousMode) { switchCont.checked = false; toggleContinuousMode(); }
  clearCanvas();
  resetResults();
  setStatus('offline', 'Conexión Inactiva');
}

// ============================================================
// Modo Escaneo Continuo
// ============================================================
function toggleContinuousMode() {
  state.continuousMode = switchCont.checked;
  if (state.continuousMode) {
    state.scanInterval = setInterval(captureAndDetect, state.scanIntervalMs);
    btnSingle.disabled = true;
    setStatus('online', 'Escaneo Continuo');
  } else {
    clearInterval(state.scanInterval);
    state.scanInterval = null;
    btnSingle.disabled = false;
    setStatus('online', 'Cámara Activa');
  }
}

// ============================================================
// Captura y Detección (100% navegador)
// ============================================================
async function captureAndDetect() {
  if (!state.cameraActive || video.readyState < 2) return;
  if (!state.modelLoaded || !state.faceModelLoaded)  { setStatus('processing', 'Cargando...'); return; }
  if (state.isProcessing) return;

  state.isProcessing = true;
  if (!state.continuousMode) loadingOverlay.classList.remove('hidden');
  setStatus('processing', 'Analizando...');

  const t0 = performance.now();

  try {
    // ── 1. Detectar rostros ──────────────────────────────────
    const preds = await state.faceDetector.estimateFaces(video, false);

    if (!preds.length) {
      clearCanvas();
      noResults.textContent = 'No se detectaron rostros. Asegúrate de estar frente a la cámara.';
      noResults.classList.remove('hidden');
      resultsList.classList.add('hidden');
      statPeople.textContent  = '0';
      statLatency.textContent = `${Math.round(performance.now() - t0)} ms`;
      setStatus('online', state.continuousMode ? 'Escaneo Continuo' : 'Cámara Activa');
      return;
    }

    // ── 2. Clasificar cada rostro ────────────────────────────
    const faces = [];
    const vw = video.videoWidth  || 640;
    const vh = video.videoHeight || 480;

    for (const pred of preds) {
      const [x1, y1] = pred.topLeft;
      const [x2, y2] = pred.bottomRight;

      // Clamping para no salirse del frame
      const x = Math.max(0, Math.round(x1));
      const y = Math.max(0, Math.round(y1));
      const w = Math.min(Math.round(x2 - x1), vw - x);
      const h = Math.min(Math.round(y2 - y1), vh - y);
      if (w <= 0 || h <= 0) continue;

      const rawPred = tf.tidy(() => {
        const img    = tf.browser.fromPixels(video);
        const roi    = img.slice([y, x, 0], [h, w, 3]);
        const scaled = tf.image.resizeBilinear(roi, [224, 224]);
        const norm   = scaled.div(255.0).expandDims(0);  // [1, 224, 224, 3]
        const out    = state.maskModel.predict(norm);
        return out;
      });

      const val = (await rawPred.data())[0];
      rawPred.dispose();

      const isMask    = val > 0.7;
      const label     = isMask ? 'CON MASCARA' : 'SIN MASCARA';
      const confidence = Math.round((isMask ? val : 1 - val) * 1000) / 10;

      faces.push({ box: [x, y, w, h], label, confidence, raw_pred: val });
    }

    statLatency.textContent = `${Math.round(performance.now() - t0)} ms`;
    renderResults(faces, vw, vh);

  } catch (err) {
    console.error('[Detección]', err);
    setStatus('offline', 'Error en detección');
  } finally {
    loadingOverlay.classList.add('hidden');
    state.isProcessing = false;
    if (state.cameraActive && !state.continuousMode) {
      setStatus('online', 'Cámara Activa');
    }
  }
}

// ============================================================
// Renderizar en Canvas
// ============================================================
function renderResults(faces, vw, vh) {
  clearCanvas();
  const sx = canvas.width  / vw;
  const sy = canvas.height / vh;

  faces.forEach(face => {
    const [x, y, w, h] = face.box;
    const cx = x * sx, cy = y * sy, cw = w * sx, ch = h * sy;
    const isMask  = face.label === 'CON MASCARA';
    const color   = isMask ? '#22c55e' : '#ef4444';
    const glow    = isMask ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)';

    ctx.shadowColor = glow;
    ctx.shadowBlur  = 18;
    ctx.strokeStyle = color;
    ctx.lineWidth   = 2.5;
    ctx.strokeRect(cx, cy, cw, ch);
    drawCorners(cx, cy, cw, ch, color, 16, 3);

    ctx.shadowBlur = 0;
    const txt  = `${face.label}  ${face.confidence.toFixed(1)}%`;
    ctx.font   = 'bold 13px Outfit, sans-serif';
    const tw   = ctx.measureText(txt).width + 16;
    const th   = 24;
    const ty   = cy > th + 6 ? cy - th - 4 : cy + ch + 4;

    ctx.fillStyle = isMask ? 'rgba(22,101,52,0.85)' : 'rgba(127,29,29,0.85)';
    ctx.beginPath();
    ctx.roundRect(cx, ty, tw, th, 5);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.fillText(txt, cx + 8, ty + 16);
  });

  statPeople.textContent = faces.length;
  updateResultsList(faces);
  setStatus('online', state.continuousMode ? 'Escaneo Continuo' : 'Cámara Activa');
}

function drawCorners(x, y, w, h, color, size, thickness) {
  ctx.strokeStyle = color;
  ctx.lineWidth   = thickness;
  ctx.shadowBlur  = 0;
  [
    [[x, y+size],   [x, y],     [x+size, y]],
    [[x+w-size, y], [x+w, y],   [x+w, y+size]],
    [[x+w, y+h-size],[x+w, y+h],[x+w-size, y+h]],
    [[x+size, y+h], [x, y+h],   [x, y+h-size]]
  ].forEach(pts => {
    ctx.beginPath();
    ctx.moveTo(...pts[0]);
    ctx.lineTo(...pts[1]);
    ctx.lineTo(...pts[2]);
    ctx.stroke();
  });
}

function updateResultsList(faces) {
  if (!faces.length) {
    noResults.textContent = 'No se detectaron rostros en este fotograma.';
    noResults.classList.remove('hidden');
    resultsList.classList.add('hidden');
    return;
  }
  noResults.classList.add('hidden');
  resultsList.classList.remove('hidden');
  resultsList.innerHTML = '';

  faces.forEach((face, i) => {
    const isMask = face.label === 'CON MASCARA';
    const css    = isMask ? 'with-mask' : 'no-mask';
    const emoji  = isMask ? '✅' : '❌';
    const el     = document.createElement('div');
    el.className = `result-item ${css}`;
    el.innerHTML = `
      <div class="result-item-header">
        <span class="face-id">👤 Rostro ${i + 1}</span>
        <span class="detection-badge ${css}">${emoji} ${face.label}</span>
      </div>
      <div class="confidence-container ${css}">
        <div class="confidence-bar-outer">
          <div class="confidence-bar-inner" style="width:${face.confidence.toFixed(1)}%"></div>
        </div>
        <span class="confidence-text">${face.confidence.toFixed(1)}%</span>
      </div>`;
    resultsList.appendChild(el);
  });
}

// ============================================================
// Utilidades
// ============================================================
function clearCanvas()  { ctx.clearRect(0, 0, canvas.width, canvas.height); }
function resetResults() {
  statPeople.textContent  = '0';
  statLatency.textContent = '0 ms';
  noResults.textContent   = 'Activa la cámara e inicia el escaneo para ver los resultados.';
  noResults.classList.remove('hidden');
  resultsList.classList.add('hidden');
  resultsList.innerHTML   = '';
}
function setStatus(type, text) {
  statusBadge.textContent = text;
  statusBadge.className   = `status-badge status-${type}`;
}
