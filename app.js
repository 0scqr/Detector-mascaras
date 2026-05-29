/**
 * app.js — Detector de Máscaras Faciales
 * Controla la cámara web, captura fotogramas y los envía a la API /api/detect.
 * Dibuja los resultados sobre un canvas superpuesto.
 */

// ============================================================
// Estado Global
// ============================================================
const state = {
  stream: null,
  cameraActive: false,
  continuousMode: false,
  scanInterval: null,
  scanIntervalMs: 700, // Milisegundos entre escaneos continuos
  lastLatency: 0,
  totalScans: 0,
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
// Inicialización
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
  btnToggle.addEventListener('click', toggleCamera);
  btnSingle.addEventListener('click', () => captureAndDetect());
  switchCont.addEventListener('change', toggleContinuousMode);
  checkApiHealth();
});

// ============================================================
// Verificar que la API está en línea
// ============================================================
async function checkApiHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    if (data.status === 'healthy') {
      const sizeMB = data.model_size_mb ? data.model_size_mb.toFixed(1) : '?';
      statModel.textContent = `OK (${sizeMB} MB)`;
      statModel.className = 'stat-value text-success';
    } else {
      throw new Error('Unhealthy');
    }
  } catch (e) {
    statModel.textContent = 'Sin conexión';
    statModel.className = 'stat-value text-danger';
    console.warn('[API] No se pudo conectar al endpoint /api/health', e);
  }
}

// ============================================================
// Control de Cámara
// ============================================================
async function toggleCamera() {
  if (state.cameraActive) {
    stopCamera();
  } else {
    await startCamera();
  }
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
    btnSingle.disabled = false;
    switchCont.disabled = false;
    setStatus('online', 'Cámara Activa');
  } catch (err) {
    console.error('[Cámara] Error:', err);
    alert('No se pudo acceder a la cámara. Por favor, permite el acceso en tu navegador.');
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  video.srcObject = null;
  state.cameraActive = false;
  placeholder.style.display = 'flex';
  btnCameraText.textContent = 'Iniciar Cámara';
  btnToggle.querySelector('.btn-icon').textContent = '📹';
  btnSingle.disabled = true;
  switchCont.disabled = true;

  // Apagar escaneo continuo si estaba encendido
  if (state.continuousMode) {
    switchCont.checked = false;
    toggleContinuousMode();
  }

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
// Capturar fotograma y enviar a la API
// ============================================================
async function captureAndDetect() {
  if (!state.cameraActive || video.readyState < 2) return;

  // Dibujar el frame del video en un canvas temporal para obtener el base64
  const tmpCanvas = document.createElement('canvas');
  tmpCanvas.width = video.videoWidth || 640;
  tmpCanvas.height = video.videoHeight || 480;
  const tmpCtx = tmpCanvas.getContext('2d');
  // Capturamos el video sin invertir (el servidor no espera imagen espejada)
  tmpCtx.drawImage(video, 0, 0, tmpCanvas.width, tmpCanvas.height);
  const imageBase64 = tmpCanvas.toDataURL('image/jpeg', 0.8);

  if (!state.continuousMode) {
    loadingOverlay.classList.remove('hidden');
  }
  setStatus('processing', 'Procesando...');

  const startTime = performance.now();
  try {
    const response = await fetch('/api/detect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64 })
    });

    const data = await response.json();
    state.lastLatency = Math.round(performance.now() - startTime);
    statLatency.textContent = `${state.lastLatency} ms`;
    state.totalScans++;

    if (data.success) {
      renderResults(data.faces, tmpCanvas.width, tmpCanvas.height);
    } else {
      console.error('[API] Error:', data.error);
    }
  } catch (err) {
    console.error('[Fetch] Error de red:', err);
    setStatus('offline', 'Error de Red');
  } finally {
    loadingOverlay.classList.add('hidden');
    if (state.cameraActive && !state.continuousMode) {
      setStatus('online', 'Cámara Activa');
    }
  }
}

// ============================================================
// Renderizar resultados en Canvas y Panel
// ============================================================
function renderResults(faces, vidW, vidH) {
  clearCanvas();

  // Escalar las coordenadas del servidor a las dimensiones del canvas en pantalla
  const scaleX = canvas.width / vidW;
  const scaleY = canvas.height / vidH;

  faces.forEach((face, index) => {
    const [x, y, w, h] = face.box;
    const cx = x * scaleX;
    const cy = y * scaleY;
    const cw = w * scaleX;
    const ch = h * scaleY;

    const isMask   = face.label === 'CON MASCARA';
    const color    = isMask ? '#22c55e' : '#ef4444';
    const glowHex  = isMask ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)';

    // Rectángulo exterior con glow
    ctx.shadowColor  = glowHex;
    ctx.shadowBlur   = 18;
    ctx.strokeStyle  = color;
    ctx.lineWidth    = 2.5;
    ctx.strokeRect(cx, cy, cw, ch);

    // Esquinas decorativas
    drawCorners(cx, cy, cw, ch, color, 16, 3);

    // Fondo del texto
    ctx.shadowBlur = 0;
    const label    = face.label;
    const confText = `${face.confidence.toFixed(1)}%`;
    const fullText = `${label}  ${confText}`;
    ctx.font       = 'bold 13px Outfit, sans-serif';
    const textW    = ctx.measureText(fullText).width + 16;
    const textH    = 24;
    const textY    = cy > textH + 6 ? cy - textH - 4 : cy + ch + 4;

    ctx.fillStyle  = isMask ? 'rgba(22,101,52,0.85)' : 'rgba(127,29,29,0.85)';
    ctx.beginPath();
    ctx.roundRect(cx, textY, textW, textH, 5);
    ctx.fill();

    ctx.fillStyle  = '#fff';
    ctx.fillText(fullText, cx + 8, textY + 16);
  });

  // Actualizar panel lateral
  statPeople.textContent = faces.length;
  updateResultsList(faces);
  setStatus(state.continuousMode ? 'online' : 'online', state.continuousMode ? 'Escaneo Continuo' : 'Cámara Activa');
}

function drawCorners(x, y, w, h, color, size, thickness) {
  ctx.strokeStyle = color;
  ctx.lineWidth   = thickness;
  ctx.shadowBlur  = 0;
  const corners = [
    [[x, y + size], [x, y], [x + size, y]],
    [[x + w - size, y], [x + w, y], [x + w, y + size]],
    [[x + w, y + h - size], [x + w, y + h], [x + w - size, y + h]],
    [[x + size, y + h], [x, y + h], [x, y + h - size]]
  ];
  corners.forEach(pts => {
    ctx.beginPath();
    ctx.moveTo(...pts[0]);
    ctx.lineTo(...pts[1]);
    ctx.lineTo(...pts[2]);
    ctx.stroke();
  });
}

function updateResultsList(faces) {
  if (faces.length === 0) {
    noResults.textContent = 'No se detectaron rostros en este fotograma.';
    noResults.classList.remove('hidden');
    resultsList.classList.add('hidden');
    return;
  }

  noResults.classList.add('hidden');
  resultsList.classList.remove('hidden');
  resultsList.innerHTML = '';

  faces.forEach((face, i) => {
    const isMask   = face.label === 'CON MASCARA';
    const cssClass = isMask ? 'with-mask' : 'no-mask';
    const emoji    = isMask ? '✅' : '❌';

    const item = document.createElement('div');
    item.className = `result-item ${cssClass}`;
    item.innerHTML = `
      <div class="result-item-header">
        <span class="face-id">👤 Rostro ${i + 1}</span>
        <span class="detection-badge ${cssClass}">${emoji} ${face.label}</span>
      </div>
      <div class="confidence-container ${cssClass}">
        <div class="confidence-bar-outer">
          <div class="confidence-bar-inner" style="width: ${face.confidence.toFixed(1)}%"></div>
        </div>
        <span class="confidence-text">${face.confidence.toFixed(1)}%</span>
      </div>
    `;
    resultsList.appendChild(item);
  });
}

// ============================================================
// Utilidades
// ============================================================
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

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
