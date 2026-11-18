'use strict';
const $ = id => document.getElementById(id);
let requestPending = false;
let hadFrame = false;
let lastMode = null;
let currentPhase = 'idle';
let streamActive = false;
let lastSourceRun = null;
let connectionErrorShown = false;
let latestSourceRun = null;
let displayedPhotoKey = null;
let photoRequest = null;
let photoUrl = null;
let photoToken = 0;
let displayedFrameId = null;

async function jsonRequest(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch(path, {...options, signal: controller.signal, cache: 'no-store'});
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || 'No se pudo completar la acción.');
    return body;
  } finally { clearTimeout(timeout); }
}

function metric(value) { return Number.isFinite(value) ? value.toFixed(3) : '—'; }
function timing(fps, ms) { return Number.isFinite(fps) && Number.isFinite(ms) ? `${fps.toFixed(1)} FPS · ${ms.toFixed(0)} ms` : '—'; }
function showError(message) { $('error').textContent = message || ''; $('error').hidden = !message; }

function cancelPhotoRequest() {
  photoToken += 1;
  if (photoRequest) photoRequest.controller.abort();
  photoRequest = null;
}

async function renderStatus(state) {
  if (latestSourceRun !== null && state.source_run < latestSourceRun) return;
  latestSourceRun = state.source_run;
  const isPhoto = state.phase === 'running' && state.mode === 'photos' && Number.isInteger(state.frame_sequence);
  if (!isPhoto) {
    cancelPhotoRequest();
    if (state.phase === 'starting' || state.phase === 'error' || state.mode === 'camera') {
      if (photoUrl) URL.revokeObjectURL(photoUrl);
      photoUrl = null; displayedPhotoKey = null;
    }
    paintStatus(state);
    return;
  }
  const key = `${state.source_run}:${state.frame_sequence}`;
  if (key === displayedPhotoKey) { paintStatus(state); return; }
  if (photoRequest && photoRequest.key === key) return;
  cancelPhotoRequest();
  const token = photoToken;
  const controller = new AbortController();
  photoRequest = {key, controller};
  const timeout = setTimeout(() => controller.abort(), 5000);
  let pendingUrl = null;
  try {
    const response = await fetch(`/api/frame.jpg?sequence=${state.frame_sequence}`, {signal: controller.signal, cache: 'no-store'});
    if (response.status === 409) return; // La siguiente consulta traerá el estado de la imagen nueva.
    if (!response.ok) throw new Error('No se pudo cargar la fotografía. Se reintentará automáticamente.');
    pendingUrl = URL.createObjectURL(await response.blob());
    const preview = new Image();
    preview.src = pendingUrl;
    await preview.decode();
    if (token !== photoToken || latestSourceRun !== state.source_run) return;
    const previousUrl = photoUrl;
    photoUrl = pendingUrl; pendingUrl = null;
    displayedPhotoKey = key;
    paintStatus(state, photoUrl);
    if (previousUrl) URL.revokeObjectURL(previousUrl);
  } catch (error) {
    if (error.name !== 'AbortError' && token === photoToken) { connectionErrorShown = true; showError(error.message); }
  } finally {
    clearTimeout(timeout);
    if (pendingUrl) URL.revokeObjectURL(pendingUrl);
    if (token === photoToken) photoRequest = null;
  }
}

function paintStatus(state, snapshotUrl = null) {
  // Una consulta anterior puede terminar después de la respuesta a Detener.
  if (lastSourceRun !== null && Number.isFinite(state.source_run) && state.source_run < lastSourceRun) return;
  currentPhase = state.phase;
  const sourceChanged = lastSourceRun !== null && state.source_run !== lastSourceRun;
  if ((sourceChanged && state.phase !== 'idle') || state.phase === 'starting' || state.phase === 'error') {
    hadFrame = false;
    streamActive = false;
    displayedFrameId = null;
    $('frame').removeAttribute('src');
  }
  lastSourceRun = state.source_run;
  $('status').textContent = state.message;
  $('source').textContent = state.source_label;
  if (state.phase === 'running') displayedFrameId = state.frame_id;
  $('frame-id').textContent = (hadFrame || state.phase === 'running') ? displayedFrameId || '' : '';
  $('piece-count').textContent = state.measurements.length;
  $('inference').textContent = timing(state.inference_fps, state.inference_ms);
  $('processing').textContent = timing(state.processing_fps, state.processing_ms);
  $('resolution').textContent = state.resolution ? state.resolution.join(' × ') : '—';
  $('source-note').textContent = state.mode === 'photos'
    ? 'Secuencia de fotografías de prueba. No es captura de cámara.'
    : state.mode === 'camera' ? 'Captura desde la cámara seleccionada.'
    : hadFrame ? 'Última imagen procesada. La fuente está detenida.'
    : 'La cámara permanece apagada hasta pulsar «Usar cámara».';
  $('timing-note').textContent = state.mode === 'photos'
    ? 'FPS por foto procesada; excluyen la pausa de inspección y el transporte al navegador. No miden captura de cámara. IDs locales a cada imagen.'
    : 'FPS de inferencia y procesamiento de imagen; excluyen transporte y dibujo del navegador. No miden frecuencia de captura. IDs por fotograma.';
  $('empty-title').textContent = state.phase === 'starting' ? 'Preparando la fuente' : state.phase === 'error' ? 'No hay una fuente activa' : 'Elige una fuente para comenzar';
  $('empty-note').textContent = state.phase === 'starting'
    ? 'La primera imagen aparecerá al terminar su procesamiento.'
    : 'Las fotografías permiten revisar el modelo sin encender la cámara.';
  const rows = $('measurements');
  rows.replaceChildren();
  for (const [index, item] of state.measurements.entries()) {
    const tr = document.createElement('tr');
    const id = document.createElement('td');
    const dot = document.createElement('span'); dot.className = `piece-dot piece-color-${index % 6}`;
    id.append(dot, document.createTextNode(`#${item.segment_id}`));
    if (item.truncated) { const label = document.createElement('span'); label.className = 'partial'; label.textContent = 'Parcial en borde'; id.append(label); }
    tr.append(id);
    for (const key of ['length', 'width']) {
      const td = document.createElement('td');
      td.textContent = Number.isFinite(item[key]) ? `${item[key].toFixed(1)} ${item.unit || 'px'}` : '—';
      tr.append(td);
    }
    rows.append(tr);
  }
  if (!state.measurements.length) {
    const tr = document.createElement('tr'); const td = document.createElement('td');
    td.colSpan = 3; td.className = 'empty-row'; td.textContent = state.phase === 'starting' ? 'Preparando la fuente…' : 'Sin piezas para mostrar.'; tr.append(td); rows.append(tr);
  }
  if (state.frames > 0 && state.phase === 'running') hadFrame = true;
  if (state.phase === 'running' && state.mode === 'photos') {
    if (snapshotUrl) $('frame').src = snapshotUrl;
    streamActive = false;
  } else if (state.phase === 'running' && hadFrame && !streamActive) {
    $('frame').src = `/stream.mjpg?t=${Date.now()}`;
    streamActive = true;
  } else if (state.phase !== 'running' && hadFrame && streamActive) {
    $('frame').src = `/api/frame.jpg?t=${Date.now()}`;
    streamActive = false;
  }
  $('frame').hidden = !hadFrame;
  $('empty').hidden = hadFrame;
  $('photos').disabled = requestPending || (state.mode === 'photos' && state.phase !== 'error');
  $('camera').disabled = requestPending || (state.mode === 'camera' && state.phase !== 'error');
  $('stop').disabled = requestPending || !['starting', 'running'].includes(state.phase);
  if (state.error) showError(state.error);
  else if (lastMode !== state.mode) showError('');
  lastMode = state.mode;
}

async function action(path, body = {}) {
  if (requestPending) return;
  requestPending = true;
  showError('');
  try {
    const state = await jsonRequest(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    requestPending = false;
    renderStatus(state);
  } catch (error) { requestPending = false; showError(error.message || 'No se pudo conectar con el panel.'); }
}

async function pollStatus() {
  try {
    const state = await jsonRequest('/api/status');
    if (connectionErrorShown) { showError(''); connectionErrorShown = false; }
    renderStatus(state);
  }
  catch (_) {
    connectionErrorShown = true;
    showError('No se pudo conectar con la demostración. Comprueba que su ventana de consola siga abierta.');
  }
  finally { setTimeout(pollStatus, 750); }
}

async function loadMetrics() {
  try {
    const data = await jsonRequest('/api/metrics');
    $('metrics').replaceChildren();
    for (const row of data.rows) {
      const tr = document.createElement('tr');
      const values = [row.model, row.available ? metric(row.balsa_iou) : 'Pendiente', row.available ? metric(row.balsa_f1) : '—', row.available ? metric(row.balsa_precision) : '—', !row.instance_capable ? 'No aplicable' : row.available ? metric(row.map) : '—', row.available ? `${row.negative_images_with_fp}/${row.negative_images}` : '—'];
      for (const value of values) { const td = document.createElement('td'); td.textContent = value; tr.append(td); }
      $('metrics').append(tr);
    }
    const available = data.rows.filter(row => row.available);
    $('metrics-note').textContent = available.length
      ? `${available[0].image_count} imágenes: ${available[0].positive_images} con balsa y ${available[0].negative_images} negativas. Los recortes comparten fuente y partición. mAP de máscaras COCO con IoU de 0,50 a 0,95. Referencias revisadas por el asistente, sin auditoría humana independiente.`
      : 'Los resultados aparecerán cuando se guarden las evaluaciones finales.';
  } catch (_) { $('metrics-note').textContent = 'No se pudieron leer los resultados en este momento.'; }
  finally { setTimeout(loadMetrics, 10000); }
}

$('photos').addEventListener('click', () => action('/api/start', {mode: 'photos'}));
$('camera').addEventListener('click', () => action('/api/start', {mode: 'camera', camera_index: Number($('camera-index').value)}));
$('stop').addEventListener('click', () => action('/api/stop'));
$('frame').addEventListener('error', () => {
  if (currentPhase === 'running' && lastMode === 'camera') setTimeout(() => {
    if (currentPhase === 'running' && lastMode === 'camera') { $('frame').src = `/stream.mjpg?t=${Date.now()}`; streamActive = true; }
  }, 1500);
});
window.addEventListener('pagehide', () => navigator.sendBeacon('/api/stop', new Blob(['{}'], {type: 'application/json'})));
pollStatus(); loadMetrics();
