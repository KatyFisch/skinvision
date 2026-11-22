const labels = {
  AKIEC: 'Actinic keratosis or intraepidermal carcinoma',
  BCC: 'Basal cell carcinoma',
  BEN_OTH: 'Other benign growths',
  BKL: 'Benign keratinocytic lesion',
  DF: 'Dermatofibroma',
  INF: 'Inflammatory or infectious conditions',
  MAL_OTH: 'Other malignant growths',
  MEL: 'Melanoma',
  NV: 'Mole (melanocytic nevus)',
  SCCKA: 'Squamous cell carcinoma or keratoacanthoma',
  VASC: 'Blood vessel lesions or bleeding',
};

const input = /** @type {HTMLInputElement} */ (document.getElementById('file-input'));
const dropzone = document.getElementById('dropzone');
const choose = document.getElementById('choose');
const preview = /** @type {HTMLCanvasElement} */ (document.getElementById('preview'));
const context = preview.getContext('2d', { alpha: false });
const cropFrame = document.getElementById('crop-frame');
const cropControls = document.getElementById('crop-controls');
const cropZoom = /** @type {HTMLInputElement} */ (document.getElementById('crop-zoom'));
const cropZoomValue = document.getElementById('crop-zoom-value');
const emptyPhoto = document.getElementById('empty-photo');
const fileRow = document.getElementById('file-row');
const fileName = document.getElementById('file-name');
const error = document.getElementById('upload-error');
const analyze = document.getElementById('analyze');
const analyzeLabel = document.getElementById('analyze-label');
const analyzeIcon = document.getElementById('analyze-icon');
const resultIdle = document.getElementById('result-idle');
const resultLoading = document.getElementById('result-loading');
const resultDone = document.getElementById('result-done');

let photo;
let selection = 0;
let crop;
let drag;
let request;

function showError(message = '') {
  error.textContent = message;
  error.hidden = !message;
}

function showResult(state) {
  resultIdle.hidden = state !== 'idle';
  resultLoading.hidden = state !== 'loading';
  resultDone.hidden = state !== 'done';
  const busy = state === 'loading';
  analyze.setAttribute('aria-disabled', String(!photo || busy));
  analyze.title = busy ? 'Analysis is running' : photo ? '' : 'Choose a photo first';
  analyzeLabel.textContent = busy ? 'Analyzing…' : 'Analyze photo';
  analyzeIcon.hidden = busy;
}

function clearAnalysis() {
  request?.abort();
  request = undefined;
  showError();
  showResult('idle');
}

function showPhoto(image, filename = '') {
  endDrag();
  photo = image;
  cropFrame.hidden = !image;
  cropControls.hidden = !image;
  dropzone.classList.toggle('has-photo', Boolean(image));
  emptyPhoto.hidden = Boolean(image);
  fileRow.hidden = !image;
  fileName.textContent = filename;
  if (image) {
    resetCrop();
  } else {
    context.clearRect(0, 0, preview.width, preview.height);
    clearAnalysis();
  }
}

async function setPhoto(file) {
  const current = ++selection;
  if (!file) {
    showPhoto(undefined);
    return;
  }
  if (!['image/jpeg', 'image/png', ''].includes(file.type)) {
    showError('Choose a JPEG or PNG photo.');
    return;
  }
  if (file.size > 20 * 1024 * 1024) {
    showError('Choose a photo under 20 MB.');
    return;
  }

  const url = URL.createObjectURL(file);
  const image = new Image();
  try {
    const header = new DataView(await file.slice(0, 8).arrayBuffer());
    const jpeg = header.byteLength >= 3 && header.getUint16(0) === 0xffd8 && header.getUint8(2) === 0xff;
    const png = header.byteLength === 8 && header.getUint32(0) === 0x89504e47 && header.getUint32(4) === 0x0d0a1a0a;
    if (!jpeg && !png) throw new Error('Unsupported image format');
    image.src = url;
    await image.decode();
    if (current !== selection) return;
    const choosing = document.activeElement === choose;
    showPhoto(image, file.name || 'Pasted photo');
    if (choosing) preview.focus();
  } catch {
    if (current === selection) showError('This photo could not be opened. Try another JPEG or PNG.');
  } finally {
    URL.revokeObjectURL(url);
  }
}

function drawCrop() {
  const ratio = preview.width / preview.height;
  const scale = cropZoom.valueAsNumber / 100;
  crop.width = Math.min(photo.naturalWidth, photo.naturalHeight * ratio) / scale;
  crop.height = crop.width / ratio;
  const halfWidth = crop.width / 2;
  const halfHeight = crop.height / 2;
  crop.x = Math.max(halfWidth, Math.min(crop.x, photo.naturalWidth - halfWidth));
  crop.y = Math.max(halfHeight, Math.min(crop.y, photo.naturalHeight - halfHeight));

  context.clearRect(0, 0, preview.width, preview.height);
  context.imageSmoothingQuality = 'high';
  context.drawImage(photo, crop.x - halfWidth, crop.y - halfHeight, crop.width, crop.height, 0, 0, preview.width, preview.height);
}

function refreshCrop() {
  const zoom = `${cropZoom.value}%`;
  cropZoomValue.textContent = zoom;
  cropZoom.setAttribute('aria-valuetext', zoom);
  drawCrop();
  clearAnalysis();
}

function resetCrop() {
  cropZoom.value = '100';
  crop = { x: photo.naturalWidth / 2, y: photo.naturalHeight / 2 };
  refreshCrop();
}

function panCrop(x, y) {
  const previousX = crop.x;
  const previousY = crop.y;
  crop.x += x * crop.width;
  crop.y += y * crop.height;
  drawCrop();
  if (crop.x !== previousX || crop.y !== previousY) clearAnalysis();
}

function endDrag() {
  if (!drag) return;
  if (preview.hasPointerCapture(drag.id)) preview.releasePointerCapture(drag.id);
  drag = undefined;
  cropFrame.classList.remove('is-panning');
}

cropZoom.addEventListener('input', refreshCrop);
document.querySelector('#reset-crop').addEventListener('click', resetCrop);
preview.addEventListener('pointerdown', (event) => {
  if (!photo || drag || event.button !== 0) return;
  preview.setPointerCapture(event.pointerId);
  drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
  cropFrame.classList.add('is-panning');
});
preview.addEventListener('pointermove', (event) => {
  if (drag?.id !== event.pointerId) return;
  const bounds = preview.getBoundingClientRect();
  panCrop((drag.x - event.clientX) / bounds.width, (drag.y - event.clientY) / bounds.height);
  drag.x = event.clientX;
  drag.y = event.clientY;
});
preview.addEventListener('lostpointercapture', endDrag);
preview.addEventListener('keydown', (event) => {
  if (event.altKey || event.ctrlKey || event.metaKey) return;
  const directions = { ArrowLeft: [1, 0], ArrowRight: [-1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] };
  if (!Object.hasOwn(directions, event.key)) return;
  const [x, y] = directions[event.key];
  event.preventDefault();
  panCrop(0.04 * x, 0.04 * y);
});

dropzone.addEventListener('click', () => {
  if (!photo) input.click();
});
document.querySelector('#replace').addEventListener('click', () => input.click());
document.querySelector('#remove').addEventListener('click', () => {
  setPhoto(undefined);
  choose.focus();
});
input.addEventListener('change', () => {
  if (input.files.length) setPhoto(input.files[0]);
  input.value = '';
});
dropzone.addEventListener('dragover', (event) => {
  if (!event.dataTransfer.types.includes('Files')) return;
  event.preventDefault();
  dropzone.classList.add('is-dragging');
});
dropzone.addEventListener('dragleave', (event) => {
  if (!(event.relatedTarget instanceof Node) || !dropzone.contains(event.relatedTarget)) {
    dropzone.classList.remove('is-dragging');
  }
});
dropzone.addEventListener('drop', (event) => {
  event.preventDefault();
  dropzone.classList.remove('is-dragging');
  if (event.dataTransfer.files.length) setPhoto(event.dataTransfer.files[0]);
});
document.addEventListener('paste', (event) => {
  const image = Array.from(event.clipboardData.files).find((file) => file.type.startsWith('image/'));
  if (!image) return;
  event.preventDefault();
  setPhoto(image);
});

analyze.addEventListener('click', async () => {
  if (!photo || request) return;
  const pending = new AbortController();
  request = pending;
  showError();
  showResult('loading');

  try {
    /** @type {Blob | null} */
    const croppedPhoto = await new Promise((resolve) => preview.toBlob(resolve, 'image/png'));
    if (request !== pending) return;
    if (!croppedPhoto) throw new Error('The crop could not be prepared. Try again.');
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: croppedPhoto,
      signal: pending.signal,
    });
    const data = await response.json();
    if (request !== pending) return;
    if (!response.ok) throw new Error(data.error);

    document.querySelector('#result-code').textContent = data.research_prediction;
    document.querySelector('#result-label').textContent = labels[data.research_prediction] ?? data.research_prediction;
    document.querySelector('#result-route').textContent = data.route === 'MEL_OR_NV' ? 'MEL vs NV' : 'Other diagnoses';
    showResult('done');
  } catch (cause) {
    if (request !== pending) return;
    showError(cause instanceof TypeError ? 'Could not reach the demo. Check that it is running and try again.' : cause.message);
    showResult('idle');
  } finally {
    if (request === pending) request = undefined;
  }
});
