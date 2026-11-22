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

const melanomaBands = {
  below: 'Below melanoma cutoff',
  low: 'Low melanoma signal',
  moderate: 'Moderate melanoma signal',
  high: 'High melanoma signal',
};

const input = /** @type {HTMLInputElement} */ (document.getElementById('file-input'));
const dropzone = document.getElementById('dropzone');
const choose = document.getElementById('choose');
const preview = /** @type {HTMLCanvasElement} */ (document.getElementById('preview'));
const context = preview.getContext('2d', { alpha: false });
const cropFrame = document.getElementById('crop-frame');
const lesionPath = document.getElementById('lesion-path');
const maskScanLine = document.getElementById('mask-scan-line');
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
const resultProgress = document.getElementById('result-progress');
const analysisPercent = document.getElementById('analysis-percent');
const resultDone = document.getElementById('result-done');
const resultEmpty = document.getElementById('result-empty');
const resultBand = document.getElementById('result-band');
const scoreSegments = resultBand.querySelectorAll('[data-score-band]');

let photo;
let selectedSample;
let selection = 0;
let crop;
let drag;
let request;

function showError(message = '') {
  error.textContent = message;
  error.hidden = !message;
}

function showResult(state) {
  const busy = state === 'segmenting' || state === 'classifying';
  resultLoading.classList.remove('is-fading');
  resultIdle.hidden = state !== 'idle';
  resultLoading.hidden = !busy;
  resultDone.hidden = state !== 'done';
  resultEmpty.hidden = state !== 'empty';
  analysisPercent.hidden = state !== 'classifying';
  if (busy) resultProgress.textContent = state === 'segmenting' ? 'Finding the lesion area…' : 'Analyzing the crop…';
  analyze.setAttribute('aria-disabled', String(!photo || busy));
  analyze.title = busy ? 'Analysis is running' : photo ? '' : 'Choose a photo first';
  analyzeLabel.textContent = busy ? 'Analyzing…' : 'Analyze photo';
  analyzeIcon.hidden = busy;
}

function showPrediction(data) {
  const bandLabel = melanomaBands[data.melanoma_score_band];
  document.querySelector('#result-code').textContent = bandLabel ? 'RESEARCH SCORE BAND' : data.research_prediction;
  document.querySelector('#result-label').textContent = bandLabel ?? labels[data.research_prediction] ?? data.research_prediction;
  const taskLabel = data.route === 'MEL_OR_NV' ? 'MEL vs NV' : 'Other diagnoses';
  document.querySelector('#result-route').textContent =
    data.analysis_input === 'original_photo' ? `${taskLabel} · original crop used` : taskLabel;
  resultBand.hidden = !bandLabel;
  if (bandLabel) {
    scoreSegments.forEach((segment) => {
      segment.classList.toggle('is-active', segment.getAttribute('data-score-band') === data.melanoma_score_band);
    });
  }
  showResult('done');
}

function clearOutline() {
  cropFrame.classList.remove('is-segmenting', 'has-outline');
  lesionPath.removeAttribute('d');
}

function clearAnalysis() {
  request?.abort();
  request = undefined;
  clearOutline();
  showError();
  showResult('idle');
}

/** @param {HTMLImageElement} [image] @param {string} [filename] @param {HTMLButtonElement} [sample] */
function showPhoto(image, filename = '', sample) {
  endDrag();
  selectedSample?.setAttribute('aria-pressed', 'false');
  selectedSample = sample;
  selectedSample?.setAttribute('aria-pressed', 'true');
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

async function loadSampleGallery() {
  let samples;
  try {
    const response = await fetch('/api/samples');
    if (!response.ok) return;
    samples = await response.json();
  } catch {
    return;
  }
  if (!samples.length) return;

  const grid = document.getElementById('sample-grid');
  const credits = document.getElementById('sample-credits');
  for (const sample of samples) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'button sample-photo';
    button.setAttribute('aria-pressed', 'false');

    const thumbnail = new Image();
    thumbnail.alt = '';
    thumbnail.src = sample.url;
    const caption = document.createElement('span');
    caption.textContent = sample.title;
    button.append(thumbnail, caption);
    button.addEventListener('click', async () => {
      const current = ++selection;
      try {
        await thumbnail.decode();
        if (current !== selection) return;
        showPhoto(thumbnail, sample.title, button);
        if (document.activeElement === button) preview.focus();
      } catch {
        if (current === selection) showError('This sample could not be opened. Choose another photo.');
      }
    });
    grid.append(button);

    const credit = document.createElement('li');
    const source = document.createElement('a');
    source.href = sample.source;
    source.target = '_blank';
    source.rel = 'noopener noreferrer';
    source.textContent = sample.title;
    credit.append(source, ` · ${sample.credit} · ${sample.license} · cropped`);
    credits.append(credit);
  }
  document.getElementById('sample-gallery').hidden = false;
}

loadSampleGallery();

/** @param {ReadableStream<Uint8Array>} body */
async function* analysisEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let tail = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) return;
      const lines = (tail + decoder.decode(value, { stream: true })).split('\n');
      tail = lines.pop();
      for (const line of lines) yield JSON.parse(line);
    }
  } finally {
    reader.releaseLock();
  }
}

function finishAnimations(element) {
  return Promise.allSettled(element.getAnimations().map((animation) => animation.finished));
}

async function revealOutline(path, pending, scanFinished) {
  await scanFinished;
  if (request !== pending) return;
  cropFrame.classList.remove('is-segmenting');
  if (!path) return;
  lesionPath.setAttribute('d', path);
  cropFrame.classList.add('has-outline');
  await finishAnimations(lesionPath);
}

analyze.addEventListener('click', async () => {
  if (!photo || request) return;
  const pending = new AbortController();
  request = pending;
  showError();
  clearOutline();
  cropFrame.classList.add('is-segmenting');
  showResult('segmenting');
  const scanFinished = finishAnimations(maskScanLine);
  let countFinished;

  try {
    /** @type {Blob | null} */
    const croppedPhoto = await new Promise((resolve) => preview.toBlob(resolve, 'image/png'));
    if (request !== pending) return;
    if (!croppedPhoto) throw new Error('The crop could not be prepared. Try again.');
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream', Accept: 'application/x-ndjson' },
      body: croppedPhoto,
      signal: pending.signal,
    });
    if (request !== pending) return;
    if (!response.ok) throw new Error((await response.json()).error);

    for await (const data of analysisEvents(response.body)) {
      if (request !== pending) return;
      if (data.error) throw new Error(data.error);
      if (typeof data.outline === 'string') {
        await revealOutline(data.outline, pending, scanFinished);
        if (request !== pending) return;
        showResult('classifying');
        countFinished = finishAnimations(analysisPercent);
        continue;
      }

      await countFinished;
      if (request !== pending) return;
      resultLoading.classList.add('is-fading');
      await finishAnimations(resultLoading);
      if (request !== pending) return;
      if (data.research_prediction === null) {
        showResult('empty');
      } else {
        showPrediction(data);
      }
      return;
    }
    throw new Error('Analysis ended early. Try again.');
  } catch (cause) {
    if (request !== pending) return;
    clearAnalysis();
    showError(cause instanceof TypeError ? 'Could not reach the demo. Check that it is running and try again.' : cause.message);
  } finally {
    if (request === pending) request = undefined;
  }
});
