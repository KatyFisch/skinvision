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

const input = document.querySelector('#file-input');
const dropzone = document.querySelector('#dropzone');
const choose = document.querySelector('#choose');
const preview = document.querySelector('#preview');
const emptyPhoto = document.querySelector('#empty-photo');
const fileRow = document.querySelector('#file-row');
const fileName = document.querySelector('#file-name');
const error = document.querySelector('#upload-error');
const analyze = document.querySelector('#analyze');
const analyzeLabel = document.querySelector('#analyze-label');
const analyzeIcon = document.querySelector('#analyze-icon');
const resultIdle = document.querySelector('#result-idle');
const resultLoading = document.querySelector('#result-loading');
const resultDone = document.querySelector('#result-done');

let photo;
let previewUrl;
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

function setPhoto(file) {
  if (file && !['image/jpeg', 'image/png', ''].includes(file.type)) {
    showError('Choose a JPEG or PNG photo.');
    return;
  }
  if (file && (!file.size || file.size > 20 * 1024 * 1024)) {
    showError('Choose a photo under 20 MB.');
    return;
  }

  request?.abort();
  request = undefined;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const choosing = document.activeElement === choose;
  photo = file;
  previewUrl = file ? URL.createObjectURL(file) : undefined;
  if (previewUrl) preview.src = previewUrl;
  else preview.removeAttribute('src');
  fileName.textContent = file?.name || 'Pasted photo';
  preview.hidden = !file;
  emptyPhoto.hidden = Boolean(file);
  fileRow.hidden = !file;
  showError();
  showResult('idle');
  if (choosing && file) analyze.focus();
}

dropzone.addEventListener('click', () => input.click());
document.querySelector('#replace').addEventListener('click', () => input.click());
document.querySelector('#remove').addEventListener('click', () => {
  setPhoto(undefined);
  choose.focus();
});
input.addEventListener('change', () => {
  if (input.files.length) setPhoto(input.files[0]);
  input.value = '';
});
preview.addEventListener('error', () => {
  if (!photo) return;
  setPhoto(undefined);
  showError('This photo could not be opened. Try another JPEG or PNG.');
});

dropzone.addEventListener('dragover', (event) => {
  if (!event.dataTransfer.types.includes('Files')) return;
  event.preventDefault();
  dropzone.classList.add('is-dragging');
});
dropzone.addEventListener('dragleave', (event) => {
  if (!dropzone.contains(event.relatedTarget)) dropzone.classList.remove('is-dragging');
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
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: photo,
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
