const fileInput   = document.getElementById('file-input');
const uploadArea  = document.querySelector('.upload-area');
const resultDiv   = document.getElementById('predict-result');

const CATEGORY_COLORS = {
  prohibition: 'badge-prohibition',
  danger:      'badge-danger',
  mandatory:   'badge-mandatory',
  other:       'badge-other',
  unknown:     'badge-other',
};

// ── File selection ─────────────────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) runPredict(fileInput.files[0]);
});

// ── Drag-and-drop on the upload area ──────────────────────────────────────────
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) runPredict(file);
});

// ── Prediction ─────────────────────────────────────────────────────────────────
async function runPredict(file) {
  resultDiv.innerHTML = '<div class="card"><div class="loading">Running inference…</div></div>';

  const form = new FormData();
  form.append('file', file);

  let data;
  try {
    const res = await fetch('/dashboard/predict', { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail ?? res.statusText);
    }
    data = await res.json();
  } catch (err) {
    resultDiv.innerHTML = `<div class="card"><div class="error">Prediction failed: ${err.message}</div></div>`;
    return;
  }

  const confidencePct = (data.confidence * 100).toFixed(1);
  const badgeClass    = CATEGORY_COLORS[data.category] ?? 'badge-other';

  const top3Rows = data.top_3.map((item, i) => `
    <li>
      <span>${i + 1}. ${item.class}</span>
      <span style="color:#64748b">${(item.confidence * 100).toFixed(2)}%</span>
    </li>`).join('');

  // Preview the uploaded image
  const objUrl = URL.createObjectURL(file);

  resultDiv.innerHTML = `
    <div class="card result-card">
      <div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start">
        <img src="${objUrl}" alt="Uploaded sign"
             style="width:120px;height:120px;object-fit:contain;border:1px solid #e2e8f0;border-radius:6px;background:#f8fafc">
        <div style="flex:1;min-width:200px">
          <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem">
            <div class="result-class">${data.class}</div>
            <span class="badge ${badgeClass}">${data.category}</span>
          </div>
          <div class="confidence-row">
            <span class="confidence-label">Confidence</span>
            <div class="confidence-bar-wrap">
              <div class="confidence-bar-fill" style="width:${confidencePct}%"></div>
            </div>
            <span class="confidence-pct">${confidencePct}%</span>
          </div>
        </div>
      </div>

      <div style="margin-top:1.25rem">
        <div style="font-size:0.8rem;font-weight:600;color:#475569;margin-bottom:0.4rem">Top 3 Predictions</div>
        <ul class="top3-list">${top3Rows}</ul>
      </div>

      <p class="disclaimer">${data.note}</p>
    </div>`;
}
