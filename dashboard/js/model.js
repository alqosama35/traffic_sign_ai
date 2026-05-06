async function init() {
  let report;
  try {
    const res = await fetch('/dashboard/data/training-report');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    report = await res.json();
  } catch (err) {
    document.getElementById('model-stats-container').innerHTML =
      `<div class="error">Failed to load training report: ${err.message}</div>`;
    return;
  }

  const macroF1      = report['macro avg']['f1-score'];
  const totalSamples = report['macro avg']['support'];
  const accuracy     = report['accuracy'];

  const cards = document.querySelectorAll('#model-stats-container .stat-card');
  if (cards.length >= 3) {
    cards[0].querySelector('.stat-value').textContent = (accuracy * 100).toFixed(2) + '%';
    cards[1].querySelector('.stat-value').textContent = (macroF1  * 100).toFixed(2) + '%';
    cards[2].querySelector('.stat-value').textContent = totalSamples.toLocaleString();
  }
}

init();
