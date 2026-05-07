function pct(v, dec = 2) { return (v * 100).toFixed(dec) + '%'; }

async function init() {
  const container = document.getElementById('comparison-container');

  let report, cv;
  try {
    [report, cv] = await Promise.all([
      fetch('/dashboard/data/training-report').then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
      fetch('/dashboard/data/cv-metrics').then(r      => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
      fetch('/dashboard/data/ea').then(r              => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); }),
    ]);
  } catch (err) {
    container.innerHTML = `<div class="error">Failed to load comparison data: ${err.message}</div>`;
    return;
  }

  // EA best run by GA val accuracy
  const eaRuns = await fetch('/dashboard/data/ea').then(r => r.json());
  const bestRun = eaRuns.reduce((a, b) => a.best_fitness > b.best_fitness ? a : b);
  const gaLabel = (() => {
    const sel = bestRun.selection === 'tournament' ? 'Tournament' : 'Roulette';
    const cx  = bestRun.crossover === 'single_point' ? 'Single-Point' : 'Uniform';
    return `${sel} × ${cx}`;
  })();

  const rows = [
    {
      model:    'CV Baseline',
      type:     '<span class="badge badge-cv">Classical</span>',
      accuracy: pct(cv.overall.accuracy),
      f1:       pct(cv.overall.macro_f1),
      samples:  cv.overall.test_samples.toLocaleString(),
      note:     '10-class subset†',
    },
    {
      model:    'MobileNetV2',
      type:     '<span class="badge badge-dl">Deep Learning</span>',
      accuracy: pct(report.accuracy),
      f1:       pct(report['macro avg']['f1-score']),
      samples:  report['macro avg'].support.toLocaleString(),
      note:     '43 classes, GTSRB test set',
    },
    {
      model:    `GA-Optimized<br><small style="color:#64748b;font-weight:400">${gaLabel}</small>`,
      type:     '<span class="badge badge-ga">GA + k-NN/SVM</span>',
      accuracy: pct(bestRun.best_fitness),
      f1:       '—',
      samples:  `${bestRun.selected_features} / ${bestRun.total_features} features`,
      note:     `${(bestRun.reduction_ratio * 100).toFixed(1)}% feature reduction`,
    },
  ];

  const tbody = rows.map(r => `
    <tr>
      <td>${r.model}</td>
      <td>${r.type}</td>
      <td class="num">${r.accuracy}</td>
      <td class="num">${r.f1}</td>
      <td class="num">${r.samples}</td>
      <td>${r.note}</td>
    </tr>`).join('');

  container.innerHTML = `
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Type</th>
            <th class="num">Accuracy</th>
            <th class="num">Macro F1</th>
            <th class="num">Evaluated On</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>${tbody}</tbody>
      </table>
    </div>`;
}

init();
