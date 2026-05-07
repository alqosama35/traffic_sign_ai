const COLORS = ['#2563eb', '#7c3aed', '#059669', '#d97706'];

function configLabel(run) {
  const sel = run.selection === 'tournament' ? 'Tournament' : 'Roulette';
  const cx  = run.crossover === 'single_point' ? 'Single-Point' : 'Uniform';
  return `${sel} × ${cx}`;
}

function pct(v, dec = 2) {
  return (v * 100).toFixed(dec) + '%';
}

async function init() {
  let runs;
  try {
    const res = await fetch('/dashboard/data/ea');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    runs = await res.json();
  } catch (err) {
    document.getElementById('ga-table-container').innerHTML =
      `<div class="error">Failed to load EA data: ${err.message}</div>`;
    return;
  }
  buildChart(runs);
  buildTable(runs);
}

function buildChart(runs) {
  const ctx = document.getElementById('ga-chart').getContext('2d');
  const genCount = runs[0].fitness_per_generation.length;

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: Array.from({ length: genCount }, (_, i) => i + 1),
      datasets: runs.map((run, i) => ({
        label: configLabel(run),
        data: run.fitness_per_generation.map(v => v * 100),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: COLORS[i % COLORS.length] + '18',
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.3,
        fill: false,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12, font: { size: 12 } } },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(3)}%`,
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Generation', font: { size: 12 } },
          grid: { color: '#f1f5f9' },
        },
        y: {
          title: { display: true, text: 'Validation Accuracy (%)', font: { size: 12 } },
          grid: { color: '#f1f5f9' },
          ticks: { callback: v => v.toFixed(1) + '%' },
        },
      },
    },
  });
}

function buildTable(runs) {
  const bestGaAcc   = Math.max(...runs.map(r => r.best_fitness));
  const bestReduc   = Math.max(...runs.map(r => r.reduction_ratio));
  const bestAccDrop = Math.min(...runs.map(r => r.accuracy_drop_pp));

  const rows = runs.map(run => {
    const isGa   = run.best_fitness    === bestGaAcc    ? ' class="num best"' : ' class="num"';
    const isR    = run.reduction_ratio === bestReduc    ? ' class="num best"' : ' class="num"';
    const isD    = run.accuracy_drop_pp=== bestAccDrop  ? ' class="num best"' : ' class="num"';
    return `
      <tr>
        <td>${configLabel(run)}</td>
        <td class="num">${pct(run.full_feature_val_accuracy)}</td>
        <td${isGa}>${pct(run.best_fitness)}</td>
        <td class="num">${run.selected_features} / ${run.total_features}</td>
        <td${isR}>${(run.reduction_ratio * 100).toFixed(1)}%</td>
        <td${isD}>${(run.accuracy_drop_pp * 100).toFixed(2)} pp</td>
      </tr>`;
  }).join('');

  document.getElementById('ga-table-container').innerHTML = `
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>Configuration</th>
            <th class="num">Full Val Acc</th>
            <th class="num">GA Val Acc</th>
            <th class="num">Features Used</th>
            <th class="num">Reduction</th>
            <th class="num">Acc Drop</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="footnote">
      Green values: best GA accuracy, highest feature reduction, smallest accuracy drop.
      All runs use generational replacement + bit-flip mutation (rate&nbsp;0.01).
    </p>`;
}

init();
