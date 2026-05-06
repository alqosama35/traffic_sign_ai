import json, re
from pathlib import Path

nb = json.loads(Path('cv/cv_project.ipynb').read_text(encoding='utf-8'))

def stdout(cell):
    s = ''
    for o in cell.get('outputs', []) or []:
        if o.get('output_type') == 'stream' and o.get('name') == 'stdout':
            t = o.get('text', '')
            if isinstance(t, list):
                t = ''.join(t)
            s += t
    return s

metrics_cell = None
report_cell = None
for c in nb.get('cells', []):
    s = stdout(c)
    if metrics_cell is None and 'Naive Bayes' in s and 'Accuracy' in s:
        if 'Naive Bayes' in s and 'Accuracy' in s and 'Precision' in s and 'Recall' in s:
            metrics_cell = s
    if report_cell is None and 'Classification Report' in s:
        report_cell = s

if metrics_cell:
    print('--- METRICS ---')
    for label in ['Accuracy', 'Precision', 'Recall']:
        m = re.search(label + r"\s*:\s*([0-9.]+)%", metrics_cell)
        if m:
            print(f'{label}% {m.group(1)}')

if report_cell:
    print('\n--- REPORT (head) ---')
    lines = report_cell.splitlines()
    # keep only the first ~45 non-empty lines to show it exists
    out = []
    for line in lines:
        if line.strip() == '' and len(out) == 0:
            continue
        out.append(line)
        if len(out) >= 45:
            break
    print('\n'.join(out))
else:
    print('No classification report found in saved outputs')
