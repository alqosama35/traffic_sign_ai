# Implementation Plan — Traffic Sign AI Dashboard

## Goal
Build a vanilla HTML/JS dashboard served by FastAPI that displays GA convergence charts, model evaluation results, a three-way model comparison table, and a live prediction tab — with all data fetched from backend relay endpoints that read from local files in dev and from S3 in prod.

## Constraints & assumptions
- Vanilla HTML/CSS/JS only; Chart.js via CDN for charts; no build step
- Dashboard served by FastAPI as StaticFiles at `/dashboard`; all fetch calls use relative paths
- Browser never touches `X-API-Key` or S3 directly; backend relay endpoints handle both
- `DATA_SOURCE` env var (`local` | `s3`) controls data origin; `local` is the default for development
- When `DATA_SOURCE=s3`, reads use boto3 with the bucket name from a new `S3_BUCKET` env var
- All 4 EA experiment JSON logs already exist in `ea/results/`; MobileNetV2 training artifacts exist in `training/outputs/` (`cm.png`, `curves.png`, `report.json`)
- CV baseline metrics sourced from the notebook output (`cv/cv_project.ipynb`); the notebook ran on 10 classes (0–9) with HOG + color histogram features — not the full 43-class GTSRB test set; this caveat must be labeled clearly on the dashboard
- Existing `api/app.py` is not refactored; new code is added as an included router and a StaticFiles mount
- FastAPI route registration order ensures API routes at `/dashboard/data/*` are matched before the StaticFiles mount at `/dashboard`

## Architecture overview
FastAPI registers a new `dashboard_router` (prefix `/dashboard`) that exposes five relay endpoints; these endpoints abstract the data source (local disk or S3) behind a simple env-var switch. The existing `/predict` function is called directly from the relay — no HTTP sub-request — so the API key stays server-side. After registering the router, the app mounts the `dashboard/` directory as StaticFiles at `/dashboard`, so the browser can load `index.html` and all assets from the same origin as the API.

## Components

### 1. Dashboard router (`api/dashboard_router.py`)
- **Responsibility:** five relay endpoints the browser calls; isolates all data-fetching logic from `app.py`
- **Inputs / Outputs:**
  - `GET /dashboard/data/ea` → JSON array of all 4 EA run objects (strips `best_chromosome` to keep payload small; ~5 KB vs ~25 KB)
  - `GET /dashboard/data/training-report` → JSON with `accuracy`, `macro avg`, `weighted avg`, and per-class metrics (read directly from `training/outputs/logs/report.json`)
  - `GET /dashboard/data/cv-metrics` → JSON read from `dashboard/data/cv_metrics.json` fixture
  - `GET /dashboard/data/confusion-matrix.png` → `FileResponse` (local) or `StreamingResponse` from `boto3.get_object` (S3)
  - `POST /dashboard/predict` → accepts `multipart/form-data` image, calls the same `predict()` logic used by `/predict`, returns identical response shape; no `X-API-Key` required from the browser
- **Key decisions:** router is imported and registered in `app.py` with `app.include_router(dashboard_router)`; a single `_read_json(local_path, s3_key)` helper switches between disk and S3 based on `DATA_SOURCE` env var

### 2. CV metrics fixture (`dashboard/data/cv_metrics.json`)
- **Responsibility:** checked-in ground truth for CV baseline numbers extracted from the notebook; decouples the dashboard from the notebook runtime
- **Inputs / Outputs:** read at request time by the router; returned verbatim to the browser
- **Key decisions:** include a `"scope"` field (`"10 classes (0–9), 80 samples/class, HOG + ColorHistogram features"`) so the dashboard can render the footnote automatically; include filter metrics, SIFT matching accuracy, and K-Means IoU per class so all CV results are in one place

### 3. Dashboard shell (`dashboard/index.html`)
- **Responsibility:** single HTML file; hosts four tab panels and loads all JS/CSS
- **Key decisions:** tabs controlled by CSS class toggling in vanilla JS; Chart.js 4.x loaded from CDN; no routing library; each tab's JS module is a separate file loaded via `<script type="module">`

### 4. GA tab (`dashboard/js/ga.js`)
- **Responsibility:** fetch `/dashboard/data/ea`, render Chart.js multi-series line chart (4 configs on one chart) and feature reduction table
- **Inputs / Outputs:** EA JSON array in → chart + HTML table out
- **Key decisions:** one dataset per run, labeled by config name (e.g. "Tournament × Single-Point"); table columns: Config | Full Val Acc | GA Val Acc | Features Used | Reduction | Acc Drop

### 5. Model results tab (`dashboard/js/model.js`)
- **Responsibility:** display confusion matrix image and MobileNetV2 summary stats
- **Key decisions:** `<img src="/dashboard/data/confusion-matrix.png">` — no JS chart needed, preserves the pre-generated 43×43 heatmap exactly; overall accuracy, macro F1, and test sample count pulled from `/dashboard/data/training-report` and rendered as stat cards

### 6. Comparison tab (`dashboard/js/comparison.js`)
- **Responsibility:** render three-way comparison table: CV Baseline vs MobileNetV2 vs GA-Optimized
- **Key decisions:** client-side `Promise.all` fetches training-report and cv-metrics simultaneously; GA row uses best `best_fitness` across all 4 runs; CV row shows the 10-class scope footnote inline

### 7. Live prediction tab (`dashboard/js/predict.js`)
- **Responsibility:** file input → `POST /dashboard/predict` → render result card
- **Key decisions:** plain `<input type="file">` (no drag-and-drop); show predicted class name, category badge, percentage confidence bar, top-3 list, and the safety disclaimer from the API response

## Implementation phases

### Phase 1 — Data relay endpoints & StaticFiles mount (S)
- Create `api/dashboard_router.py` with the `_read_json(local_path, s3_key)` helper and all five endpoints
- Add `DATA_SOURCE=local` and `S3_BUCKET=` to `.env.example`
- In `api/app.py`: `app.include_router(dashboard_router)`, then `app.mount("/dashboard", StaticFiles(directory="dashboard", html=True))`
- Create `dashboard/` directory with placeholder `index.html` so the mount doesn't error on startup

Done-when: `curl /dashboard/data/ea` returns a JSON array with 4 objects; `curl /dashboard/data/confusion-matrix.png` returns a PNG binary.

### Phase 2 — CV metrics fixture (S)
- Create `dashboard/data/cv_metrics.json` with the following values from the notebook:
  - Overall accuracy 95.62%, macro precision 95.74%, macro recall 95.62%
  - Per-class F1 for all 10 classes
  - Filter metrics: Gaussian (PSNR 24.54 dB, SSIM 0.8162), Median (PSNR 22.21 dB, SSIM 0.6739)
  - SIFT matching accuracy 5.9% (1 good match / 17 total)
  - K-Means IoU: Speed 20 → 0.44, Speed 30 → 0.67, Speed 50 → 0.49, Speed 60 → 0.38, Speed 70 → 0.59
  - `"scope"` field as described above

Done-when: `GET /dashboard/data/cv-metrics` returns this JSON with the scope field present.

### Phase 3 — Dashboard shell & navigation (S)
- `dashboard/index.html`: four tab buttons + four `<section>` panels; Chart.js CDN `<script>` tag
- `dashboard/css/style.css`: minimal academic look (no CSS framework)
- Tab switching logic in `dashboard/js/tabs.js`: toggle `active` class on sections and buttons

Done-when: `/dashboard/` loads in browser; four tabs switch panels correctly; no JS errors in console.

### Phase 4 — GA convergence tab (S)
- `dashboard/js/ga.js`: fetch EA data, build Chart.js line chart, build HTML table and insert into DOM
- Each series labeled by `selection` + `crossover` from the JSON; x-axis is generation index, y-axis is fitness (0–1)

Done-when: tab shows a chart with 4 labeled series and a table with 4 rows of correct numbers.

### Phase 5 — Model results tab (S)
- `dashboard/js/model.js`: set `<img>` src to confusion matrix URL; fetch training-report; render three stat cards (overall accuracy, macro F1, test samples)

Done-when: confusion matrix image displays; stat cards show 95.26% overall accuracy and 93.83% macro F1.

### Phase 6 — Comparison tab (S)
- `dashboard/js/comparison.js`: `Promise.all` both endpoints, build three-row HTML table; attach footnote to CV row

Done-when: three-row table shows CV Baseline (95.62%, 10-class†), MobileNetV2 (95.26%, 43-class), GA-Optimized (~98.05% val, 43-class features).

### Phase 7 — Live prediction tab (M)
- `dashboard/js/predict.js`: wire file input change event → `FormData` POST → parse response → render result card
- Result card: sign name (`class`), category badge (colour-coded by category string), confidence bar, top-3 list, disclaimer paragraph

Done-when: uploading any GTSRB test image renders a complete result card with no JS errors; uploading a non-image returns a readable error message.

## What to cut first
1. Training curves image (`curves.png`) on the Model Results tab — `cm.png` is the SRS requirement; curves are informational only
2. Per-class F1 table on the Model Results tab — the confusion matrix image already conveys this; add only if the comparison tab is complete with time to spare
3. K-Means IoU detail table on the CV tab — include only the headline accuracy in the comparison table if time is tight; the IoU data is in the fixture and can be added later

## Open questions
1. **CV scope mismatch:** The notebook ran on 10 classes with a training-set split — not the full 43-class GTSRB test set used by MobileNetV2. The dashboard will label the CV row clearly, but decide whether to re-run the CV pipeline on all 43 classes before the demo for a proper apples-to-apples comparison.
2. **report.json top-5 accuracy field:** The training report has `accuracy` and per-class metrics but no `top_5_accuracy` key visible. Confirm whether the training notebook also saved this value, or whether the stat card should omit it.
3. **S3 paths for images:** `cm.png` and `curves.png` are currently only on local disk. If `DATA_SOURCE=s3` must serve the confusion matrix, the training member needs to upload `training/outputs/plots/` to S3 and confirm the key prefix.
4. **S3 bucket name:** Confirm the agreed bucket name so `S3_BUCKET` can be documented in `.env.example` and the GitHub Actions secrets.
