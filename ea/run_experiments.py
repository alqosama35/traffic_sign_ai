"""Entry point: run all 4 mandatory GA experiments and save JSON logs.

Usage:
    python -m ea.run_experiments          # recommended
    python ea/run_experiments.py          # also works

Fault tolerance:
    - Per-experiment : if a completed JSON log already exists in ea/results/,
      that experiment is skipped automatically on restart.
    - Per-generation : after every generation a checkpoint file is saved to
      ea/results/checkpoint_<run_id>.npz.  If the process is interrupted the
      next run resumes from the last completed generation instead of starting
      over.  The checkpoint is deleted once the JSON log is successfully saved.
"""

import os
import sys

# Limit BLAS/OpenMP internal thread pools to 1 thread each.
# Must be set BEFORE numpy/sklearn are imported — these libraries read the
# variables once at import time and ignore later changes.
# With _N_JOBS=-1 (N Python threads) × 1 internal BLAS thread each = N total
# threads, cleanly mapping one thread per CPU core without oversubscription.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import time
from pathlib import Path

# Make `import ea` work when the script is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ea.experiment import get_extension_configs, get_mandatory_configs
from ea.ga import run_ga
from ea.logger import build_log, save_log

_DATA_DIR    = Path(__file__).resolve().parent / "data"
_RESULTS_DIR = Path(__file__).resolve().parent / "results"

_SUBSAMPLE_SIZE = 5000   # training rows per fitness eval (was 8000 — stratified sampling)
_N_JOBS         = -1    # use all CPU cores via ThreadPoolExecutor (was 1)


def _load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    X_train = np.load(_DATA_DIR / "X_train.npy")
    y_train = np.load(_DATA_DIR / "y_train.npy")
    X_val   = np.load(_DATA_DIR / "X_val.npy")
    y_val   = np.load(_DATA_DIR / "y_val.npy")
    baseline_acc = float((_DATA_DIR / "baseline_accuracy.txt").read_text().strip())
    return X_train, y_train, X_val, y_val, baseline_acc


def _try_s3_upload(log_path: Path, run_id: str) -> None:
    try:
        from ea.upload_s3 import upload_log  # Phase 7
        url = upload_log(str(log_path), run_id)
        print(f"  Uploaded -> {url}")
    except Exception as exc:
        print(f"  S3 upload skipped: {exc}")


def _box(lines: list[str], width: int = 62) -> str:
    top  = "+" + "=" * width + "+"
    bot  = "+" + "=" * width + "+"
    body = "\n".join("|  " + line.ljust(width - 2) + "|" for line in lines)
    return f"{top}\n{body}\n{bot}"


def _apply_or_load_filter(
    X_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, "np.ndarray | None"]:
    """Apply variance pre-filter or load cached result.

    Returns (X_train_filtered, X_val_filtered, filter_mask).
    filter_mask is None when the original dimension is already <= 900.
    """
    from ea.extract_features import apply_variance_filter

    _FILTER_TARGET = 900
    filter_mask_path      = _DATA_DIR / "feature_filter_mask.npy"
    X_train_filtered_path = _DATA_DIR / "X_train_filtered.npy"
    X_val_filtered_path   = _DATA_DIR / "X_val_filtered.npy"

    orig_dim = X_train.shape[1]
    if orig_dim <= _FILTER_TARGET:
        print(f"  Pre-filter skipped: original dim {orig_dim} <= target {_FILTER_TARGET}")
        return X_train, X_val, None

    if X_train_filtered_path.exists() and filter_mask_path.exists():
        filter_mask = np.load(filter_mask_path)
        X_tr_f      = np.load(X_train_filtered_path)
        X_va_f      = np.load(X_val_filtered_path)
        print(
            f"  Loaded cached pre-filtered features: "
            f"{int(filter_mask.sum())} / {orig_dim} kept"
        )
        return X_tr_f, X_va_f, filter_mask

    print(f"  Applying variance pre-filter (top {_FILTER_TARGET} / {orig_dim} features)...")
    X_tr_f, filter_mask = apply_variance_filter(X_train, target=_FILTER_TARGET)
    X_va_f = X_val[:, filter_mask]
    np.save(X_train_filtered_path, X_tr_f)
    np.save(X_val_filtered_path,   X_va_f)
    np.save(filter_mask_path,      filter_mask)
    removed = orig_dim - int(filter_mask.sum())
    print(
        f"  Pre-filter done: {int(filter_mask.sum())} kept, "
        f"{removed} removed ({removed / orig_dim:.1%}) — cached to data/"
    )
    return X_tr_f, X_va_f, filter_mask


def main() -> None:
    actual_workers = os.cpu_count() if _N_JOBS == -1 else max(1, _N_JOBS)

    print(_box([
        "EA Feature Selection -- Experiment Runner",
        "",
        f"Subsample size  : {_SUBSAMPLE_SIZE:,} / ~33k rows (~{_SUBSAMPLE_SIZE/33318:.0%}) — stratified",
        f"Parallel workers: {actual_workers} threads (n_jobs={_N_JOBS})",
        f"Configs to run  : 4 mandatory + 2 extension = 6 total (FR-EA-12, FR-EA-09b, FR-EA-10b)",
        "",
        f"OMP_NUM_THREADS    = {os.environ['OMP_NUM_THREADS']}  (limits OpenMP thread pool)",
        f"OPENBLAS_NUM_THREADS = {os.environ['OPENBLAS_NUM_THREADS']}  (limits OpenBLAS thread pool)",
        f"MKL_NUM_THREADS    = {os.environ['MKL_NUM_THREADS']}  (limits MKL thread pool)",
        "",
        "Fault tolerance : checkpoints saved after every generation",
        "                  completed experiments auto-skipped on restart",
    ]))

    print("\nLoading feature arrays...")
    X_train, y_train, X_val, y_val, baseline_acc = _load_features()
    print(f"  Train : {X_train.shape}  ({X_train.nbytes / 1e6:.0f} MB)")
    print(f"  Val   : {X_val.shape}  ({X_val.nbytes / 1e6:.0f} MB)")

    # ── Variance pre-filter ───────────────────────────────────────────────────
    X_train, X_val, filter_mask = _apply_or_load_filter(X_train, X_val)

    n           = X_train.shape[1]
    sigma_share = 0.2 * n

    print(f"  n={n}  sigma_share={sigma_share:.0f}  baseline_acc={baseline_acc:.4f}")

    configs = get_mandatory_configs() + get_extension_configs()
    for cfg in configs:
        cfg.sigma_share    = sigma_share
        cfg.subsample_size = _SUBSAMPLE_SIZE

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows   = []
    t_total_start  = time.time()

    for exp_num, cfg in enumerate(configs, start=1):
        log_path        = _RESULTS_DIR / f"{cfg.run_id}.json"
        checkpoint_path = _RESULTS_DIR / f"checkpoint_{cfg.run_id}.npz"

        # ── Level 2: skip completed experiments ───────────────────────────────
        if log_path.exists():
            print("\n" + _box([
                f"Experiment {exp_num} / {len(configs)} -- SKIPPED (already complete)",
                cfg.run_id,
                f"JSON log found: {log_path.name}",
            ]))
            existing = json.loads(log_path.read_text())
            summary_rows.append({
                "run_id"         : cfg.run_id,
                "best_fitness"   : existing["best_fitness"],
                "reduction_ratio": existing["reduction_ratio"],
                "selected"       : existing["selected_features"],
                "generations_run": existing["generations_run"],
                "run_time_min"   : existing.get("run_time_seconds", 0) / 60,
                "skipped"        : True,
            })
            continue

        print("\n" + _box([
            f"Experiment {exp_num} / {len(configs)}",
            cfg.run_id,
            "",
            f"Selection : {cfg.selection:<12}  Crossover : {cfg.crossover}",
            f"Mutation  : {cfg.mutation:<12}  Survivor  : {cfg.survivor}",
            f"Pop size  : {cfg.pop_size:<12}  Max gens  : {cfg.max_generations}",
            f"Subsample : {cfg.subsample_size:<12}  sigma_share: {cfg.sigma_share:.0f}",
            f"Seed      : {cfg.seed:<12}  Workers   : {actual_workers}",
            "",
            f"Checkpoint: {checkpoint_path.name}",
        ]))

        t0 = time.time()

        # ── Level 1: pass checkpoint path so run_ga can save/resume ──────────
        result = run_ga(
            X_train, y_train, X_val, y_val, cfg,
            verbose=True,
            n_jobs=_N_JOBS,
            checkpoint_path=checkpoint_path,
        )

        run_time = time.time() - t0

        log      = build_log(cfg, result, baseline_acc, run_time, filter_mask=filter_mask)
        log_path = save_log(log, str(_RESULTS_DIR))

        # ── Delete checkpoint only after JSON is safely written ───────────────
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"  Checkpoint removed: {checkpoint_path.name}")

        print("\n" + _box([
            f"Experiment {exp_num} complete",
            "",
            f"Best fitness    : {result['best_fitness']:.4f}",
            f"Baseline acc    : {baseline_acc:.4f}",
            f"Accuracy drop   : {baseline_acc - result['best_fitness']:.4f} pp",
            f"Selected feats  : {result['selected_features']:,} / {result['total_features']:,}"
            f"  ({result['reduction_ratio']:.1%} reduction)",
            f"Generations run : {result['generations_run']}",
            f"Run time        : {run_time/60:.1f} min",
            "",
            f"Saved -> {log_path.name}",
        ]))

        _try_s3_upload(log_path, cfg.run_id)

        summary_rows.append({
            "run_id"         : cfg.run_id,
            "best_fitness"   : result["best_fitness"],
            "reduction_ratio": result["reduction_ratio"],
            "selected"       : result["selected_features"],
            "generations_run": result["generations_run"],
            "run_time_min"   : run_time / 60,
            "skipped"        : False,
        })

    # ── Final summary ─────────────────────────────────────────────────────────
    total_min = (time.time() - t_total_start) / 60
    col = 52
    print("\n" + "=" * (col + 36))
    print(f"  {'FINAL SUMMARY':^{col + 32}}")
    print("=" * (col + 36))
    header = f"  {'run_id':<{col}} {'best':>6}  {'reduc':>6}  {'sel':>5}  {'gens':>4}  {'min':>5}  {'status'}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in summary_rows:
        passes = r["best_fitness"] >= 0.88 and r["reduction_ratio"] >= 0.30
        flag   = "[OK]" if passes else "[--]"
        status = "skipped" if r.get("skipped") else "done"
        print(
            f"  {r['run_id']:<{col}} {r['best_fitness']:>6.4f}  "
            f"{r['reduction_ratio']:>6.2%}  {r['selected']:>5d}  "
            f"{r['generations_run']:>4d}  {r['run_time_min']:>5.1f}  {flag} {status}"
        )
    print(f"\n  Total wall time: {total_min:.1f} min")
    print("=" * (col + 36))


if __name__ == "__main__":
    main()
