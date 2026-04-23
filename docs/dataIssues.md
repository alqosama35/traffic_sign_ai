# Data Pipeline Review — `setup_dataset.py` & `EDA/EDA.ipynb`

---

## setup_dataset.py

### CRITICAL

#### 1. `class_id` Stored as Float in Annotations CSV
**Location:** `build_annotation()` — the reverse-mapping loop sets `class_id = None` as default.  
**Problem:** When any `class_id` stays `None`, Pandas casts the entire column to `float64`, so every ID becomes `35.0` instead of `35`. This is confirmed in the EDA output (`class_id: 35.0`).  
**Fix:**
```python
# After the reverse-mapping loop, raise an error instead of silently writing None
if class_id is None:
    print(f"WARNING: no class_id found for folder '{class_name}' — skipping")
    continue
```
Also cast explicitly before saving:
```python
df["class_id"] = df["class_id"].astype(int)
```

---

#### 2. Silent Data Loss on Image Errors
**Location:** `resize_image()` — `except Exception as e: print(...)` swallows all errors.  
**Problem:** A corrupted or missing image is silently skipped, leaving a broken entry in the filesystem with no indication of how many images were lost.  
**Fix:** Collect failures and report a summary at the end:
```python
failed = []
# inside resize_image, append to failed instead of just printing
# after the loop: if failed: print(f"WARNING: {len(failed)} images failed")
```

---

#### 3. `shutil.rmtree("dataset")` Without Safeguard
**Location:** `prepare_data()` line 189.  
**Problem:** Unconditionally deletes the entire `dataset/` directory. Re-running the script destroys a completed processed dataset with no confirmation or backup.  
**Fix:** Add a guard flag or prompt:
```python
if os.path.exists("dataset"):
    print("WARNING: 'dataset/' already exists and will be deleted. Press Ctrl+C to cancel.")
    import time; time.sleep(5)
    shutil.rmtree("dataset")
```

---

### HIGH

#### 4. Hardcoded Windows-Capitalized Paths Are OS-Sensitive
**Location:** `prepare_data()` — `"Train"` and `"Test"` are capitalized to match the Kaggle download layout.  
**Problem:** On Linux/macOS (or if the Kaggle layout changes) the paths fail silently.  
**Fix:** Use `os.path.join` with a case-insensitive search or document the exact expected folder structure prominently.

---

#### 5. `build_annotation` Has No File Extension Filter
**Location:** `build_annotation()` — iterates `os.listdir(class_path)` with no filter.  
**Problem:** Any non-image file (`.DS_Store`, `thumbs.db`, temp files) will be included as a row in `annotations.csv`.  
**Fix:**
```python
VALID_EXTS = {".png", ".jpg", ".jpeg", ".ppm"}
for img in os.listdir(class_path):
    if Path(img).suffix.lower() not in VALID_EXTS:
        continue
```

---

#### 6. `clean_name` Can Produce Collisions
**Location:** `clean_name()`.  
**Problem:** `"No passing >3.5t"` → `"No_passing_3.5t"` and a hypothetical class `"No passing 3.5t"` would produce the same folder name. The current class list has no collision, but the function is fragile with no assertion.  
**Fix:** Add a startup assertion that no two cleaned names collide:
```python
cleaned = [clean_name(v) for v in CLASS_NAMES.values()]
assert len(cleaned) == len(set(cleaned)), "Cleaned name collision detected"
```

---

#### 7. Unused Variables in `split_train_val`
**Location:** Lines 69–70 — `train_out` and `val_out` are defined but never used; the actual paths are re-constructed inside the loop via string concatenation.  
**Fix:** Remove `train_out`/`val_out` or use them consistently inside the loop.

---

### MEDIUM

#### 8. No Progress Bar for the Resize Loop
**Location:** `split_train_val()` and `organize_test()`.  
**Problem:** Resizing ~50k images gives no feedback; the script appears frozen for several minutes.  
**Fix:** Add `tqdm`:
```python
from tqdm import tqdm
for img in tqdm(img_list, desc=f"{split}/{class_name}"):
```

---

#### 9. `print()` Instead of `logging`
**Problem:** All status messages use bare `print()`, making it impossible to control verbosity or route output to a log file without code changes.  
**Fix:** Replace with `logging.getLogger(__name__)` and set the level via config.

---

#### 10. Duplicate Section Numbering
**Location:** Comments label both `build_annotation` and `prepare_data` as `# 5)`.  
**Fix:** Renumber to `# 5)` and `# 6)` respectively.

---

#### 11. `os.path.basename` in `organize_test` Is Fragile
**Location:** `organize_test()` — `img_name = os.path.basename(row["Path"])`.  
**Problem:** Assumes the CSV `Path` column always contains a bare filename. If Kaggle changes the column format to a relative path with directories (`Test/00000.png`), `basename` still works — but if it includes a duplicate directory prefix the file won't be found at `os.path.join(test_path, img_name)`.  
**Fix:** Assert the file exists before calling `resize_image`:
```python
if not os.path.exists(src):
    print(f"Missing: {src}")
    continue
```

---

## EDA/EDA.ipynb

### CRITICAL

#### 12. Hardcoded Absolute Path `D:/AML_Project`
**Location:** Cell 17 — `BASE_DIR = Path(r"D:/AML_Project")`.  
**Problem:** The notebook is completely non-portable. Anyone running it on a different machine or directory structure will get `FileNotFoundError` on every `cv2.imread` call.  
**Fix:** Derive the base path dynamically:
```python
BASE_DIR = Path.cwd().parent  # or use __file__ if converted to .py
```
Or read from an environment variable / config file.

---

#### 13. `cv2.imread` Can Return `None` → Crash in `show_samples`
**Location:** `show_samples()` — `img = cv2.imread(row.image_path)` followed immediately by `cv2.cvtColor(img, ...)`.  
**Problem:** If the path is wrong (e.g., due to the `D:/AML_Project` hardcoding being off), `imread` returns `None` and the next line throws `cv2.error: NULL pointer`.  
**Fix:**
```python
img = cv2.imread(row.image_path)
if img is None:
    print(f"Cannot read: {row.image_path}")
    continue
```

---

### HIGH

#### 14. `df.sample()` Without `random_state` — Non-Reproducible
**Location:** `show_samples(df)` and `df["image_path"].sample(500)` in the size distribution cell.  
**Problem:** Every kernel restart gives a different sample, making the notebook non-reproducible.  
**Fix:** Pass `random_state=42` to both `.sample()` calls.

---

#### 15. Fragile Path Deduplication Workaround
**Location:** Cell 17 — `df["image_path"].str.replace("dataset/dataset", "dataset")`.  
**Problem:** This is a workaround for the `image_path` column bug in `annotations.csv` (duplicate `dataset/` prefix from `build_annotation`). The root cause is that `build_annotation` concatenates `dataset_path` with a path that already includes `dataset/` in `split_path`. Fixing the root cause removes the need for this hack.  
**Root Fix in `build_annotation`:** Use relative paths (e.g., `split/class_name/img`) and let consumers resolve to absolute.

---

### MEDIUM

#### 16. Imbalance Analysis Is Incomplete
**Problem:** The notebook identifies an 11x imbalance ratio but offers no mitigation analysis (e.g., which classes are underrepresented, whether the split preserves the ratio, or whether augmentation/oversampling is recommended). This leaves the findings without actionable conclusions.

---

#### 17. Most Plots Are Not Saved to Disk
**Problem:** Only in-memory display via `plt.show()`. Reproducibility and reporting require saved artifacts.  
**Fix:** Add `plt.savefig("outputs/eda/<name>.png", dpi=150, bbox_inches="tight")` before each `plt.show()`.

---

#### 18. No Markdown Findings — Only Section Headers
**Problem:** Every markdown cell is just a title (e.g., `# Top 10 Classes`). There are no written observations, no stated conclusions, and no data quality notes. An EDA notebook should document what was found, not just what was plotted.
