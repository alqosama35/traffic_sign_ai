# EA Engineer Tutorial — GA Feature Selection Engine

This tutorial is written for the **EA Engineer (Member 4)** implementing **FR-EA-01 → FR-EA-16** in the SRS.

Your deliverable is an **offline GA experiment engine** that:

- Takes **feature vectors** extracted from the trained MobileNetV2 (penultimate layer)
- Runs **multiple GA configurations** (selection × crossover, plus recommended variants)
- Logs each run as **JSON** (schema in SRS) and **uploads logs to S3**
- Demonstrates that GA can reduce feature dimensionality while keeping accuracy high

The GA engine is not part of the runtime API; it produces artifacts consumed by the dashboard (and stored in S3).

---

## 1) What you must deliver (mapped to SRS)

### Core (must-have)

- **Custom GA implementation (Python + NumPy)** — no DEAP/scipy optimizers (FR-EA-01)
- **Binary chromosome** length $n$ = feature dimension (FR-EA-02)
- **Fitness** = validation accuracy of a fast classifier (k-NN or linear SVM) on selected feature subset (FR-EA-03)
- **Population** size 50, random init (FR-EA-04)
- **Selection**: Tournament (k=3) and Roulette Wheel (FR-EA-05, FR-EA-06)
- **Crossover**: Single-point and Uniform (FR-EA-07, FR-EA-08)
- **Mutation**: bit-flip, $p=0.01$ per bit (FR-EA-09)
- **Termination**: max 100 generations OR 20 gens without improvement (FR-EA-10)
- **Fitness sharing** to maintain diversity (FR-EA-11)
- **Run at least 4 configurations**: {Tournament, Roulette} × {Single-point, Uniform} (FR-EA-12)
- **JSON log per run** with the exact fields requested (FR-EA-13)
- **Upload logs to S3 prefix** `ea-experiments/` (FR-EA-14)
- **Hit targets**: ≥30% reduction and ≥88% validation accuracy, with <3pp drop vs full features (FR-EA-15, FR-EA-16)

### Recommended (do if time permits)

- **Second mutation operator** + ≥2 additional experiment configurations comparing it vs bit-flip (FR-EA-09b)
- **Two survivor selection strategies** (e.g., generational vs elitism/steady-state) and compare their impact (FR-EA-10b)

---

## 2) Dependencies + how EA fits the whole system

### What you need from the ML Engineer (Member 3)

You cannot start GA until you have:

- A trained MobileNetV2 checkpoint (SRS: `traffic_sign_model_mobilenetv2.pth` uploaded to S3)

In this repo today, the API loads a MobileNetV2 checkpoint from:

- `training/outputs/model/mobilenetv2_inference.pth`

That file can be used locally for feature extraction as long as the architecture matches.

### What you hand off to Cloud/Dashboard (Member 7)

- 4 JSON logs (minimum) in `s3://<bucket>/ea-experiments/` following the SRS schema.
- Optional: extra logs for recommended experiments (clearly named so the dashboard can include/exclude them).

---

## 3) Repo-specific starting point (what already exists)

These are useful to you as the EA engineer:

- Dataset prep script: `setup_dataset.py`
  - Creates `dataset/train`, `dataset/val`, `dataset/test`
  - Writes `dataset/annotations.csv` (image path + split + class_id)
- Training outputs (for AML reporting): `training/outputs/logs/config.json` and `training/outputs/logs/report.json`
- API model loading logic (shows class naming conventions and MobileNetV2 head replacement): `api/app.py`

There is **no** `ea/` package yet in the repo—so your tutorial-level plan should assume you will add one.

---

## 4) The big idea: make GA fast by precomputing features

The performance constraint **NFR-PERF-04** says GA evaluation per generation must finish in <60s. If your fitness evaluation requires running the deep model repeatedly, you’ll miss that.

Best-practice approach:

1. **Precompute feature vectors once** for train/val splits using MobileNetV2.
2. Store features + labels on disk (e.g., `.npz`) so each GA fitness evaluation becomes:
   - Slice columns by chromosome mask
   - Train a small classifier (k-NN or linear model)
   - Evaluate on validation features

This is also consistent with the SRS definition of “Feature Vector”.

---

## 5) Step-by-step implementation guide

### Step A — Extract feature vectors (FR-EA-02 prerequisite)

You need a function that returns the **penultimate layer activations**.

For torchvision MobileNetV2, the typical feature dimension is `model.last_channel` (often 1280), but **do not hard-code it**; infer it from the model.

Minimal pattern (conceptual):

```python
# Idea only: return penultimate features (before classifier)
features = model.features(x)        # conv backbone
features = features.mean([2, 3])    # global average pool -> (B, C)
# features.shape[1] == n (chromosome length)
```

Key decisions:

- Use the **same preprocessing** as training/API (`Resize(224,224)` + ImageNet normalization).
- Extract features for **train** and **val** splits (test optional for final reporting).
- Save as something like:
  - `X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`
  - or a single `features_train_val.npz`

Why this is needed:

- GA needs $n$ (feature dimension) to define chromosome length (FR-EA-02)
- Fitness evaluation must be lightweight (FR-EA-03, NFR-PERF-04)

### Step B — Compute a “full features” baseline (needed for FR-EA-16)

Before you run GA, establish baseline validation accuracy using **all features**.

- Train the fitness classifier on full `X_train`
- Evaluate on full `X_val`
- Store as `full_feature_val_accuracy`

Why this matters:

- FR-EA-16 requires your GA solution to be within <3 percentage points of this baseline.

### Step C — Implement the GA core (FR-EA-01 → FR-EA-11)

#### Chromosome representation (FR-EA-02)

- Chromosome is a NumPy binary vector: `shape = (n,)`, dtype `bool` or `uint8`.
- `1` means keep the feature; `0` means drop it.

Practical constraint (strongly recommended):

- Disallow “all zeros” chromosomes (they select no features). If it happens, force at least one bit to 1.

#### Population init (FR-EA-04)

- Population size: 50
- Initialize uniformly at random.

Implementation detail:

- “Uniform at random” can mean each bit is Bernoulli(0.5). If you do that, expected features selected is 50%.
- If you want to bias towards stronger reduction, you *can* use a lower probability (e.g., 0.3) **but document it** (and keep it consistent across runs for fair comparison).

#### Fitness function (FR-EA-03)

Fitness is **validation accuracy** of a quick classifier trained on selected features:

- k-NN: simple, but can get slow if dataset is large.
- linear SVM: fast if implemented with a linear solver.

SRS allows “k-NN or linear SVM”. A practical choice is:

- `sklearn.svm.LinearSVC` or `sklearn.linear_model.SGDClassifier(loss="hinge")`

Important: FR-EA-01 bans black-box **optimization libraries for GA**, but using scikit-learn for the **classifier inside fitness** is usually acceptable (it’s not doing the GA). If your instructor is strict, confirm this assumption early.

Fitness evaluation pseudocode:

```python
mask = chromosome.astype(bool)
Xtr = X_train[:, mask]
Xva = X_val[:, mask]
clf.fit(Xtr, y_train)
acc = accuracy(clf.predict(Xva), y_val)
return acc
```

#### Selection operators (FR-EA-05, FR-EA-06)

- Tournament (k=3): pick 3 random individuals, choose the best.
- Roulette: choose proportionally to fitness.

Note: roulette is fragile if many fitness values are equal/near zero. Stabilize by shifting:

- `p_i = (f_i - f_min + eps) / sum(...)`

#### Crossover operators (FR-EA-07, FR-EA-08)

- Single-point crossover
- Uniform crossover (each bit swapped with probability 0.5)

#### Mutation (FR-EA-09)

- Bit-flip with $p=0.01$ per bit.

#### Termination (FR-EA-10)

Stop when either:

- generation == 100
- OR best fitness hasn’t improved for 20 consecutive generations

Make sure you log `generations_run` correctly in JSON.

#### Fitness sharing (FR-EA-11)

Goal: prevent premature convergence by reducing the effective fitness of crowded niches.

Common method:

- Distance: Hamming distance $d(i,j)$ between chromosomes
- Sharing function:

$$
sh(d) = \begin{cases}
1 - \left(\frac{d}{\sigma_{share}}\right)^\alpha & d < \sigma_{share}\\
0 & \text{otherwise}
\end{cases}
$$

Shared fitness:

$$
\tilde{f}_i = \frac{f_i}{\sum_j sh(d(i,j))}
$$

Where:

- $\sigma_{share}$ controls “niche radius” (try 0.1–0.3 of $n$ as a starting point)
- $\alpha$ often set to 1

Use **shared fitness** for selection probability (roulette) and/or tournament comparisons.

### Step D — Survivor selection strategy (ties into FR-EA-10b)

Even if you don’t do the recommended comparison, you must still decide how the next generation is formed.

Two common strategies:

1. **Generational replacement**: parents -> children, replace whole population
2. **Elitism**: keep top `E` individuals from current gen, fill the rest with offspring

If you implement both, you satisfy FR-EA-10b (recommended) by running the same configuration twice and comparing convergence.

### Step E — Run the required 4 experiments (FR-EA-12)

You must run exactly these base configurations:

1. Tournament + Single-point crossover
2. Tournament + Uniform crossover
3. Roulette + Single-point crossover
4. Roulette + Uniform crossover

Keep everything else identical across them:

- population size = 50
- mutation rate = 0.01
- termination rule
- fitness sharing settings
- classifier choice + hyperparameters

This makes comparisons fair.

### Step F — Logging + JSON schema (FR-EA-13)

For each experiment, write one JSON file matching the SRS schema:

Required fields:

- `run_id`
- `selection`
- `crossover`
- `population_size`
- `generations_run`
- `fitness_per_generation` (list[float])
- `best_fitness`
- `best_chromosome` (list[int])
- `selected_feature_indices` (list[int])
- `total_features`
- `selected_features`
- `reduction_ratio`

Highly recommended additions (keep backward-compatible):

- `mutation`: e.g., `bit_flip`
- `mutation_rate`: 0.01
- `survivor_selection`: `generational` / `elitism`
- `fitness_sharing`: config object with `sigma_share`, `alpha`
- `full_feature_val_accuracy` and `accuracy_drop_pp`
- `seed`

As long as you keep all required keys, adding extra keys won’t break a robust dashboard.

### Step G — Upload logs to S3 (FR-EA-14 + NFR-MAINT-03)

SRS requires logs uploaded under:

- `s3://<bucket>/ea-experiments/`

Do not hard-code bucket names/credentials in code.

Recommended environment variables:

- `S3_BUCKET`
- `AWS_REGION` (optional)

Upload should happen **after** the JSON file is fully written (NFR-REL-02).

---

## 6) Acceptance checklist (what your supervisor will look for)

### SRS compliance

- GA written in your own code (FR-EA-01)
- Correct operator implementations and parameters (FR-EA-04…FR-EA-10)
- Fitness sharing implemented and used (FR-EA-11)
- Exactly 4 required base runs completed (FR-EA-12)
- JSON logs conform to schema and are uploaded to S3 (FR-EA-13, FR-EA-14)

### Quantitative targets

- Best GA configuration reaches **≥88% validation accuracy** (FR-EA-15)
- Feature reduction is **≥30%** (FR-EA-15)
- Accuracy drop vs full-feature baseline is **<3 percentage points** (FR-EA-16)

### Evidence of “EA-ness”

- Convergence curves show improvement over generations (EA acceptance criteria)
- Diversity mechanism visibly helps (fitness sharing; and recommended mutation/survivor comparisons strengthen this)

---

## 7) Common pitfalls (and how to avoid them)

- **Overfitting the validation set:** GA directly optimizes val accuracy. To reduce risk:
  - keep a held-out test split for final reporting only
  - or use cross-validation inside fitness (but this may violate time budget)
- **Fitness evaluation too slow:** precompute features; consider a linear classifier over k-NN.
- **Roulette wheel instability:** protect against near-equal fitness by shifting/scaling.
- **All-zero chromosome:** enforce at least 1 selected feature.
- **Non-reproducible results:** set and log seeds for NumPy (and scikit-learn where applicable).

---

## 8) Minimal experiment naming convention (so the dashboard stays clean)

Suggested `run_id` format (human + machine readable):

- `<selection>_<crossover>_<survivor>_<mutation>_<seed>`

Examples:

- `tournament_singlepoint_generational_bitflip_42`
- `roulette_uniform_elitism_bitflip_42`

This makes it easy for Member 7 to map each log to the right chart line.
