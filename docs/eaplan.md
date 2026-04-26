# EA Engineer Implementation Plan
## Traffic Sign Intelligence Platform — Member 4 (FR-EA-01 → FR-EA-16)

**Date:** 2026-04-25  
**Role:** EA Engineer (Member 4)  
**Scope:** GA Feature Selection Engine — offline batch component  
**Source of truth:** `docs/srs.md` §3.4, §9.2, §11.2 (Member 4 role)

---

## Executive Summary

You will build a **custom Genetic Algorithm** in Python + NumPy (no DEAP, no scipy.optimize) that selects a subset of features extracted from the trained MobileNetV2 model. The GA reduces the feature dimension while keeping classification accuracy within 3 percentage points of the full-feature baseline.

**Your primary deliverables:**
- `ea/` Python package with all GA source code
- 4 mandatory JSON experiment logs (one per operator configuration)
- Those logs uploaded to `s3://<bucket>/ea-experiments/`
- Validation confirming all SRS acceptance criteria are met

**Your dependency chain:**
```
[M3: trained MobileNetV2 checkpoint] → [You: EA engine] → [M7: Dashboard]
```

The checkpoint already exists locally at `training/outputs/model/mobilenetv2_inference.pth`.

---

## Explicit Assumptions

The following assumptions were required because they cannot be derived directly from the codebase:

| # | Assumption | Justification |
|---|---|---|
| A1 | `sklearn` classifiers are permitted inside the fitness function | FR-EA-01 bans GA optimization libraries (DEAP, scipy.optimize), not classifiers. `eatutorial.md §5 Step C` explicitly states "using scikit-learn for the classifier inside fitness is usually acceptable". **Confirm with instructor before submission.** |
| A2 | Feature dimension `n = 1280` | MobileNetV2 `model.last_channel = 1280`. The plan does not hard-code this — it is always inferred from the loaded model. |
| A3 | LinearSVC on the full training set (33,318 samples) may exceed the 60-second/generation budget | Fitness budget per chromosome = 60s ÷ 50 = 1.2s. LinearSVC on 33k samples × 1280 features is potentially 2–5s. **Benchmark during Phase 3 and subsample if needed.** |
| A4 | `dataset/train` and `dataset/val` directories exist locally | Created by `setup_dataset.py`. If missing, run `python setup_dataset.py` before starting Phase 2. |
| A5 | The S3 bucket is created by Member 7 (Cloud engineer) before Phase 7 | EA engineer only uploads to the bucket; does not create it. Coordinate on bucket name via `S3_BUCKET` env variable. |
| A6 | Same seed (42) is used for all 4 mandatory experiments | Required for fair comparison across operator configurations. |

---

## Phase Overview

| Phase | Name | Key Output | Estimated Effort |
|---|---|---|---|
| 1 | Environment Setup | Working `ea/` structure | 0.5 day |
| 2 | Feature Extraction | `.npy` feature arrays + full-feature baseline | 1 day |
| 3 | GA Core Operators | All atomic GA building blocks | 2 days |
| 4 | Fitness Sharing & Diversity | Niche-based diversity preservation | 1 day |
| 5 | GA Loop, Runner & JSON Logging | 4 complete experiment runs + JSON logs | 2 days |
| 6 | Recommended Extensions ✅ | Bonus mutation + survivor selection variants | 1–2 days |
| 7 | S3 Upload & Cloud Integration | Logs uploaded to S3 | 0.5 day |
| 8 | Validation & Acceptance Testing | All SRS criteria confirmed; handoff artifacts ready | 1 day |

---

## Phase 1 — Environment Setup & Prerequisite Review

### a) Learning Requirements
- How to load a PyTorch `.pth` file and switch a model to eval mode
- NumPy array operations: boolean indexing, broadcasting
- Scikit-learn API: `fit`, `predict`, `accuracy_score`
- How the existing codebase is organized: `api/app.py` (model loading pattern), `training/outputs/` (model weights), `dataset/` (organized splits)
- SRS §3.4 (all EA requirements), §9.2 (acceptance criteria), §11.2 Member 4 role

### b) Implementation Tasks
1. Read `docs/eatutorial.md` end-to-end — it is the implementation guide for this role.
2. Read `api/app.py` — understand class naming conventions and how MobileNetV2 is loaded.
3. Verify checkpoint exists: `training/outputs/model/mobilenetv2_inference.pth` (9 MB).
4. Verify dataset exists: `dataset/train/`, `dataset/val/`, `dataset/test/`. If not, run `python setup_dataset.py`.
5. Create the `ea/` top-level directory with the following structure:

```
ea/
├── __init__.py
├── extract_features.py        # Phase 2
├── fitness.py                 # Phase 3
├── population.py              # Phase 3
├── ga.py                      # Phase 5 (main GA loop)
├── experiment.py              # Phase 5 (config + runner)
├── logger.py                  # Phase 5 (JSON log builder)
├── upload_s3.py               # Phase 7
├── validate.py                # Phase 8
├── operators/
│   ├── __init__.py
│   ├── selection.py           # Phase 3
│   ├── crossover.py           # Phase 3
│   ├── mutation.py            # Phase 3, 6
│   └── survivor.py            # Phase 4, 6
├── diversity.py               # Phase 4
├── data/                      # gitignored — stores precomputed features
│   └── .gitkeep
└── results/                   # gitignored — stores JSON logs
    └── .gitkeep
```

6. Install required packages:
   ```
   pip install numpy scikit-learn boto3
   # torch and torchvision already installed for the training module
   ```
7. Add `ea/data/` and `ea/results/` to `.gitignore` (feature arrays are large; JSON logs go to S3).

### c) Key Decisions
- **Directory placement**: `ea/` at the repository root, consistent with the SRS NFR-MAINT-01 pattern (`trainer/`, `api/`, `dashboard/`).
- **No `ea/requirements.txt`**: The EA module shares dependencies with the rest of the project. Pin all versions in the root `requirements.txt`.

### d) Deliverables
- `ea/` directory with all subdirectories and `__init__.py` files in place.
- `.gitignore` updated for `ea/data/` and `ea/results/`.

### e) Validation Criteria
- `python -c "import torch; m = torch.load('training/outputs/model/mobilenetv2_inference.pth', map_location='cpu'); print(type(m))"` prints without error.
- `dataset/train`, `dataset/val`, `dataset/test` are non-empty directories.
- `python -c "import ea"` runs without ImportError.

### f) Purpose
Establishes the working environment and confirms all upstream dependencies exist before any GA code is written. Prevents wasted debugging time later.

### g) Dependencies
- **Depends on:** AML engineer's trained checkpoint at `training/outputs/model/mobilenetv2_inference.pth`. This already exists in the repo.
- **Future phases depend on:** correct directory structure (all phases import from `ea/`).

---

## Phase 2 — Feature Extraction Pipeline

### a) Learning Requirements
- MobileNetV2 architecture: `model.features` (conv backbone) → global average pool → `model.classifier` (linear head). The **penultimate layer** is the output after the global average pool, shape `(batch, 1280)`.
- PyTorch inference mode: `model.eval()`, `torch.no_grad()`, `.cpu().numpy()`
- `torchvision.datasets.ImageFolder` for loading the organized dataset
- Same preprocessing transforms used during training: `Resize(224,224)` + ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- NumPy `.npy` and `.npz` file I/O (`np.save`, `np.load`)

### b) Implementation Tasks

**File: `ea/extract_features.py`**

1. Define `load_mobilenetv2(checkpoint_path)`:
   - Load `torchvision.models.mobilenet_v2()` and replace classifier head with `nn.Linear(1280, 43)` — same as `api/app.py`.
   - Load state dict from checkpoint.
   - Set `model.eval()`.
   - Return model.

2. Define `build_feature_extractor(model)`:
   - Return a function that runs `model.features(x).mean([2, 3])` — this extracts the global-average-pooled backbone output (1280 dims) without the classifier head.
   - Do NOT hard-code 1280; read it as `model.features[-1][0].out_channels` or infer from a single forward pass.

3. Define `extract_split(split_dir, feature_fn, batch_size=64)` → `(X, y)` as float32 numpy arrays:
   - Load split using `ImageFolder(split_dir, transform=val_transforms)` where `val_transforms` = Resize + ToTensor + Normalize (no augmentation — consistent features required).
   - Run in batches, accumulate features to avoid OOM.
   - Return `X.shape = (N, n)`, `y.shape = (N,)`.

4. Main block:
   - Call `extract_split` for `dataset/train` → save `ea/data/X_train.npy`, `ea/data/y_train.npy`.
   - Call `extract_split` for `dataset/val` → save `ea/data/X_val.npy`, `ea/data/y_val.npy`.
   - Print: feature dimension `n`, sample counts, extraction time.

5. Compute the **full-feature baseline** (required for FR-EA-16):
   - Train a `LinearSVC` on full `X_train`, evaluate on full `X_val`.
   - Save baseline accuracy to `ea/data/baseline_accuracy.txt`.
   - This is `full_feature_val_accuracy` referenced in every JSON log.

### c) Key Decisions

**Penultimate layer extraction strategy:**
- Option A: `model.features(x).mean([2,3])` — manually extract backbone output before classifier. Clean, explicit.
- Option B: Register a forward hook on `model.classifier[0]` (the dropout before the linear layer). More fragile, architecture-dependent.
- **Decision: Option A.** It directly matches `eatutorial.md §5 Step A` and is readable without deep PyTorch hook knowledge.

**Train vs val transforms:**
- Training images used augmentation (rotation, jitter). Feature extraction must use **validation transforms only** (no augmentation) to produce stable, deterministic features.
- Using augmented transforms would produce different feature vectors each time the same image is processed, making the GA's fitness function non-deterministic.

**Feature array storage format:**
- `.npy` files (separate train/val) rather than a single `.npz`, because GA only loads val features during fitness evaluation. Keeping them separate avoids loading 170+ MB when only 30 MB is needed.

**Memory estimate:**
- `X_train`: 33,318 × 1280 × 4 bytes ≈ 170 MB
- `X_val`: 5,891 × 1280 × 4 bytes ≈ 30 MB
- Total: ~200 MB — manageable on a standard laptop.

### d) Deliverables
- `ea/extract_features.py` (runnable: `python -m ea.extract_features`)
- `ea/data/X_train.npy`, `ea/data/y_train.npy`
- `ea/data/X_val.npy`, `ea/data/y_val.npy`
- `ea/data/baseline_accuracy.txt` (one float, e.g., `0.9312`)
- Console output confirming shapes and extraction time.

### e) Validation Criteria
- `X_train.shape == (33318, 1280)`, `X_val.shape == (5891, 1280)` (or similar counts matching dataset splits)
- `n = X_train.shape[1]` is read dynamically, not hard-coded
- Full-feature LinearSVC baseline achieves **>90% validation accuracy** (confirms features are meaningful — MobileNetV2 was trained to 98.25% val accuracy, so a linear classifier on its features should easily exceed 90%)
- Feature extraction completes in <15 minutes (GPU) or <30 minutes (CPU)

### f) Purpose
Decouples the slow deep learning model from the inner GA loop. Without precomputed features, every fitness evaluation would require a DNN forward pass, making 100 generations × 50 evaluations = 5,000 DNN inferences, far exceeding the NFR-PERF-04 budget of <60 seconds/generation. With precomputed features, each fitness evaluation is a cheap linear classifier.

### g) Dependencies
- **Depends on:** Phase 1 (environment), MobileNetV2 checkpoint, organized dataset.
- **All later phases depend on:** `ea/data/X_train.npy`, `ea/data/X_val.npy`, `y_train.npy`, `y_val.npy`, and `n` (chromosome length). Nothing else can start without these files.

---

## Phase 3 — GA Core Operators

### a) Learning Requirements
- Binary representation: NumPy `bool` or `uint8` arrays; boolean indexing for feature masking
- Tournament selection: select `k` random individuals, return the fittest
- Roulette wheel selection: fitness-proportionate probability, numerical stabilization
- Single-point crossover: split two parents at a random point, swap tails
- Uniform crossover: independently swap each bit with probability 0.5
- Bit-flip mutation: flip each bit independently with probability `p`
- Scikit-learn `LinearSVC` or `SGDClassifier` usage for multiclass classification
- NFR-PERF-04: <60 seconds per generation constraint

### b) Implementation Tasks

#### `ea/population.py`
1. `init_population(pop_size, n, seed=42)` → `ndarray shape (pop_size, n)` dtype `bool`
   - Each bit is Bernoulli(0.5) — expected 50% features selected.
   - Enforce no all-zero chromosomes: if any occur, flip one random bit to 1.

#### `ea/fitness.py`
1. `evaluate_fitness(chromosome, X_train, y_train, X_val, y_val, subsample_size=None)` → float
   - `mask = chromosome.astype(bool)`
   - If `mask.sum() == 0`: return 0.0 (invalid chromosome)
   - `Xtr = X_train[:, mask]` (optionally subsample rows — see Key Decisions)
   - `Xva = X_val[:, mask]`
   - Fit `LinearSVC(max_iter=2000, C=0.1)` on `Xtr`
   - Return `accuracy_score(y_val, clf.predict(Xva))`
2. `benchmark_fitness_eval(X_train, y_train, X_val, y_val)` → prints time for a single evaluation with a random chromosome. Run this to decide whether to subsample.

#### `ea/operators/selection.py`
1. `tournament_selection(population, fitnesses, k=3)` → one parent chromosome
   - Sample `k` unique indices from population, return chromosome with highest fitness.
2. `select_parents_tournament(population, fitnesses, n_parents, k=3)` → array of `n_parents` chromosomes
3. `roulette_wheel_selection(population, fitnesses, n_parents)` → array of `n_parents` chromosomes
   - Stabilize: `probs = (fitnesses - fitnesses.min() + 1e-8)`; normalize to sum to 1.
   - Use `np.random.choice(len(population), size=n_parents, p=probs, replace=True)`.

#### `ea/operators/crossover.py`
1. `single_point_crossover(parent1, parent2)` → (child1, child2)
   - Draw crossover point uniformly from `[1, n-1]`; swap tails.
2. `uniform_crossover(parent1, parent2, p=0.5)` → (child1, child2)
   - For each bit position, swap with probability `p`.
3. `apply_crossover(parents, crossover_fn, crossover_rate=0.8)` → offspring array
   - Pair parents, apply crossover with probability `crossover_rate`, else copy parents unchanged.

#### `ea/operators/mutation.py`
1. `bit_flip_mutation(chromosome, rate=0.01)` → mutated chromosome
   - Flip each bit independently with probability `rate`.
   - Enforce no all-zero result: if `sum == 0`, flip one random bit back to 1.
2. `apply_mutation(population, mutation_fn, **kwargs)` → mutated population

### c) Key Decisions

**Fitness classifier: LinearSVC vs alternatives**

| Classifier | Training speed (33k samples) | Accuracy | Notes |
|---|---|---|---|
| `LinearSVC(C=0.1)` | ~1–3s | High | Best option if within time budget |
| `SGDClassifier(loss='hinge')` | <0.5s | Slightly lower | Faster, less stable |
| `KNeighborsClassifier(k=3)` | 0s training, slow predict | Moderate | Impractical for large val set |
| `LogisticRegression` | ~2–5s | High | Slower than LinearSVC |

**Decision: Use `LinearSVC(C=0.1, max_iter=2000)`.**  
First, benchmark a single evaluation (Phase 3, `benchmark_fitness_eval`). If one call exceeds 1.2s, enable subsampling: randomly sample 8,000–10,000 training rows, use the same subsample for all evaluations in a run (subsample once at run start, not per-chromosome). Document this in JSON logs under an optional `subsample_size` field.

**Roulette wheel when all fitnesses are equal:**  
Early in evolution, many chromosomes may have similar fitness. Roulette degenerates to uniform random selection — this is acceptable behavior. The stabilization formula (`f - f_min + eps`) handles the case where `f_min = 0` (invalid chromosome).

**Crossover rate:**  
Apply crossover with probability 0.8 (i.e., 20% of parent pairs are copied unchanged). This is standard in GA literature and not mandated by the SRS — document it.

**Mutation rate:**  
FR-EA-09 specifies exactly `p = 0.01` per bit. Do not deviate for the 4 mandatory runs.

### d) Deliverables
- `ea/population.py`
- `ea/fitness.py`
- `ea/operators/__init__.py`, `selection.py`, `crossover.py`, `mutation.py`
- Benchmark printout confirming single fitness eval time

### e) Validation Criteria
- `tournament_selection` with k=3 from a population of 50 returns exactly 1 chromosome
- `roulette_wheel_selection` probabilities (before sampling) sum to 1.0 within floating-point tolerance
- After `single_point_crossover`, child1 and child2 together contain all bits of both parents (no bits created or destroyed)
- `bit_flip_mutation` with rate=0.01 flips ~12.8 bits on average over 100 runs on a 1280-bit chromosome (verify empirically)
- Single fitness evaluation: LinearSVC time ≤ 1.2s (so 50 evaluations ≤ 60s/generation — NFR-PERF-04)

### f) Purpose
These are the atomic building blocks. Each is small, independently testable, and swappable — this modular design is what makes the 4-configuration experiment matrix (FR-EA-12) implementable as a simple config substitution rather than code duplication.

### g) Dependencies
- **Depends on:** Phase 2 (feature arrays must exist to benchmark fitness)
- **Phases 4 and 5 depend on:** all operators imported from this phase

---

## Phase 4 — Fitness Sharing & Diversity Preservation

### a) Learning Requirements
- Fitness sharing theory: niche radius σ_share, sharing function `sh(d)`, shared fitness formula
- Hamming distance between binary vectors: `np.sum(a != b)` or vectorized over all pairs
- Survivor selection strategies: generational replacement vs elitism
- Effect of elitism on convergence: best fitness is monotonically non-decreasing with elitism
- Termination by stagnation: tracking generations without improvement

### b) Implementation Tasks

#### `ea/diversity.py` — Fitness Sharing (FR-EA-11)

1. `hamming_distances(population)` → `ndarray shape (pop_size, pop_size)`
   - Vectorized: `dists[i,j] = np.sum(population[i] != population[j])`
   - Efficient form: `np.sum(population[:, None] != population[None, :], axis=2)`

2. `sharing_function(d, sigma_share, alpha=1)` → float
   - Returns `max(0, 1 - (d / sigma_share) ** alpha)`

3. `apply_fitness_sharing(raw_fitnesses, population, sigma_share, alpha=1)` → shared_fitnesses
   - Compute all pairwise Hamming distances
   - For each individual `i`: `niche_count_i = sum_j(sh(d(i,j)))` (includes self: `sh(0) = 1`)
   - `shared_f_i = raw_f_i / niche_count_i`
   - Return shared fitness array (used for selection, not for termination check or JSON logging)

**Note:** Fitness sharing is applied **before selection** only. The raw best fitness is what gets logged in `fitness_per_generation` and used for the termination stagnation check (FR-EA-13, FR-EA-10).

#### `ea/operators/survivor.py` — Survivor Selection

1. `generational_replacement(population, offspring)` → next_population
   - Simply returns `offspring`. The entire parent generation is replaced.

2. `elitism(population, offspring, raw_fitnesses, offspring_fitnesses, elite_size=5)` → next_population
   - Identify top `elite_size` individuals from `population` by raw fitness.
   - Replace the worst `elite_size` individuals in `offspring` with the elite.
   - Return merged next population.

**Default for the 4 mandatory runs:** `generational_replacement`. Elitism is used only in Phase 6 (recommended extensions).

#### Termination Logic (in `ea/ga.py`, implemented in Phase 5)

- Maintain `best_history` list of best raw fitness per generation.
- After each generation: if `len(best_history) > 20` and `best_history[-20] == best_history[-1]`, set `stagnation_flag = True`.
- Terminate when `gen == 100` OR `stagnation_flag`.
- Log `generations_run` = actual generation count when terminated.

### c) Key Decisions

**σ_share value:**
- Too small (e.g., σ = 10): nearly no sharing; no diversity benefit.
- Too large (e.g., σ = 1000): all 1280-bit chromosomes are within each other's niche; selection pressure collapses.
- Reasonable starting point: `σ_share = 0.2 × n = 256` (20% of max Hamming distance).
- **Decision: σ_share = 0.2 × n, α = 1** for all mandatory runs. Log in JSON so the dashboard can display it.
- If convergence is premature, increase σ_share. If the GA converges too slowly, decrease it.

**Shared fitness for which selection method:**
- For tournament selection: compare shared fitnesses within each tournament (not raw).
- For roulette wheel: use shared fitnesses as the probability weights.
- Both methods receive shared fitness — this is the whole point of fitness sharing.

**Elitism size:**
- `elite_size = 5` (10% of population = 50) — a common heuristic.
- Larger values → faster convergence but reduced exploration.

**Termination stagnation check — raw or shared fitness?**
- Use **raw best fitness** for stagnation. Shared fitness fluctuates as population changes (even if the best chromosome is stable, its shared fitness changes when neighbors move). Using raw fitness gives a stable convergence signal.

### d) Deliverables
- `ea/diversity.py` (fitness sharing)
- `ea/operators/survivor.py` (generational + elitism)
- Termination logic stub in `ea/ga.py` (to be wired in Phase 5)

### e) Validation Criteria
- With a synthetic all-identical population (50 copies of the same chromosome), fitness sharing reduces each individual's shared fitness by a factor of 50 (niche count = 50 for all).
- With a perfectly diverse population (all chromosomes differ by >σ_share), shared fitness equals raw fitness (niche count = 1 for all).
- Elitism: assert `max(next_population_fitnesses) >= max(parent_fitnesses)` after every generation.
- Termination: unit test where a mock fitness function returns the same value for 20+ generations → GA terminates before generation 100.

### f) Purpose
- Fitness sharing (FR-EA-11) prevents the GA from converging to a single dominant solution and discarding diversity. Without it, all 50 chromosomes may cluster around one feature subset after a few generations — missing better solutions in other regions of the search space.
- Elitism provides a contrasting survivor selection strategy for the recommended FR-EA-10b comparison.

### g) Dependencies
- **Depends on:** Phase 3 (operators, fitness function)
- **Phase 5 depends on:** `apply_fitness_sharing` and `survivor` functions imported by the GA loop

---

## Phase 5 — GA Main Loop, Experiment Runner & JSON Logging

### a) Learning Requirements
- Python dataclasses (`@dataclass`) for structured configuration
- Functional composition: passing operator functions as arguments (strategy pattern)
- JSON serialization: `json.dump`, handling numpy types (`int64`, `float32` → Python native)
- `time.time()` for measuring wall-clock performance
- Python `random.seed`, `np.random.seed` for reproducibility
- SRS §5.3 JSON log schema — every field name must match exactly

### b) Implementation Tasks

#### `ea/ga.py` — Main GA Loop

```python
def run_ga(X_train, y_train, X_val, y_val, config) -> dict:
    """
    Returns a result dict:
      {best_chromosome, fitness_per_generation, generations_run, best_fitness,
       selected_feature_indices, total_features, selected_features, reduction_ratio}
    """
    np.random.seed(config.seed)
    n = X_train.shape[1]
    population = init_population(config.pop_size, n, config.seed)
    best_history = []

    for gen in range(config.max_generations):
        # Evaluate
        raw_fitnesses = [evaluate_fitness(chrom, X_train, y_train, X_val, y_val,
                                          config.subsample_size) for chrom in population]
        # Diversity
        shared_fitnesses = apply_fitness_sharing(
            np.array(raw_fitnesses), population, config.sigma_share, config.alpha)
        # Log best raw fitness
        best_idx = np.argmax(raw_fitnesses)
        best_history.append(raw_fitnesses[best_idx])

        # Termination check
        if gen >= 20 and best_history[-20] == best_history[-1]:
            break
        if gen == config.max_generations - 1:
            break

        # Selection (uses shared fitness)
        parents = config.selection_fn(population, shared_fitnesses, config.pop_size)
        # Crossover + Mutation
        offspring = apply_crossover(parents, config.crossover_fn, config.crossover_rate)
        offspring = apply_mutation(offspring, config.mutation_fn, config.mutation_rate)
        # Survivor selection
        offspring_fitnesses = [evaluate_fitness(c, X_train, y_train, X_val, y_val,
                                                config.subsample_size) for c in offspring]
        population = config.survivor_fn(population, offspring,
                                        np.array(raw_fitnesses), np.array(offspring_fitnesses))

    best_chrom = population[np.argmax(raw_fitnesses)]
    selected_indices = np.where(best_chrom)[0].tolist()
    return {
        "best_chromosome": best_chrom,
        "fitness_per_generation": best_history,
        "generations_run": len(best_history),
        "best_fitness": max(best_history),
        "selected_feature_indices": selected_indices,
        "total_features": n,
        "selected_features": len(selected_indices),
        "reduction_ratio": 1.0 - len(selected_indices) / n,
    }
```

**Performance note:** The loop above evaluates each chromosome twice (once for parents, once for offspring after mutation). Optimization: cache fitness evaluations where the chromosome hasn't changed. For the mandatory runs, the double evaluation is acceptable; cache only if runtime is too slow.

#### `ea/experiment.py` — Configuration System

```python
@dataclass
class ExperimentConfig:
    run_id: str
    selection: str            # 'tournament' | 'roulette'
    crossover: str            # 'single_point' | 'uniform'
    mutation: str             # 'bit_flip' (default)
    survivor: str             # 'generational' (default)
    selection_fn: callable
    crossover_fn: callable
    mutation_fn: callable
    survivor_fn: callable
    pop_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    sigma_share: float = None  # set to 0.2 * n after feature extraction
    alpha: float = 1.0
    seed: int = 42
    subsample_size: int = None  # set if LinearSVC is too slow (Assumption A3)
```

The 4 mandatory configurations:
```python
MANDATORY_CONFIGS = [
    ExperimentConfig(
        run_id="tournament_singlepoint_generational_bitflip_42",
        selection="tournament", crossover="single_point",
        mutation="bit_flip",    survivor="generational",
        selection_fn=select_parents_tournament,
        crossover_fn=single_point_crossover,
        mutation_fn=bit_flip_mutation,
        survivor_fn=generational_replacement,
    ),
    ExperimentConfig(
        run_id="tournament_uniform_generational_bitflip_42",
        selection="tournament", crossover="uniform", ...
    ),
    ExperimentConfig(
        run_id="roulette_singlepoint_generational_bitflip_42",
        selection="roulette",   crossover="single_point", ...
    ),
    ExperimentConfig(
        run_id="roulette_uniform_generational_bitflip_42",
        selection="roulette",   crossover="uniform", ...
    ),
]
```

#### `ea/logger.py` — JSON Log Builder

1. `build_log(config, result, full_feature_val_accuracy)` → dict
   - All required SRS §5.3 fields: `run_id`, `selection`, `crossover`, `population_size`, `generations_run`, `fitness_per_generation`, `best_fitness`, `best_chromosome`, `selected_feature_indices`, `total_features`, `selected_features`, `reduction_ratio`
   - Additional recommended fields: `mutation`, `mutation_rate`, `survivor_selection`, `fitness_sharing` (dict: `sigma_share`, `alpha`), `full_feature_val_accuracy`, `accuracy_drop_pp`, `seed`, `run_time_seconds`
2. `save_log(log_dict, output_dir='ea/results/')` → writes `<run_id>.json`
   - Convert all numpy types to Python native before `json.dump`.
   - Use `indent=2` for readable output.

#### `ea/run_experiments.py` — Entry Point

```
python -m ea.run_experiments
```
1. Load feature arrays from `ea/data/`
2. Load full-feature baseline from `ea/data/baseline_accuracy.txt`
3. Set `sigma_share = 0.2 * n`
4. For each config in `MANDATORY_CONFIGS`:
   - Print `=== Running <run_id> ===`
   - Print per-generation progress: `Gen 42 | Best: 0.8912 | Time: 34.2s`
   - Run `run_ga(...)` → result
   - Build and save JSON log
   - Upload to S3 immediately (Phase 7 hook — no-op if S3 not configured yet)
5. Print summary table: run_id, best_fitness, reduction_ratio, generations_run

### c) Key Decisions

**Sequential vs parallel experiment runs:**
- Option A: Run 4 configs sequentially (simpler, predictable memory use)
- Option B: Python `multiprocessing.Pool` (runs 4 configs concurrently)
- **Decision: Sequential.** Each run may take 30–120 minutes depending on hardware. Running in parallel would require 4× the RAM for feature arrays. The engineering complexity is not justified for a single-engineer project. If time is critical, parallelize configs 1–2 and 3–4 as two pairs.

**Termination logging:**
- `generations_run` in JSON = the actual number of generations completed, including the final generation where early stopping triggered. This matches the SRS §5.3 schema.

**Fitness per generation — raw or shared?**
- Log **raw best fitness** per generation (the best chromosome's actual validation accuracy without niche sharing adjustment). This is what the dashboard's convergence curve shows and what the supervisor evaluates.

**Numpy type serialization:**
- `json.dump` cannot serialize `np.int64`, `np.float32`, etc. Add a custom encoder: `class NumpyEncoder(json.JSONEncoder)` that converts numpy scalars to Python int/float.

### d) Deliverables
- `ea/ga.py` (complete GA loop)
- `ea/experiment.py` (config dataclasses + 4 mandatory configs)
- `ea/logger.py` (JSON builder and saver)
- `ea/run_experiments.py` (entry point)
- `ea/results/tournament_singlepoint_generational_bitflip_42.json`
- `ea/results/tournament_uniform_generational_bitflip_42.json`
- `ea/results/roulette_singlepoint_generational_bitflip_42.json`
- `ea/results/roulette_uniform_generational_bitflip_42.json`

### e) Validation Criteria
- 4 JSON files exist in `ea/results/`, each parseable by `json.load`
- Each file contains all required SRS §5.3 fields
- `len(log['fitness_per_generation']) == log['generations_run']` for every log
- At least one log has `reduction_ratio >= 0.30` (FR-EA-15)
- At least one log has `best_fitness >= 0.88` (FR-EA-15)
- `full_feature_val_accuracy - best_fitness < 0.03` for the best-performing log (FR-EA-16)
- Convergence is visible: `fitness_per_generation[-1] > fitness_per_generation[0]` in all runs (or at worst flat, never negative trending)

### f) Purpose
This is the integration phase — all components from Phases 2–4 are composed into a working, runnable GA experiment pipeline. It produces the primary deliverables (JSON logs) that the Dashboard engineer (Member 7) needs.

### g) Dependencies
- **Depends on:** Phases 2 (features), 3 (operators), 4 (diversity, survivor selection)
- **Phase 6 depends on:** working GA loop (adds new operator implementations)
- **Phase 7 depends on:** completed JSON logs in `ea/results/`
- **Member 7 (Dashboard) depends on:** JSON logs in S3 (after Phase 7)

---

## Phase 6 — Recommended Extensions (Bonus Operators & Experiments) ✅ DONE

*Complete only after all mandatory runs in Phase 5 are verified. These satisfy FR-EA-09b and FR-EA-10b for bonus marks.*

**Status (2026-04-26):** Phase 6 implemented. `swap_mutation` added to `ea/operators/mutation.py`. Three extension configs added via `get_extension_configs()` in `ea/experiment.py`: `tournament_singlepoint_generational_swapmutation_42`, `roulette_uniform_generational_swapmutation_42`, `tournament_singlepoint_elitism5_bitflip_42`. `run_experiments.py` now runs all 6 configs (4 mandatory + 2 extension). Elitism was already implemented in `ea/operators/survivor.py` from Phase 4.

### a) Learning Requirements
- Swap mutation behavior on binary strings (differs from permutation representations)
- Steady-state GA: replace one or few individuals per generation rather than the whole population
- How to fairly compare algorithm variants: control all variables, change only one at a time
- Statistical characterization: convergence speed (generation at which best fitness stabilizes) vs final quality

### b) Implementation Tasks

#### Second Mutation Operator — Swap Mutation (FR-EA-09b)

1. Add `swap_mutation(chromosome, rate=0.01)` to `ea/operators/mutation.py`:
   - For each bit, with probability `rate`, select another random position and swap values.
   - For binary chromosomes, a swap is meaningful only when positions differ (one is 0, one is 1); otherwise it is a no-op.
   - This is subtly less disruptive than bit-flip because it preserves the number of selected features, changing only which features are selected.

2. Add ≥2 new configs to `ea/experiment.py`:
   - `tournament_singlepoint_generational_swapmutation_42`
   - `roulette_uniform_generational_swapmutation_42`

3. Run these configs and save JSON logs.

4. In `ea/results/analysis.md`, compare:
   - Convergence speed: how many generations until best fitness stabilizes?
   - Final accuracy: best fitness of bit-flip vs swap mutation
   - Feature reduction ratio: does swap mutation tend toward different reduction levels?

#### Survivor Selection Comparison (FR-EA-10b)

1. Run the same base configuration (Tournament + Single-point) with elitism:
   - Config: `tournament_singlepoint_elitism5_bitflip_42`
   - `elite_size = 5`

2. Compare against `tournament_singlepoint_generational_bitflip_42` (already done):
   - Plot both convergence curves on the same chart (best fitness per generation)
   - Expected: elitism curve is strictly non-decreasing; generational may dip

3. Report in `ea/results/analysis.md`:
   - Does elitism converge faster? Does it get stuck in local optima?
   - Does generational replacement maintain more diversity?

### c) Key Decisions

**Swap mutation on binary strings — academic honesty:**
- Swap mutation is traditionally designed for permutation representations (e.g., TSP route) where it meaningfully preserves ordering.
- On a binary chromosome, swap is equivalent to flipping two bits (one 0→1 and one 1→0) while keeping the count of selected features constant. This **constrains** the search to chromosomes with the same number of features.
- **Decision:** Implement swap mutation as specified (FR-EA-09b requires it). Document the behavioral difference from bit-flip explicitly in the report. If the constraint-preserving behavior is academically interesting, note it as a research observation.

**Elitism size:**
- `elite_size = 5` (10% of population). This is the same value as Phase 4's implementation.
- Larger values (e.g., 10–15) might be needed if generational diversity causes good solutions to be lost. Tune only if validation in Phase 8 fails.

### d) Deliverables
- Updated `ea/operators/mutation.py` with `swap_mutation`
- ≥2 additional JSON logs for mutation comparison
- ≥1 additional JSON log for elitism survivor selection
- `ea/results/analysis.md` with comparison tables and observations

### e) Validation Criteria
- All new JSON logs contain all required SRS §5.3 fields
- Analysis compares at least 2 mutation operators quantitatively (accuracy + convergence speed)
- Analysis compares at least 2 survivor selection strategies quantitatively
- Elitism run's `fitness_per_generation` is monotonically non-decreasing (assert this)

### f) Purpose
These extensions satisfy the "recommended" requirements (FR-EA-09b, FR-EA-10b) that earn bonus marks in the EA course assessment. They also strengthen the academic narrative of the project: demonstrating that you understood how operator choice affects convergence is a core EA learning objective.

### g) Dependencies
- **Depends on:** Phase 5 (GA loop must be verified and working before adding more configs)
- **Phase 7 depends on:** all JSON logs (mandatory + recommended) are complete

---

## Phase 7 — S3 Upload & Cloud Integration

### a) Learning Requirements
- AWS boto3 SDK: `boto3.client('s3')`, `s3.upload_file`, `s3.put_object`
- AWS credential chain: environment variables → `~/.aws/credentials` → IAM instance role
- Environment variable management: `os.environ.get('S3_BUCKET')`
- SRS NFR-SEC-04: no credentials in code or Dockerfiles
- SRS NFR-REL-02: logs persisted to S3 before process exits

### b) Implementation Tasks

1. Write `ea/upload_s3.py`:

```python
import boto3, os, json, pathlib

def upload_log(local_path: str, run_id: str) -> str:
    bucket = os.environ['S3_BUCKET']       # never hard-code
    key = f"ea-experiments/{run_id}.json"
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"

def upload_all_logs(results_dir='ea/results/') -> list[str]:
    uploaded = []
    for path in pathlib.Path(results_dir).glob('*.json'):
        run_id = path.stem
        url = upload_log(str(path), run_id)
        print(f"Uploaded: {url}")
        uploaded.append(url)
    return uploaded
```

2. Integrate into `ea/run_experiments.py`: call `upload_log(log_path, run_id)` immediately after each JSON file is saved (not at the end of all runs). This satisfies NFR-REL-02 — if the process crashes mid-run, completed runs are already persisted.

3. Document required environment variable: `S3_BUCKET=<your-bucket-name>` in a `.env.example` file at the repo root. Never commit a `.env` file with real values.

4. Verify uploads: after all runs, print a listing of the S3 prefix:
   ```python
   s3.list_objects_v2(Bucket=bucket, Prefix='ea-experiments/')
   ```

5. Coordinate with Member 7 (Cloud/Dashboard engineer): they create the S3 bucket; you upload to it. Agree on the exact bucket name before Phase 7 starts.

### c) Key Decisions

**Upload timing — per-run vs batch:**
- Option A: Upload all logs after all 4 runs complete.
- Option B: Upload each log immediately after the run completes.
- **Decision: Option B (per-run upload).** If the process is killed after run 2 completes, runs 1 and 2 are already in S3. Option A would lose all 4 if the process crashes on run 3. This directly satisfies NFR-REL-02.

**Which files to upload:**
- Upload: JSON experiment logs only (`ea/results/*.json`).
- Do NOT upload: feature arrays (`ea/data/*.npy`) — they are large (200 MB) and reproducible from the model checkpoint. The SRS does not require them in S3.

**AWS credentials:**
- Use environment variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (or an IAM instance role on EC2 — no action needed if running on EC2 with the correct role).
- Never place credentials in `ea/upload_s3.py` or any committed file.

### d) Deliverables
- `ea/upload_s3.py`
- `.env.example` at repo root documenting `S3_BUCKET` (and optionally `AWS_REGION`)
- All 4+ JSON logs visible at `s3://<bucket>/ea-experiments/` after execution

### e) Validation Criteria
- `aws s3 ls s3://$S3_BUCKET/ea-experiments/` lists ≥4 `.json` files
- Downloaded copies are identical to local files (md5 comparison)
- No credentials or bucket names in source code (grep check: `git grep -r "s3://" ea/` should find nothing in `.py` files)

### f) Purpose
Satisfies FR-EA-14 (upload logs to S3 prefix) and NFR-REL-02 (persistence before exit). Enables Member 7 to build the dashboard without depending on the EA engineer's local machine.

### g) Dependencies
- **Depends on:** Phase 5 (complete JSON logs), S3 bucket created by Member 7
- **Member 7 (Dashboard) depends on:** logs at the S3 prefix `s3://<bucket>/ea-experiments/`

---

## Phase 8 — Validation, Acceptance Testing & Handoff

### a) Learning Requirements
- SRS §9.2 EA acceptance criteria (the exact thresholds to check)
- Python assertions and exit codes for CI-friendly validation scripts
- Matplotlib for basic line plots (convergence curves)
- CSV writing with `csv.DictWriter`

### b) Implementation Tasks

#### `ea/validate.py` — Acceptance Criteria Checker

```python
"""
Run: python -m ea.validate
Exits 0 if all criteria pass, non-zero otherwise.
"""
```

Checks to implement:
1. **4 mandatory configs exist**: for each of the 4 required `(selection, crossover)` combinations, at least one JSON log must exist.
2. **Feature reduction ≥ 30%**: `assert best_log['reduction_ratio'] >= 0.30` (FR-EA-15)
3. **Accuracy ≥ 88%**: `assert best_log['best_fitness'] >= 0.88` (FR-EA-15)
4. **Accuracy drop < 3pp**: `assert best_log['full_feature_val_accuracy'] - best_log['best_fitness'] < 0.03` (FR-EA-16)
5. **Schema compliance**: every required SRS §5.3 field is present in every log
6. **Convergence visible**: `best_history[-1] >= best_history[0]` in every run (fitness improves or stays flat, never degrades overall)
7. **All logs in S3**: list `s3://<bucket>/ea-experiments/` and assert each local JSON is present

#### Convergence Plot: `ea/results/convergence.png`

- One line per experiment run on a single chart.
- X-axis: generation number (0 to `generations_run`).
- Y-axis: best fitness (validation accuracy).
- Legend: `run_id` for each line.
- Save as PNG — this is the local preview; the final chart is on the dashboard.

```python
import matplotlib.pyplot as plt
for log in logs:
    plt.plot(log['fitness_per_generation'], label=log['run_id'])
plt.xlabel('Generation')
plt.ylabel('Best Fitness (Val Accuracy)')
plt.legend(fontsize=6)
plt.savefig('ea/results/convergence.png', dpi=150)
```

#### Feature Reduction Table: `ea/results/feature_reduction.csv`

Columns: `run_id`, `total_features`, `selected_features`, `reduction_ratio`, `best_fitness`, `accuracy_drop_pp`

This is the source data for the dashboard's FR-DASH-02 table.

#### Handoff Summary

Write a brief `ea/results/HANDOFF.md` (not committed — this is ephemeral documentation):
- S3 prefix: `s3://<bucket>/ea-experiments/`
- Full-feature baseline accuracy: value from `baseline_accuracy.txt`
- Best configuration name and its metrics
- Date of last upload
- Any known issues or deviations from SRS

### c) Key Decisions
- **Validation as an executable script**: `python -m ea.validate` returns exit code 0 on pass. This lets Member 6 (CI/CD) or Member 7 (Dashboard) run it independently to verify the data before building the dashboard.
- **Best log definition**: "best" = highest `best_fitness` across all mandatory runs. FR-EA-15 and FR-EA-16 require at least one run to hit the thresholds; they need not all hit them simultaneously.

### d) Deliverables
- `ea/validate.py` (exits 0 on success)
- `ea/results/convergence.png`
- `ea/results/feature_reduction.csv`
- `ea/results/HANDOFF.md` (not committed)

### e) Validation Criteria (Final Gate)
- `python -m ea.validate` exits with code 0
- Best run: `best_fitness >= 0.88`, `reduction_ratio >= 0.30`, `accuracy_drop < 0.03`
- All 4 mandatory JSON logs in S3, parseable and schema-compliant
- Convergence plot shows visible improvement over generations in at least 3 of 4 runs

### f) Purpose
Formally confirms that all FR-EA-01 → FR-EA-16 acceptance criteria are met before handing off to Member 7. Without this phase, there is no way to be confident the dashboard will display valid data or that the supervisor will see the required metrics.

### g) Dependencies
- **Depends on:** All previous phases
- **No future phases:** This is the final EA phase. After this, your work is complete. Member 7 reads from S3.

---

## Summary: File Inventory

| File | Phase | SRS Requirements |
|---|---|---|
| `ea/__init__.py` | 1 | — |
| `ea/extract_features.py` | 2 | FR-EA-02 (chromosome length), FR-EA-16 (baseline) |
| `ea/population.py` | 3 | FR-EA-04 |
| `ea/fitness.py` | 3 | FR-EA-03, NFR-PERF-04 |
| `ea/operators/selection.py` | 3 | FR-EA-05, FR-EA-06 |
| `ea/operators/crossover.py` | 3 | FR-EA-07, FR-EA-08 |
| `ea/operators/mutation.py` | 3, 6 | FR-EA-09, FR-EA-09b |
| `ea/diversity.py` | 4 | FR-EA-11 |
| `ea/operators/survivor.py` | 4, 6 | FR-EA-10b |
| `ea/ga.py` | 4, 5 | FR-EA-10 (termination), full GA loop |
| `ea/experiment.py` | 5 | FR-EA-12 (4 configs) |
| `ea/logger.py` | 5 | FR-EA-13 (JSON schema) |
| `ea/run_experiments.py` | 5 | FR-EA-12, FR-EA-13, FR-EA-14 |
| `ea/upload_s3.py` | 7 | FR-EA-14, NFR-REL-02 |
| `ea/validate.py` | 8 | FR-EA-15, FR-EA-16, §9.2 |

---

## Dependency Graph

```
Phase 1: Environment Setup
    │
    ▼
Phase 2: Feature Extraction  ← needs: trained model + dataset
    │   (produces: X_train.npy, X_val.npy, baseline_accuracy.txt)
    ▼
Phase 3: GA Core Operators
    │   (produces: selection, crossover, mutation, fitness)
    ▼
Phase 4: Fitness Sharing & Diversity
    │   (produces: diversity.py, survivor.py, termination logic)
    ▼
Phase 5: GA Loop + Runner + JSON Logging
    │   (produces: 4 JSON logs in ea/results/)
    ├──▶ Phase 6: Recommended Extensions (parallel, optional)
    │            (produces: additional JSON logs)
    ▼
Phase 7: S3 Upload
    │   (produces: logs at s3://<bucket>/ea-experiments/)
    ▼
Phase 8: Validation & Handoff
        (confirms: all acceptance criteria met)
        (produces: convergence.png, feature_reduction.csv)
            │
            ▼
    [Member 7: Dashboard reads from S3]
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Fitness eval too slow (>1.2s/chrom) | Medium | High — blocks all runs | Benchmark in Phase 3. Subsample 8–10k training rows if needed (Assumption A3). |
| GA doesn't hit ≥88% accuracy | Low | High — fails FR-EA-15 | MobileNetV2 val accuracy is 98.25%; even a 10% linear classifier on 640 features should easily clear 88%. If not, check feature extraction bugs. |
| Roulette wheel instability (near-zero fitnesses early) | Medium | Low | Stabilization formula in Phase 3 handles this. |
| All-zero chromosome crashes LinearSVC | Low | Medium | Guard added in `evaluate_fitness` and `bit_flip_mutation`. |
| S3 bucket not ready when Phase 7 starts | Medium | Low — only delays handoff | Test upload locally with `--dry-run` mode. Phase 7 can run independently once bucket is available. |
| Instructor disallows sklearn for fitness | Low | High — invalidates fitness function | Flag assumption A1 early. Alternative: implement a simple k-NN in NumPy without sklearn. |
| Runs take too long (4 configs × 100 gens) | Medium | Medium — timeline pressure | Start Phase 5 early; use stagnation termination (may stop before 100 gens). Run configs 1–2 and 3–4 in separate terminal windows if sequential is too slow. |
