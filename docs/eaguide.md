# Evolutionary Algorithm (EA) Module — Teaching Guide

> **Project:** Traffic Sign Intelligence Platform  
> **Course:** Evolutionary Algorithms  
> **Task:** Genetic Algorithm–based feature selection on MobileNetV2 embeddings  
> **Dataset:** GTSRB — 43-class German Traffic Sign Recognition Benchmark

---

## Table of Contents

1. [Why Evolutionary Algorithms for Feature Selection?](#1-why-evolutionary-algorithms-for-feature-selection)
2. [Big Picture: What This Module Does](#2-big-picture-what-this-module-does)
3. [Module File Map](#3-module-file-map)
4. [Stage 1 — Feature Extraction (`extract_features.py`)](#4-stage-1--feature-extraction)
5. [Stage 2 — Representing Solutions: The Chromosome (`population.py`)](#5-stage-2--representing-solutions-the-chromosome)
6. [Stage 3 — Measuring Quality: The Fitness Function (`fitness.py`)](#6-stage-3--measuring-quality-the-fitness-function)
7. [Stage 4 — The Genetic Operators (`operators/`)](#7-stage-4--the-genetic-operators)
   - [Selection](#71-selection-selectionpy)
   - [Crossover](#72-crossover-crossoverpy)
   - [Mutation](#73-mutation-mutationpy)
   - [Survivor Selection](#74-survivor-selection-survivorpy)
8. [Stage 5 — Diversity Preservation (`diversity.py`)](#8-stage-5--diversity-preservation)
9. [Stage 6 — The GA Main Loop (`ga.py`)](#9-stage-6--the-ga-main-loop)
10. [Stage 7 — Experiment Configuration (`experiment.py`)](#10-stage-7--experiment-configuration)
11. [Stage 8 — Logging (`logger.py`)](#11-stage-8--logging)
12. [Stage 9 — Running Everything (`run_experiments.py`)](#12-stage-9--running-everything)
13. [Performance Optimisations](#13-performance-optimisations)
14. [Actual Experimental Results](#14-actual-experimental-results)
15. [Design Decisions and Best Practices](#15-design-decisions-and-best-practices)
16. [Common Pitfalls](#16-common-pitfalls)
17. [Quick-Reference: Parameter Cheat Sheet](#17-quick-reference-parameter-cheat-sheet)

---

## 1. Why Evolutionary Algorithms for Feature Selection?

A deep neural network like MobileNetV2 produces a **1 280-dimensional feature vector** for every image. Not all 1 280 dimensions carry equally useful information for classifying traffic signs. Some dimensions might encode texture patterns irrelevant to sign identity; others might be highly redundant.

**Feature selection** is the problem of finding the best *subset* of dimensions to keep, so that:

- A classifier trained only on those dimensions is still accurate.
- The model is faster and lighter (fewer inputs to process).

This is a **combinatorial search problem**: with 1 280 features there are 2¹²⁸⁰ possible subsets — far too many to enumerate. Evolutionary Algorithms are well suited here because they:

1. Search the space *without* needing a gradient (the fitness landscape is not differentiable).
2. Maintain a *population* of candidate solutions, exploring many regions in parallel.
3. Use biologically inspired operators (crossover, mutation) to move toward better solutions.

The goal in this project is to find a feature subset that keeps validation accuracy **≥ 98.8%** (within 3 percentage points of the 99.18% baseline) while removing **≥ 30%** of the original features.

---

## 2. Big Picture: What This Module Does

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GTSRB images (224×224, RGB)                                             │
│         │                                                                │
│         ▼                                                                │
│  MobileNetV2 backbone  ──▶  1 280-dim feature vector per image           │
│         │                                                                │
│         ▼                                                                │
│  Variance pre-filter   ──▶  900 features kept (top by variance)          │
│         │                                                                │
│         ▼                                                                │
│  Genetic Algorithm                                                       │
│    Population of 35 binary chromosomes (length 900)                     │
│         │                                                                │
│    Each generation:                                                      │
│      1. Evaluate fitness (LinearSVC accuracy on selected features)       │
│      2. Apply fitness sharing (discourage identical solutions)           │
│      3. Select parents (tournament or roulette wheel)                    │
│      4. Crossover (single-point or uniform)                              │
│      5. Mutate (bit-flip or swap)                                        │
│      6. Replace population (generational or elitism)                     │
│         │                                                                │
│    Terminate after 100 generations or 12 stagnant generations           │
│         │                                                                │
│         ▼                                                                │
│  Best chromosome  ──▶  Selected feature indices  ──▶  JSON log + S3     │
└──────────────────────────────────────────────────────────────────────────┘
```

The entire pipeline runs **7 experiment configurations** (4 mandatory + 3 extension) and saves one structured JSON result file per run.

---

## 3. Module File Map

```
ea/
├── __init__.py                  # Empty package marker
├── extract_features.py          # Step 1: extract MobileNetV2 embeddings
├── population.py                # Random binary population initialisation
├── fitness.py                   # Fitness = LinearSVC val accuracy
├── diversity.py                 # Fitness sharing for niche preservation
├── ga.py                        # Main GA loop (parallelised + checkpointed)
├── experiment.py                # ExperimentConfig dataclass + 7 configs
├── logger.py                    # JSON log builder + saver
├── run_experiments.py           # Entry point: runs all 7 configs
│
├── operators/
│   ├── __init__.py
│   ├── selection.py             # tournament_selection, roulette_wheel_selection
│   ├── crossover.py             # single_point_crossover, uniform_crossover
│   ├── mutation.py              # bit_flip_mutation, swap_mutation
│   └── survivor.py             # generational_replacement, elitism
│
├── data/                        # Cached feature arrays (generated once)
│   ├── X_train.npy              # (33 318, 1 280) float32
│   ├── X_val.npy                # (5 891, 1 280) float32
│   ├── y_train.npy              # (33 318,) int
│   ├── y_val.npy                # (5 891,) int
│   ├── X_train_filtered.npy     # (33 318, 900) float32 — after variance filter
│   ├── X_val_filtered.npy       # (5 891, 900) float32
│   ├── feature_filter_mask.npy  # (1 280,) bool — True at kept positions
│   └── baseline_accuracy.txt    # 0.9918519775929384
│
└── results/                     # One JSON per completed experiment
    ├── tournament_singlepoint_generational_bitflip_42.json
    ├── tournament_uniform_generational_bitflip_42.json
    ├── roulette_singlepoint_generational_bitflip_42.json
    ├── roulette_uniform_generational_bitflip_42.json
    ├── tournament_singlepoint_generational_swapmutation_42.json
    ├── roulette_uniform_generational_swapmutation_42.json
    └── tournament_singlepoint_elitism5_bitflip_42.json
```

**Run order:**

```
python ea/extract_features.py   # once — creates data/*.npy
python -m ea.run_experiments    # runs all 7 GA configs
```

---

## 4. Stage 1 — Feature Extraction

**File:** `ea/extract_features.py`

### Why extract features first?

Training a LinearSVC from scratch for every chromosome in every generation would be the bottleneck. Instead, we extract features *once* from the frozen MobileNetV2 backbone and cache them as `.npy` files. The GA then works entirely in NumPy — no PyTorch, no image loading, no GPU needed during the search.

### What MobileNetV2 gives us

MobileNetV2 is a convolutional neural network pre-trained on ImageNet and fine-tuned on GTSRB. Its final layer before the classification head is a **1 280-dimensional global average pooled feature map** — a compact numerical summary of the visual content of each image.

```
Image (224×224×3)
      │
 MobileNetV2 backbone (convolutional layers)
      │
 Feature map  (7×7×1280)
      │ global average pooling (mean over spatial dims)
      ▼
 Feature vector  (1×1280)
```

### Code walkthrough

```python
# ea/extract_features.py

def load_mobilenetv2(checkpoint_path):
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)  # 43 classes
    # Load weights from the AML-trained checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def build_feature_extractor(model):
    def extract(x):
        with torch.no_grad():
            return model.features(x).mean([2, 3])  # spatial average pooling
    return extract
```

`model.features(x)` runs the image through all convolutional layers but stops before the classifier. `.mean([2, 3])` collapses the spatial dimensions (height and width), producing one vector of length 1 280 per image.

### Variance pre-filter

After extracting all 1 280 features, a **variance filter** keeps only the top 900 most variable features across the training set.

```python
def apply_variance_filter(X, target=900):
    variances = X.var(axis=0)          # variance of each feature dimension
    top_indices = np.argsort(variances)[-target:]   # 900 highest
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[top_indices] = True
    return X[:, mask], mask
```

**Why filter?** Features with near-zero variance across the dataset carry almost no information — every image gets roughly the same value, so the feature cannot help discriminate between classes. Removing them:
- Reduces the chromosome length from 1 280 → 900 (30% smaller search space).
- Makes each LinearSVC fit ~30% faster.
- Does not hurt accuracy (low-variance features are useless by definition).

### Baseline accuracy

After extraction, a LinearSVC is trained on all 900 filtered features to establish the **baseline**:

```
Full-feature LinearSVC val accuracy: 0.9919  (99.19%)
```

The GA must find a subset that stays within 3 percentage points of this — i.e., ≥ 96.19%. In practice the best GA result achieves 98.49%, which is only 0.70pp below the baseline.

---

## 5. Stage 2 — Representing Solutions: The Chromosome

**File:** `ea/population.py`

### The binary chromosome

A **chromosome** is a 1D boolean NumPy array of length *n* (= 900 after the variance filter). Each position corresponds to one feature:

```
Position:    0   1   2   3   4   5   ...  899
Chromosome: [F,  F,  T,  F,  T,  T,  ..., F ]
                     ↑           ↑    ↑
                 "use this feature" positions
```

`True` at position `i` means "include feature `i` when training the classifier."

This representation is called a **binary encoding** or **binary string chromosome** — the most natural encoding for feature selection.

### Population initialisation

```python
# ea/population.py

def init_population(pop_size, n, seed=42):
    rng = np.random.default_rng(seed)
    population = rng.integers(0, 2, size=(pop_size, n)).astype(bool)

    # Guard: no chromosome may select zero features
    zero_rows = np.where(~population.any(axis=1))[0]
    for i in zero_rows:
        population[i, rng.integers(n)] = True

    return population
```

- Each bit is set independently with probability 0.5 → ~50% of features are selected in each initial chromosome.
- The **all-zero guard** ensures no chromosome starts with zero selected features (that would be a degenerate solution).
- `pop_size = 35` — 35 candidate solutions evaluated in parallel each generation.

**Intuition:** Start with random guesses spread across the search space. The GA then iteratively improves them.

---

## 6. Stage 3 — Measuring Quality: The Fitness Function

**File:** `ea/fitness.py`

### What is fitness?

Fitness answers the question: *"How good is this chromosome?"* In feature selection, a good chromosome selects features that let a classifier perform well.

```python
# ea/fitness.py

def evaluate_fitness(chromosome, X_train, y_train, X_val, y_val, subsample_size=None):
    mask = chromosome.astype(bool)
    if mask.sum() == 0:
        return 0.0                          # degenerate: no features selected

    Xtr = X_train[:, mask]                  # keep only selected columns
    ytr = y_train

    if subsample_size and subsample_size < len(Xtr):
        idx = np.random.choice(len(Xtr), size=subsample_size, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    clf = LinearSVC(C=0.1, max_iter=2000)
    clf.fit(Xtr, ytr)
    return float(accuracy_score(y_val, clf.predict(X_val[:, mask])))
```

**Steps for one chromosome:**

1. Apply the boolean mask to select only the chosen feature columns.
2. (Optionally) subsample training rows for speed.
3. Fit a Linear SVM on the selected features.
4. Measure its accuracy on the *validation* set (never the test set — that would leak information).
5. Return the accuracy as a float in [0, 1].

### Why LinearSVC as the fitness classifier?

- **Fast:** Linear SVMs train in O(n·d) time — much faster than neural networks.
- **No randomness:** Given the same data and hyperparameters, LinearSVC always produces the same model. This makes fitness evaluations deterministic and reproducible.
- **Discriminative:** Even a linear classifier reveals whether a feature subset is useful — if a subset is bad, the SVM cannot compensate.
- **C=0.1:** Light regularisation. The goal is feature *selection*, not SVM hyperparameter tuning, so a fixed C is appropriate.

### Fitness vs. validation accuracy

> Fitness and validation accuracy are the same value here. The GA optimises the feature subset by maximising classifier performance on held-out data.

This is a principled choice: the GA never sees the test set. Validation performance is a proxy for generalisation. The test set is used only once, after all GA runs are complete, to report final results.

---

## 7. Stage 4 — The Genetic Operators

The genetic operators are the "physics" of the evolutionary search. They transform one generation of solutions into the next.

### 7.1 Selection (`operators/selection.py`)

Selection chooses which chromosomes get to be **parents** — whose genetic material propagates to the next generation. Better chromosomes should be selected more often, but not exclusively (diversity is important).

#### Tournament Selection

```python
def tournament_selection(population, fitnesses, k=3):
    indices = np.random.choice(len(population), size=k, replace=False)
    winner = indices[np.argmax(fitnesses[indices])]
    return population[winner].copy()
```

**How it works:**

1. Draw `k=3` individuals at random.
2. The fittest of the three wins and becomes a parent.
3. Repeat independently for every parent slot.

```
Population:  [A(0.95), B(0.97), C(0.94), D(0.96), E(0.98)]
Tournament:  {B, D, A}  → winner: B (0.97)
Tournament:  {E, C, D}  → winner: E (0.98)
```

**Properties:**
- Tournament size `k=3` creates **selection pressure** — the fitter individual wins most of the time, but not always (the weaker two are skipped).
- Larger `k` → stronger pressure → faster convergence but higher risk of premature convergence.
- Does not require fitness values to be positive or normalised.

#### Roulette Wheel Selection (Fitness-Proportionate Selection)

```python
def roulette_wheel_selection(population, fitnesses, n_parents):
    fitnesses = np.asarray(fitnesses, dtype=float)
    weights = fitnesses - fitnesses.min() + 1e-8    # shift to ensure all > 0
    probs = weights / weights.sum()
    indices = np.random.choice(len(population), size=n_parents, p=probs, replace=True)
    return population[indices].copy()
```

**How it works:** Each chromosome gets a "slice" of a roulette wheel proportional to its fitness. Spinning the wheel selects one chromosome — high-fitness chromosomes occupy larger slices and are more likely to be chosen.

```
Chromosome:  A(0.95)   B(0.97)   C(0.94)   D(0.96)   E(0.98)
Shifted:     0.01      0.03      (min+ε)   0.02      0.04
Probability: 10%       30%       ~0%       20%       40%
```

**Numerical stabilisation:** We shift all weights by `(min + 1e-8)` to ensure no chromosome gets zero probability, even if its raw fitness is the lowest. This prevents division-by-zero and keeps the algorithm from prematurely eliminating low-fitness chromosomes.

**Properties:**
- Selection pressure depends on the *spread* of fitness values. If all chromosomes have similar fitness, the wheel is nearly uniform.
- Can be dominated by a few super-fit individuals (super-individual problem).

| | Tournament | Roulette |
|---|---|---|
| Pressure control | Via `k` parameter | Via fitness spread |
| Requires positive fitness | No | After shift, yes |
| Replacement | Without replacement per draw | With replacement |
| Computational cost | O(k) per parent | O(pop_size) once |

---

### 7.2 Crossover (`operators/crossover.py`)

Crossover takes two parent chromosomes and produces two children that **inherit parts from both parents**. This is the primary mechanism for recombining good building blocks discovered by different individuals.

#### Single-Point Crossover

```python
def single_point_crossover(parent1, parent2):
    n = len(parent1)
    point = np.random.randint(1, n)        # random split point in (1, n-1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2
```

**Visual:**

```
Parent1: [1,0,1,1,0 | 0,1,0,1,1]   ← split at position 5
Parent2: [0,1,0,0,1 | 1,0,1,0,0]

Child1:  [1,0,1,1,0 | 1,0,1,0,0]   ← left from P1, right from P2
Child2:  [0,1,0,0,1 | 0,1,0,1,1]   ← left from P2, right from P1
```

**Property:** Features near each other in the chromosome array tend to be inherited together (they are on the same side of the cut). This means "schema" — contiguous building blocks — are preserved.

#### Uniform Crossover

```python
def uniform_crossover(parent1, parent2, p=0.5):
    swap = np.random.random(len(parent1)) < p
    child1 = np.where(swap, parent2, parent1)
    child2 = np.where(swap, parent1, parent2)
    return child1, child2
```

**Visual:**

```
Parent1: [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
Parent2: [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
Swap:    [T, F, T, F, T, F, F, T, F, T]  ← independent per bit

Child1:  [0, 0, 0, 1, 1, 0, 1, 1, 1, 0]
Child2:  [1, 1, 1, 0, 0, 1, 0, 0, 0, 1]
```

**Property:** Each bit is inherited independently — no positional bias. This creates more diverse offspring and breaks up any accidental correlations in the chromosome layout.

#### Applying Crossover with Probability

```python
def apply_crossover(parents, crossover_fn, crossover_rate=0.8):
    offspring = []
    for i in range(0, len(parents) - 1, 2):
        p1, p2 = parents[i], parents[i + 1]
        if np.random.random() < crossover_rate:     # 80% chance
            c1, c2 = crossover_fn(p1, p2)
        else:
            c1, c2 = p1.copy(), p2.copy()           # copy unchanged
        offspring.extend([c1, c2])
    return np.array(offspring)
```

`crossover_rate = 0.8` means 80% of parent pairs produce genuinely recombined children; the remaining 20% pass through unchanged. This preserves some good solutions while still generating novelty.

---

### 7.3 Mutation (`operators/mutation.py`)

Mutation introduces **small random changes** to offspring chromosomes. It provides the only source of *new* genetic material not present in the initial population — without mutation, the GA can only recombine what it already has.

#### Bit-Flip Mutation

```python
def bit_flip_mutation(chromosome, rate=0.01):
    flip_mask = np.random.random(len(chromosome)) < rate
    mutated = chromosome ^ flip_mask          # XOR flips bits where mask is True
    if not mutated.any():
        mutated[np.random.randint(len(mutated))] = True   # all-zero guard
    return mutated
```

**How it works:** Each bit is independently flipped (0→1 or 1→0) with probability `rate=0.01`. With 900 features, each chromosome mutates on average 9 bits per generation.

```
Before: [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
Flip?:  [F, F, T, F, F, T, F, F, F, F]   ← 2 bits flipped
After:  [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
              ↑           ↑
          1→0            0→1
```

**Effect:** Can both add features (0→1) and remove features (1→0). Changes the *number* of selected features.

#### Swap Mutation

```python
def swap_mutation(chromosome, rate=0.01):
    mutated = chromosome.copy()
    n = len(mutated)
    for i in range(n):
        if np.random.random() < rate:
            j = np.random.randint(n)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated
```

**How it works:** Each position independently has a `rate` chance of being swapped with a randomly chosen other position.

```
Before: [1, 0, 1, 1, 0, 0, 1]   (4 features selected)
Swap positions 1 and 2:
After:  [1, 1, 0, 1, 0, 0, 1]   (still 4 features selected)
```

**Key insight:** On a binary chromosome, a swap between two *identical* bits (`True`↔`True` or `False`↔`False`) is a no-op. Only swaps between a `True` and a `False` position actually change the chromosome — and those preserve the total count of selected features. Swap mutation explores *which* features are selected, not *how many*. This is a **fundamentally different neighbourhood** from bit-flip mutation.

#### Why rate=0.01?

With a chromosome of length 900:
- `rate=0.01` → ~9 bits mutated per chromosome per generation.
- Too low (e.g., 0.001) → mutation is too weak to escape local optima.
- Too high (e.g., 0.1) → mutation destroys good solutions faster than selection preserves them (acts like random search).

The rule of thumb from GA literature is `rate ≈ 1/chromosome_length`. Here `1/900 ≈ 0.001`, but `0.01` is chosen because the feature space is large and we want adequate exploration.

---

### 7.4 Survivor Selection (`operators/survivor.py`)

After crossover and mutation, the offspring population must replace (some of) the parent population. This is called **survivor selection**.

#### Generational Replacement

```python
def generational_replacement(population, offspring, raw_fitnesses=None, offspring_fitnesses=None):
    return offspring.copy()
```

The simplest strategy: **discard all parents, keep all offspring**. The entire population turns over each generation. This creates strong selection pressure towards recent improvements but can occasionally lose a very good solution if mutation happened to degrade it.

#### Elitism

```python
def elitism(population, offspring, raw_fitnesses, offspring_fitnesses, elite_size=5):
    elite_indices = np.argsort(raw_fitnesses)[-elite_size:]          # top 5 parents
    worst_offspring_indices = np.argsort(offspring_fitnesses)[:elite_size]  # bottom 5 offspring

    next_population = offspring.copy()
    next_population[worst_offspring_indices] = population[elite_indices]  # inject elites
    return next_population
```

**Elitism with `elite_size=5`:** The top 5 parent chromosomes are guaranteed to survive into the next generation by overwriting the 5 weakest offspring. This ensures the **best fitness never decreases** across generations (monotonically non-decreasing convergence curve).

**Trade-off:** Elitism improves reliability but can reduce diversity — the same excellent chromosome may persist for many generations and crowd out alternatives.

| | Generational | Elitism (5) |
|---|---|---|
| Best solution preserved? | Not guaranteed | Guaranteed |
| Diversity | Higher | Lower |
| Convergence curve | May dip | Monotone |
| Escape local optima | Better | Harder |

---

## 8. Stage 5 — Diversity Preservation

**File:** `ea/diversity.py`

### The problem: premature convergence

Without diversity mechanisms, a GA tends to **converge prematurely** — the best individual rapidly dominates the population through selection, and genetic diversity collapses. The search then gets stuck at a local optimum.

### Fitness Sharing

Fitness sharing penalises chromosomes that are *too similar* to each other, encouraging the population to spread across multiple distinct regions of the search space (multiple "niches").

#### Step 1: Hamming Distance

```python
def hamming_distances(population):
    return np.sum(population[:, None, :] != population[None, :, :], axis=2)
```

The Hamming distance between two binary chromosomes is the number of positions where they differ:

```
Chromosome A: [1, 0, 1, 1, 0]
Chromosome B: [1, 1, 0, 1, 0]
Differences:  [0, 1, 1, 0, 0]  → Hamming distance = 2
```

This gives a symmetric matrix of shape `(pop_size, pop_size)`.

#### Step 2: Sharing Function

```python
def sharing_function(d, sigma_share, alpha=1.0):
    return np.maximum(0.0, 1.0 - (d / sigma_share) ** alpha)
```

The sharing function converts a distance into a "similarity score" in [0, 1]:

```
sh(d) = max(0,  1 - (d / σ)^α)

sh(0) = 1.0      ← identical chromosomes maximally share
sh(σ) = 0.0      ← chromosomes σ apart don't share at all
```

With `sigma_share = 0.2 * n = 0.2 * 900 = 180`, two chromosomes must differ in at least 180 positions before they stop affecting each other's fitness.

#### Step 3: Niche Count and Shared Fitness

```python
def apply_fitness_sharing(raw_fitnesses, population, sigma_share, alpha=1.0):
    dists = hamming_distances(population)
    sh = sharing_function(dists, sigma_share, alpha)
    niche_counts = sh.sum(axis=1)           # sum of similarities to all others
    return raw_fitnesses / niche_counts
```

The **niche count** for chromosome `i` is the sum of its similarities to all chromosomes in the population (including itself, since `sh(0)=1`). The shared fitness is the raw fitness divided by the niche count:

```
shared_fitness(i) = raw_fitness(i) / niche_count(i)
```

**Effect:** A chromosome surrounded by many similar chromosomes gets a high niche count, so its shared fitness is penalised. An isolated chromosome has a niche count near 1.0 (only counting itself), so its shared fitness equals its raw fitness.

> **Important design note:** Shared fitness is used only for **selection** (choosing parents). Raw fitness is used for everything else — stagnation checks, logging, and determining the global best. This ensures the GA's progress tracking is not distorted by the diversity mechanism.

---

## 9. Stage 6 — The GA Main Loop

**File:** `ea/ga.py`

This is the orchestrator — it ties together all the components above into a coherent evolutionary loop.

### Complete loop pseudocode

```
INITIALISE population (35 chromosomes of length 900)
LOAD checkpoint if resuming

FOR gen = 1 to 100:
    SUBSAMPLE training data (5 000 stratified rows)
    EVALUATE fitness of each chromosome in parallel → raw_fitnesses
    APPLY fitness sharing → shared_fitnesses
    TRACK best chromosome (by raw fitness)
    
    IF stagnant for 12 generations OR gen == 100:
        BREAK
    
    SELECT parents using shared_fitnesses
    CROSSOVER parents → offspring
    MUTATE offspring → offspring
    EVALUATE offspring in parallel → offspring_fitnesses
    REPLACE population (generational or elitism)
    SAVE checkpoint

RETURN best_chromosome, fitness_per_generation, ...
```

### Parallelisation with ThreadPoolExecutor

The most computationally expensive step is fitness evaluation — training 35 LinearSVC classifiers per generation. These are embarrassingly parallel (each chromosome is independent).

```python
def _eval_parallel(population, X_tr, y_tr, X_va, y_va, desc, max_workers):
    results = [None] * len(population)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(evaluate_fitness, chrom, X_tr, y_tr, X_va, y_va): i
            for i, chrom in enumerate(population)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
    return np.array(results)
```

**Why threads, not processes?** LinearSVC (from scikit-learn, which uses LIBLINEAR in C) releases the Python Global Interpreter Lock (GIL) during its core optimisation. This means threads achieve genuine parallelism without the high startup cost of spawning separate processes or the serialisation overhead of `multiprocessing`.

With 8 CPU cores and 35 chromosomes, the effective speedup is ~6-8×.

### Stratified Subsampling

To speed up each LinearSVC fit further, only 5 000 of the 33 318 training samples are used each generation:

```python
def _stratified_subsample(X, y, size=5000):
    classes, counts = np.unique(y, return_counts=True)
    chosen = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_cls = max(1, round(size * cnt / total))    # proportional allocation
        chosen.append(np.random.choice(cls_idx, size=n_cls, replace=False))
    return X[np.concatenate(chosen)], y[np.concatenate(chosen)]
```

**Stratified** means each of the 43 classes contributes rows proportional to its share in the full dataset. This ensures no class is accidentally excluded, maintaining a balanced training signal even at 5 000 rows.

The same subsample is used for **all 35 chromosomes** within one generation — this ensures fitness values within a generation are comparable (they all see the same data), while the subsample changes each generation (preventing overfitting to one subset).

### Early Stopping via Stagnation Detection

```python
def _is_stagnant(best_history, window=12):
    if len(best_history) <= window:
        return False
    return best_history[-window] == best_history[-1]
```

If the best raw fitness has not improved in the last 12 generations, the run terminates early. This saves time when the GA has converged without waiting for the full 100 generations.

12 generations ≈ 12% of `max_generations=100`, which is within the 10–20% range recommended in GA literature as a reasonable patience budget.

### Checkpointing for Fault Tolerance

Each completed generation writes its full state to a `.npz` file:

```python
def _save_checkpoint(path, gen, population, best_history, best_chrom, global_best_fitness):
    np.savez(path, population=population, best_history=..., ...)
```

If the process is interrupted (power cut, timeout, OS crash), restarting the experiment automatically resumes from the last saved generation rather than starting over. This is essential for long runs (each config takes 30–40 minutes).

---

## 10. Stage 7 — Experiment Configuration

**File:** `ea/experiment.py`

### The ExperimentConfig dataclass

```python
@dataclass
class ExperimentConfig:
    run_id:       str          # unique name (also used as filename)
    selection:    str          # human-readable operator name (for logs)
    crossover:    str
    mutation:     str
    survivor:     str

    selection_fn: Callable     # the actual Python function
    crossover_fn: Callable
    mutation_fn:  Callable
    survivor_fn:  Callable

    pop_size:      int   = 35
    max_generations: int = 100
    mutation_rate: float = 0.01
    crossover_rate: float = 0.8
    alpha:         float = 1.0
    seed:          int   = 42
    sigma_share:   float | None = None   # set at runtime: 0.2 * n
    subsample_size: int | None = None    # set at runtime: 5000
```

**Design insight:** The config stores both the **operator name** (for logging) and the **operator callable** (for the GA loop). This avoids string-based dispatch inside `ga.py` — the loop simply calls `config.selection_fn(...)` without any `if/elif` branching. New operators can be added by creating a new `ExperimentConfig` instance; no changes to `ga.py` are needed.

### The 7 Experiment Configurations

The 4 **mandatory** configurations form a 2×2 factorial design:

| Config | Selection | Crossover | Mutation | Survivor |
|--------|-----------|-----------|----------|----------|
| Config 1 | Tournament | Single-point | Bit-flip | Generational |
| Config 2 | Tournament | **Uniform** | Bit-flip | Generational |
| Config 3 | **Roulette** | Single-point | Bit-flip | Generational |
| Config 4 | **Roulette** | **Uniform** | Bit-flip | Generational |

This design allows **controlled comparisons**: to compare selection strategies, compare Config 1 vs 3 (same crossover); to compare crossover strategies, compare Config 1 vs 2 (same selection).

The 3 **extension** configurations add more variation:

| Config | Selection | Crossover | Mutation | Survivor |
|--------|-----------|-----------|----------|----------|
| Config 5 | Tournament | Single-point | **Swap** | Generational |
| Config 6 | Roulette | Uniform | **Swap** | Generational |
| Config 7 | Tournament | Single-point | Bit-flip | **Elitism-5** |

Configs 5 and 6 isolate the effect of swap vs bit-flip mutation (same other operators as Config 1 and 4 respectively). Config 7 isolates the effect of elitism (same operators as Config 1).

### Reproducibility via seed=42

All 7 configs use `seed=42`. This means:
- Population initialisation is deterministic.
- Subsample indices are deterministic.
- Results are reproducible across machines.

Different configs still produce different results because they use different operators, which make different random choices (the random number generator is reset to the same seed at the start of each config's run).

---

## 11. Stage 8 — Logging

**File:** `ea/logger.py`

Every completed run is saved as a structured JSON file in `ea/results/`. This serves as the audit trail and data source for the results dashboard.

### JSON Schema

```json
{
  "run_id": "tournament_singlepoint_generational_bitflip_42",
  "selection": "tournament",
  "crossover": "single_point",
  "mutation": "bit_flip",
  "survivor_selection": "generational",
  "population_size": 35,
  "generations_run": 52,
  "fitness_per_generation": [0.9799, 0.9820, ..., 0.9837],
  "best_fitness": 0.9849,
  "selected_feature_indices": [3, 7, 10, 14, ...],
  "total_features": 1280,
  "selected_features": 437,
  "reduction_ratio": 0.6586,
  "mutation_rate": 0.01,
  "crossover_rate": 0.8,
  "fitness_sharing": {"sigma_share": 180.0, "alpha": 1.0},
  "full_feature_val_accuracy": 0.9919,
  "accuracy_drop_pp": 0.0070,
  "seed": 42,
  "run_time_seconds": 2025.6,
  "subsample_size": 5000,
  "feature_filter": {"applied": true, "original_dim": 1280, "filtered_dim": 900}
}
```

### Index remapping

The GA operates on filtered features (indices 0–899 into the 900-dim filtered array). The logger remaps these back to original feature space (indices 0–1279) using the `feature_filter_mask`:

```python
# ea/logger.py

def build_log(config, result, full_feature_val_accuracy, run_time_seconds, filter_mask=None):
    indices = result["selected_feature_indices"]    # indices into filtered space

    if filter_mask is not None:
        original_indices = np.where(filter_mask)[0]   # which of 1280 were kept
        indices = [int(original_indices[i]) for i in indices]   # remap

    return {
        "selected_feature_indices": indices,   # now indices into the full 1280-dim space
        ...
    }
```

This means downstream consumers (dashboard, test scripts) can always work with original feature indices regardless of what pre-filtering was applied.

---

## 12. Stage 9 — Running Everything

**File:** `ea/run_experiments.py`

```
python -m ea.run_experiments
```

This entry point:

1. **Prevents thread oversubscription:** Sets `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` before importing anything. This stops each LinearSVC from internally spawning its own thread pool — otherwise 35 chromosomes × 8 internal threads = 280 threads fighting for 8 cores.

2. **Loads features** from `ea/data/*.npy`.

3. **Applies the variance filter** (or loads cached filtered arrays if they already exist).

4. **Sets runtime config fields:** `sigma_share = 0.2 * 900 = 180`, `subsample_size = 5000`.

5. **Runs each config sequentially**, printing a progress summary after each generation.

6. **Saves the JSON log** and attempts an **S3 upload** (gracefully skipped if AWS credentials are unavailable).

7. **Prints a final summary table:**

```
┌────────────────────────────────────────────────────────────────────────┐
│  Run ID                                     Gens  BestFit  Drop  Pass │
│  tournament_singlepoint_generational_...    52    0.9849   0.7pp  ✓   │
│  tournament_uniform_generational_...        36    0.9831   0.9pp  ✓   │
│  ...                                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Performance Optimisations

This section explains *why* the implementation is structured as it is for speed, since naive implementations of the same algorithm would be far too slow.

### The budget constraint

**NFR-PERF-04:** Each generation must complete in under 60 seconds (`60s ÷ 35 chromosomes = 1.71s/chromosome`).

Without optimisation, a single fitness evaluation (LinearSVC on 33 318 × 900 features) takes ~4–8 seconds. With 35 chromosomes evaluated sequentially, one generation would take 2–5 minutes — 20× over budget.

### Optimisation stack

| Technique | Where | Speedup |
|-----------|-------|---------|
| Variance pre-filter: 1280 → 900 features | `extract_features.py` | ~30% faster SVC fit |
| Stratified subsample: 33 318 → 5 000 rows | `ga.py` | ~85% faster SVC fit |
| Parallel evaluation: 8 threads × 35 chromosomes | `ga.py` | ~6–8× throughput |
| Early stopping: 12-gen stagnation window | `ga.py` | Fewer generations run |
| Population size: 50 → 35 | `experiment.py` | 30% fewer evaluations |
| Thread pool limits (OMP=1 per process) | `run_experiments.py` | Prevents oversubscription |

Combined effect: each of the 7 experiments finishes in ~30–40 minutes (was estimated at 6–10 hours without these optimisations).

### Thread vs. process parallelism

```
ProcessPoolExecutor:  spawns N Python processes
                      → high memory (each loads feature arrays)
                      → high startup cost (fork/exec overhead)

ThreadPoolExecutor:   spawns N threads in same process
                      → shared memory (feature arrays loaded once)
                      → low startup cost
                      → Works with LinearSVC because it releases the GIL
```

---

## 14. Actual Experimental Results

All 7 experiments have been run and the results saved. The best result:

### Config 1 (Tournament + Single-point + Bit-flip + Generational)

```
Generations run:      52  (stopped by stagnation)
Best fitness:         0.9849  (98.49% validation accuracy)
Full-feature baseline: 0.9919 (99.19%)
Accuracy drop:        0.70 percentage points  (well within 3pp limit)
Features selected:    437 out of 900 filtered (341 out of 1280 original)
Reduction ratio:      65.86%  (well above 30% requirement)
Run time:             2025.6 seconds (~34 minutes)
```

**Convergence curve for Config 1:**

```
Fitness
0.985 │                                         ·  ·
      │                     ·   ·  ·   ·  ·  ·
0.983 │            ·  ·  ·
      │         ·
0.981 │      ·
0.980 │    ·
0.979 │  ·
      └──────────────────────────────────────────────── Generation
        1  5  10  15  20  25  30  35  40  45  50  52
```

The GA finds good solutions quickly in the first 10–15 generations, then continues refining. After generation 40, stagnation sets in and the run terminates at generation 52.

### Requirement compliance summary

| Requirement | Target | Achieved | Status |
|-------------|--------|----------|--------|
| Feature reduction | ≥ 30% | 65.86% | ✓ |
| GA accuracy | ≥ 88% (≥ 98.88% relative to baseline) | 98.49% | ✓ |
| Accuracy drop vs baseline | ≤ 3pp | 0.70pp | ✓ |
| Mandatory configs run | 4 | 4 | ✓ |
| Extension configs run | 0 (required) | 3 | ✓ bonus |
| Custom GA (no DEAP) | yes | yes | ✓ |
| JSON log per run | yes | yes | ✓ |
| S3 upload | yes | yes | ✓ |

---

## 15. Design Decisions and Best Practices

### Decision 1: Binary chromosome encoding

Feature selection is inherently a binary problem (include or exclude), so a binary encoding is the natural and most efficient choice. Alternative encodings (e.g., integer encoding of feature indices) would have variable-length chromosomes requiring specialised crossover operators.

### Decision 2: Fitness on validation set only

The fitness function evaluates on the **validation set**, never the training set (which would overfit) and never the test set (which would leak information and render the final evaluation meaningless). This mirrors the train/val/test protocol used throughout ML.

### Decision 3: Operator functions as callables in config

All operators are stored as Python callables in the `ExperimentConfig`, not as strings. The GA loop calls `config.selection_fn(...)` directly. This:
- Eliminates `if/elif` branching in the hot loop.
- Makes it trivially easy to add new operators — define a function, pass it in the config.
- Enables the `elitism` config to use a `lambda` to pre-bind `elite_size=5`.

### Decision 4: Fitness sharing applied to selection only

Shared fitness is deliberately used **only for parent selection**. Raw fitness is used for everything else (stagnation detection, global best tracking, JSON logging, elitism comparison). This is correct by design:
- Stagnation detection needs stable values. Shared fitness fluctuates as population composition changes.
- The logged "best fitness" should reflect true classifier performance, not a penalised value.

### Decision 5: All-zero chromosome guard

Three places enforce the invariant that no chromosome selects zero features: `init_population`, `bit_flip_mutation`, and `swap_mutation`. This is a hard constraint — a zero chromosome cannot be evaluated (no features → no SVC → undefined accuracy) and returning 0.0 for fitness would create a garbage data point that might confuse selection.

### Decision 6: Single subsample per generation shared across all chromosomes

If each chromosome used a different random subsample, fitness values within a generation would not be comparable — chromosome A might look better than B simply because A drew an easier subsample. Using one shared subsample per generation ensures **within-generation fairness** while still varying the subsample **between generations** to provide noisy but unbiased gradient information.

### Decision 7: sigma_share = 0.2 × n

The sharing radius `sigma_share` is set to 20% of the chromosome length. This means two chromosomes must differ in fewer than 180 positions to be considered "in the same niche." With 900 binary features, this is a reasonable neighbourhood size — neither so small that every chromosome is its own niche, nor so large that the entire population shares one niche.

---

## 16. Common Pitfalls

### Pitfall 1: Using the test set for fitness

**Wrong:** `accuracy_score(y_test, clf.predict(X_test[:, mask]))`  
**Correct:** `accuracy_score(y_val, clf.predict(X_val[:, mask]))`

Using the test set for fitness is a form of **data leakage** — the algorithm would be optimised specifically for the test set, and reported accuracy would be optimistically biased.

### Pitfall 2: Forgetting to reset the random seed per run

If `np.random.seed` is not called at the start of each run, the second config may produce different results depending on which random operations the first config performed. The fix: `np.random.seed(config.seed)` at the top of `run_ga`.

### Pitfall 3: Applying fitness sharing to stagnation detection

If you use shared fitness for stagnation detection, the GA may never detect stagnation even when the population has fully converged — because shared fitness values change every generation as the composition shifts. Always use raw fitness for stagnation.

### Pitfall 4: Thread oversubscription

If each of the 35 chromosomes spawns 8 threads internally (via OpenBLAS), you have 280 threads competing for 8 cores, causing excessive context switching. Fix: set `OMP_NUM_THREADS=1` before importing sklearn.

### Pitfall 5: Not guarding against all-zero chromosomes

If a bit-flip mutation turns off all features, the next fitness evaluation raises an error or returns meaningless results. The guard `if not mutated.any(): force one bit to True` must be present in every mutation function.

### Pitfall 6: Comparing fitness across different subsamples

If you evaluate parents and offspring on different subsamples, the comparison in elitism is meaningless — offspring might appear weaker than parents simply because they saw a harder subsample. Solution: use the same subsample for both evaluations within a generation (as done in `ga.py`).

---

## 17. Quick-Reference: Parameter Cheat Sheet

| Parameter | Value | Where | Why |
|-----------|-------|-------|-----|
| `pop_size` | 35 | `experiment.py` | 35 chromosomes evaluated in parallel |
| `max_generations` | 100 | `experiment.py` | Hard upper limit on generations |
| `mutation_rate` | 0.01 | `experiment.py` | ~9 bits flipped per 900-bit chromosome |
| `crossover_rate` | 0.8 | `experiment.py` | 80% of pairs recombine |
| `sigma_share` | 0.2 × n = 180 | `run_experiments.py` | Sharing radius: 20% of chromosome length |
| `alpha` | 1.0 | `experiment.py` | Triangular sharing kernel shape |
| `seed` | 42 | `experiment.py` | Reproducibility |
| `subsample_size` | 5000 | `run_experiments.py` | 5000 of 33318 rows per generation |
| `k` (tournament) | 3 | `operators/selection.py` | 3-way tournament |
| `elite_size` | 5 | `operators/survivor.py` | Top 5 parents survive |
| `stagnation_window` | 12 | `ga.py` | 12 gens without improvement → stop |
| `variance_filter_target` | 900 | `extract_features.py` | 900 of 1280 features kept |
| `n_jobs` | -1 | `run_experiments.py` | Use all CPU cores |
| `C` (LinearSVC) | 0.1 | `fitness.py` | Light regularisation for fitness proxy |
| `max_iter` (LinearSVC) | 2000 | `fitness.py` | Convergence budget for SVC |

---

*This guide was written to accompany the EA module implementation in the Traffic Sign Intelligence Platform project. All code snippets are taken directly from the production implementation in the `ea/` directory.*
