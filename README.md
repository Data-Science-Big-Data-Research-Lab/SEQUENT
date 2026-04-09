# SEQUENT — Selective Qubit Entanglement for Efficient Quantum Feature Map Construction

> **Quantum Machine Intelligence** (Springer Nature) — under review

SEQUENT is a methodology for constructing **data-adaptive, resource-efficient quantum feature maps** for NISQ-era devices. Instead of using fixed entanglement patterns (linear, full, ring), SEQUENT uses classical feature selection and a metaheuristic search to discover which qubit pairs should actually be entangled for a given dataset — reducing two-qubit gate count and circuit depth without sacrificing classification accuracy.

---

## Core Idea

A quantum feature map encodes classical data into a quantum state. The entanglement structure of that map — which qubits interact with which — is typically fixed by hand (linear, full, circular). SEQUENT argues this choice should be **learned from the data**.

The pipeline has four stages:

```
Raw data
   │
   ▼
① Classical Feature Selection (ANOVA / Autoencoder)
   │   Reduces dimensionality → fewer qubits needed
   ▼
② Entanglement Space Construction
   │   Pearson correlation between selected features → candidate qubit pairs
   │   Binary vector of length C(k,2): bit i = 1 means "entangle pair i"
   ▼
③ Metaheuristic Search (Simulated Annealing / Genetic Algorithm)
   │   Searches over 2^C(k,2) possible entanglement configurations
   │   Each candidate is evaluated by training a quantum model on a validation split
   │   GA uses internal k-fold CV to prevent overfitting to small validation sets
   ▼
④ Selected Feature Map → Quantum Model Training & Evaluation
      PegasosQSVC (QSVM) or NeuralNetworkClassifier (QNN)
      Tested in three environments: statevector · noise emulator · real IBM hardware
```

The key novelty is that **no prior work combines all four of these elements**:

| Component | What makes it novel |
|---|---|
| FS → entanglement | Feature correlations directly determine which qubits to entangle |
| Binary search space | Entanglement is a discrete selection problem, not a parameter optimisation |
| NISQ-aware objective | Fitness function penalises accuracy loss; circuit depth reduced as a side effect |
| Three-environment validation | Statevector · noise emulator (AerSimulator + IBM noise model) · real IBM hardware |

---

## Repository Structure

```
SEQUENT/
├── main.py               # Single entry point — run_experiment()
├── tools.py              # Data loading, feature maps, models, metrics
├── metaheuristicas.py    # Simulated Annealing and Genetic Algorithm
├── requirements.txt      # Python dependencies
├── datasets/             # Local datasets (TSV/CSV format)
│   ├── breast-w.tsv
│   ├── fitness_class_2212.csv
│   ├── flare.tsv
│   └── heart.csv
├── results/              # Auto-created on first run
│   ├── benchmark_results.csv   # Aggregated summary (one row per experiment)
│   └── *.json                  # Full per-run detail (metrics, solution, complexity)
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/sequent.git
cd sequent
pip install -r requirements.txt
```

> **IBM Quantum credentials** (required for noise emulator and hardware modes only):
> ```python
> from qiskit_ibm_runtime import QiskitRuntimeService
> QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
> ```

---

## Quick Start

```python
from main import run_experiment

# Minimal run — PMLB dataset, statevector, SA, no FS
run_experiment(dataset="corral", option=1, mode="statevector")

# Full experiment — local dataset, ANOVA FS, SA, 5 independent runs
run_experiment(
    dataset="heart",
    option=0,
    path="./datasets/heart.csv",
    use_fs=True,
    fs_method="anova",
    k=5,
    mode="statevector",
    metaheuristic="sa",
    n_runs=5,
    run_baselines=True,
    run_ablation=True,
)
```

Results are saved automatically:
- `results/benchmark_results.csv` — one row per experiment (appended, never overwritten)
- `results/<run_id>.json` — full log including per-class metrics, circuit complexity, significance tests and the binary entanglement solution vector

---

## Parameters

### `run_experiment()`

| Parameter | Default | Description |
|---|---|---|
| `dataset` | — | PMLB dataset name (`option=1`) or local file tag (`option=0`) |
| `option` | `1` | `0` = local CSV/TSV at `path`, `1` = PMLB |
| `path` | `None` | Path to local file (required when `option=0`) |
| `model_type` | `"qsvm"` | `"qsvm"` (PegasosQSVC) or `"qnn"` (NeuralNetworkClassifier) |
| `mode` | `"statevector"` | `"statevector"` · `"noise"` · `"hardware"` |
| `reps` | `1` | Repetitions of the ZZFeatureMap encoding block |
| `use_fs` | `True` | Apply feature selection before building the entanglement map |
| `fs_method` | `"anova"` | `"anova"` · `"autoencoder"` |
| `k` | `5` | Number of features to keep when `use_fs=True` |
| `metaheuristic` | `"sa"` | `"sa"` (Simulated Annealing) or `"ga"` (Genetic Algorithm) |
| `n_runs` | `5` | Independent runs with different data splits for statistical reporting |
| `run_baselines` | `True` | Evaluate linear / ring / full entanglement baselines |
| `run_ablation` | `True` | Run FS-only and entanglement-only ablation variants |

### SA hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `sa_initial_temp` | `10.0` | Starting temperature — set ~10× typical Δcost to allow early exploration |
| `sa_cooling_rate` | `0.90` | Multiplicative cooling factor per iteration |
| `sa_stopping_temp` | `0.01` | Algorithm halts when T falls below this value |
| `sa_max_iterations` | `20` | Maximum number of temperature steps |
| `sa_num_neighbors` | `5` | Candidates evaluated per temperature step (~100 total evals/run) |

### GA hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `ga_population_size` | `20` | Individuals per generation |
| `ga_n_generations` | `10` | Generations to evolve |
| `ga_crossover_rate` | `0.8` | Probability of crossover between two parents |
| `ga_mutation_rate` | `None` | Per-bit flip probability; defaults to `1/chromosome_length` |
| `ga_tournament_size` | `2` | Selection pressure — lower values preserve more diversity |
| `ga_elitism_count` | `2` | Top individuals copied unchanged to the next generation |
| `ga_cv_folds` | `3` | Internal stratified k-fold CV folds used as GA fitness function. Prevents overfitting to a small validation split. Set to `0` to revert to single val split |

---

## Statistical Rigor

Each of the `n_runs` repetitions uses `seed = BASE_SEED + run_idx` for **both** the train/val/test partition and the metaheuristic. This means every run sees a genuinely different data split, making reported mean ± std and 95% CI estimates reflect model uncertainty over unseen data, not just metaheuristic variance. This design is equivalent to repeated random sub-sampling.

### Confidence intervals

All aggregated metrics report a **95% confidence interval** computed using the t-distribution with `n_runs − 1` degrees of freedom:

```
CI₉₅ = t₀.₉₇₅(n−1) × std / √n
```

### Significance tests

After all runs complete, three paired one-sided tests (H₁: SEQUENT > baseline) compare SEQUENT against each quantum baseline (linear, ring, full entanglement) evaluated on the **same splits**:

| Test | Type | Notes |
|---|---|---|
| **Permutation test** | Non-parametric, exact | Primary test. No minimum sample size. Enumerates all 2ⁿ sign assignments of paired differences |
| **Sign test** | Non-parametric, exact | Counts wins vs losses; ties dropped. Most conservative |
| **Paired t-test** | Parametric | Included for reviewer convention; interpret with caution at n < 10 |

The comparison against the classical SVM is also reported for reference, but the **primary statistical claim** is the quantum-to-quantum comparison (SEQUENT vs linear / ring / full entanglement).

When `model_type="qnn"`, an additional paired test compares the QNN against the classical MLP baseline.

Results are printed to console, saved to the JSON log and appended to the CSV with columns `linear_perm_p`, `ring_perm_p`, `full_perm_p`, etc.

---

## Experiment Grid

The full factorial experiment covers all combinations of:

| Axis | Values |
|---|---|
| Model | `qsvm`, `qnn` |
| Metaheuristic | `sa`, `ga` |
| Feature selection | `anova`, `autoencoder` |
| Reps | `1`, `3`, `5` |
| n\_runs | `5` (fixed) |
| k | `5` (fixed) |

This produces **24 configurations × 5 datasets = 120 experiments**.

Baselines (linear / ring / full entanglement) are evaluated only at `reps=1`, since their circuits are independent of reps. The ablation study (FS-only, entanglement-only) is only run for the canonical configuration (`reps=1`, `anova`, `sa`) to avoid redundant computational expense.

---

## Datasets

| Dataset | Samples | Features | Class balance | Source |
|---|---|---|---|---|
| Corral | 160 | 6 | 56% / 44% | PMLB |
| BreastW | 699 | 9 | 63% benign / 37% malignant | Local TSV |
| Fitness | 1500 | 6 | 70% class 1 / 30% class 0 | Local CSV |
| Flare | ~1066 | **11** | varies | Local TSV |
| Heart | 918 | varies | 44% normal / 56% disease | Local CSV |

> **Note on Flare**: the deprecated version of the dataset is used, which contains **11 features**. Some mirrors of the dataset report 10 features; the version used here is the original with the additional attribute included.

Three datasets (BreastW, Fitness, Heart) are class-imbalanced. For these, accuracy alone is insufficient — the paper reports and analyses macro-averaged precision, recall and F1, as well as per-class breakdowns.

---

## Models

### PegasosQSVC (QSVM)

The default model. Uses a `FidelityQuantumKernel` built on the SEQUENT feature map, optimised with the Pegasos SGD-SVM algorithm (`C=1000`, `num_steps=100`).

### NeuralNetworkClassifier (QNN)

A parameterised quantum circuit composed of the SEQUENT feature map followed by a `TwoLocal` ansatz (Ry + Rz rotations, CX linear entanglement). Trained with COBYLA (maxiter=100).

| Mode | Backend | Notes |
|---|---|---|
| `statevector` | Default Qiskit sampler | No noise, fastest |
| `noise` | `AerSimulator` + IBM noise model | GPU-accelerated via tensor network method |
| `hardware` | Real IBM Quantum device | ALAP scheduling + XX dynamical decoupling |

### Classical MLP baseline (QNN track only)

When `model_type="qnn"`, a shallow classical MLP is evaluated on every split alongside the QNN. The architecture (Linear → 64 → 32 → n\_classes, BatchNorm + ReLU + Dropout) is deliberately comparable in depth to the quantum circuits, ensuring a fair classical-vs-quantum comparison. Training uses Adam with early stopping on a 10% held-out split. A paired significance test (SEQUENT QNN vs MLP) is computed and saved alongside the quantum baseline comparisons.

---

## Metaheuristics

Both optimisers share the same interface and are fully interchangeable.

### Simulated Annealing

At each temperature step, `num_neighbors` candidates are generated by flipping a random bit in the current solution. A worse candidate is accepted with probability exp(−Δcost / T), allowing escape from local optima early in the search. The initial temperature `sa_initial_temp=10.0` is set approximately 10× the typical accuracy difference between neighbouring solutions, ensuring genuine thermal exploration before cooling constrains the search.

### Genetic Algorithm

A steady-state GA with:
- **Tournament selection** with `tournament_size=2` — lower pressure than the canonical value of 3, preserving diversity in small populations
- **Uniform crossover** — each bit independently chosen from either parent with p = 0.5
- **Bit-flip mutation** with default rate `1/chromosome_length`
- **Elitism** — top `elitism_count=2` individuals always survive
- **Internal k-fold CV** (`ga_cv_folds=3`) — fitness is the mean accuracy over stratified 3-fold CV on the combined train+val set. This prevents the GA from converging to entanglement maps that overfit a small validation partition, a failure mode that does not affect SA due to its natural thermal exploration
- **Memoisation** — quantum evaluations already computed are never repeated

---

## Metrics and Output

Every experiment reports the following metrics on the test set:

| Metric | Why it matters |
|---|---|
| Accuracy | Overall correctness |
| Precision (macro) | Per-class average — not dominated by majority class |
| Recall (macro) | Per-class average — critical for imbalanced datasets |
| F1 (macro) | Harmonic mean of precision and recall |
| Per-class breakdown | Precision / recall / F1 / support for each class individually |
| Training time | Seconds |
| Inference time | Seconds |
| Circuit depth | Hardware friendliness indicator |
| Total gates | Resource cost |
| Two-qubit gates | Primary source of noise in NISQ devices |
| Search space size | 2^C(k,2) — scalability analysis |

Macro averaging is used throughout because three of the five datasets are class-imbalanced. Per-class metrics are logged to JSON, printed to console and averaged with ± std and 95% CI across runs.

### Console output example

```
╔══════════════════════════════════════════════════════════════════════╗
║  SEQUENT  |  heart  |  QSVM  |  statevector  |  SA  |  FS:True     ║
╚══════════════════════════════════════════════════════════════════════╝

┌── Baseline: Linear Entanglement  (5 runs, raw features, mean ± std [95% CI]) ┐
│  test (aggregated)  Acc:0.8312 ±0.0201 [95%CI ±0.0250] ...
└──────────────────────────────────────────────────────────────────────────────┘

┌── SEQUENT  —  Aggregated (5 runs, mean ± std [95% CI]) ──────────────────────┐
│  test               Acc:0.8804 ±0.0098 [95%CI ±0.0122]  F1:0.8798 ±0.0102
│  Per-class test metrics (mean ± std over 5 runs):
│    class 0     f1:0.8611 ±0.0143 [95%CI ±0.0178]
│    class 1     f1:0.8985 ±0.0071 [95%CI ±0.0088]
└──────────────────────────────────────────────────────────────────────────────┘

┌── Statistical Significance  (SEQUENT vs Linear entanglement) ────────────────┐
│  n_runs = 5  (one-sided tests, H1: SEQUENT > baseline)
│  Permutation test  (primary):    p = 0.0312  *** p<0.05
│  Sign test         (robust):     p = 0.0625  not significant  (wins=4, ties=0)
│  Paired t-test     (parametric): p = 0.0289  *** p<0.05
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Ablation Study

Setting `run_ablation=True` runs two additional variants that isolate the contribution of each component:

| Variant | FS | Search | Purpose |
|---|---|---|---|
| **FS-only** | ✓ | ✗ (all pairs entangled) | How much does feature reduction alone explain? |
| **Entanglement-only** | ✗ | ✓ (raw features) | How much does entanglement search alone explain? |
| **SEQUENT** | ✓ | ✓ | Full methodology |

Ablation is run only for the canonical configuration (`reps=1`, `anova`, `sa`) on split-0, since its purpose is qualitative component analysis rather than statistical estimation.

---

## Reproducibility

All experiments use a fixed base seed:

```python
BASE_SEED = 12345
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
algorithm_globals.random_seed = BASE_SEED
```

Multi-run experiments offset the seed by run index (`BASE_SEED + run_idx`) for **both** the data partition and the metaheuristic, ensuring each run is independent while remaining fully reproducible. The complete list of split seeds used is saved to the JSON log under `split_seeds`.

---

## Citation

<!--
```bibtex
@article{sequent2025,
  title   = {Selective Qubit Entanglement for Efficient Quantum Feature Map Construction},
  author  = {F. Rodríguez-Díaz and D. Gutiérrez-Avilés and A. Troncoso and F. Martínez-Álvarez},
  journal = {Quantum Machine Intelligence},
  year    = {2025},
  publisher = {Springer Nature}
}
```
-->

---

## License

This project is licensed under the MIT License.
