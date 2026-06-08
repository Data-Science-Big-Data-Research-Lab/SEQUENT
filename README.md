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
③ Metaheuristic Search (Simulated Annealing / Iterated Local Search)
   │   Searches over 2^C(k,2) possible entanglement configurations
   │   Each candidate is evaluated by training a quantum model on a validation split
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
├── metaheuristicas.py    # Simulated Annealing and Iterated Local Search
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
run_experiment(dataset="corral", option=1, path=None, cfg=dict(
    mode="statevector", model_type="qsvm", metaheuristic="sa",
    use_fs=False, k=5, reps=1, n_runs=1, run_baselines=False,
    objective_metric="accuracy",
))

# Full experiment — local dataset, ANOVA FS, SA, 10 independent runs
run_experiment(
    dataset="heart",
    option=0,
    path="./datasets/heart.csv",
    cfg=dict(
        mode="statevector",
        model_type="qsvm",
        metaheuristic="sa",
        use_fs=True,
        fs_method="anova",
        k=5,
        reps=1,
        n_runs=10,
        run_baselines=True,
        objective_metric="accuracy",
    )
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
| `metaheuristic` | `"sa"` | `"sa"` (Simulated Annealing) or `"ils"` (Iterated Local Search) |
| `n_runs` | `10` | Independent runs with different data splits for statistical reporting |
| `run_baselines` | `True` | Evaluate linear / ring / full entanglement baselines |
| `objective_metric` | `"accuracy"` | `"accuracy"` or `"f1_macro"` (auto-set for imbalanced datasets) |

### SA hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `sa_initial_temp` | `10.0` | Starting temperature — set ~10× typical Δcost to allow early exploration |
| `sa_cooling_rate` | `0.90` | Multiplicative cooling factor per iteration |
| `sa_stopping_temp` | `0.01` | Algorithm halts when T falls below this value |
| `sa_max_iterations` | `10` | Maximum number of temperature steps |
| `sa_num_neighbors` | `5` | Candidates evaluated per temperature step (~50 total evals/run) |

### ILS hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `ils_n_restarts` | `3` | Number of perturbation-and-search restarts |
| `ils_perturbation_strength` | `0.3` | Fraction of bits randomly flipped at each perturbation |
| `ils_local_search_iters` | `15` | SA iterations for each local search phase |
| `ils_local_search_initial_temp` | `5.0` | Initial temperature of the local SA |
| `ils_local_search_cooling_rate` | `0.85` | Cooling rate of the local SA |
| `ils_local_search_stopping_temp` | `1e-3` | Stopping temperature of the local SA |
| `ils_local_search_num_neighbors` | `3` | Neighbours evaluated per local SA step |
| `ils_use_warm_start` | `True` | Initialise from correlation-based warm start |
| `ils_repair` | `True` | Repair solutions that violate active-pair constraints |

---

## Statistical Rigor

Each of the `n_runs` repetitions uses `seed = BASE_SEED + run_idx` for **both** the train/val/test partition and the metaheuristic. This means every run sees a genuinely different data split, making reported mean ± std estimates reflect model uncertainty over unseen data, not just metaheuristic variance. This design is equivalent to repeated random sub-sampling.

### Significance test

After all runs complete, a **one-sided Wilcoxon signed-rank test** (H₁: SEQUENT > baseline) compares SEQUENT against each quantum baseline (linear, ring, full entanglement) evaluated on the **same splits**:

| Test | Type | Notes |
|---|---|---|
| **Wilcoxon signed-rank** | Non-parametric, paired | Primary test. Ranks the magnitudes of paired differences; more powerful than the sign test and does not assume normality |

The comparison against the classical SVM is also reported for reference, but the **primary statistical claim** is the quantum-to-quantum comparison (SEQUENT vs linear / ring / full entanglement).

Results are printed to console and saved to the JSON log and CSV.

---

## Experiment Grid

The full factorial experiment covers all combinations of:

| Axis | Values |
|---|---|
| Model | `qsvm`, `qnn` |
| Metaheuristic | `sa`, `ils` |
| Feature selection | `anova`, `autoencoder` |
| Reps | `1`, `2`, `3` |
| n\_runs | `10` |
| k | `5` |

Each configuration is run as an independent job. The `SEQUENT_JOB_INDEX` environment variable can be used to run a single job from the grid, enabling parallelisation via SLURM array jobs.

---

## Datasets

| Dataset | Samples | Features | Class balance | Source |
|---|---|---|---|---|
| Corral | 160 | 6 | 56% / 44% | PMLB |
| BreastW | 699 | 9 | 63% benign / 37% malignant | Local TSV |
| Fitness | 1500 | 7 | 70% class 1 / 30% class 0 | Local CSV |
| Flare | 1066 | 10 | varies | Local TSV |
| Heart | 918 | 11 | 44% normal / 56% disease | Local CSV |

Three datasets (BreastW, Fitness) are class-imbalanced. For these, accuracy alone is insufficient — the paper reports and analyses macro-averaged precision, recall and F1, as well as per-class breakdowns. The objective metric is automatically set to `f1_macro` for datasets with imbalance ratio > 1.5.

---

## Models

### PegasosQSVC (QSVM)

The default model. Uses a `FidelityQuantumKernel` built on the SEQUENT feature map, optimised with the Pegasos SGD-SVM algorithm (`C=1000`, `num_steps=500`).

### NeuralNetworkClassifier (QNN)

A parameterised quantum circuit composed of the SEQUENT feature map followed by a `TwoLocal` ansatz (Ry + Rz rotations, CX linear entanglement). Trained with COBYLA (maxiter=100).

| Mode | Backend | Notes |
|---|---|---|
| `statevector` | Default Qiskit sampler | No noise, fastest |
| `noise` | `AerSimulator` + IBM noise model | GPU-accelerated via tensor network method |
| `hardware` | Real IBM Quantum device | ALAP scheduling + XX dynamical decoupling |

### Classical MLP baseline (QNN track only)

When `model_type="qnn"`, a shallow classical MLP is evaluated on every split alongside the QNN. The architecture (Linear → 64 → 32 → n\_classes, BatchNorm + ReLU + Dropout) is deliberately comparable in depth to the quantum circuits, ensuring a fair classical-vs-quantum comparison. Training uses Adam with early stopping on a 10% held-out split. A paired Wilcoxon significance test (SEQUENT QNN vs MLP) is computed and saved alongside the quantum baseline comparisons.

---

## Metaheuristics

Both optimisers share the same interface and are fully interchangeable.

### Simulated Annealing

At each temperature step, `num_neighbors` candidates are generated by flipping a random bit in the current solution. A worse candidate is accepted with probability exp(−Δcost / T), allowing escape from local optima early in the search. The initial temperature `sa_initial_temp=10.0` is set approximately 10× the typical accuracy difference between neighbouring solutions, ensuring genuine thermal exploration before cooling constrains the search.

### Iterated Local Search

ILS alternates between a local search phase (a short SA run) and a perturbation step that randomly flips a fraction of bits in the current best solution, allowing the search to escape local optima. Key features:

- **Warm start**: the initial solution is derived from the Pearson correlation structure of the selected features, biasing the search toward highly correlated qubit pairs
- **Repair**: solutions violating the active-pair cardinality constraints (min/max active pairs) are automatically repaired after perturbation
- **Memoisation**: quantum evaluations already computed are never repeated across restarts

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
  journal = {},
  year    = {2025}
}
```
-->

---

## License

This project is licensed under the MIT License.
