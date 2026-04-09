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
① Classical Feature Selection (ANOVA / Mutual Information)
   │   Reduces dimensionality → fewer qubits needed
   ▼
② Entanglement Space Construction
   │   Pearson correlation between selected features → candidate qubit pairs
   │   Binary vector of length C(k,2): bit i = 1 means "entangle pair i"
   ▼
③ Metaheuristic Search (Simulated Annealing / Genetic Algorithm)
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
├── metaheuristicas.py    # Simulated Annealing and Genetic Algorithm
├── requirements.txt      # Python dependencies
├── datasets/             # Local datasets (TSV format)
│   ├── breast-w.tsv
│   ├── fitness_class_2212.tsv
│   ├── flare.tsv
│   └── heart.tsv
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
    path="./datasets/heart.tsv",
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
- `results/<run_id>.json` — full log including per-class metrics, circuit complexity and the binary entanglement solution vector

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
| `fs_method` | `"anova"` | `"anova"` · `"mutual_info"` · `"autoencoder"` (placeholder) |
| `k` | `5` | Number of features to keep when `use_fs=True` |
| `metaheuristic` | `"sa"` | `"sa"` (Simulated Annealing) or `"ga"` (Genetic Algorithm) |
| `n_runs` | `5` | Independent optimisation runs for statistical reporting (mean ± std) |
| `run_baselines` | `True` | Evaluate linear / ring / full entanglement baselines |
| `run_ablation` | `True` | Run FS-only and entanglement-only ablation variants |

### Speed guide for metaheuristic search

Each metaheuristic iteration trains a full quantum model, so the number of evaluations directly controls wall-clock time. The values below give a good accuracy/speed trade-off on datasets with ≤ 200 samples in statevector mode:

| Parameter | Recommended value | Rationale |
|---|---|---|
| `C` (QSVM) | `1000` | Negligible accuracy loss vs 5000 on small datasets |
| `num_steps` (Pegasos) | `100` | **5× faster** than 500; accuracy impact < 1% |
| `sa_initial_temp` | `1.0` | Low T₀ → fast convergence, sufficient for small search spaces |
| `sa_max_iterations` | `10` | |
| `sa_num_neighbors` | `3` | 10 × 3 = **~30 evaluations per run** |
| `ga_population_size` | `10` | |
| `ga_n_generations` | `5` | **~35–50 unique evaluations** (memoisation eliminates repeats) |

With `mode="statevector"` each evaluation takes ~3–8 s → one SA run ≈ 1.5–4 min.

---

## Datasets

Four real-world binary classification datasets are included in `datasets/`. All are in TSV format with a `target` column.

| Dataset | Samples | Features (after preprocessing) | Class balance |
|---|---|---|---|
| BreastW | 699 | 9 | 63% benign / 37% malignant |
| Fitness | 1500 | 6 | 70% class 1 / 30% class 0 |
| Flare | ~1000 | 10–11 | varies |
| Heart | 918 | varies | 44% normal / 56% disease |

The `corral` dataset is loaded directly from PMLB (`option=1`, no local file needed).

---

## Models

### PegasosQSVC (QSVM)

The default model. Uses a `FidelityQuantumKernel` built on the SEQUENT feature map, optimised with the Pegasos SGD-SVM algorithm. Fast enough for metaheuristic search at `num_steps=100`.

### NeuralNetworkClassifier (QNN)

A parameterised quantum circuit composed of the SEQUENT feature map followed by a `TwoLocal` ansatz (Ry + Rz rotations, CX linear entanglement). Trained with COBYLA. Three execution variants:

| Mode | Backend | Notes |
|---|---|---|
| `statevector` | Default Qiskit sampler | No noise, fastest |
| `noise` | `AerSimulator` + IBM noise model | GPU-accelerated via tensor network method |
| `hardware` | Real IBM Quantum device | ALAP scheduling + XX dynamical decoupling |

---

## Metaheuristics

Both optimisers share the same interface and are fully interchangeable.

### Simulated Annealing (`metaheuristicas.py`)

At each temperature step, `num_neighbors` candidates are generated by flipping a random bit in the current solution. A worse candidate is accepted with probability exp(−Δcost / T), allowing escape from local optima early in the search.

### Genetic Algorithm (`metaheuristicas.py`)

A steady-state GA with:
- **Tournament selection** (pressure controlled by `tournament_size`)
- **Uniform crossover** — each bit independently chosen from either parent with p = 0.5, preserving useful patterns regardless of position
- **Bit-flip mutation** with default rate 1/chromosome_length (canonical for binary encoding)
- **Elitism** — top `elitism_count` individuals always survive
- **Memoisation** — quantum evaluations already computed are never repeated

Both optimisers minimise cost = −accuracy, so maximising accuracy is equivalent.

---

## Metrics and Output

Every experiment reports the following metrics, computed on the test set:

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

Macro averaging is used throughout because three of the five datasets are class-imbalanced (BreastW, Fitness, Heart). Per-class metrics are logged both to JSON and printed to console.

### Console output example

```
╔══════════════════════════════════════════════════════════════════════╗
║  SEQUENT  |  heart  |  QSVM  |  statevector  |  SA  |  FS:True     ║
╚══════════════════════════════════════════════════════════════════════╝

┌── Baseline: Classical RBF-SVM  (raw features, test set) ───────────┐
│  SVM test               Acc:0.8478       Prec:0.8401  Rec:0.8502  F1:0.8449
└────────────────────────────────────────────────────────────────────┘

┌── SEQUENT  —  Aggregated (5 runs, mean ± std) ─────────────────────┐
│  validation             Acc:0.8750 ±0.0120  Prec:0.8701  Rec:0.8790  F1:0.8745 ±0.0115
│  test                   Acc:0.8804 ±0.0098  Prec:0.8756  Rec:0.8841  F1:0.8798 ±0.0102
│  Per-class test metrics (mean over 5 runs):
│    class 0       f1:0.8611 ±0.0143
│    class 1       f1:0.8985 ±0.0071
│  complexity             depth:24  total_gates:67  2Q-gates:8  search_space:1024
└────────────────────────────────────────────────────────────────────┘
```

---

## Ablation Study

Setting `run_ablation=True` runs two additional variants that isolate the contribution of each component:

| Variant | FS | Search | Purpose |
|---|---|---|---|
| **FS-only** | ✓ | ✗ (all pairs entangled) | How much does feature reduction alone explain? |
| **Entanglement-only** | ✗ | ✓ (raw features) | How much does entanglement search alone explain? |
| **SEQUENT** | ✓ | ✓ | Full methodology |

---

## Reproducibility

All experiments use fixed seeds:

```python
random.seed(12345)
np.random.seed(12345)
algorithm_globals.random_seed = 12345
```

Multi-run experiments (`n_runs > 1`) offset the seed by run index (`12345 + run_idx`) to ensure each run is independent while remaining fully reproducible.

---

## Citation

```bibtex
@article{sequent2025,
  title   = {Selective Qubit Entanglement for Efficient Quantum Feature Map Construction},
  author  = {Rodríguez-Díaz, Francesc and Gutiérrez-Avilés, David and Martínez-Álvarez, Francisco},
  journal = {Quantum Machine Intelligence},
  year    = {2025},
  publisher = {Springer Nature}
}
```

---

## License

This project is licensed under the MIT License.
