"""
tools.py — SEQUENT toolkit.

Models  : PegasosQSVC (QSVM) and NeuralNetworkClassifier (QNN).
Modes   : statevector (ideal), noise emulator, real hardware.
Metrics : accuracy, macro precision/recall/F1, per-class breakdown (R2-10).

AUTOENCODER (semi-supervised)
──────────────────────────────
The _Autoencoder now uses a combined loss:

    Loss = MSE(reconstruction, x) + alpha * CrossEntropy(latent → y)

The classifier head injects supervised signal into the latent space,
forcing it to be discriminative for the target classes. This is critical
for small datasets (< 100 samples per split) where a purely unsupervised
AE tends to capture class-irrelevant variance.

  alpha = 0.0  →  pure unsupervised AE (original behaviour)
  alpha = 0.5  →  balanced reconstruction + classification (default)
  alpha = 1.0  →  classification-only latent space

PARALLELISM MODEL
─────────────────
This version is designed for sequential execution inside one SLURM task.
The Python driver does not spawn workers for the metaheuristic search; instead,
each quantum/classical fit can use the CPUs allocated to the job.

By default `_threads_per_worker()` uses:

  1. SEQUENT_THREADS_PER_WORKER, if set;
  2. SLURM_CPUS_PER_TASK, if set;
  3. OMP_NUM_THREADS, if set;
  4. os.cpu_count() as fallback.

To force a specific number of simulator threads, export:
    SEQUENT_THREADS_PER_WORKER=<N>
before launching the job.
"""

import os
import time
import random
from qiskit_aer.primitives import Sampler as AerSampler
import numpy as np
import pandas as pd

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
from qiskit.circuit.library import XGate

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import BackendSampler

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from pmlb import fetch_data

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("QISKIT_AER_CUQUANTUM", "1")
os.environ.setdefault("CUQUANTUM_MGPU", "1")

np.random.seed(12345)
random.seed(12345)
algorithm_globals.random_seed = 12345


# ═══════════════════════════════════════════════════════════════
#  THREAD-COUNT HELPER (sequential job: maximise resources per fit)
# ═══════════════════════════════════════════════════════════════

def _threads_per_worker():
    """
    Number of OpenMP/Aer threads used by each simulator instance.

    In this sequential version there is no Python-level worker fan-out inside
    the metaheuristic, so one fit may use the CPUs assigned to the SLURM task.

    Priority:
      1. SEQUENT_THREADS_PER_WORKER
      2. SLURM_CPUS_PER_TASK
      3. OMP_NUM_THREADS
      4. os.cpu_count()
    """
    for name in ("SEQUENT_THREADS_PER_WORKER", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        env = os.environ.get(name)
        if env:
            try:
                return max(1, int(env))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)
try:
    _torch_threads = _threads_per_worker()
    torch.set_num_threads(_torch_threads)
    torch.set_num_interop_threads(max(1, min(4, _torch_threads)))
except Exception:
    pass



#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def _preprocess_heart(df):
    df = df.drop(['RestingECG', 'ST_Slope', 'Age', 'Sex'], axis=1)
    df = pd.get_dummies(df, columns=['ChestPainType', 'ExerciseAngina'])
    if 'HeartDisease' in df.columns:
        df = df.rename(columns={'HeartDisease': 'target'})
    return df

def _preprocess_fitness(df):
    _TARGET_CANDIDATES = ['attended', 'status', 'label', 'class', 'target']
    for _col in _TARGET_CANDIDATES:
        if _col in df.columns and _col != 'target':
            df = df.rename(columns={_col: 'target'})
            break
    df = df.drop(['booking_id'], axis=1, errors='ignore')
    df['day_of_week'] = df['day_of_week'].map(
        {'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7})
    df['time']        = df['time'].map({'AM':0,'PM':1})
    df['category']    = df['category'].map({'Strength':1,'HIIT':2,'Cycling':3,'Aqua':4})
    df['days_before'] = df['days_before'].str.extract(r'(\d+)').astype(int)
    df['day_of_week'].fillna(df['day_of_week'].mode()[0], inplace=True)
    df['category'].fillna(df['category'].mode()[0], inplace=True)
    df['weight'].fillna(df['weight'].mean(), inplace=True)
    df['weight'] = df['weight'].round().astype(int)
    return df

def load_data(path=None, option=0, dataset=None):
    """
    option=0 → local CSV/TSV at `path`.
    option=1 → PMLB dataset by name.
    Returns X (DataFrame), y (Series).
    """
    if option == 1:
        df = fetch_data(dataset)
    else:
        if "breast" in path or "flare" in path:
            df = pd.read_csv(path, sep="\t")
        else:
            df = pd.read_csv(path)
        if "fitness" in path: df = _preprocess_fitness(df)
        if "heart"   in path: df = _preprocess_heart(df)

    if 'target' not in df.columns:
        _candidates = ['attended', 'status', 'label', 'class',
                       'Target', 'TARGET', 'y', 'outcome']
        _found = [c for c in _candidates if c in df.columns]
        if _found:
            df = df.rename(columns={_found[0]: 'target'})
        else:
            raise KeyError(
                f"No target column found. Available columns: {list(df.columns)}"
            )

    y = df['target']
    X = df.drop(['target'], axis=1).astype(int)
    return X, y


# ═══════════════════════════════════════════════════════════════
#  FEATURE SELECTION  (ANOVA, mutual_info, semi-supervised AE)
# ═══════════════════════════════════════════════════════════════

class _Autoencoder(nn.Module):
    """
    Semi-supervised autoencoder for feature compression.

    Architecture
    ────────────
    Encoder     : n_features → hidden_dim → latent_dim (k)
    Decoder     : k          → hidden_dim → n_features
    Classifier  : k          → n_classes   (auxiliary supervised head)

    Loss = MSE(reconstruction, x) + alpha * CrossEntropy(latent → y)

    The classifier head injects supervised signal directly into the latent
    space, forcing it to be discriminative for the target classes. This is
    critical for small datasets (< 100 samples) where an unsupervised AE
    tends to learn class-irrelevant variance.

    alpha = 0.0  →  pure unsupervised autoencoder (original behaviour)
    alpha = 0.5  →  balanced reconstruction + classification (recommended)
    alpha = 1.0  →  classification-only latent space

    The classifier head is discarded at inference — only encode() is used.

    Parameters
    ----------
    n_features  : int   — number of input features
    latent_dim  : int   — target latent dimensions (= k)
    hidden_dim  : int   — size of intermediate hidden layer
    n_classes   : int   — number of target classes
    alpha       : float — weight of the classification loss term
    """

    def __init__(self, n_features: int, latent_dim: int, hidden_dim: int,
                 n_classes: int = 2, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            # No activation: latent space must remain unconstrained
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        # Supervised head: latent → class logits
        # Lightweight (single linear layer) to avoid overfitting on small data.
        self.classifier_head = nn.Linear(latent_dim, n_classes)

    def forward(self, x, y=None):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        if y is not None and self.alpha > 0.0:
            logits = self.classifier_head(latent)
            return recon, logits
        return recon, None

    def encode(self, x):
        return self.encoder(x)


def _train_autoencoder(
    X_scaled: np.ndarray,
    latent_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
    seed: int,
    y_train: np.ndarray = None,
    alpha: float = 0.5,
) -> _Autoencoder:
    """
    Trains the semi-supervised autoencoder.

    Loss = MSE(reconstruction) + alpha * CrossEntropy(classifier_head(latent), y)

    When y_train is None or alpha=0, falls back to pure unsupervised MSE,
    preserving backward compatibility.

    Early stopping is applied on a 10% held-out validation split using the
    combined loss so both reconstruction and classification objectives are
    considered for convergence.

    Returns the trained model in eval() mode. The classifier head is retained
    in the returned model but is ignored at inference (encode() is used).
    """
    torch.manual_seed(seed)

    n, n_feat = X_scaled.shape
    val_size  = max(1, int(0.1 * n))
    rng       = np.random.default_rng(seed)
    idx       = rng.permutation(n)
    tr_idx, val_idx = idx[val_size:], idx[:val_size]

    X_tr  = torch.tensor(X_scaled[tr_idx],  dtype=torch.float32, device=device)
    X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32, device=device)

    # Supervised signal setup
    use_supervised = (y_train is not None and alpha > 0.0)
    n_classes = 2
    y_tr_t    = None
    y_val_t   = None

    if use_supervised:
        classes   = np.unique(y_train)
        n_classes = len(classes)
        label_map = {c: i for i, c in enumerate(classes)}
        y_mapped  = np.array([label_map[c] for c in y_train], dtype=np.int64)
        y_tr_t  = torch.tensor(y_mapped[tr_idx],  dtype=torch.long, device=device)
        y_val_t = torch.tensor(y_mapped[val_idx], dtype=torch.long, device=device)

    # Build DataLoader — include y only when supervised
    if use_supervised:
        loader = DataLoader(
            TensorDataset(X_tr, y_tr_t),
            batch_size=min(batch_size, len(tr_idx)),
            shuffle=True,
        )
    else:
        loader = DataLoader(
            TensorDataset(X_tr, X_tr),
            batch_size=min(batch_size, len(tr_idx)),
            shuffle=True,
        )

    model  = _Autoencoder(n_feat, latent_dim, hidden_dim,
                          n_classes=n_classes,
                          alpha=alpha if use_supervised else 0.0).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse_fn = nn.MSELoss()
    ce_fn  = nn.CrossEntropyLoss() if use_supervised else None

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    model.train()
    for epoch in range(1, epochs + 1):
        for batch in loader:
            xb = batch[0]
            yb = batch[1] if use_supervised else None
            opt.zero_grad()
            recon, logits = model(xb, yb)
            loss = mse_fn(recon, xb)
            if use_supervised and logits is not None:
                loss = loss + alpha * ce_fn(logits, yb)
            loss.backward()
            opt.step()

        # Validation loss (combined)
        model.eval()
        with torch.no_grad():
            recon_val, logits_val = model(X_val, y_val_t)
            val_loss = mse_fn(recon_val, X_val).item()
            if use_supervised and logits_val is not None:
                val_loss = val_loss + alpha * ce_fn(logits_val, y_val_t).item()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    → early stop at epoch {epoch}  "
                      f"(best val_loss={best_val_loss:.6f})")
                break

        if epoch % 50 == 0:
            print(f"    epoch {epoch:>4}  val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def fit_feature_selector(X, y=None, method="anova", k=5,
                         ae_epochs=200,
                         ae_batch_size=32,
                         ae_lr=1e-3,
                         ae_patience=20,
                         ae_hidden_factor=2,
                         ae_seed=12345,
                         ae_alpha=0.5):
    """
    Fit a feature-selection / representation-learning transform on a
    training set only, so validation and test data remain unseen.

    For method="autoencoder", ae_alpha controls the supervised loss weight:
      ae_alpha=0.0  →  pure unsupervised (original behaviour)
      ae_alpha=0.5  →  semi-supervised (recommended, default)
    """
    k = min(k, X.shape[1])

    if method in ("anova", "mutual_info"):
        if y is None:
            raise ValueError(f"method='{method}' requires y (target labels).")
        score_func = f_classif if method == "anova" else mutual_info_classif
        selector   = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        cols = X.columns[selector.get_support()]
        print(f"    → kept {k} features: {list(cols)}")
        return {
            "method":   method,
            "selector": selector,
            "cols":     pd.Index(cols),
        }

    if method == "autoencoder":
        n_features = X.shape[1]
        hidden_dim = max(ae_hidden_factor * k, n_features // 2, k + 1)

        print(f"    [AE] architecture: {n_features} → {hidden_dim} → {k} "
              f"→ {hidden_dim} → {n_features}")
        print(f"    [AE] epochs={ae_epochs}  batch={ae_batch_size}  "
              f"lr={ae_lr}  patience={ae_patience}  alpha={ae_alpha}")

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X.values.astype(np.float32))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    [AE] device: {device}  "
              f"({'semi-supervised' if ae_alpha > 0 and y is not None else 'unsupervised'})")

        model = _train_autoencoder(
            X_scaled   = X_scaled,
            latent_dim = k,
            hidden_dim = hidden_dim,
            epochs     = ae_epochs,
            batch_size = ae_batch_size,
            lr         = ae_lr,
            patience   = ae_patience,
            device     = device,
            seed       = ae_seed,
            y_train    = y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y),
            alpha      = ae_alpha,
        )
        cols = pd.Index([f"ae_{i}" for i in range(k)])
        return {
            "method": method,
            "scaler": scaler,
            "model":  model,
            "cols":   cols,
            "device": device,
        }

    raise ValueError(
        f"fs_method must be 'anova', 'mutual_info' or 'autoencoder'; got '{method}'"
    )


def transform_with_feature_selector(fitted_selector, X):
    """
    Apply a fitted feature selector / autoencoder transform to any split
    without re-fitting it.
    """
    method = fitted_selector["method"]
    cols   = fitted_selector["cols"]

    if method in ("anova", "mutual_info"):
        selector = fitted_selector["selector"]
        X_arr    = selector.transform(X)
        return pd.DataFrame(X_arr, columns=cols, index=X.index)

    if method == "autoencoder":
        scaler = fitted_selector["scaler"]
        model  = fitted_selector["model"]
        device = fitted_selector["device"]

        X_scaled = scaler.transform(X.values.astype(np.float32))
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            latent   = model.encode(X_tensor).cpu().numpy()
        return pd.DataFrame(latent, columns=cols, index=X.index)

    raise ValueError(f"Unsupported fitted selector method: {method}")


def fit_transform_feature_selection(X_train, X_other_list, y_train=None,
                                    method="anova", k=5,
                                    ae_epochs=200, ae_batch_size=32, ae_lr=1e-3,
                                    ae_patience=20, ae_hidden_factor=2,
                                    ae_seed=12345, ae_alpha=0.5):
    """
    Convenience helper: fit on X_train only, then transform X_train and each
    dataset in X_other_list with the same fitted transform.

    ae_alpha is forwarded to fit_feature_selector for the autoencoder method.
    """
    fitted    = fit_feature_selector(
        X_train, y=y_train, method=method, k=k,
        ae_epochs=ae_epochs, ae_batch_size=ae_batch_size, ae_lr=ae_lr,
        ae_patience=ae_patience, ae_hidden_factor=ae_hidden_factor,
        ae_seed=ae_seed, ae_alpha=ae_alpha,
    )
    X_train_t  = transform_with_feature_selector(fitted, X_train)
    X_others_t = [transform_with_feature_selector(fitted, Xo) for Xo in X_other_list]
    return fitted, X_train_t, X_others_t


def apply_feature_selection(X, y=None, method="anova", k=5,
                             ae_epochs=200, ae_batch_size=32, ae_lr=1e-3,
                             ae_patience=20, ae_hidden_factor=2,
                             ae_seed=12345, ae_alpha=0.5):
    """
    Backward-compatible convenience wrapper. Fits the transform on X and
    immediately applies it to X itself.
    """
    fitted    = fit_feature_selector(
        X, y=y, method=method, k=k,
        ae_epochs=ae_epochs, ae_batch_size=ae_batch_size, ae_lr=ae_lr,
        ae_patience=ae_patience, ae_hidden_factor=ae_hidden_factor,
        ae_seed=ae_seed, ae_alpha=ae_alpha,
    )
    X_reduced = transform_with_feature_selector(fitted, X)
    cols      = fitted["cols"]

    if method == "autoencoder":
        print(f"    [AE] latent shape: {X_reduced.shape}  "
              f"(columns: {list(cols)})")

    return X_reduced, cols


# ═══════════════════════════════════════════════════════════════
#  CORRELATION → ENTANGLEMENT MAP
# ═══════════════════════════════════════════════════════════════

def transformCorrelations(correlation_matrix):
    """Upper-triangular values of the correlation matrix as a flat array."""
    upper = np.triu(correlation_matrix.values, k=1)
    return upper[np.triu_indices_from(upper, k=1)]

def createCouples(triangular_array, columns):
    """Maps flat-array positions back to (col_i, col_j) pairs."""
    couples, n, idx = [], len(columns), 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(triangular_array):
                couples.append((columns[i], columns[j]))
            idx += 1
    return couples


# ═══════════════════════════════════════════════════════════════
#  FEATURE MAPS  (R1-02: linear, ring, full baselines)
# ═══════════════════════════════════════════════════════════════

def createFeatureMapLinear(num_features, reps=1):
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement="linear")

def createFeatureMapRing(num_features, reps=1):
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement="circular")

def createFeatureMapFull(num_features, reps=1):
    return ZZFeatureMap(feature_dimension=num_features, reps=reps, entanglement="full")

def createFeatureMap(couples, columns, reps=1):
    """
    SEQUENT selective-entanglement feature map.
    Entangles only the qubit pairs in `couples`; all qubits get a single-qubit
    phase gate regardless to ensure no qubit is left unencoded.
    """
    ent_list = [(columns.get_loc(p[0]), columns.get_loc(p[1])) for p in couples]
    ent_dict = {
        1: [(i,) for i in range(len(columns))],
        2: ent_list,
    }
    fm  = ZZFeatureMap(feature_dimension=len(columns), reps=reps, entanglement=ent_dict)
    dec = fm.decompose()

    num_qubits  = dec.num_qubits
    qubit_gates = {i: [] for i in range(num_qubits)}
    for instr, qargs, _ in dec.data:
        for qa in qargs:
            qubit_gates[qa._index].append(instr)

    x = ParameterVector('x', num_qubits)
    for i in range(num_qubits):
        if [g.name for g in qubit_gates[i]] == ['h']:
            dec.p(2.0 * x[i], i)
    return dec


def topology_pairs(columns, topology):
    """
    R2: pair list of a fixed topology over `columns`, in createFeatureMap format,
    so fixed and optimized entanglement go through the same circuit builder.
    """
    n = len(columns)
    if topology == "linear":
        idx = [(i, i + 1) for i in range(n - 1)]
    elif topology == "ring":
        idx = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if n > 2 else [])
    elif topology == "full":
        idx = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return [(columns[i], columns[j]) for i, j in idx]



# ═══════════════════════════════════════════════════════════════
#  SPLIT
# ═══════════════════════════════════════════════════════════════

def splitData(X, y, test_size=0.4, random_state=12345):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


# ═══════════════════════════════════════════════════════════════
#  METRICS  (R2-10: per-class precision/recall/F1)
# ═══════════════════════════════════════════════════════════════

def compute_metrics(model, X, y_true):
    t0      = time.time()
    y_pred  = model.predict(X)
    elapsed = time.time() - t0
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(   y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(       y_true, y_pred, average="macro", zero_division=0)),
        "inference_time":  float(elapsed),
    }


def _subset_rows(X, idx):
    """Return rows by integer positions for NumPy arrays or pandas objects."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    return X[idx]


def _add_train_metrics_if_requested(
    metrics,
    model,
    train_features,
    train_labels,
    return_train_metrics=False,
    train_metric_sample_size=None,
    train_metric_seed=12345,
):
    """
    Optionally add train_accuracy and train_f1_macro to a metrics dict.

    The beta overfitting penalty only needs an estimate of the train-side
    metric. When train_metric_sample_size is set, the estimate is computed on
    a deterministic subsample to avoid a full extra quantum prediction for
    every candidate mask.
    """
    if not return_train_metrics:
        return metrics

    n = len(train_labels)
    if train_metric_sample_size is not None and train_metric_sample_size < n:
        rng = np.random.default_rng(train_metric_seed)
        idx = rng.choice(n, size=int(train_metric_sample_size), replace=False)
        X_eval = _subset_rows(train_features, idx)
        y_eval = _subset_rows(train_labels, idx)
    else:
        X_eval = train_features
        y_eval = train_labels

    train_m = compute_metrics(model, X_eval, y_eval)
    metrics["train_accuracy"] = train_m["accuracy"]
    metrics["train_f1_macro"] = train_m["f1_macro"]
    return metrics

def compute_metrics_per_class(model, X, y_true):
    y_pred  = model.predict(X)
    report  = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = sorted([k for k in report if k not in ("accuracy","macro avg","weighted avg")])
    return {cls: {
        "precision": float(report[cls]["precision"]),
        "recall":    float(report[cls]["recall"]),
        "f1":        float(report[cls]["f1-score"]),
        "support":   int(report[cls]["support"]),
    } for cls in classes}


# ═══════════════════════════════════════════════════════════════
#  CIRCUIT COMPLEXITY  (R1-08)
# ═══════════════════════════════════════════════════════════════

def circuit_complexity(feature_map):
    dec   = feature_map.decompose()
    two_q = sum(1 for _, qargs, _ in dec.data if len(qargs) == 2)
    n     = dec.num_qubits
    return {
        "n_qubits":        n,
        "depth":           dec.depth(),
        "total_gates":     dec.size(),
        "two_qubit_gates": two_q,
        "search_space":    2 ** (n * (n - 1) // 2),
    }


# ═══════════════════════════════════════════════════════════════
#  QNN BUILDING BLOCKS
# ═══════════════════════════════════════════════════════════════

def parity(x):
    return bin(x).count("1") % 2

def build_ansatz(n_qubits, reps=1):
    ansatz        = TwoLocal(n_qubits, ["ry","rz"], "cx", entanglement="linear", reps=reps)
    ordered_params = sorted(ansatz.parameters, key=lambda p: p.name)
    return ansatz, ordered_params, n_qubits

def create_qnn_circuit(num_qubits, feature_map, ansatz):
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.barrier()
    qc.compose(ansatz, inplace=True)
    in_params = sorted(feature_map.parameters, key=lambda p: p.name)
    wt_params = sorted(ansatz.parameters,      key=lambda p: p.name)
    return qc, in_params, wt_params

def _make_qnn_classifier(qnn, wt_params):
    return NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=30, tol=1e-4),
        one_hot=False,
        initial_point=np.random.uniform(0, 2 * np.pi, len(wt_params)),
    )


# ═══════════════════════════════════════════════════════════════
#  QSVM — THREE EXECUTION MODES
# ═══════════════════════════════════════════════════════════════

_NOISE_MODELS = {}


def get_noise_model(backend_name):
    """
    R2: noise model fetched once per process. If SEQUENT_NOISE_MODEL_FILE points to
    a pickled NoiseModel snapshot it is loaded instead, so every job shares one
    calibration (NoiseModel.from_dict does not round-trip Kraus errors in Aer 0.17).
    """
    if backend_name not in _NOISE_MODELS:
        path = os.environ.get("SEQUENT_NOISE_MODEL_FILE", "").strip()
        if path and os.path.exists(path):
            import pickle
            with open(path, "rb") as f:
                _NOISE_MODELS[backend_name] = pickle.load(f)
            print(f"  [noise] frozen noise model loaded from {path}", flush=True)
        else:
            _NOISE_MODELS[backend_name] = NoiseModel.from_backend(
                QiskitRuntimeService().backend(backend_name))
    return _NOISE_MODELS[backend_name]



_QSVM_C         = 100
_QSVM_NUM_STEPS = 50
_QSVM_SEED      = 12345
_QNN_COBYLA_SEARCH = 30
_QNN_COBYLA_FINAL  = 50

def evaluate_qsvm_statevector(feature_map, train_features, train_labels,
                               val_features, val_labels,
                               return_train_metrics=False,
                               train_metric_sample_size=None,
                               train_metric_seed=12345,
                               **kwargs):
    _n_threads = _threads_per_worker()
    _aer = AerSimulator(
        method="statevector",
        precision="single",
        max_parallel_threads=_n_threads,
        max_parallel_experiments=_n_threads,
        max_parallel_shots=1,
        fusion_enable=True,
        fusion_max_qubit=5,
        fusion_threshold=1,
    )
    _sampler  = BackendSampler(backend=_aer)
    _fidelity = ComputeUncompute(sampler=_sampler)
    qkernel   = FidelityQuantumKernel(feature_map=feature_map, fidelity=_fidelity)
    clf       = PegasosQSVC(quantum_kernel=qkernel, C=_QSVM_C,
                             num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(
        metrics, clf, train_features, train_labels,
        return_train_metrics=return_train_metrics,
        train_metric_sample_size=train_metric_sample_size,
        train_metric_seed=train_metric_seed,
    )
    metrics["training_time"] = float(train_t)
    metrics["fit_time"] = float(train_t)
    return metrics, clf


# ─── Method-selection thresholds ─────────────────────────────────────────────
# Below _MPS_MIN_QUBITS, statevector is dramatically faster than MPS for the
# noisy simulator. MPS only pays off for many qubits with limited entanglement.

def _build_sim_kw(circuit, noise_model, gpu, n_threads):
    n_2q      = circuit.num_nonlocal_gates()
    n_qubits  = circuit.num_qubits
    has_noise = noise_model is not None

    # Density matrix only fits comfortably up to ~12 qubits (256 MB at 12q).
    # Above that, fall back to statevector trajectories.
    if has_noise and n_qubits <= 12:
        method = "density_matrix"
    else:
        method = "statevector"

    # GPU launch overhead dominates below ~14 qubits — keep it on CPU.
    use_gpu = gpu is not None and n_qubits >= 14

    print(f"  [sim] qubits={n_qubits}  2Q-gates={n_2q}  "
          f"method={method}  device={'GPU' if use_gpu else 'CPU'}  "
          f"threads={n_threads}", flush=True)

    kw = dict(
        method=method,
        noise_model=noise_model,
        precision="single",
        max_parallel_threads=n_threads,
        fusion_enable=True,
        fusion_threshold=1,
        fusion_max_qubit=5,
    )

    if use_gpu:
        kw["device"] = "GPU"
    else:
        kw["max_parallel_experiments"] = 1
        kw["max_parallel_shots"] = 1

    # shot_branching_* only applies to statevector trajectory mode
    if method == "statevector" and has_noise:
        kw["shot_branching_enable"] = True
        kw["shot_branching_sampling_enable"] = True

    return kw


def _build_sim_kw_fallback(noise_model, n_threads):
    # Fallback to statevector CPU in case density_matrix fails for any reason
    # (custom gates not supported as Kraus, memory pressure on larger circuits…)
    print(f"  [sim] fallback → statevector CPU  threads={n_threads}", flush=True)
    return dict(
        method="statevector",
        noise_model=noise_model,
        precision="single",
        max_parallel_threads=n_threads,
        max_parallel_experiments=1,
        max_parallel_shots=1,
        shot_branching_enable=True,
        shot_branching_sampling_enable=True,
        fusion_enable=True,
        fusion_threshold=1,
        fusion_max_qubit=5,
    )

def evaluate_qsvm_noise_sim(feature_map, train_features, train_labels,
                             val_features, val_labels,
                             backend_name="ibm_boston", gpu=None,
                             return_train_metrics=False,
                             train_metric_sample_size=None,
                             train_metric_seed=12345,
                             **kwargs):
    _n_threads = _threads_per_worker()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    noise_model  = get_noise_model(backend_name)
    sim_kw       = _build_sim_kw(feature_map.decompose(), noise_model, gpu, _n_threads)

    def _make_clf(sim_kw_local):
        _aer = AerSimulator(**sim_kw_local)
        _samp = BackendSampler(backend=_aer, options={"shots": 50})
        _fid = ComputeUncompute(sampler=_samp)
        _qk = FidelityQuantumKernel(feature_map=feature_map, fidelity=_fid)
        return PegasosQSVC(quantum_kernel=_qk, C=_QSVM_C,
                           num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)

    clf = _make_clf(sim_kw)
    t0  = time.time()
    try:
        clf.fit(train_features, train_labels)
    except Exception as e:
        print(f"  [sim] primary method failed ({type(e).__name__}: "
              f"{str(e)[:100]}) → fallback statevector CPU", flush=True)
        sim_kw2 = _build_sim_kw_fallback(noise_model, _n_threads)
        clf     = _make_clf(sim_kw2)
        clf.fit(train_features, train_labels)
    train_t = time.time() - t0

    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(
        metrics, clf, train_features, train_labels,
        return_train_metrics=return_train_metrics,
        train_metric_sample_size=train_metric_sample_size,
        train_metric_seed=train_metric_seed,
    )
    metrics["training_time"] = float(train_t)
    metrics["fit_time"] = float(train_t)
    return metrics, clf

def evaluate_qsvm_hardware(feature_map, train_features, train_labels,
                            val_features, val_labels,
                            backend_name="ibm_boston", shots=10,
                            return_train_metrics=False,
                            train_metric_sample_size=None,
                            train_metric_seed=12345,
                            **kwargs):
    backend = QiskitRuntimeService().backend(backend_name)
    pm      = generate_preset_pass_manager(backend=backend, optimization_level=2)
    pm.scheduling = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
    ])
    sampler  = SamplerV2(backend)
    sampler.options.default_shots = shots
    fidelity = ComputeUncompute(sampler, pass_manager=pm)
    qkernel  = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    clf      = PegasosQSVC(quantum_kernel=qkernel, C=_QSVM_C,
                           num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(
        metrics, clf, train_features, train_labels,
        return_train_metrics=return_train_metrics,
        train_metric_sample_size=train_metric_sample_size,
        train_metric_seed=train_metric_seed,
    )
    metrics["training_time"] = float(train_t)
    metrics["fit_time"] = float(train_t)
    return metrics, clf


# ═══════════════════════════════════════════════════════════════
#  QNN — THREE EXECUTION MODES
# ═══════════════════════════════════════════════════════════════

def evaluate_qnn_statevector(feature_map, train_features, train_labels,
                              val_features, val_labels, reps_ansatz=1,
                              fast_eval=False,
                              return_train_metrics=False,
                              train_metric_sample_size=None,
                              train_metric_seed=12345,
                              **kwargs):
    _n_threads  = _threads_per_worker()
    _cobyla_max = _QNN_COBYLA_SEARCH if fast_eval else _QNN_COBYLA_FINAL
    _aer = AerSimulator(
        method="statevector",
        precision="single",
        max_parallel_threads=_n_threads,
        max_parallel_experiments=_n_threads,
        max_parallel_shots=1,
        fusion_enable=True,
        fusion_max_qubit=5,
        fusion_threshold=1,
    )
    _sampler = BackendSampler(backend=_aer, options={"shots": 50})
    num_q    = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    qnn = SamplerQNN(circuit=qc, input_params=in_params, weight_params=wt_params,
                     interpret=parity, output_shape=2, sampler=_sampler)
    clf = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=_cobyla_max, tol=1e-4),
        one_hot=False,
        initial_point=np.random.uniform(0, 2 * np.pi, len(wt_params)),
    )
    t0  = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(
        metrics, clf, train_features, train_labels,
        return_train_metrics=return_train_metrics,
        train_metric_sample_size=train_metric_sample_size,
        train_metric_seed=train_metric_seed,
    )
    metrics["training_time"] = float(train_t)
    metrics["fit_time"] = float(train_t)
    return metrics, clf

def evaluate_qnn_noise_sim(feature_map, train_features, train_labels,
                            val_features, val_labels,
                            backend_name="ibm_boston", reps_ansatz=1,
                            gpu=None,
                            return_train_metrics=False,
                            train_metric_sample_size=None,
                            train_metric_seed=12345,
                            **kwargs):
    _n_threads   = _threads_per_worker()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    noise_model  = get_noise_model(backend_name)
    num_q        = train_features.shape[1]
    ansatz, _, wt_params     = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    sim_kw       = _build_sim_kw(qc.decompose(), noise_model, gpu, _n_threads)

    def _make_qnn_clf(skw):
        # AerSampler (qiskit_aer.primitives) — admite shots=None para obtener
        # P(x) exacta desde la diagonal de ρ. BackendSampler no soporta esto.
        _samp = AerSampler(
            backend_options=skw,
            run_options={"shots": None},   # ← exacto, sin shot noise
        )
        _qnn  = SamplerQNN(circuit=qc, input_params=in_params,
                           weight_params=wt_params, sampler=_samp,
                           interpret=parity, output_shape=2)
        return _make_qnn_classifier(_qnn, wt_params)

    clf = _make_qnn_clf(sim_kw)
    t0  = time.time()
    try:
        clf.fit(train_features, train_labels)
    except Exception as e:
        print(f"  [sim] primary method failed ({type(e).__name__}: "
              f"{str(e)[:100]}) → fallback statevector CPU", flush=True)
        clf = _make_qnn_clf(_build_sim_kw_fallback(noise_model, _n_threads))
        clf.fit(train_features, train_labels)
    train_t = time.time() - t0

    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(
        metrics, clf, train_features, train_labels,
        return_train_metrics=return_train_metrics,
        train_metric_sample_size=train_metric_sample_size,
        train_metric_seed=train_metric_seed,
    )
    metrics["training_time"] = float(train_t)
    metrics["fit_time"] = float(train_t)
    return metrics, clf

def evaluate_qnn_hardware(feature_map, train_features, train_labels,
                           val_features, val_labels,
                           backend_name="ibm_boston", shots=10,
                           job_tag="SEQUENT_QNN", reps_ansatz=1,
                           return_train_metrics=False,
                           train_metric_sample_size=None,
                           train_metric_seed=12345,
                           **kwargs):
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    pm      = generate_preset_pass_manager(backend=backend, optimization_level=2)
    pm.scheduling = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
    ])
    sampler = SamplerV2(backend)
    sampler.options.default_shots = shots
    sampler.options.environment.job_tags = [job_tag]
    num_q   = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    qc.measure_all()
    qc_t = pm.run(qc)
    qnn  = SamplerQNN(circuit=qc_t, input_params=in_params, weight_params=wt_params,
                      sampler=sampler, interpret=parity, output_shape=2)
    clf  = _make_qnn_classifier(qnn, wt_params)
    t0   = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(
        metrics, clf, train_features, train_labels,
        return_train_metrics=return_train_metrics,
        train_metric_sample_size=train_metric_sample_size,
        train_metric_seed=train_metric_seed,
    )
    metrics["training_time"] = float(train_t)
    metrics["fit_time"] = float(train_t)
    return metrics, clf


# ═══════════════════════════════════════════════════════════════
#  CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════════

def evaluate_classical_svm(train_features, train_labels,
                            test_features, test_labels, kernel="rbf"):
    clf = SVC(kernel=kernel, C=1000, random_state=12345)
    t0  = time.time()
    clf.fit(train_features, train_labels)
    train_t = time.time() - t0
    metrics = compute_metrics(clf, test_features, test_labels)
    metrics["training_time"] = float(train_t)
    metrics["fit_time"] = float(train_t)
    return metrics, clf


class _MLPClassifierTorch(nn.Module):
    def __init__(self, n_in: int, n_classes: int,
                 hidden1: int = 64, hidden2: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class _SklearnWrapperMLP:
    def __init__(self, model, classes, device, scaler):
        self.model_   = model
        self.classes_ = classes
        self._device  = device
        self._scaler  = scaler

    def predict(self, X):
        X_s    = self._scaler.transform(X)
        tensor = torch.tensor(X_s, dtype=torch.float32, device=self._device)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(tensor)
            preds  = logits.argmax(dim=1).cpu().numpy()
        return np.array([self.classes_[p] for p in preds])


def evaluate_classical_mlp(train_features, train_labels,
                            test_features, test_labels,
                            hidden1=64, hidden2=32, dropout=0.3,
                            epochs=200, batch_size=32, lr=1e-3,
                            patience=20, seed=12345):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.unique(train_labels)
    n_in    = train_features.shape[1]
    n_cls   = len(classes)
    label_map = {c: i for i, c in enumerate(classes)}

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(train_features.astype(np.float32))
    X_te_s  = scaler.transform(test_features.astype(np.float32))

    n_val   = max(1, int(0.1 * len(X_tr_s)))
    idx     = np.random.permutation(len(X_tr_s))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _to_tensors(X, y_raw):
        y_mapped = np.array([label_map[c] for c in y_raw], dtype=np.int64)
        return (torch.tensor(X,        dtype=torch.float32, device=device),
                torch.tensor(y_mapped, dtype=torch.long,    device=device))

    X_t, y_t = _to_tensors(X_tr_s[tr_idx],  train_labels[tr_idx])
    X_v, y_v = _to_tensors(X_tr_s[val_idx], train_labels[val_idx])

    loader = DataLoader(TensorDataset(X_t, y_t),
                        batch_size=min(batch_size, len(tr_idx)),
                        shuffle=True)

    model   = _MLPClassifierTorch(n_in, n_cls, hidden1, hidden2, dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state    = None
    wait          = 0

    t0 = time.time()
    model.train()
    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_v), y_v).item()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            wait          = 0
        else:
            wait += 1
            if wait >= patience:
                break

    train_t = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    wrapper = _SklearnWrapperMLP(model, classes, device, scaler)
    metrics = compute_metrics(wrapper, test_features, test_labels)
    metrics["training_time"] = float(train_t)
    return metrics, wrapper