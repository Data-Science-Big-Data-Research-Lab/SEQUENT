"""
tools.py — SEQUENT toolkit.

Models  : PegasosQSVC (QSVM) and NeuralNetworkClassifier (QNN).
Modes   : statevector (ideal), noise simulator, real hardware.
Metrics : accuracy, macro precision/recall/F1, per-class breakdown.

Feature selection:
  - ANOVA / mutual_info  (sklearn SelectKBest)
  - Semi-supervised autoencoder (PyTorch):
      Loss = MSE(reconstruction) + alpha * CrossEntropy(latent → y)
      alpha=0.0 → pure unsupervised | alpha=0.5 → balanced (default)

Hardware QSVM:
  Uses _HardwareKernel, which builds a concrete ISA fidelity circuit and
  batches the entire kernel matrix in a single SamplerV2 PUB, avoiding
  QPY parameter UUID binding errors.

Thread count (sequential SLURM jobs):
  Priority: SEQUENT_THREADS_PER_WORKER > SLURM_CPUS_PER_TASK >
            OMP_NUM_THREADS > os.cpu_count()
"""

import os
import re
import time
import random

import numpy as np
import pandas as pd

from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, XGate
from qiskit.transpiler import generate_preset_pass_manager, PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.primitives import BackendSampler
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Sampler as AerSampler

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
try:
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
except ImportError:
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
#  THREAD-COUNT HELPER
# ═══════════════════════════════════════════════════════════════

def _threads_per_worker():
    for name in ("SEQUENT_THREADS_PER_WORKER", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        env = os.environ.get(name)
        if env:
            try:
                return max(1, int(env))
            except ValueError:
                pass
    return max(1, os.cpu_count() or 1)


try:
    _t = _threads_per_worker()
    torch.set_num_threads(_t)
    torch.set_num_interop_threads(max(1, min(4, _t)))
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════

def _preprocess_heart(df):
    df = df.drop(['RestingECG', 'ST_Slope', 'Age', 'Sex'], axis=1)
    df = pd.get_dummies(df, columns=['ChestPainType', 'ExerciseAngina'])
    if 'HeartDisease' in df.columns:
        df = df.rename(columns={'HeartDisease': 'target'})
    return df


def _preprocess_fitness(df):
    for col in ['attended', 'status', 'label', 'class']:
        if col in df.columns:
            df = df.rename(columns={col: 'target'})
            break
    df = df.drop(['booking_id'], axis=1, errors='ignore')
    df['day_of_week'] = df['day_of_week'].map({'Mon':1,'Tue':2,'Wed':3,'Thu':4,'Fri':5,'Sat':6,'Sun':7})
    df['time'] = df['time'].map({'AM':0,'PM':1})
    df['category'] = df['category'].map({'Strength':1,'HIIT':2,'Cycling':3,'Aqua':4})
    df['days_before'] = df['days_before'].str.extract(r'(\d+)').astype(int)
    for col in ['day_of_week', 'category']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['weight'].fillna(df['weight'].mean(), inplace=True)
    df['weight'] = df['weight'].round().astype(int)
    return df


def load_data(path=None, option=0, dataset=None):
    """
    Load dataset. option=0 → local CSV/TSV at path; option=1 → PMLB by name.
    Returns X (DataFrame), y (Series).
    """
    if option == 1:
        df = fetch_data(dataset)
    else:
        sep = "\t" if ("breast" in path or "flare" in path) else ","
        df = pd.read_csv(path, sep=sep)
        if "fitness" in path:
            df = _preprocess_fitness(df)
        if "heart" in path:
            df = _preprocess_heart(df)

    if 'target' not in df.columns:
        candidates = ['attended', 'status', 'label', 'class', 'Target', 'TARGET', 'y', 'outcome']
        found = [c for c in candidates if c in df.columns]
        if found:
            df = df.rename(columns={found[0]: 'target'})
        else:
            raise KeyError(f"No target column found. Available: {list(df.columns)}")

    y = df['target']
    X = df.drop(['target'], axis=1).astype(int)
    return X, y


# ═══════════════════════════════════════════════════════════════
#  FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════

class _Autoencoder(nn.Module):
    """
    Semi-supervised autoencoder.
    Loss = MSE(reconstruction) + alpha * CrossEntropy(latent → y)
    alpha=0.0 → unsupervised | alpha=0.5 → balanced (default) | alpha=1.0 → classification only
    """
    def __init__(self, n_features, latent_dim, hidden_dim, n_classes=2, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        self.classifier_head = nn.Linear(latent_dim, n_classes)

    def forward(self, x, y=None):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        if y is not None and self.alpha > 0.0:
            return recon, self.classifier_head(latent)
        return recon, None

    def encode(self, x):
        return self.encoder(x)


def _train_autoencoder(X_scaled, latent_dim, hidden_dim, epochs, batch_size,
                       lr, patience, device, seed, y_train=None, alpha=0.5):
    torch.manual_seed(seed)
    n, n_feat = X_scaled.shape
    val_size = max(1, int(0.1 * n))
    idx = np.random.default_rng(seed).permutation(n)
    tr_idx, val_idx = idx[val_size:], idx[:val_size]

    X_tr = torch.tensor(X_scaled[tr_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(X_scaled[val_idx], dtype=torch.float32, device=device)

    use_supervised = y_train is not None and alpha > 0.0
    n_classes, y_tr_t, y_val_t = 2, None, None

    if use_supervised:
        classes = np.unique(y_train)
        n_classes = len(classes)
        label_map = {c: i for i, c in enumerate(classes)}
        y_mapped = np.array([label_map[c] for c in y_train], dtype=np.int64)
        y_tr_t = torch.tensor(y_mapped[tr_idx], dtype=torch.long, device=device)
        y_val_t = torch.tensor(y_mapped[val_idx], dtype=torch.long, device=device)

    loader = (
        DataLoader(TensorDataset(X_tr, y_tr_t), batch_size=min(batch_size, len(tr_idx)), shuffle=True)
        if use_supervised else
        DataLoader(TensorDataset(X_tr, X_tr), batch_size=min(batch_size, len(tr_idx)), shuffle=True)
    )

    model = _Autoencoder(n_feat, latent_dim, hidden_dim, n_classes=n_classes,
                         alpha=alpha if use_supervised else 0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse_fn = nn.MSELoss()
    ce_fn = nn.CrossEntropyLoss() if use_supervised else None

    best_val_loss, best_state, wait = float("inf"), None, 0
    model.train()
    for epoch in range(1, epochs + 1):
        for batch in loader:
            xb, yb = batch[0], batch[1] if use_supervised else None
            opt.zero_grad()
            recon, logits = model(xb, yb)
            loss = mse_fn(recon, xb)
            if use_supervised and logits is not None:
                loss = loss + alpha * ce_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            recon_val, logits_val = model(X_val, y_val_t)
            val_loss = mse_fn(recon_val, X_val).item()
            if use_supervised and logits_val is not None:
                val_loss += alpha * ce_fn(logits_val, y_val_t).item()
        model.train()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    → early stop at epoch {epoch}  (best val_loss={best_val_loss:.6f})")
                break

        if epoch % 50 == 0:
            print(f"    epoch {epoch:>4}  val_loss={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def fit_feature_selector(X, y=None, method="anova", k=5,
                         ae_epochs=200, ae_batch_size=32, ae_lr=1e-3,
                         ae_patience=20, ae_hidden_factor=2,
                         ae_seed=12345, ae_alpha=0.5):
    """Fit a feature selector on X_train only."""
    k = min(k, X.shape[1])

    if method in ("anova", "mutual_info"):
        if y is None:
            raise ValueError(f"method='{method}' requires y.")
        score_func = f_classif if method == "anova" else mutual_info_classif
        selector = SelectKBest(score_func=score_func, k=k).fit(X, y)
        cols = X.columns[selector.get_support()]
        print(f"    → kept {k} features: {list(cols)}")
        return {"method": method, "selector": selector, "cols": pd.Index(cols)}

    if method == "autoencoder":
        n_features = X.shape[1]
        hidden_dim = max(ae_hidden_factor * k, n_features // 2, k + 1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    [AE] {n_features} → {hidden_dim} → {k}  device={device}  alpha={ae_alpha}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values.astype(np.float32))
        model = _train_autoencoder(X_scaled, k, hidden_dim, ae_epochs, ae_batch_size,
                                   ae_lr, ae_patience, device, ae_seed,
                                   y_train=y.to_numpy() if hasattr(y, 'to_numpy') else np.asarray(y),
                                   alpha=ae_alpha)
        cols = pd.Index([f"ae_{i}" for i in range(k)])
        return {"method": method, "scaler": scaler, "model": model, "cols": cols, "device": device}

    raise ValueError(f"method must be 'anova', 'mutual_info' or 'autoencoder'; got '{method}'")


def transform_with_feature_selector(fitted_selector, X):
    """Apply a fitted selector without re-fitting."""
    method = fitted_selector["method"]
    cols = fitted_selector["cols"]

    if method in ("anova", "mutual_info"):
        X_arr = fitted_selector["selector"].transform(X)
        return pd.DataFrame(X_arr, columns=cols, index=X.index)

    if method == "autoencoder":
        scaler, model, device = fitted_selector["scaler"], fitted_selector["model"], fitted_selector["device"]
        X_scaled = scaler.transform(X.values.astype(np.float32))
        with torch.no_grad():
            latent = model.encode(torch.tensor(X_scaled, dtype=torch.float32, device=device)).cpu().numpy()
        return pd.DataFrame(latent, columns=cols, index=X.index)

    raise ValueError(f"Unsupported method: {method}")


def fit_transform_feature_selection(X_train, X_other_list, y_train=None,
                                    method="anova", k=5,
                                    ae_epochs=200, ae_batch_size=32, ae_lr=1e-3,
                                    ae_patience=20, ae_hidden_factor=2,
                                    ae_seed=12345, ae_alpha=0.5):
    """Fit on X_train, transform X_train and all datasets in X_other_list."""
    fitted = fit_feature_selector(X_train, y=y_train, method=method, k=k,
                                  ae_epochs=ae_epochs, ae_batch_size=ae_batch_size,
                                  ae_lr=ae_lr, ae_patience=ae_patience,
                                  ae_hidden_factor=ae_hidden_factor,
                                  ae_seed=ae_seed, ae_alpha=ae_alpha)
    X_train_t = transform_with_feature_selector(fitted, X_train)
    X_others_t = [transform_with_feature_selector(fitted, Xo) for Xo in X_other_list]
    return fitted, X_train_t, X_others_t


# ═══════════════════════════════════════════════════════════════
#  CORRELATION → ENTANGLEMENT MAP
# ═══════════════════════════════════════════════════════════════

def transformCorrelations(correlation_matrix):
    """Upper-triangular values of the correlation matrix as a flat array."""
    upper = np.triu(correlation_matrix.values, k=1)
    return upper[np.triu_indices_from(upper, k=1)]


def createCouples(triangular_array, columns):
    """Map flat-array positions back to (col_i, col_j) feature pairs."""
    couples, n, idx = [], len(columns), 0
    for i in range(n):
        for j in range(i + 1, n):
            if idx < len(triangular_array):
                couples.append((columns[i], columns[j]))
            idx += 1
    return couples


# ═══════════════════════════════════════════════════════════════
#  FEATURE MAPS
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
    Entangles only the qubit pairs in `couples`; all qubits receive a
    single-qubit phase gate to ensure no qubit is left unencoded.
    Uses numeric parameter index sorting to handle n >= 11 features correctly.
    """
    ent_list = [(columns.get_loc(p[0]), columns.get_loc(p[1])) for p in couples]
    ent_dict = {1: [(i,) for i in range(len(columns))], 2: ent_list}
    fm = ZZFeatureMap(feature_dimension=len(columns), reps=reps, entanglement=ent_dict)
    dec = fm.decompose()

    num_qubits = dec.num_qubits
    qubit_gates = {i: [] for i in range(num_qubits)}
    for instr, qargs, _ in dec.data:
        for qa in qargs:
            qubit_gates[qa._index].append(instr)

    def _param_idx(p):
        m = re.match(r'.*\[(\d+)\]$', p.name)
        return int(m.group(1)) if m else 0

    existing_params = sorted(fm.parameters, key=_param_idx)
    param_by_qubit = {i: existing_params[i] for i in range(num_qubits)}

    for i in range(num_qubits):
        if [g.name for g in qubit_gates[i]] == ["h"]:
            dec.p(2.0 * param_by_qubit[i], i)

    return dec


# ═══════════════════════════════════════════════════════════════
#  DATA SPLIT
# ═══════════════════════════════════════════════════════════════

def splitData(X, y, test_size=0.3, random_state=12345):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ═══════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════

def compute_metrics(model, X, y_true):
    t0 = time.time()
    y_pred = model.predict(X)
    elapsed = time.time() - t0
    return {
        "accuracy":        float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":        float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "inference_time":  float(elapsed),
    }


def compute_metrics_per_class(model, X, y_true):
    y_pred = model.predict(X)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes = sorted(k for k in report if k not in ("accuracy", "macro avg", "weighted avg"))
    return {
        cls: {
            "precision": float(report[cls]["precision"]),
            "recall":    float(report[cls]["recall"]),
            "f1":        float(report[cls]["f1-score"]),
            "support":   int(report[cls]["support"]),
        }
        for cls in classes
    }


def circuit_complexity(feature_map):
    dec = feature_map.decompose()
    two_q = sum(1 for _, qargs, _ in dec.data if len(qargs) == 2)
    n = dec.num_qubits
    return {
        "n_qubits":        n,
        "depth":           dec.depth(),
        "total_gates":     dec.size(),
        "two_qubit_gates": two_q,
        "search_space":    2 ** (n * (n - 1) // 2),
    }


def _add_train_metrics_if_requested(metrics, model, train_features, train_labels,
                                    return_train_metrics=False,
                                    train_metric_sample_size=None,
                                    train_metric_seed=12345):
    if not return_train_metrics:
        return metrics
    n = len(train_labels)
    if train_metric_sample_size is not None and train_metric_sample_size < n:
        idx = np.random.default_rng(train_metric_seed).choice(n, size=int(train_metric_sample_size), replace=False)
        X_eval = train_features[idx] if not hasattr(train_features, "iloc") else train_features.iloc[idx]
        y_eval = train_labels[idx] if not hasattr(train_labels, "iloc") else train_labels.iloc[idx]
    else:
        X_eval, y_eval = train_features, train_labels
    m = compute_metrics(model, X_eval, y_eval)
    metrics["train_accuracy"] = m["accuracy"]
    metrics["train_f1_macro"] = m["f1_macro"]
    return metrics


# ═══════════════════════════════════════════════════════════════
#  QNN HELPERS
# ═══════════════════════════════════════════════════════════════

def parity(x):
    return bin(x).count("1") % 2


def build_ansatz(n_qubits, reps=1):
    ansatz = TwoLocal(n_qubits, ["ry", "rz"], "cx", entanglement="linear", reps=reps)
    ordered = sorted(ansatz.parameters, key=lambda p: p.name)
    return ansatz, ordered, n_qubits


def create_qnn_circuit(num_qubits, feature_map, ansatz):
    qc = QuantumCircuit(num_qubits)
    qc.compose(feature_map, inplace=True)
    qc.barrier()
    qc.compose(ansatz, inplace=True)
    in_params = sorted(feature_map.parameters, key=lambda p: p.name)
    wt_params = sorted(ansatz.parameters, key=lambda p: p.name)
    return qc, in_params, wt_params


def _make_qnn_classifier(qnn, wt_params, maxiter=100):
    return NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=COBYLA(maxiter=maxiter, tol=1e-4),
        one_hot=False,
        initial_point=np.random.uniform(0, 2 * np.pi, len(wt_params)),
    )


# ═══════════════════════════════════════════════════════════════
#  SIMULATOR HELPERS
# ═══════════════════════════════════════════════════════════════

_QSVM_C = 1000
_QSVM_NUM_STEPS = 500
_QSVM_SEED = 12345
_QNN_COBYLA_SEARCH = 30
_QNN_COBYLA_FINAL = 100
_GPU_MIN_QUBITS = 9
_MPS_MIN_QUBITS = 13


def _build_sim_kw(circuit, noise_model, gpu, n_threads):
    n_qubits = circuit.num_qubits
    use_gpu = gpu is not None and n_qubits >= _GPU_MIN_QUBITS
    if use_gpu:
        print(f"  [sim] qubits={n_qubits} method=matrix_product_state device=GPU")
        return dict(method="matrix_product_state", noise_model=noise_model,
                    precision="single", device="GPU",
                    fusion_enable=True, fusion_threshold=1, fusion_max_qubit=5)
    if n_qubits < _MPS_MIN_QUBITS:
        print(f"  [sim] qubits={n_qubits} method=statevector device=CPU threads={n_threads}")
        return dict(method="statevector", noise_model=noise_model, precision="single",
                    max_parallel_threads=n_threads, max_parallel_experiments=1,
                    max_parallel_shots=1, shot_branching_enable=True,
                    shot_branching_sampling_enable=True,
                    fusion_enable=True, fusion_threshold=1, fusion_max_qubit=5)
    print(f"  [sim] qubits={n_qubits} method=matrix_product_state device=CPU threads={n_threads}")
    return dict(method="matrix_product_state", noise_model=noise_model, precision="single",
                max_parallel_threads=n_threads, max_parallel_experiments=1,
                max_parallel_shots=1,
                fusion_enable=True, fusion_threshold=1, fusion_max_qubit=5)


def _build_sim_kw_fallback(noise_model, n_threads):
    print(f"  [sim] fallback → statevector CPU threads={n_threads}")
    return dict(method="statevector", noise_model=noise_model, precision="single",
                max_parallel_threads=n_threads, max_parallel_experiments=1,
                max_parallel_shots=1, shot_branching_enable=True,
                shot_branching_sampling_enable=True,
                fusion_enable=True, fusion_threshold=1, fusion_max_qubit=5)


# ═══════════════════════════════════════════════════════════════
#  QSVM EVALUATORS
# ═══════════════════════════════════════════════════════════════

def evaluate_qsvm_statevector(feature_map, train_features, train_labels,
                               val_features, val_labels,
                               return_train_metrics=False,
                               train_metric_sample_size=None,
                               train_metric_seed=12345, **kwargs):
    n_threads = _threads_per_worker()
    aer = AerSimulator(method="statevector", precision="single",
                       max_parallel_threads=n_threads, max_parallel_experiments=n_threads,
                       max_parallel_shots=1,
                       fusion_enable=True, fusion_max_qubit=5, fusion_threshold=1)
    sampler = BackendSampler(backend=aer)
    fidelity = ComputeUncompute(sampler=sampler)
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)
    clf = PegasosQSVC(quantum_kernel=qkernel, C=_QSVM_C,
                      num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(metrics, clf, train_features, train_labels,
                                              return_train_metrics, train_metric_sample_size, train_metric_seed)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, clf


def evaluate_qsvm_noise_sim(feature_map, train_features, train_labels,
                             val_features, val_labels,
                             backend_name="ibm_pittsburgh", gpu=None,
                             return_train_metrics=False,
                             train_metric_sample_size=None,
                             train_metric_seed=12345, **kwargs):
    n_threads = _threads_per_worker()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    noise_model = NoiseModel.from_backend(QiskitRuntimeService().backend(backend_name))
    sim_kw = _build_sim_kw(feature_map.decompose(), noise_model, gpu, n_threads)

    def _make_clf(skw):
        aer = AerSimulator(**skw)
        samp = BackendSampler(backend=aer, options={"shots": 50})
        fid = ComputeUncompute(sampler=samp)
        qk = FidelityQuantumKernel(feature_map=feature_map, fidelity=fid)
        return PegasosQSVC(quantum_kernel=qk, C=_QSVM_C,
                           num_steps=_QSVM_NUM_STEPS, seed=_QSVM_SEED)

    clf = _make_clf(sim_kw)
    t0 = time.time()
    try:
        clf.fit(train_features, train_labels)
    except Exception as e:
        print(f"  [sim] primary failed ({type(e).__name__}: {str(e)[:80]}) → fallback CPU")
        clf = _make_clf(_build_sim_kw_fallback(noise_model, n_threads))
        clf.fit(train_features, train_labels)
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(metrics, clf, train_features, train_labels,
                                              return_train_metrics, train_metric_sample_size, train_metric_seed)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, clf


# ═══════════════════════════════════════════════════════════════
#  HARDWARE QSVM KERNEL
# ═══════════════════════════════════════════════════════════════

class _HardwareKernel:
    """
    Quantum kernel for real hardware.
    Implements K(x, y) = |<0| U†(y) U(x) |0>|² = P(measure |0...0>).
    Batches the entire kernel matrix in a single SamplerV2 PUB.
    """

    def __init__(self, feature_map, backend, shots, pm,
                 runtime_session=None, job_tag=None):
        self._n_qubits = feature_map.num_qubits
        self._zero_state = '0' * self._n_qubits

        mode = runtime_session if runtime_session is not None else backend
        self._sampler = SamplerV2(mode=mode)
        self._sampler.options.default_shots = shots
        if job_tag is not None:
            try:
                self._sampler.options.environment.job_tags = [job_tag]
            except Exception:
                pass

        self._isa_circuit, self._x_indices, self._y_indices = \
            self._build_isa_template(feature_map, pm)

    def _build_isa_template(self, feature_map, pm):
        def _feat_idx(p):
            m = re.match(r'.*\[(\d+)\]$', p.name)
            return int(m.group(1)) if m else 0

        orig_params = sorted(feature_map.parameters, key=_feat_idx)
        n = len(orig_params)
        xa, xb = ParameterVector('xA', n), ParameterVector('xB', n)

        fm_xa = feature_map.assign_parameters(dict(zip(orig_params, list(xa))), inplace=False)
        fm_xb = feature_map.assign_parameters(dict(zip(orig_params, list(xb))), inplace=False)

        qc = QuantumCircuit(self._n_qubits)
        qc.compose(fm_xa, inplace=True)
        qc.compose(fm_xb.inverse(), inplace=True)
        qc.measure_all()
        isa_qc = pm.run(qc)

        name_to_pos = {p.name: i for i, p in enumerate(isa_qc.parameters)}
        x_indices = [name_to_pos[xa[m].name] for m in range(n)]
        y_indices = [name_to_pos[xb[m].name] for m in range(n)]
        return isa_qc, x_indices, y_indices

    def evaluate_matrix(self, X1, X2=None):
        X1 = np.asarray(X1, dtype=float)
        symmetric = X2 is None
        X2 = X1 if symmetric else np.asarray(X2, dtype=float)
        n1, n2 = len(X1), len(X2)

        pairs = [(i, j) for i in range(n1) for j in range(n2)
                 if not (symmetric and j <= i)]
        K = np.zeros((n1, n2), dtype=float)

        if pairs:
            n_total = len(self._isa_circuit.parameters)
            param_values = np.zeros((len(pairs), n_total), dtype=float)
            for k, (i, j) in enumerate(pairs):
                param_values[k, self._x_indices] = X1[i]
                param_values[k, self._y_indices] = X2[j]

            pub_result = self._sampler.run([(self._isa_circuit, param_values)]).result()[0]
            bit_array = pub_result.data.meas

            for k, (i, j) in enumerate(pairs):
                counts = bit_array.get_counts() if bit_array.ndim == 0 else bit_array.get_counts(loc=k)
                total = sum(counts.values())
                p0 = counts.get(self._zero_state, 0) / total if total else 0.0
                K[i, j] = p0
                if symmetric:
                    K[j, i] = p0

        if symmetric:
            np.fill_diagonal(K, 1.0)
        return K


class _HardwareQSVC:
    def __init__(self, hardware_kernel, C=1000):
        self._hk = hardware_kernel
        self._svc = None
        self._X_train = None

    def fit(self, X, y):
        self._X_train = np.asarray(X, dtype=float)
        K_train = self._hk.evaluate_matrix(self._X_train)
        self._svc = SVC(kernel="precomputed", C=self._hk._C if hasattr(self._hk, '_C') else 1000)
        self._svc.fit(K_train, y)
        return self

    def predict(self, X):
        return self._svc.predict(self._hk.evaluate_matrix(np.asarray(X, dtype=float), self._X_train))


def evaluate_qsvm_hardware(feature_map, train_features, train_labels,
                            val_features, val_labels,
                            backend_name="ibm_pittsburgh", shots=128,
                            runtime_session=None, job_tag="SEQUENT_QSVM",
                            return_train_metrics=False,
                            train_metric_sample_size=None,
                            train_metric_seed=12345, **kwargs):
    backend = QiskitRuntimeService().backend(backend_name)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    pm.scheduling = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
    ])
    hk = _HardwareKernel(feature_map, backend, shots, pm,
                         runtime_session=runtime_session, job_tag=job_tag)
    clf = _HardwareQSVC(hk, C=_QSVM_C)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(metrics, clf, train_features, train_labels,
                                              return_train_metrics, train_metric_sample_size, train_metric_seed)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, clf


# ═══════════════════════════════════════════════════════════════
#  QNN EVALUATORS
# ═══════════════════════════════════════════════════════════════

def evaluate_qnn_statevector(feature_map, train_features, train_labels,
                              val_features, val_labels,
                              reps_ansatz=1, fast_eval=False,
                              return_train_metrics=False,
                              train_metric_sample_size=None,
                              train_metric_seed=12345, **kwargs):
    n_threads = _threads_per_worker()
    cobyla_max = _QNN_COBYLA_SEARCH if fast_eval else _QNN_COBYLA_FINAL
    aer = AerSimulator(method="statevector", precision="single",
                       max_parallel_threads=n_threads, max_parallel_experiments=n_threads,
                       max_parallel_shots=1, fusion_enable=True,
                       fusion_max_qubit=5, fusion_threshold=1)
    sampler = BackendSampler(backend=aer, options={"shots": 50})
    num_q = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    qnn = SamplerQNN(circuit=qc, input_params=in_params, weight_params=wt_params,
                     interpret=parity, output_shape=2, sampler=sampler)
    clf = _make_qnn_classifier(qnn, wt_params, maxiter=cobyla_max)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(metrics, clf, train_features, train_labels,
                                              return_train_metrics, train_metric_sample_size, train_metric_seed)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, clf


def evaluate_qnn_noise_sim(feature_map, train_features, train_labels,
                            val_features, val_labels,
                            backend_name="ibm_pittsburgh", reps_ansatz=1, gpu=None,
                            return_train_metrics=False,
                            train_metric_sample_size=None,
                            train_metric_seed=12345, **kwargs):
    n_threads = _threads_per_worker()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    noise_model = NoiseModel.from_backend(QiskitRuntimeService().backend(backend_name))
    num_q = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    sim_kw = _build_sim_kw(qc.decompose(), noise_model, gpu, n_threads)

    def _make_qnn_clf(skw):
        samp = BackendSampler(backend=AerSimulator(**skw))
        qnn = SamplerQNN(circuit=qc, input_params=in_params, weight_params=wt_params,
                         sampler=samp, interpret=parity, output_shape=2)
        return _make_qnn_classifier(qnn, wt_params)

    clf = _make_qnn_clf(sim_kw)
    t0 = time.time()
    try:
        clf.fit(train_features, train_labels)
    except Exception as e:
        print(f"  [sim] primary failed ({type(e).__name__}: {str(e)[:80]}) → fallback CPU")
        clf = _make_qnn_clf(_build_sim_kw_fallback(noise_model, n_threads))
        clf.fit(train_features, train_labels)
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(metrics, clf, train_features, train_labels,
                                              return_train_metrics, train_metric_sample_size, train_metric_seed)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, clf


def evaluate_qnn_hardware(feature_map, train_features, train_labels,
                           val_features, val_labels,
                           backend_name="ibm_pittsburgh", shots=128,
                           runtime_session=None, job_tag="SEQUENT_QNN",
                           reps_ansatz=1, qnn_hardware_maxiter=5,
                           return_train_metrics=False,
                           train_metric_sample_size=None,
                           train_metric_seed=12345, **kwargs):
    backend = QiskitRuntimeService().backend(backend_name)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=2)
    pm.scheduling = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
    ])
    mode = runtime_session if runtime_session is not None else backend
    sampler = SamplerV2(mode=mode)
    sampler.options.default_shots = shots
    if job_tag is not None:
        try:
            sampler.options.environment.job_tags = [job_tag]
        except Exception:
            pass
    num_q = train_features.shape[1]
    ansatz, _, wt_params = build_ansatz(num_q, reps=reps_ansatz)
    qc, in_params, wt_params = create_qnn_circuit(num_q, feature_map, ansatz)
    qnn = SamplerQNN(circuit=qc, input_params=in_params, weight_params=wt_params,
                     sampler=sampler, pass_manager=pm, interpret=parity, output_shape=2)
    clf = _make_qnn_classifier(qnn, wt_params, maxiter=qnn_hardware_maxiter)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    metrics = compute_metrics(clf, val_features, val_labels)
    metrics = _add_train_metrics_if_requested(metrics, clf, train_features, train_labels,
                                              return_train_metrics, train_metric_sample_size, train_metric_seed)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, clf


# ═══════════════════════════════════════════════════════════════
#  CLASSICAL BASELINES
# ═══════════════════════════════════════════════════════════════

def evaluate_classical_svm(train_features, train_labels,
                            test_features, test_labels, kernel="rbf"):
    clf = SVC(kernel=kernel, C=1000, random_state=12345)
    t0 = time.time()
    clf.fit(train_features, train_labels)
    metrics = compute_metrics(clf, test_features, test_labels)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, clf


class _MLPClassifierTorch(nn.Module):
    def __init__(self, n_in, n_classes, hidden1=64, hidden2=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden1), nn.BatchNorm1d(hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.BatchNorm1d(hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class _SklearnWrapperMLP:
    def __init__(self, model, classes, device, scaler):
        self.model_ = model
        self.classes_ = classes
        self._device = device
        self._scaler = scaler

    def predict(self, X):
        X_s = self._scaler.transform(X)
        tensor = torch.tensor(X_s, dtype=torch.float32, device=self._device)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(tensor).argmax(dim=1).cpu().numpy()
        return np.array([self.classes_[p] for p in preds])


def evaluate_classical_mlp(train_features, train_labels,
                            test_features, test_labels,
                            hidden1=64, hidden2=32, dropout=0.3,
                            epochs=200, batch_size=32, lr=1e-3,
                            patience=20, seed=12345):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.unique(train_labels)
    n_in, n_cls = train_features.shape[1], len(classes)
    label_map = {c: i for i, c in enumerate(classes)}

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(train_features.astype(np.float32))
    n_val = max(1, int(0.1 * len(X_tr_s)))
    idx = np.random.permutation(len(X_tr_s))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    def _to_tensors(X, y_raw):
        y_mapped = np.array([label_map[c] for c in y_raw], dtype=np.int64)
        return (torch.tensor(X, dtype=torch.float32, device=device),
                torch.tensor(y_mapped, dtype=torch.long, device=device))

    X_t, y_t = _to_tensors(X_tr_s[tr_idx], train_labels[tr_idx])
    X_v, y_v = _to_tensors(X_tr_s[val_idx], train_labels[val_idx])
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=min(batch_size, len(tr_idx)), shuffle=True)

    model = _MLPClassifierTorch(n_in, n_cls, hidden1, hidden2, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss, best_state, wait = float("inf"), None, 0

    t0 = time.time()
    model.train()
    for _ in range(1, epochs + 1):
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
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    wrapper = _SklearnWrapperMLP(model, classes, device, scaler)
    metrics = compute_metrics(wrapper, test_features, test_labels)
    metrics["training_time"] = float(time.time() - t0)
    return metrics, wrapper
