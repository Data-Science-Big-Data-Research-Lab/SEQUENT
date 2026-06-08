"""
main.py — SEQUENT Experiment Runner
====================================
Runs the SEQUENT feature-map search experiment for one or more dataset/model
configurations, fully sequentially (no multiprocessing, no thread fans).

Configuration is at the bottom under "EXPERIMENT GRID". Adjust datasets,
models, metaheuristics, feature-selection methods, and execution mode there.

Execution mode:
  "statevector"  — ideal AerSimulator, fast, for local testing
  "noise"        — noise-model AerSimulator (requires IBM account)
  "hardware"     — real IBM Quantum backend (requires IBM account + credits)

IBM credentials:
  Set IBM_QUANTUM_TOKEN and IBM_QUANTUM_INSTANCE as environment variables,
  or hard-code them in the CREDENTIALS section below.
"""

import os
import datetime
import json
import random
from collections import Counter
from contextlib import nullcontext

import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import QiskitRuntimeService, Session

import tools as t
from metaheuristicas import (
    simulated_annealing,
    tabu_search,
    iterated_local_search,
    genetic_algorithm,
)

# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS  (prefer environment variables over hard-coding)
# ─────────────────────────────────────────────────────────────────────────────
_IBM_TOKEN    = os.environ.get("IBM_QUANTUM_TOKEN", "")
_IBM_INSTANCE = os.environ.get("IBM_QUANTUM_INSTANCE", "")

if _IBM_TOKEN:
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=_IBM_TOKEN,
        instance=_IBM_INSTANCE,
        overwrite=True,
        set_as_default=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SEED
# ─────────────────────────────────────────────────────────────────────────────
BASE_SEED = 12345
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
algorithm_globals.random_seed = BASE_SEED

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH TABLES
# ─────────────────────────────────────────────────────────────────────────────
_EVALUATE = {
    "qsvm": {
        "statevector": t.evaluate_qsvm_statevector,
        "noise":       t.evaluate_qsvm_noise_sim,
        "hardware":    t.evaluate_qsvm_hardware,
    },
    "qnn": {
        "statevector": t.evaluate_qnn_statevector,
        "noise":       t.evaluate_qnn_noise_sim,
        "hardware":    t.evaluate_qnn_hardware,
    },
}

_OPTIMISER = {
    "sa":  simulated_annealing,
    "ts":  tabu_search,
    "ils": iterated_local_search,
    "ga":  genetic_algorithm,
}

_FM_FACTORY = {
    "linear": t.createFeatureMapLinear,
    "ring":   t.createFeatureMapRing,
    "full":   t.createFeatureMapFull,
}

os.makedirs("results", exist_ok=True)
_CSV_PATH = os.path.join("results", "benchmark_results.csv")


# ═════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _header(text, width=70):
    bar = "=" * width
    print(f"\n+{bar}+")
    for chunk in [text[i:i + width] for i in range(0, len(text), width)]:
        print(f"|  {chunk:<{width - 2}}|")
    print(f"+{bar}+")


def _section(title):
    print(f"\n--- {title} {'-' * max(0, 65 - len(title))}")


def _section_end():
    print("-" * 72)


def _row(label, metrics):
    acc  = metrics.get("accuracy", metrics.get("accuracy_mean", 0))
    prec = metrics.get("precision_macro", metrics.get("precision_macro_mean", 0))
    rec  = metrics.get("recall_macro", metrics.get("recall_macro_mean", 0))
    f1   = metrics.get("f1_macro", metrics.get("f1_macro_mean", 0))
    tt   = metrics.get("training_time", metrics.get("training_time_mean", 0))
    it   = metrics.get("inference_time", metrics.get("inference_time_mean", 0))

    acc_str = f"{acc:.4f}"
    if (s := metrics.get("accuracy_std")) is not None:
        acc_str += f" +/-{s:.4f}"
    if (c := metrics.get("accuracy_ci95")) is not None:
        acc_str += f" [95%CI +/-{c:.4f}]"

    f1_str = f"{f1:.4f}"
    if (s := metrics.get("f1_macro_std")) is not None:
        f1_str += f" +/-{s:.4f}"

    print(f"|  {label:<22} Acc:{acc_str:<30} Prec:{prec:.4f}  Rec:{rec:.4f}  F1:{f1_str}")
    if tt > 0:
        print(f"|  {'':22} Train:{tt:.2f}s  Infer:{f'{it:.3f}s' if it > 0 else '-'}")


def _circuit_row(label, complexity):
    print(
        f"|  {label:<22} depth:{complexity.get('depth', 0)}  "
        f"total_gates:{complexity.get('total_gates', 0)}  "
        f"2Q-gates:{complexity.get('two_qubit_gates', 0)}  "
        f"search_space:{complexity.get('search_space', '-')}"
    )


def _class_rows(per_class):
    for cls, m in per_class.items():
        print(f"|    class {cls:<6}  prec:{m['precision']:.4f}  rec:{m['recall']:.4f}  "
              f"f1:{m['f1']:.4f}  n={m['support']}")


# ═════════════════════════════════════════════════════════════════════════════
# STATISTICS HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _aggregate(results_list):
    if not results_list:
        return {}
    out = {}
    for k in results_list[0]:
        vals = [r[k] for r in results_list
                if isinstance(r.get(k), (int, float)) and not np.isnan(r[k])]
        if vals:
            n = len(vals)
            mu = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n)
            ci95 = float(sp_stats.t.ppf(0.975, df=max(n - 1, 1))) * sem
            out[f"{k}_mean"] = mu
            out[f"{k}_std"]  = std
            out[f"{k}_ci95"] = ci95
    return out


def _significance_tests(a_list, b_list):
    """One-sided Wilcoxon signed-rank test: H1 = SEQUENT (a) > baseline (b)."""
    n = len(a_list)
    res = {"n_runs": n}
    if n < 2:
        res["note"] = "Need n_runs >= 2"
        return res

    a = np.array(a_list, dtype=float)
    b = np.array(b_list, dtype=float)
    diffs = a - b

    # Wilcoxon signed-rank test (one-sided: H1 = a > b)
    try:
        wins = int(np.sum(diffs > 0))
        losses = int(np.sum(diffs < 0))
        ties = int(np.sum(diffs == 0))
        stat_w, p_w_two = sp_stats.wilcoxon(a, b, alternative="two-sided")
        # Convert two-sided p to one-sided (greater)
        p_w = float(p_w_two / 2) if np.sum(diffs) > 0 else 1.0
        res.update(
            wilcoxon_stat=float(stat_w),
            wilcoxon_p=p_w,
            wilcoxon_sig_05=p_w < 0.05,
            wilcoxon_wins=wins,
            wilcoxon_losses=losses,
            wilcoxon_ties=ties,
        )
    except Exception as e:
        res["wilcoxon_error"] = str(e)

    return res


def _print_significance(sig, label):
    _section(f"Statistical Significance ({label})")
    if "note" in sig:
        print(f"|  {sig['note']}")
        _section_end()
        return
    print(f"|  n={sig['n_runs']}  (one-sided Wilcoxon, H1: SEQUENT > baseline)")
    if "wilcoxon_p" in sig:
        print(f"|  Wilcoxon:    W={sig['wilcoxon_stat']:.4f}  p={sig['wilcoxon_p']:.4f}  "
              f"{'*** p<0.05' if sig['wilcoxon_sig_05'] else 'n.s.'}  "
              f"(wins={sig.get('wilcoxon_wins', '?')}, losses={sig.get('wilcoxon_losses', '?')}, "
              f"ties={sig.get('wilcoxon_ties', 0)})")
    if "wilcoxon_error" in sig:
        print(f"|  Wilcoxon error: {sig['wilcoxon_error']}")
    _section_end()


def _directional_comparison(a_list, b_list):
    if not a_list or not b_list:
        return {}
    a, b = np.array(a_list, dtype=float), np.array(b_list, dtype=float)
    diffs = a - b
    return {
        "wins": int(np.sum(diffs > 0)),
        "losses": int(np.sum(diffs < 0)),
        "ties": int(np.sum(diffs == 0)),
        "mean_diff": float(np.mean(diffs)),
        "sequent_acc_mean": float(np.mean(a)),
        "baseline_acc_mean": float(np.mean(b)),
    }


def _build_warm_start(scores, activation_ratio=0.30):
    if scores is None:
        return None
    arr = np.nan_to_num(np.abs(np.asarray(scores, dtype=float)),
                        nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    if arr.size == 0 or np.all(np.isneginf(arr)):
        warm = [0] * len(arr)
        if warm:
            warm[0] = 1
        return warm
    n_active = max(1, min(len(arr), int(round(len(arr) * activation_ratio))))
    warm = [0] * len(arr)
    for idx in np.argsort(arr)[::-1][:n_active]:
        warm[int(idx)] = 1
    return warm


# ═════════════════════════════════════════════════════════════════════════════
# OBJECTIVE FUNCTION
# ═════════════════════════════════════════════════════════════════════════════

class _Objective:
    """Callable objective: evaluates a binary entanglement mask on val set."""

    MIN_ACTIVE = 3
    MAX_ACTIVE = 5
    PENALTY_STEP = 0.01

    def __init__(self, couples_array, columns, tr, y_tr, val, y_val,
                 evaluate_fn, reps, cache, objective_metric="accuracy"):
        self.couples_array = couples_array
        self.columns = columns
        self.tr, self.y_tr = tr, y_tr
        self.val, self.y_val = val, y_val
        self.evaluate_fn = evaluate_fn
        self.reps = reps
        self.cache = cache
        self.objective_metric = objective_metric

    def _penalty(self, vector):
        n = int(np.sum(vector))
        if n < self.MIN_ACTIVE:
            return self.PENALTY_STEP * (self.MIN_ACTIVE - n)
        if n > self.MAX_ACTIVE:
            return self.PENALTY_STEP * (n - self.MAX_ACTIVE)
        return 0.0

    def __call__(self, vector):
        key = str(vector)
        if key in self.cache:
            return self.cache[key]["cost"], self.cache[key]["aux"]

        selected = [self.couples_array[i] for i, bit in enumerate(vector) if bit == 1]
        fm = t.createFeatureMap(selected, self.columns, reps=self.reps)
        metrics, model = self.evaluate_fn(fm, self.tr, self.y_tr, self.val, self.y_val)

        val_metric = metrics.get(self.objective_metric, metrics["accuracy"])
        penalty = self._penalty(vector)
        cost = -(val_metric - penalty)

        self.cache[key] = {
            "solution": list(vector), "cost": float(cost),
            "aux": metrics.get("training_time", 0.0),
            "metrics": metrics, "model": model, "feature_map": fm,
            "active_pairs": int(np.sum(vector)), "mask_penalty": float(penalty),
        }
        return cost, self.cache[key]["aux"]


# ═════════════════════════════════════════════════════════════════════════════
# DATA SPLIT & FEATURE-SELECTION HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _make_splits(X, y, seed):
    X_outer, X_test, y_outer, y_test = t.splitData(X, y, test_size=0.3, random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_outer, y_outer, test_size=0.3, random_state=seed, stratify=y_outer)
    return dict(outer_train_X=X_outer, outer_train_y=y_outer,
                search_train_X=X_tr, search_train_y=y_tr,
                search_val_X=X_val, search_val_y=y_val,
                test_X=X_test, test_y=y_test)


def _prepare_fs_views(splits, use_fs, fs_method, k, seed):
    X_str, y_str = splits["search_train_X"], splits["search_train_y"]

    if use_fs:
        _, X_str_fs, others = t.fit_transform_feature_selection(
            X_str,
            [splits["search_val_X"], splits["outer_train_X"], splits["test_X"]],
            y_train=y_str, method=fs_method, k=k, ae_seed=seed,
        )
        X_val_fs, X_outer_fs, X_test_fs = others
    else:
        X_str_fs = X_str.copy()
        X_val_fs = splits["search_val_X"].copy()
        X_outer_fs = splits["outer_train_X"].copy()
        X_test_fs = splits["test_X"].copy()

    fs_cols = X_str_fs.columns
    fs_scores = t.transformCorrelations(X_str_fs.corr())
    pairs_fs = t.createCouples(fs_scores, fs_cols)
    warm_start = _build_warm_start(fs_scores)

    return dict(
        fs_cols=fs_cols, pairs_fs=pairs_fs, warm_start=warm_start,
        search_train_np=X_str_fs.to_numpy(),
        search_val_np=X_val_fs.to_numpy(),
        outer_train_np=X_outer_fs.to_numpy(),
        test_np=X_test_fs.to_numpy(),
    )


# ═════════════════════════════════════════════════════════════════════════════
# CSV SAVE HELPER
# ═════════════════════════════════════════════════════════════════════════════

def _append_csv(row):
    df = pd.DataFrame([row])
    if os.path.exists(_CSV_PATH):
        df.to_csv(_CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(_CSV_PATH, mode="w", header=True, index=False)


def _save_json(data, path):
    def _clean(obj):
        if isinstance(obj, dict):    return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [_clean(v) for v in obj]
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        return obj
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)


def _already_done(dataset, model_type, mode, metaheuristic, fs_method, reps):
    if not os.path.exists(_CSV_PATH):
        return False
    try:
        df = pd.read_csv(_CSV_PATH)
        mask = (
            (df["dataset"] == dataset) & (df["model_type"] == model_type) &
            (df["mode"] == mode) & (df["metaheuristic"] == metaheuristic) &
            (df["fs_method"] == fs_method) & (df["reps"] == reps)
        )
        return bool(mask.any())
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMISER DISPATCH
# ═════════════════════════════════════════════════════════════════════════════

def _run_optimiser(metaheuristic, obj, n_pairs, cfg, seed, warm_start):
    mh = cfg["metaheuristic"]
    if mh == "sa":
        return simulated_annealing(
            obj, n_pairs,
            initial_temp=cfg["sa_initial_temp"],
            cooling_rate=cfg["sa_cooling_rate"],
            stopping_temp=cfg["sa_stopping_temp"],
            max_iterations=cfg["sa_max_iterations"],
            num_neighbors=cfg["sa_num_neighbors"],
            seed=seed,
        )
    if mh == "ts":
        return tabu_search(
            obj, n_pairs,
            max_iterations=cfg["ts_max_iterations"],
            tabu_tenure=cfg["ts_tabu_tenure"],
            max_no_improve=cfg["ts_max_no_improve"],
            neighborhood_sample_size=cfg["ts_neighborhood_sample_size"],
            restart_fraction=cfg["ts_restart_fraction"],
            seed=seed,
        )
    if mh == "ils":
        return iterated_local_search(
            obj, n_pairs,
            n_restarts=cfg["ils_n_restarts"],
            perturbation_strength=cfg["ils_perturbation_strength"],
            local_search_iters=cfg["ils_local_search_iters"],
            local_search_initial_temp=cfg["ils_local_search_initial_temp"],
            local_search_cooling_rate=cfg["ils_local_search_cooling_rate"],
            local_search_stopping_temp=cfg["ils_local_search_stopping_temp"],
            local_search_num_neighbors=cfg["ils_local_search_num_neighbors"],
            warm_start=warm_start if cfg.get("ils_use_warm_start") else None,
            repair=cfg.get("ils_repair", True),
            seed=seed,
        )
    if mh == "ga":
        return genetic_algorithm(
            obj, n_pairs,
            population_size=cfg["ga_population_size"],
            n_generations=cfg["ga_n_generations"],
            crossover_rate=cfg["ga_crossover_rate"],
            mutation_rate=cfg["ga_mutation_rate"],
            tournament_size=cfg["ga_tournament_size"],
            elitism_count=cfg["ga_elitism_count"],
            seed=seed,
        )
    raise ValueError(f"Unknown metaheuristic: {mh}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(dataset, option, path, cfg):
    """
    Run one SEQUENT experiment end-to-end and save results to CSV + JSON.

    Parameters
    ----------
    dataset : str
    option  : int  (0=local file, 1=PMLB)
    path    : str | None
    cfg     : dict  (see EXPERIMENT GRID section for keys)
    """
    mode           = cfg["mode"]
    model_type     = cfg["model_type"]
    metaheuristic  = cfg["metaheuristic"]
    fs_method      = cfg["fs_method"]
    k              = cfg["k"]
    reps           = cfg["reps"]
    n_runs         = cfg["n_runs"]
    use_fs         = cfg.get("use_fs", True)
    run_baselines  = cfg.get("run_baselines", True)
    objective_metric = cfg.get("objective_metric", "accuracy")
    backend_name   = cfg.get("backend_name", "ibm_pittsburgh")
    hardware_shots = cfg.get("hardware_shots", 128)

    if model_type not in _EVALUATE:
        raise ValueError(f"model_type must be 'qsvm' or 'qnn'; got '{model_type}'")
    if mode not in _EVALUATE[model_type]:
        raise ValueError(f"mode must be 'statevector'|'noise'|'hardware'; got '{mode}'")
    if metaheuristic not in _OPTIMISER:
        raise ValueError(f"metaheuristic must be sa/ts/ils/ga; got '{metaheuristic}'")

    # Build evaluate function (wraps hardware/noise extras)
    if mode == "hardware":
        from functools import partial
        evaluate_fn = partial(_EVALUATE[model_type][mode],
                              shots=hardware_shots, backend_name=backend_name)
    else:
        evaluate_fn = _EVALUATE[model_type][mode]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{dataset}_{model_type}_{mode}_{metaheuristic}_{fs_method}_reps{reps}_{timestamp}"

    _header(f"SEQUENT | {dataset} | {model_type.upper()} | {mode} | "
            f"{metaheuristic.upper()} | FS:{use_fs}({fs_method},k={k}) | reps={reps} | runs={n_runs}")

    X_orig, y_orig = t.load_data(path=path, option=option, dataset=dataset)
    print(f"\n  Dataset: {X_orig.shape[0]} samples x {X_orig.shape[1]} features")
    print(f"  Class distribution: {dict(sorted(Counter(y_orig).items()))}")

    # ── Classical baselines (split 0 only, for reference) ────────────────────
    splits0 = _make_splits(X_orig, y_orig, BASE_SEED)
    tr0_np  = splits0["outer_train_X"].to_numpy()
    y_tr0   = splits0["outer_train_y"].to_numpy()
    te0_np  = splits0["test_X"].to_numpy()
    y_te0   = splits0["test_y"].to_numpy()

    _section("Classical SVM baseline (split 0)")
    svm_m0, _ = t.evaluate_classical_svm(tr0_np, y_tr0, te0_np, y_te0)
    _row("SVM test", svm_m0)
    _section_end()

    mlp_m0 = None
    if model_type == "qnn":
        _section("Classical MLP baseline (split 0)")
        mlp_m0, _ = t.evaluate_classical_mlp(tr0_np, y_tr0, te0_np, y_te0, seed=BASE_SEED)
        _row("MLP test", mlp_m0)
        _section_end()

    # ── Quantum baselines (Linear, Ring, Full) ────────────────────────────────
    # Baselines use all features (no FS) for methodological correctness.
    # Each run uses a different seed matching the corresponding SEQUENT run,
    # so pairs are valid for the Wilcoxon signed-rank test.
    bl_test_accs = {k_: [] for k_ in _FM_FACTORY}
    bl_test_all  = {k_: [] for k_ in _FM_FACTORY}

    if run_baselines:
        bl_session = (Session(backend=QiskitRuntimeService().backend(backend_name))
                      if mode == "hardware" else nullcontext(None))
        with bl_session as session:
            for run_i in range(n_runs):
                seed_i = BASE_SEED + run_i
                sp = _make_splits(X_orig, y_orig, seed_i)
                tr_np = sp["outer_train_X"].to_numpy()
                y_tr  = sp["outer_train_y"].to_numpy()
                te_np = sp["test_X"].to_numpy()
                y_te  = sp["test_y"].to_numpy()

                for bl_key, bl_factory in _FM_FACTORY.items():
                    fm_bl = bl_factory(X_orig.shape[1], reps=reps)
                    extra = {}
                    if mode == "hardware":
                        extra = dict(runtime_session=session,
                                     job_tag=f"SEQUENT_BL_{dataset}_{model_type}_{bl_key}_run{run_i+1}")
                    val_m, clf_bl = evaluate_fn(fm_bl, tr_np, y_tr, tr_np, y_tr, **extra)
                    test_m = t.compute_metrics(clf_bl, te_np, y_te)
                    bl_test_accs[bl_key].append(test_m["accuracy"])
                    bl_test_all[bl_key].append(test_m)
                    print(f"  [BL {bl_key:<6} run {run_i+1}/{n_runs}]  "
                          f"test_acc={test_m['accuracy']:.4f}")

        for bl_key in _FM_FACTORY:
            agg = _aggregate(bl_test_all[bl_key])
            fm0 = _FM_FACTORY[bl_key](X_orig.shape[1], reps=reps)
            _section(f"Baseline: {bl_key.capitalize()} ({n_runs} runs)")
            _row("test (aggregated)", agg)
            _circuit_row("circuit", t.circuit_complexity(fm0))
            _section_end()

    # ── SEQUENT runs ──────────────────────────────────────────────────────────
    _header(f"SEQUENT ({metaheuristic.upper()}, {n_runs} runs)")

    all_test, all_comp, all_cls = [], [], []
    svm_accs_per_run, mlp_accs_per_run = [], []
    run_logs = []
    best_acc, best_solution = -1.0, None

    for run_i in range(n_runs):
        seed_i = BASE_SEED + run_i
        splits = _make_splits(X_orig, y_orig, seed_i)
        fv = _prepare_fs_views(splits, use_fs, fs_method, k, seed_i)

        tr_np   = fv["search_train_np"]
        y_tr    = splits["search_train_y"].to_numpy()
        val_np  = fv["search_val_np"]
        y_val   = splits["search_val_y"].to_numpy()
        out_np  = fv["outer_train_np"]
        y_out   = splits["outer_train_y"].to_numpy()
        te_np   = fv["test_np"]
        y_te    = splits["test_y"].to_numpy()

        # Classical per-run baselines
        svm_m_i, _ = t.evaluate_classical_svm(out_np, y_out, te_np, y_te)
        svm_accs_per_run.append(svm_m_i["accuracy"])

        if model_type == "qnn":
            mlp_m_i, _ = t.evaluate_classical_mlp(out_np, y_out, te_np, y_te, seed=seed_i)
            mlp_accs_per_run.append(mlp_m_i["accuracy"])

        # Open hardware session per SEQUENT run if needed
        seq_session = (Session(backend=QiskitRuntimeService().backend(backend_name))
                       if mode == "hardware" else nullcontext(None))

        with seq_session as session:
            if mode == "hardware":
                from functools import partial
                search_fn = partial(evaluate_fn, fast_eval=True, runtime_session=session,
                                    job_tag=f"SEQUENT_{dataset}_{model_type}_search_run{run_i+1}")
            else:
                search_fn = evaluate_fn

            cache = {}
            obj = _Objective(fv["pairs_fs"], fv["fs_cols"],
                             tr_np, y_tr, val_np, y_val,
                             search_fn, reps, cache,
                             objective_metric=objective_metric)

            best_sol, _ = _run_optimiser(metaheuristic, obj,
                                         len(fv["pairs_fs"]), cfg, seed_i,
                                         fv["warm_start"])

            # Final evaluation with full training set
            selected_pairs = [fv["pairs_fs"][i] for i, b in enumerate(best_sol) if b == 1]
            fm_final = t.createFeatureMap(selected_pairs, fv["fs_cols"], reps=reps)

            if mode == "hardware":
                final_m, final_model = evaluate_fn(
                    fm_final, out_np, y_out, te_np, y_te,
                    fast_eval=False, runtime_session=session,
                    job_tag=f"SEQUENT_{dataset}_{model_type}_final_run{run_i+1}",
                )
            else:
                final_m, final_model = evaluate_fn(fm_final, out_np, y_out, te_np, y_te)

        test_m  = t.compute_metrics(final_model, te_np, y_te)
        test_m["training_time"] = float(final_m.get("training_time", 0.0))
        per_cls = t.compute_metrics_per_class(final_model, te_np, y_te)
        comp    = t.circuit_complexity(fm_final)

        all_test.append(test_m)
        all_comp.append(comp)
        all_cls.append(per_cls)

        print(f"\n  -- Run {run_i + 1}/{n_runs}  (seed={seed_i})")
        _row(f"Run {run_i+1} test", test_m)
        print(f"|  {'':22} SVM: {svm_m_i['accuracy']:.4f}"
              + (f"  MLP: {mlp_m_i['accuracy']:.4f}" if model_type == "qnn" else ""))
        _class_rows(per_cls)
        _circuit_row(f"Run {run_i+1} circuit", comp)

        run_logs.append(dict(seed=seed_i, test=test_m, per_class=per_cls,
                             complexity=comp, solution=list(best_sol),
                             svm_accuracy=svm_m_i["accuracy"],
                             mlp_accuracy=mlp_m_i["accuracy"] if model_type == "qnn" else None))

        if test_m["accuracy"] > best_acc:
            best_acc, best_solution = test_m["accuracy"], list(best_sol)

    # ── Aggregated results ───────────────────────────────────────────────────
    agg_test = _aggregate(all_test)
    agg_comp = _aggregate(all_comp)

    _section(f"SEQUENT Aggregated ({n_runs} runs)")
    _row("test", agg_test)
    _circuit_row("complexity", agg_comp)
    _section_end()

    # ── Statistical significance ─────────────────────────────────────────────
    sequent_accs = [r["test"]["accuracy"] for r in run_logs]

    sig_results = {}
    for bl_key in _FM_FACTORY:
        if bl_test_accs[bl_key]:
            sig_results[bl_key] = _significance_tests(sequent_accs, bl_test_accs[bl_key])
            _print_significance(sig_results[bl_key], f"SEQUENT vs {bl_key.capitalize()}")

    sig_svm = _significance_tests(sequent_accs, svm_accs_per_run)
    _print_significance(sig_svm, "SEQUENT vs classical SVM")

    sig_mlp = {}
    if model_type == "qnn" and mlp_accs_per_run:
        sig_mlp = _significance_tests(sequent_accs, mlp_accs_per_run)
        _print_significance(sig_mlp, "QNN vs classical MLP")

    # ── Save ─────────────────────────────────────────────────────────────────
    full_log = dict(
        run_id=run_id, dataset=dataset, model_type=model_type, mode=mode,
        metaheuristic=metaheuristic, fs_method=fs_method, k=k, reps=reps,
        n_runs=n_runs, objective_metric=objective_metric,
        hardware_shots=hardware_shots if mode == "hardware" else None,
        backend_name=backend_name if mode == "hardware" else None,
        aggregated_test=agg_test, aggregated_complexity=agg_comp,
        runs=run_logs,
        svm_accs_per_run=svm_accs_per_run,
        mlp_accs_per_run=mlp_accs_per_run or None,
        baseline_test_accs=bl_test_accs,
        significance_vs_baselines=sig_results,
        significance_vs_svm=sig_svm,
        significance_vs_mlp=sig_mlp or None,
        best_solution=best_solution,
    )

    json_path = os.path.join("results", f"{run_id}.json")
    _save_json(full_log, json_path)
    print(f"\n  Full log → {json_path}")

    csv_row = dict(
        run_id=run_id, dataset=dataset, model_type=model_type, mode=mode,
        metaheuristic=metaheuristic, fs_method=fs_method if use_fs else "none",
        k=k if use_fs else X_orig.shape[1], reps=reps, n_runs=n_runs,
        hardware_shots=hardware_shots if mode == "hardware" else None,
        backend_name=backend_name if mode == "hardware" else None,
        objective_metric=objective_metric,
        svm_acc_mean=float(np.mean(svm_accs_per_run)),
        mlp_acc_mean=float(np.mean(mlp_accs_per_run)) if mlp_accs_per_run else None,
        **{f"{bl_key}_test_acc_mean": float(np.mean(bl_test_accs[bl_key])) if bl_test_accs[bl_key] else None
           for bl_key in _FM_FACTORY},
        test_acc_mean=agg_test.get("accuracy_mean", 0),
        test_acc_std=agg_test.get("accuracy_std", 0),
        test_acc_ci95=agg_test.get("accuracy_ci95", 0),
        test_f1_mean=agg_test.get("f1_macro_mean", 0),
        depth_mean=agg_comp.get("depth_mean", 0),
        two_qubit_gates_mean=agg_comp.get("two_qubit_gates_mean", 0),
        timestamp=timestamp,
    )
    _append_csv(csv_row)
    print(f"  Summary  → {_CSV_PATH}")
    return full_log


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT GRID  ← edit this section to configure experiments
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Each entry: (name, option, path)
    #   option=0 → local file at path
    #   option=1 → PMLB dataset (path ignored)
    DATASETS = [
        ("flare",  0, "./datasets/flare.tsv"),
        # ("corral", 1, None),
        # ("breast-w", 0, "./datasets/breast-w.tsv"),
        # ("heart",   0, "./datasets/heart.csv"),
	# ("fitness_class_2212", 0, "./datasets/fitness_class_2212.csv"),
    ]

    # ── Execution mode ────────────────────────────────────────────────────────
    # "statevector"  → ideal simulator (fastest, local testing)
    # "noise"        → noise-model simulator (requires IBM account)
    # "hardware"     → real IBM backend (requires IBM account + credits)
    MODE = os.environ.get("SEQUENT_MODE", "noise")

    # ── Reps grid ─────────────────────────────────────────────────────────────
    REPS_GRID = [1, 2, 3]

    # ── Default config ────────────────────────────────────────────────────────
    # Override individual keys per-dataset/model in the loop below if needed.
    DEFAULT_CFG = dict(
        mode=MODE,
        model_type="qsvm",        # "qsvm" | "qnn"
        metaheuristic="sa",       # "sa" | "ils" 
        use_fs=True,
        fs_method="anova",        # "anova" | "mutual_info" | "autoencoder"
        k=5,
        reps=1,                   # overridden per job from REPS_GRID
        n_runs=10,
        run_baselines=True,
        objective_metric="accuracy",  # overridden to "f1_macro" for imbalanced

        # SA hyperparameters
        sa_initial_temp=10.0,
        sa_cooling_rate=0.90,
        sa_stopping_temp=0.01,
        sa_max_iterations=10,
        sa_num_neighbors=5,

        # ILS hyperparameters
        ils_n_restarts=3,
        ils_perturbation_strength=0.3,
        ils_local_search_iters=15,
        ils_local_search_initial_temp=5.0,
        ils_local_search_cooling_rate=0.85,
        ils_local_search_stopping_temp=1e-3,
        ils_local_search_num_neighbors=3,
        ils_use_warm_start=True,
        ils_repair=True,

        # Hardware settings
        hardware_shots=512,
        backend_name="ibm_pittsburgh",
    )

    # ── Build job list ────────────────────────────────────────────────────────
    IMBALANCE_THRESHOLD = 1.5

    jobs = []
    for ds, opt, path in DATASETS:
        try:
            _, y_ds = t.load_data(path=path, option=opt, dataset=ds)
            counts = Counter(y_ds)
            ir = max(counts.values()) / min(counts.values())
            imbalanced = ir > IMBALANCE_THRESHOLD
            print(f"  [{ds}] IR={ir:.2f} → {'f1_macro' if imbalanced else 'accuracy'}")
        except Exception as e:
            print(f"  [WARN] Could not pre-load {ds}: {e}. Defaulting to accuracy.")
            imbalanced = False

        for model_type in ["qsvm", "qnn"]:
            for mh in ["sa", "ils"]:
                for fs in ["anova", "autoencoder"]:
                    for rep in REPS_GRID:
                        cfg = dict(DEFAULT_CFG)
                        cfg["model_type"] = model_type
                        cfg["metaheuristic"] = mh
                        cfg["fs_method"] = fs
                        cfg["reps"] = rep
                        if imbalanced:
                            cfg["objective_metric"] = "f1_macro"
                        jobs.append((ds, opt, path, cfg))

    # ── Optional: filter to a single job index (for SLURM array jobs) ─────────
    job_idx_env = os.environ.get("SEQUENT_JOB_INDEX", "").strip()
    if job_idx_env:
        jobs = [jobs[int(job_idx_env)]]
        print(f"  Running single job index {job_idx_env}")
    else:
        print(f"  Running full grid: {len(jobs)} jobs")

    # ── Execute ───────────────────────────────────────────────────────────────
    for ds, opt, path, cfg in jobs:
        if _already_done(ds, cfg["model_type"], cfg["mode"],
                         cfg["metaheuristic"], cfg["fs_method"], cfg["reps"]):
            print(f"  [SKIP] {ds}/{cfg['model_type']}/{cfg['metaheuristic']}/{cfg['fs_method']} already in CSV")
            continue

        print(f"\n  Running: {ds} / {cfg['model_type']} / {cfg['metaheuristic']} / {cfg['fs_method']}")
        try:
            run_experiment(ds, opt, path, cfg)
        except Exception as e:
            import traceback
            print(f"\n  [ERROR] {ds}/{cfg['model_type']}: {e}")
            traceback.print_exc()
