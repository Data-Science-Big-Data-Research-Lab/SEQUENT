"""
main.py - SEQUENT Experiment Runner
===================================

This version keeps the 24 jobs independent. Quantum baselines are computed only
for the canonical configurations where run_baselines=True. Other jobs skip the
expensive Linear/Ring/Full quantum baselines and save only their own SEQUENT
results. Baseline comparisons can be joined later in post-processing.
"""

import os
import time
import json
import random
import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from qiskit_algorithms.utils import algorithm_globals
from qiskit_ibm_runtime import QiskitRuntimeService, Session
import tools as t
from metaheuristicas import (
    simulated_annealing,
    tabu_search,
    iterated_local_search,
    genetic_algorithm,
)

if os.environ.get("QISKIT_IBM_TOKEN"):
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform",
        token=os.environ["QISKIT_IBM_TOKEN"],
        instance="crn:v1:bluemix:public:quantum-computing:us-east:a/517eab0b52bb4662bd562b5ce2e52f1c:62e8a60f-b741-4446-9085-330bdf4187f5::",
        overwrite=True,
        set_as_default=True
    )


BASE_SEED = 12345
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
algorithm_globals.random_seed = BASE_SEED


_EVALUATE = {
    "qsvm": {
        "statevector": t.evaluate_qsvm_statevector,
        "noise": t.evaluate_qsvm_noise_sim,
        "hardware": t.evaluate_qsvm_hardware,
    },
    "qnn": {
        "statevector": t.evaluate_qnn_statevector,
        "noise": t.evaluate_qnn_noise_sim,
        "hardware": t.evaluate_qnn_hardware,
    },
}

_OPTIMISER = {
    "sa": simulated_annealing,
    "ts": tabu_search,
    "ils": iterated_local_search,
    "ga": genetic_algorithm,
}

os.makedirs("results2", exist_ok=True)
_CSV_PATH = os.path.join("results2", "benchmark_results.csv")


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def _header(text, width=70):
    bar = "=" * width
    print(f"\n+{bar}+")
    lines = [text[i:i + width] for i in range(0, len(text), width)]
    for line in lines:
        print(f"|  {line:<{width - 2}}|")
    print(f"+{bar}+")


def _section(title):
    print(f"\n--- {title} {'-' * max(0, 65 - len(title))}")


def _section_end():
    print("-" * 72)


def _row(label, metrics):
    acc = metrics.get("accuracy", metrics.get("accuracy_mean", 0))
    prec = metrics.get("precision_macro", metrics.get("precision_macro_mean", 0))
    rec = metrics.get("recall_macro", metrics.get("recall_macro_mean", 0))
    f1 = metrics.get("f1_macro", metrics.get("f1_macro_mean", 0))
    ttime = metrics.get("training_time", metrics.get("training_time_mean", 0))
    itime = metrics.get("inference_time", metrics.get("inference_time_mean", 0))

    std_acc = metrics.get("accuracy_std", None)
    std_f1 = metrics.get("f1_macro_std", None)
    ci_acc = metrics.get("accuracy_ci95", None)
    ci_f1 = metrics.get("f1_macro_ci95", None)

    acc_str = f"{acc:.4f}"
    if std_acc is not None:
        acc_str += f" +/-{std_acc:.4f}"
    if ci_acc is not None:
        acc_str += f" [95%CI +/-{ci_acc:.4f}]"

    f1_str = f"{f1:.4f}"
    if std_f1 is not None:
        f1_str += f" +/-{std_f1:.4f}"
    if ci_f1 is not None:
        f1_str += f" [95%CI +/-{ci_f1:.4f}]"

    print(
        f"|  {label:<22} Acc:{acc_str:<30} "
        f"Prec:{prec:.4f}  Rec:{rec:.4f}  F1:{f1_str}"
    )

    if ttime > 0:
        inf_str = f"{itime:.3f}s" if itime > 0 else "-"
        print(f"|  {'':22} Train:{ttime:.2f}s  Infer:{inf_str}")


def _circuit_row(label, complexity):
    d = complexity.get("depth", complexity.get("depth_mean", 0))
    tg = complexity.get("total_gates", complexity.get("total_gates_mean", 0))
    tq = complexity.get("two_qubit_gates", complexity.get("two_qubit_gates_mean", 0))
    ss = complexity.get("search_space", "-")

    print(
        f"|  {label:<22} depth:{d}  total_gates:{tg}  "
        f"2Q-gates:{tq}  search_space:{ss}"
    )


def _class_rows(per_class):
    for cls, m in per_class.items():
        print(
            f"|    class {cls:<6}  prec:{m['precision']:.4f}  "
            f"rec:{m['recall']:.4f}  f1:{m['f1']:.4f}  n={m['support']}"
        )


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------

def _aggregate(results_list):
    if not results_list:
        return {}

    keys = results_list[0].keys()
    out = {}

    for k in keys:
        vals = [
            r[k]
            for r in results_list
            if isinstance(r.get(k), (int, float)) and not np.isnan(r[k])
        ]

        if vals:
            n = len(vals)
            mu = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            sem = std / np.sqrt(n)
            t_crit = float(sp_stats.t.ppf(0.975, df=max(n - 1, 1)))
            ci95 = float(t_crit * sem)

            out[f"{k}_mean"] = mu
            out[f"{k}_std"] = std
            out[f"{k}_ci95"] = ci95

    return out


def _directional_comparison(sequent_accs, baseline_accs):
    if not sequent_accs or not baseline_accs:
        return {}

    a = np.array(sequent_accs, dtype=float)
    b = np.array(baseline_accs, dtype=float)
    diffs = a - b

    return {
        "n_runs": len(diffs),
        "wins": int(np.sum(diffs > 0)),
        "losses": int(np.sum(diffs < 0)),
        "ties": int(np.sum(diffs == 0)),
        "mean_diff": float(np.mean(diffs)),
        "median_diff": float(np.median(diffs)),
        "sequent_acc_mean": float(np.mean(a)),
        "baseline_acc_mean": float(np.mean(b)),
        "sequent_better_mean": bool(np.mean(a) > np.mean(b)),
    }


def _build_warm_start(scores, activation_ratio=0.30):
    if scores is None:
        return None

    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return []

    arr = np.nan_to_num(
        np.abs(arr),
        nan=-np.inf,
        posinf=-np.inf,
        neginf=-np.inf,
    )

    if np.all(np.isneginf(arr)):
        warm = [0] * len(arr)
        if warm:
            warm[0] = 1
        return warm

    n_active = max(1, int(round(len(arr) * activation_ratio)))
    n_active = min(n_active, len(arr))
    top_idx = np.argsort(arr)[::-1][:n_active]

    warm = [0] * len(arr)
    for idx in top_idx:
        warm[int(idx)] = 1

    return warm


def _significance_tests(sequent_accs, baseline_accs):
    n = len(sequent_accs)
    results = {"n_runs": n}

    if n < 2:
        results["note"] = "Need n_runs >= 2 for significance tests"
        return results

    a = np.array(sequent_accs, dtype=float)
    b = np.array(baseline_accs, dtype=float)
    diffs = a - b

    try:
        observed_stat = float(np.sum(diffs))
        count_ge = 0
        total = 0

        for signs in range(1 << n):
            sign_vec = np.array(
                [1 if (signs >> i) & 1 else -1 for i in range(n)],
                dtype=float,
            )
            perm_stat = float(np.sum(np.abs(diffs) * sign_vec))
            if perm_stat >= observed_stat:
                count_ge += 1
            total += 1

        p_perm = count_ge / total
        results["permutation_stat"] = observed_stat
        results["permutation_p"] = float(p_perm)
        results["permutation_sig_05"] = bool(p_perm < 0.05)

    except Exception as e:
        results["permutation_error"] = str(e)

    try:
        wins = int(np.sum(diffs > 0))
        losses = int(np.sum(diffs < 0))
        n_tied = int(np.sum(diffs == 0))
        n_eff = wins + losses

        if n_eff > 0:
            try:
                p_sign = float(
                    sp_stats.binom_test(
                        wins,
                        n_eff,
                        0.5,
                        alternative="greater",
                    )
                )
            except AttributeError:
                p_sign = float(
                    sp_stats.binomtest(
                        wins,
                        n_eff,
                        0.5,
                        alternative="greater",
                    ).pvalue
                )
        else:
            p_sign = 1.0

        results.update(
            sign_wins=wins,
            sign_losses=losses,
            sign_ties=n_tied,
            sign_p=p_sign,
            sign_sig_05=bool(p_sign < 0.05),
        )

    except Exception as e:
        results["sign_error"] = str(e)

    try:
        stat_t, p_t_two = sp_stats.ttest_rel(a, b)
        p_t = float(p_t_two / 2) if stat_t > 0 else 1.0

        results.update(
            ttest_stat=float(stat_t),
            ttest_p=p_t,
            ttest_sig_05=bool(p_t < 0.05),
        )

    except Exception as e:
        results["ttest_error"] = str(e)

    return results


def _print_significance_block(sig_dict, label):
    _section(f"Statistical Significance ({label})")

    n = sig_dict.get("n_runs", 0)

    if "note" in sig_dict:
        print(f"|  {sig_dict['note']}")
        _section_end()
        return

    print(f"|  n_runs = {n}  (one-sided tests, H1: SEQUENT > baseline)")

    if "permutation_p" in sig_dict:
        p = sig_dict["permutation_p"]
        sig = sig_dict["permutation_sig_05"]
        print(
            f"|  Permutation test  (primary):    p = {p:.4f}  "
            f"{'*** p<0.05' if sig else 'not significant'}"
        )

    if "sign_p" in sig_dict:
        p = sig_dict["sign_p"]
        sig = sig_dict["sign_sig_05"]
        wins = sig_dict.get("sign_wins", "?")
        ties = sig_dict.get("sign_ties", 0)
        print(
            f"|  Sign test         (robust):     p = {p:.4f}  "
            f"{'*** p<0.05' if sig else 'not significant'}"
            f"  (wins={wins}, ties={ties})"
        )

    if "ttest_p" in sig_dict:
        p = sig_dict["ttest_p"]
        sig = sig_dict["ttest_sig_05"]
        print(
            f"|  Paired t-test     (parametric): p = {p:.4f}  "
            f"{'*** p<0.05' if sig else 'not significant'}"
        )

    _section_end()


# -----------------------------------------------------------------------------
# Objective function
# -----------------------------------------------------------------------------

class _Objective:
    def __init__(
        self,
        couples_array,
        columns,
        tr,
        y_tr,
        val,
        y_val,
        evaluate_fn,
        reps,
        cache,
        cv_folds=0,
        objective_metric="accuracy",
    ):
        self.couples_array = couples_array
        self.columns = columns
        self.tr = tr
        self.y_tr = y_tr
        self.val = val
        self.y_val = y_val
        self.evaluate_fn = evaluate_fn
        self.reps = reps
        self.cache = cache
        self.cv_folds = cv_folds
        self.objective_metric = objective_metric

    def _mask_penalty(self, vector):
        n_active = int(np.sum(vector))
        min_active = 3
        max_active = 5
        penalty_step = 0.01

        if n_active < min_active:
            return penalty_step * (min_active - n_active)

        if n_active > max_active:
            return penalty_step * (n_active - max_active)

        return 0.0

    def __call__(self, vector):
        key = str(vector)

        if key in self.cache:
            return self.cache[key]["cost"], self.cache[key]["aux"]

        selected = [
            self.couples_array[i]
            for i, bit in enumerate(vector)
            if bit == 1
        ]

        fm = t.createFeatureMap(
            selected,
            self.columns,
            reps=self.reps,
        )

        metrics, model = self.evaluate_fn(
            fm,
            self.tr,
            self.y_tr,
            self.val,
            self.y_val,
        )

        val_metric = metrics.get(
            self.objective_metric,
            metrics["accuracy"],
        )

        mask_penalty = self._mask_penalty(vector)
        cost = -(val_metric - mask_penalty)

        self.cache[key] = {
            "solution": list(vector),
            "cost": float(cost),
            "aux": metrics.get("training_time", 0.0),
            "metrics": metrics,
            "model": model,
            "feature_map": fm,
            "active_pairs": int(np.sum(vector)),
            "mask_penalty": float(mask_penalty),
        }

        return self.cache[key]["cost"], self.cache[key]["aux"]


def _build_objective(
    couples_array,
    columns,
    tr,
    y_tr,
    val,
    y_val,
    evaluate_fn,
    reps,
    cache,
    cv_folds=0,
    objective_metric="accuracy",
    overfitting_beta=None,
    base_seed=0,
):
    return _Objective(
        couples_array,
        columns,
        tr,
        y_tr,
        val,
        y_val,
        evaluate_fn,
        reps,
        cache,
        cv_folds=cv_folds,
        objective_metric=objective_metric,
    )


# -----------------------------------------------------------------------------
# Split helpers
# -----------------------------------------------------------------------------

_FM_FACTORY = {
    "linear": t.createFeatureMapLinear,
    "ring": t.createFeatureMapRing,
    "full": t.createFeatureMapFull,
}

_FM_NAME = {
    "linear": "Linear",
    "ring": "Ring",
    "full": "Full",
}


def _make_splits(X, y, seed):
    X_outer, X_test, y_outer, y_test = t.splitData(
        X,
        y,
        test_size=0.3,
        random_state=seed,
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_outer,
        y_outer,
        test_size=0.3,
        random_state=seed,
        stratify=y_outer,
    )

    return {
        "outer_train_X": X_outer,
        "outer_train_y": y_outer,
        "search_train_X": X_tr,
        "search_train_y": y_tr,
        "search_val_X": X_val,
        "search_val_y": y_val,
        "test_X": X_test,
        "test_y": y_test,
    }


def _prepare_fs_views(splits, use_fs, fs_method, k, seed):
    X_str = splits["search_train_X"]
    y_str = splits["search_train_y"]

    if use_fs:
        fitted_fs, X_str_fs, others = t.fit_transform_feature_selection(
            X_str,
            [
                splits["search_val_X"],
                splits["outer_train_X"],
                splits["test_X"],
            ],
            y_train=y_str,
            method=fs_method,
            k=k,
            ae_seed=seed,
        )

        X_val_fs, X_outer_fs, X_test_fs = others
        fs_cols = X_str_fs.columns

    else:
        fitted_fs = None
        X_str_fs = X_str.copy()
        X_val_fs = splits["search_val_X"].copy()
        X_outer_fs = splits["outer_train_X"].copy()
        X_test_fs = splits["test_X"].copy()
        fs_cols = X_str_fs.columns

    fs_scores = t.transformCorrelations(X_str_fs.corr())
    pairs_fs = t.createCouples(fs_scores, fs_cols)
    warm_start_fs = _build_warm_start(fs_scores, activation_ratio=0.30)

    return {
        "fitted_fs": fitted_fs,
        "fs_cols": fs_cols,
        "pairs_fs": pairs_fs,
        "warm_start_fs": warm_start_fs,
        "search_train_np": X_str_fs.to_numpy(),
        "search_val_np": X_val_fs.to_numpy(),
        "outer_train_np": X_outer_fs.to_numpy(),
        "test_np": X_test_fs.to_numpy(),
    }


# -----------------------------------------------------------------------------
# Workers
# -----------------------------------------------------------------------------

def _baseline_worker(args):
    run_idx_bl, key, ctx = args

    X_orig = ctx["X_orig"]
    y_orig = ctx["y_orig"]
    evaluate_fn = ctx["evaluate_fn"]
    reps = ctx["reps"]
    base_seed = ctx["base_seed"]

    seed_bl = base_seed + run_idx_bl
    splits = _make_splits(X_orig, y_orig, seed_bl)

    tr_np = splits["outer_train_X"].to_numpy()
    y_tr_np = splits["outer_train_y"].to_numpy()
    val_np = splits["search_val_X"].to_numpy()
    y_val_np = splits["search_val_y"].to_numpy()
    te_np = splits["test_X"].to_numpy()
    y_te_np = splits["test_y"].to_numpy()

    fm = _FM_FACTORY[key](X_orig.shape[1], reps=reps)

    val_m, clf = evaluate_fn(
        fm,
        tr_np,
        y_tr_np,
        val_np,
        y_val_np,
    )

    test_m = t.compute_metrics(clf, te_np, y_te_np)

    return {
        "run_idx": run_idx_bl,
        "key": key,
        "name": _FM_NAME[key],
        "seed": seed_bl,
        "val": val_m,
        "test": test_m,
    }


def _sequent_worker(args):
    run_idx, ctx = args

    X_orig = ctx["X_orig"]
    y_orig = ctx["y_orig"]
    use_fs = ctx["use_fs"]
    fs_method = ctx["fs_method"]
    k = ctx["k"]
    evaluate_fn = ctx["evaluate_fn"]
    reps = ctx["reps"]
    base_seed = ctx["base_seed"]
    model_type = ctx["model_type"]
    metaheuristic = ctx["metaheuristic"]
    ga_cv_folds = ctx["ga_cv_folds"]
    objective_metric = ctx["objective_metric"]
    overfitting_beta = ctx["overfitting_beta"]

    sa_initial_temp = ctx["sa_initial_temp"]
    sa_cooling_rate = ctx["sa_cooling_rate"]
    sa_stopping_temp = ctx["sa_stopping_temp"]
    sa_max_iterations = ctx["sa_max_iterations"]
    sa_num_neighbors = ctx["sa_num_neighbors"]

    ga_population_size = ctx["ga_population_size"]
    ga_n_generations = ctx["ga_n_generations"]
    ga_crossover_rate = ctx["ga_crossover_rate"]
    ga_mutation_rate = ctx["ga_mutation_rate"]
    ga_tournament_size = ctx["ga_tournament_size"]
    ga_elitism_count = ctx["ga_elitism_count"]

    ts_max_iterations = ctx["ts_max_iterations"]
    ts_tabu_tenure = ctx["ts_tabu_tenure"]
    ts_max_no_improve = ctx["ts_max_no_improve"]
    ts_neighborhood_sample_size = ctx["ts_neighborhood_sample_size"]
    ts_restart_fraction = ctx["ts_restart_fraction"]

    ils_n_restarts = ctx["ils_n_restarts"]
    ils_perturbation_strength = ctx["ils_perturbation_strength"]
    ils_local_search_iters = ctx["ils_local_search_iters"]
    ils_local_search_initial_temp = ctx["ils_local_search_initial_temp"]
    ils_local_search_cooling_rate = ctx["ils_local_search_cooling_rate"]
    ils_local_search_stopping_temp = ctx["ils_local_search_stopping_temp"]
    ils_local_search_num_neighbors = ctx["ils_local_search_num_neighbors"]
    ils_use_warm_start = ctx["ils_use_warm_start"]
    ils_repair = ctx["ils_repair"]

    optimise = _OPTIMISER[metaheuristic]
    seed_i = base_seed + run_idx

    splits = _make_splits(X_orig, y_orig, seed_i)
    fs_views = _prepare_fs_views(
        splits,
        use_fs,
        fs_method,
        k,
        seed_i,
    )

    tr_np = fs_views["search_train_np"]
    y_tr_np = splits["search_train_y"].to_numpy()
    val_np = fs_views["search_val_np"]
    y_val_np = splits["search_val_y"].to_numpy()
    outer_np = fs_views["outer_train_np"]
    y_out_np = splits["outer_train_y"].to_numpy()
    test_np = fs_views["test_np"]
    y_te_np = splits["test_y"].to_numpy()

    svm_m_i, _ = t.evaluate_classical_svm(
        outer_np,
        y_out_np,
        test_np,
        y_te_np,
    )

    mlp_m_i = None
    if model_type == "qnn":
        mlp_m_i, _ = t.evaluate_classical_mlp(
            outer_np,
            y_out_np,
            test_np,
            y_te_np,
            seed=seed_i,
        )

    cache = {}
    _cv = ga_cv_folds if metaheuristic == "ga" else 0

    from functools import partial as _partial
    _search_fn = _partial(evaluate_fn, fast_eval=True)

    obj = _build_objective(
        fs_views["pairs_fs"],
        fs_views["fs_cols"],
        tr_np,
        y_tr_np,
        val_np,
        y_val_np,
        _search_fn,
        reps,
        cache,
        cv_folds=_cv,
        objective_metric=objective_metric,
        overfitting_beta=overfitting_beta,
        base_seed=seed_i,
    )

    if metaheuristic == "sa":
        best_sol, _ = optimise(
            obj,
            chromosome_length=len(fs_views["pairs_fs"]),
            initial_temp=sa_initial_temp,
            cooling_rate=sa_cooling_rate,
            stopping_temp=sa_stopping_temp,
            max_iterations=sa_max_iterations,
            num_neighbors=sa_num_neighbors,
            seed=seed_i,
        )

    elif metaheuristic == "ts":
        best_sol, _ = optimise(
            obj,
            chromosome_length=len(fs_views["pairs_fs"]),
            max_iterations=ts_max_iterations,
            tabu_tenure=ts_tabu_tenure,
            max_no_improve=ts_max_no_improve,
            neighborhood_sample_size=ts_neighborhood_sample_size,
            restart_fraction=ts_restart_fraction,
            seed=seed_i,
        )

    elif metaheuristic == "ils":
        _warm = fs_views["warm_start_fs"] if ils_use_warm_start else None

        best_sol, _ = optimise(
            obj,
            chromosome_length=len(fs_views["pairs_fs"]),
            initial_solution=_warm,
            warm_start=_warm,
            n_restarts=ils_n_restarts,
            perturbation_strength=ils_perturbation_strength,
            local_search_iters=ils_local_search_iters,
            local_search_initial_temp=ils_local_search_initial_temp,
            local_search_cooling_rate=ils_local_search_cooling_rate,
            local_search_stopping_temp=ils_local_search_stopping_temp,
            local_search_num_neighbors=ils_local_search_num_neighbors,
            repair=ils_repair,
            seed=seed_i,
        )

    else:
        best_sol, _ = optimise(
            obj,
            chromosome_length=len(fs_views["pairs_fs"]),
            population_size=ga_population_size,
            n_generations=ga_n_generations,
            crossover_rate=ga_crossover_rate,
            mutation_rate=ga_mutation_rate,
            tournament_size=ga_tournament_size,
            elitism_count=ga_elitism_count,
            seed=seed_i,
        )

    selected_pairs = [
        fs_views["pairs_fs"][i]
        for i, b in enumerate(best_sol)
        if b == 1
    ]

    fm_final = t.createFeatureMap(
        selected_pairs,
        fs_views["fs_cols"],
        reps=reps,
    )

    # -----------------------------------------------------------------
    # CAMBIO 1: capturar training_time del refit final
    # -----------------------------------------------------------------
    final_metrics, final_model = evaluate_fn(
        fm_final,
        outer_np,
        y_out_np,
        test_np,
        y_te_np,
        fast_eval=False,
    )

    search_val_m = t.compute_metrics(final_model, val_np, y_val_np)
    test_m = t.compute_metrics(final_model, test_np, y_te_np)
    # Propagar el training_time del refit al diccionario de métricas de test
    test_m["training_time"] = final_metrics.get("training_time", 0.0)

    per_cls = t.compute_metrics_per_class(final_model, test_np, y_te_np)
    comp = t.circuit_complexity(fm_final)

    return {
        "run_idx": run_idx,
        "seed": seed_i,
        "search_val_m": search_val_m,
        "test_m": test_m,
        "per_cls": per_cls,
        "comp": comp,
        "solution": list(best_sol),
        "svm_accuracy": svm_m_i["accuracy"],
        "mlp_accuracy": mlp_m_i["accuracy"] if mlp_m_i else None,
    }


# -----------------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------------

def _save_json(data, path):
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj

        return str(obj)

    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)


def _append_csv(row):
    df_new = pd.DataFrame([row])

    if os.path.exists(_CSV_PATH):
        df_new.to_csv(_CSV_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(_CSV_PATH, mode="w", header=True, index=False)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_experiment(
    dataset,
    option=1,
    path=None,
    model_type="qsvm",
    mode="statevector",
    reps=1,
    use_fs=True,
    fs_method="anova",
    k=5,
    metaheuristic="sa",
    sa_initial_temp=10.0,
    sa_cooling_rate=0.90,
    sa_stopping_temp=0.01,
    sa_max_iterations=5,
    sa_num_neighbors=3,
    ga_population_size=10,
    ga_n_generations=10,
    ga_crossover_rate=0.8,
    ga_mutation_rate=0.2,
    ga_tournament_size=2,
    ga_elitism_count=1,
    ga_cv_folds=0,
    ts_max_iterations=12,
    ts_tabu_tenure=4,
    ts_max_no_improve=4,
    ts_neighborhood_sample_size=None,
    ts_restart_fraction=0.2,
    ils_n_restarts=5,
    ils_perturbation_strength=0.3,
    ils_local_search_iters=5,
    ils_local_search_initial_temp=5.0,
    ils_local_search_cooling_rate=0.85,
    ils_local_search_stopping_temp=0.001,
    ils_local_search_num_neighbors=3,
    ils_use_warm_start=True,
    ils_repair=True,
    n_runs=5,
    use_smote=False,
    run_baselines=True,
    run_ablation=True,
    objective_metric="accuracy",
    overfitting_beta=0.5,
    gpu=None,
):
    if model_type not in _EVALUATE:
        raise ValueError(
            f"model_type must be 'qsvm' or 'qnn'; got '{model_type}'"
        )

    if mode not in _EVALUATE[model_type]:
        raise ValueError(
            f"mode must be 'statevector'|'noise'|'hardware'; got '{mode}'"
        )

    if metaheuristic not in _OPTIMISER:
        raise ValueError(
            f"metaheuristic must be 'sa','ts','ils' or 'ga'; got '{metaheuristic}'"
        )

    _eval_raw = _EVALUATE[model_type][mode]

    if mode == "noise" and gpu is not None:
        from functools import partial
        evaluate_fn = partial(_eval_raw, gpu=gpu)
    else:
        evaluate_fn = _eval_raw

    optimise = _OPTIMISER[metaheuristic]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{dataset}_{model_type}_{mode}_{metaheuristic}_"
        f"{fs_method if use_fs else 'none'}_reps{reps}_{timestamp}"
    )

    _header(
        f"SEQUENT | {dataset} | {model_type.upper()} | {mode} | "
        f"{metaheuristic.upper()} | FS:{use_fs}({fs_method},k={k}) | "
        f"reps={reps} | runs={n_runs}"
    )

    full_log = {
        "run_id": run_id,
        "dataset": dataset,
        "model_type": model_type,
        "mode": mode,
        "metaheuristic": metaheuristic,
        "use_fs": use_fs,
        "fs_method": fs_method,
        "k": k,
        "reps": reps,
        "n_runs": n_runs,
        "use_smote": use_smote,
        "run_baselines": run_baselines,
        "run_ablation": run_ablation,
        "ga_cv_folds": ga_cv_folds if metaheuristic == "ga" else 0,
        "objective_metric": objective_metric,
        "overfitting_beta": overfitting_beta,
        "base_seed": BASE_SEED,
        "split_seeds": list(range(BASE_SEED, BASE_SEED + n_runs)),
        "ils_n_restarts": ils_n_restarts if metaheuristic == "ils" else None,
        "ils_perturbation_strength": (
            ils_perturbation_strength if metaheuristic == "ils" else None
        ),
        "ils_local_search_iters": (
            ils_local_search_iters if metaheuristic == "ils" else None
        ),
        "ils_local_search_initial_temp": (
            ils_local_search_initial_temp if metaheuristic == "ils" else None
        ),
        "ils_local_search_cooling_rate": (
            ils_local_search_cooling_rate if metaheuristic == "ils" else None
        ),
        "ils_local_search_stopping_temp": (
            ils_local_search_stopping_temp if metaheuristic == "ils" else None
        ),
        "ils_local_search_num_neighbors": (
            ils_local_search_num_neighbors if metaheuristic == "ils" else None
        ),
        "ils_use_warm_start": (
            ils_use_warm_start if metaheuristic == "ils" else None
        ),
        "ils_repair": ils_repair if metaheuristic == "ils" else None,
    }

    X_orig, y_orig = t.load_data(path=path, option=option, dataset=dataset)

    print(f"\n  Dataset: {X_orig.shape[0]} samples x {X_orig.shape[1]} features")
    print(f"  Class distribution: {dict(y_orig.value_counts().sort_index())}")
    print(f"  Split seeds: {BASE_SEED} ... {BASE_SEED + n_runs - 1}")

    if use_fs:
        print(f"\n  FS inside each split: {fs_method}, k={k}")
    else:
        print("\n  No FS: raw features")

    first_seed = BASE_SEED
    first_splits = _make_splits(X_orig, y_orig, first_seed)

    tr_o = first_splits["outer_train_X"].to_numpy()
    y_tr_o = first_splits["outer_train_y"].to_numpy()
    val_o = first_splits["search_val_X"].to_numpy()
    y_val_o = first_splits["search_val_y"].to_numpy()
    test_o = first_splits["test_X"].to_numpy()
    y_te_o = first_splits["test_y"].to_numpy()

    _section("Baseline: Classical RBF-SVM (raw features, test set, split 0)")

    svm_m_first, svm_model_first = t.evaluate_classical_svm(
        tr_o,
        y_tr_o,
        test_o,
        y_te_o,
    )

    _row("SVM test", svm_m_first)
    _section_end()

    full_log["baseline_svm_split0"] = svm_m_first

    mlp_m_first = None

    if model_type == "qnn":
        _section("Baseline: Classical MLP (raw features, test set, split 0)")

        mlp_m_first, _ = t.evaluate_classical_mlp(
            tr_o,
            y_tr_o,
            test_o,
            y_te_o,
            seed=first_seed,
        )

        _row("MLP test", mlp_m_first)
        _section_end()

        full_log["baseline_mlp_split0"] = mlp_m_first

    baseline_results = {}
    _bl_all_val = {"linear": [], "ring": [], "full": []}
    _bl_all_test = {"linear": [], "ring": [], "full": []}
    _bl_run_logs = {"linear": [], "ring": [], "full": []}

    if run_baselines:
        _total_cpus = int(
            os.environ.get(
                "SLURM_CPUS_PER_TASK",
                multiprocessing.cpu_count(),
            )
        )

        _n_workers_bl = min(5, max(1, _total_cpus), n_runs * 3)

        bl_ctx = {
            "X_orig": X_orig,
            "y_orig": y_orig,
            "evaluate_fn": evaluate_fn,
            "reps": reps,
            "base_seed": BASE_SEED,
        }

        bl_tasks = [
            (i, k_, bl_ctx)
            for i in range(n_runs)
            for k_ in ("linear", "ring", "full")
        ]

        print(
            f"\n  [baselines] launching {len(bl_tasks)} tasks "
            f"on {_n_workers_bl} workers",
            flush=True,
        )

        _old_threads_env = os.environ.get("SEQUENT_THREADS_PER_WORKER")
        os.environ["SEQUENT_THREADS_PER_WORKER"] = "1"

        try:
            _mp_ctx = mp.get_context("spawn") if gpu is not None else None
            with ProcessPoolExecutor(max_workers=_n_workers_bl, mp_context=_mp_ctx) as executor:
                futures = {
                    executor.submit(_baseline_worker, task): task
                    for task in bl_tasks
                }

                done_count = 0

                for future in as_completed(futures):
                    r = future.result()
                    done_count += 1

                    _bl_all_val[r["key"]].append(r["val"])
                    _bl_all_test[r["key"]].append(r["test"])
                    _bl_run_logs[r["key"]].append({
                        "seed": r["seed"],
                        "val": r["val"],
                        "test": r["test"],
                    })

                    print(
                        f"  [baseline {done_count:>2}/{len(bl_tasks)}] "
                        f"{r['name']:<6} reps={reps}  "
                        f"run {r['run_idx'] + 1}/{n_runs} "
                        f"val_acc={r['val']['accuracy']:.4f}  "
                        f"test_acc={r['test']['accuracy']:.4f}  "
                        f"train_time={r['val'].get('training_time', 0):.1f}s",
                        flush=True,
                    )

        finally:
            if _old_threads_env is None:
                os.environ.pop("SEQUENT_THREADS_PER_WORKER", None)
            else:
                os.environ["SEQUENT_THREADS_PER_WORKER"] = _old_threads_env

        for name, key in [
            ("Linear", "linear"),
            ("Ring", "ring"),
            ("Full", "full"),
        ]:
            agg_bl_val = _aggregate(_bl_all_val[key])
            agg_bl_test = _aggregate(_bl_all_test[key])

            fm0 = _FM_FACTORY[key](X_orig.shape[1], reps=reps)
            comp_bl = t.circuit_complexity(fm0)

            _section(
                f"Baseline: {name} Entanglement "
                f"({n_runs} runs, raw features, reps={reps}, "
                f"mean +/- std [95% CI])"
            )

            _row("val  (aggregated)", agg_bl_val)
            _row("test (aggregated)", agg_bl_test)
            _circuit_row("circuit", comp_bl)
            _section_end()

            baseline_results[key] = {
                "reps": reps,
                "aggregated_val": agg_bl_val,
                "aggregated_test": agg_bl_test,
                "runs": _bl_run_logs[key],
                "complexity": comp_bl,
            }

    else:
        print(
            "\n  [baselines] skipped for this configuration "
            "(will be joined later in post-processing)",
            flush=True,
        )

    full_log["baseline_linear_accs_per_run"] = [
        r["test"]["accuracy"]
        for r in _bl_run_logs["linear"]
    ]

    full_log["baseline_ring_accs_per_run"] = [
        r["test"]["accuracy"]
        for r in _bl_run_logs["ring"]
    ]

    full_log["baseline_full_accs_per_run"] = [
        r["test"]["accuracy"]
        for r in _bl_run_logs["full"]
    ]

    if run_baselines:
        full_log["baselines"] = baseline_results

    if baseline_results:
        full_log["baseline_linear_test_acc_mean"] = (
            baseline_results["linear"]["aggregated_test"].get("accuracy_mean")
        )
        full_log["baseline_ring_test_acc_mean"] = (
            baseline_results["ring"]["aggregated_test"].get("accuracy_mean")
        )
        full_log["baseline_full_test_acc_mean"] = (
            baseline_results["full"]["aggregated_test"].get("accuracy_mean")
        )

    if run_ablation:
        ablation_log = {}

        ab_splits = _make_splits(X_orig, y_orig, first_seed)
        ab_fs = _prepare_fs_views(
            ab_splits,
            use_fs,
            fs_method,
            k,
            first_seed,
        )

        ab_outer_np = ab_fs["outer_train_np"]
        ab_y_out_np = ab_splits["outer_train_y"].to_numpy()
        ab_test_np = ab_fs["test_np"]
        ab_y_te_np = ab_splits["test_y"].to_numpy()

        _section("Ablation: FS-only (all pairs, refit outer_train, split 0)")

        fm_all = t.createFeatureMap(
            ab_fs["pairs_fs"],
            ab_fs["fs_cols"],
            reps=reps,
        )

        _, m_fs_only = evaluate_fn(
            fm_all,
            ab_outer_np,
            ab_y_out_np,
            ab_test_np,
            ab_y_te_np,
            fast_eval=False,
        )

        tst_fs_only = t.compute_metrics(
            m_fs_only,
            ab_test_np,
            ab_y_te_np,
        )

        cls_fs_only = t.compute_metrics_per_class(
            m_fs_only,
            ab_test_np,
            ab_y_te_np,
        )

        cmp_fs_only = t.circuit_complexity(fm_all)

        _row("test", tst_fs_only)
        _class_rows(cls_fs_only)
        _circuit_row("circuit", cmp_fs_only)
        _section_end()

        ablation_log["fs_only"] = {
            "test": tst_fs_only,
            "per_class": cls_fs_only,
            "complexity": cmp_fs_only,
        }

        _section(
            "Ablation: Entanglement-only "
            "(raw features, refit outer_train, split 0)"
        )

        _orig_scores = t.transformCorrelations(
            ab_splits["search_train_X"].corr()
        )

        pairs_orig = t.createCouples(
            _orig_scores,
            ab_splits["search_train_X"].columns,
        )

        warm_start_orig = _build_warm_start(
            _orig_scores,
            activation_ratio=0.30,
        )

        cache_eo = {}
        _cv_eo = ga_cv_folds if metaheuristic == "ga" else 0

        obj_eo = _build_objective(
            pairs_orig,
            ab_splits["search_train_X"].columns,
            ab_splits["search_train_X"].to_numpy(),
            ab_splits["search_train_y"].to_numpy(),
            ab_splits["search_val_X"].to_numpy(),
            ab_splits["search_val_y"].to_numpy(),
            (
                evaluate_fn
                if model_type == "qsvm"
                else __import__("functools").partial(evaluate_fn, fast_eval=True)
            ),
            reps,
            cache_eo,
            cv_folds=_cv_eo,
            objective_metric=objective_metric,
            overfitting_beta=overfitting_beta,
            base_seed=first_seed,
        )

        if metaheuristic == "sa":
            best_eo, _ = optimise(
                obj_eo,
                chromosome_length=len(pairs_orig),
                initial_temp=sa_initial_temp,
                cooling_rate=sa_cooling_rate,
                stopping_temp=sa_stopping_temp,
                max_iterations=5,
                num_neighbors=3,
                seed=first_seed,
            )

        elif metaheuristic == "ts":
            best_eo, _ = optimise(
                obj_eo,
                chromosome_length=len(pairs_orig),
                max_iterations=ts_max_iterations,
                tabu_tenure=ts_tabu_tenure,
                max_no_improve=ts_max_no_improve,
                neighborhood_sample_size=ts_neighborhood_sample_size,
                restart_fraction=ts_restart_fraction,
                seed=first_seed,
            )

        elif metaheuristic == "ils":
            _warm = warm_start_orig if ils_use_warm_start else None

            best_eo, _ = optimise(
                obj_eo,
                chromosome_length=len(pairs_orig),
                initial_solution=_warm,
                warm_start=_warm,
                n_restarts=ils_n_restarts,
                perturbation_strength=ils_perturbation_strength,
                local_search_iters=ils_local_search_iters,
                local_search_initial_temp=ils_local_search_initial_temp,
                local_search_cooling_rate=ils_local_search_cooling_rate,
                local_search_stopping_temp=ils_local_search_stopping_temp,
                local_search_num_neighbors=ils_local_search_num_neighbors,
                repair=ils_repair,
                seed=first_seed,
            )

        else:
            best_eo, _ = optimise(
                obj_eo,
                chromosome_length=len(pairs_orig),
                population_size=ga_population_size,
                n_generations=ga_n_generations,
                crossover_rate=ga_crossover_rate,
                mutation_rate=ga_mutation_rate,
                tournament_size=ga_tournament_size,
                elitism_count=ga_elitism_count,
                seed=first_seed,
            )

        selected_eo = [
            pairs_orig[i]
            for i, b in enumerate(best_eo)
            if b == 1
        ]

        fm_eo = t.createFeatureMap(
            selected_eo,
            ab_splits["search_train_X"].columns,
            reps=reps,
        )

        ab_outer_raw_np = ab_splits["outer_train_X"].to_numpy()
        ab_y_out_raw_np = ab_splits["outer_train_y"].to_numpy()
        ab_test_raw_np = ab_splits["test_X"].to_numpy()
        ab_y_te_raw_np = ab_splits["test_y"].to_numpy()

        _, model_eo = evaluate_fn(
            fm_eo,
            ab_outer_raw_np,
            ab_y_out_raw_np,
            ab_test_raw_np,
            ab_y_te_raw_np,
            fast_eval=False,
        )

        tst_eo = t.compute_metrics(
            model_eo,
            ab_test_raw_np,
            ab_y_te_raw_np,
        )

        cls_eo = t.compute_metrics_per_class(
            model_eo,
            ab_test_raw_np,
            ab_y_te_raw_np,
        )

        cmp_eo = t.circuit_complexity(fm_eo)

        _row("test", tst_eo)
        _class_rows(cls_eo)
        _circuit_row("circuit", cmp_eo)
        _section_end()

        ablation_log["entanglement_only"] = {
            "test": tst_eo,
            "per_class": cls_eo,
            "complexity": cmp_eo,
        }

        full_log["ablation"] = ablation_log

    _header(
        f"SEQUENT ({metaheuristic.upper()}, "
        f"{n_runs} independent runs, sequential)"
    )

    seq_ctx = {
        "X_orig": X_orig,
        "y_orig": y_orig,
        "use_fs": use_fs,
        "fs_method": fs_method,
        "k": k,
        "evaluate_fn": evaluate_fn,
        "reps": reps,
        "base_seed": BASE_SEED,
        "model_type": model_type,
        "metaheuristic": metaheuristic,
        "ga_cv_folds": ga_cv_folds,
        "objective_metric": objective_metric,
        "overfitting_beta": overfitting_beta,
        "sa_initial_temp": sa_initial_temp,
        "sa_cooling_rate": sa_cooling_rate,
        "sa_stopping_temp": sa_stopping_temp,
        "sa_max_iterations": sa_max_iterations,
        "sa_num_neighbors": sa_num_neighbors,
        "ga_population_size": ga_population_size,
        "ga_n_generations": ga_n_generations,
        "ga_crossover_rate": ga_crossover_rate,
        "ga_mutation_rate": ga_mutation_rate,
        "ga_tournament_size": ga_tournament_size,
        "ga_elitism_count": ga_elitism_count,
        "ts_max_iterations": ts_max_iterations,
        "ts_tabu_tenure": ts_tabu_tenure,
        "ts_max_no_improve": ts_max_no_improve,
        "ts_neighborhood_sample_size": ts_neighborhood_sample_size,
        "ts_restart_fraction": ts_restart_fraction,
        "ils_n_restarts": ils_n_restarts,
        "ils_perturbation_strength": ils_perturbation_strength,
        "ils_local_search_iters": ils_local_search_iters,
        "ils_local_search_initial_temp": ils_local_search_initial_temp,
        "ils_local_search_cooling_rate": ils_local_search_cooling_rate,
        "ils_local_search_stopping_temp": ils_local_search_stopping_temp,
        "ils_local_search_num_neighbors": ils_local_search_num_neighbors,
        "ils_use_warm_start": ils_use_warm_start,
        "ils_repair": ils_repair,
    }

    all_test = [None] * n_runs
    all_comp = [None] * n_runs
    all_cls = [None] * n_runs
    run_logs = [None] * n_runs
    svm_accs_per_run = [None] * n_runs
    mlp_accs_per_run = []

    best_acc = -1.0
    best_info = None

    for i in range(n_runs):
        r = _sequent_worker((i, seq_ctx))

        print(f"\n  -- Run {i + 1}/{n_runs}  (seed={r['seed']}) completed")

        _row(f"Run {i + 1} search_val", r["search_val_m"])
        _row(f"Run {i + 1} test", r["test_m"])

        print(f"|  {'':22} SVM (same split): {r['svm_accuracy']:.4f}")

        if r["mlp_accuracy"] is not None:
            print(f"|  {'':22} MLP (same split): {r['mlp_accuracy']:.4f}")

        _class_rows(r["per_cls"])
        _circuit_row(f"Run {i + 1} circuit", r["comp"])

        all_test[i] = r["test_m"]
        all_comp[i] = r["comp"]
        all_cls[i] = r["per_cls"]
        svm_accs_per_run[i] = r["svm_accuracy"]

        if r["mlp_accuracy"] is not None:
            mlp_accs_per_run.append(r["mlp_accuracy"])

        run_logs[i] = {
            "seed": r["seed"],
            "search_val": r["search_val_m"],
            "test": r["test_m"],
            "svm_accuracy": r["svm_accuracy"],
            "mlp_accuracy": r["mlp_accuracy"],
            "per_class": r["per_cls"],
            "complexity": r["comp"],
            "solution": r["solution"],
        }

        if r["test_m"]["accuracy"] > best_acc:
            best_acc = r["test_m"]["accuracy"]
            best_info = {"solution": r["solution"]}

    agg_test = _aggregate(all_test)
    agg_comp = _aggregate(all_comp)

    _section(
        f"SEQUENT - Aggregated "
        f"({n_runs} runs, mean +/- std [95% CI])"
    )

    _row("test", agg_test)
    _circuit_row("complexity", agg_comp)

    if all_cls:
        classes = all_cls[0].keys()
        print(f"|  Per-class test metrics (mean +/- std over {n_runs} runs):")

        for cls in classes:
            f1s = [r[cls]["f1"] for r in all_cls]
            n = len(f1s)
            mu = np.mean(f1s)
            std = np.std(f1s, ddof=1) if n > 1 else 0.0
            ci = float(sp_stats.t.ppf(0.975, df=max(n - 1, 1))) * std / np.sqrt(n)

            print(
                f"|    class {cls:<6}  "
                f"f1:{mu:.4f} +/-{std:.4f} [95%CI +/-{ci:.4f}]"
            )

    _section_end()

    sequent_accs = [r["test"]["accuracy"] for r in run_logs]

    sig_results_linear = {}
    sig_results_ring = {}
    sig_results_full = {}

    if run_baselines:
        for log_key, var_name, label in [
            ("baseline_linear_accs_per_run", "sig_results_linear", "SEQUENT vs Linear"),
            ("baseline_ring_accs_per_run", "sig_results_ring", "SEQUENT vs Ring"),
            ("baseline_full_accs_per_run", "sig_results_full", "SEQUENT vs Full"),
        ]:
            bl_accs = full_log.get(log_key, [])

            if bl_accs:
                result = _significance_tests(sequent_accs, bl_accs)
                _print_significance_block(result, label)

                if var_name == "sig_results_linear":
                    sig_results_linear = result
                elif var_name == "sig_results_ring":
                    sig_results_ring = result
                else:
                    sig_results_full = result
    else:
        print(
            "\n  [stats] quantum baseline significance skipped "
            "because run_baselines=False for this configuration",
            flush=True,
        )

    cmp_linear = _directional_comparison(
        sequent_accs,
        full_log.get("baseline_linear_accs_per_run", []),
    )

    cmp_ring = _directional_comparison(
        sequent_accs,
        full_log.get("baseline_ring_accs_per_run", []),
    )

    cmp_full = _directional_comparison(
        sequent_accs,
        full_log.get("baseline_full_accs_per_run", []),
    )

    sig_results = _significance_tests(sequent_accs, svm_accs_per_run)
    _print_significance_block(sig_results, "SEQUENT vs classical SVM")

    cmp_svm = _directional_comparison(sequent_accs, svm_accs_per_run)

    sig_results_mlp = {}
    cmp_mlp = {}

    if model_type == "qnn" and mlp_accs_per_run:
        sig_results_mlp = _significance_tests(
            sequent_accs,
            mlp_accs_per_run,
        )

        _print_significance_block(sig_results_mlp, "QNN vs classical MLP")
        cmp_mlp = _directional_comparison(sequent_accs, mlp_accs_per_run)

    full_log.update({
        "aggregated_test": agg_test,
        "aggregated_complexity": agg_comp,
        "runs": run_logs,
        "svm_accs_per_run": svm_accs_per_run,
        "mlp_accs_per_run": mlp_accs_per_run if mlp_accs_per_run else None,
        "significance_tests_vs_linear": sig_results_linear if sig_results_linear else None,
        "significance_tests_vs_ring": sig_results_ring if sig_results_ring else None,
        "significance_tests_vs_full": sig_results_full if sig_results_full else None,
        "significance_tests_vs_svm": sig_results,
        "significance_tests_mlp": sig_results_mlp if sig_results_mlp else None,
        "accuracy_comparison_vs_linear": cmp_linear if cmp_linear else None,
        "accuracy_comparison_vs_ring": cmp_ring if cmp_ring else None,
        "accuracy_comparison_vs_full": cmp_full if cmp_full else None,
        "accuracy_comparison_vs_svm": cmp_svm if cmp_svm else None,
        "accuracy_comparison_vs_mlp": cmp_mlp if cmp_mlp else None,
        "best_solution": best_info["solution"] if best_info else None,
    })

    json_path = os.path.join("results2", f"{run_id}.json")
    _save_json(full_log, json_path)

    print(f"\n  Full log -> {json_path}")

    csv_row = {
        "run_id": run_id,
        "dataset": dataset,
        "model_type": model_type,
        "mode": mode,
        "metaheuristic": metaheuristic,
        "fs_method": fs_method if use_fs else "none",
        "k": k if use_fs else X_orig.shape[1],
        "reps": reps,
        "n_runs": n_runs,
        "run_baselines": run_baselines,
        "run_ablation": run_ablation,
        "ga_cv_folds": ga_cv_folds if metaheuristic == "ga" else 0,
        "ils_n_restarts": ils_n_restarts if metaheuristic == "ils" else None,
        "ils_perturbation_strength": (
            ils_perturbation_strength if metaheuristic == "ils" else None
        ),
        "ils_local_search_iters": (
            ils_local_search_iters if metaheuristic == "ils" else None
        ),
        "ils_use_warm_start": (
            ils_use_warm_start if metaheuristic == "ils" else None
        ),
        "ils_repair": ils_repair if metaheuristic == "ils" else None,
        "objective_metric": objective_metric,
        "overfitting_beta": overfitting_beta,
        "use_smote": use_smote,
        "svm_test_acc_split0": svm_m_first["accuracy"],
        "svm_test_f1_split0": svm_m_first["f1_macro"],
        "svm_acc_mean": float(np.mean(svm_accs_per_run)),
        "svm_acc_std": float(np.std(svm_accs_per_run, ddof=1)) if len(svm_accs_per_run) > 1 else 0.0,
        "mlp_acc_mean": float(np.mean(mlp_accs_per_run)) if mlp_accs_per_run else None,
        "mlp_acc_std": float(np.std(mlp_accs_per_run, ddof=1)) if len(mlp_accs_per_run) > 1 else None,
        "linear_test_acc_mean": baseline_results.get("linear", {}).get("aggregated_test", {}).get("accuracy_mean"),
        "linear_test_acc_std": baseline_results.get("linear", {}).get("aggregated_test", {}).get("accuracy_std"),
        "ring_test_acc_mean": baseline_results.get("ring", {}).get("aggregated_test", {}).get("accuracy_mean"),
        "ring_test_acc_std": baseline_results.get("ring", {}).get("aggregated_test", {}).get("accuracy_std"),
        "full_test_acc_mean": baseline_results.get("full", {}).get("aggregated_test", {}).get("accuracy_mean"),
        "full_test_acc_std": baseline_results.get("full", {}).get("aggregated_test", {}).get("accuracy_std"),
        "test_acc_mean": agg_test.get("accuracy_mean", 0),
        "test_acc_std": agg_test.get("accuracy_std", 0),
        "test_acc_ci95": agg_test.get("accuracy_ci95", 0),
        "test_f1_mean": agg_test.get("f1_macro_mean", 0),
        "test_f1_std": agg_test.get("f1_macro_std", 0),
        "test_f1_ci95": agg_test.get("f1_macro_ci95", 0),
        "depth_mean": agg_comp.get("depth_mean", 0),
        "two_qubit_gates_mean": agg_comp.get("two_qubit_gates_mean", 0),
        "train_time_mean": agg_test.get("training_time_mean", 0),
        "linear_perm_p": sig_results_linear.get("permutation_p", float("nan")) if sig_results_linear else None,
        "linear_perm_sig_05": sig_results_linear.get("permutation_sig_05", False) if sig_results_linear else None,
        "linear_sign_p": sig_results_linear.get("sign_p", float("nan")) if sig_results_linear else None,
        "linear_sign_sig_05": sig_results_linear.get("sign_sig_05", False) if sig_results_linear else None,
        "linear_ttest_p": sig_results_linear.get("ttest_p", float("nan")) if sig_results_linear else None,
        "linear_ttest_sig_05": sig_results_linear.get("ttest_sig_05", False) if sig_results_linear else None,
        "sequent_gt_linear_wins": cmp_linear.get("wins") if cmp_linear else None,
        "sequent_gt_linear_losses": cmp_linear.get("losses") if cmp_linear else None,
        "sequent_gt_linear_ties": cmp_linear.get("ties") if cmp_linear else None,
        "sequent_minus_linear_acc_mean": cmp_linear.get("mean_diff") if cmp_linear else None,
        "ring_perm_p": sig_results_ring.get("permutation_p", float("nan")) if sig_results_ring else None,
        "ring_perm_sig_05": sig_results_ring.get("permutation_sig_05", False) if sig_results_ring else None,
        "ring_sign_p": sig_results_ring.get("sign_p", float("nan")) if sig_results_ring else None,
        "ring_sign_sig_05": sig_results_ring.get("sign_sig_05", False) if sig_results_ring else None,
        "ring_ttest_p": sig_results_ring.get("ttest_p", float("nan")) if sig_results_ring else None,
        "ring_ttest_sig_05": sig_results_ring.get("ttest_sig_05", False) if sig_results_ring else None,
        "sequent_gt_ring_wins": cmp_ring.get("wins") if cmp_ring else None,
        "sequent_gt_ring_losses": cmp_ring.get("losses") if cmp_ring else None,
        "sequent_gt_ring_ties": cmp_ring.get("ties") if cmp_ring else None,
        "sequent_minus_ring_acc_mean": cmp_ring.get("mean_diff") if cmp_ring else None,
        "full_perm_p": sig_results_full.get("permutation_p", float("nan")) if sig_results_full else None,
        "full_perm_sig_05": sig_results_full.get("permutation_sig_05", False) if sig_results_full else None,
        "full_sign_p": sig_results_full.get("sign_p", float("nan")) if sig_results_full else None,
        "full_sign_sig_05": sig_results_full.get("sign_sig_05", False) if sig_results_full else None,
        "full_ttest_p": sig_results_full.get("ttest_p", float("nan")) if sig_results_full else None,
        "full_ttest_sig_05": sig_results_full.get("ttest_sig_05", False) if sig_results_full else None,
        "sequent_gt_full_wins": cmp_full.get("wins") if cmp_full else None,
        "sequent_gt_full_losses": cmp_full.get("losses") if cmp_full else None,
        "sequent_gt_full_ties": cmp_full.get("ties") if cmp_full else None,
        "sequent_minus_full_acc_mean": cmp_full.get("mean_diff") if cmp_full else None,
        "svm_perm_p": sig_results.get("permutation_p", float("nan")),
        "svm_perm_sig_05": sig_results.get("permutation_sig_05", False),
        "svm_sign_p": sig_results.get("sign_p", float("nan")),
        "svm_sign_sig_05": sig_results.get("sign_sig_05", False),
        "svm_ttest_p": sig_results.get("ttest_p", float("nan")),
        "svm_ttest_sig_05": sig_results.get("ttest_sig_05", False),
        "sequent_gt_svm_wins": cmp_svm.get("wins") if cmp_svm else None,
        "sequent_gt_svm_losses": cmp_svm.get("losses") if cmp_svm else None,
        "sequent_gt_svm_ties": cmp_svm.get("ties") if cmp_svm else None,
        "sequent_minus_svm_acc_mean": cmp_svm.get("mean_diff") if cmp_svm else None,
        "mlp_sign_p": sig_results_mlp.get("sign_p", float("nan")) if sig_results_mlp else None,
        "mlp_sign_sig_05": sig_results_mlp.get("sign_sig_05", False) if sig_results_mlp else None,
        "mlp_ttest_p": sig_results_mlp.get("ttest_p", float("nan")) if sig_results_mlp else None,
        "mlp_ttest_sig_05": sig_results_mlp.get("ttest_sig_05", False) if sig_results_mlp else None,
        "sequent_gt_mlp_wins": cmp_mlp.get("wins") if cmp_mlp else None,
        "sequent_gt_mlp_losses": cmp_mlp.get("losses") if cmp_mlp else None,
        "sequent_gt_mlp_ties": cmp_mlp.get("ties") if cmp_mlp else None,
        "sequent_minus_mlp_acc_mean": cmp_mlp.get("mean_diff") if cmp_mlp else None,
        "timestamp": timestamp,
    }

    _append_csv(csv_row)
    print(f"  Summary -> {_CSV_PATH}")
    return full_log


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    # _ALL_DATASETS = [
    #     ("breast-w", 0, "./datasets/breast-w.tsv"),
    #     ("corral", 1, None),
    #     ("fitness_class_2212", 0, "./datasets/fitness_class_2212.csv"),
    #     ("flare", 0, "./datasets/flare.tsv"),
    #     ("heart", 0, "./datasets/heart.csv"),
    #     ("credit", 0, "./datasets/Credit_Card_Applications.csv"),
    #     ("inosphere", 0, "./datasets/inosphere.csv"),
    # ]

    _ALL_DATASETS = [
       ("flare", 0, "./datasets/flare.tsv"),
    ]

    _ds_filter = os.environ.get("SEQUENT_DATASET", "").strip()

    DATASETS = (
        [d for d in _ALL_DATASETS if d[0] == _ds_filter]
        if _ds_filter
        else _ALL_DATASETS
    )

    _IMBALANCE_THRESHOLD = 1.5

    def _imbalance_ratio(y):
        from collections import Counter
        counts = Counter(y)
        return max(counts.values()) / min(counts.values())

    _MODELS = ["qsvm", "qnn"]
    _MHS = ["sa", "ils"]
    _FS = ["anova", "autoencoder"]
    _REPS = [1]

    def _make_configs(imbalanced=False, mode="statevector"):
        """
        Build the experiment grid.

        mode is forced to noise regardless of the argument passed.

        Quantum baselines are computed only for canonical configurations:
            metaheuristic == "sa" and fs_method == "anova"

        gpu=0 is included in every config so the noise simulator uses the
        GPU allocated by SLURM via CUDA_VISIBLE_DEVICES.
        """
        mode = "noise"
        is_noise = mode == "noise"

        configs = []

        for _model in _MODELS:
            for _mh in _MHS:
                for _fs in _FS:
                    for _reps in _REPS:
                        configs.append(dict(
                            model_type=_model,
                            mode=mode,
                            metaheuristic=_mh,
                            use_fs=True,
                            fs_method=_fs,
                            k=5,
                            reps=_reps,
                            n_runs=1,
                            sa_max_iterations=5,
                            sa_num_neighbors=3,
                            ils_local_search_iters=5,
                            ils_n_restarts=3,
                            run_baselines=True,
                            run_ablation=True,
                            objective_metric=(
                                "f1_macro"
                                if imbalanced
                                else "accuracy"
                            ),
                            gpu=None,
                        ))

        return configs

    def _already_done(dataset, cfg):
        if not os.path.exists(_CSV_PATH):
            return False

        try:
            df_done = pd.read_csv(_CSV_PATH)

            mask = (
                (df_done["dataset"] == dataset)
                & (df_done["model_type"] == cfg["model_type"])
                & (df_done["mode"] == cfg["mode"])
                & (df_done["metaheuristic"] == cfg["metaheuristic"])
                & (df_done["fs_method"] == cfg["fs_method"])
                & (df_done["reps"] == cfg["reps"])
            )

            return mask.any()

        except Exception:
            return False

    all_jobs = []

    for ds, opt, p in DATASETS:
        try:
            _, _y_ds = t.load_data(path=p, option=opt, dataset=ds)

        except Exception as _e:
            print(
                f"  [WARN] Could not pre-load {ds}: {_e}. "
                f"Defaulting to accuracy."
            )
            _y_ds = None

        if _y_ds is not None:
            _ir = _imbalance_ratio(_y_ds)
            _ds_imbalanced = _ir > _IMBALANCE_THRESHOLD

            print(
                f"  [{ds}]  IR = {_ir:.2f}  ->  "
                f"{'f1_macro (imbalanced)' if _ds_imbalanced else 'accuracy  (balanced)'}"
            )

        else:
            _ds_imbalanced = False

        _ds_configs = _make_configs(
            imbalanced=_ds_imbalanced,
            mode=os.environ.get("SEQUENT_MODE", "statevector").strip(),
        )

        if ds == DATASETS[0][0]:
            print(
                f"\n  Grid: {len(_ds_configs)} configs x {len(DATASETS)} datasets "
                f"= {len(_ds_configs) * len(DATASETS)} experiments\n"
            )

            print(
                f"  {'idx':<4} {'dataset':<10} {'model':<6} {'mh':<4} "
                f"{'fs':<12} {'reps':<5} {'baseline':<9} {'ablation'}"
            )
            print(f"  {'-' * 72}")

        for cfg in _ds_configs:
            job_idx = len(all_jobs)
            all_jobs.append((job_idx, ds, opt, p, cfg))

            if ds == DATASETS[0][0]:
                print(
                    f"  {job_idx:<4} {ds:<10} {cfg['model_type']:<6} "
                    f"{cfg['metaheuristic']:<4} {cfg['fs_method']:<12} "
                    f"{cfg['reps']:<5} {str(cfg['run_baselines']):<9} "
                    f"{cfg['run_ablation']}"
                )

    if all_jobs:
        print()

    # -------------------------------------------------------------------------
    # CAMBIO 3: leer rango de índices SEQUENT_JOB_START / SEQUENT_JOB_END
    # -------------------------------------------------------------------------
    _job_start_env = os.environ.get("SEQUENT_JOB_START", "").strip()
    _job_end_env   = os.environ.get("SEQUENT_JOB_END",   "").strip()

    if _job_start_env and _job_end_env:
        try:
            _start = int(_job_start_env)
            _end   = int(_job_end_env)   # inclusivo
        except ValueError as exc:
            raise ValueError(
                f"SEQUENT_JOB_START/END must be integers, "
                f"got '{_job_start_env}' / '{_job_end_env}'"
            ) from exc

        if _start < 0 or _end >= len(all_jobs) or _start > _end:
            raise IndexError(
                f"SEQUENT_JOB_START={_start} / SEQUENT_JOB_END={_end} "
                f"fuera de rango [0, {len(all_jobs) - 1}]"
            )

        jobs_to_run = [j for j in all_jobs if _start <= j[0] <= _end]
        print(f"  Running jobs {_start}–{_end} ({len(jobs_to_run)} configs)")

    else:
        jobs_to_run = all_jobs
        print(f"  Running full grid: {len(jobs_to_run)} jobs")

    for job_idx, ds, opt, p, cfg in jobs_to_run:
        print(
            f"\n  [JOB {job_idx}] {ds} / {cfg['model_type']} / "
            f"{cfg['metaheuristic']} / {cfg['fs_method']} / reps={cfg['reps']}"
        )

        if _already_done(ds, cfg):
            print("  [SKIP] already in CSV")
            continue

        try:
            run_experiment(
                dataset=ds,
                option=opt,
                path=p,
                **cfg,
            )

        except Exception as e:
            print(
                f"\n  [ERROR] job={job_idx}  "
                f"{ds} / {cfg['model_type']} / {cfg['mode']} : {e}"
            )

            import traceback
            traceback.print_exc()