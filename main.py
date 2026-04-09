"""
main.py — SEQUENT Experiment Runner
====================================
Covers all experimental tasks from the reviewer revision plan:

  R1-02  Three entanglement baselines: linear, ring, full
  R1-03  Multi-seed runs (n_runs) → mean ± std
  R1-07  Deep ablation: FS-only, Entanglement-only, FS+Search (SEQUENT)
  R1-08  Circuit complexity printed and saved
  R2-10  Per-class precision / recall / F1 (class imbalance analysis)
  R2-12  Configurable feature-map reps
  R2-13  ANOVA vs mutual_info FS methods

  CLASSICAL MLP BASELINE (QNN counterpart):
  ──────────────────────────────────────────
  When model_type="qnn", a shallow classical MLP (64→32 units, same depth as
  the quantum circuit) is evaluated on every split alongside the SVM.  This
  lets the paper answer "does the QNN beat not only SVM but also a classical
  network of comparable size?" — a common reviewer request for quantum ML work.

  STATISTICAL RIGOR (reviewer comment 3):
  ─────────────────────────────────────────
  Each run i uses a DIFFERENT train/val/test split (random_state = BASE_SEED + i).
  This makes the n_runs repeats analogous to repeated random sub-sampling, so
  the reported mean ± std / 95 % CI reflects model uncertainty over unseen data,
  not just metaheuristic variance.

  After n_runs completions, a one-sample Wilcoxon signed-rank test checks whether
  SEQUENT test accuracy is significantly above the single-split SVM baseline.
  For a paired comparison (SEQUENT vs SVM both evaluated on the same n_runs
  splits), a paired Wilcoxon / t-test is also computed when n_runs ≥ 5.

QSVM speed guide (to avoid slow metaheuristic runs)
────────────────────────────────────────────────────
  C=1000, num_steps=100  →  ~5× faster than C=5000/steps=500
  SA: initial_temp=1.0, max_iterations=10, num_neighbors=3  → ~30 evals/run
  GA: population_size=10, n_generations=5                   → ~30-50 evals/run
  Statevector: each eval ≈ 3-8 s  →  total ≈ 1.5-4 min/run

Output
──────
  Console : colour-free ASCII tables, one block per section
  CSV     : results/benchmark_results.csv  — appended after each run
  JSON    : results/{dataset}_{model}_{mode}_{mh}_{ts}.json — full detail
"""

import os
import time
import json
import random
import datetime
import numpy as np
import pandas as pd
import scipy.stats as sp_stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from qiskit_algorithms.utils import algorithm_globals

import tools as t
from metaheuristicas import simulated_annealing, genetic_algorithm

# ─── Base seed (partitions are BASE_SEED + run_idx) ───────────────────────────
BASE_SEED = 12345
random.seed(BASE_SEED)
np.random.seed(BASE_SEED)
algorithm_globals.random_seed = BASE_SEED

# ─── Model/mode dispatch ──────────────────────────────────────────────────────
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
    "sa": simulated_annealing,
    "ga": genetic_algorithm,
}

os.makedirs("results", exist_ok=True)
_CSV_PATH = os.path.join("results", "benchmark_results.csv")


# ═══════════════════════════════════════════════════════════════
#  OUTPUT HELPERS
# ═══════════════════════════════════════════════════════════════

def _header(text, width=70):
    bar = "═" * width
    print(f"\n╔{bar}╗")
    lines = [text[i:i+width] for i in range(0, len(text), width)]
    for line in lines:
        print(f"║  {line:<{width-2}}║")
    print(f"╚{bar}╝")

def _section(title):
    print(f"\n┌── {title} {'─'*max(0, 65-len(title))}┐")

def _section_end():
    print(f"└{'─'*69}┘")

def _row(label, metrics):
    """Print a single labelled metrics row inside a section."""
    acc  = metrics.get("accuracy",        metrics.get("accuracy_mean",        0))
    prec = metrics.get("precision_macro", metrics.get("precision_macro_mean",  0))
    rec  = metrics.get("recall_macro",    metrics.get("recall_macro_mean",     0))
    f1   = metrics.get("f1_macro",        metrics.get("f1_macro_mean",         0))
    ttime = metrics.get("training_time",  metrics.get("training_time_mean",    0))
    itime = metrics.get("inference_time", metrics.get("inference_time_mean",   0))

    std_acc = metrics.get("accuracy_std", None)
    std_f1  = metrics.get("f1_macro_std", None)
    ci_acc  = metrics.get("accuracy_ci95", None)
    ci_f1   = metrics.get("f1_macro_ci95", None)

    acc_str = f"{acc:.4f}"
    if std_acc is not None:
        acc_str += f" ±{std_acc:.4f}"
    if ci_acc is not None:
        acc_str += f" [95%CI ±{ci_acc:.4f}]"

    f1_str = f"{f1:.4f}"
    if std_f1 is not None:
        f1_str += f" ±{std_f1:.4f}"
    if ci_f1 is not None:
        f1_str += f" [95%CI ±{ci_f1:.4f}]"

    print(f"│  {label:<22} Acc:{acc_str:<30} Prec:{prec:.4f}  Rec:{rec:.4f}  F1:{f1_str}")
    if ttime > 0:
        inf_str = f"{itime:.3f}s" if itime > 0 else "—"
        print(f"│  {'':22} Train:{ttime:.2f}s  Infer:{inf_str}")

def _circuit_row(label, complexity):
    d   = complexity.get("depth", complexity.get("depth_mean", 0))
    tg  = complexity.get("total_gates", complexity.get("total_gates_mean", 0))
    tq  = complexity.get("two_qubit_gates", complexity.get("two_qubit_gates_mean", 0))
    ss  = complexity.get("search_space", "—")
    print(f"│  {label:<22} depth:{d}  total_gates:{tg}  2Q-gates:{tq}  search_space:{ss}")

def _class_rows(per_class):
    for cls, m in per_class.items():
        print(f"│    class {cls:<6}  prec:{m['precision']:.4f}  "
              f"rec:{m['recall']:.4f}  f1:{m['f1']:.4f}  n={m['support']}")


# ═══════════════════════════════════════════════════════════════
#  STATS HELPERS
# ═══════════════════════════════════════════════════════════════

def _aggregate(results_list):
    """
    Mean ± std ± 95 % CI for every numeric key across multiple runs.

    The 95 % CI uses the t-distribution with (n-1) degrees of freedom,
    which is appropriate for small samples (n_runs typically 5–10).
    """
    if not results_list:
        return {}
    keys = results_list[0].keys()
    out  = {}
    for k in keys:
        vals = [r[k] for r in results_list
                if isinstance(r[k], (int, float)) and not np.isnan(r[k])]
        if vals:
            n   = len(vals)
            mu  = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))   # sample std (ddof=1)
            sem = std / np.sqrt(n)
            # t critical value at 97.5th percentile, df = n-1
            t_crit = float(sp_stats.t.ppf(0.975, df=max(n - 1, 1)))
            ci95   = float(t_crit * sem)
            out[f"{k}_mean"] = mu
            out[f"{k}_std"]  = std
            out[f"{k}_ci95"] = ci95
    return out


def _significance_tests(sequent_accs, baseline_accs):
    """
    Paired statistical tests comparing SEQUENT vs a baseline across n_runs.

    Test battery (all one-sided, H1: SEQUENT > baseline):
    ─────────────────────────────────────────────────────
    1. Permutation test (primary)
       Exact, non-parametric, no minimum sample size. Works by enumerating
       all 2^n sign assignments of the paired differences and computing the
       fraction that are >= the observed sum. This is the recommended test
       for small-n comparisons in ML (Demšar, JMLR 2006).

    2. Sign test (fallback / complementary)
       Counts how many runs SEQUENT wins. Exact binomial p-value under H0
       that each run is a coin flip. Robust to ties (tied runs are dropped).
       Very conservative but valid at any n.

    3. Paired t-test (reported for completeness / journal convention)
       Parametric; normality assumption is unlikely to hold for n<10, so
       use with caution. Included because many reviewers expect to see it.

    Parameters
    ----------
    sequent_accs   : list[float]  — SEQUENT test accuracy per run.
    baseline_accs  : list[float]  — Baseline test accuracy on the same splits.

    Returns
    -------
    dict with all test statistics and boolean significance flags.
    """
    n = len(sequent_accs)
    results = {"n_runs": n}

    if n < 2:
        results["note"] = "Need n_runs >= 2 for significance tests"
        return results

    a = np.array(sequent_accs,  dtype=float)
    b = np.array(baseline_accs, dtype=float)
    diffs = a - b

    # ── 1. Permutation test (exact, one-sided) ────────────────────────────
    try:
        observed_stat = float(np.sum(diffs))
        # Enumerate all 2^n sign flips
        count_ge = 0
        total    = 0
        for signs in range(1 << n):
            sign_vec = np.array(
                [1 if (signs >> i) & 1 else -1 for i in range(n)],
                dtype=float)
            perm_stat = float(np.sum(np.abs(diffs) * sign_vec))
            if perm_stat >= observed_stat:
                count_ge += 1
            total += 1
        p_perm = count_ge / total
        results["permutation_stat"]   = observed_stat
        results["permutation_p"]      = float(p_perm)
        results["permutation_sig_05"] = bool(p_perm < 0.05)
    except Exception as e:
        results["permutation_error"] = str(e)

    # ── 2. Sign test (exact binomial, one-sided) ──────────────────────────
    try:
        wins   = int(np.sum(diffs > 0))
        losses = int(np.sum(diffs < 0))
        n_tied = int(np.sum(diffs == 0))
        n_eff  = wins + losses          # drop ties
        if n_eff > 0:
            # P(X >= wins) under Binomial(n_eff, 0.5)
            p_sign = float(sp_stats.binom_test(wins, n_eff, 0.5,
                                               alternative="greater"))
        else:
            p_sign = 1.0                # all ties → no evidence
        results["sign_wins"]    = wins
        results["sign_losses"]  = losses
        results["sign_ties"]    = n_tied
        results["sign_p"]       = p_sign
        results["sign_sig_05"]  = bool(p_sign < 0.05)
    except Exception as e:
        # binom_test deprecated in scipy >=1.11 → use binomtest
        try:
            bt = sp_stats.binomtest(wins, n_eff, 0.5, alternative="greater")
            p_sign = float(bt.pvalue)
            results["sign_wins"]    = wins
            results["sign_losses"]  = losses
            results["sign_ties"]    = n_tied
            results["sign_p"]       = p_sign
            results["sign_sig_05"]  = bool(p_sign < 0.05)
        except Exception as e2:
            results["sign_error"] = str(e2)

    # ── 3. Paired t-test (parametric, for completeness) ───────────────────
    try:
        stat_t, p_t_two = sp_stats.ttest_rel(a, b)
        p_t = float(p_t_two / 2) if stat_t > 0 else 1.0   # one-sided
        results["ttest_stat"]    = float(stat_t)
        results["ttest_p"]       = p_t
        results["ttest_sig_05"]  = bool(p_t < 0.05)
    except Exception as e:
        results["ttest_error"] = str(e)

    return results


def _print_significance_block(sig_dict, label):
    _section(f"Statistical Significance  ({label})")
    n = sig_dict.get("n_runs", 0)
    if "note" in sig_dict:
        print(f"│  {sig_dict['note']}")
        _section_end()
        return

    print(f"│  n_runs = {n}  (one-sided tests, H1: SEQUENT > baseline)")

    # Permutation test
    if "permutation_p" in sig_dict:
        p   = sig_dict["permutation_p"]
        sig = sig_dict["permutation_sig_05"]
        print(f"│  Permutation test  (primary):    "
              f"p = {p:.4f}  {'*** p<0.05' if sig else 'not significant'}")

    # Sign test
    if "sign_p" in sig_dict:
        p    = sig_dict["sign_p"]
        sig  = sig_dict["sign_sig_05"]
        wins = sig_dict.get("sign_wins", "?")
        ties = sig_dict.get("sign_ties", 0)
        print(f"│  Sign test         (robust):     "
              f"p = {p:.4f}  {'*** p<0.05' if sig else 'not significant'}"
              f"  (wins={wins}, ties={ties})")

    # t-test
    if "ttest_p" in sig_dict:
        p   = sig_dict["ttest_p"]
        sig = sig_dict["ttest_sig_05"]
        print(f"│  Paired t-test     (parametric): "
              f"p = {p:.4f}  {'*** p<0.05' if sig else 'not significant'}")

    _section_end()


# ═══════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION FACTORY
# ═══════════════════════════════════════════════════════════════

def _build_objective(couples_array, columns, tr, y_tr, val, y_val,
                     evaluate_fn, reps, cache, cv_folds=0):
    """
    Returns a memoised f(vector) -> (cost, training_time).
    cost = -accuracy so optimisers that minimise cost maximise accuracy.

    Parameters
    ----------
    cv_folds : int
        0  → use the single (tr, val) split as before.  Fast; used for SA.
        >0 → use stratified k-fold CV on tr+val combined.  Slower (k× evals
             per solution) but prevents overfitting to a small val split.
             Recommended for the GA, which has high selection pressure and
             converges quickly to solutions that overfit a single val set.
    """
    def objective(vector):
        key = str(vector)
        if key in cache:
            return cache[key]["cost"], cache[key]["aux"]

        selected = [couples_array[i] for i, b in enumerate(vector) if b == 1]
        fm       = t.createFeatureMap(selected, columns, reps=reps)

        if cv_folds < 2:
            # ── Original behaviour: single val split ──────────────────────
            metrics, model = evaluate_fn(fm, tr, y_tr, val, y_val)
            cost = -metrics["accuracy"]
            cache[key] = {
                "solution":    vector[:],
                "cost":        cost,
                "aux":         metrics["training_time"],
                "metrics":     metrics,
                "model":       model,
                "feature_map": fm,
            }
        else:
            # ── k-fold CV on train+val combined ───────────────────────────
            # Merge tr and val so the CV uses all available labelled data.
            X_all = np.vstack([tr, val])
            y_all = np.concatenate([y_tr, y_val])
            skf   = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                    random_state=42)
            fold_accs  = []
            fold_precs = []
            fold_recs  = []
            fold_f1s   = []
            last_model = None
            t_start = time.time()
            for fold_tr_idx, fold_val_idx in skf.split(X_all, y_all):
                fold_m, fold_model = evaluate_fn(
                    fm,
                    X_all[fold_tr_idx], y_all[fold_tr_idx],
                    X_all[fold_val_idx], y_all[fold_val_idx],
                )
                fold_accs.append(fold_m["accuracy"])
                fold_precs.append(fold_m.get("precision_macro", float("nan")))
                fold_recs.append(fold_m.get("recall_macro",    float("nan")))
                fold_f1s.append(fold_m.get("f1_macro",         float("nan")))
                last_model = fold_model
            cv_acc  = float(np.mean(fold_accs))
            cv_std  = float(np.std(fold_accs))
            elapsed = time.time() - t_start
            cost    = -cv_acc
            # Synthetic metrics dict averaging all folds — keeps the rest of
            # the pipeline (display, aggregation) working unchanged.
            metrics = {
                "accuracy":        cv_acc,
                "accuracy_std_cv": cv_std,
                "precision_macro": float(np.nanmean(fold_precs)),
                "recall_macro":    float(np.nanmean(fold_recs)),
                "f1_macro":        float(np.nanmean(fold_f1s)),
                "inference_time":  0.0,
                "training_time":   elapsed,
            }
            cache[key] = {
                "solution":    vector[:],
                "cost":        cost,
                "aux":         elapsed,
                "metrics":     metrics,
                "model":       last_model,
                "feature_map": fm,
            }

        return cache[key]["cost"], cache[key]["aux"]
    return objective


# ═══════════════════════════════════════════════════════════════
#  SAVING HELPERS
# ═══════════════════════════════════════════════════════════════

def _save_json(data, path):
    """Save dict to JSON, skipping non-serialisable objects."""
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        return str(obj)   # models, circuits → skip gracefully
    with open(path, "w") as f:
        json.dump(_clean(data), f, indent=2)

def _append_csv(row: dict):
    """Append a single result row to the benchmark CSV (creates if missing)."""
    df_new = pd.DataFrame([row])
    if os.path.exists(_CSV_PATH):
        df_new.to_csv(_CSV_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(_CSV_PATH, mode="w", header=True, index=False)


# ═══════════════════════════════════════════════════════════════
#  MAIN EXPERIMENT FUNCTION
# ═══════════════════════════════════════════════════════════════

def run_experiment(
    dataset,
    option=1,
    path=None,
    # ── Model ─────────────────────────────────
    model_type="qsvm",          # "qsvm" | "qnn"
    mode="statevector",         # "statevector" | "noise" | "hardware"
    reps=1,                     # feature-map repetitions         (R2-12)
    # ── Feature selection ─────────────────────
    use_fs=True,
    fs_method="anova",          # "anova" | "mutual_info"         (R2-13)
    k=5,
    # ── Optimisation ──────────────────────────
    metaheuristic="sa",         # "sa" | "ga"
    # SA hyperparameters
    sa_initial_temp=10.0,       # T0 raised: ~10× typical Δcost so the search
                                # can escape local optima early. T0=1.0 was
                                # too cold — the SA froze by iteration 3.
    sa_cooling_rate=0.90,       # Slightly faster cooling to compensate for
                                # the higher T0 within the iteration budget.
    sa_stopping_temp=0.01,
    sa_max_iterations=20,       # 20 iters × 5 neighbors = 100 evals — still
    sa_num_neighbors=5,         # fast on statevector (~8 min/run).
    # GA hyperparameters
    ga_population_size=20,      # Larger population reduces premature convergence
    ga_n_generations=10,        # More generations to compensate larger pop
    ga_crossover_rate=0.8,
    ga_mutation_rate=None,      # Defaults to 1/chromosome_length in GA code
    ga_tournament_size=2,       # Lower pressure (was 3) → more diversity,
                                # less risk of converging to a val-overfit
    ga_elitism_count=2,         # Keep top-2 so good solutions aren't lost
    ga_cv_folds=3,              # Internal CV folds for GA objective function.
                                # 0 = use single val split (original behaviour).
                                # >0 = use k-fold CV on train+val combined,
                                # which prevents overfitting to the val split.
    # ── Statistical rigor ─────────────────────
    n_runs=5,                   # R1-03 + reviewer comment 3: multiple seeds
    # ── Scope ─────────────────────────────────
    run_baselines=True,         # linear / ring / full maps        (R1-02)
    run_ablation=True,          # FS-only, Ent-only                (R1-07)
):
    """
    Full SEQUENT experiment.  See module docstring for parameter guide.

    Statistical design
    ------------------
    Each of the n_runs repetitions uses seed = BASE_SEED + run_idx for BOTH
    the train/val/test partition AND the metaheuristic.  This means every run
    sees a genuinely different data split, making mean ± std / CI estimates
    reflect uncertainty over unseen data rather than only metaheuristic variance.

    After all runs, a paired Wilcoxon signed-rank test (and a paired t-test)
    compare SEQUENT test accuracy against the SVM baseline evaluated on the
    same splits.  Results are printed, saved to JSON and appended to the CSV.
    """
    if model_type not in _EVALUATE:
        raise ValueError(f"model_type must be 'qsvm' or 'qnn'; got '{model_type}'")
    if mode not in _EVALUATE[model_type]:
        raise ValueError(f"mode must be 'statevector'|'noise'|'hardware'; got '{mode}'")
    if metaheuristic not in _OPTIMISER:
        raise ValueError(f"metaheuristic must be 'sa' or 'ga'; got '{metaheuristic}'")

    evaluate_fn = _EVALUATE[model_type][mode]
    optimise    = _OPTIMISER[metaheuristic]
    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id      = f"{dataset}_{model_type}_{mode}_{metaheuristic}_{timestamp}"

    _header(f"SEQUENT  |  {dataset}  |  {model_type.upper()}  |  {mode}  "
            f"|  {metaheuristic.upper()}  |  FS:{use_fs}({fs_method},k={k})  "
            f"|  reps={reps}  |  runs={n_runs}")

    full_log = {
        "run_id": run_id, "dataset": dataset, "model_type": model_type,
        "mode": mode, "metaheuristic": metaheuristic, "use_fs": use_fs,
        "fs_method": fs_method, "k": k, "reps": reps, "n_runs": n_runs,
        "ga_cv_folds": ga_cv_folds if metaheuristic == "ga" else 0,
        "base_seed": BASE_SEED,
        "split_seeds": list(range(BASE_SEED, BASE_SEED + n_runs)),
    }

    # ── 1. Load raw data ───────────────────────────────────────────────────────
    X_orig, y_orig = t.load_data(path=path, option=option, dataset=dataset)
    print(f"\n  Dataset: {X_orig.shape[0]} samples × {X_orig.shape[1]} features")
    print(f"  Class distribution: {dict(y_orig.value_counts().sort_index())}")
    print(f"  Split seeds: {BASE_SEED} … {BASE_SEED + n_runs - 1}  "
          f"(one distinct partition per run)")

    # ── 2. Feature selection (done once on the full dataset) ──────────────────
    # FS is applied to the full X/y before any split so that the selected
    # feature subset is fixed across runs; only the train/val/test partition
    # changes.  This mirrors the typical ML pipeline where the feature
    # engineering step is frozen and cross-validation varies the split.
    if use_fs:
        print(f"\n  Applying FS: method={fs_method}, k={k}")
        X_fs, fs_cols = t.apply_feature_selection(X_orig, y_orig,
                                                   method=fs_method, k=k)
    else:
        X_fs, fs_cols = X_orig, X_orig.columns

    pairs_fs = t.createCouples(
        t.transformCorrelations(X_fs.corr()), X_fs.columns)

    # ── 3. Baselines and ablation on the FIRST split only ────────────────────
    # Baselines (linear/ring/full) and ablation components are deterministic
    # given the data, so running them on the first split is representative.
    # Their purpose is a qualitative comparison, not a statistical estimate.
    first_seed = BASE_SEED  # split seed for the baseline/ablation split

    X_tr_o, X_te_o, y_tr_o, y_te_o = t.splitData(
        X_orig, y_orig, random_state=first_seed)
    X_tr2_o, X_v_o, y_tr2_o, y_v_o = train_test_split(
        X_tr_o, y_tr_o, test_size=0.2, random_state=first_seed)
    tr_o   = X_tr2_o.to_numpy();  y_tr_o_np  = y_tr2_o.to_numpy()
    val_o  = X_v_o.to_numpy();    y_val_o_np = y_v_o.to_numpy()
    test_o = X_te_o.to_numpy();   y_te_o_np  = y_te_o.to_numpy()

    # ── 4. Classical baselines (first split) ──────────────────────────────────
    _section("Baseline: Classical RBF-SVM  (raw features, test set, split 0)")
    svm_m_first, svm_model_first = t.evaluate_classical_svm(
        tr_o, y_tr_o_np, test_o, y_te_o_np)
    _row("SVM test", svm_m_first)
    _section_end()
    full_log["baseline_svm_split0"] = svm_m_first

    # Classical MLP — evaluated only when model_type="qnn" so the paper has
    # a direct classical-vs-quantum comparison for the neural network track.
    mlp_m_first = None
    if model_type == "qnn":
        _section("Baseline: Classical MLP  (raw features, test set, split 0)")
        mlp_m_first, _ = t.evaluate_classical_mlp(
            tr_o, y_tr_o_np, test_o, y_te_o_np, seed=first_seed)
        _row("MLP test", mlp_m_first)
        _section_end()
        full_log["baseline_mlp_split0"] = mlp_m_first

    # ── 5. Quantum entanglement baselines (all n_runs splits) ────────────────
    # Running the baselines on the same n_runs splits as SEQUENT serves two
    # purposes:
    #   (a) Their mean ± std is directly comparable to SEQUENT's aggregated
    #       metrics (same data partitions, apples-to-apples).
    #   (b) The per-run accuracies are used for the paired Wilcoxon / t-test
    #       against SEQUENT, which is the statistically meaningful comparison
    #       in a quantum ML paper (not SEQUENT vs classical SVM).
    # The best baseline for significance testing is Linear, as it is the most
    # common reference in the QML feature-map literature.
    if run_baselines:
        baseline_results  = {}
        # Accumulators for the n_runs loop — keyed by baseline name
        _bl_all_test  = {"linear": [], "ring": [], "full": []}
        _bl_run_logs  = {"linear": [], "ring": [], "full": []}

        for run_idx_bl in range(n_runs):
            seed_bl = BASE_SEED + run_idx_bl
            # Raw-feature split for this run (same seeds as SEQUENT loop)
            X_tr_bl, X_te_bl, y_tr_bl, y_te_bl = t.splitData(
                X_orig, y_orig, random_state=seed_bl)
            X_tr2_bl, X_v_bl, y_tr2_bl, y_v_bl = train_test_split(
                X_tr_bl, y_tr_bl, test_size=0.2, random_state=seed_bl)
            tr_bl   = X_tr2_bl.to_numpy();  y_tr_bl_np  = y_tr2_bl.to_numpy()
            val_bl  = X_v_bl.to_numpy();    y_val_bl_np = y_v_bl.to_numpy()
            test_bl = X_te_bl.to_numpy();   y_te_bl_np  = y_te_bl.to_numpy()

            for name, fm_fn, key in [
                ("Linear", t.createFeatureMapLinear, "linear"),
                ("Ring",   t.createFeatureMapRing,   "ring"),
                ("Full",   t.createFeatureMapFull,   "full"),
            ]:
                fm        = fm_fn(X_orig.shape[1], reps=reps)
                val_m_bl, model_bl = evaluate_fn(fm, tr_bl, y_tr_bl_np,
                                                  val_bl, y_val_bl_np)
                test_m_bl = t.compute_metrics(model_bl, test_bl, y_te_bl_np)
                _bl_all_test[key].append(test_m_bl)
                _bl_run_logs[key].append({"seed": seed_bl, "test": test_m_bl})

        # Print aggregated results + single-split detail for split-0
        for name, key in [("Linear","linear"),("Ring","ring"),("Full","full")]:
            agg_bl = _aggregate(_bl_all_test[key])
            _section(f"Baseline: {name} Entanglement  "
                     f"({n_runs} runs, raw features, mean ± std [95% CI])")
            _row("test (aggregated)", agg_bl)
            # Also show split-0 detail with circuit info (circuit is data-independent)
            fm0 = {"linear": t.createFeatureMapLinear,
                   "ring":   t.createFeatureMapRing,
                   "full":   t.createFeatureMapFull}[key](X_orig.shape[1], reps=reps)
            comp_bl = t.circuit_complexity(fm0)
            per_cls_bl0 = t.compute_metrics_per_class(
                evaluate_fn(fm0, tr_o, y_tr_o_np, val_o, y_val_o_np)[1],
                test_o, y_te_o_np)
            _class_rows(per_cls_bl0)
            _circuit_row("circuit", comp_bl)
            _section_end()
            baseline_results[key] = {
                "aggregated_test": agg_bl,
                "runs":            _bl_run_logs[key],
                "complexity":      comp_bl,
            }

        full_log["baselines"]        = baseline_results
        full_log["baseline_linear_accs_per_run"] = [
            r["test"]["accuracy"] for r in _bl_run_logs["linear"]]
        full_log["baseline_ring_accs_per_run"] = [
            r["test"]["accuracy"] for r in _bl_run_logs["ring"]]
        full_log["baseline_full_accs_per_run"] = [
            r["test"]["accuracy"] for r in _bl_run_logs["full"]]

    # ── 6. Ablation (first split) ─────────────────────────────────────────────
    X_tr_fs0, X_te_fs0, y_tr_fs0, y_te_fs0 = t.splitData(
        X_fs, y_orig, random_state=first_seed)
    X_tr2_fs0, X_v_fs0, y_tr2_fs0, y_v_fs0 = train_test_split(
        X_tr_fs0, y_tr_fs0, test_size=0.2, random_state=first_seed)
    tr_fs0   = X_tr2_fs0.to_numpy();  y_tr_fs0_np  = y_tr2_fs0.to_numpy()
    val_fs0  = X_v_fs0.to_numpy();    y_val_fs0_np = y_v_fs0.to_numpy()
    test_fs0 = X_te_fs0.to_numpy();   y_te_fs0_np  = y_te_fs0.to_numpy()

    if run_ablation:
        ablation_log = {}

        # ── 6a. FS-only: all pairs, no search ─────────────────────────────
        _section("Ablation: FS-only  (all pairs, no metaheuristic search, split 0)")
        fm_all      = t.createFeatureMap(pairs_fs, X_fs.columns, reps=reps)
        val_fs_only, m_fs_only = evaluate_fn(
            fm_all, tr_fs0, y_tr_fs0_np, val_fs0, y_val_fs0_np)
        tst_fs_only = t.compute_metrics(m_fs_only, test_fs0, y_te_fs0_np)
        cls_fs_only = t.compute_metrics_per_class(m_fs_only, test_fs0, y_te_fs0_np)
        cmp_fs_only = t.circuit_complexity(fm_all)
        _row("validation", val_fs_only)
        _row("test",       tst_fs_only)
        _class_rows(cls_fs_only)
        _circuit_row("circuit", cmp_fs_only)
        _section_end()
        ablation_log["fs_only"] = {
            "val": val_fs_only, "test": tst_fs_only,
            "per_class": cls_fs_only, "complexity": cmp_fs_only,
        }

        # ── 6b. Entanglement-only: search on raw features, no FS ──────────
        _section("Ablation: Entanglement-only  (search on raw features, no FS, split 0)")
        pairs_orig = t.createCouples(
            t.transformCorrelations(X_orig.corr()), X_orig.columns)
        cache_eo   = {}
        _cv_eo     = ga_cv_folds if metaheuristic == "ga" else 0
        obj_eo     = _build_objective(pairs_orig, X_orig.columns,
                                       tr_o, y_tr_o_np, val_o, y_val_o_np,
                                       evaluate_fn, reps, cache_eo,
                                       cv_folds=_cv_eo)
        if metaheuristic == "sa":
            best_eo, _ = optimise(obj_eo, chromosome_length=len(pairs_orig),
                                   initial_temp=sa_initial_temp,
                                   cooling_rate=sa_cooling_rate,
                                   stopping_temp=sa_stopping_temp,
                                   max_iterations=sa_max_iterations,
                                   num_neighbors=sa_num_neighbors,
                                   seed=first_seed)
        else:
            best_eo, _ = optimise(obj_eo, chromosome_length=len(pairs_orig),
                                   population_size=ga_population_size,
                                   n_generations=ga_n_generations,
                                   crossover_rate=ga_crossover_rate,
                                   mutation_rate=ga_mutation_rate,
                                   tournament_size=ga_tournament_size,
                                   elitism_count=ga_elitism_count,
                                   seed=first_seed)
        info_eo      = cache_eo[str(best_eo)]
        tst_eo       = t.compute_metrics(info_eo["model"], test_o, y_te_o_np)
        cls_eo       = t.compute_metrics_per_class(info_eo["model"], test_o, y_te_o_np)
        cmp_eo       = t.circuit_complexity(info_eo["feature_map"])
        _row("validation", info_eo["metrics"])
        _row("test",       tst_eo)
        _class_rows(cls_eo)
        _circuit_row("circuit", cmp_eo)
        _section_end()
        ablation_log["entanglement_only"] = {
            "val": info_eo["metrics"], "test": tst_eo,
            "per_class": cls_eo, "complexity": cmp_eo,
        }
        full_log["ablation"] = ablation_log

    # ── 7. SEQUENT: FS + metaheuristic, n_runs independent seeds ─────────────
    # KEY CHANGE (reviewer comment 3):
    #   seed_i = BASE_SEED + run_idx is used for BOTH the data split AND the
    #   metaheuristic, so each run is a truly independent experiment on a
    #   different subset of the data.  The resulting mean ± std / CI is an
    #   estimate of the model's generalisation performance, not just
    #   metaheuristic variance.
    _header(f"SEQUENT  ({metaheuristic.upper()}, {n_runs} independent runs, "
            f"different split per run)")
    all_val  = []; all_test  = []; all_comp  = []
    all_cls  = []; run_logs  = []
    svm_accs_per_run = []          # SVM re-evaluated on each run's split
    mlp_accs_per_run = []          # MLP re-evaluated on each run's split (QNN only)
    best_acc = -1.0; best_info = None

    for run_idx in range(n_runs):
        seed_i = BASE_SEED + run_idx
        print(f"\n  ── Run {run_idx+1}/{n_runs}  (seed={seed_i}) "
              f"──────────────────────────────────")

        # ── Data split for this run ────────────────────────────────────────
        X_tr_fs, X_te_fs, y_tr_fs, y_te_fs = t.splitData(
            X_fs, y_orig, random_state=seed_i)
        X_tr2_fs, X_v_fs, y_tr2_fs, y_v_fs = train_test_split(
            X_tr_fs, y_tr_fs, test_size=0.2, random_state=seed_i)
        tr_fs   = X_tr2_fs.to_numpy();  y_tr_fs_np  = y_tr2_fs.to_numpy()
        val_fs  = X_v_fs.to_numpy();    y_val_fs_np = y_v_fs.to_numpy()
        test_fs = X_te_fs.to_numpy();   y_te_fs_np  = y_te_fs.to_numpy()

        # Re-evaluate classical SVM on the same split for a fair comparison
        X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = t.splitData(
            X_orig, y_orig, random_state=seed_i)
        X_tr2_raw, _, y_tr2_raw, _ = train_test_split(
            X_tr_raw, y_tr_raw, test_size=0.2, random_state=seed_i)
        svm_m_i, _ = t.evaluate_classical_svm(
            X_tr2_raw.to_numpy(), y_tr2_raw.to_numpy(),
            X_te_raw.to_numpy(),  y_te_raw.to_numpy())
        svm_accs_per_run.append(svm_m_i["accuracy"])

        # Re-evaluate classical MLP on the same split (QNN track only)
        mlp_m_i = None
        if model_type == "qnn":
            mlp_m_i, _ = t.evaluate_classical_mlp(
                X_tr2_raw.to_numpy(), y_tr2_raw.to_numpy(),
                X_te_raw.to_numpy(),  y_te_raw.to_numpy(),
                seed=seed_i)
            mlp_accs_per_run.append(mlp_m_i["accuracy"])

        # ── Metaheuristic search ───────────────────────────────────────────
        cache = {}
        # For the GA, use k-fold CV as the objective to prevent overfitting
        # to the small validation split (high selection pressure + small n
        # causes the GA to find entanglement maps that memorise val).
        _cv = ga_cv_folds if metaheuristic == "ga" else 0
        obj   = _build_objective(pairs_fs, X_fs.columns,
                                  tr_fs, y_tr_fs_np, val_fs, y_val_fs_np,
                                  evaluate_fn, reps, cache, cv_folds=_cv)

        if metaheuristic == "sa":
            best_sol, _ = optimise(obj, chromosome_length=len(pairs_fs),
                                    initial_temp=sa_initial_temp,
                                    cooling_rate=sa_cooling_rate,
                                    stopping_temp=sa_stopping_temp,
                                    max_iterations=sa_max_iterations,
                                    num_neighbors=sa_num_neighbors,
                                    seed=seed_i)
        else:
            best_sol, _ = optimise(obj, chromosome_length=len(pairs_fs),
                                    population_size=ga_population_size,
                                    n_generations=ga_n_generations,
                                    crossover_rate=ga_crossover_rate,
                                    mutation_rate=ga_mutation_rate,
                                    tournament_size=ga_tournament_size,
                                    elitism_count=ga_elitism_count,
                                    seed=seed_i)

        info      = cache[str(best_sol)]
        val_m     = info["metrics"]
        test_m    = t.compute_metrics(info["model"], test_fs, y_te_fs_np)
        per_cls   = t.compute_metrics_per_class(info["model"], test_fs, y_te_fs_np)
        comp      = t.circuit_complexity(info["feature_map"])

        all_val.append(val_m); all_test.append(test_m); all_comp.append(comp)
        all_cls.append(per_cls)

        _row(f"Run {run_idx+1} val",  val_m)
        _row(f"Run {run_idx+1} test", test_m)
        print(f"│  {'':22} SVM (same split): {svm_m_i['accuracy']:.4f}")
        if mlp_m_i is not None:
            print(f"│  {'':22} MLP (same split): {mlp_m_i['accuracy']:.4f}")
        _class_rows(per_cls)
        _circuit_row(f"Run {run_idx+1} circuit", comp)

        run_logs.append({"seed": seed_i,
                         "val": val_m, "test": test_m,
                         "svm_accuracy": svm_m_i["accuracy"],
                         "mlp_accuracy": mlp_m_i["accuracy"] if mlp_m_i else None,
                         "per_class": per_cls, "complexity": comp,
                         "solution": info["solution"]})

        if val_m["accuracy"] > best_acc:
            best_acc  = val_m["accuracy"]
            best_info = info

    # ── 8. Aggregate across runs ──────────────────────────────────────────────
    agg_val   = _aggregate(all_val)
    agg_test  = _aggregate(all_test)
    agg_comp  = _aggregate(all_comp)

    _section(f"SEQUENT  —  Aggregated ({n_runs} runs, mean ± std [95% CI])")
    _row("validation", agg_val)
    _row("test",       agg_test)
    _circuit_row("complexity", agg_comp)

    if all_cls:
        classes = all_cls[0].keys()
        print(f"│  Per-class test metrics (mean ± std over {n_runs} runs):")
        for cls in classes:
            f1s = [r[cls]["f1"] for r in all_cls]
            n   = len(f1s)
            mu  = np.mean(f1s)
            std = np.std(f1s, ddof=1)
            ci  = float(sp_stats.t.ppf(0.975, df=max(n-1,1))) * std / np.sqrt(n)
            print(f"│    class {cls:<6}  f1:{mu:.4f} ±{std:.4f} [95%CI ±{ci:.4f}]")
    _section_end()

    # ── 9. Statistical significance ───────────────────────────────────────────
    # Primary test: SEQUENT vs Linear entanglement baseline — the natural
    # quantum-to-quantum comparison that justifies the contribution.
    # Secondary test: SEQUENT vs classical SVM — kept for completeness but
    # less meaningful as a main claim in a QML paper.
    sequent_accs = [r["test"]["accuracy"] for r in run_logs]

    sig_results_linear = {}
    sig_results_ring   = {}
    sig_results_full   = {}
    if run_baselines:
        for key, var_name, label in [
            ("baseline_linear_accs_per_run", "sig_results_linear",
             "SEQUENT vs Linear entanglement"),
            ("baseline_ring_accs_per_run",   "sig_results_ring",
             "SEQUENT vs Ring entanglement"),
            ("baseline_full_accs_per_run",   "sig_results_full",
             "SEQUENT vs Full entanglement"),
        ]:
            bl_accs = full_log.get(key, [])
            if bl_accs:
                result = _significance_tests(sequent_accs, bl_accs)
                _print_significance_block(result, label)
                if var_name == "sig_results_linear":
                    sig_results_linear = result
                elif var_name == "sig_results_ring":
                    sig_results_ring = result
                else:
                    sig_results_full = result

    sig_results  = _significance_tests(sequent_accs, svm_accs_per_run)
    _print_significance_block(
        sig_results,
        "SEQUENT vs classical SVM — reference")

    sig_results_mlp = {}
    if model_type == "qnn" and mlp_accs_per_run:
        sig_results_mlp = _significance_tests(sequent_accs, mlp_accs_per_run)
        _print_significance_block(sig_results_mlp, "QNN vs classical MLP")

    # ── 10. Save results ──────────────────────────────────────────────────────
    full_log.update({
        "aggregated_val":        agg_val,
        "aggregated_test":       agg_test,
        "aggregated_complexity": agg_comp,
        "runs":                  run_logs,
        "svm_accs_per_run":      svm_accs_per_run,
        "mlp_accs_per_run":      mlp_accs_per_run if mlp_accs_per_run else None,
        "significance_tests_vs_linear": sig_results_linear if sig_results_linear else None,
        "significance_tests_vs_ring":   sig_results_ring   if sig_results_ring   else None,
        "significance_tests_vs_full":   sig_results_full   if sig_results_full   else None,
        "significance_tests_vs_svm":    sig_results,
        "significance_tests_mlp":       sig_results_mlp if sig_results_mlp else None,
        "best_solution":         best_info["solution"] if best_info else None,
    })

    json_path = os.path.join("results", f"{run_id}.json")
    _save_json(full_log, json_path)
    print(f"\n  Full log → {json_path}")

    csv_row = {
        "run_id":              run_id,
        "dataset":             dataset,
        "model_type":          model_type,
        "mode":                mode,
        "metaheuristic":       metaheuristic,
        "fs_method":           fs_method if use_fs else "none",
        "k":                   k if use_fs else X_orig.shape[1],
        "reps":                reps,
        "n_runs":              n_runs,
        "ga_cv_folds":         ga_cv_folds if metaheuristic == "ga" else 0,
        # SVM baseline (split-0 reference)
        "svm_test_acc_split0": svm_m_first["accuracy"],
        "svm_test_f1_split0":  svm_m_first["f1_macro"],
        # SVM re-evaluated per run (mean across runs)
        "svm_acc_mean":        float(np.mean(svm_accs_per_run)),
        "svm_acc_std":         float(np.std(svm_accs_per_run, ddof=1)),
        # MLP re-evaluated per run — only populated for model_type="qnn"
        "mlp_acc_mean":        float(np.mean(mlp_accs_per_run)) if mlp_accs_per_run else None,
        "mlp_acc_std":         float(np.std(mlp_accs_per_run, ddof=1)) if mlp_accs_per_run else None,
        "mlp_wilcoxon_p":      sig_results_mlp.get("wilcoxon_p",   float("nan")) if sig_results_mlp else None,
        "mlp_wilcoxon_sig_05": sig_results_mlp.get("wilcoxon_sig_05", False)     if sig_results_mlp else None,
        # SEQUENT aggregated
        "val_acc_mean":        agg_val.get("accuracy_mean", 0),
        "val_acc_std":         agg_val.get("accuracy_std",  0),
        "val_acc_ci95":        agg_val.get("accuracy_ci95", 0),
        "val_f1_mean":         agg_val.get("f1_macro_mean", 0),
        "test_acc_mean":       agg_test.get("accuracy_mean", 0),
        "test_acc_std":        agg_test.get("accuracy_std",  0),
        "test_acc_ci95":       agg_test.get("accuracy_ci95", 0),
        "test_f1_mean":        agg_test.get("f1_macro_mean", 0),
        "test_f1_std":         agg_test.get("f1_macro_std",  0),
        "test_f1_ci95":        agg_test.get("f1_macro_ci95", 0),
        "depth_mean":          agg_comp.get("depth_mean",    0),
        "two_qubit_gates_mean":agg_comp.get("two_qubit_gates_mean", 0),
        "train_time_mean":     agg_val.get("training_time_mean", 0),
        # Significance — quantum baselines (linear / ring / full)
        "linear_perm_p":       sig_results_linear.get("permutation_p",      float("nan")) if sig_results_linear else None,
        "linear_perm_sig_05":  sig_results_linear.get("permutation_sig_05", False)        if sig_results_linear else None,
        "linear_sign_p":       sig_results_linear.get("sign_p",             float("nan")) if sig_results_linear else None,
        "linear_ttest_p":      sig_results_linear.get("ttest_p",            float("nan")) if sig_results_linear else None,
        "ring_perm_p":         sig_results_ring.get("permutation_p",      float("nan")) if sig_results_ring else None,
        "ring_perm_sig_05":    sig_results_ring.get("permutation_sig_05", False)        if sig_results_ring else None,
        "ring_sign_p":         sig_results_ring.get("sign_p",             float("nan")) if sig_results_ring else None,
        "ring_ttest_p":        sig_results_ring.get("ttest_p",            float("nan")) if sig_results_ring else None,
        "full_perm_p":         sig_results_full.get("permutation_p",      float("nan")) if sig_results_full else None,
        "full_perm_sig_05":    sig_results_full.get("permutation_sig_05", False)        if sig_results_full else None,
        "full_sign_p":         sig_results_full.get("sign_p",             float("nan")) if sig_results_full else None,
        "full_ttest_p":        sig_results_full.get("ttest_p",            float("nan")) if sig_results_full else None,
        # Significance — reference: SEQUENT vs classical SVM
        "svm_perm_p":          sig_results.get("permutation_p",      float("nan")),
        "svm_perm_sig_05":     sig_results.get("permutation_sig_05", False),
        "svm_sign_p":          sig_results.get("sign_p",             float("nan")),
        "svm_ttest_p":         sig_results.get("ttest_p",            float("nan")),
        "timestamp":           timestamp,
    }
    _append_csv(csv_row)
    print(f"  Summary → {_CSV_PATH}")

    return full_log


# ═══════════════════════════════════════════════════════════════
#  BENCHMARK ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Dataset manifest ──────────────────────────────────────────────────────
    DATASETS = [
        ("corral",              1, None),
        ("breast-w",            0, "./datasets/breast-w.tsv"),
        ("fitness_class_2212",  0, "./datasets/fitness_class_2212.csv"),
        ("flare",               0, "./datasets/flare.tsv"),
        ("heart",               0, "./datasets/heart.csv"),
    ]

    # ── Experiment grid ───────────────────────────────────────────────────────
    # Full factorial design:
    #   model_type : qsvm, qnn              (2)
    #   metaheuristic: sa, ga               (2)
    #   fs_method  : anova, autoencoder     (2)
    #   reps       : 1, 3, 5               (3)
    #   n_runs     : 5  (fixed)
    #   k          : 5  (fixed)
    #   Total      : 2×2×2×3 = 24 configs × 5 datasets = 120 experiments
    #
    # run_baselines=True only on reps=1 configs — baselines use fixed
    # entanglement and don't depend on reps, so running them once is enough.
    # run_ablation=True only on reps=1 + anova + sa (the canonical setting).
    #
    # SA defaults inherited from function signature:
    #   sa_initial_temp=10.0, sa_cooling_rate=0.90,
    #   sa_max_iterations=20, sa_num_neighbors=5
    # GA defaults inherited from function signature:
    #   ga_population_size=20, ga_n_generations=10,
    #   ga_tournament_size=2,  ga_cv_folds=3

    _MODELS = ["qsvm", "qnn"]
    _MHS    = ["sa", "ga"]
    _FS     = ["anova", "autoencoder"]
    _REPS   = [1, 3, 5]

    CONFIGS = []
    for _model in _MODELS:
        for _mh in _MHS:
            for _fs in _FS:
                for _reps in _REPS:
                    # Baselines (linear/ring/full) only on reps=1 — they are
                    # independent of reps so one evaluation is sufficient.
                    _baselines = (_reps == 1)
                    # Ablation only on the canonical config: reps=1, anova, SA.
                    # Running ablation for every combination would be redundant
                    # and very expensive.
                    _ablation  = (_reps == 1 and _fs == "anova" and _mh == "sa")
                    CONFIGS.append(dict(
                        model_type    = _model,
                        mode          = "statevector",
                        metaheuristic = _mh,
                        use_fs        = True,
                        fs_method     = _fs,
                        k             = 5,
                        reps          = _reps,
                        n_runs        = 5,
                        run_baselines = _baselines,
                        run_ablation  = _ablation,
                    ))

    # Quick sanity-check: print the grid before running
    print(f"\n  Experiment grid: {len(CONFIGS)} configs × {len(DATASETS)} datasets "
          f"= {len(CONFIGS)*len(DATASETS)} experiments\n")
    _header_cols = f"{'model':<6} {'mh':<4} {'fs':<12} {'reps':<5} {'baselines':<10} {'ablation'}"
    print(f"  {_header_cols}")
    print(f"  {'-'*55}")
    for _c in CONFIGS:
        print(f"  {_c['model_type']:<6} {_c['metaheuristic']:<4} "
              f"{_c['fs_method']:<12} {_c['reps']:<5} "
              f"{str(_c['run_baselines']):<10} {_c['run_ablation']}")
    print()

    for ds, opt, p in DATASETS:
        for cfg in CONFIGS:
            try:
                run_experiment(
                    dataset=ds, option=opt, path=p,
                    **cfg,
                )
            except Exception as e:
                print(f"\n  [ERROR] {ds} / {cfg['model_type']} / {cfg['mode']} : {e}")
                import traceback
                traceback.print_exc()