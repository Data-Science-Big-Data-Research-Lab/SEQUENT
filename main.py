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
import json
import random
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals

import tools as t
from metaheuristicas import simulated_annealing, genetic_algorithm

# ─── Seeds ────────────────────────────────────────────────────────────────────
random.seed(12345)
np.random.seed(12345)
algorithm_globals.random_seed = 12345

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

    acc_str = f"{acc:.4f}" + (f" ±{std_acc:.4f}" if std_acc is not None else "")
    f1_str  = f"{f1:.4f}"  + (f" ±{std_f1:.4f}"  if std_f1  is not None else "")

    print(f"│  {label:<22} Acc:{acc_str:<12} Prec:{prec:.4f}  Rec:{rec:.4f}  F1:{f1_str}")
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
    """Mean ± std for every numeric key across multiple runs."""
    if not results_list:
        return {}
    keys = results_list[0].keys()
    out  = {}
    for k in keys:
        vals = [r[k] for r in results_list if isinstance(r[k], (int, float))]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"]  = float(np.std(vals))
    return out


# ═══════════════════════════════════════════════════════════════
#  OBJECTIVE FUNCTION FACTORY
# ═══════════════════════════════════════════════════════════════

def _build_objective(couples_array, columns, tr, y_tr, val, y_val,
                     evaluate_fn, reps, cache):
    """
    Returns a memoised f(vector) -> (cost, training_time).
    cost = -accuracy so optimisers that minimise cost maximise accuracy.
    All evaluated solutions are stored in `cache` (passed by reference).
    """
    def objective(vector):
        key = str(vector)
        if key in cache:
            return cache[key]["cost"], cache[key]["aux"]
        selected = [couples_array[i] for i, b in enumerate(vector) if b == 1]
        fm       = t.createFeatureMap(selected, columns, reps=reps)
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
        return cost, metrics["training_time"]
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
    sa_initial_temp=1.0,        # Low T0: fast convergence, less exploration
    sa_cooling_rate=0.95,
    sa_stopping_temp=0.001,
    sa_max_iterations=10,       # 10 iters × 3 neighbors = 30 evals → fast
    sa_num_neighbors=3,
    # GA hyperparameters
    ga_population_size=10,      # 10 init + ~5 new/gen × 5 gens ≈ 35 evals
    ga_n_generations=5,
    ga_crossover_rate=0.8,
    ga_mutation_rate=None,
    ga_tournament_size=3,
    ga_elitism_count=1,
    # ── Statistical rigor ─────────────────────
    n_runs=5,                   # R1-03: multiple seeds            (R1-03)
    # ── Scope ─────────────────────────────────
    run_baselines=True,         # linear / ring / full maps        (R1-02)
    run_ablation=True,          # FS-only, Ent-only                (R1-07)
):
    """
    Full SEQUENT experiment.  See module docstring for parameter guide.
    Results saved to results/ after each run.
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
    }

    # ── 1. Load raw data (used for baselines and entanglement-only ablation) ──
    X_orig, y_orig = t.load_data(path=path, option=option, dataset=dataset)
    print(f"\n  Dataset: {X_orig.shape[0]} samples × {X_orig.shape[1]} features")
    print(f"  Class distribution: {dict(y_orig.value_counts().sort_index())}")

    X_tr_o, X_te_o, y_tr_o, y_te_o = t.splitData(X_orig, y_orig)
    X_tr2_o, X_v_o, y_tr2_o, y_v_o = train_test_split(
        X_tr_o, y_tr_o, test_size=0.2, random_state=12345)
    tr_o   = X_tr2_o.to_numpy();  y_tr_o  = y_tr2_o.to_numpy()
    val_o  = X_v_o.to_numpy();    y_val_o = y_v_o.to_numpy()
    test_o = X_te_o.to_numpy();   y_te_o  = y_te_o.to_numpy()

    # ── 2. Classical SVM (always, on raw data) ────────────────────────────────
    _section("Baseline: Classical RBF-SVM  (raw features, test set)")
    svm_m, svm_model = t.evaluate_classical_svm(tr_o, y_tr_o, test_o, y_te_o)
    _row("SVM test", svm_m)
    _section_end()
    full_log["baseline_svm"] = svm_m

    # ── 3. Quantum entanglement baselines (on raw data) ───────────────────────
    if run_baselines:
        baseline_results = {}
        for name, fm_fn in [
            ("Linear",  t.createFeatureMapLinear),
            ("Ring",    t.createFeatureMapRing),
            ("Full",    t.createFeatureMapFull),
        ]:
            _section(f"Baseline: {name} Entanglement  (raw features)")
            fm       = fm_fn(X_orig.shape[1], reps=reps)
            val_m, model = evaluate_fn(fm, tr_o, y_tr_o, val_o, y_val_o)
            test_m   = t.compute_metrics(model, test_o, y_te_o)
            per_cls  = t.compute_metrics_per_class(model, test_o, y_te_o)
            comp     = t.circuit_complexity(fm)
            _row("validation", val_m)
            _row("test",       test_m)
            _class_rows(per_cls)
            _circuit_row("circuit", comp)
            _section_end()
            baseline_results[name.lower()] = {
                "val": val_m, "test": test_m,
                "per_class": per_cls, "complexity": comp,
            }
        full_log["baselines"] = baseline_results

    # ── 4. Feature selection (used exclusively for SEQUENT + FS-only ablation) ─
    print(f"\n  Applying FS: method={fs_method}, k={k}")
    X_fs, fs_cols = t.apply_feature_selection(X_orig, y_orig, method=fs_method, k=k)

    X_tr_fs, X_te_fs, y_tr_fs, y_te_fs = t.splitData(X_fs, y_orig)
    X_tr2_fs, X_v_fs, y_tr2_fs, y_v_fs = train_test_split(
        X_tr_fs, y_tr_fs, test_size=0.2, random_state=12345)
    tr_fs   = X_tr2_fs.to_numpy();  y_tr_fs  = y_tr2_fs.to_numpy()
    val_fs  = X_v_fs.to_numpy();    y_val_fs = y_v_fs.to_numpy()
    test_fs = X_te_fs.to_numpy();   y_te_fs  = y_te_fs.to_numpy()

    pairs_fs = t.createCouples(
        t.transformCorrelations(X_fs.corr()), X_fs.columns)

    # ── 5. Ablation ───────────────────────────────────────────────────────────
    if run_ablation:
        ablation_log = {}

        # ── 5a. FS-only: all pairs, no search ─────────────────────────────
        _section("Ablation: FS-only  (all pairs, no metaheuristic search)")
        fm_all      = t.createFeatureMap(pairs_fs, X_fs.columns, reps=reps)
        val_fs_only, m_fs_only = evaluate_fn(fm_all, tr_fs, y_tr_fs, val_fs, y_val_fs)
        tst_fs_only = t.compute_metrics(m_fs_only, test_fs, y_te_fs)
        cls_fs_only = t.compute_metrics_per_class(m_fs_only, test_fs, y_te_fs)
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

        # ── 5b. Entanglement-only: search on raw features, no FS ──────────
        _section("Ablation: Entanglement-only  (search on raw features, no FS)")
        pairs_orig = t.createCouples(
            t.transformCorrelations(X_orig.corr()), X_orig.columns)
        cache_eo   = {}
        obj_eo     = _build_objective(pairs_orig, X_orig.columns,
                                       tr_o, y_tr_o, val_o, y_val_o,
                                       evaluate_fn, reps, cache_eo)
        if metaheuristic == "sa":
            best_eo, _ = optimise(obj_eo, chromosome_length=len(pairs_orig),
                                   initial_temp=sa_initial_temp,
                                   cooling_rate=sa_cooling_rate,
                                   stopping_temp=sa_stopping_temp,
                                   max_iterations=sa_max_iterations,
                                   num_neighbors=sa_num_neighbors,
                                   seed=12345)
        else:
            best_eo, _ = optimise(obj_eo, chromosome_length=len(pairs_orig),
                                   population_size=ga_population_size,
                                   n_generations=ga_n_generations,
                                   crossover_rate=ga_crossover_rate,
                                   mutation_rate=ga_mutation_rate,
                                   tournament_size=ga_tournament_size,
                                   elitism_count=ga_elitism_count,
                                   seed=12345)
        info_eo      = cache_eo[str(best_eo)]
        tst_eo       = t.compute_metrics(info_eo["model"], test_o, y_te_o)
        cls_eo       = t.compute_metrics_per_class(info_eo["model"], test_o, y_te_o)
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

    # ── 6. SEQUENT: FS + metaheuristic, n_runs independent seeds ─────────────
    _header(f"SEQUENT  ({metaheuristic.upper()}, {n_runs} independent runs)")
    all_val  = []; all_test  = []; all_comp  = []
    all_cls  = []; run_logs  = []
    best_acc = -1.0; best_info = None

    for run_idx in range(n_runs):
        print(f"\n  ── Run {run_idx+1}/{n_runs} ──────────────────────────────────")
        cache = {}
        obj   = _build_objective(pairs_fs, X_fs.columns,
                                  tr_fs, y_tr_fs, val_fs, y_val_fs,
                                  evaluate_fn, reps, cache)

        seed_i = 12345 + run_idx
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
        test_m    = t.compute_metrics(info["model"], test_fs, y_te_fs)
        per_cls   = t.compute_metrics_per_class(info["model"], test_fs, y_te_fs)
        comp      = t.circuit_complexity(info["feature_map"])

        all_val.append(val_m); all_test.append(test_m); all_comp.append(comp)
        # Aggregate per-class f1 across runs
        all_cls.append(per_cls)

        _row(f"Run {run_idx+1} val",  val_m)
        _row(f"Run {run_idx+1} test", test_m)
        _class_rows(per_cls)
        _circuit_row(f"Run {run_idx+1} circuit", comp)

        run_logs.append({"val": val_m, "test": test_m,
                         "per_class": per_cls, "complexity": comp,
                         "solution": info["solution"]})

        if val_m["accuracy"] > best_acc:
            best_acc  = val_m["accuracy"]
            best_info = info

    # ── 7. Aggregate across runs ──────────────────────────────────────────────
    agg_val   = _aggregate(all_val)
    agg_test  = _aggregate(all_test)
    agg_comp  = _aggregate(all_comp)

    _section(f"SEQUENT  —  Aggregated ({n_runs} runs, mean ± std)")
    _row("validation", agg_val)
    _row("test",       agg_test)
    _circuit_row("complexity", agg_comp)

    # Per-class mean across runs
    if all_cls:
        classes = all_cls[0].keys()
        print(f"│  Per-class test metrics (mean over {n_runs} runs):")
        for cls in classes:
            f1s = [r[cls]["f1"] for r in all_cls]
            print(f"│    class {cls:<6}  f1:{np.mean(f1s):.4f} ±{np.std(f1s):.4f}")
    _section_end()

    # ── 8. Save results ───────────────────────────────────────────────────────
    full_log.update({
        "aggregated_val":      agg_val,
        "aggregated_test":     agg_test,
        "aggregated_complexity": agg_comp,
        "runs":                run_logs,
        "best_solution":       best_info["solution"] if best_info else None,
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
        # SVM baseline
        "svm_test_acc":        svm_m["accuracy"],
        "svm_test_f1":         svm_m["f1_macro"],
        # SEQUENT aggregated
        "val_acc_mean":        agg_val.get("accuracy_mean", 0),
        "val_acc_std":         agg_val.get("accuracy_std",  0),
        "val_f1_mean":         agg_val.get("f1_macro_mean", 0),
        "test_acc_mean":       agg_test.get("accuracy_mean", 0),
        "test_acc_std":        agg_test.get("accuracy_std",  0),
        "test_f1_mean":        agg_test.get("f1_macro_mean", 0),
        "test_f1_std":         agg_test.get("f1_macro_std",  0),
        "depth_mean":          agg_comp.get("depth_mean",    0),
        "two_qubit_gates_mean":agg_comp.get("two_qubit_gates_mean", 0),
        "train_time_mean":     agg_val.get("training_time_mean", 0),
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
    # Each entry: (dataset_tag, option, path)
    #   option=1  → PMLB by name
    #   option=0  → local file; path must point to the CSV/TSV
    DATASETS = [
        ("corral",              1, None),
        ("breast-w",            0, "./datasets/breast-w.tsv"),
        ("fitness_class_2212",  0, "./datasets/fitness_class_2212.csv"),
        ("flare",               0, "./datasets/flare.tsv"),
        ("heart",               0, "./datasets/heart.csv"),
    ]

    # ── Experiment grid ───────────────────────────────────────────────────────
    # Uncomment/extend as needed for the revision plan. / Probar reps 1, 3 y 5
    CONFIGS = [
        # R1-03: statevector, SA, ANOVA, 5 runs (main result table)
        dict(model_type="qsvm", mode="statevector",
             metaheuristic="sa", use_fs=True, fs_method="anova", k=5,
             reps=1, n_runs=5, run_baselines=True, run_ablation=True),

        # R2-13: same with mutual_info FS
        dict(model_type="qsvm", mode="statevector",
             metaheuristic="sa", use_fs=True, fs_method="mutual_info", k=5,
             reps=1, n_runs=5, run_baselines=False, run_ablation=False),

        # Alternative metaheuristic: GA
        dict(model_type="qsvm", mode="statevector",
             metaheuristic="ga", use_fs=True, fs_method="anova", k=5,
             reps=1, n_runs=5, run_baselines=False, run_ablation=False),

        # R2-12: effect of reps=2
        dict(model_type="qsvm", mode="statevector",
             metaheuristic="sa", use_fs=True, fs_method="anova", k=5,
             reps=2, n_runs=5, run_baselines=True, run_ablation=False),

        # Second model: QNN statevector
        dict(model_type="qnn", mode="statevector",
             metaheuristic="sa", use_fs=True, fs_method="anova", k=5,
             reps=1, n_runs=3, run_baselines=True, run_ablation=True),
    ]

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