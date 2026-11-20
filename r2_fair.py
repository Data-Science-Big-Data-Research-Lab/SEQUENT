"""
r2_fair.py — Fixed topologies on the same selected features (QMI revision, R2)
==============================================================================
Editor, comment 1: "The revised evaluation should include fixed and optimized
topologies using the same features, number of qubits, learning model, and
experimental conditions."

In R1 the Linear / Ring / Full references of the results tables (ideal simulator
and noisy emulator) used all m original features, while SEQUENT used the k = 5
selected ones. This script runs Linear-5 / Ring-5 / Full-5 on exactly the k = 5
features SEQUENT sees in each run, with the R1 code (Hercules main.py / tools.py):

  modes     statevector (ideal simulator) and noise (emulator)
  grid      5 datasets x {QSVM, QNN} x {ANOVA, AE} x Reps {1, 2, 3} x 10 seeds
  metrics   accuracy, precision, recall, F1 (macro), 2Q gates, depth
  times     training_time (fit, same definition as the R1 tables), predict_time,
            total_time (fit + predict on test)

Pairing with SEQUENT: same BASE_SEED + run_idx, same _make_splits and same feature
selection (_prepare_fs_views) as main.py; the model is trained on the outer train
split and evaluated on test, as SEQUENT's final model. The fixed circuit does not
depend on the metaheuristic, so the same value serves the SA and ILS rows of a table.

Usage
-----
  python r2_fair.py prepare [--freeze-backend --backend ibm_pittsburgh]
  python r2_fair.py plan --mode statevector|noise
  python r2_fair.py run --dataset corral --model qsvm --fs anova --reps 1 \
                        --mode statevector [--runs 0-9]
"""

import argparse
import fcntl
import json
import os
import pickle
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

# main.py saves an IBM account at import time; skip it (the account saved on
# Hercules is used when the noise model has to be fetched).
import qiskit_ibm_runtime
qiskit_ibm_runtime.QiskitRuntimeService.save_account = staticmethod(lambda *a, **k: None)

from qiskit_algorithms.utils import algorithm_globals  # noqa: E402

import tools as t  # noqa: E402
from main import BASE_SEED, _EVALUATE, _make_splits, _prepare_fs_views  # noqa: E402

DATASETS = {
    "corral":             (1, None),
    "breast-w":           (0, "./datasets/breast-w.tsv"),
    "fitness_class_2212": (0, "./datasets/fitness_class_2212.csv"),
    "flare":              (0, "./datasets/flare.tsv"),
    "heart":              (0, "./datasets/heart.csv"),
}
MODELS = ["qsvm", "qnn"]
FS_METHODS = ["anova", "autoencoder"]
REPS_GRID = [1, 2, 3]
TOPOLOGIES = ["linear", "ring", "full"]
K = 5
N_RUNS = 10

# Values of the Hercules tools.py, the code that produced the R1 SEQUENT JSONs.
# C=1000 / num_steps=500 / COBYLA 100 (first R2 campaign) did not reproduce the R1
# fs_only ablation (Full on the k features, seed 12345, same process as SEQUENT):
# 0/20 QSVM and 0/20 QNN matches. r2_tables.py repeats that parity check.
HYPERPARAMS = dict(_QSVM_C=100, _QSVM_NUM_STEPS=50, _QSVM_SEED=12345,
                   _QNN_COBYLA_SEARCH=30, _QNN_COBYLA_FINAL=50)
for _name, _value in HYPERPARAMS.items():
    setattr(t, _name, _value)

CACHE_DIR = "r2_cache"
VIEWS_DIR = os.path.join(CACHE_DIR, "views")
ROWS_DIR = os.path.join("results_r2", "rows_c100_s50")   # the C=1000/500 rows stay in rows/
ROWS_R1_DIR = os.path.join("results_r2", "rows_r1regime")  # same runs in the R1 baselines' regime
R1_WORKERS = 15
NOISE_FILE = os.path.join(CACHE_DIR, "noise_model_{backend}.pkl")


def _view_path(dataset, fs, seed):
    return os.path.join(VIEWS_DIR, f"{dataset}_{fs}_seed{seed}.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# PREPARE: freeze the k = 5 views once (the autoencoder is not bit-reproducible
# across nodes), and optionally the noise model, so every job shares them.
# ═════════════════════════════════════════════════════════════════════════════

def _main_py_account():
    """channel / token / instance of the save_account call in main.py (read, not executed)."""
    import ast
    with open("main.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save_account"):
            kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords
                  if k.arg in ("channel", "token", "instance")}
            if "token" in kw:
                return kw
    raise RuntimeError("no save_account(...) with a token in main.py")


def _find_backend(name):
    """
    Look for `name` in the default saved IBM account and then in every other saved
    account (~/.qiskit/qiskit-ibm.json). If none sees it, list what each one sees.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService
    accounts = ["main.py"] + [None] + [a for a in QiskitRuntimeService.saved_accounts() if a]
    seen = {}
    for acc in accounts:
        tag = acc or "default"
        try:
            if acc == "main.py":      # the SEQUENT account R1 saved on every import
                service = QiskitRuntimeService(**_main_py_account())
            else:
                service = QiskitRuntimeService() if acc is None else QiskitRuntimeService(name=acc)
            names = sorted(b.name for b in service.backends())
        except Exception as e:
            seen[tag] = f"error: {type(e).__name__}: {str(e)[:120]}"
            continue
        seen[tag] = names
        if name in names:
            print(f"Backend {name} found in saved account '{tag}'")
            return service.backend(name)
    print(f"Backend {name} not visible from any saved account:")
    for tag, names in seen.items():
        print(f"  {tag}: {names}")
    sys.exit(1)


def prepare(args):
    os.makedirs(VIEWS_DIR, exist_ok=True)
    for ds in args.datasets or list(DATASETS):
        opt, path = DATASETS[ds]
        X, y = t.load_data(path=path, option=opt, dataset=ds)
        print(f"[{ds}] {X.shape}  classes={dict(Counter(y))}")
        for fs in FS_METHODS:
            for run_i in range(N_RUNS):
                seed = BASE_SEED + run_i
                out = _view_path(ds, fs, seed)
                if os.path.exists(out) and not args.overwrite:
                    continue
                sp = _make_splits(X, y, seed)
                fv = _prepare_fs_views(sp, True, fs, K, seed)
                view = dict(
                    dataset=ds, fs=fs, seed=seed, cols=list(fv["fs_cols"]),
                    outer_train_X=fv["outer_train_np"], outer_train_y=sp["outer_train_y"].to_numpy(),
                    test_X=fv["test_np"], test_y=sp["test_y"].to_numpy(),
                )
                with open(out + ".tmp", "wb") as f:
                    pickle.dump(view, f)
                os.replace(out + ".tmp", out)
                print(f"  {ds}/{fs}/seed{seed}: {view['cols']}")

    if args.freeze_backend:
        from qiskit_aer.noise import NoiseModel
        backend = _find_backend(args.backend)
        with open(NOISE_FILE.format(backend=args.backend), "wb") as f:
            pickle.dump(NoiseModel.from_backend(backend), f)
        with open(os.path.join(CACHE_DIR, f"frozen_{args.backend}.json"), "w") as f:
            json.dump(dict(backend=args.backend, frozen_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                           num_qubits=backend.num_qubits,
                           basis_gates=sorted(backend.operation_names)), f, indent=2)
        print(f"Frozen noise model of {args.backend} → {CACHE_DIR}/")


# ═════════════════════════════════════════════════════════════════════════════
# PLAN
# ═════════════════════════════════════════════════════════════════════════════

def plan(args):
    lines = []
    for ds in DATASETS:
        for model in MODELS:
            for fs in FS_METHODS:
                for reps in REPS_GRID:
                    base = (f"python r2_fair.py run --dataset {ds} --model {model} --fs {fs} "
                            f"--reps {reps} --mode {args.mode}")
                    if args.r1_regime:         # the 30 tasks of a configuration share one node
                        lines.append(f"{base} --runs 0-{N_RUNS - 1} --r1-regime")
                    elif args.mode == "noise":   # emulation is slow: one job per seed
                        lines += [f"{base} --runs {r}" for r in range(N_RUNS)]
                    else:
                        lines.append(f"{base} --runs 0-{N_RUNS - 1}")
    name = f"cmds_r2_{'r1regime_' if args.r1_regime else ''}{args.mode}.txt"
    with open(name, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)} commands → {name}\n  sbatch --array=1-{len(lines)} slurm_r2.sh {name}")


# ═════════════════════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════════════════════

def _parse_runs(spec):
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def _seed_all(seed):
    np.random.seed(seed)                  # QNN initial weights
    random.seed(seed)
    algorithm_globals.random_seed = seed  # shot sampling stays unseeded, as in R1


def _fit_row(job, seed, topo):
    """Fit one fixed topology on the k = 5 view of `seed` and return its row."""
    eval_kw = dict(fast_eval=False)
    if job.mode == "noise":
        eval_kw["backend_name"] = job.backend
    evaluate_fn = _EVALUATE[job.model][job.mode]
    v_path = _view_path(job.dataset, job.fs, seed)
    if not os.path.exists(v_path):
        sys.exit(f"Missing {v_path}: run `python r2_fair.py prepare` first.")
    with open(v_path, "rb") as f:
        v = pickle.load(f)
    cols = pd.Index(v["cols"])
    pairs = t.topology_pairs(cols, topo)
    fm = t.createFeatureMap(pairs, cols, reps=job.reps)

    _seed_all(seed)
    t0 = time.time()
    m, model = evaluate_fn(fm, v["outer_train_X"], v["outer_train_y"],
                           v["test_X"], v["test_y"], **eval_kw)
    t1 = time.time()
    test_m = t.compute_metrics(model, v["test_X"], v["test_y"])
    predict_time = time.time() - t1
    comp = t.circuit_complexity(fm)

    return dict(
        dataset=job.dataset, model=job.model, fs=job.fs, reps=job.reps, mode=job.mode,
        arm=topo, seed=seed, source="r2",
        backend=job.backend if job.mode == "noise" else "",
        n_qubits=comp["n_qubits"], n_pairs=len(pairs),
        depth=comp["depth"], total_gates=comp["total_gates"],
        two_qubit_gates=comp["two_qubit_gates"],
        accuracy=test_m["accuracy"], precision_macro=test_m["precision_macro"],
        recall_macro=test_m["recall_macro"], f1_macro=test_m["f1_macro"],
        majority_rate=float(max(Counter(v["test_y"]).values()) / len(v["test_y"])),
        # training_time = fit only, the definition used for SEQUENT in the R1 tables
        training_time=float(m["training_time"]),
        predict_time=float(predict_time),
        total_time=float(m["training_time"] + predict_time),
        wall_time=float(time.time() - t0),
        qsvm_C=t._QSVM_C, qsvm_num_steps=t._QSVM_NUM_STEPS, qnn_cobyla=t._QNN_COBYLA_FINAL,
        threads=t._threads_per_worker(), host=os.uname().nodename,
        regime="r1_pool" if job.r1_regime else "sequential",
        workers=job.workers if job.r1_regime else 1,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _r1_task(args):
    job, seed, topo = args
    return _fit_row(job, seed, topo)


def _save_row(out_csv, row):
    """Merge one row into the configuration CSV (read-merge-replace under a file lock)."""
    df = pd.DataFrame([row])
    with open(out_csv + ".lock", "w") as lk:
        try:
            fcntl.flock(lk, fcntl.LOCK_EX)
        except OSError:           # filesystem without flock: write unlocked
            pass
        if os.path.exists(out_csv):
            prev = pd.read_csv(out_csv)
            prev = prev[[k != (row["seed"], row["arm"]) for k in zip(prev["seed"], prev["arm"])]]
            df = pd.concat([prev, df], ignore_index=True)
        df.to_csv(out_csv + ".tmp", index=False)
        os.replace(out_csv + ".tmp", out_csv)


def run(job):
    if job.mode == "noise":
        nm_file = NOISE_FILE.format(backend=job.backend)
        if os.path.exists(nm_file):
            os.environ.setdefault("SEQUENT_NOISE_MODEL_FILE", nm_file)
        else:
            print(f"  [WARN] {nm_file} not found: noise model fetched live "
                  f"(run `prepare --freeze-backend` so all jobs share one calibration).")

    rows_dir = ROWS_R1_DIR if job.r1_regime else ROWS_DIR
    os.makedirs(rows_dir, exist_ok=True)
    job_id = f"{job.dataset}_{job.model}_{job.fs}_reps{job.reps}_{job.mode}"
    out_csv = os.path.join(rows_dir, f"{job_id}.csv")
    done = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["seed"], prev["arm"]))
    tasks = [(BASE_SEED + run_i, topo) for run_i in job.runs for topo in job.topologies
             if (BASE_SEED + run_i, topo) not in done]

    if job.r1_regime:
        # Execution regime of the R1 baselines (main.py, logged as "30 tasks on 15 parallel
        # workers (1 thread(s) per worker)"): every (seed, topology) of the configuration in a
        # process pool on one node, SEQUENT_THREADS_PER_WORKER=1 in the workers.
        from concurrent.futures import ProcessPoolExecutor, as_completed
        os.environ["SEQUENT_THREADS_PER_WORKER"] = "1"
        print(f"== {job_id}  {len(tasks)} tasks on {job.workers} parallel workers (1 thread per worker)",
              flush=True)
        with ProcessPoolExecutor(max_workers=job.workers) as executor:
            futures = [executor.submit(_r1_task, (job, seed, topo)) for seed, topo in tasks]
            for fut in as_completed(futures):
                row = fut.result()
                _save_row(out_csv, row)
                print(f"  [seed {row['seed']}] {row['arm']:<6} acc={row['accuracy']:.4f} "
                      f"f1={row['f1_macro']:.4f} train={row['training_time']:.1f}s", flush=True)
    else:
        print(f"== {job_id}  runs={job.runs}  threads={t._threads_per_worker()}", flush=True)
        for seed, topo in tasks:
            # written after every arm: jobs can be resumed
            row = _fit_row(job, seed, topo)
            _save_row(out_csv, row)
            print(f"  [seed {seed}] {topo:<6} acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} "
                  f"train={row['training_time']:.1f}s 2Q={row['two_qubit_gates']}", flush=True)
    print(f"  rows → {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--datasets", nargs="*", choices=list(DATASETS))
    p.add_argument("--backend", default="ibm_pittsburgh")
    p.add_argument("--freeze-backend", action="store_true",
                   help="save the backend noise model (needs the IBM account)")
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("plan")
    p.add_argument("--mode", choices=["statevector", "noise"], required=True)
    p.add_argument("--r1-regime", action="store_true")

    p = sub.add_parser("run")
    p.add_argument("--dataset", choices=list(DATASETS), required=True)
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--fs", choices=FS_METHODS, required=True)
    p.add_argument("--reps", type=int, choices=REPS_GRID, required=True)
    p.add_argument("--mode", choices=["statevector", "noise"], required=True)
    p.add_argument("--runs", type=_parse_runs, default=list(range(N_RUNS)))
    p.add_argument("--backend", default="ibm_pittsburgh")
    p.add_argument("--topologies", nargs="+", choices=TOPOLOGIES, default=TOPOLOGIES,
                   help="run in this order; each arm is saved as soon as it finishes")
    p.add_argument("--r1-regime", action="store_true",
                   help="R1 baselines' regime: all (seed, topology) tasks in a pool of --workers "
                        "processes with 1 thread each; rows → results_r2/rows_r1regime/")
    p.add_argument("--workers", type=int, default=R1_WORKERS)

    args = ap.parse_args()
    {"prepare": prepare, "plan": plan, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
