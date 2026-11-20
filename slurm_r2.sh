#!/bin/bash
# R2 fixed topologies on the k=5 features — one line of the command file per array task.
# R1 regime: one node, 15 CPUs, the (seed, topology) tasks of a configuration in a pool of
# 15 processes with 1 thread each, as the R1 baselines ran.
#   python r2_fair.py plan --mode statevector --r1-regime
#   python r2_fair.py plan --mode noise --r1-regime
#   sbatch --array=1-60 slurm_r2.sh cmds_r2_r1regime_statevector.txt
#   sbatch --array=1-60 slurm_r2.sh cmds_r2_r1regime_noise.txt
#SBATCH --job-name=SEQUENT_R2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --partition=standard
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

CMDS=${1:?usage: sbatch --array=1-N slurm_r2.sh cmds_file.txt}

cd "${SLURM_SUBMIT_DIR}"

module purge
module load Miniconda3/24.5.0
eval "$(conda shell.bash hook)"
conda activate /home/frarodriguez/conda/envs/SEQUENT_ENV

export SEQUENT_THREADS_PER_WORKER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${CMDS}")
echo "[$(date)] $(hostname) cpus=${SLURM_CPUS_PER_TASK} :: ${CMD}"
eval "${CMD}"
echo "[$(date)] exit=$?"
