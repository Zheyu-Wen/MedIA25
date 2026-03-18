#!/bin/bash
#SBATCH -J clinical-multimodal-ODE
#SBATCH -o /home1/08171/zheyw1/Alzh_tau_abeta_model_new/job_sh_scripts/log_clinical0
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -p vm-small
#SBATCH -t 48:00:00

source ~/.bashrc
conda activate
module load launcher
export OMP_NUM_THREADS=4

export LAUNCHER_WORKDIR=/home1/08171/zheyw1/Alzh_tau_abeta_model_new
export LAUNCHER_JOB_FILE=/home1/08171/zheyw1/Alzh_tau_abeta_model_new/job_sh_scripts/paramsrun_clinical0.sh
${LAUNCHER_DIR}/paramrun
