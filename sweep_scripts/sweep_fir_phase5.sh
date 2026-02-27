#!/bin/bash
# MuLoCo K=1 Phase 5: Optimized warmdown schedule
# Key insight: 12-epoch run (3.403) had sharper warmdown than 20-epoch (3.412)
# Strategy: more full-LR epochs but keep warmdown period sharp (6 epochs)
set -e

WANDB_KEY="REDACTED"
ACCOUNT="rrg-bengioy-ad_gpu"

declare -a CONFIGS=(
    # format: EPOCHS OLR OMOM H LRM WD WARMDOWN FINAL_LR
    # 16 epochs, warmdown=0.375 → 6 epochs warmdown starting at ep 10
    "16 0.5 0.5 5 0.25 1.6 0.375 0.0"
    "16 0.5 0.5 10 0.25 1.6 0.375 0.0"
    # 18 epochs, warmdown=0.333 → 6 epochs warmdown starting at ep 12
    "18 0.5 0.5 5 0.25 1.6 0.333 0.0"
    "18 0.5 0.5 10 0.25 1.6 0.333 0.0"
    # 14 epochs, warmdown=0.43 → 6 epochs warmdown starting at ep 8
    "14 0.5 0.5 5 0.25 1.6 0.43 0.0"
    "14 0.5 0.5 10 0.25 1.6 0.43 0.0"
    # 20 epochs, warmdown=0.3 → 6 epochs warmdown starting at ep 14
    "20 0.5 0.5 5 0.25 1.6 0.3 0.0"
    "20 0.5 0.5 10 0.25 1.6 0.3 0.0"
)

for cfg in "${CONFIGS[@]}"; do
    read -r EPOCHS OLR OMOM H LRM WD WARMDOWN FINALLR <<< "$cfg"
    RUN_NAME="p5_e${EPOCHS}_olr${OLR}_omom${OMOM}_H${H}_lrm${LRM}_wd${WD}_wdr${WARMDOWN}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=logs/${RUN_NAME}_%j.out

set -e
echo "=== Job \$SLURM_JOB_ID on \$(hostname) ==="
echo "Config: epochs=${EPOCHS}, olr=${OLR}, omom=${OMOM}, H=${H}, lrm=${LRM}, wd=${WD}, warmdown=${WARMDOWN}"
nvidia-smi -L

module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0
python -m venv --system-site-packages \$SLURM_TMPDIR/venv
source \$SLURM_TMPDIR/venv/bin/activate
pip install --no-index torch tiktoken datasets triton wandb numpy 2>&1 | tail -3

export TIKTOKEN_CACHE_DIR=~/scratch/slowrun/tiktoken_cache
export HF_DATASETS_OFFLINE=1
export WANDB_API_KEY=${WANDB_KEY}

cp -r ~/scratch/slowrun \$SLURM_TMPDIR/slowrun
cd \$SLURM_TMPDIR/slowrun

torchrun --standalone --nproc_per_node=4 train_muloco.py \\
    --run-name ${RUN_NAME} \\
    --wandb_group phase5_warmdown \\
    --outer-lr ${OLR} \\
    --outer-momentum ${OMOM} \\
    --sync-interval ${H} \\
    --num-epochs ${EPOCHS} \\
    --lr_multiplier ${LRM} \\
    --weight-decay ${WD} \\
    --warmdown-ratio ${WARMDOWN} \\
    --final-lr-frac ${FINALLR} \\
    --device-batch-size 4 \\
    --dropout 0.1

echo "=== Done: \$(date) ==="
EOF

    echo "Submitted: ${RUN_NAME}"
done

echo "Phase 5: submitted ${#CONFIGS[@]} jobs"
