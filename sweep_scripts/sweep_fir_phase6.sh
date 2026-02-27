#!/bin/bash
# MuLoCo K=1 Phase 6: Micro-tuning around the best config
# Best result: H=5, olr=0.5, omom=0.5, 12ep → 3.403 (gap: 0.001)
# Strategy: tiny tweaks to weight_decay, lr_multiplier, dropout
set -e

WANDB_KEY="REDACTED"
ACCOUNT="rrg-bengioy-ad_gpu"

# format: EPOCHS OLR OMOM H LRM WD DROPOUT
declare -a CONFIGS=(
    "12 0.5 0.5 5 0.25 1.3 0.1"    # lower weight decay
    "12 0.5 0.5 5 0.25 1.0 0.1"    # even lower wd
    "12 0.5 0.5 5 0.20 1.6 0.1"    # lower lr_multiplier
    "12 0.5 0.5 5 0.30 1.6 0.1"    # higher lr_multiplier
    "12 0.5 0.5 10 0.25 1.3 0.1"   # H=10, lower wd
    "12 0.5 0.5 10 0.25 1.0 0.1"   # H=10, even lower wd
    "13 0.5 0.5 5 0.25 1.6 0.1"    # 13 epochs
    "13 0.5 0.5 10 0.25 1.6 0.1"   # 13 epochs, H=10
)

for cfg in "${CONFIGS[@]}"; do
    read -r EPOCHS OLR OMOM H LRM WD DROPOUT <<< "$cfg"
    RUN_NAME="p6_e${EPOCHS}_olr${OLR}_H${H}_lrm${LRM}_wd${WD}_do${DROPOUT}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=logs/${RUN_NAME}_%j.out

set -e
echo "=== Job \$SLURM_JOB_ID on \$(hostname) ==="
echo "Config: epochs=${EPOCHS}, olr=${OLR}, omom=${OMOM}, H=${H}, lrm=${LRM}, wd=${WD}, dropout=${DROPOUT}"
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
    --wandb_group phase6_microtune \\
    --outer-lr ${OLR} \\
    --outer-momentum ${OMOM} \\
    --sync-interval ${H} \\
    --num-epochs ${EPOCHS} \\
    --lr_multiplier ${LRM} \\
    --weight-decay ${WD} \\
    --device-batch-size 4 \\
    --dropout ${DROPOUT}

echo "=== Done: \$(date) ==="
EOF

    echo "Submitted: ${RUN_NAME}"
done

echo "Phase 6: submitted ${#CONFIGS[@]} jobs"
