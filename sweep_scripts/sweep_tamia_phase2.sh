#!/bin/bash
# MuLoCo K=1 Phase 2: Tuning to beat 3.402 val loss
# Findings from Phase 1: olr=0.7, omom=0.6, H=20 got 3.457 at 12 epochs (still improving)
# Strategy: more epochs + tune inner LR + refine outer params
set -e

WANDB_KEY="REDACTED"

# Phase 2a: More epochs with best config
declare -a CONFIGS=(
    # More epochs, same params
    "15 0.7 0.6 20 0.25 1.6"   # 15 epochs
    "20 0.7 0.6 20 0.25 1.6"   # 20 epochs
    # Higher inner LR
    "15 0.7 0.6 20 0.30 1.6"   # lr_mult 0.3
    "15 0.7 0.6 20 0.35 1.6"   # lr_mult 0.35
    # Lower weight decay
    "15 0.7 0.6 20 0.25 1.0"   # wd 1.0
    "15 0.7 0.6 20 0.25 1.3"   # wd 1.3
    # Adjusted outer params
    "15 0.5 0.5 20 0.25 1.6"   # lower outer
    "15 0.9 0.7 20 0.25 1.6"   # higher outer
    # Smaller sync interval
    "15 0.7 0.6 10 0.25 1.6"   # H=10
    "15 0.5 0.6 10 0.25 1.6"   # H=10, lower olr
)

for cfg in "${CONFIGS[@]}"; do
    read -r EPOCHS OLR OMOM H LRM WD <<< "$cfg"
    RUN_NAME="p2_e${EPOCHS}_olr${OLR}_omom${OMOM}_H${H}_lrm${LRM}_wd${WD}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=aip-irina
#SBATCH --gpus-per-node=h200:8
#SBATCH --nodes=1
#SBATCH --mem=950G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=logs/${RUN_NAME}_%j.out

set -e
echo "=== Job \$SLURM_JOB_ID on \$(hostname) ==="
echo "Start: \$(date)"
echo "Config: epochs=${EPOCHS}, outer_lr=${OLR}, outer_momentum=${OMOM}, H=${H}, lr_mult=${LRM}, wd=${WD}"
nvidia-smi -L

module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0
python -m venv --system-site-packages \$SLURM_TMPDIR/venv
source \$SLURM_TMPDIR/venv/bin/activate
pip install --no-index torch tiktoken datasets triton wandb numpy 2>&1 | tail -3

export TIKTOKEN_CACHE_DIR=~/scratch/slowrun/tiktoken_cache
export HF_DATASETS_OFFLINE=1
export WANDB_API_KEY=${WANDB_KEY}
export WANDB_MODE=offline

cp -r ~/scratch/slowrun \$SLURM_TMPDIR/slowrun
cd \$SLURM_TMPDIR/slowrun

torchrun --standalone --nproc_per_node=8 train_muloco.py \
    --run-name ${RUN_NAME} \
    --wandb_group phase2a \
    --outer-lr ${OLR} \
    --outer-momentum ${OMOM} \
    --sync-interval ${H} \
    --num-epochs ${EPOCHS} \
    --lr_multiplier ${LRM} \
    --weight-decay ${WD} \
    --device-batch-size 4 \
    --dropout 0.1

echo "=== Done: \$(date) ==="
EOF

    echo "Submitted: ${RUN_NAME}"
done

echo ""
echo "Phase 2a: submitted ${#CONFIGS[@]} jobs"
