#!/bin/bash
# MuLoCo K=1 hyperparameter sweep on fir (H100, 4 GPUs per node)
# Phase 1b: Sweep sync_interval with promising outer_lr/momentum combos
set -e

WANDB_KEY="REDACTED"

# Phase 1b: Sweep sync_interval with two promising outer param combos
SYNC_INTERVALS=(5 10 30 50)
# Combo 1: moderate
OLR1=0.7
OMOM1=0.6
# Combo 2: aggressive
OLR2=0.5
OMOM2=0.5

# Determine best account to use
ACCOUNT="rrg-bengioy-ad_gpu"

for H in "${SYNC_INTERVALS[@]}"; do
  for IDX in 1 2; do
    if [ "$IDX" -eq 1 ]; then OLR=$OLR1; OMOM=$OMOM1; fi
    if [ "$IDX" -eq 2 ]; then OLR=$OLR2; OMOM=$OMOM2; fi
    RUN_NAME="muloco_olr${OLR}_omom${OMOM}_H${H}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT}
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=48
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --job-name=${RUN_NAME}
#SBATCH --output=logs/${RUN_NAME}_%j.out

set -e
echo "=== Job \$SLURM_JOB_ID on \$(hostname) ==="
echo "Start: \$(date)"
echo "Config: outer_lr=${OLR}, outer_momentum=${OMOM}, sync_interval=${H}"
nvidia-smi -L

# Load modules
module load python/3.11.5 cuda/12.6 gcc arrow/23.0.1 thrift/0.22.0

# Create virtualenv in fast local storage
python -m venv --system-site-packages \$SLURM_TMPDIR/venv
source \$SLURM_TMPDIR/venv/bin/activate

# Install deps
pip install --no-index torch tiktoken datasets triton wandb 2>&1 | tail -3
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available(), "GPUs:", torch.cuda.device_count())')"

# Environment
export TIKTOKEN_CACHE_DIR=~/scratch/slowrun/tiktoken_cache
export HF_DATASETS_OFFLINE=1
export WANDB_API_KEY=${WANDB_KEY}

# Copy code and data to fast local storage
cp -r ~/scratch/slowrun \$SLURM_TMPDIR/slowrun
cd \$SLURM_TMPDIR/slowrun

echo ""
echo "=== Running MuLoCo K=1 training (4 GPUs, 2x grad accum) ==="
torchrun --standalone --nproc_per_node=4 train_muloco.py \
    --run-name ${RUN_NAME} \
    --wandb_group phase1b \
    --outer-lr ${OLR} \
    --outer-momentum ${OMOM} \
    --sync-interval ${H} \
    --num-epochs 12 \
    --device-batch-size 4 \
    --dropout 0.1

echo ""
echo "=== Done: \$(date) ==="
EOF

    echo "Submitted: ${RUN_NAME}"
  done
done

echo ""
echo "Phase 1b: submitted ${#SYNC_INTERVALS[@]} x 2 = $((${#SYNC_INTERVALS[@]} * 2)) jobs"
