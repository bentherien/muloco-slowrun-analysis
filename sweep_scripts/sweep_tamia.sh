#!/bin/bash
# MuLoCo K=1 hyperparameter sweep on tamia (H200, 8 GPUs per node)
# Phase 1: Sweep outer_lr, outer_momentum, sync_interval
set -e

WANDB_KEY="REDACTED"

# Phase 1a: outer_lr x outer_momentum with fixed sync_interval=20
OUTER_LRS=(0.3 0.5 0.7 1.0)
OUTER_MOMS=(0.4 0.6 0.8)
SYNC_INTERVAL=20

for OLR in "${OUTER_LRS[@]}"; do
  for OMOM in "${OUTER_MOMS[@]}"; do
    RUN_NAME="muloco_olr${OLR}_omom${OMOM}_H${SYNC_INTERVAL}"

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
echo "Config: outer_lr=${OLR}, outer_momentum=${OMOM}, sync_interval=${SYNC_INTERVAL}"
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
export WANDB_MODE=offline

# Copy code and data to fast local storage
cp -r ~/scratch/slowrun \$SLURM_TMPDIR/slowrun
cd \$SLURM_TMPDIR/slowrun

echo ""
echo "=== Running MuLoCo K=1 training ==="
torchrun --standalone --nproc_per_node=8 train_muloco.py \
    --run-name ${RUN_NAME} \
    --wandb_group phase1a \
    --outer-lr ${OLR} \
    --outer-momentum ${OMOM} \
    --sync-interval ${SYNC_INTERVAL} \
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
echo "Phase 1a: submitted ${#OUTER_LRS[@]} x ${#OUTER_MOMS[@]} = $((${#OUTER_LRS[@]} * ${#OUTER_MOMS[@]})) jobs"
