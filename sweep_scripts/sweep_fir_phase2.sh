#!/bin/bash
# MuLoCo K=1 Phase 2 on fir: More epochs and tuned params
set -e

WANDB_KEY="REDACTED"
ACCOUNT="rrg-bengioy-ad_gpu"

# Best findings so far: H=20, olr=0.7, omom=0.6 got 3.457 at 12 epochs
# Try: more epochs, tune inner LR
declare -a CONFIGS=(
    "15 0.7 0.6 20 0.25 1.6"   # 15 epochs
    "20 0.7 0.6 20 0.25 1.6"   # 20 epochs
    "15 0.5 0.5 20 0.25 1.6"   # lower outer params
    "15 0.7 0.6 20 0.30 1.6"   # higher inner LR
)

for cfg in "${CONFIGS[@]}"; do
    read -r EPOCHS OLR OMOM H LRM WD <<< "$cfg"
    RUN_NAME="p2_e${EPOCHS}_olr${OLR}_omom${OMOM}_H${H}_lrm${LRM}_wd${WD}"

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
echo "Config: epochs=${EPOCHS}, olr=${OLR}, omom=${OMOM}, H=${H}, lrm=${LRM}, wd=${WD}"
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

torchrun --standalone --nproc_per_node=4 train_muloco.py \
    --run-name ${RUN_NAME} \
    --wandb_group phase2 \
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

echo "Phase 2: submitted ${#CONFIGS[@]} jobs"
