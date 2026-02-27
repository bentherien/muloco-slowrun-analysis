#!/bin/bash
# MuLoCo K=1 Phase 2 on tamia H100 nodes (4 GPUs)
set -e

WANDB_KEY="REDACTED"

# Key runs from Phase 2 on H100 for faster scheduling
declare -a CONFIGS=(
    "15 0.7 0.6 20 0.25 1.6"   # 15 epochs, best config
    "20 0.7 0.6 20 0.25 1.6"   # 20 epochs, best config
    "15 0.7 0.6 20 0.30 1.6"   # higher inner LR
    "15 0.5 0.5 20 0.25 1.6"   # lower outer
)

for cfg in "${CONFIGS[@]}"; do
    read -r EPOCHS OLR OMOM H LRM WD <<< "$cfg"
    RUN_NAME="p2h_e${EPOCHS}_olr${OLR}_omom${OMOM}_H${H}_lrm${LRM}_wd${WD}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --account=aip-irina
#SBATCH --gpus-per-node=h100:4
#SBATCH --nodes=1
#SBATCH --mem=450G
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
export WANDB_MODE=offline

cp -r ~/scratch/slowrun \$SLURM_TMPDIR/slowrun
cd \$SLURM_TMPDIR/slowrun

torchrun --standalone --nproc_per_node=4 train_muloco.py \
    --run-name ${RUN_NAME} \
    --wandb_group phase2_h100 \
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

echo "Phase 2 H100: submitted ${#CONFIGS[@]} jobs"
