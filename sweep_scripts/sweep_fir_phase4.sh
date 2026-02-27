#!/bin/bash
# MuLoCo K=1 Phase 4: Reduced warmdown experiments
set -e

WANDB_KEY="REDACTED"
ACCOUNT="rrg-bengioy-ad_gpu"

# Test reduced warmdown with best configs
# Hypothesis: keeping inner LR higher later helps MuLoCo outer updates
declare -a CONFIGS=(
    # format: EPOCHS OLR OMOM H LRM WD WARMDOWN FINAL_LR
    "20 0.5 0.5 10 0.25 1.6 0.3 0.0"    # H=10, less warmdown
    "20 0.5 0.5 10 0.25 1.6 0.5 0.1"    # H=10, higher final LR
    "20 0.5 0.5 5 0.25 1.6 0.3 0.0"     # H=5, less warmdown
    "20 0.7 0.6 20 0.25 1.6 0.3 0.0"    # H=20 best tamia config, less warmdown
    "20 0.7 0.6 20 0.25 1.6 0.5 0.1"    # H=20, higher final LR
    "20 0.5 0.5 10 0.30 1.6 0.3 0.0"    # H=10, higher inner LR, less warmdown
)

for cfg in "${CONFIGS[@]}"; do
    read -r EPOCHS OLR OMOM H LRM WD WARMDOWN FINALLR <<< "$cfg"
    RUN_NAME="p4_e${EPOCHS}_olr${OLR}_omom${OMOM}_H${H}_lrm${LRM}_wd${WD}_wd${WARMDOWN}_flr${FINALLR}"

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
echo "Config: epochs=${EPOCHS}, olr=${OLR}, omom=${OMOM}, H=${H}, lrm=${LRM}, wd=${WD}, warmdown=${WARMDOWN}, final_lr=${FINALLR}"
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
    --wandb_group phase4_warmdown \\
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

echo "Phase 4: submitted ${#CONFIGS[@]} jobs"
