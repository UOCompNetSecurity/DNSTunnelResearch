#!/bin/bash
#SBATCH --job-name=dns_bert_timing_gpu
#SBATCH --output=/projects/cis431_531/khogan/DNS_project/machine_learning/logs/%x_%j.out
#SBATCH --error=/projects/cis431_531/khogan/DNS_project/machine_learning/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Usage: sbatch timing_gpu.sh <model> <safe_query_file>
# MODEL options: Default, Fast, HighAccuracy, LowMemory, Regularized, TwoPass
MODEL=${1}
SAFE_QUERY_FILE=${2}

EXPECTED_ARGS=2
if [ $# -ne $EXPECTED_ARGS ]; then
    echo "Error: This script requires exactly $EXPECTED_ARGS arguments."
    echo "Usage: sbatch timing_gpu.sh <model> <safe_query_file>"
    exit 1
fi

VALID_MODELS=("Default" "Fast" "HighAccuracy" "LowMemory" "Regularized" "TwoPass")
VALID=false
for VALID_MODEL in "${VALID_MODELS[@]}"; do
    if [ "$MODEL" == "$VALID_MODEL" ]; then
        VALID=true
        break
    fi
done

if [ "$VALID" = false ]; then
    echo "Error: Invalid MODEL '$MODEL'"
    echo "Valid options: Default, Fast, HighAccuracy, LowMemory, Regularized, TwoPass"
    exit 1
fi

#======================================================================
# Environment Setup
#======================================================================
module load miniconda3
source activate dns_bert

export PYTHONNOUSERSITE=1
export LD_PRELOAD=/home/khogan/.conda/envs/dns_bert/lib/libstdc++.so.6
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Paths
PROJECT_DIR=/projects/cis431_531/khogan/DNS_project/machine_learning/BERT_models
MODEL_DIR=$PROJECT_DIR/model_weights/dns_bert_$MODEL
LOG_DIR=/projects/cis431_531/khogan/DNS_project/machine_learning/logs

mkdir -p $LOG_DIR
cd $PROJECT_DIR

#======================================================================
# Timing
#======================================================================
echo "========================================"
echo "Model:      $MODEL"
echo "Device:     gpu"
echo "Start time: $(date)"
echo "========================================"

srun python3 timing.py cuda $MODEL_DIR $SAFE_QUERY_FILE

echo "========================================"
echo "End time: $(date)"
echo "========================================"