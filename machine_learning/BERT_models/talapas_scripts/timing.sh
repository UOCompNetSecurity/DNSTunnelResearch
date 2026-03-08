#!/bin/bash
#SBATCH --job-name=dns_bert_timing
#SBATCH --output=/projects/cis431_531/khogan/DNS_project/machine_learning/logs/%x_%j.out
#SBATCH --error=/projects/cis431_531/khogan/DNS_project/machine_learning/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Usage: sbatch timing.sh <model> <device> <safe_query_file>
# MODEL options: Default, Fast, HighAccuracy, LowMemory, Regularized, TwoPass
# DEVICE options: cpu, gpu
MODEL=${1}
DEVICE=${2}
SAFE_QUERY_FILE=${3}

EXPECTED_ARGS=3
if [ $# -ne $EXPECTED_ARGS ]; then
    echo "Error: This script requires exactly $EXPECTED_ARGS arguments."
    echo "Usage: sbatch timing.sh <model> <device> <safe_query_file>"
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
    echo "Error: Invalid MODEL '$MODEL'"    # Fix 2: was printing $ARG_TYPE instead of $MODEL
    echo "Valid options: Default, Fast, HighAccuracy, LowMemory, Regularized, TwoPass"
    exit 1
fi

# Validate and configure device
if [ "$DEVICE" == "gpu" ]; then
    echo "Test"
    scontrol update jobid=$SLURM_JOB_ID partition=gpu
    scontrol update jobid=$SLURM_JOB_ID gres=gpu:1
elif [ "$DEVICE" == "cpu" ]; then
    scontrol update jobid=$SLURM_JOB_ID partition=compute
    export CUDA_VISIBLE_DEVICES=""          # Force CPU-only mode
else
    echo "Error: Invalid DEVICE '$DEVICE'"
    echo "Valid options: cpu, gpu"
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
echo "Device:     $DEVICE"
echo "Start time: $(date)"
echo "========================================"

if [ "$DEVICE" == "gpu" ]; then
    srun python3 timing.py cuda $MODEL_DIR $SAFE_QUERY_FILE   
elif [ "$DEVICE" == "cpu" ]; then
    srun python3 timing.py cpu $MODEL_DIR $SAFE_QUERY_FILE   
else
    echo "Error: Invalid DEVICE '$DEVICE'"
    exit 1
fi

echo "========================================"
echo "End time: $(date)"
echo "========================================"