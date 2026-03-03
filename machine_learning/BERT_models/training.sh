#!/bin/bash
#SBATCH --job-name=dns_bert_training
#SBATCH --output=/projects/cis431_531/khogan/DNS_project/machine_learning/logs/%x_%j.out
#SBATCH --error=/projects/cis431_531/khogan/DNS_project/machine_learning/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Usage: sbatch training.sh <INPUT_FILE> [ARG_TYPE] [MODEL_NAME]
# ARG_TYPE options: Default, Fast, HighAccuracy, LowMemory, Regularized
# MODEL_NAME options: Default, Pretrained
INPUT_FILE=${1}
ARG_TYPE=${2:-Default}
MODEL_NAME=${3:-Default}

if [ -z "$INPUT_FILE" ]; then
    echo "Error: INPUT_FILE is required"
    echo "Usage: sbatch train_one.sh <INPUT_FILE> [ARG_TYPE] [MODEL_NAME]"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

VALID_TYPES=("Default" "Fast" "HighAccuracy" "LowMemory" "Regularized")
VALID=false
for TYPE in "${VALID_TYPES[@]}"; do
    if [ "$ARG_TYPE" == "$TYPE" ]; then
        VALID=true
        break
    fi
done

if [ "$VALID" = false ]; then
    echo "Error: Invalid ARG_TYPE '$ARG_TYPE'"
    echo "Valid options: Default, Fast, HighAccuracy, LowMemory, Regularized"
    exit 1
fi

VALID_MODELS=("Default" "Pretrained")
VALID_MODEL=false
for MODEL in "${VALID_MODELS[@]}"; do
    if [ "$MODEL_NAME" == "$MODEL" ]; then
        VALID_MODEL=true
        break
    fi
done

if [ "$VALID_MODEL" = false ]; then
    echo "Error: Invalid MODEL_NAME '$MODEL_NAME'"
    echo "Valid options: Default, Pretrained"
    exit 1
fi

#======================================================================
# Environment Setup
#======================================================================
module load miniconda3
conda activate dns_bert

export PYTHONNOUSERSITE=1
export LD_PRELOAD=/home/khogan/.conda/envs/dns_bert/lib/libstdc++.so.6
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Paths
PROJECT_DIR=/projects/cis431_531/khogan/DNS_project/machine_learning/BERT_models
OUTPUT_DIR=$PROJECT_DIR/models/dns_bert_$ARG_TYPE_MODEL_NAME
LOG_DIR=/projects/cis431_531/khogan/DNS_project/machine_learning/logs

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

cd $PROJECT_DIR

#======================================================================
# Train
#======================================================================
echo "========================================"
echo "ARG_TYPE:   $ARG_TYPE"
echo "Model:      $MODEL_NAME"
echo "Input:      $INPUT_FILE"
echo "Output:     $OUTPUT_DIR"
echo "Start time: $(date)"
echo "========================================"

srun python3 training.py $INPUT_FILE $OUTPUT_DIR $ARG_TYPE $MODEL_NAME

echo "========================================"
echo "Finished: $ARG_TYPE"
echo "End time: $(date)"
echo "========================================"