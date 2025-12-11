#!/bin/bash
# GRPO Training Script
# ====================

set -e

# Default values
CONFIG="${CONFIG:-configs/default.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo-run}"
MODEL="${MODEL:-}"
DATASET="${DATASET:-}"
GPUS="${GPUS:-1}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config CONFIG    Path to config file (default: configs/default.yaml)"
            echo "  -o, --output DIR       Output directory (default: outputs/grpo-run)"
            echo "  -m, --model MODEL      Model name (overrides config)"
            echo "  -d, --dataset DATASET  Dataset name (overrides config)"
            echo "  -g, --gpus NUM         Number of GPUs to use (default: 1)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build command
CMD="grpo-train -c $CONFIG -o $OUTPUT_DIR"

if [[ -n "$MODEL" ]]; then
    CMD="$CMD -m $MODEL"
fi

if [[ -n "$DATASET" ]]; then
    CMD="$CMD -d $DATASET"
fi

# Run training
echo "=========================================="
echo "GRPO Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "=========================================="

if [[ "$GPUS" -gt 1 ]]; then
    echo "Running distributed training with $GPUS GPUs..."
    torchrun --nproc_per_node=$GPUS -m grpo_trainer.cli train -c $CONFIG -o $OUTPUT_DIR
else
    echo "Running single-GPU training..."
    $CMD
fi

echo "Training complete!"
