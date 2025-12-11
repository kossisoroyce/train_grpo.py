#!/bin/bash
# GRPO Training with DeepSpeed
# ============================

set -e

# Default values
CONFIG="${CONFIG:-configs/default.yaml}"
DS_CONFIG="${DS_CONFIG:-configs/deepspeed_zero2.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo-deepspeed}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
MASTER_PORT="${MASTER_PORT:-29500}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        --ds-config)
            DS_CONFIG="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --config CONFIG     Path to training config (default: configs/default.yaml)"
            echo "  --ds-config CONFIG      Path to DeepSpeed config (default: configs/deepspeed_zero2.json)"
            echo "  -o, --output DIR        Output directory"
            echo "  -g, --gpus NUM          Number of GPUs (default: all available)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "GRPO Training with DeepSpeed"
echo "=========================================="
echo "Training Config: $CONFIG"
echo "DeepSpeed Config: $DS_CONFIG"
echo "Output: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    -m grpo_trainer.cli train \
    -c $CONFIG \
    -o $OUTPUT_DIR \
    --deepspeed $DS_CONFIG

echo "Training complete!"
