#!/bin/bash
# GRPO Model Evaluation Script
# ============================

set -e

# Default values
MODEL_PATH="${1:-outputs/grpo-default}"
DATASET="${DATASET:-gsm8k}"
SPLIT="${SPLIT:-test}"
OUTPUT_FILE="${OUTPUT_FILE:-}"
NUM_SAMPLES="${NUM_SAMPLES:-}"

# Parse arguments
shift 2>/dev/null || true

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -s|--split)
            SPLIT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 MODEL_PATH [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  MODEL_PATH             Path to the trained model"
            echo ""
            echo "Options:"
            echo "  -d, --dataset DATASET  Dataset to evaluate on (default: gsm8k)"
            echo "  -s, --split SPLIT      Dataset split (default: test)"
            echo "  -o, --output FILE      Output file for results"
            echo "  -n, --num-samples NUM  Number of samples to evaluate"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "GRPO Model Evaluation"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET ($SPLIT)"
echo "=========================================="

# Build command
CMD="grpo-eval $MODEL_PATH -d $DATASET -s $SPLIT"

if [[ -n "$OUTPUT_FILE" ]]; then
    CMD="$CMD -o $OUTPUT_FILE"
fi

if [[ -n "$NUM_SAMPLES" ]]; then
    CMD="$CMD -n $NUM_SAMPLES"
fi

# Run evaluation
$CMD

echo "Evaluation complete!"
