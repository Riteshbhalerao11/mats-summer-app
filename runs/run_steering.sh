#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, and pipe failures

module load conda

conda activate myenv
# Run steering model script
# This script demonstrates how to run steer_model.py with all required arguments

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root for consistent paths
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Steering Model Execution Script"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# Set up model and paths
MODEL="llama"
MODEL_PATH="/pscratch/sd/r/ritesh11/temp/Llama-3.2-3B-Instruct"

# Configuration
VEC_TYPE="trust_orthogonal"  # or your vector type

# Base directories (script will construct full paths automatically)
# Base directories (script will construct full paths automatically)
DATASET_NAME="xstest"                              # Name of dataset in eval_data/
EVAL_DATA_DIR="eval_data"                              # Base directory for datasets
PROBE_DIR="probe_checkpoints"                          # Base directory for probe checkpoints
REFUSAL_DIR="refusal_svs"                              # Base directory for refusal vectors
BASE_OUTPUT_DIR="steered-outs"      # Will create: BASE_OUTPUT_DIR/MODEL/VEC_TYPE/DATASET_NAME

# Model dimensions (adjust based on your model)
D_MODEL=3072
MODEL_LEN=28

# Steering configuration
STEER_ON_USER=true
STEER_ON_GENERATION=true
STEER_ON_SYSTEM=true
RANDOM_VECTOR=false  # Set to true to use random vectors with matching mean/std

# Trust vector config (optional)
PROMPT_CATEGORY="no_refusal"
ACT_NAME="PRE_ASSISTANT"

# Generation settings
BATCH_SIZE=50
MAX_NEW_TOKENS=3000
TEMPERATURE=0.0
TOP_P=1
DTYPE="bfloat16"

# Seed(s) - can be single or multiple
SEEDS=(42)

# Multiple seeds for robustness (uncomment to use)
# SEEDS=(42 43 44 45 46)

# Many seeds for statistical analysis (uncomment to use)
# SEEDS=($(seq 0 30))  # Seeds 1 through 10

# Steering parameters
LAYER_RANGE=(11 11)  # First and last layer to consider
NUM_LAYERS=(1)    # Number of layers to steer
STRENGTH=(-3)  # Positive for eval (increase awareness), negative for deploy (decrease awareness)
# EXACT_MULTIPLIERS=(1.5 2.5 3.5 4.5) # Uncomment to use specific multipliers per layer (must match NUM_LAYERS length)

# Construct full paths that will be used
FULL_OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL/$VEC_TYPE/$DATASET_NAME"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Vec type: $VEC_TYPE"
echo "  Dataset: $DATASET_NAME"
echo "  Eval Data Dir: $EVAL_DATA_DIR"
echo "  Probe Dir: $PROBE_DIR"
echo "  Refusal Dir: $REFUSAL_DIR"
echo "  Random vector: $RANDOM_VECTOR"
echo "  Seeds: ${SEEDS[@]} (${#SEEDS[@]} seed(s))"
echo "  Num layers: ${NUM_LAYERS[@]}"
echo "  Strengths: ${STRENGTH[@]}"
if [ "$RANDOM_VECTOR" = true ]; then
  echo "  Output: $FULL_OUTPUT_DIR/L_{layers}_S_{s}_random/"
else
  echo "  Output: $FULL_OUTPUT_DIR/L_{layers}_S_{s}/"
fi
echo ""

# Build the base command (conditionally add strength or exact_multipliers)
if [ -n "${EXACT_MULTIPLIERS:-}" ]; then
    STRENGTH_ARG="--exact_multipliers ${EXACT_MULTIPLIERS[@]}"
else
    STRENGTH_ARG="--strength ${STRENGTH[@]}"
fi

python steer_model.py \
  --model "$MODEL" \
  --model_path "$MODEL_PATH" \
  --vec_type "$VEC_TYPE" \
  --out_dir "$BASE_OUTPUT_DIR" \
  --dataset_name "$DATASET_NAME" \
  --eval_data_dir "$EVAL_DATA_DIR" \
  --probe_dir "$PROBE_DIR" \
  --refusal_dir "$REFUSAL_DIR" \
  --prompt_category "$PROMPT_CATEGORY" \
  --act_name "$ACT_NAME" \
  --dtype "$DTYPE" \
  --seed ${SEEDS[@]} \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --d_model $D_MODEL \
  --model_len $MODEL_LEN \
  --layer_range ${LAYER_RANGE[@]} \
  --num_layers ${NUM_LAYERS[@]} \
  $STRENGTH_ARG \
  $([ "$STEER_ON_USER" = true ] && echo "--steer_on_user" || echo "") \
  $([ "$STEER_ON_GENERATION" = true ] && echo "--steer_on_generation" || echo "") \
  $([ "$STEER_ON_SYSTEM" = true ] && echo "--steer_on_system" || echo "") \
  $([ "$RANDOM_VECTOR" = true ] && echo "--random-vector" || echo "")

echo ""
echo "=========================================="
echo "Steering execution completed!"
echo "=========================================="
