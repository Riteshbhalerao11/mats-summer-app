#!/usr/bin/env bash
set -euo pipefail  # Exit on error, undefined vars, and pipe failures

module load conda
conda activate myenv

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Steering Model Execution Script (Multi-Vec)"
echo "=========================================="
echo "Project root: $PROJECT_ROOT"
echo ""

# -------------------- MODEL --------------------
MODEL="llama"
MODEL_PATH="/pscratch/sd/r/ritesh11/temp/Llama-3.2-3B-Instruct"

# -------------------- VEC TYPES --------------------
VEC_TYPES=(
 trust_orthogonal trust_complete
)

# -------------------- PATHS --------------------
DATASET_NAME="user_description_single"
EVAL_DATA_DIR="eval_data"
PROBE_DIR="probe_checkpoints"
REFUSAL_DIR="refusal_svs"
BASE_OUTPUT_DIR="steered-outs"

# -------------------- MODEL PARAMS --------------------
D_MODEL=3072
MODEL_LEN=28

# -------------------- STEERING FLAGS --------------------
STEER_ON_USER=true
STEER_ON_GENERATION=true
STEER_ON_SYSTEM=true
RANDOM_VECTOR=false

# -------------------- TRUST CONFIG --------------------
PROMPT_CATEGORY="refusal"
ACT_NAME="PRE_ASSISTANT"

# -------------------- GENERATION --------------------
BATCH_SIZE=50
MAX_NEW_TOKENS=3000
TEMPERATURE=0.0
TOP_P=1
DTYPE="bfloat16"

# -------------------- SEEDS --------------------
SEEDS=(42)

# -------------------- STEERING --------------------
LAYER_RANGE=(11 11)
NUM_LAYERS=(1)
STRENGTH=(3)

# -------------------- STRENGTH ARG --------------------
if [ -n "${EXACT_MULTIPLIERS:-}" ]; then
    STRENGTH_ARG="--exact_multipliers ${EXACT_MULTIPLIERS[@]}"
else
    STRENGTH_ARG="--strength ${STRENGTH[@]}"
fi

# ==================== MAIN LOOP ====================
for VEC_TYPE in "${VEC_TYPES[@]}"; do
    echo "------------------------------------------"
    echo "Running vec_type = $VEC_TYPE"
    echo "------------------------------------------"

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
      $([ "$STEER_ON_USER" = true ] && echo "--steer_on_user") \
      $([ "$STEER_ON_GENERATION" = true ] && echo "--steer_on_generation") \
      $([ "$STEER_ON_SYSTEM" = true ] && echo "--steer_on_system") \
      $([ "$RANDOM_VECTOR" = true ] && echo "--random-vector")

done

echo ""
echo "=========================================="
echo "All steering runs completed!"
echo "=========================================="
