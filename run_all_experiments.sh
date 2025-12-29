#!/bin/bash
# Run CAB experiments with different models, AST vs non-AST
# Framework: OpenHands
# Uses 'dataset' command which runs BOTH generation AND evaluation in one step

set -e

# Source AWS credentials
source /home/mysoo/CodeStruct/test.txt

cd /home/mysoo/CodeStruct/CodeAssistBench
source venv/bin/activate
export PYTHONPATH="/home/mysoo/CodeStruct/CodeAssistBench/src:$PYTHONPATH"

DATASET="dataset/cab_verified.jsonl"
OUTPUT_BASE="results"

mkdir -p "$OUTPUT_BASE"

declare -a MODELS=("gpt5" "gpt5mini" "gptnano" "qwen32b")

# Parse arguments
AST_ONLY=false
NOAST_ONLY=false
SPECIFIC_MODEL=""
SEQUENTIAL=false
GENERATION_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ast-only) AST_ONLY=true; shift ;;
        --noast-only) NOAST_ONLY=true; shift ;;
        --model) SPECIFIC_MODEL="$2"; shift 2 ;;
        --sequential) SEQUENTIAL=true; shift ;;
        --generation-only) GENERATION_ONLY=true; shift ;;
        --help)
            echo "Usage: $0 [--ast-only|--noast-only] [--model MODEL] [--sequential] [--generation-only]"
            echo "  --generation-only: Run only generation (no judge evaluation)"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

[[ -n "$SPECIFIC_MODEL" ]] && MODELS=("$SPECIFIC_MODEL")

echo "=== CAB OpenHands Experiments ==="
echo "Models: ${MODELS[*]}"
echo "Mode: $( $GENERATION_ONLY && echo 'Generation only' || echo 'Generation + Evaluation (judge)' )"

PIDS=()

run_experiment() {
    local model=$1
    local ast_flag=$2
    local suffix=$3
    
    local OUTPUT_DIR="${OUTPUT_BASE}/${model}_${suffix}"
    local LOG_FILE="${OUTPUT_BASE}/${model}_${suffix}.log"
    
    mkdir -p "$OUTPUT_DIR"
    echo "Starting: $model ($suffix)"
    
    local CMD
    if $GENERATION_ONLY; then
        # Generation only (no judge)
        CMD="python -c 'from cab_evaluation.cli import main; main()' generation-dataset $DATASET \
            --output $OUTPUT_DIR/generation_results.jsonl \
            --agent-models '{\"maintainer\": \"$model\", \"user\": \"sonnet45\"}' \
            --agent-framework '{\"maintainer\": \"openhands\"}' \
            --resume \
            $ast_flag"
    else
        # Complete workflow: generation + evaluation (judge) in one step
        CMD="python -c 'from cab_evaluation.cli import main; main()' dataset $DATASET \
            --output-dir $OUTPUT_DIR \
            --agent-models '{\"maintainer\": \"$model\", \"user\": \"sonnet45\", \"judge\": \"sonnet45\"}' \
            --agent-framework '{\"maintainer\": \"openhands\"}' \
            --resume \
            $ast_flag"
    fi
    
    if $SEQUENTIAL; then
        eval "$CMD" 2>&1 | tee "$LOG_FILE"
    else
        eval "$CMD" 2>&1 | tee "$LOG_FILE" &
        PIDS+=($!)
        sleep 3
    fi
}

for model in "${MODELS[@]}"; do
    echo "========================================"
    echo "Model: $model"
    echo "========================================"
    
    # AST enabled
    if ! $NOAST_ONLY; then
        run_experiment "$model" "" "openhands_ast"
    fi
    
    # AST disabled
    if ! $AST_ONLY; then
        run_experiment "$model" "--disable-ast-tools" "openhands_noast"
    fi
done

if ! $SEQUENTIAL && [[ ${#PIDS[@]} -gt 0 ]]; then
    echo ""
    echo "PIDs: ${PIDS[*]}"
    echo "Monitor: tail -f ${OUTPUT_BASE}/*.log"
    wait
fi

echo "All experiments completed!"
