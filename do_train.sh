#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --run_name [RUN_NAME] [other arguments]"
    exit 1
}

RUN_NAME=""
LOG="--logger wandb"
HEADLESS="--headless"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)
            if [[ -n "$2" && ! "$2" =~ ^-- ]]; then
                RUN_NAME="$2"
                shift 2
            else
                echo "Error: --run_name requires a non-empty argument."
                usage
            fi
            ;;
        --no-log)
            LOG=""
            shift 1
            ;;
        --no-headless)
            HEADLESS=""
            shift 1
            ;;
        *)
            OTHER_ARGS+="$1 "
            shift
            ;;
    esac
done

# If RUN_NAME is still empty, show usage and exit
if [[ -z "$RUN_NAME" ]]; then
    echo "Error: --run_name is a mandatory argument."
    usage
fi

echo "Training with $RUN_NAME"

# Execute the command with --run_name
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task gcrl-train \
    --log_project_name gcrl \
    --max_iterations 50000 \
    $LOG \
    $HEADLESS \
    --run_name "$RUN_NAME" $OTHER_ARGS

