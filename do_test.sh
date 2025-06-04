#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 --load_run [RUN_NAME] [other arguments]"
    exit 1
}

LOAD_RUN=""
NUM_ENVS=""
HEADLESS_FLAG=""
REAL_TIME_FLAG=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --headless)
            HEADLESS_FLAG="--headless"
            shift
            ;;
        --real_time)
            REAL_TIME_FLAG="--real_time"
            shift
            ;;
        --load_run)
            if [[ -n "$2" ]]; then
                LOAD_RUN="--load_run $2"
                shift 2
            else
                echo "Error: --load_run requires an argument."
                usage
            fi
            ;;
        --num_envs)
            if [[ -n "$2" ]] && [[ "$2" =~ ^[0-9]+$ ]]; then
                NUM_ENVS="--num_envs $2"
                shift 2
            else
                echo "Error: --num_envs requires a valid integer."
                exit 1
            fi
            ;;
        *)
            OTHER_ARGS+="$1 "
            shift
            ;;
    esac
done

# If LOAD_RUN is still empty, show usage and exit
if [[ -z "$LOAD_RUN" ]]; then
    echo "Error: --load_run is a mandatory argument."
    usage
fi

echo "Testing with $LOAD_RUN"

# Execute the command with --load_run
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task gcrl-play \
    --experiment_name g1_gcrl \
    --video --video_length 500 \
    $HEADLESS_FLAG \
    $REAL_TIME_FLAG \
    $LOAD_RUN \
    $NUM_ENVS \
    $OTHER_ARGS

