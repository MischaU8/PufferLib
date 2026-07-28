#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODE="${1:-status}"
if [[ "$MODE" != "remote-run" ]]; then
    : "${VAST_HOST:?set VAST_HOST, for example root@vast-host}"
    : "${VAST_DIR:?set VAST_DIR to an isolated remote checkout}"
    : "${GPU_ID:?set GPU_ID after inspecting the remote GPU inventory}"
fi

RUN_ID="${RUN_ID:-gc-lr-learning-curve-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-33554432}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-16}"
EVAL_GAMES="${EVAL_GAMES:-512}"
TRAIN_SEED="${TRAIN_SEED:-101}"
EVAL_SEED="${EVAL_SEED:-9101}"
LEARNING_RATES="${LEARNING_RATES:-0.0001 0.001 0.005 0.015}"
ROLES="${ROLES:-guerrilla coin}"
LOCAL_ARTIFACT_DIR="${LOCAL_ARTIFACT_DIR:-.runtime/lr-learning-curve/$RUN_ID}"
REMOTE_RUN_DIR=".runtime/lr-learning-curve/$RUN_ID"
REMOTE_SOURCE_MARKER=".runtime/gc-performance-source-head"

ssh_run() {
    ssh "$VAST_HOST" "$@"
}

sync_source() {
    local archive head remote_head source_sha
    local explicit_files=(
        build.sh
        config/guerrillacheckers.ini
        ocean/guerrillacheckers/guerrillacheckers.c
        ocean/guerrillacheckers/guerrillacheckers.h
        tools/guerrillacheckers/run_lr_learning_curve_pilot.sh
    )
    head=$(git rev-parse HEAD)
    ssh_run "mkdir -p '$VAST_DIR'"
    remote_head=$(ssh_run "cat '$VAST_DIR/$REMOTE_SOURCE_MARKER' 2>/dev/null || true")
    if [[ "$remote_head" != "$head" ]]; then
        archive=$(mktemp /tmp/gc-performance-lab-source.XXXXXX)
        git archive --format=tar --output="$archive" HEAD
        scp "$archive" "$VAST_HOST:$VAST_DIR/source.tar"
        ssh_run "cd '$VAST_DIR' && tar -xf source.tar && rm source.tar"
        rm -f "$archive"
    fi
    rsync -azR "${explicit_files[@]}" "$VAST_HOST:$VAST_DIR/"
    source_sha=$(shasum -a 256 "${explicit_files[@]}" |
        shasum -a 256 | awk '{print $1}')
    ssh_run "mkdir -p '$VAST_DIR/$REMOTE_RUN_DIR' && \
        printf '%s\\n' '$head' >'$VAST_DIR/$REMOTE_SOURCE_MARKER' && \
        printf '%s\\n' '$head' >'$VAST_DIR/$REMOTE_RUN_DIR/source-head' && \
        printf '%s\\n' '$source_sha' >'$VAST_DIR/$REMOTE_RUN_DIR/source-files-sha256'"
}

remote_status() {
    ssh_run "cd '$VAST_DIR' && \
        if test -f '$REMOTE_RUN_DIR/pid'; then \
            pid=\$(cat '$REMOTE_RUN_DIR/pid'); \
            if kill -0 \"\$pid\" 2>/dev/null; then echo status=running pid=\"\$pid\"; \
            else echo status=stopped pid=\"\$pid\"; fi; \
        else echo status=not-started; fi; \
        test ! -f '$REMOTE_RUN_DIR/stage' || sed 's/^/stage=/' '$REMOTE_RUN_DIR/stage'; \
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
          --format=csv,noheader; \
        test ! -f '$REMOTE_RUN_DIR/results.tsv' || tail -n 20 '$REMOTE_RUN_DIR/results.tsv'; \
        test ! -f '$REMOTE_RUN_DIR/run.log' || tail -n 25 '$REMOTE_RUN_DIR/run.log'"
}

remote_stop() {
    ssh_run "cd '$VAST_DIR' && \
        if test -f '$REMOTE_RUN_DIR/pid'; then \
            pid=\$(cat '$REMOTE_RUN_DIR/pid'); \
            pkill -TERM -P \"\$pid\" 2>/dev/null || true; \
            kill -TERM \"\$pid\" 2>/dev/null || true; \
            printf '%s\\n' stopped-by-user >'$REMOTE_RUN_DIR/stage'; \
        fi"
    remote_status
}

remote_run() {
    local run_dir=$1
    local run_id=$2
    local total_timesteps=$3
    local checkpoint_interval=$4
    local eval_games=$5
    local train_seed=$6
    local eval_seed=$7
    local learning_rates=$8
    local roles=$9
    local gpu_id=${10}

    export PATH="/usr/local/cuda-13.0/bin:$PATH"
    mkdir -p "$run_dir" .runtime/checkpoints .runtime/logs
    local run_key
    run_key=$(printf '%s' "$run_id" | sha256sum | awk '{print substr($1, 1, 8)}')
    printf 'role\tlearning_rate\ttimesteps\topponent\tperf\tcheckpoint\tsha256\n' \
        >"$run_dir/results.tsv"
    {
        printf 'run_id=%s\n' "$run_id"
        printf 'total_timesteps=%s\n' "$total_timesteps"
        printf 'checkpoint_interval=%s\n' "$checkpoint_interval"
        printf 'eval_games=%s\n' "$eval_games"
        printf 'train_seed=%s\n' "$train_seed"
        printf 'eval_seed=%s\n' "$eval_seed"
        printf 'learning_rates=%s\n' "$learning_rates"
        printf 'roles=%s\n' "$roles"
        printf 'gpu_id=%s\n' "$gpu_id"
    } >"$run_dir/manifest.txt"

    evaluate_checkpoint() {
        local side=$1
        local opponent=$2
        local iterations=$3
        local checkpoint=$4
        local output=$5

        CUDA_VISIBLE_DEVICES="$gpu_id" ./puffer eval_bot guerrillacheckers \
            "base.load_model_path=$checkpoint" \
            "base.num_games=$eval_games" \
            "base.eval_agents=$eval_games" \
            "base.seed=$eval_seed" \
            base.burnin_games=0 \
            base.gpu_offset=0 \
            selfplay.enabled=0 \
            vec.num_frozen_banks=0 \
            vec.frozen_bank_pct=0 \
            env.selfplay=0 \
            "env.side=$side" \
            "env.opponent=$opponent" \
            "env.mcts_iterations=$iterations" \
            env.mcts_rollout=1 \
            2>&1 | tee "$output" >&2

        tr '\r' '\n' <"$output" |
            awk 'match($0, /perf=[0-9.]+/) {
                value=substr($0, RSTART + 5, RLENGTH - 5)
            } END {
                if (value == "") exit 1
                print value
            }'
    }

    ./build.sh guerrillacheckers --fast
    ./build.sh guerrillacheckers --float

    local role learning_rate side role_key lr_key arm checkpoint_dir
    local checkpoint basename timesteps perf sha eval_file
    for role in $roles; do
        if [[ "$role" == "guerrilla" ]]; then
            side=1
            role_key=g
        elif [[ "$role" == "coin" ]]; then
            side=2
            role_key=c
        else
            echo "error: unsupported role $role" >&2
            exit 2
        fi
        for learning_rate in $learning_rates; do
            lr_key=${learning_rate//./p}
            arm="gc-${run_key}-${role_key}-lr${lr_key}-s${train_seed}"
            if [[ ${#arm} -gt 63 ]]; then
                echo "error: native run id exceeds 63 characters: $arm" >&2
                exit 2
            fi
            checkpoint_dir=".runtime/checkpoints/guerrillacheckers/$arm"
            printf '%s\n' "$arm" >"$run_dir/stage"

            CUDA_VISIBLE_DEVICES="$gpu_id" ./puffer train guerrillacheckers \
                "base.run_id=$arm" \
                "base.seed=$train_seed" \
                base.gpu_offset=0 \
                base.checkpoint_dir=.runtime/checkpoints \
                base.log_dir=.runtime/logs \
                "base.checkpoint_interval=$checkpoint_interval" \
                base.eval_episodes=0 \
                selfplay.enabled=0 \
                vec.num_frozen_banks=0 \
                vec.frozen_bank_pct=0 \
                vec.num_threads=32 \
                vec.total_agents=4096 \
                env.selfplay=0 \
                "env.side=$side" \
                env.opponent=0 \
                env.mcts_iterations=1 \
                env.mcts_rollout=1 \
                env.capture_reward_guerrilla=0.05 \
                env.capture_reward_coin=0.03 \
                "train.total_timesteps=$total_timesteps" \
                "train.learning_rate=$learning_rate" \
                train.min_lr_ratio=0.1 \
                train.gamma=0.98 \
                2>&1 | tee "$run_dir/$arm.train.log"

            mapfile -t checkpoints < <(
                find "$checkpoint_dir" -maxdepth 1 -type f -name '*.bin' |
                    sort -V
            )
            if [[ ${#checkpoints[@]} -eq 0 ]]; then
                echo "error: no checkpoints produced for $arm" >&2
                exit 1
            fi
            for checkpoint in "${checkpoints[@]}"; do
                basename=$(basename "$checkpoint" .bin)
                timesteps=$((10#$basename))
                eval_file="$run_dir/$arm-$basename-random.eval.log"
                perf=$(evaluate_checkpoint  "$side" 0 1 "$checkpoint" "$eval_file")
                sha=$(sha256sum "$checkpoint" | awk '{print $1}')
                printf '%s\t%s\t%d\trandom\t%s\t%s\t%s\n' \
                    "$role" "$learning_rate" "$timesteps" "$perf" \
                    "$checkpoint" "$sha" | tee -a "$run_dir/results.tsv"
            done

            checkpoint=${checkpoints[$((${#checkpoints[@]} - 1))]}
            for opponent_spec in "greedy:1:1" "mcts32:2:32"; do
                IFS=: read -r opponent_name opponent_id iterations \
                    <<<"$opponent_spec"
                eval_file="$run_dir/$arm-final-$opponent_name.eval.log"
                perf=$(evaluate_checkpoint "$side" "$opponent_id" "$iterations" \
                    "$checkpoint" "$eval_file")
                sha=$(sha256sum "$checkpoint" | awk '{print $1}')
                printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\n' \
                    "$role" "$learning_rate" "$total_timesteps" \
                    "$opponent_name" "$perf" "$checkpoint" "$sha" |
                    tee -a "$run_dir/results.tsv"
            done
        done
    done

    printf '%s\n' complete >"$run_dir/stage"
    echo "learning-rate curve pilot complete"
}

case "$MODE" in
    sync)
        sync_source
        ;;
    start)
        sync_source
        ssh_run "cd '$VAST_DIR' && mkdir -p '$REMOTE_RUN_DIR' && \
            if test -f '$REMOTE_RUN_DIR/pid' && \
                kill -0 \$(cat '$REMOTE_RUN_DIR/pid') 2>/dev/null; then \
                echo 'error: learning curve already running' >&2; exit 1; fi; \
            nohup ./tools/guerrillacheckers/run_lr_learning_curve_pilot.sh remote-run \
                '$REMOTE_RUN_DIR' '$RUN_ID' '$TOTAL_TIMESTEPS' \
                '$CHECKPOINT_INTERVAL' '$EVAL_GAMES' '$TRAIN_SEED' \
                '$EVAL_SEED' '$LEARNING_RATES' '$ROLES' '$GPU_ID' \
                >'$REMOTE_RUN_DIR/run.log' 2>&1 </dev/null & \
            echo \$! >'$REMOTE_RUN_DIR/pid'; \
            echo started pid=\$(cat '$REMOTE_RUN_DIR/pid')"
        remote_status
        ;;
    status)
        remote_status
        ;;
    stop)
        remote_stop
        ;;
    download)
        mkdir -p "$LOCAL_ARTIFACT_DIR"
        rsync -az "$VAST_HOST:$VAST_DIR/$REMOTE_RUN_DIR/" "$LOCAL_ARTIFACT_DIR/"
        echo "downloaded to $LOCAL_ARTIFACT_DIR"
        ;;
    remote-run)
        if [[ $# -ne 11 ]]; then
            echo "error: invalid remote-run arguments" >&2
            exit 2
        fi
        remote_run "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}"
        ;;
    *)
        echo "usage: $0 sync|start|status|stop|download" >&2
        exit 2
        ;;
esac
