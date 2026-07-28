#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODE="${1:-status}"
if [[ "$MODE" != "remote-run" ]]; then
    : "${VAST_HOST:?set VAST_HOST, for example root@vast-host}"
    : "${VAST_DIR:?set VAST_DIR to an isolated remote checkout}"
    : "${GPU_ID:?set GPU_ID after inspecting the remote GPU inventory}"
fi

RUN_ID="${RUN_ID:-gc-objective-continuation-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-33554432}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-16}"
EVAL_GAMES="${EVAL_GAMES:-512}"
TRAIN_SEED="${TRAIN_SEED:-202}"
EVAL_SEED="${EVAL_SEED:-9201}"
LEARNING_RATE="${LEARNING_RATE:-0.005}"
ROLES="${ROLES:-guerrilla coin}"
TREATMENTS="${TREATMENTS:-shaped terminal98 terminal100}"
G_START_CHECKPOINT="${G_START_CHECKPOINT:-}"
G_START_SHA="${G_START_SHA:-}"
C_START_CHECKPOINT="${C_START_CHECKPOINT:-}"
C_START_SHA="${C_START_SHA:-}"
LOCAL_ARTIFACT_DIR="${LOCAL_ARTIFACT_DIR:-.runtime/objective-continuation/$RUN_ID}"
REMOTE_RUN_DIR=".runtime/objective-continuation/$RUN_ID"
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
        tools/guerrillacheckers/run_objective_continuation_pilot.sh
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
    local learning_rate=$8
    local roles=$9
    local treatments=${10}
    local g_start=${11}
    local g_sha=${12}
    local c_start=${13}
    local c_sha=${14}
    local gpu_id=${15}

    export PATH="/usr/local/cuda-13.0/bin:$PATH"
    mkdir -p "$run_dir" .runtime/checkpoints .runtime/logs
    local run_key
    run_key=$(printf '%s' "$run_id" | sha256sum | awk '{print substr($1, 1, 8)}')
    printf 'role\ttreatment\ttimesteps\topponent\tperf\tcheckpoint\tsha256\n' \
        >"$run_dir/results.tsv"
    {
        printf 'run_id=%s\n' "$run_id"
        printf 'total_timesteps=%s\n' "$total_timesteps"
        printf 'checkpoint_interval=%s\n' "$checkpoint_interval"
        printf 'eval_games=%s\n' "$eval_games"
        printf 'train_seed=%s\n' "$train_seed"
        printf 'eval_seed=%s\n' "$eval_seed"
        printf 'learning_rate=%s\n' "$learning_rate"
        printf 'roles=%s\n' "$roles"
        printf 'treatments=%s\n' "$treatments"
        printf 'g_start_checkpoint=%s\n' "$g_start"
        printf 'g_start_sha=%s\n' "$g_sha"
        printf 'c_start_checkpoint=%s\n' "$c_start"
        printf 'c_start_sha=%s\n' "$c_sha"
        printf 'gpu_id=%s\n' "$gpu_id"
    } >"$run_dir/manifest.txt"

    verify_start() {
        local checkpoint=$1
        local expected=$2
        local actual
        if [[ -z "$checkpoint" || -z "$expected" || ! -f "$checkpoint" ]]; then
            echo "error: missing start checkpoint or expected SHA: $checkpoint" >&2
            exit 1
        fi
        actual=$(sha256sum "$checkpoint" | awk '{print $1}')
        if [[ "$actual" != "$expected" ]]; then
            echo "error: start checkpoint SHA mismatch for $checkpoint" >&2
            echo "expected=$expected actual=$actual" >&2
            exit 1
        fi
    }
    verify_start "$g_start" "$g_sha"
    verify_start "$c_start" "$c_sha"

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

    local role treatment side role_key treatment_key start_checkpoint
    local gamma reward_g reward_c arm checkpoint_dir checkpoint basename
    local timesteps perf sha eval_file opponent_spec opponent_name
    local opponent_id iterations
    for role in $roles; do
        if [[ "$role" == "guerrilla" ]]; then
            side=1
            role_key=g
            start_checkpoint=$g_start
        elif [[ "$role" == "coin" ]]; then
            side=2
            role_key=c
            start_checkpoint=$c_start
        else
            echo "error: unsupported role $role" >&2
            exit 2
        fi
        for treatment in $treatments; do
            case "$treatment" in
                shaped)
                    treatment_key=sh
                    gamma=0.98
                    reward_g=0.05
                    reward_c=0.03
                    ;;
                terminal98)
                    treatment_key=t98
                    gamma=0.98
                    reward_g=0
                    reward_c=0
                    ;;
                terminal100)
                    treatment_key=t100
                    gamma=1.0
                    reward_g=0
                    reward_c=0
                    ;;
                *)
                    echo "error: unsupported treatment $treatment" >&2
                    exit 2
                    ;;
            esac
            arm="gc-${run_key}-${role_key}-${treatment_key}-s${train_seed}"
            if [[ ${#arm} -gt 63 ]]; then
                echo "error: native run id exceeds 63 characters: $arm" >&2
                exit 2
            fi
            checkpoint_dir=".runtime/checkpoints/guerrillacheckers/$arm"
            printf '%s\n' "$arm" >"$run_dir/stage"

            CUDA_VISIBLE_DEVICES="$gpu_id" ./puffer train guerrillacheckers \
                "base.run_id=$arm" \
                "base.seed=$train_seed" \
                "base.load_model_path=$start_checkpoint" \
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
                env.opponent=1 \
                env.mcts_iterations=1 \
                env.mcts_rollout=1 \
                "env.capture_reward_guerrilla=$reward_g" \
                "env.capture_reward_coin=$reward_c" \
                "train.total_timesteps=$total_timesteps" \
                "train.learning_rate=$learning_rate" \
                train.min_lr_ratio=0.1 \
                "train.gamma=$gamma" \
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
                eval_file="$run_dir/$arm-$basename-greedy.eval.log"
                perf=$(evaluate_checkpoint "$side" 1 1 "$checkpoint" "$eval_file")
                sha=$(sha256sum "$checkpoint" | awk '{print $1}')
                printf '%s\t%s\t%d\tgreedy\t%s\t%s\t%s\n' \
                    "$role" "$treatment" "$timesteps" "$perf" \
                    "$checkpoint" "$sha" | tee -a "$run_dir/results.tsv"
            done

            checkpoint=${checkpoints[$((${#checkpoints[@]} - 1))]}
            for opponent_spec in \
                    "random:0:1" "mcts32:2:32" \
                    "mcts128:2:128" "mcts512:2:512"; do
                IFS=: read -r opponent_name opponent_id iterations \
                    <<<"$opponent_spec"
                eval_file="$run_dir/$arm-final-$opponent_name.eval.log"
                perf=$(evaluate_checkpoint "$side" "$opponent_id" "$iterations" \
                    "$checkpoint" "$eval_file")
                sha=$(sha256sum "$checkpoint" | awk '{print $1}')
                printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\n' \
                    "$role" "$treatment" "$total_timesteps" \
                    "$opponent_name" "$perf" "$checkpoint" "$sha" |
                    tee -a "$run_dir/results.tsv"
            done
        done
    done

    printf '%s\n' complete >"$run_dir/stage"
    echo "objective continuation pilot complete"
}

case "$MODE" in
    sync)
        sync_source
        ;;
    start)
        : "${G_START_CHECKPOINT:?set G_START_CHECKPOINT}"
        : "${G_START_SHA:?set G_START_SHA}"
        : "${C_START_CHECKPOINT:?set C_START_CHECKPOINT}"
        : "${C_START_SHA:?set C_START_SHA}"
        sync_source
        ssh_run "cd '$VAST_DIR' && mkdir -p '$REMOTE_RUN_DIR' && \
            if test -f '$REMOTE_RUN_DIR/pid' && \
                kill -0 \$(cat '$REMOTE_RUN_DIR/pid') 2>/dev/null; then \
                echo 'error: objective continuation already running' >&2; exit 1; fi; \
            nohup ./tools/guerrillacheckers/run_objective_continuation_pilot.sh \
                remote-run '$REMOTE_RUN_DIR' '$RUN_ID' '$TOTAL_TIMESTEPS' \
                '$CHECKPOINT_INTERVAL' '$EVAL_GAMES' '$TRAIN_SEED' \
                '$EVAL_SEED' '$LEARNING_RATE' '$ROLES' '$TREATMENTS' \
                '$G_START_CHECKPOINT' '$G_START_SHA' \
                '$C_START_CHECKPOINT' '$C_START_SHA' '$GPU_ID' \
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
        if [[ $# -ne 16 ]]; then
            echo "error: invalid remote-run arguments" >&2
            exit 2
        fi
        remote_run "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" \
            "${10}" "${11}" "${12}" "${13}" "${14}" "${15}" "${16}"
        ;;
    *)
        echo "usage: $0 sync|start|status|stop|download" >&2
        exit 2
        ;;
esac
