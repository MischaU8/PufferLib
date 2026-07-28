#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODE="${1:-status}"
if [[ "$MODE" != "remote-run" ]]; then
    : "${VAST_HOST:?set VAST_HOST, for example root@vast-host}"
    : "${VAST_DIR:?set VAST_DIR to an isolated remote checkout}"
    : "${GPU_ID:?set GPU_ID after inspecting the remote GPU inventory}"
fi

RUN_ID="${RUN_ID:-gc-role-objective-pilot-v1}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-4194304}"
MCTS_ITERATIONS="${MCTS_ITERATIONS:-512}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
SCREEN_GAMES="${SCREEN_GAMES:-20}"
LOCAL_ARTIFACT_DIR="${LOCAL_ARTIFACT_DIR:-.runtime/role-objective-pilot/$RUN_ID}"
REMOTE_RUN_DIR=".runtime/role-objective-pilot/$RUN_ID"

ssh_run() {
    ssh "$VAST_HOST" "$@"
}

sync_source() {
    local archive head remote_head
    head=$(git rev-parse HEAD)
    ssh_run "mkdir -p '$VAST_DIR'"
    remote_head=$(ssh_run "cat '$VAST_DIR/$REMOTE_RUN_DIR/source-head' 2>/dev/null" || true)
    if [[ "$remote_head" != "$head" ]]; then
        archive=$(mktemp /tmp/gc-performance-lab-source.XXXXXX)
        git archive --format=tar --output="$archive" HEAD
        scp "$archive" "$VAST_HOST:$VAST_DIR/source.tar"
        ssh_run "cd '$VAST_DIR' && tar -xf source.tar && rm source.tar"
        rm -f "$archive"
    fi
    rsync -azR \
        build.sh \
        config/guerrillacheckers.ini \
        ocean/guerrillacheckers/guerrillacheckers.c \
        ocean/guerrillacheckers/guerrillacheckers.h \
        tools/guerrillacheckers/run_role_objective_pilot.sh \
        "$VAST_HOST:$VAST_DIR/"

    local diff_sha
    diff_sha=$(git diff --binary | shasum -a 256 | awk '{print $1}')
    ssh_run "mkdir -p '$VAST_DIR/$REMOTE_RUN_DIR' && \
        printf '%s\\n' '$head' > '$VAST_DIR/$REMOTE_RUN_DIR/source-head' && \
        printf '%s\\n' '$diff_sha' > '$VAST_DIR/$REMOTE_RUN_DIR/source-diff-sha256'"
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
        test ! -f '$REMOTE_RUN_DIR/run.log' || tail -n 50 '$REMOTE_RUN_DIR/run.log'"
}

remote_run() {
    local run_dir=$1
    local run_id=$2
    local total_timesteps=$3
    local mcts_iterations=$4
    local learning_rate=$5
    local screen_games=$6
    local gpu_id=$7

    export PATH="/usr/local/cuda-13.0/bin:$PATH"
    mkdir -p "$run_dir" .runtime/checkpoints .runtime/logs

    run_arm() {
        local role=$1
        local objective=$2
        local seed=$3
        local side gamma reward_g reward_c
        if [[ "$role" == "guerrilla" ]]; then side=1; else side=2; fi
        if [[ "$objective" == "terminal" ]]; then
            gamma=1.0
            reward_g=0
            reward_c=0
        else
            gamma=0.98
            reward_g=0.05
            reward_c=0.03
        fi

        local arm="${run_id}-${role}-${objective}-s${seed}"
        local checkpoint_dir=".runtime/checkpoints/guerrillacheckers/$arm"
        printf '%s\n' "$arm" >"$run_dir/stage"
        echo "starting arm=$arm side=$side gamma=$gamma reward_g=$reward_g reward_c=$reward_c"

        CUDA_VISIBLE_DEVICES="$gpu_id" ./puffer train guerrillacheckers \
            "base.run_id=$arm" \
            "base.seed=$seed" \
            base.gpu_offset=0 \
            base.checkpoint_dir=.runtime/checkpoints \
            base.log_dir=.runtime/logs \
            base.checkpoint_interval=4 \
            base.eval_episodes=0 \
            selfplay.enabled=0 \
            vec.num_frozen_banks=0 \
            vec.frozen_bank_pct=0 \
            vec.num_threads=32 \
            vec.total_agents=4096 \
            env.selfplay=0 \
            "env.side=$side" \
            env.opponent=2 \
            "env.mcts_iterations=$mcts_iterations" \
            env.mcts_rollout=1 \
            "env.capture_reward_guerrilla=$reward_g" \
            "env.capture_reward_coin=$reward_c" \
            "train.total_timesteps=$total_timesteps" \
            "train.learning_rate=$learning_rate" \
            train.min_lr_ratio=0.1 \
            "train.gamma=$gamma" \
            2>&1 | tee "$run_dir/$arm.train.log"

        local checkpoint
        checkpoint=$(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.bin' |
            sort | tail -n 1)
        if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
            echo "error: no checkpoint produced for $arm" >&2
            return 1
        fi
        sha256sum "$checkpoint" >"$run_dir/$arm.checkpoint.sha256"
        printf '%s\n' "$checkpoint" >"$run_dir/$arm.checkpoint"

        if [[ "$role" == "guerrilla" ]]; then
            ./guerrillacheckers --candidate-pair 6 3 "$screen_games" "$checkpoint" |
                tee "$run_dir/$arm.vs-puffer.tsv"
            ./guerrillacheckers --candidate-pair 6 4 "$screen_games" "$checkpoint" |
                tee "$run_dir/$arm.vs-mcts2k.tsv"
            ./guerrillacheckers --candidate-pair 6 5 "$screen_games" "$checkpoint" |
                tee "$run_dir/$arm.vs-mcts10k.tsv"
        else
            ./guerrillacheckers --candidate-pair 3 6 "$screen_games" "$checkpoint" |
                tee "$run_dir/$arm.vs-puffer.tsv"
            ./guerrillacheckers --candidate-pair 4 6 "$screen_games" "$checkpoint" |
                tee "$run_dir/$arm.vs-mcts2k.tsv"
            ./guerrillacheckers --candidate-pair 5 6 "$screen_games" "$checkpoint" |
                tee "$run_dir/$arm.vs-mcts10k.tsv"
        fi
    }

    ./build.sh guerrillacheckers --fast
    ./build.sh guerrillacheckers --float

    # Same initialization seed within a role isolates the learning objective.
    # The role comparison itself is intentionally separate because the games
    # and action semantics are asymmetric.
    run_arm guerrilla shaped 101
    run_arm guerrilla terminal 101
    run_arm coin shaped 101
    run_arm coin terminal 101

    printf '%s\n' complete >"$run_dir/stage"
    echo "role/objective pilot complete"
}

case "$MODE" in
    sync)
        sync_source
        ;;
    smoke)
        sync_source
        ssh_run "cd '$VAST_DIR' && \
            export PATH=/usr/local/cuda-13.0/bin:\$PATH && \
            ./build.sh guerrillacheckers --fast && \
            ./build.sh guerrillacheckers --float && \
            ./guerrillacheckers --tournament 1"
        ;;
    start)
        sync_source
        ssh_run "cd '$VAST_DIR' && mkdir -p '$REMOTE_RUN_DIR' && \
            if test -f '$REMOTE_RUN_DIR/pid' && \
                kill -0 \$(cat '$REMOTE_RUN_DIR/pid') 2>/dev/null; then \
                echo 'error: pilot already running' >&2; exit 1; fi; \
            nohup ./tools/guerrillacheckers/run_role_objective_pilot.sh remote-run \
                '$REMOTE_RUN_DIR' '$RUN_ID' '$TOTAL_TIMESTEPS' \
                '$MCTS_ITERATIONS' '$LEARNING_RATE' '$SCREEN_GAMES' '$GPU_ID' \
                >'$REMOTE_RUN_DIR/run.log' 2>&1 </dev/null & \
            echo \$! >'$REMOTE_RUN_DIR/pid'; \
            echo started pid=\$(cat '$REMOTE_RUN_DIR/pid')"
        remote_status
        ;;
    status)
        remote_status
        ;;
    download)
        mkdir -p "$LOCAL_ARTIFACT_DIR"
        rsync -az "$VAST_HOST:$VAST_DIR/$REMOTE_RUN_DIR/" "$LOCAL_ARTIFACT_DIR/"
        echo "downloaded to $LOCAL_ARTIFACT_DIR"
        ;;
    remote-run)
        if [[ $# -ne 8 ]]; then
            echo "error: invalid remote-run arguments" >&2
            exit 2
        fi
        remote_run "$2" "$3" "$4" "$5" "$6" "$7" "$8"
        ;;
    *)
        echo "usage: $0 sync|smoke|start|status|download" >&2
        exit 2
        ;;
esac
