#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODE="${1:-status}"
if [[ "$MODE" != "remote-run" ]]; then
    : "${VAST_HOST:?set VAST_HOST, for example root@vast-host}"
    : "${VAST_DIR:?set VAST_DIR to an isolated remote checkout}"
    : "${GPU_ID:?set GPU_ID after inspecting the remote GPU inventory}"
fi

RUN_ID="${RUN_ID:-gc-role-curriculum-pilot-v1}"
STAGE_TIMESTEPS="${STAGE_TIMESTEPS:-1048576}"
MAX_STAGE_ROUNDS="${MAX_STAGE_ROUNDS:-3}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
GATE_GAMES="${GATE_GAMES:-256}"
SCREEN_GAMES="${SCREEN_GAMES:-20}"
LOCAL_ARTIFACT_DIR="${LOCAL_ARTIFACT_DIR:-.runtime/role-curriculum-pilot/$RUN_ID}"
REMOTE_RUN_DIR=".runtime/role-curriculum-pilot/$RUN_ID"
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
        tools/guerrillacheckers/run_role_curriculum_pilot.sh
    )
    head=$(git rev-parse HEAD)
    ssh_run "mkdir -p '$VAST_DIR'"
    remote_head=$(ssh_run "cat '$VAST_DIR/$REMOTE_SOURCE_MARKER' 2>/dev/null || \
        find '$VAST_DIR/.runtime/role-objective-pilot' -name source-head \
        -type f -print -quit 2>/dev/null | xargs -r cat" || true)
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
        test ! -f '$REMOTE_RUN_DIR/results.tsv' || tail -n 12 '$REMOTE_RUN_DIR/results.tsv'; \
        test ! -f '$REMOTE_RUN_DIR/run.log' || tail -n 35 '$REMOTE_RUN_DIR/run.log'"
}

remote_run() {
    local run_dir=$1
    local run_id=$2
    local stage_timesteps=$3
    local max_stage_rounds=$4
    local learning_rate=$5
    local gate_games=$6
    local screen_games=$7
    local gpu_id=$8

    export PATH="/usr/local/cuda-13.0/bin:$PATH"
    mkdir -p "$run_dir" .runtime/checkpoints .runtime/logs
    local run_key
    run_key=$(printf '%s' "$run_id" | sha256sum | awk '{print substr($1, 1, 8)}')
    if [[ ! -f "$run_dir/results.tsv" ]]; then
        printf 'role\tobjective\tstage\tround\topponent\titerations\tthreshold\tperf\tcheckpoint\tsha256\tdecision\n' \
            >"$run_dir/results.tsv"
    fi

    evaluate_gate() {
        local role=$1
        local opponent=$2
        local iterations=$3
        local checkpoint=$4
        local output=$5
        local side opponent_args
        if [[ "$role" == "guerrilla" ]]; then side=1; else side=2; fi
        if [[ "$opponent" == "random" ]]; then
            opponent_args="env.opponent=0 env.mcts_iterations=1"
        elif [[ "$opponent" == "greedy" ]]; then
            opponent_args="env.opponent=1 env.mcts_iterations=1"
        else
            opponent_args="env.opponent=2 env.mcts_iterations=$iterations"
        fi

        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES="$gpu_id" ./puffer eval_bot guerrillacheckers \
            "base.load_model_path=$checkpoint" \
            "base.num_games=$gate_games" \
            "base.eval_agents=$gate_games" \
            base.burnin_games=0 \
            base.gpu_offset=0 \
            selfplay.enabled=0 \
            vec.num_frozen_banks=0 \
            vec.frozen_bank_pct=0 \
            env.selfplay=0 \
            "env.side=$side" \
            $opponent_args \
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

    train_stage_round() {
        local role=$1
        local objective=$2
        local seed=$3
        local stage=$4
        local opponent=$5
        local iterations=$6
        local round=$7
        local input_checkpoint=$8
        local side gamma reward_g reward_c opponent_id
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
        if [[ "$opponent" == "random" ]]; then opponent_id=0
        elif [[ "$opponent" == "greedy" ]]; then opponent_id=1
        else opponent_id=2
        fi

        local role_key=${role:0:1}
        local objective_key=${objective:0:1}
        local stage_run="gc-${run_key}-${role_key}${objective_key}-${stage}-r${round}-s${seed}"
        if [[ ${#stage_run} -gt 63 ]]; then
            echo "error: native run id exceeds 63 characters: $stage_run" >&2
            return 1
        fi
        local checkpoint_dir=".runtime/checkpoints/guerrillacheckers/$stage_run"
        local pointer="$run_dir/$stage_run.checkpoint"
        if [[ -f "$pointer" ]] && [[ -f "$(cat "$pointer")" ]]; then
            cat "$pointer"
            return
        fi

        local load_args=()
        if [[ -n "$input_checkpoint" ]]; then
            load_args+=("base.load_model_path=$input_checkpoint")
        fi
        printf '%s\n' "$stage_run" >"$run_dir/stage"
        echo "starting stage_run=$stage_run opponent=$opponent iterations=$iterations input=${input_checkpoint:-scratch}"

        CUDA_VISIBLE_DEVICES="$gpu_id" ./puffer train guerrillacheckers \
            "base.run_id=$stage_run" \
            "base.seed=$seed" \
            base.gpu_offset=0 \
            base.checkpoint_dir=.runtime/checkpoints \
            base.log_dir=.runtime/logs \
            base.checkpoint_interval=2 \
            base.eval_episodes=0 \
            "${load_args[@]}" \
            selfplay.enabled=0 \
            vec.num_frozen_banks=0 \
            vec.frozen_bank_pct=0 \
            vec.num_threads=32 \
            vec.total_agents=4096 \
            env.selfplay=0 \
            "env.side=$side" \
            "env.opponent=$opponent_id" \
            "env.mcts_iterations=$iterations" \
            env.mcts_rollout=1 \
            "env.capture_reward_guerrilla=$reward_g" \
            "env.capture_reward_coin=$reward_c" \
            "train.total_timesteps=$stage_timesteps" \
            "train.learning_rate=$learning_rate" \
            train.min_lr_ratio=0.1 \
            "train.gamma=$gamma" \
            2>&1 | tee "$run_dir/$stage_run.train.log"

        local checkpoint
        checkpoint=$(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.bin' |
            sort | tail -n 1)
        if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
            echo "error: no checkpoint produced for $stage_run" >&2
            return 1
        fi
        printf '%s\n' "$checkpoint" >"$pointer"
        sha256sum "$checkpoint" >"$run_dir/$stage_run.checkpoint.sha256"
        printf '%s\n' "$checkpoint"
    }

    external_screen() {
        local role=$1
        local arm=$2
        local checkpoint=$3
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

    run_arm() {
        local role=$1
        local objective=$2
        local seed=$3
        local arm="${run_id}-${role}-${objective}-s${seed}"
        local arm_complete="$run_dir/$arm.complete"
        if [[ -f "$arm_complete" ]]; then
            echo "skipping completed arm=$arm"
            return
        fi

        # stage:opponent:iterations:promotion threshold. COIN starts with a
        # structural advantage in the preserved field, so its gates are higher.
        local -a stages
        if [[ "$role" == "guerrilla" ]]; then
            stages=(
                "random:random:1:0.55"
                "greedy:greedy:1:0.35"
                "mcts32:mcts:32:0.20"
                "mcts128:mcts:128:0.10"
                "mcts512:mcts:512:0.05"
            )
        else
            stages=(
                "random:random:1:0.70"
                "greedy:greedy:1:0.60"
                "mcts32:mcts:32:0.45"
                "mcts128:mcts:128:0.30"
                "mcts512:mcts:512:0.20"
            )
        fi
        local input_checkpoint=""
        local failed=0
        for spec in "${stages[@]}"; do
            IFS=: read -r stage opponent iterations threshold <<<"$spec"
            local promoted=0
            for round in $(seq 1 "$max_stage_rounds"); do
                local checkpoint eval_file perf sha decision
                checkpoint=$(train_stage_round "$role" "$objective" "$seed" \
                    "$stage" "$opponent" "$iterations" "$round" "$input_checkpoint" |
                    tail -n 1)
                eval_file="$run_dir/$arm-$stage-r$round.gate.log"
                perf=$(evaluate_gate "$role" "$opponent" "$iterations" \
                    "$checkpoint" "$eval_file")
                sha=$(sha256sum "$checkpoint" | awk '{print $1}')
                decision=retry
                if awk -v score="$perf" -v gate="$threshold" \
                        'BEGIN { exit !(score >= gate) }'; then
                    decision=promote
                    promoted=1
                elif [[ "$round" -eq "$max_stage_rounds" ]]; then
                    decision=stop
                fi
                printf '%s\t%s\t%s\t%d\t%s\t%d\t%s\t%s\t%s\t%s\t%s\n' \
                    "$role" "$objective" "$stage" "$round" "$opponent" \
                    "$iterations" "$threshold" "$perf" "$checkpoint" "$sha" \
                    "$decision" | tee -a "$run_dir/results.tsv"
                input_checkpoint=$checkpoint
                [[ "$promoted" -eq 1 ]] && break
            done
            if [[ "$promoted" -ne 1 ]]; then
                failed=1
                break
            fi
        done

        if [[ -n "$input_checkpoint" ]]; then
            external_screen "$role" "$arm" "$input_checkpoint"
            printf '%s\n' "$input_checkpoint" >"$run_dir/$arm.final-checkpoint"
        fi
        if [[ "$failed" -eq 1 ]]; then
            printf '%s\n' stopped-at-gate >"$arm_complete"
        else
            printf '%s\n' completed-curriculum >"$arm_complete"
        fi
    }

    ./build.sh guerrillacheckers --fast
    ./build.sh guerrillacheckers --float

    # Identical initialization within each role isolates the objective bundle.
    run_arm guerrilla shaped 101
    run_arm guerrilla terminal 101
    run_arm coin shaped 101
    run_arm coin terminal 101

    printf '%s\n' complete >"$run_dir/stage"
    echo "role curriculum pilot complete"
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
                echo 'error: curriculum already running' >&2; exit 1; fi; \
            nohup ./tools/guerrillacheckers/run_role_curriculum_pilot.sh remote-run \
                '$REMOTE_RUN_DIR' '$RUN_ID' '$STAGE_TIMESTEPS' \
                '$MAX_STAGE_ROUNDS' '$LEARNING_RATE' '$GATE_GAMES' \
                '$SCREEN_GAMES' '$GPU_ID' \
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
        if [[ $# -ne 9 ]]; then
            echo "error: invalid remote-run arguments" >&2
            exit 2
        fi
        remote_run "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9"
        ;;
    *)
        echo "usage: $0 sync|start|status|download" >&2
        exit 2
        ;;
esac
