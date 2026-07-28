#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

MODE="${1:-status}"
if [[ "$MODE" != "remote-run" ]]; then
    : "${VAST_HOST:?set VAST_HOST, for example root@vast-host}"
    : "${VAST_DIR:?set VAST_DIR to the isolated remote source snapshot}"
fi

RUN_ID="${RUN_ID:-gc-deployment-field-20260728-v1}"
GAMES="${GAMES:-100}"
PARALLELISM="${PARALLELISM:-6}"
LOCAL_FINALIST_DIR="${LOCAL_FINALIST_DIR:-.runtime/finalists/gc-performance-20260728}"
LOCAL_ARTIFACT_DIR="${LOCAL_ARTIFACT_DIR:-.runtime/deployment-field/$RUN_ID}"
REMOTE_RUN_DIR=".runtime/deployment-field/$RUN_ID"

ssh_run() {
    ssh "$VAST_HOST" "$@"
}

sync_inputs() {
    local explicit_files=(
        build.sh
        ocean/guerrillacheckers/guerrillacheckers.c
        ocean/guerrillacheckers/guerrillacheckers.h
        tools/guerrillacheckers/run_deployment_field_tournament.sh
    )
    local finalist_files=(
        coin-terminal100-s202.bin
        coin-terminal100-s203.bin
        coin-terminal100-s204.bin
        guerrilla-shaped-s202.bin
        guerrilla-shaped-s203.bin
        guerrilla-shaped-s204.bin
    )
    local file

    for file in "${finalist_files[@]}"; do
        if [[ ! -f "$LOCAL_FINALIST_DIR/$file" ]]; then
            echo "error: missing finalist $LOCAL_FINALIST_DIR/$file" >&2
            return 1
        fi
    done

    ssh_run "mkdir -p '$VAST_DIR/$REMOTE_RUN_DIR/finalists'"
    rsync -azR "${explicit_files[@]}" "$VAST_HOST:$VAST_DIR/"
    for file in "${finalist_files[@]}"; do
        rsync -az "$LOCAL_FINALIST_DIR/$file" \
            "$VAST_HOST:$VAST_DIR/$REMOTE_RUN_DIR/finalists/$file"
    done
}

remote_status() {
    ssh_run "cd '$VAST_DIR' && \
        if test -f '$REMOTE_RUN_DIR/pid'; then \
            pid=\$(cat '$REMOTE_RUN_DIR/pid'); \
            if kill -0 \"\$pid\" 2>/dev/null; then echo status=running pid=\"\$pid\"; \
            else echo status=stopped pid=\"\$pid\"; fi; \
        else echo status=not-started; fi; \
        test ! -f '$REMOTE_RUN_DIR/stage' || sed 's/^/stage=/' '$REMOTE_RUN_DIR/stage'; \
        echo active-pairings; \
        ps -eo pid,ppid,etime,%cpu,%mem,args | \
            grep '[g]uerrillacheckers --candidate-pair' || true; \
        echo results; \
        test ! -f '$REMOTE_RUN_DIR/results.tsv' || cat '$REMOTE_RUN_DIR/results.tsv'; \
        echo log-tail; \
        test ! -f '$REMOTE_RUN_DIR/run.log' || tail -n 30 '$REMOTE_RUN_DIR/run.log'"
}

remote_run() {
    local run_dir=$1
    local games=$2
    local parallelism=$3

    if [[ "$parallelism" -lt 1 || "$parallelism" -gt 6 ]]; then
        echo "error: PARALLELISM must be between 1 and 6" >&2
        return 2
    fi

    mkdir -p "$run_dir/logs" "$run_dir/outputs"
    printf 'build\n' >"$run_dir/stage"
    ./build.sh guerrillacheckers --fast

    local -a finalists=(
        "guerrilla:202:guerrilla-shaped-s202.bin:bcc3a67d6d509125bfb446d941a60699dc8018d4c3e220d21eaf4868e754be35"
        "guerrilla:203:guerrilla-shaped-s203.bin:e764bac2218ee0bf811c7faf3be87495053f35ca3963d5951ac567f0a9a9fb6f"
        "guerrilla:204:guerrilla-shaped-s204.bin:a39e2cba4b3fdf0524fb49e6b04dd12bb2b85aabff752188562aa1ee2c311a47"
        "coin:202:coin-terminal100-s202.bin:6c705d59659fbf64f4a977f1f4ab5f29247ce24407f9e15b6a92e6f49b01c162"
        "coin:203:coin-terminal100-s203.bin:45257dfce43b5994ca9cd02a5de3dabdc3d511f940a37d640a630f4110a1d71d"
        "coin:204:coin-terminal100-s204.bin:dfe63c20bfe06f5d14976141891bc672dcefa438a64f20557018bf72de680b06"
    )
    local -a opponents=(
        "puffer:3"
        "mcts2k:4"
        "mcts10k:5"
    )

    local role seed file expected_sha actual_sha
    : >"$run_dir/finalists.sha256"
    for spec in "${finalists[@]}"; do
        IFS=: read -r role seed file expected_sha <<<"$spec"
        actual_sha=$(sha256sum "$run_dir/finalists/$file" | awk '{print $1}')
        if [[ "$actual_sha" != "$expected_sha" ]]; then
            echo "error: checksum mismatch for $file" >&2
            return 1
        fi
        printf '%s  %s\n' "$actual_sha" "$file" >>"$run_dir/finalists.sha256"
    done

    rebuild_results() {
        local tmp="$run_dir/results.tsv.tmp"
        local opponent level stem output meta g_wins c_wins candidate_wins
        local opponent_wins elapsed rate checkpoint sha
        printf 'role\tseed\topponent\tgames\tcandidate_wins\topponent_wins\twin_rate\telapsed_s\tcheckpoint\tsha256\n' \
            >"$tmp"
        for opponent_spec in "${opponents[@]}"; do
            IFS=: read -r opponent level <<<"$opponent_spec"
            for spec in "${finalists[@]}"; do
                IFS=: read -r role seed file sha <<<"$spec"
                stem="${role}-s${seed}-vs-${opponent}"
                output="$run_dir/outputs/$stem.tsv"
                meta="$run_dir/outputs/$stem.elapsed"
                [[ -f "$output.done" ]] || continue
                IFS=$'\t' read -r _ _ g_wins c_wins _ _ <"$output"
                if [[ "$role" == "guerrilla" ]]; then
                    candidate_wins=$g_wins
                    opponent_wins=$c_wins
                else
                    candidate_wins=$c_wins
                    opponent_wins=$g_wins
                fi
                elapsed=$(cat "$meta")
                rate=$(awk -v wins="$candidate_wins" -v total="$games" \
                    'BEGIN { printf "%.6f", wins / total }')
                checkpoint="finalists/$file"
                printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$role" "$seed" "$opponent" "$games" "$candidate_wins" \
                    "$opponent_wins" "$rate" "$elapsed" "$checkpoint" "$sha" \
                    >>"$tmp"
            done
        done
        mv "$tmp" "$run_dir/results.tsv"
    }

    run_match() {
        local role=$1
        local seed=$2
        local file=$3
        local opponent=$4
        local level=$5
        local stem="${role}-s${seed}-vs-${opponent}"
        local output="$run_dir/outputs/$stem.tsv"
        local log="$run_dir/logs/$stem.log"
        local checkpoint="$run_dir/finalists/$file"
        local start_s end_s g_level c_level total

        if [[ -f "$output.done" ]]; then
            echo "skip completed $stem"
            return
        fi
        if [[ "$role" == "guerrilla" ]]; then
            g_level=6
            c_level=$level
        else
            g_level=$level
            c_level=6
        fi

        start_s=$(date +%s)
        ./guerrillacheckers --candidate-pair "$g_level" "$c_level" \
            "$games" "$checkpoint" >"$output.tmp" 2>"$log"
        end_s=$(date +%s)
        total=$(awk -F '\t' 'NF >= 4 { print $3 + $4 }' "$output.tmp")
        if [[ "$total" != "$games" ]]; then
            echo "error: invalid result for $stem: total=$total expected=$games" >&2
            return 1
        fi
        mv "$output.tmp" "$output"
        printf '%s\n' "$((end_s - start_s))" >"$run_dir/outputs/$stem.elapsed"
        printf 'complete\n' >"$output.done"
        echo "completed $stem elapsed=$((end_s - start_s))s"
    }

    local opponent level spec active failed pid
    local -a pids=()
    for opponent_spec in "${opponents[@]}"; do
        IFS=: read -r opponent level <<<"$opponent_spec"
        printf '%s\n' "$opponent" >"$run_dir/stage"
        pids=()
        active=0
        failed=0
        for spec in "${finalists[@]}"; do
            IFS=: read -r role seed file expected_sha <<<"$spec"
            run_match "$role" "$seed" "$file" "$opponent" "$level" &
            pids+=("$!")
            active=$((active + 1))
            if [[ "$active" -ge "$parallelism" ]]; then
                for pid in "${pids[@]}"; do
                    if ! wait "$pid"; then failed=1; fi
                done
                pids=()
                active=0
            fi
        done
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then failed=1; fi
        done
        rebuild_results
        if [[ "$failed" -ne 0 ]]; then
            echo "error: one or more $opponent pairings failed" >&2
            return 1
        fi
    done

    printf 'complete\n' >"$run_dir/stage"
    rebuild_results
    echo "deployment-field tournament complete"
}

case "$MODE" in
    sync)
        sync_inputs
        ;;
    start)
        sync_inputs
        ssh_run "cd '$VAST_DIR' && \
            mkdir -p '$REMOTE_RUN_DIR' && \
            if test -f '$REMOTE_RUN_DIR/pid' && \
                kill -0 \$(cat '$REMOTE_RUN_DIR/pid') 2>/dev/null; then \
                echo 'error: tournament already running' >&2; exit 1; fi; \
            nohup ./tools/guerrillacheckers/run_deployment_field_tournament.sh \
                remote-run '$REMOTE_RUN_DIR' '$GAMES' '$PARALLELISM' \
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
        if [[ $# -ne 4 ]]; then
            echo "error: invalid remote-run arguments" >&2
            exit 2
        fi
        remote_run "$2" "$3" "$4"
        ;;
    *)
        echo "usage: $0 sync|start|status|download" >&2
        exit 2
        ;;
esac
