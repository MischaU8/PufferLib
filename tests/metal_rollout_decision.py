import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = pathlib.Path(tempfile.gettempdir()) / "puffer-metal-rollout-decision"
DEFAULT_TIMESTEPS = 2_097_152
DEFAULT_FORWARD_ITERS = 100
DEFAULT_GPU_TFLOPS = 10.4
SELF_CONSISTENCY_TOLERANCE = 0.02
METRICS = [
    ("SPS", "SPS"),
    ("rollout_phase_SPS", None),
    ("rollout", "perf/rollout"),
    ("policy_sample_step", None),
    ("eval_policy", "perf/eval_gpu"),
    ("eval_env", "perf/eval_env"),
    ("metal_readback", "perf/metal_readback"),
    ("metal_policy_sample", "perf/metal_policy_sample"),
    ("metal_action_read", "perf/metal_action_read"),
    ("metal_cpu_step", "perf/metal_cpu_step"),
    ("metal_obs_upload", "perf/metal_obs_upload"),
    ("metal_python_overhead", "perf/metal_python_overhead"),
    ("train", "perf/train"),
    ("train_forward", "perf/train_forward"),
    ("train_forward_calls", "perf/train_forward_calls"),
    ("train_forward_mean", "perf/train_forward_mean"),
    ("train_misc", "perf/train_misc"),
    ("train_misc_transfer", "perf/train_misc_transfer"),
    ("train_misc_advantage", "perf/train_misc_advantage"),
    ("train_misc_replay", "perf/train_misc_replay"),
    ("train_misc_loss_opt", "perf/train_misc_loss_opt"),
    ("train_misc_other", "perf/train_misc_other"),
    ("train_other", None),
    ("train_forward_share", None),
    ("torch_train_epoch", "loss/torch_train_epoch"),
    ("torch_train_transfer", "loss/torch_train_transfer"),
    ("torch_train_advantage", "loss/torch_train_advantage"),
    ("torch_train_replay", "loss/torch_train_replay"),
    ("torch_train_forward", "loss/torch_train_forward"),
    ("torch_train_loss_opt", "loss/torch_train_loss_opt"),
    ("torch_train_other", "loss/torch_train_other"),
    ("torch_num_threads", "util/torch_num_threads"),
    ("omp_num_threads", "util/omp_num_threads"),
]

MODES = [
    ("cpu", "CPU baseline"),
    ("mps", "PyTorch MPS"),
    ("metal_python", "per-step Python Metal"),
    ("metal_native", "native-scheduler Metal"),
    ("metal_python_mps_train", "per-step Python Metal + MPS train"),
    ("metal_native_mps_train", "native-scheduler Metal + MPS train"),
]


def _puffer_bin():
    candidate = ROOT / ".venv" / "bin" / "puffer"
    if candidate.exists():
        return str(candidate)

    found = shutil.which("puffer")
    if found:
        return found

    raise RuntimeError("Could not find puffer. Expected .venv/bin/puffer or PATH entry.")


def _run(name, command, env, stream=False):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    if stream:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.stdout


def _parse_key_values(line):
    values = {}
    for part in line.split()[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            values[key] = float(value.rstrip("x"))
        except ValueError:
            values[key] = value
    return values


def _run_forward_microbenchmark(args, base_env):
    cmd = [
        sys.executable,
        str(ROOT / "tests" / "metal_breakout_forward.py"),
        "--batch",
        str(args.agents),
        "--iters",
        str(args.forward_iters),
        "--rollout-horizon",
        str(args.horizon),
        "--sample-step",
        "0",
        "--gpu-tflops",
        str(args.gpu_tflops),
    ]
    output = _run(
        "forward microbenchmark",
        cmd,
        base_env,
        stream=args.stream_forward_benchmark,
    )
    if args.stream_forward_benchmark:
        return None

    benchmark = None
    for line in output.splitlines():
        print(line)
        if line.startswith("benchmark "):
            benchmark = _parse_key_values(line)
    if benchmark is None:
        raise RuntimeError("forward microbenchmark did not print a benchmark line")
    return benchmark


def _latest_log(log_dir):
    env_dir = pathlib.Path(log_dir) / "breakout"
    logs = sorted(env_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise RuntimeError(f"No JSON logs found in {env_dir}")
    return logs[-1]


def _load_summary(path, agents, horizon):
    with open(path) as f:
        data = json.load(f)

    metrics = data["metrics"]
    summary = {"log": str(path), "windows": []}
    train_values = metrics.get("perf/train", [])
    train_indices = [
        idx for idx, value in enumerate(train_values)
        if float(value) > 0.0
    ]
    if not train_indices:
        train_indices = list(range(len(metrics.get("SPS", []))))
    tail_indices = train_indices[1:] if len(train_indices) > 1 else train_indices
    last_idx = train_indices[-1] if train_indices else None

    for idx in train_indices:
        summary["windows"].append(_window(metrics, idx, agents, horizon))

    tail_windows = [_window(metrics, idx, agents, horizon) for idx in tail_indices]
    if not tail_windows:
        fallback_idx = last_idx if last_idx is not None else -1
        tail_windows = [_window(metrics, fallback_idx, agents, horizon)]

    last_window = _window(metrics, last_idx if last_idx is not None else -1, agents, horizon)
    for label, _key in METRICS:
        summary[label] = last_window[label]
        summary[f"{label}_tail_mean"] = sum(
            window[label] for window in tail_windows) / len(tail_windows)

    return summary


def _assert_self_consistent(mode, summary):
    checks = [
        ("last", summary["SPS"], summary["rollout_phase_SPS"]),
        ("tail_mean", summary["SPS_tail_mean"], summary["rollout_phase_SPS_tail_mean"]),
    ]
    for label, end_to_end_sps, rollout_sps in checks:
        if not end_to_end_sps:
            continue
        minimum = end_to_end_sps * (1.0 - SELF_CONSISTENCY_TOLERANCE)
        if rollout_sps < minimum:
            raise RuntimeError(
                f"impossible timing for {mode} {label}: "
                f"rollout_phase_SPS={rollout_sps:.0f} < "
                f"end_to_end_SPS={end_to_end_sps:.0f}. "
                "Rollout is a subset of end-to-end work; check synchronization."
            )


def _metric(metrics, key, idx):
    values = metrics.get(key, [])
    if not values:
        return 0.0
    return float(values[idx])


def _window_steps(metrics, idx, agents, horizon):
    values = metrics.get("agent_steps", [])
    batch = float(agents * horizon)
    if not values:
        return batch
    if idx < 0:
        idx = len(values) + idx
    if idx < 0 or idx >= len(values):
        return batch

    current = float(values[idx])
    previous = float(values[idx - 1]) if idx > 0 else 0.0
    return max(batch, current - previous)


def _window(metrics, idx, agents, horizon):
    window = {}
    for label, key in METRICS:
        if key is None:
            continue
        window[label] = _metric(metrics, key, idx)

    steps = _window_steps(metrics, idx, agents, horizon)
    window["agent_steps"] = steps
    train = window["train"]
    train_forward = window["train_forward"]
    window["train_other"] = max(0.0, train - train_forward)
    window["train_forward_share"] = train_forward / train if train else 0.0
    window["rollout_phase_SPS"] = steps / window["rollout"] if window["rollout"] else 0.0
    policy_sample = window["metal_policy_sample"] or window["eval_policy"]
    window["policy_sample_step"] = policy_sample / horizon if horizon else 0.0
    return window


def _fmt(value, unit):
    if unit == "sps":
        return f"{value:,.0f}"
    if unit == "percent":
        return f"{100.0 * value:7.1f}%"
    if unit == "count":
        return f"{value:8.1f}"
    if unit == "step_seconds":
        return f"{value * 1000.0:8.3f} ms"
    return f"{value * 1000.0:8.2f} ms"


def _unit(label):
    if label in ("SPS", "rollout_phase_SPS"):
        return "sps"
    if label == "train_forward_share":
        return "percent"
    if label in ("train_forward_calls", "torch_num_threads", "omp_num_threads"):
        return "count"
    if label == "policy_sample_step":
        return "step_seconds"
    return "seconds"


def _print_train_split(cpu, metal, suffix):
    rows = ["SPS", "train", "train_forward", "train_other", "train_forward_share"]
    print(f"\n=== train split: {suffix} ===")
    print(f"{'metric':<22} {'cpu':>14} {'metal':>14} {'metal/cpu':>12}")
    for label in rows:
        unit = _unit(label)
        cpu_value = cpu[label]
        metal_value = metal[label]
        ratio = metal_value / cpu_value if cpu_value else 0.0
        ratio_text = f"{ratio:>11.3f}x" if cpu_value else f"{'n/a':>12}"
        print(
            f"{label:<22} "
            f"{_fmt(cpu_value, unit):>14} "
            f"{_fmt(metal_value, unit):>14} "
            f"{ratio_text}"
        )


def _print_per_window(cpu, metal):
    print("\n=== per-window SPS and train split ===")
    print(
        f"{'win':>3} {'cpu_sps':>10} {'metal_sps':>10} {'speedup':>8} "
        f"{'cpu_train':>10} {'metal_train':>11} {'metal_fwd':>10} "
        f"{'metal_other':>11} {'metal_fwd%':>10}"
    )
    for idx, (cpu_window, metal_window) in enumerate(
            zip(cpu["windows"], metal["windows"]), start=1):
        speedup = (
            metal_window["SPS"] / cpu_window["SPS"]
            if cpu_window["SPS"] else 0.0
        )
        print(
            f"{idx:>3} "
            f"{cpu_window['SPS']:>10.0f} "
            f"{metal_window['SPS']:>10.0f} "
            f"{speedup:>7.3f}x "
            f"{cpu_window['train'] * 1000.0:>9.1f}ms "
            f"{metal_window['train'] * 1000.0:>10.1f}ms "
            f"{metal_window['train_forward'] * 1000.0:>9.1f}ms "
            f"{metal_window['train_other'] * 1000.0:>10.1f}ms "
            f"{100.0 * metal_window['train_forward_share']:>9.1f}%"
        )


def _print_table(cpu, metal):
    _print_train_split(
        {label: cpu[f"{label}_tail_mean"] for label, _key in METRICS},
        {label: metal[f"{label}_tail_mean"] for label, _key in METRICS},
        "training tail mean",
    )
    _print_train_split(cpu, metal, "last training log point")
    _print_per_window(cpu, metal)

    print("\n=== training tail mean, first training log point discarded when available ===")
    print(f"{'metric':<18} {'cpu':>14} {'metal':>14} {'metal/cpu':>12}")
    for label, _key in METRICS:
        unit = _unit(label)
        cpu_value = cpu[f"{label}_tail_mean"]
        metal_value = metal[f"{label}_tail_mean"]
        ratio = metal_value / cpu_value if cpu_value else 0.0
        ratio_text = f"{ratio:>11.3f}x" if cpu_value else f"{'n/a':>12}"
        print(
            f"{label:<18} "
            f"{_fmt(cpu_value, unit):>14} "
            f"{_fmt(metal_value, unit):>14} "
            f"{ratio_text}"
        )

    print("\n=== last training log point ===")
    print(f"{'metric':<18} {'cpu':>14} {'metal':>14} {'metal/cpu':>12}")
    for label, _key in METRICS:
        unit = _unit(label)
        cpu_value = cpu[label]
        metal_value = metal[label]
        ratio = metal_value / cpu_value if cpu_value else 0.0
        ratio_text = f"{ratio:>11.3f}x" if cpu_value else f"{'n/a':>12}"
        print(
            f"{label:<18} "
            f"{_fmt(cpu_value, unit):>14} "
            f"{_fmt(metal_value, unit):>14} "
            f"{ratio_text}"
        )

    print("\nlogs:")
    print(f"  cpu:   {cpu['log']}")
    print(f"  metal: {metal['log']}")


def _print_repeat_summary(pairs):
    if len(pairs) <= 1:
        return

    print("\n=== repeat summary: training tail means ===")
    print(
        f"{'run':>3} {'cpu_sps':>10} {'metal_sps':>10} {'speedup':>8} "
        f"{'metal_train':>12} {'metal_fwd':>10} {'metal_other':>11} "
        f"{'metal_fwd%':>10}"
    )
    speedups = []
    for idx, (cpu, metal) in enumerate(pairs, start=1):
        cpu_sps = cpu["SPS_tail_mean"]
        metal_sps = metal["SPS_tail_mean"]
        speedup = metal_sps / cpu_sps if cpu_sps else 0.0
        speedups.append(speedup)
        print(
            f"{idx:>3} "
            f"{cpu_sps:>10.0f} "
            f"{metal_sps:>10.0f} "
            f"{speedup:>7.3f}x "
            f"{metal['train_tail_mean'] * 1000.0:>11.1f}ms "
            f"{metal['train_forward_tail_mean'] * 1000.0:>9.1f}ms "
            f"{metal['train_other_tail_mean'] * 1000.0:>10.1f}ms "
            f"{100.0 * metal['train_forward_share_tail_mean']:>9.1f}%"
        )

    mean = sum(speedups) / len(speedups)
    variance = sum((value - mean) ** 2 for value in speedups) / len(speedups)
    print(f"\nspeedup mean={mean:.3f}x min={min(speedups):.3f}x max={max(speedups):.3f}x std={variance ** 0.5:.3f}x")


def _env_for_mode(base_env, mode):
    env = base_env.copy()
    env.pop("PUFFER_METAL_ROLLOUT", None)
    env.pop("PUFFER_METAL_TRAIN_DEVICE", None)

    if mode == "cpu":
        env["PUFFER_TORCH_DEVICE"] = "cpu"
    elif mode == "mps":
        env["PUFFER_TORCH_DEVICE"] = "mps"
    elif mode == "metal_python":
        env["PUFFER_TORCH_DEVICE"] = "cpu"
        env["PUFFER_METAL_ROLLOUT"] = "python"
    elif mode == "metal_native":
        env["PUFFER_TORCH_DEVICE"] = "cpu"
        env["PUFFER_METAL_ROLLOUT"] = "native"
    elif mode == "metal_python_mps_train":
        env["PUFFER_TORCH_DEVICE"] = "cpu"
        env["PUFFER_METAL_ROLLOUT"] = "python"
        env["PUFFER_METAL_TRAIN_DEVICE"] = "mps"
    elif mode == "metal_native_mps_train":
        env["PUFFER_TORCH_DEVICE"] = "cpu"
        env["PUFFER_METAL_ROLLOUT"] = "native"
        env["PUFFER_METAL_TRAIN_DEVICE"] = "mps"
    else:
        raise ValueError(f"unknown mode: {mode}")
    return env


def _print_four_way(repeat_results, suffix):
    labels = [mode for mode, _name in MODES if mode in repeat_results]
    print(f"\n=== rollout decision: {suffix} ===")
    print(f"{'metric':<22}" + "".join(f" {mode:>18}" for mode in labels))
    for label, _key in METRICS:
        unit = _unit(label)
        row = f"{label:<22}"
        for mode in labels:
            value = repeat_results[mode][f"{label}_tail_mean"]
            row += f" {_fmt(value, unit):>18}"
        print(row)

    if "mps" in repeat_results:
        mps_sps = repeat_results["mps"]["SPS_tail_mean"]
        print("\nSPS speedup vs PyTorch MPS:")
        for mode in labels:
            sps = repeat_results[mode]["SPS_tail_mean"]
            ratio = sps / mps_sps if mps_sps else 0.0
            print(f"  {mode:<18} {ratio:.3f}x")
        mps_rollout_sps = repeat_results["mps"]["rollout_phase_SPS_tail_mean"]
        print("\nrollout-phase speedup vs PyTorch MPS:")
        for mode in labels:
            rollout_sps = repeat_results[mode]["rollout_phase_SPS_tail_mean"]
            ratio = rollout_sps / mps_rollout_sps if mps_rollout_sps else 0.0
            print(f"  {mode:<18} {ratio:.3f}x")

    print("\nlogs:")
    for mode in labels:
        print(f"  {mode:<18} {repeat_results[mode]['log']}")


def _print_final_summary(all_results):
    if not all_results:
        return

    print("\n=== repeat summary: tail-mean SPS ===")
    print(f"{'mode':<22} {'mean':>12} {'min':>12} {'max':>12}")
    for mode, _name in MODES:
        values = [
            repeat[mode]["SPS_tail_mean"]
            for repeat in all_results
            if mode in repeat
        ]
        if not values:
            continue
        print(
            f"{mode:<22} "
            f"{sum(values) / len(values):>12.0f} "
            f"{min(values):>12.0f} "
            f"{max(values):>12.0f}"
        )

    if all("mps" in repeat for repeat in all_results):
        mps_values = [
            repeat["mps"]["SPS_tail_mean"]
            for repeat in all_results
        ]
        mps_mean = sum(mps_values) / len(mps_values)
        if mps_mean:
            print("\nSPS speedup vs PyTorch MPS mean:")
            for mode, _name in MODES:
                values = [
                    repeat[mode]["SPS_tail_mean"]
                    for repeat in all_results
                    if mode in repeat
                ]
                if not values:
                    continue
                print(f"  {mode:<26} {(sum(values) / len(values)) / mps_mean:.3f}x")

    if all("mps" in repeat and "metal_native" in repeat for repeat in all_results):
        speedups = [
            repeat["metal_native"]["rollout_phase_SPS_tail_mean"] /
            repeat["mps"]["rollout_phase_SPS_tail_mean"]
            for repeat in all_results
            if repeat["mps"]["rollout_phase_SPS_tail_mean"]
        ]
        if speedups:
            mean = sum(speedups) / len(speedups)
            print(
                "\nnative Metal vs PyTorch MPS rollout-phase speedup: "
                f"mean={mean:.3f}x min={min(speedups):.3f}x max={max(speedups):.3f}x"
            )


def _print_forward_summary(benchmark):
    if not benchmark:
        return

    print("\n=== forward microbenchmark: batch-1024-equivalent gate signal ===")
    print(f"forward FLOPs/step: {benchmark['forward_flops']:.0f}")
    print(
        "hardware ceiling: "
        f"{benchmark['ceiling_us']:.3f} us at {benchmark['gpu_tflops']:.3f} TFLOP/s"
    )
    print(
        "native Metal forward: "
        f"warm/sync={benchmark['metal_sync_ms'] * 1000.0:.1f} us "
        f"queued={benchmark['metal_queued_ms'] * 1000.0:.1f} us "
        f"queued/ceiling={benchmark['metal_queued_vs_ceiling']:.1f}x"
    )
    print(
        "native Metal forward+sample: "
        f"{benchmark['metal_forward_sample_ms'] * 1000.0:.1f} us "
        f"vs ceiling={benchmark['metal_forward_sample_vs_ceiling']:.1f}x"
    )
    if "mps_sync_ms" in benchmark:
        print(
            "PyTorch MPS forward: "
            f"warm/sync={benchmark['mps_sync_ms'] * 1000.0:.1f} us "
            f"queued={benchmark['mps_queued_ms'] * 1000.0:.1f} us"
        )
        print(
            "native/MPS forward speed: "
            f"sync={benchmark['metal_sync_vs_mps_sync']:.3f}x "
            f"queued={benchmark['metal_queued_vs_mps_queued']:.3f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=32768)
    parser.add_argument("--replay-ratio", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000_000_000)
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--forward-iters", type=int, default=DEFAULT_FORWARD_ITERS)
    parser.add_argument("--gpu-tflops", type=float, default=DEFAULT_GPU_TFLOPS)
    parser.add_argument("--skip-forward-benchmark", action="store_true")
    parser.add_argument("--stream-forward-benchmark", action="store_true")
    parser.add_argument("--stream", action="store_true",
        help="Stream puffer dashboard output instead of keeping subprocess output quiet.")
    parser.add_argument("--mode", action="append", choices=[mode for mode, _name in MODES],
        help="Run only this mode. Repeat for multiple modes. Defaults to all modes.")
    parser.add_argument("--extra-arg", action="append", default=[],
        help="Extra argument passed to both puffer train invocations. Repeat for multiple args.")
    parser.add_argument("--keep-log-dir", action="store_true")
    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    if log_dir.exists() and not args.keep_log_dir:
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["PUFFER_METAL_DIAGNOSTICS"] = "1"
    if args.torch_num_threads is not None:
        base_env["PUFFER_TORCH_NUM_THREADS"] = str(args.torch_num_threads)
        base_env.setdefault("OMP_NUM_THREADS", str(args.torch_num_threads))
        base_env.setdefault("MKL_NUM_THREADS", str(args.torch_num_threads))

    forward_benchmark = None
    if not args.skip_forward_benchmark:
        forward_benchmark = _run_forward_microbenchmark(args, base_env)
        _print_forward_summary(forward_benchmark)

    all_results = []
    modes = [entry for entry in MODES if args.mode is None or entry[0] in args.mode]
    for repeat in range(args.repeats):
        repeat_root = log_dir / f"run-{repeat + 1}" if args.repeats > 1 else log_dir
        repeat_root.mkdir(parents=True, exist_ok=True)
        order = modes if repeat % 2 == 0 else list(reversed(modes))
        repeat_results = {}

        for mode, name in order:
            run_log_dir = repeat_root / mode
            run_log_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir = (
                pathlib.Path(args.checkpoint_dir) / f"run-{repeat + 1}" / mode
                if args.checkpoint_dir is not None and args.repeats > 1 else
                pathlib.Path(args.checkpoint_dir) / mode
                if args.checkpoint_dir is not None else
                run_log_dir / "checkpoints"
            )

            base_cmd = [
                _puffer_bin(),
                "train",
                "breakout",
                "--vec.total-agents",
                str(args.agents),
                "--vec.num-threads",
                str(args.threads),
                "--train.horizon",
                str(args.horizon),
                "--train.minibatch-size",
                str(args.minibatch_size),
                "--train.total-timesteps",
                str(args.timesteps),
                "--train.replay-ratio",
                str(args.replay_ratio),
                "--checkpoint-interval",
                str(args.checkpoint_interval),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--log-dir",
                str(run_log_dir),
            ] + args.extra_arg

            label = f"run {repeat + 1}/{args.repeats} {name}"
            _run(label, base_cmd, _env_for_mode(base_env, mode), stream=args.stream)
            repeat_results[mode] = _load_summary(_latest_log(run_log_dir), args.agents, args.horizon)
            _assert_self_consistent(mode, repeat_results[mode])

        _print_four_way(repeat_results, f"run {repeat + 1}/{args.repeats} tail means")
        all_results.append(repeat_results)

    _print_final_summary(all_results)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
