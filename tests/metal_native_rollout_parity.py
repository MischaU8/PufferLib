import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def _puffer_bin():
    candidate = ROOT / ".venv" / "bin" / "puffer"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("puffer")
    if found:
        return found
    raise RuntimeError("Could not find puffer. Expected .venv/bin/puffer or PATH entry.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=32768)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=2048)
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir) if args.log_dir else (
        pathlib.Path(tempfile.gettempdir()) / "puffer-metal-native-rollout-parity")
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PUFFER_TORCH_DEVICE"] = "cpu"
    env["PUFFER_METAL_ROLLOUT"] = "native"
    env["PUFFER_METAL_DIAGNOSTICS"] = "1"

    cmd = [
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
        "1.0",
        "--checkpoint-interval",
        "1000000000",
        "--checkpoint-dir",
        str(log_dir / "checkpoints"),
        "--log-dir",
        str(log_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    print("native Metal rollout parity ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)
