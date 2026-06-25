import argparse

import numpy as np
import torch

import metal_breakout_forward as metal_forward


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted, dtype=np.float64)
    return exp / exp.sum(axis=1, keepdims=True)


def _counts_1d(actions):
    return np.bincount(actions, minlength=metal_forward.ACTION_SIZE)


def _case_counts(actions, num_cases, agents_per_case):
    counts = np.zeros((num_cases, metal_forward.ACTION_SIZE), dtype=np.int64)
    for case_idx in range(num_cases):
        start = case_idx * agents_per_case
        stop = start + agents_per_case
        counts[case_idx] = _counts_1d(actions[:, start:stop].reshape(-1))
    return counts


def _case_step_counts(actions, num_cases, agents_per_case):
    horizon = actions.shape[0]
    counts = np.zeros(
        (num_cases, horizon, metal_forward.ACTION_SIZE),
        dtype=np.int64,
    )
    for case_idx in range(num_cases):
        start = case_idx * agents_per_case
        stop = start + agents_per_case
        for step in range(horizon):
            counts[case_idx, step] = _counts_1d(actions[step, start:stop])
    return counts


def _case_block_counts(actions, num_cases, agents_per_case, blocks):
    if agents_per_case % blocks != 0:
        raise ValueError("agents_per_case must be divisible by blocks")
    block_size = agents_per_case // blocks
    counts = np.zeros(
        (num_cases, blocks, metal_forward.ACTION_SIZE),
        dtype=np.int64,
    )
    for case_idx in range(num_cases):
        case_start = case_idx * agents_per_case
        for block in range(blocks):
            start = case_start + block * block_size
            stop = start + block_size
            counts[case_idx, block] = _counts_1d(actions[:, start:stop].reshape(-1))
    return counts


def _expected_probs(expected_probs, count_ndim):
    shape = (expected_probs.shape[0],) + (1,) * (count_ndim - 2) + (
        expected_probs.shape[1],
    )
    return expected_probs.reshape(shape)


def _chi_square(counts, expected_probs, draws):
    expected = _expected_probs(expected_probs, counts.ndim) * float(draws)
    return ((counts - expected) ** 2 / expected).sum(axis=-1)


def _total_variation(counts, expected_probs, draws):
    empirical = counts / float(draws)
    return 0.5 * np.abs(empirical - _expected_probs(expected_probs, counts.ndim)).sum(axis=-1)


def _accumulate(dst, src):
    if dst is None:
        return src.copy()
    dst += src
    return dst


def _load_fixed_logits(lib, context, config, logits):
    observations = np.zeros(
        (config.batch_size, metal_forward.OBS_SIZE),
        dtype=np.float32,
    )
    state = np.zeros(
        (metal_forward.NUM_LAYERS, config.batch_size, metal_forward.HIDDEN_SIZE),
        dtype=np.float32,
    )
    policy = metal_forward._policy()
    arrays = metal_forward._weight_arrays(policy)
    weights = metal_forward._weights_struct(arrays)
    metal_forward._metal_load_resident(
        lib,
        context,
        config,
        weights,
        observations,
        state,
    )
    values = np.zeros((config.batch_size,), dtype=np.float32)
    metal_forward._metal_load_sample_logits_resident(
        lib,
        context,
        config,
        logits,
        values,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-per-case", type=int, default=8192)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--batch-blocks", type=int, default=16)
    parser.add_argument("--max-tv-exact", type=float, default=0.01)
    parser.add_argument("--max-tv-torch", type=float, default=0.015)
    parser.add_argument("--max-seed-tv-exact", type=float, default=0.015)
    parser.add_argument("--max-step-tv-exact", type=float, default=0.03)
    parser.add_argument("--max-block-tv-exact", type=float, default=0.02)
    parser.add_argument("--max-chi-square", type=float, default=30.0)
    parser.add_argument("--max-seed-chi-square", type=float, default=40.0)
    parser.add_argument("--max-step-chi-square", type=float, default=45.0)
    parser.add_argument("--max-block-chi-square", type=float, default=45.0)
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()

    fixed_logits = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, -0.01, 0.0],
            [1.25, -0.5, 0.25],
            [-0.75, 1.0, 0.5],
            [0.2, -0.8, 1.3],
            [4.0, -2.0, -3.0],
            [-16.0, 16.0, -16.0],
            [16.0, -16.0, -16.0],
        ],
        dtype=np.float32,
    )
    num_logits = fixed_logits.shape[0]
    if args.agents_per_case % args.batch_blocks != 0:
        raise ValueError("--agents-per-case must be divisible by --batch-blocks")
    batch = num_logits * args.agents_per_case
    draws_per_logits = args.agents_per_case * args.horizon * args.seeds
    if draws_per_logits < 1_000_000:
        raise ValueError(
            "sampler gate requires at least 1,000,000 native draws per logits case; "
            f"got {draws_per_logits}"
        )
    logits = np.repeat(fixed_logits, args.agents_per_case, axis=0)
    probs = _softmax(fixed_logits)

    metal_forward._compile_library(args.force_build)
    lib = metal_forward._load_library()
    context = metal_forward._create_context(lib)

    native_counts = None
    native_seed_counts = []
    native_step_counts = None
    native_block_counts = None
    try:
        config = metal_forward.PolicyForwardConfig(
            batch,
            metal_forward.OBS_SIZE,
            metal_forward.HIDDEN_SIZE,
            metal_forward.NUM_LAYERS,
            metal_forward.ACTION_SIZE,
        )
        sample_config = metal_forward.PolicySampleConfig(
            0,
            args.horizon,
            args.seed,
            0,
        )
        _load_fixed_logits(lib, context, config, logits)
        for seed_idx in range(args.seeds):
            seed = args.seed + seed_idx
            for step in range(args.horizon):
                sample_config = metal_forward.PolicySampleConfig(
                    step,
                    args.horizon,
                    seed,
                    0,
                )
                metal_forward._metal_sample_rollout_resident(
                    lib,
                    context,
                    config,
                    sample_config,
                )
            metal_actions, _logprobs, _values = metal_forward._metal_read_rollout(
                lib,
                context,
                config,
                sample_config,
            )
            seed_counts = _case_counts(metal_actions, num_logits, args.agents_per_case)
            native_seed_counts.append(seed_counts)
            native_counts = _accumulate(native_counts, seed_counts)
            native_step_counts = _accumulate(
                native_step_counts,
                _case_step_counts(metal_actions, num_logits, args.agents_per_case),
            )
            native_block_counts = _accumulate(
                native_block_counts,
                _case_block_counts(
                    metal_actions,
                    num_logits,
                    args.agents_per_case,
                    args.batch_blocks,
                ),
            )
    finally:
        lib.puffer_metal_policy_destroy(context)

    native_seed_counts = np.stack(native_seed_counts, axis=1)
    generator = torch.Generator().manual_seed(args.seed)
    torch_counts = np.zeros_like(native_counts)
    for idx in range(num_logits):
        torch_actions = torch.multinomial(
            torch.from_numpy(probs[idx]),
            draws_per_logits,
            replacement=True,
            generator=generator,
        ).numpy()
        torch_counts[idx] = _counts_1d(torch_actions)

    native_tv_exact = _total_variation(native_counts, probs, draws_per_logits)
    torch_tv_exact = _total_variation(torch_counts, probs, draws_per_logits)
    native_tv_torch = 0.5 * np.abs(
        native_counts / float(draws_per_logits) -
        torch_counts / float(draws_per_logits)
    ).sum(axis=1)
    native_chi_square = _chi_square(native_counts, probs, draws_per_logits)
    torch_chi_square = _chi_square(torch_counts, probs, draws_per_logits)

    seed_draws = args.agents_per_case * args.horizon
    step_draws = args.agents_per_case * args.seeds
    block_draws = (args.agents_per_case // args.batch_blocks) * args.horizon * args.seeds
    native_seed_tv_exact = _total_variation(native_seed_counts, probs, seed_draws)
    native_seed_chi_square = _chi_square(native_seed_counts, probs, seed_draws)
    native_step_tv_exact = _total_variation(native_step_counts, probs, step_draws)
    native_step_chi_square = _chi_square(native_step_counts, probs, step_draws)
    native_block_tv_exact = _total_variation(native_block_counts, probs, block_draws)
    native_block_chi_square = _chi_square(native_block_counts, probs, block_draws)

    print(
        "sampler_distribution "
        f"draws_per_logits={draws_per_logits} "
        f"agents_per_case={args.agents_per_case} "
        f"horizon={args.horizon} "
        f"seeds={args.seeds} "
        f"seed={args.seed} "
        f"max_native_tv_exact={native_tv_exact.max():.8g} "
        f"max_torch_tv_exact={torch_tv_exact.max():.8g} "
        f"max_native_tv_torch={native_tv_torch.max():.8g} "
        f"max_native_chi_square={native_chi_square.max():.8g} "
        f"max_torch_chi_square={torch_chi_square.max():.8g} "
        f"max_native_seed_tv_exact={native_seed_tv_exact.max():.8g} "
        f"max_native_seed_chi_square={native_seed_chi_square.max():.8g} "
        f"max_native_step_tv_exact={native_step_tv_exact.max():.8g} "
        f"max_native_step_chi_square={native_step_chi_square.max():.8g} "
        f"max_native_block_tv_exact={native_block_tv_exact.max():.8g} "
        f"max_native_block_chi_square={native_block_chi_square.max():.8g} "
        f"max_tv_exact_threshold={args.max_tv_exact:.8g} "
        f"max_tv_torch_threshold={args.max_tv_torch:.8g} "
        f"max_seed_tv_exact_threshold={args.max_seed_tv_exact:.8g} "
        f"max_step_tv_exact_threshold={args.max_step_tv_exact:.8g} "
        f"max_block_tv_exact_threshold={args.max_block_tv_exact:.8g} "
        f"max_chi_square_threshold={args.max_chi_square:.8g} "
        f"max_seed_chi_square_threshold={args.max_seed_chi_square:.8g} "
        f"max_step_chi_square_threshold={args.max_step_chi_square:.8g} "
        f"max_block_chi_square_threshold={args.max_block_chi_square:.8g}"
    )
    for idx in range(num_logits):
        print(
            "sampler_distribution_case "
            f"idx={idx} "
            f"probs={probs[idx].tolist()} "
            f"native_counts={native_counts[idx].tolist()} "
            f"torch_counts={torch_counts[idx].tolist()} "
            f"native_tv_exact={native_tv_exact[idx]:.8g} "
            f"torch_tv_exact={torch_tv_exact[idx]:.8g} "
            f"native_tv_torch={native_tv_torch[idx]:.8g} "
            f"native_chi_square={native_chi_square[idx]:.8g} "
            f"torch_chi_square={torch_chi_square[idx]:.8g}"
        )

    if (
            native_tv_exact.max() > args.max_tv_exact or
            native_tv_torch.max() > args.max_tv_torch or
            native_seed_tv_exact.max() > args.max_seed_tv_exact or
            native_step_tv_exact.max() > args.max_step_tv_exact or
            native_block_tv_exact.max() > args.max_block_tv_exact or
            native_chi_square.max() > args.max_chi_square or
            native_seed_chi_square.max() > args.max_seed_chi_square or
            native_step_chi_square.max() > args.max_step_chi_square or
            native_block_chi_square.max() > args.max_block_chi_square):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
