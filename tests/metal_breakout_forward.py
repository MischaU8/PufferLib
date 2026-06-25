import argparse
import ctypes
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

import pufferlib.models
from pufferlib.torch_pufferl import compute_puff_advantage


ROOT = pathlib.Path(__file__).resolve().parents[1]
LIB_PATH = ROOT / "build" / "metal" / "libpuffer_metal.dylib"
KERNEL_PATH = ROOT / "pufferlib" / "metal" / "kernels" / "policy.metal"

OBS_SIZE = 118
HIDDEN_SIZE = 64
NUM_LAYERS = 2
ACTION_SIZE = 3
MAX_HEADS = 8
ACT_SIZES = (3,)
IS_CONTINUOUS = False
DEFAULT_GPU_TFLOPS = 10.4


class PolicyForwardConfig(ctypes.Structure):
    _fields_ = [
        ("batch_size", ctypes.c_uint32),
        ("obs_size", ctypes.c_uint32),
        ("hidden_size", ctypes.c_uint32),
        ("num_layers", ctypes.c_uint32),
        ("action_size", ctypes.c_uint32),
        ("num_atns", ctypes.c_uint32),
        ("act_sizes", ctypes.c_uint32 * MAX_HEADS),
        ("is_continuous", ctypes.c_uint32),
    ]


class PolicyForwardWeights(ctypes.Structure):
    _fields_ = [
        ("encoder_weight", ctypes.POINTER(ctypes.c_float)),
        ("encoder_bias", ctypes.POINTER(ctypes.c_float)),
        ("gru_weights", ctypes.POINTER(ctypes.c_float)),
        ("decoder_weight", ctypes.POINTER(ctypes.c_float)),
        ("decoder_bias", ctypes.POINTER(ctypes.c_float)),
        ("value_weight", ctypes.POINTER(ctypes.c_float)),
        ("value_bias", ctypes.POINTER(ctypes.c_float)),
        ("decoder_logstd", ctypes.POINTER(ctypes.c_float)),
    ]


class PolicySampleConfig(ctypes.Structure):
    _fields_ = [
        ("rollout_step", ctypes.c_uint32),
        ("rollout_horizon", ctypes.c_uint32),
        ("seed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class PuffAdvantageConfig(ctypes.Structure):
    _fields_ = [
        ("num_steps", ctypes.c_uint32),
        ("horizon", ctypes.c_uint32),
        ("gamma", ctypes.c_float),
        ("gae_lambda", ctypes.c_float),
        ("vtrace_rho_clip", ctypes.c_float),
        ("vtrace_c_clip", ctypes.c_float),
    ]


def _compile_library(force=False):
    source = ROOT / "pufferlib" / "metal" / "puffer_metal.mm"
    header = ROOT / "pufferlib" / "metal" / "puffer_metal.h"
    kernel = ROOT / "pufferlib" / "metal" / "kernels" / "policy.metal"
    needs_build = force or not LIB_PATH.exists()
    if not needs_build:
        newest_input = max(source.stat().st_mtime, header.stat().st_mtime, kernel.stat().st_mtime)
        needs_build = LIB_PATH.stat().st_mtime < newest_input

    if not needs_build:
        return

    cmd = ["bash", "build.sh", "breakout", "--metal-native"]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_library():
    lib = ctypes.CDLL(str(LIB_PATH))
    ctx_p = ctypes.POINTER(ctypes.c_void_p)

    lib.puffer_metal_policy_create.argtypes = [
        ctypes.c_char_p,
        ctx_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_create.restype = ctypes.c_int
    lib.puffer_metal_policy_destroy.argtypes = [ctypes.c_void_p]
    lib.puffer_metal_policy_destroy.restype = None
    lib.puffer_metal_policy_load_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(PolicyForwardWeights),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_load_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_forward_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_forward_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_forward_resident_async.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_forward_resident_async.restype = ctypes.c_int
    lib.puffer_metal_policy_sample_rollout_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(PolicySampleConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_sample_rollout_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_load_sample_logits_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_load_sample_logits_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_sample_rollout_resident_async.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(PolicySampleConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_sample_rollout_resident_async.restype = ctypes.c_int
    lib.puffer_metal_policy_forward_sample_rollout_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(PolicySampleConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_forward_sample_rollout_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_synchronize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_synchronize.restype = ctypes.c_int
    lib.puffer_metal_policy_read_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_read_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_read_actions_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_read_actions_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_read_rollout_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(PolicySampleConfig),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_read_rollout_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_load_advantage_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PuffAdvantageConfig),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_load_advantage_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_compute_advantage_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PuffAdvantageConfig),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_compute_advantage_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_read_advantage_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PuffAdvantageConfig),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_read_advantage_resident.restype = ctypes.c_int
    lib.puffer_metal_policy_advantage.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PuffAdvantageConfig),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_advantage.restype = ctypes.c_int
    lib.puffer_metal_policy_forward_context.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PolicyForwardConfig),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(PolicyForwardWeights),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.puffer_metal_policy_forward_context.restype = ctypes.c_int
    return lib


def _raise_on_error(rc, error):
    if rc != 0:
        message = error.value.decode("utf-8", "replace")
        raise RuntimeError(message)


def _policy():
    torch.manual_seed(7)
    encoder = pufferlib.models.DefaultEncoder(OBS_SIZE, HIDDEN_SIZE)
    network = pufferlib.models.MinGRU(HIDDEN_SIZE, NUM_LAYERS)
    decoder = pufferlib.models.DefaultDecoder(tuple(ACT_SIZES), HIDDEN_SIZE)
    policy = pufferlib.models.Policy(encoder, decoder, network)
    return policy.eval()


def _numpy_param(tensor):
    return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32, copy=False))


def _weight_arrays(policy):
    state = policy.state_dict()
    gru_layers = [_numpy_param(state[f"network.layers.{i}.weight"])
                  for i in range(NUM_LAYERS)]
    gru_weights = np.ascontiguousarray(np.concatenate(gru_layers, axis=0))
    if IS_CONTINUOUS:
        decoder_weight = _numpy_param(state["decoder.decoder_mean.weight"])
        decoder_bias = _numpy_param(state["decoder.decoder_mean.bias"])
        decoder_logstd = _numpy_param(state["decoder.decoder_logstd"]).reshape(-1)
    else:
        decoder_weight = _numpy_param(state["decoder.decoder.weight"])
        decoder_bias = _numpy_param(state["decoder.decoder.bias"])
        decoder_logstd = np.zeros((1,), dtype=np.float32)
    return {
        "encoder_weight": _numpy_param(state["encoder.encoder.weight"]),
        "encoder_bias": _numpy_param(state["encoder.encoder.bias"]),
        "gru_weights": gru_weights,
        "decoder_weight": decoder_weight,
        "decoder_bias": decoder_bias,
        "value_weight": _numpy_param(state["decoder.value_function.weight"]).reshape(-1),
        "value_bias": _numpy_param(state["decoder.value_function.bias"]),
        "decoder_logstd": decoder_logstd,
    }


def _weights_struct(arrays):
    ptr = ctypes.POINTER(ctypes.c_float)
    return PolicyForwardWeights(
        *(arrays[name].ctypes.data_as(ptr) for name, _ in PolicyForwardWeights._fields_)
    )


def _create_context(lib):
    error = ctypes.create_string_buffer(4096)
    context = ctypes.c_void_p()
    rc = lib.puffer_metal_policy_create(
        str(KERNEL_PATH).encode("utf-8"),
        ctypes.byref(context),
        error,
        len(error),
    )
    _raise_on_error(rc, error)
    return context


def _metal_forward(lib, context, config, weights, observations, state):
    logits = np.empty((config.batch_size, ACTION_SIZE), dtype=np.float32)
    values = np.empty((config.batch_size,), dtype=np.float32)
    state_out = np.empty((NUM_LAYERS, config.batch_size, HIDDEN_SIZE), dtype=np.float32)
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_forward_context(
        context,
        ctypes.byref(config),
        observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(weights),
        logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        state_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error,
        len(error),
    )
    _raise_on_error(rc, error)
    return logits, values, state_out


def _metal_load_resident(lib, context, config, weights, observations, state):
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_load_resident(
        context,
        ctypes.byref(config),
        observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.byref(weights),
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_forward_resident(lib, context, config):
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_forward_resident(
        context,
        ctypes.byref(config),
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_forward_resident_async(lib, context, config):
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_forward_resident_async(
        context,
        ctypes.byref(config),
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_sample_rollout_resident(lib, context, config, sample_config):
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_sample_rollout_resident(
        context,
        ctypes.byref(config),
        ctypes.byref(sample_config),
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_load_sample_logits_resident(lib, context, config, logits, values):
    error = ctypes.create_string_buffer(4096)
    logits = np.ascontiguousarray(logits.astype(np.float32, copy=False))
    values = np.ascontiguousarray(values.astype(np.float32, copy=False))
    rc = lib.puffer_metal_policy_load_sample_logits_resident(
        context,
        ctypes.byref(config),
        logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_sample_rollout_resident_async(lib, context, config, sample_config):
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_sample_rollout_resident_async(
        context,
        ctypes.byref(config),
        ctypes.byref(sample_config),
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_forward_sample_rollout_resident(lib, context, config, sample_config):
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_forward_sample_rollout_resident(
        context,
        ctypes.byref(config),
        ctypes.byref(sample_config),
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_synchronize(lib, context):
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_synchronize(
        context,
        error,
        len(error),
    )
    _raise_on_error(rc, error)


def _metal_read_actions(lib, context, config):
    actions = np.empty((config.batch_size, config.num_atns), dtype=np.int32)
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_read_actions_resident(
        context,
        ctypes.byref(config),
        actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        error,
        len(error),
    )
    _raise_on_error(rc, error)
    return actions


def _metal_read_rollout(lib, context, config, sample_config):
    shape = (sample_config.rollout_horizon, config.batch_size)
    actions = np.empty((sample_config.rollout_horizon, config.batch_size, config.num_atns), dtype=np.int32)
    logprobs = np.empty(shape, dtype=np.float32)
    values = np.empty(shape, dtype=np.float32)
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_read_rollout_resident(
        context,
        ctypes.byref(config),
        ctypes.byref(sample_config),
        actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        logprobs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error,
        len(error),
    )
    _raise_on_error(rc, error)
    return actions, logprobs, values


def _metal_advantage(lib, context, config, values, rewards, terminals, ratio):
    advantages = np.empty_like(values, dtype=np.float32)
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_advantage(
        context,
        ctypes.byref(config),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        terminals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ratio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        advantages.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error,
        len(error),
    )
    _raise_on_error(rc, error)
    return advantages


def _reference_advantage(values, rewards, terminals, ratio, config):
    advantages = np.zeros_like(values, dtype=np.float32)
    for row in range(values.shape[0]):
        last = np.float32(0.0)
        for t in range(values.shape[1] - 2, -1, -1):
            nextnonterminal = np.float32(1.0) - terminals[row, t + 1]
            rho_t = np.minimum(ratio[row, t], np.float32(config.vtrace_rho_clip))
            c_t = np.minimum(ratio[row, t], np.float32(config.vtrace_c_clip))
            delta = (
                rho_t * rewards[row, t + 1]
                + np.float32(config.gamma) * values[row, t + 1] * nextnonterminal
                - values[row, t]
            )
            last = (
                delta
                + np.float32(config.gamma)
                * np.float32(config.gae_lambda)
                * c_t
                * last
                * nextnonterminal
            )
            advantages[row, t] = last
    return advantages


def _cpu_extension_advantage(values, rewards, terminals, ratio, config):
    advantages = torch.zeros_like(torch.from_numpy(values.copy()))
    compute_puff_advantage(
        torch.from_numpy(values.copy()),
        torch.from_numpy(rewards.copy()),
        torch.from_numpy(terminals.copy()),
        torch.from_numpy(ratio.copy()),
        advantages,
        float(config.gamma),
        float(config.gae_lambda),
        float(config.vtrace_rho_clip),
        float(config.vtrace_c_clip),
    )
    return advantages.numpy()


def _hash_u32(x):
    x = np.uint32(x)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def _uniform01(seed, step, batch, head=0):
    idx = np.arange(batch, dtype=np.uint64)
    x = (
        (int(seed) & 0xFFFFFFFF)
        ^ ((int(step) * 0x9E3779B9) & 0xFFFFFFFF)
        ^ ((idx * 0x85EBCA6B) & 0xFFFFFFFF)
        ^ ((int(head) * 0xC2B2AE35) & 0xFFFFFFFF)
    ).astype(np.uint32, copy=False)
    h = _hash_u32(x)
    return (((h >> np.uint32(8)) & np.uint32(0x00FFFFFF)).astype(np.float32) + 0.5) / 16777216.0


def _reference_sample(logits, seed, step):
    # Mirrors the kernel's multi-head cumulative categorical pick: each head owns
    # its act_sizes segment of the concatenated logits, is sampled with a
    # head-keyed uniform, and the joint log-probability is the sum of the
    # per-head log-softmax values. Returns actions [batch, num_atns] and the
    # summed logprob [batch], matching the torch multi-discrete convention.
    clean = np.nan_to_num(logits.astype(np.float32, copy=False), nan=0.0, posinf=1e20, neginf=-1e20)
    clean = np.clip(clean, -1e20, 1e20)
    batch = clean.shape[0]
    actions = np.empty((batch, len(ACT_SIZES)), dtype=np.int32)
    logprobs = np.zeros((batch,), dtype=np.float32)
    offset = 0
    for head, head_size in enumerate(ACT_SIZES):
        seg = clean[:, offset:offset + head_size]
        shifted = seg - seg.max(axis=1, keepdims=True)
        probs = np.exp(shifted, dtype=np.float32)
        probs /= probs.sum(axis=1, keepdims=True)
        u = _uniform01(seed, step, batch, head)
        cumsum = np.cumsum(probs, axis=1)
        act = (u[:, None] < cumsum).argmax(axis=1).astype(np.int32)
        act[u >= cumsum[:, -1]] = head_size - 1
        actions[:, head] = act
        logsumexp = seg.max(axis=1) + np.log(np.exp(shifted, dtype=np.float32).sum(axis=1))
        logprobs += seg[np.arange(batch), act] - logsumexp
        offset += head_size
    return actions, logprobs.astype(np.float32, copy=False)


def _metal_read_resident(lib, context, config):
    logits = np.empty((config.batch_size, ACTION_SIZE), dtype=np.float32)
    values = np.empty((config.batch_size,), dtype=np.float32)
    state_out = np.empty((NUM_LAYERS, config.batch_size, HIDDEN_SIZE), dtype=np.float32)
    error = ctypes.create_string_buffer(4096)
    rc = lib.puffer_metal_policy_read_resident(
        context,
        ctypes.byref(config),
        logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        state_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        error,
        len(error),
    )
    _raise_on_error(rc, error)
    return logits, values, state_out


def _time_call(fn, iters):
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - start) / iters


def _bench_torch(policy, observations, state, device, iters):
    model = policy.to(device)
    obs = observations.to(device)
    recurrent_state = (state.to(device),)
    with torch.no_grad():
        for _ in range(10):
            _logits, _values, recurrent_state = model.forward_eval(obs, recurrent_state)
        if device == "mps":
            torch.mps.synchronize()

        def run():
            nonlocal recurrent_state
            _logits, _values, recurrent_state = model.forward_eval(obs, recurrent_state)
            if device == "mps":
                torch.mps.synchronize()

        return _time_call(run, iters)


def _bench_torch_throughput(policy, observations, state, device, iters):
    model = policy.to(device)
    obs = observations.to(device)
    recurrent_state = (state.to(device),)
    with torch.no_grad():
        for _ in range(10):
            _logits, _values, recurrent_state = model.forward_eval(obs, recurrent_state)
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            _logits, _values, recurrent_state = model.forward_eval(obs, recurrent_state)
        if device == "mps":
            torch.mps.synchronize()
        return (time.perf_counter() - start) / iters


def _torch_sample_logits(logits, device):
    # Throughput timing only (correctness is covered by the parity gate).
    if isinstance(logits, torch.distributions.Normal):
        action = logits.sample()
        return action, logits.log_prob(action).sum(-1)
    # Multi-discrete forward_eval returns per-head segments; concatenate and
    # sample over the joined space as a representative proxy.
    if isinstance(logits, (tuple, list)):
        logits = torch.cat(list(logits), dim=1)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.nan_to_num(log_probs.exp(), 1e-8, 1e-8, 1e-8)
    sample_probs = probs.cpu() if device == "mps" else probs
    actions = torch.multinomial(sample_probs, 1, replacement=True).to(logits.device)
    logprob = log_probs.gather(1, actions).squeeze(1)
    return actions, logprob


def _bench_torch_forward_sample(policy, observations, state, device, iters):
    model = policy.to(device)
    obs = observations.to(device)
    recurrent_state = (state.to(device),)
    with torch.no_grad():
        for _ in range(10):
            logits, values, recurrent_state = model.forward_eval(obs, recurrent_state)
            _torch_sample_logits(logits, device)
        if device == "mps":
            torch.mps.synchronize()

        start = time.perf_counter()
        for i in range(iters):
            logits, values, recurrent_state = model.forward_eval(obs, recurrent_state)
            # Compute (and thus time) forward + sample; results are not stored as
            # this is a throughput benchmark and action shapes vary by policy.
            _torch_sample_logits(logits, device)
            if device == "mps":
                torch.mps.synchronize()
        return (time.perf_counter() - start) / iters


def _forward_flops_per_step(batch):
    encoder = 2 * OBS_SIZE * HIDDEN_SIZE
    mingru = NUM_LAYERS * 3 * 2 * HIDDEN_SIZE * HIDDEN_SIZE
    decoder = 2 * HIDDEN_SIZE * ACTION_SIZE
    value = 2 * HIDDEN_SIZE
    return int(batch) * (encoder + mingru + decoder + value)


def _hardware_ceiling_seconds(batch, tflops):
    return _forward_flops_per_step(batch) / (float(tflops) * 1.0e12)


def main():
    global OBS_SIZE, HIDDEN_SIZE, ACTION_SIZE, NUM_LAYERS, ACT_SIZES, IS_CONTINUOUS
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--obs-size", type=int, default=OBS_SIZE,
        help="Encoder input size (default: Breakout 118).")
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE,
        help="MinGRU hidden size (default: Breakout 64; max 512).")
    parser.add_argument("--action-size", type=int, default=ACTION_SIZE,
        help="Single-head discrete action count (default: Breakout 3). Ignored if --act-sizes is set.")
    parser.add_argument("--act-sizes", type=str, default=None,
        help="Comma-separated per-head action sizes for a multi-discrete policy, e.g. 9,5.")
    parser.add_argument("--continuous", type=int, default=0,
        help="If >0, use a continuous (Gaussian) policy with this many action dims.")
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS,
        help="MinGRU layer count (default: Breakout 2).")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--rollout-horizon", type=int, default=4)
    parser.add_argument("--sample-step", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=12345)
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--sample-tolerance", type=float, default=2e-5)
    parser.add_argument("--advantage-tolerance", type=float, default=2e-5)
    parser.add_argument("--gpu-tflops", type=float, default=DEFAULT_GPU_TFLOPS,
        help="Nominal FP32 TFLOP/s used for the forward hardware-ceiling estimate.")
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()
    OBS_SIZE = args.obs_size
    HIDDEN_SIZE = args.hidden_size
    NUM_LAYERS = args.num_layers
    IS_CONTINUOUS = args.continuous > 0
    if IS_CONTINUOUS:
        ACT_SIZES = tuple(1 for _ in range(args.continuous))  # nvec all 1s => continuous
        ACTION_SIZE = args.continuous                          # one mean per dim
    elif args.act_sizes:
        ACT_SIZES = tuple(int(x) for x in args.act_sizes.split(","))
        ACTION_SIZE = sum(ACT_SIZES)
    else:
        ACT_SIZES = (args.action_size,)
        ACTION_SIZE = sum(ACT_SIZES)
    if not (0 < HIDDEN_SIZE <= 512):
        raise ValueError("--hidden-size must be in [1, 512]")
    if OBS_SIZE <= 0 or any(a <= 0 for a in ACT_SIZES):
        raise ValueError("--obs-size and all action sizes must be positive")
    if not (1 <= len(ACT_SIZES) <= MAX_HEADS):
        raise ValueError(f"number of action heads must be in [1, {MAX_HEADS}]")
    if NUM_LAYERS < 1:
        raise ValueError("--num-layers must be >= 1")
    if args.rollout_horizon <= 0:
        raise ValueError("--rollout-horizon must be positive")
    if args.sample_step < 0 or args.sample_step >= args.rollout_horizon:
        raise ValueError("--sample-step must be within --rollout-horizon")

    _compile_library(args.force_build)
    lib = _load_library()
    context = _create_context(lib)

    try:
        policy = _policy()
        arrays = _weight_arrays(policy)
        weights = _weights_struct(arrays)

        generator = torch.Generator().manual_seed(11)
        observations = torch.randn(args.batch, OBS_SIZE, generator=generator)
        state = torch.randn(NUM_LAYERS, args.batch, HIDDEN_SIZE, generator=generator)

        act_sizes_arr = (ctypes.c_uint32 * MAX_HEADS)(
            *(list(ACT_SIZES) + [0] * (MAX_HEADS - len(ACT_SIZES))))
        config = PolicyForwardConfig(
            args.batch,
            OBS_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS,
            ACTION_SIZE,
            len(ACT_SIZES),
            act_sizes_arr,
            1 if IS_CONTINUOUS else 0,
        )
        sample_config = PolicySampleConfig(
            args.sample_step,
            args.rollout_horizon,
            args.sample_seed,
            0,
        )
        advantage_config = PuffAdvantageConfig(
            args.batch,
            args.rollout_horizon,
            0.998,
            0.95,
            1.0,
            1.0,
        )
        obs_np = np.ascontiguousarray(observations.numpy().astype(np.float32, copy=False))
        state_np = np.ascontiguousarray(state.numpy().astype(np.float32, copy=False))

        with torch.no_grad():
            cpu_logits, cpu_values, cpu_state = policy.forward_eval(
                observations.clone(),
                (state.clone(),),
            )
            cpu_std = None
            if isinstance(cpu_logits, torch.distributions.Normal):
                # Continuous: the Metal "logits" are the decoder means.
                cpu_std = cpu_logits.scale
                cpu_logits = cpu_logits.loc
            elif isinstance(cpu_logits, (tuple, list)):
                # Multi-discrete forward_eval returns per-head logit segments; the
                # Metal path emits them concatenated, so join for comparison.
                cpu_logits = torch.cat(list(cpu_logits), dim=1)

        metal_logits, metal_values, metal_state = _metal_forward(
            lib, context, config, weights, obs_np, state_np)

        _metal_load_resident(lib, context, config, weights, obs_np, state_np)
        _metal_forward_resident(lib, context, config)
        resident_logits, resident_values, resident_state = _metal_read_resident(
            lib, context, config)
        _metal_sample_rollout_resident(lib, context, config, sample_config)
        metal_actions = _metal_read_actions(lib, context, config)
        rollout_actions, rollout_logprobs, rollout_values = _metal_read_rollout(
            lib, context, config, sample_config)
        logits_diff = np.max(np.abs(metal_logits - cpu_logits.numpy()))
        values_diff = np.max(np.abs(metal_values - cpu_values.reshape(-1).numpy()))
        state_diff = np.max(np.abs(metal_state - cpu_state[0].numpy()))
        resident_logits_diff = np.max(np.abs(resident_logits - cpu_logits.numpy()))
        resident_values_diff = np.max(np.abs(resident_values - cpu_values.reshape(-1).numpy()))
        resident_state_diff = np.max(np.abs(resident_state - cpu_state[0].numpy()))

        dist_z_mean = 0.0
        dist_z_std = 1.0
        dist_ok = True
        if IS_CONTINUOUS:
            num_dims = len(ACT_SIZES)
            # Kernel actions are floats stored in the int32 buffers; reinterpret.
            metal_actions_f = metal_actions.view(np.float32)
            roll_step_f = rollout_actions[args.sample_step].view(np.float32)
            # The per-step buffer and the rollout slice come from the same
            # dispatch, so they must be bit-identical.
            sample_actions_match = bool(np.array_equal(metal_actions_f, roll_step_f))
            rollout_actions_match = sample_actions_match
            # Log-prob of the kernel's sampled actions under the torch Gaussian.
            means = resident_logits.reshape(args.batch, num_dims)
            std = cpu_std.numpy().reshape(args.batch, num_dims)
            z = (roll_step_f.reshape(args.batch, num_dims) - means) / std
            ref_logprobs = (
                -0.5 * z * z - np.log(std) - 0.5 * np.log(2.0 * np.pi)
            ).sum(axis=1).astype(np.float32)
            rollout_logprobs_diff = np.max(np.abs(
                rollout_logprobs[args.sample_step] - ref_logprobs))
            rollout_values_diff = np.max(np.abs(
                rollout_values[args.sample_step] - resident_values))
            # Standardized residuals across all (batch, dim) should be ~N(0, 1).
            # Use 5-standard-error bounds so the gate is robust to the sample
            # count (mean SE = 1/sqrt(n), std SE = 1/sqrt(2n)).
            z_flat = z.reshape(-1)
            n = z_flat.size
            dist_z_mean = float(z_flat.mean())
            dist_z_std = float(z_flat.std())
            tol_mean = max(0.02, 5.0 / np.sqrt(n))
            tol_std = max(0.02, 5.0 / np.sqrt(2.0 * n))
            dist_ok = abs(dist_z_mean) < tol_mean and abs(dist_z_std - 1.0) < tol_std
        else:
            expected_actions, expected_logprobs = _reference_sample(
                resident_logits, args.sample_seed, args.sample_step)
            sample_actions_match = bool(np.array_equal(metal_actions, expected_actions))
            rollout_actions_match = bool(np.array_equal(
                rollout_actions[args.sample_step], expected_actions))
            rollout_logprobs_diff = np.max(np.abs(
                rollout_logprobs[args.sample_step] - expected_logprobs))
            rollout_values_diff = np.max(np.abs(
                rollout_values[args.sample_step] - resident_values))
        max_diff = max(
            logits_diff,
            values_diff,
            state_diff,
            resident_logits_diff,
            resident_values_diff,
            resident_state_diff,
        )
        sample_max_diff = max(rollout_logprobs_diff, rollout_values_diff)

        print(
            "parity "
            f"batch={args.batch} "
            f"logits_max_abs={logits_diff:.8g} "
            f"values_max_abs={values_diff:.8g} "
            f"state_max_abs={state_diff:.8g} "
            f"resident_logits_max_abs={resident_logits_diff:.8g} "
            f"resident_values_max_abs={resident_values_diff:.8g} "
            f"resident_state_max_abs={resident_state_diff:.8g} "
            f"tolerance={args.tolerance:.8g}"
        )
        if max_diff > args.tolerance:
            return 1
        print(
            "sample_parity "
            f"batch={args.batch} "
            f"continuous={int(IS_CONTINUOUS)} "
            f"rollout_horizon={args.rollout_horizon} "
            f"rollout_step={args.sample_step} "
            f"seed={args.sample_seed} "
            f"actions_match={sample_actions_match} "
            f"rollout_actions_match={rollout_actions_match} "
            f"logprobs_max_abs={rollout_logprobs_diff:.8g} "
            f"values_max_abs={rollout_values_diff:.8g} "
            f"dist_z_mean={dist_z_mean:.4g} dist_z_std={dist_z_std:.4g} dist_ok={dist_ok} "
            f"tolerance={args.sample_tolerance:.8g}"
        )
        if (not sample_actions_match or not rollout_actions_match or not dist_ok or
                sample_max_diff > args.sample_tolerance):
            return 1

        rng = np.random.default_rng(23)
        advantage_values = rng.normal(
            loc=0.0,
            scale=0.5,
            size=(args.batch, args.rollout_horizon),
        ).astype(np.float32)
        advantage_rewards = np.clip(
            rng.normal(size=(args.batch, args.rollout_horizon)),
            -1.0,
            1.0,
        ).astype(np.float32)
        advantage_terminals = (
            rng.random(size=(args.batch, args.rollout_horizon)) < 0.08
        ).astype(np.float32)
        advantage_terminals[:, 0] = 0.0
        advantage_ratio = rng.uniform(
            0.2,
            1.8,
            size=(args.batch, args.rollout_horizon),
        ).astype(np.float32)
        metal_advantages = _metal_advantage(
            lib,
            context,
            advantage_config,
            np.ascontiguousarray(advantage_values),
            np.ascontiguousarray(advantage_rewards),
            np.ascontiguousarray(advantage_terminals),
            np.ascontiguousarray(advantage_ratio),
        )
        expected_advantages = _reference_advantage(
            advantage_values,
            advantage_rewards,
            advantage_terminals,
            advantage_ratio,
            advantage_config,
        )
        cpu_extension_advantages = _cpu_extension_advantage(
            np.ascontiguousarray(advantage_values),
            np.ascontiguousarray(advantage_rewards),
            np.ascontiguousarray(advantage_terminals),
            np.ascontiguousarray(advantage_ratio),
            advantage_config,
        )
        advantage_numpy_diff = np.max(np.abs(metal_advantages - expected_advantages))
        advantage_cpu_extension_diff = np.max(
            np.abs(metal_advantages - cpu_extension_advantages))
        advantage_diff = max(advantage_numpy_diff, advantage_cpu_extension_diff)
        print(
            "advantage_parity "
            f"num_steps={args.batch} "
            f"horizon={args.rollout_horizon} "
            f"numpy_max_abs={advantage_numpy_diff:.8g} "
            f"cpu_extension_max_abs={advantage_cpu_extension_diff:.8g} "
            f"tolerance={args.advantage_tolerance:.8g}"
        )
        if advantage_diff > args.advantage_tolerance:
            return 1

        _metal_load_resident(lib, context, config, weights, obs_np, state_np)
        for _ in range(10):
            _metal_forward_resident(lib, context, config)
        metal_sync_seconds = _time_call(
            lambda: _metal_forward_resident(lib, context, config),
            args.iters,
        )

        _metal_load_resident(lib, context, config, weights, obs_np, state_np)
        for _ in range(10):
            _metal_forward_resident_async(lib, context, config)
        _metal_synchronize(lib, context)
        start = time.perf_counter()
        for _ in range(args.iters):
            _metal_forward_resident_async(lib, context, config)
        _metal_synchronize(lib, context)
        metal_seconds = (time.perf_counter() - start) / args.iters
        _metal_load_resident(lib, context, config, weights, obs_np, state_np)
        for _ in range(10):
            _metal_forward_sample_rollout_resident(lib, context, config, sample_config)
        metal_forward_sample_seconds = _time_call(
            lambda: _metal_forward_sample_rollout_resident(lib, context, config, sample_config),
            args.iters,
        )
        cpu_seconds = _bench_torch(policy, observations, state, "cpu", args.iters)
        cpu_forward_sample_seconds = _bench_torch_forward_sample(
            policy, observations, state, "cpu", args.iters)

        mps_seconds = None
        mps_throughput_seconds = None
        mps_forward_sample_seconds = None
        if torch.backends.mps.is_available():
            mps_policy = _policy()
            mps_policy.load_state_dict(policy.state_dict())
            mps_seconds = _bench_torch(mps_policy, observations, state, "mps", args.iters)
            mps_throughput_policy = _policy()
            mps_throughput_policy.load_state_dict(policy.state_dict())
            mps_throughput_seconds = _bench_torch_throughput(
                mps_throughput_policy, observations, state, "mps", args.iters)
            mps_forward_sample_policy = _policy()
            mps_forward_sample_policy.load_state_dict(policy.state_dict())
            mps_forward_sample_seconds = _bench_torch_forward_sample(
                mps_forward_sample_policy, observations, state, "mps", args.iters)

        forward_flops = _forward_flops_per_step(args.batch)
        ceiling_seconds = _hardware_ceiling_seconds(args.batch, args.gpu_tflops)
        print(
            "benchmark "
            f"batch={args.batch} "
            f"iters={args.iters} "
            f"forward_flops={forward_flops} "
            f"ceiling_us={ceiling_seconds * 1.0e6:.3f} "
            f"gpu_tflops={args.gpu_tflops:.3f} "
            f"metal_sync_ms={metal_sync_seconds * 1000.0:.4f} "
            f"metal_queued_ms={metal_seconds * 1000.0:.4f} "
            f"metal_forward_sample_ms={metal_forward_sample_seconds * 1000.0:.4f} "
            f"metal_queued_vs_ceiling={metal_seconds / ceiling_seconds:.1f}x "
            f"metal_forward_sample_vs_ceiling={metal_forward_sample_seconds / ceiling_seconds:.1f}x "
            f"cpu_ms={cpu_seconds * 1000.0:.4f} "
            f"cpu_forward_sample_ms={cpu_forward_sample_seconds * 1000.0:.4f} "
            f"metal_sync_vs_cpu={cpu_seconds / metal_sync_seconds:.3f}x "
            f"metal_queued_vs_cpu={cpu_seconds / metal_seconds:.3f}x "
            f"metal_forward_sample_vs_cpu_forward_sample={cpu_forward_sample_seconds / metal_forward_sample_seconds:.3f}x"
            + (
                f" mps_sync_ms={mps_seconds * 1000.0:.4f} "
                f"mps_queued_ms={mps_throughput_seconds * 1000.0:.4f} "
                f"mps_forward_sample_ms={mps_forward_sample_seconds * 1000.0:.4f} "
                f"metal_sync_vs_mps_sync={mps_seconds / metal_sync_seconds:.3f}x "
                f"metal_queued_vs_mps_queued={mps_throughput_seconds / metal_seconds:.3f}x "
                f"metal_forward_sample_vs_mps_forward_sample={mps_forward_sample_seconds / metal_forward_sample_seconds:.3f}x"
                if mps_seconds is not None
                else " mps_ms=unavailable"
            )
        )
        return 0
    finally:
        lib.puffer_metal_policy_destroy(context)


if __name__ == "__main__":
    sys.exit(main())
