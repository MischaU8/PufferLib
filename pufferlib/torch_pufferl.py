## puffer [train | eval | sweep] [env_name] [optional args] -- See https://puffer.ai for full detail0
# This is the same as python -m pufferlib.pufferl [train | eval | sweep] [env_name] [optional args]
# Distributed example: torchrun --standalone --nnodes=1 --nproc-per-node=6 -m pufferlib.pufferl train puffer_nmmo3

import os
import glob
import time
import ctypes
import pathlib
import subprocess
import sys
from collections import defaultdict

import numpy as np

import torch
import torch.distributed
from torch.distributions.utils import logits_to_probs

import pufferlib
import pufferlib.models
import pufferlib.pufferl
from pufferlib.muon import Muon
from pufferlib import _C
if _C.precision_bytes != 4:
    raise RuntimeError(
        f'_C was compiled with bf16 precision (precision_bytes={_C.precision_bytes}). '
        'The PyTorch backend requires float32. Rerun build.sh with --float'
    )

_OBS_DTYPE_MAP = {
    'ByteTensor':   torch.uint8,
    'FloatTensor':  torch.float32,
}

_TORCH_TO_TYPESTR = {
    torch.uint8:   '|u1',
    torch.float32: '<f4',
}

def _log_prob(logits, value):
    value = value.long().unsqueeze(-1)
    value, log_pmf = torch.broadcast_tensors(value, logits)
    value = value[..., :1]
    return log_pmf.gather(-1, value).squeeze(-1)

def _entropy(logits):
    min_real = torch.finfo(logits.dtype).min
    logits = torch.clamp(logits, min=min_real)
    p_log_p = logits * logits_to_probs(logits)
    return -p_log_p.sum(-1)

def _split_action_mask(action_mask, logits):
    sizes = [l.shape[-1] for l in logits]
    return torch.split(action_mask.reshape(-1, sum(sizes)), sizes, dim=-1)

def _apply_action_mask(logits, action_mask):
    if action_mask is None:
        return logits
    if isinstance(logits, torch.Tensor):
        mask = action_mask.reshape(logits.shape).to(
            device=logits.device, dtype=torch.bool)
        return logits.masked_fill_(~mask, -1.0e4)
    masks = _split_action_mask(action_mask, logits)
    masked = []
    for l, m in zip(logits, masks):
        mask = m.to(device=l.device, dtype=torch.bool)
        masked.append(l.masked_fill_(~mask, -1.0e4))
    return tuple(masked)

def _action_mask_for_sampled_logits(action_mask, logits):
    if action_mask is None:
        return None
    if isinstance(logits, torch.Tensor):
        return action_mask.reshape(logits.shape).to(
            device=logits.device, dtype=torch.bool).unsqueeze(0)
    masks = _split_action_mask(action_mask, logits)
    return torch.nn.utils.rnn.pad_sequence(
        [m.to(device=l.device, dtype=torch.bool).transpose(0, 1)
         for l, m in zip(logits, masks)],
        batch_first=False,
        padding_value=False,
    ).permute(1, 2, 0)

def _repair_sampled_actions(action, action_mask):
    sampled_is_legal = action_mask.gather(
        -1, action.long().unsqueeze(-1)).squeeze(-1)
    has_legal_action = action_mask.any(dim=-1)
    action_indices = torch.arange(
        action_mask.shape[-1], device=action_mask.device).view(
            *([1] * (action_mask.ndim - 1)), action_mask.shape[-1])
    fallback = torch.where(action_mask, action_indices, -1).amax(dim=-1)
    repair = has_legal_action & ~sampled_is_legal
    return torch.where(repair, fallback.to(action.dtype), action)

def sample_logits(logits, action=None, action_mask=None):
    is_discrete = isinstance(logits, torch.Tensor)
    if isinstance(logits, torch.distributions.Normal):
        if action_mask is not None:
            raise RuntimeError('Action masks are only supported for discrete policies')
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    sampled = action is None
    sample_action_mask = (
        _action_mask_for_sampled_logits(action_mask, logits)
        if sampled else None
    )
    logits = _apply_action_mask(logits, action_mask)
    if is_discrete:
        logits = logits.unsqueeze(0)
    else: # multi-discrete
        logits = torch.nn.utils.rnn.pad_sequence(
            [l.transpose(0,1) for l in logits],
            batch_first=False,
            padding_value=-torch.inf
        ).permute(1,2,0)

    normalized_logits = logits - logits.logsumexp(dim=-1, keepdim=True)
    probs = logits_to_probs(logits)

    if action is None:
        probs = torch.nan_to_num(probs, 1e-8, 1e-8, 1e-8)
        action = _multinomial(probs.reshape(-1, probs.shape[-1]), 1, replacement=True).int()
        action = action.reshape(probs.shape[:-1])
        if sample_action_mask is not None:
            action = _repair_sampled_actions(action, sample_action_mask)
    else:
        batch = logits[0].shape[0]
        action = action.view(batch, -1).T

    logprob = _log_prob(normalized_logits, action)
    logits_entropy = _entropy(normalized_logits).sum(0)

    if is_discrete:
        return action.T, logprob.squeeze(0), logits_entropy.squeeze(0)

    return action.T, logprob.sum(0), logits_entropy

class _CudaPtr:
    '''Wraps a raw CUDA pointer so torch.as_tensor can consume it via
    __cuda_array_interface__ without any copy or C++ torch dependency.'''
    def __init__(self, ptr, shape, dtype):
        self.__cuda_array_interface__ = {
            'data':    (ptr, False),
            'shape':   shape,
            'typestr': _TORCH_TO_TYPESTR[dtype],
            'version': 2,
        }

_TORCH_TO_CTYPE = {
    torch.uint8:   ctypes.c_uint8,
    torch.float32: ctypes.c_float,
}

def _resolve_device(config):
    '''Resolve --train.device (auto|cpu|cuda|mps) to a torch device string.'''
    requested = str(config.get('device', 'auto')).lower()
    if requested == 'auto':
        return 'cuda' if _C.gpu else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('--train.device mps but PyTorch MPS is not available')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('--train.device cuda but PyTorch CUDA is not available')
    if requested not in ('cpu', 'cuda', 'mps'):
        raise ValueError(
            f'--train.device must be auto, cpu, cuda, or mps (got {requested!r})')
    return requested

def _multinomial(probs, num_samples, replacement=True):
    if probs.device.type != 'mps':
        return torch.multinomial(probs, num_samples, replacement=replacement)
    if replacement:
        # torch.multinomial is not supported on MPS; inverse-CDF sampling
        # keeps prioritized replay on-device.
        cdf = torch.cumsum(probs, dim=-1)
        total = cdf[..., -1:]
        sample_shape = probs.shape[:-1] + (num_samples,)
        u = torch.rand(sample_shape, device=probs.device, dtype=probs.dtype) * total
        return (cdf.unsqueeze(-1) < u.unsqueeze(-2)).sum(dim=-2).long()
    idx = torch.multinomial(probs.cpu(), num_samples, replacement=replacement)
    return idx.to(probs.device)

def _actions_for_vec_step(action):
    if action.dim() == 1:
        action = action.unsqueeze(-1)
    return action.to(dtype=torch.float32).contiguous()

def _cpu_tensor(ptr, shape, dtype):
    '''Zero-copy CPU tensor from a raw pointer via ctypes.'''
    ctype = _TORCH_TO_CTYPE[dtype]
    n = 1
    for s in shape:
        n *= s
    arr = (ctype * n).from_address(ptr)
    return torch.frombuffer(arr, dtype=dtype).reshape(shape)

_METAL_MAX_HEADS = 8

class _MetalPolicyForwardConfig(ctypes.Structure):
    _fields_ = [
        ('batch_size', ctypes.c_uint32),
        ('obs_size', ctypes.c_uint32),
        ('hidden_size', ctypes.c_uint32),
        ('num_layers', ctypes.c_uint32),
        ('action_size', ctypes.c_uint32),
        ('num_atns', ctypes.c_uint32),
        ('act_sizes', ctypes.c_uint32 * _METAL_MAX_HEADS),
        ('is_continuous', ctypes.c_uint32),
    ]

class _MetalPolicyForwardWeights(ctypes.Structure):
    _fields_ = [
        ('encoder_weight', ctypes.POINTER(ctypes.c_float)),
        ('encoder_bias', ctypes.POINTER(ctypes.c_float)),
        ('gru_weights', ctypes.POINTER(ctypes.c_float)),
        ('decoder_weight', ctypes.POINTER(ctypes.c_float)),
        ('decoder_bias', ctypes.POINTER(ctypes.c_float)),
        ('value_weight', ctypes.POINTER(ctypes.c_float)),
        ('value_bias', ctypes.POINTER(ctypes.c_float)),
        ('decoder_logstd', ctypes.POINTER(ctypes.c_float)),
    ]

class _MetalPolicySampleConfig(ctypes.Structure):
    _fields_ = [
        ('rollout_step', ctypes.c_uint32),
        ('rollout_horizon', ctypes.c_uint32),
        ('seed', ctypes.c_uint32),
        ('has_mask', ctypes.c_uint32),
    ]

# The Metal rollout path is shape-parametric over the default PufferLib policy
# (DefaultEncoder -> N-layer MinGRU -> DefaultDecoder + value head). obs/hidden/
# action shapes and the layer count are derived from the env and policy at
# runtime. Discrete, multi-discrete, and continuous (Gaussian) action spaces
# are supported.
class _MetalPolicyRollout:
    MAX_HIDDEN_SIZE = 512

    def __init__(self, pufferl):
        if sys.platform != 'darwin':
            raise RuntimeError('--train.rollout-backend metal requires macOS')
        if pufferl.gpu:
            raise RuntimeError(
                '--train.rollout-backend metal requires the CPU vec backend')

        vec = pufferl._vec
        policy = pufferl.policy

        # Topology guards (shape-agnostic).
        if not isinstance(policy.encoder, pufferlib.models.DefaultEncoder):
            raise RuntimeError('--train.rollout-backend metal requires DefaultEncoder')
        if not isinstance(policy.network, pufferlib.models.MinGRU):
            raise RuntimeError('--train.rollout-backend metal requires MinGRU')
        if not isinstance(policy.decoder, pufferlib.models.DefaultDecoder):
            raise RuntimeError('--train.rollout-backend metal requires DefaultDecoder')
        if policy.network.num_layers < 1:
            raise RuntimeError('--train.rollout-backend metal requires num_layers >= 1')
        act_sizes = tuple(int(a) for a in vec.act_sizes)
        if vec.num_atns != len(act_sizes) or not (1 <= len(act_sizes) <= _METAL_MAX_HEADS):
            raise RuntimeError(
                f'Metal rollout supports 1..{_METAL_MAX_HEADS} action heads/dims')
        if any(a < 1 for a in act_sizes):
            raise RuntimeError('Metal rollout requires positive action sizes')
        if tuple(int(a) for a in policy.decoder.nvec) != act_sizes:
            raise RuntimeError('Metal rollout decoder/action-space mismatch')

        # Derive shapes from the env and policy. For a discrete policy ACTION_SIZE
        # is the total logit count (sum over heads); for a continuous (Gaussian)
        # policy it is the number of action dimensions, with one mean per dim.
        self.IS_CONTINUOUS = bool(getattr(policy.decoder, 'is_continuous', False))
        self.OBS_SIZE = int(vec.obs_size)
        self.HIDDEN_SIZE = int(policy.network.hidden_size)
        self.NUM_LAYERS = int(policy.network.num_layers)
        self.NUM_ATNS = len(act_sizes)
        self.ACT_SIZES = act_sizes if not self.IS_CONTINUOUS else tuple(1 for _ in act_sizes)
        self.ACTION_SIZE = self.NUM_ATNS if self.IS_CONTINUOUS else sum(act_sizes)
        self.action_dtype = np.float32 if self.IS_CONTINUOUS else np.int32
        if not (0 < self.HIDDEN_SIZE <= self.MAX_HIDDEN_SIZE):
            raise RuntimeError(
                f'Metal rollout supports hidden_size in [1, {self.MAX_HIDDEN_SIZE}], '
                f'got {self.HIDDEN_SIZE}')

        self.root = pathlib.Path(__file__).resolve().parents[1]
        self.kernel_path = self.root / 'pufferlib' / 'metal' / 'kernels' / 'policy.metal'
        self.lib_path = self.root / 'build' / 'metal' / 'libpuffer_metal.dylib'
        self._compile_library()
        self.lib = self._load_library()
        act_sizes_arr = (ctypes.c_uint32 * _METAL_MAX_HEADS)(
            *(list(self.ACT_SIZES) + [0] * (_METAL_MAX_HEADS - self.NUM_ATNS)))
        self.config = _MetalPolicyForwardConfig(
            pufferl.total_agents,
            self.OBS_SIZE,
            self.HIDDEN_SIZE,
            self.NUM_LAYERS,
            self.ACTION_SIZE,
            self.NUM_ATNS,
            act_sizes_arr,
            1 if self.IS_CONTINUOUS else 0,
        )
        self.context = self._create_context()
        self.actions = np.empty((pufferl.total_agents, self.NUM_ATNS), dtype=self.action_dtype)

    def close(self):
        context = getattr(self, 'context', None)
        if context:
            self.lib.puffer_metal_policy_destroy(context)
            self.context = None

    def _compile_library(self):
        source = self.root / 'pufferlib' / 'metal' / 'puffer_metal.mm'
        header = self.root / 'pufferlib' / 'metal' / 'puffer_metal.h'
        newest_input = max(
            source.stat().st_mtime,
            header.stat().st_mtime,
            self.kernel_path.stat().st_mtime,
        )
        if self.lib_path.exists() and self.lib_path.stat().st_mtime >= newest_input:
            return

        self.lib_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            'clang++', '-std=c++17', '-ObjC++', '-fobjc-arc', '-O2', '-dynamiclib',
            str(source),
            '-framework', 'Foundation',
            '-framework', 'Metal',
            '-framework', 'MetalPerformanceShaders',
            '-o', str(self.lib_path),
        ]
        subprocess.run(cmd, cwd=self.root, check=True)

    def _load_library(self):
        lib = ctypes.CDLL(str(self.lib_path))
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
            ctypes.POINTER(_MetalPolicyForwardConfig),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(_MetalPolicyForwardWeights),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.puffer_metal_policy_load_resident.restype = ctypes.c_int
        lib.puffer_metal_policy_write_observations_resident.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MetalPolicyForwardConfig),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.puffer_metal_policy_write_observations_resident.restype = ctypes.c_int
        lib.puffer_metal_policy_forward_sample_rollout_resident.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MetalPolicyForwardConfig),
            ctypes.POINTER(_MetalPolicySampleConfig),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.puffer_metal_policy_forward_sample_rollout_resident.restype = ctypes.c_int
        lib.puffer_metal_policy_read_actions_resident.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MetalPolicyForwardConfig),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.puffer_metal_policy_read_actions_resident.restype = ctypes.c_int
        lib.puffer_metal_policy_read_rollout_resident.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MetalPolicyForwardConfig),
            ctypes.POINTER(_MetalPolicySampleConfig),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.puffer_metal_policy_read_rollout_resident.restype = ctypes.c_int
        return lib

    def _raise_on_error(self, rc, error):
        if rc != 0:
            message = error.value.decode('utf-8', 'replace')
            raise RuntimeError(message)

    def _create_context(self):
        error = ctypes.create_string_buffer(4096)
        context = ctypes.c_void_p()
        rc = self.lib.puffer_metal_policy_create(
            str(self.kernel_path).encode('utf-8'),
            ctypes.byref(context),
            error,
            len(error),
        )
        self._raise_on_error(rc, error)
        return context

    def _array(self, tensor):
        return np.ascontiguousarray(tensor.detach().cpu().numpy().astype(np.float32, copy=False))

    def _weights(self, policy):
        state = policy.state_dict()
        # Stack all MinGRU layer weights into one contiguous [num_layers,
        # 3*hidden, hidden] block (concatenated along axis 0) matching the
        # gru_weights ABI the resident path reads with a per-layer offset.
        gru_layers = [self._array(state[f'network.layers.{i}.weight'])
                      for i in range(self.NUM_LAYERS)]
        gru_weights = np.ascontiguousarray(np.concatenate(gru_layers, axis=0))
        if self.IS_CONTINUOUS:
            # Continuous: the decoder is a mean head plus a learned logstd vector.
            decoder_weight = self._array(state['decoder.decoder_mean.weight'])
            decoder_bias = self._array(state['decoder.decoder_mean.bias'])
            decoder_logstd = self._array(state['decoder.decoder_logstd']).reshape(-1)
        else:
            decoder_weight = self._array(state['decoder.decoder.weight'])
            decoder_bias = self._array(state['decoder.decoder.bias'])
            decoder_logstd = np.zeros((1,), dtype=np.float32)  # unused by the discrete sampler
        arrays = {
            'encoder_weight': self._array(state['encoder.encoder.weight']),
            'encoder_bias': self._array(state['encoder.encoder.bias']),
            'gru_weights': gru_weights,
            'decoder_weight': decoder_weight,
            'decoder_bias': decoder_bias,
            'value_weight': self._array(state['decoder.value_function.weight']).reshape(-1),
            'value_bias': self._array(state['decoder.value_function.bias']),
            'decoder_logstd': decoder_logstd,
        }
        ptr = ctypes.POINTER(ctypes.c_float)
        weights = _MetalPolicyForwardWeights(
            *(arrays[name].ctypes.data_as(ptr) for name, _ in _MetalPolicyForwardWeights._fields_)
        )
        return arrays, weights

    def _obs_array(self, obs):
        return np.ascontiguousarray(torch.as_tensor(obs).numpy().astype(np.float32, copy=False))

    def load(self, policy, observations, state):
        arrays, weights = self._weights(policy)
        obs = self._obs_array(observations)
        state_arr = self._array(state[0])
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.puffer_metal_policy_load_resident(
            self.context,
            ctypes.byref(self.config),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            state_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(weights),
            error,
            len(error),
        )
        self._raise_on_error(rc, error)
        return arrays

    def write_observations(self, observations):
        obs = self._obs_array(observations)
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.puffer_metal_policy_write_observations_resident(
            self.context,
            ctypes.byref(self.config),
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error,
            len(error),
        )
        self._raise_on_error(rc, error)

    def launch_forward_sample_step(self, rollout_step, rollout_horizon, seed, action_mask=None):
        sample_config = _MetalPolicySampleConfig(
            rollout_step,
            rollout_horizon,
            seed,
            1 if action_mask is not None else 0,
        )
        mask_ptr = None
        if action_mask is not None:
            mask_ptr = ctypes.cast(
                int(action_mask.data_ptr()), ctypes.POINTER(ctypes.c_uint8))
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.puffer_metal_policy_forward_sample_rollout_resident(
            self.context,
            ctypes.byref(self.config),
            ctypes.byref(sample_config),
            mask_ptr,
            error,
            len(error),
        )
        self._raise_on_error(rc, error)

    def read_actions(self):
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.puffer_metal_policy_read_actions_resident(
            self.context,
            ctypes.byref(self.config),
            self.actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            error,
            len(error),
        )
        self._raise_on_error(rc, error)
        return self.actions

    def forward_sample_step(self, rollout_step, rollout_horizon, seed,
            action_mask=None):
        self.launch_forward_sample_step(
            rollout_step, rollout_horizon, seed, action_mask)
        return self.read_actions()

    def read_rollout(self, rollout_horizon, seed):
        sample_config = _MetalPolicySampleConfig(
            0,
            rollout_horizon,
            seed,
            0,
        )
        shape = (rollout_horizon, self.config.batch_size)
        actions = np.empty((rollout_horizon, self.config.batch_size, self.NUM_ATNS), dtype=self.action_dtype)
        logprobs = np.empty(shape, dtype=np.float32)
        values = np.empty(shape, dtype=np.float32)
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.puffer_metal_policy_read_rollout_resident(
            self.context,
            ctypes.byref(self.config),
            ctypes.byref(sample_config),
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            logprobs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            error,
            len(error),
        )
        self._raise_on_error(rc, error)
        return actions, logprobs, values

class PuffeRL:
    def __init__(self, args, vec, policy, verbose=True):
        config = args['train']
        rollout_backend = str(config.get('rollout_backend', 'torch')).lower()
        if rollout_backend not in ('torch', 'metal'):
            raise ValueError(
                f'--train.rollout-backend must be torch or metal (got {rollout_backend!r})')
        use_metal = rollout_backend == 'metal'
        self.train_device = torch.device(_resolve_device(config))
        # The Metal rollout is scheduled from Python over CPU-side env buffers,
        # so rollout storage stays on CPU; training tensors use train_device.
        device = 'cpu' if use_metal else str(self.train_device)
        self.device = device

        torch.set_float32_matmul_precision('high')
        if _C.gpu:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        self._vec = vec
        self.gpu = vec.gpu
        total_agents = vec.total_agents
        self.total_agents = total_agents
        obs_dtype = _OBS_DTYPE_MAP.get(vec.obs_dtype, torch.uint8)
        if vec.action_mask_size:
            policy_act_sizes = tuple(
                int(a) for a in getattr(policy.decoder, 'nvec', vec.act_sizes))
            policy_action_size = sum(policy_act_sizes)
            if int(vec.action_mask_size) != policy_action_size:
                raise RuntimeError(
                    'Action mask size mismatch: '
                    f'vec.action_mask_size={int(vec.action_mask_size)} but '
                    f'sum(act_sizes)={policy_action_size}. Masked envs must '
                    'define MY_ACTION_MASK equal to the total policy logit count.'
                )

        if self.gpu:
            self.vec_obs = torch.as_tensor(_CudaPtr(vec.gpu_obs_ptr,
                (total_agents, vec.obs_size), obs_dtype))
            self.vec_rewards = torch.as_tensor(_CudaPtr(vec.gpu_rewards_ptr,
                (total_agents,), torch.float32))
            self.vec_terminals = torch.as_tensor(_CudaPtr(vec.gpu_terminals_ptr,
                (total_agents,), torch.float32))
            self.vec_action_mask = (
                torch.as_tensor(_CudaPtr(vec.gpu_action_mask_ptr,
                    (total_agents, vec.action_mask_size), torch.uint8))
                if vec.action_mask_size else None
            )
        else:
            self.vec_obs = _cpu_tensor(vec.obs_ptr,
                (total_agents, vec.obs_size), obs_dtype)
            self.vec_rewards = _cpu_tensor(vec.rewards_ptr,
                (total_agents,), torch.float32)
            self.vec_terminals = _cpu_tensor(vec.terminals_ptr,
                (total_agents,), torch.float32)
            self.vec_action_mask = (
                _cpu_tensor(vec.action_mask_ptr,
                    (total_agents, vec.action_mask_size), torch.uint8)
                if vec.action_mask_size else None
            )

        vec.reset()
        horizon = config['horizon']
        num_atns = vec.num_atns

        self.observations = torch.zeros(horizon, total_agents, vec.obs_size,
            dtype=obs_dtype, device=device)
        self.actions = torch.zeros(horizon, total_agents, num_atns, device=device)
        self.values = torch.zeros(horizon, total_agents, device=device)
        self.logprobs = torch.zeros(horizon, total_agents, device=device)
        self.rewards = torch.zeros(horizon, total_agents, device=device)
        self.terminals = torch.zeros(horizon, total_agents, device=device)
        self.action_masks = (
            torch.zeros(horizon, total_agents, vec.action_mask_size,
                dtype=torch.bool, device=device)
            if vec.action_mask_size else None
        )
        self.ratio = torch.ones(total_agents, horizon, device=device)
        self.state = policy.initial_state(total_agents, device=device)

        self.batch_size = total_agents * horizon
        self.minibatch_segments = config['minibatch_size'] // horizon
        self.total_epochs = max(1, config['total_timesteps'] // self.batch_size)

        self.args = args
        self.config = config
        self.world_size = args['world_size']

        self.policy = policy
        self._metal_rollout = None
        if use_metal:
            self._metal_rollout = _MetalPolicyRollout(self)
            self.args['torch_device'] = f'METAL+{self.train_device.type.upper()}'
        else:
            self.args['torch_device'] = str(device).upper()
        self.optimizer = Muon(
            self.policy.parameters(),
            lr=config['learning_rate'],
            momentum=config['beta1'],
            eps=config['eps'],
        )

        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.profile = Profile(gpu=self.gpu, device=self.train_device)
        self.verbose = verbose

        self.model_size = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        if verbose:
            pufferlib.pufferl.print_dashboard(args, self.model_size, {}, clear=True)


    @property
    def uptime(self):
        return time.time() - self.start_time

    @property
    def sps(self):
        if self.global_step == self.last_log_step:
            return 0

        return (self.global_step - self.last_log_step) / (time.time() - self.last_log_time)

    def num_params(self):
        return self.model_size

    def rollouts(self):
        if self._metal_rollout is not None:
            return self._metal_rollouts()

        prof = self.profile
        config = self.config
        device = self.device
        horizon = config['horizon']

        self.state = tuple(torch.zeros_like(s) for s in self.state) if self.state else ()
        o = self.vec_obs
        r = torch.zeros(self.total_agents, device=device)
        d = torch.zeros(self.total_agents, device=device)
        m = self.vec_action_mask

        P = Profile
        prof.mark(0)
        for t in range(horizon):
            o_device = torch.as_tensor(o, device=device)
            m_device = (
                torch.as_tensor(m, device=device).to(dtype=torch.bool)
                if m is not None else None
            )

            prof.mark(1)
            with torch.no_grad():
                logits, value, state = self.policy.forward_eval(o_device, self.state)
                action, logprob, _ = sample_logits(logits, action_mask=m_device)
            prof.mark(2)

            with torch.no_grad():
                self.state = state
                self.observations[t] = o_device
                if self.action_masks is not None:
                    self.action_masks[t] = m_device
                self.actions[t] = action
                self.logprobs[t] = logprob
                self.rewards[t] = torch.as_tensor(r, device=device)
                self.terminals[t] = torch.as_tensor(d, device=device).float()
                self.values[t] = value.flatten()

            prof.mark(2)
            actions_flat = _actions_for_vec_step(action)
            if self.gpu:
                actions_flat = actions_flat.to('cuda')
                self._vec.gpu_step(actions_flat.data_ptr())
                torch.cuda.synchronize()
            else:
                actions_flat = actions_flat.cpu()
                self._vec.cpu_step(actions_flat.data_ptr())

            o, r, d, m = self.vec_obs, self.vec_rewards, self.vec_terminals, self.vec_action_mask
            prof.mark(3)
            prof.elapsed(P.EVAL_GPU, 1, 2)
            prof.elapsed(P.EVAL_ENV, 2, 3)

        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)
        self.global_step += self.total_agents * horizon
        self.env_logs = self._vec.log()

    def _metal_rollouts(self):
        prof = self.profile
        config = self.config
        device = self.device
        horizon = config['horizon']

        self.state = tuple(torch.zeros_like(s) for s in self.state) if self.state else ()
        o = self.vec_obs
        r = torch.zeros(self.total_agents, device=device)
        d = torch.zeros(self.total_agents, device=device)
        m = self.vec_action_mask
        rollout_seed = (
            int(self.args.get('seed', config.get('seed', 0))) +
            int(self.epoch) * 0x9e3779b9
        ) & 0xffffffff

        P = Profile
        prof.mark(0)
        self._metal_rollout.load(self.policy, o, self.state)
        for t in range(horizon):
            if self.action_masks is not None:
                self.action_masks[t] = torch.as_tensor(m, device=device).to(
                    dtype=torch.bool)

            prof.mark(1)
            self._metal_rollout.launch_forward_sample_step(
                t, horizon, rollout_seed, m)
            actions = self._metal_rollout.read_actions()
            prof.mark(2)

            with torch.no_grad():
                self.observations[t] = torch.as_tensor(o, device=device)
                self.rewards[t] = torch.as_tensor(r, device=device)
                self.terminals[t] = torch.as_tensor(d, device=device).float()

            actions_flat = torch.from_numpy(
                actions.astype(np.float32, copy=True).reshape(
                    self.total_agents, self._metal_rollout.NUM_ATNS)
            ).contiguous()
            self._vec.cpu_step(actions_flat.data_ptr())

            o, r, d, m = self.vec_obs, self.vec_rewards, self.vec_terminals, self.vec_action_mask
            if t + 1 < horizon:
                self._metal_rollout.write_observations(o)
            prof.mark(3)
            prof.elapsed(P.EVAL_GPU, 1, 2)
            prof.elapsed(P.EVAL_ENV, 2, 3)

        rollout_actions, rollout_logprobs, rollout_values = self._metal_rollout.read_rollout(
            horizon,
            rollout_seed,
        )
        with torch.no_grad():
            self.actions.copy_(torch.from_numpy(
                rollout_actions.astype(np.float32, copy=False)))
            self.logprobs.copy_(torch.from_numpy(rollout_logprobs))
            self.values.copy_(torch.from_numpy(rollout_values))

        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)
        self.global_step += self.total_agents * horizon
        self.env_logs = self._vec.log()

    def train(self):
        prof = self.profile
        losses = defaultdict(float)
        config = self.config
        device = self.train_device

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        self.ratio[:] = 1
        # The Metal hybrid trains on MPS but keeps rollout storage, advantage
        # computation, and importance ratios on CPU, transferring minibatch
        # inputs to the device in a few coalesced copies.
        use_metal_mps_train = (
            device.type == 'mps' and self._metal_rollout is not None)

        learning_rate = config['learning_rate']
        if config['anneal_lr'] and self.epoch > 0:
            lr_ratio = self.epoch / self.total_epochs
            lr_min = config['learning_rate'] * config['min_lr_ratio']
            learning_rate = lr_min + 0.5*(learning_rate - lr_min) * (1 + np.cos(np.pi * lr_ratio))
            self.optimizer.param_groups[0]['lr'] = learning_rate

        # Transpose from [horizon, agents] (contiguous writes) to [agents, horizon] (minibatch indexing)
        obs_cpu = self.observations.transpose(0, 1).contiguous()
        act_cpu = self.actions.transpose(0, 1).contiguous()
        val_cpu = self.values.T.contiguous()
        lp_cpu = self.logprobs.T.contiguous()
        rew_cpu = self.rewards.T.contiguous().clamp(-1, 1)
        ter_cpu = self.terminals.T.contiguous()
        mask_cpu = (
            self.action_masks.transpose(0, 1).contiguous()
            if self.action_masks is not None else None
        )

        obs = obs_cpu.to(device)
        masks = mask_cpu.to(device) if mask_cpu is not None else None
        if use_metal_mps_train:
            if act_cpu.shape[-1] == 1:
                # Single action head: coalesce act/val/lp into one transfer.
                train_float = torch.stack(
                    (act_cpu.squeeze(-1), val_cpu, lp_cpu), dim=0
                ).to(device)
                act = train_float[0].unsqueeze(-1)
                val = train_float[1]
                lp = train_float[2]
            else:
                # Multi-discrete: act carries a head dim, so transfer it on its
                # own and coalesce only the scalar val/lp tensors.
                val_lp = torch.stack((val_cpu, lp_cpu), dim=0).to(device)
                act = act_cpu.to(device)
                val = val_lp[0]
                lp = val_lp[1]
            advantages_cpu = torch.zeros_like(val_cpu)
            rew = None
            ter = None
        else:
            act = act_cpu.to(device)
            val = val_cpu.to(device)
            lp = lp_cpu.to(device)
            rew = rew_cpu.to(device)
            ter = ter_cpu.to(device)
            advantages_cpu = None

        P = Profile
        prof.mark(0)
        num_minibatches = int(config['replay_ratio'] * self.batch_size / config['minibatch_size'])
        for mb in range(num_minibatches):
            shape = val.shape
            if use_metal_mps_train:
                # _C's advantage kernels are CPU/CUDA only; computing on the
                # CPU copy and transferring the result beats a torch loop over
                # horizon on MPS.
                advantages_cpu.zero_()
                compute_puff_advantage(val_cpu, rew_cpu,
                    ter_cpu, self.ratio, advantages_cpu, config['gamma'],
                    config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
                advantages = advantages_cpu.to(device)
            else:
                advantages = torch.zeros(shape, device=device)
                advantages = compute_puff_advantage(val, rew,
                    ter, self.ratio, advantages, config['gamma'],
                    config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])

            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6)/(prio_weights.sum() + 1e-6)
            idx = _multinomial(prio_probs, self.minibatch_segments, replacement=True)
            idx_cpu = (
                idx.cpu()
                if use_metal_mps_train and mb + 1 < num_minibatches
                else None
            )
            mb_prio = (self.total_agents*prio_probs[idx, None])**-anneal_beta

            mb_obs = obs[idx]
            mb_actions = act[idx]
            mb_logprobs = lp[idx]
            mb_masks = masks[idx] if masks is not None else None
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]

            prof.mark(1)
            logits, newvalue = self.policy(mb_obs)
            actions, newlogprob, entropy = sample_logits(
                logits, action=mb_actions, action_mask=mb_masks)
            prof.mark(2)
            prof.elapsed(P.TRAIN_FORWARD, 1, 2)

            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            if use_metal_mps_train:
                if idx_cpu is not None:
                    self.ratio[idx_cpu] = ratio.detach().cpu()
            else:
                self.ratio[idx] = ratio.detach()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > config['clip_coef']).float().mean()

            adv = mb_advantages
            adv = mb_prio * (adv - adv.mean()) / (adv.std() + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5*torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss + config['vf_coef']*v_loss - config['ent_coef']*entropy_loss
            val[idx] = newvalue.detach().float()
            if use_metal_mps_train and idx_cpu is not None:
                val_cpu[idx_cpu] = newvalue.detach().float().cpu()

            losses['policy_loss'] += pg_loss
            losses['value_loss'] += v_loss
            losses['entropy'] += entropy_loss
            losses['old_approx_kl'] += old_approx_kl
            losses['approx_kl'] += approx_kl
            losses['clipfrac'] += clipfrac
            losses['importance'] += ratio.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config['max_grad_norm'])
            self.optimizer.step()
            self.optimizer.zero_grad()

        prof.mark(1)
        prof.elapsed(P.TRAIN, 0, 1)

        losses = {k: v.item() / num_minibatches for k, v in losses.items()}
        y_pred = val.flatten()
        y_true = advantages.flatten() + val.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var

        self.losses = losses
        self.epoch += 1

    def log(self):
        P = Profile
        perf = self.profile.read_and_reset()
        logs = {
            'SPS': self.sps * self.world_size,
            'agent_steps': self.global_step * self.world_size,
            'uptime': time.time() - self.start_time,
            'epoch': self.epoch,
            'env': dict(getattr(self, 'env_logs', {})),
            'loss': dict(getattr(self, 'losses', {})),
            'perf': {
                'rollout': perf[P.ROLLOUT],
                'eval_gpu': perf[P.EVAL_GPU],
                'eval_env': perf[P.EVAL_ENV],
                'train': perf[P.TRAIN],
                'train_misc': perf[P.TRAIN_MISC],
                'train_forward': perf[P.TRAIN_FORWARD],
            },
            'util': (
                dict(_C.get_utilization(self.args.get('gpu_id', 0)))
                if self.gpu else
                {'cpu_mem_gb': pufferlib.pufferl.current_rss_gb()}
            ),
        }
        self.last_log_time = time.time()
        self.last_log_step = self.global_step
        return logs

    eval_log = log

    def save_weights(self, path):
        torch.save(self.policy.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path, map_location=self.device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.policy.load_state_dict(state_dict)

    def render(self, env_id=0):
        self._vec.render(env_id)

    def close(self):
        if self._metal_rollout is not None:
            self._metal_rollout.close()
            self._metal_rollout = None
        self.vec_obs = None
        self.vec_rewards = None
        self.vec_terminals = None
        self._vec.close()

    @classmethod
    def create_pufferl(cls, args):
        '''Matches _C.create_pufferl(args) interface.'''
        # DDP setup
        if _C.gpu and 'LOCAL_RANK' in os.environ:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

        args['vec']['num_buffers'] = 1
        vec = _C.create_vec(args, _C.gpu)
        policy = load_policy(args, vec)

        if _C.gpu and 'LOCAL_RANK' in os.environ:
            torch.distributed.init_process_group(backend='nccl', world_size=world_size)
            policy = policy.to(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(
                policy, device_ids=[local_rank], output_device=local_rank)
            if hasattr(policy, 'lstm'):
                model.hidden_size = policy.hidden_size
            model.forward_eval = policy.forward_eval
            model.initial_state = policy.initial_state
            policy = model.to(local_rank)

        return cls(args, vec, policy)

def _compute_puff_advantage_torch(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    num_steps, horizon = values.shape
    last = torch.zeros(num_steps, device=values.device, dtype=values.dtype)
    for t in range(horizon - 2, -1, -1):
        nextnonterminal = 1.0 - terminals[:, t + 1]
        importance = ratio[:, t]
        rho_t = torch.clamp(importance, max=vtrace_rho_clip)
        c_t = torch.clamp(importance, max=vtrace_c_clip)
        delta = rho_t * rewards[:, t + 1] + gamma * values[:, t + 1] * nextnonterminal - values[:, t]
        last = delta + gamma * gae_lambda * c_t * last * nextnonterminal
        advantages[:, t] = last
    return advantages

def compute_puff_advantage(values, rewards, terminals,
        ratio, advantages, gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip):
    if values.device.type == 'mps':
        return _compute_puff_advantage_torch(values, rewards,
            terminals, ratio, advantages, gamma, gae_lambda,
            vtrace_rho_clip, vtrace_c_clip)

    num_steps, horizon = values.shape
    fn = _C.puff_advantage if values.is_cuda else _C.puff_advantage_cpu
    fn(
        values.data_ptr(), rewards.data_ptr(), terminals.data_ptr(),
        ratio.data_ptr(), advantages.data_ptr(),
        num_steps, horizon,
        gamma, gae_lambda, vtrace_rho_clip, vtrace_c_clip)
    return advantages

class Profile:
    '''Matches pufferlib.cu profiling: accumulate ms, report seconds.'''
    ROLLOUT, EVAL_GPU, EVAL_ENV, TRAIN, TRAIN_MISC, TRAIN_FORWARD, NUM = range(7)

    def __init__(self, gpu=True, device='cpu'):
        self.accum = [0.0] * Profile.NUM
        self.gpu = gpu
        # MPS executes asynchronously; sync at marks so timings are honest.
        self.sync_mps = torch.device(device).type == 'mps'
        if gpu:
            self._events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        else:
            self._stamps = [0.0] * 4

    def mark(self, idx):
        if self.gpu:
            self._events[idx].record()
        else:
            if self.sync_mps:
                torch.mps.synchronize()
            self._stamps[idx] = time.perf_counter()

    def elapsed(self, idx, start_ev, end_ev):
        if self.gpu:
            self._events[end_ev].synchronize()
            elapsed_ms = self._events[start_ev].elapsed_time(self._events[end_ev])
        else:
            elapsed_ms = (self._stamps[end_ev] - self._stamps[start_ev]) * 1000.0
        self.accum[idx] += elapsed_ms
        return elapsed_ms / 1000.0

    def read_and_reset(self):
        out = [v / 1000.0 for v in self.accum]
        self.accum = [0.0] * Profile.NUM
        return out

def load_policy(args, vec):
    import pufferlib.models
    # Sweep-populated configs store float means for integer hyperparameters
    # (e.g. num_layers = 2.11327). The Protein sweep rounds integer params before
    # use (pufferlib/sweep.py: is_integer -> round); mirror that here on a local
    # copy so direct/.ini runs build a valid network instead of crashing on
    # range(float) / nn.Linear(float). The original args are left untouched so the
    # sampled float is still preserved for logging.
    policy_kwargs = dict(args['policy'])
    for key in ('num_layers', 'hidden_size'):
        if isinstance(policy_kwargs.get(key), float):
            policy_kwargs[key] = round(policy_kwargs[key])
    network_cls = getattr(pufferlib.models, args['torch']['network'])
    encoder_cls = getattr(pufferlib.models, args['torch']['encoder'])
    decoder_cls = getattr(pufferlib.models, args['torch']['decoder'])

    network = network_cls(**policy_kwargs)
    encoder = encoder_cls(vec.obs_size, policy_kwargs['hidden_size'])
    decoder = decoder_cls(vec.act_sizes, policy_kwargs['hidden_size'])
    policy = pufferlib.models.Policy(encoder, decoder, network)

    device = _resolve_device(args['train'])
    policy = policy.to(device)

    load_id = args['load_id']
    if load_id is not None:
        if args['wandb']:
            import wandb
            artifact = wandb.use_artifact(f'{load_id}:latest')
            data_dir = artifact.download()
            path = f'{data_dir}/{max(os.listdir(data_dir))}'
        else:
            raise ValueError('load_id requires --wandb')

        state_dict = torch.load(path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    load_path = args['load_model_path']
    if load_path == 'latest':
        pattern = os.path.join(args['checkpoint_dir'], args['env_name'], '**', '*.bin')
        candidates = glob.glob(pattern, recursive=True)
        load_path = max(candidates, key=os.path.getctime)

    if load_path is not None:
        state_dict = torch.load(load_path, map_location=device)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        policy.load_state_dict(state_dict)

    return policy
