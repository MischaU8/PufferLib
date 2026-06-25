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

def sample_logits(logits, action=None):
    is_discrete = isinstance(logits, torch.Tensor)
    if isinstance(logits, torch.distributions.Normal):
        batch = logits.loc.shape[0]
        if action is None:
            action = logits.sample().view(batch, -1)
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().view(batch, -1).sum(1)
        return action, log_probs, logits_entropy
    elif is_discrete:
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

def _select_device():
    override = os.environ.get('PUFFER_TORCH_DEVICE')
    if override:
        override = override.lower()
        if override == 'auto':
            if _C.gpu:
                return 'cuda'
            return 'mps' if torch.backends.mps.is_available() else 'cpu'
        if override == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('PUFFER_TORCH_DEVICE=mps but PyTorch MPS is not available')
        if override == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('PUFFER_TORCH_DEVICE=cuda but PyTorch CUDA is not available')
        return override
    if _C.gpu:
        return 'cuda'
    return 'cpu'

def _select_metal_train_device(default_device):
    override = os.environ.get('PUFFER_METAL_TRAIN_DEVICE')
    if not override:
        return torch.device(default_device)

    override = override.lower()
    if override == 'cpu':
        return torch.device('cpu')
    if override == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('PUFFER_METAL_TRAIN_DEVICE=mps but PyTorch MPS is not available')
        return torch.device('mps')
    raise RuntimeError(
        'PUFFER_METAL_TRAIN_DEVICE must be unset, cpu, or mps '
        f'(got {override!r})')

def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ('0', 'false', 'no', 'off')

def _multinomial(probs, num_samples, replacement=True):
    use_mps_device_sampler = (
        probs.device.type == 'mps' and
        replacement and
        os.environ.get('PUFFER_METAL_TRAIN_DEVICE', '').lower() == 'mps'
    )
    if use_mps_device_sampler:
        cdf = torch.cumsum(probs, dim=-1)
        total = cdf[..., -1:]
        sample_shape = probs.shape[:-1] + (num_samples,)
        u = torch.rand(sample_shape, device=probs.device, dtype=probs.dtype) * total
        return (cdf.unsqueeze(-1) < u.unsqueeze(-2)).sum(dim=-2).long()
    if probs.device.type == 'mps':
        idx = torch.multinomial(probs.cpu(), num_samples, replacement=replacement)
        return idx.to(probs.device)
    return torch.multinomial(probs, num_samples, replacement=replacement)

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
        ('reserved', ctypes.c_uint32),
    ]

class _MetalPolicyNativeConfig(ctypes.Structure):
    _fields_ = [
        ('static_vec_ptr', ctypes.c_uint64),
        ('total_agents', ctypes.c_uint32),
        ('horizon', ctypes.c_uint32),
        ('seed', ctypes.c_uint32),
        ('reserved', ctypes.c_uint32),
    ]

class _MetalPolicyNativeProfile(ctypes.Structure):
    _fields_ = [
        ('policy_sample_seconds', ctypes.c_double),
        ('action_read_seconds', ctypes.c_double),
        ('cpu_step_seconds', ctypes.c_double),
        ('obs_upload_seconds', ctypes.c_double),
    ]

# The Metal rollout path is shape-parametric over the default PufferLib policy
# (DefaultEncoder -> 2-layer MinGRU -> DefaultDecoder + value head) with a single
# discrete action head. obs/hidden/action are derived from the env and policy at
# runtime; only num_layers is fixed at 2 because the weight ABI passes two GRU
# buffers. The native (CPU-env scheduler) mode remains Breakout-specific.
class _MetalPolicyRollout:
    MAX_HIDDEN_SIZE = 512

    def __init__(self, pufferl):
        if sys.platform != 'darwin':
            raise RuntimeError('PUFFER_METAL_ROLLOUT requires macOS')
        if pufferl.gpu:
            raise RuntimeError('PUFFER_METAL_ROLLOUT currently requires the CPU vec backend')
        if torch.device(pufferl.device).type != 'cpu':
            raise RuntimeError('PUFFER_METAL_ROLLOUT currently requires PUFFER_TORCH_DEVICE=cpu')

        args = pufferl.args
        vec = pufferl._vec
        policy = pufferl.policy
        mode = os.environ.get('PUFFER_METAL_ROLLOUT', 'python').lower()
        self.native = mode in ('native', 'scheduler')

        # Topology guards (shape-agnostic).
        if not isinstance(policy.encoder, pufferlib.models.DefaultEncoder):
            raise RuntimeError('PUFFER_METAL_ROLLOUT requires DefaultEncoder')
        if not isinstance(policy.network, pufferlib.models.MinGRU):
            raise RuntimeError('PUFFER_METAL_ROLLOUT requires MinGRU')
        if not isinstance(policy.decoder, pufferlib.models.DefaultDecoder):
            raise RuntimeError('PUFFER_METAL_ROLLOUT requires DefaultDecoder')
        if policy.network.num_layers < 1:
            raise RuntimeError('PUFFER_METAL_ROLLOUT requires num_layers >= 1')
        act_sizes = tuple(int(a) for a in vec.act_sizes)
        if vec.num_atns != len(act_sizes) or not (1 <= len(act_sizes) <= _METAL_MAX_HEADS):
            raise RuntimeError(
                f'PUFFER_METAL_ROLLOUT supports 1..{_METAL_MAX_HEADS} action heads/dims')
        if any(a < 1 for a in act_sizes):
            raise RuntimeError('PUFFER_METAL_ROLLOUT requires positive action sizes')
        if tuple(int(a) for a in policy.decoder.nvec) != act_sizes:
            raise RuntimeError('PUFFER_METAL_ROLLOUT decoder/action-space mismatch')

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
                f'PUFFER_METAL_ROLLOUT supports hidden_size in [1, {self.MAX_HIDDEN_SIZE}], '
                f'got {self.HIDDEN_SIZE}')

        # The native scheduler steps the Breakout CPU env directly, so it stays
        # Breakout-specific (single discrete head); the production python mode is
        # env-agnostic.
        if self.native and (args.get('env_name') != 'breakout' or self.NUM_ATNS != 1):
            raise RuntimeError(
                'PUFFER_METAL_ROLLOUT=native only supports breakout; use '
                'PUFFER_METAL_ROLLOUT=python for other environments')

        self.root = pathlib.Path(__file__).resolve().parents[1]
        self.kernel_path = self.root / 'pufferlib' / 'metal' / 'kernels' / 'policy.metal'
        self.lib_path = self.root / 'build' / 'metal' / 'libpuffer_metal.dylib'
        self._compile_library()
        self.lib = self._load_library()
        self.context = None
        self.native_context = None
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
        if self.native:
            self.base_seed = int(
                pufferl.args.get('seed', pufferl.config.get('seed', 0))
            ) & 0xffffffff
            self.native_config = _MetalPolicyNativeConfig(
                int(pufferl._vec.vec_ptr),
                pufferl.total_agents,
                pufferl.config['horizon'],
                self.base_seed,
                0,
            )
            self.native_context = self._create_native_context()
        else:
            self.context = self._create_context()
            self.actions = np.empty((pufferl.total_agents, self.NUM_ATNS), dtype=self.action_dtype)

    def close(self):
        context = getattr(self, 'context', None)
        if context:
            self.lib.puffer_metal_policy_destroy(context)
            self.context = None
        native_context = getattr(self, 'native_context', None)
        if native_context:
            self.lib.puffer_metal_policy_native_destroy(native_context)
            self.native_context = None

    def _compile_library(self):
        source = self.root / 'pufferlib' / 'metal' / 'puffer_metal.mm'
        header = self.root / 'pufferlib' / 'metal' / 'puffer_metal.h'
        newest_input = max(
            source.stat().st_mtime,
            header.stat().st_mtime,
            (self.root / 'build.sh').stat().st_mtime,
            (self.root / 'ocean' / 'breakout' / 'breakout.h').stat().st_mtime,
            self.kernel_path.stat().st_mtime,
        )
        if self.lib_path.exists() and self.lib_path.stat().st_mtime >= newest_input:
            return

        cmd = ['bash', 'build.sh', 'breakout', '--metal-native']
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
        lib.puffer_metal_policy_native_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(_MetalPolicyNativeConfig),
            ctx_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.puffer_metal_policy_native_create.restype = ctypes.c_int
        lib.puffer_metal_policy_native_destroy.argtypes = [ctypes.c_void_p]
        lib.puffer_metal_policy_native_destroy.restype = None
        lib.puffer_metal_policy_native_rollouts.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_MetalPolicyForwardWeights),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint32,
            ctypes.POINTER(_MetalPolicyNativeProfile),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.puffer_metal_policy_native_rollouts.restype = ctypes.c_int
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

    def _create_native_context(self):
        error = ctypes.create_string_buffer(4096)
        context = ctypes.c_void_p()
        rc = self.lib.puffer_metal_policy_native_create(
            str(self.kernel_path).encode('utf-8'),
            ctypes.byref(self.native_config),
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

    def launch_forward_sample_step(self, rollout_step, rollout_horizon, seed):
        sample_config = _MetalPolicySampleConfig(
            rollout_step,
            rollout_horizon,
            seed,
            0,
        )
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.puffer_metal_policy_forward_sample_rollout_resident(
            self.context,
            ctypes.byref(self.config),
            ctypes.byref(sample_config),
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

    def forward_sample_step(self, rollout_step, rollout_horizon, seed):
        self.launch_forward_sample_step(rollout_step, rollout_horizon, seed)
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

    def native_rollouts(self, policy, state, epoch):
        arrays, weights = self._weights(policy)
        state_arr = self._array(state[0])
        horizon = self.native_config.horizon
        batch = self.native_config.total_agents
        self.native_config.seed = (
            int(self.base_seed) + int(epoch) * 0x9e3779b9
        ) & 0xffffffff

        observations = np.empty((horizon, batch, self.OBS_SIZE), dtype=np.float32)
        rewards = np.empty((horizon, batch), dtype=np.float32)
        terminals = np.empty((horizon, batch), dtype=np.float32)
        actions = np.empty((horizon, batch), dtype=np.float32)
        logprobs = np.empty((horizon, batch), dtype=np.float32)
        values = np.empty((horizon, batch), dtype=np.float32)
        native_profile = _MetalPolicyNativeProfile()
        error = ctypes.create_string_buffer(4096)
        rc = self.lib.puffer_metal_policy_native_rollouts(
            self.native_context,
            ctypes.byref(weights),
            state_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rewards.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            terminals.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            logprobs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self.native_config.seed,
            ctypes.byref(native_profile),
            error,
            len(error),
        )
        self._raise_on_error(rc, error)
        return observations, rewards, terminals, actions, logprobs, values, native_profile

class PuffeRL:
    def __init__(self, args, vec, policy, verbose=True):
        config = args['train']
        device = _select_device()
        self.device = device
        torch_num_threads = os.environ.get('PUFFER_TORCH_NUM_THREADS')
        if torch_num_threads:
            torch.set_num_threads(int(torch_num_threads))

        torch.set_float32_matmul_precision('high')
        if _C.gpu:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        self._vec = vec
        self.gpu = vec.gpu
        total_agents = vec.total_agents
        self.total_agents = total_agents
        obs_dtype = _OBS_DTYPE_MAP.get(vec.obs_dtype, torch.uint8)

        if self.gpu:
            self.vec_obs = torch.as_tensor(_CudaPtr(vec.gpu_obs_ptr,
                (total_agents, vec.obs_size), obs_dtype))
            self.vec_rewards = torch.as_tensor(_CudaPtr(vec.gpu_rewards_ptr,
                (total_agents,), torch.float32))
            self.vec_terminals = torch.as_tensor(_CudaPtr(vec.gpu_terminals_ptr,
                (total_agents,), torch.float32))
        else:
            self.vec_obs = _cpu_tensor(vec.obs_ptr,
                (total_agents, vec.obs_size), obs_dtype)
            self.vec_rewards = _cpu_tensor(vec.rewards_ptr,
                (total_agents,), torch.float32)
            self.vec_terminals = _cpu_tensor(vec.terminals_ptr,
                (total_agents,), torch.float32)

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
        if os.environ.get('PUFFER_METAL_ROLLOUT'):
            self._metal_rollout = _MetalPolicyRollout(self)
            self.args['torch_device'] = 'METAL'
        self.train_device = _select_metal_train_device(device)
        if self.train_device.type != torch.device(device).type:
            if self._metal_rollout is None:
                raise RuntimeError('PUFFER_METAL_TRAIN_DEVICE requires PUFFER_METAL_ROLLOUT')
            self.policy = self.policy.to(self.train_device)
            self.args['torch_device'] = (
                f'{self.args["torch_device"]}+'
                f'{self.train_device.type.upper()}_TRAIN'
            )
        self.optimizer = Muon(
            self.policy.parameters(),
            lr=config['learning_rate'],
            momentum=config['beta1'],
            eps=config['eps'],
        )

        self.args.setdefault('torch_device', str(device).upper())
        self.epoch = 0
        self.global_step = 0
        self.last_log_step = 0
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.profile_diagnostics = _env_flag('PUFFER_METAL_DIAGNOSTICS', False)
        profile_device = self.train_device if self.train_device.type == 'mps' else device
        self.profile = Profile(
            gpu=self.gpu,
            device=profile_device,
            sync_mps=self.profile_diagnostics,
        )
        self.train_forward_calls_since_log = 0
        self.train_forward_seconds_since_log = 0.0
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

        P = Profile
        prof.mark(0)
        for t in range(horizon):
            o_device = torch.as_tensor(o, device=device)

            prof.mark(1)
            with torch.no_grad():
                logits, value, state = self.policy.forward_eval(o_device, self.state)
                action, logprob, _ = sample_logits(logits)
            prof.mark(2)

            with torch.no_grad():
                self.state = state
                self.observations[t] = o_device
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

            o, r, d = self.vec_obs, self.vec_rewards, self.vec_terminals
            prof.mark(3)
            prof.elapsed(P.EVAL_GPU, 1, 2)
            prof.elapsed(P.EVAL_ENV, 2, 3)

        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)
        self.global_step += self.total_agents * horizon
        self.env_logs = self._vec.log()

    def _metal_rollouts(self):
        if self._metal_rollout.native:
            return self._metal_native_rollouts()

        prof = self.profile
        config = self.config
        device = self.device
        horizon = config['horizon']

        self.state = tuple(torch.zeros_like(s) for s in self.state) if self.state else ()
        o = self.vec_obs
        r = torch.zeros(self.total_agents, device=device)
        d = torch.zeros(self.total_agents, device=device)
        rollout_seed = (
            int(self.args.get('seed', config.get('seed', 0))) +
            int(self.epoch) * 0x9e3779b9
        ) & 0xffffffff

        P = Profile
        diagnostics = self.profile_diagnostics
        prof.mark(0)
        self._metal_rollout.load(self.policy, o, self.state)
        for t in range(horizon):
            sample_start = prof.timer() if diagnostics else None
            self._metal_rollout.launch_forward_sample_step(
                t,
                horizon,
                rollout_seed,
            )
            sample_seconds = (
                prof.elapsed_since(P.METAL_POLICY_SAMPLE, sample_start)
                if diagnostics else 0.0
            )

            action_read_start = prof.timer() if diagnostics else None
            actions = self._metal_rollout.read_actions()
            action_read_seconds = (
                prof.elapsed_since(P.METAL_ACTION_READ, action_read_start)
                if diagnostics else 0.0
            )
            if diagnostics:
                prof.accum[P.EVAL_GPU] += (
                    sample_seconds + action_read_seconds
                ) * 1000.0

            python_overhead_start = time.perf_counter() if diagnostics else None
            with torch.no_grad():
                self.observations[t] = torch.as_tensor(o, device=device)
                self.rewards[t] = torch.as_tensor(r, device=device)
                self.terminals[t] = torch.as_tensor(d, device=device).float()

            actions_flat = torch.from_numpy(
                actions.astype(np.float32, copy=True).reshape(
                    self.total_agents, self._metal_rollout.NUM_ATNS)
            ).contiguous()
            if diagnostics:
                prof.accum[P.METAL_PYTHON_OVERHEAD] += (
                    time.perf_counter() - python_overhead_start
                ) * 1000.0

            cpu_step_start = prof.timer() if diagnostics else None
            self._vec.cpu_step(actions_flat.data_ptr())
            cpu_step_seconds = (
                prof.elapsed_since(P.METAL_CPU_STEP, cpu_step_start)
                if diagnostics else 0.0
            )

            o, r, d = self.vec_obs, self.vec_rewards, self.vec_terminals
            if t + 1 < horizon:
                obs_upload_start = prof.timer() if diagnostics else None
                self._metal_rollout.write_observations(o)
                obs_upload_seconds = (
                    prof.elapsed_since(P.METAL_OBS_UPLOAD, obs_upload_start)
                    if diagnostics else 0.0
                )
            else:
                obs_upload_seconds = 0.0
            if diagnostics:
                prof.accum[P.EVAL_ENV] += (
                    cpu_step_seconds + obs_upload_seconds
                ) * 1000.0

        prof.mark(1)
        rollout_actions, rollout_logprobs, rollout_values = self._metal_rollout.read_rollout(
            horizon,
            rollout_seed,
        )
        prof.mark(2)
        if diagnostics:
            prof.elapsed(P.METAL_READBACK, 1, 2)
        python_overhead_start = time.perf_counter() if diagnostics else None
        with torch.no_grad():
            self.actions.copy_(torch.from_numpy(
                rollout_actions.astype(np.float32, copy=False)))
            self.logprobs.copy_(torch.from_numpy(rollout_logprobs))
            self.values.copy_(torch.from_numpy(rollout_values))
        if diagnostics:
            prof.accum[P.METAL_PYTHON_OVERHEAD] += (
                time.perf_counter() - python_overhead_start
            ) * 1000.0

        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)
        self.global_step += self.total_agents * horizon
        self.env_logs = self._vec.log()

    def _metal_native_rollouts(self):
        prof = self.profile
        horizon = self.config['horizon']

        self.state = tuple(torch.zeros_like(s) for s in self.state) if self.state else ()
        P = Profile
        prof.mark(0)
        (
            observations,
            rewards,
            terminals,
            actions,
            logprobs,
            values,
            native_profile,
        ) = self._metal_rollout.native_rollouts(self.policy, self.state, self.epoch)
        prof.mark(1)
        prof.elapsed(P.ROLLOUT, 0, 1)

        if self.profile_diagnostics:
            ref_device = next(self.policy.parameters()).device
            ref_state = (
                tuple(torch.zeros_like(s, device=ref_device) for s in self.state)
                if self.state else ()
            )
            max_logprob_diff = 0.0
            max_value_diff = 0.0
            with torch.no_grad():
                for t in range(horizon):
                    obs_t = torch.from_numpy(observations[t]).to(ref_device)
                    action_t = torch.from_numpy(actions[t]).to(ref_device).reshape(self.total_agents, 1)
                    logits, value, ref_state = self.policy.forward_eval(obs_t, ref_state)
                    _action, logprob, _entropy = sample_logits(logits, action_t)
                    max_logprob_diff = max(
                        max_logprob_diff,
                        float(torch.max(torch.abs(logprob.cpu() - torch.from_numpy(logprobs[t])))))
                    max_value_diff = max(
                        max_value_diff,
                        float(torch.max(torch.abs(value.flatten().cpu() - torch.from_numpy(values[t])))))
            tolerance = 2e-5
            if max_logprob_diff > tolerance or max_value_diff > tolerance:
                raise RuntimeError(
                    'native Metal rollout parity failed: '
                    f'logprobs_max_abs={max_logprob_diff:.8g} '
                    f'values_max_abs={max_value_diff:.8g} '
                    f'tolerance={tolerance:.8g}')

        if self.profile_diagnostics:
            prof.accum[P.METAL_POLICY_SAMPLE] += native_profile.policy_sample_seconds * 1000.0
            prof.accum[P.METAL_ACTION_READ] += native_profile.action_read_seconds * 1000.0
            prof.accum[P.METAL_CPU_STEP] += native_profile.cpu_step_seconds * 1000.0
            prof.accum[P.METAL_OBS_UPLOAD] += native_profile.obs_upload_seconds * 1000.0
            prof.accum[P.EVAL_GPU] += (
                native_profile.policy_sample_seconds +
                native_profile.action_read_seconds
            ) * 1000.0
            prof.accum[P.EVAL_ENV] += (
                native_profile.cpu_step_seconds +
                native_profile.obs_upload_seconds
            ) * 1000.0

        with torch.no_grad():
            self.observations.copy_(torch.from_numpy(observations))
            self.rewards.copy_(torch.from_numpy(rewards))
            self.terminals.copy_(torch.from_numpy(terminals))
            self.actions.copy_(torch.from_numpy(actions[:, :, None]))
            self.logprobs.copy_(torch.from_numpy(logprobs))
            self.values.copy_(torch.from_numpy(values))

        self.global_step += self.total_agents * horizon
        self.env_logs = self._vec.log()

    def train(self):
        prof = self.profile
        losses = defaultdict(float)
        config = self.config
        device = self.train_device
        P = Profile
        diagnostics = self.profile_diagnostics
        train_start = prof.timer()

        b0 = config['prio_beta0']
        a = config['prio_alpha']
        clip_coef = config['clip_coef']
        vf_clip = config['vf_clip_coef']
        anneal_beta = b0 + (1 - b0)*a*self.epoch/self.total_epochs
        use_metal_mps_train = (
            device.type == 'mps' and
            self._metal_rollout is not None
        )

        learning_rate = config['learning_rate']
        if config['anneal_lr'] and self.epoch > 0:
            lr_ratio = self.epoch / self.total_epochs
            lr_min = config['learning_rate'] * config['min_lr_ratio']
            learning_rate = lr_min + 0.5*(learning_rate - lr_min) * (1 + np.cos(np.pi * lr_ratio))
            self.optimizer.param_groups[0]['lr'] = learning_rate

        transfer_start = prof.timer() if diagnostics else None
        # Transpose from [horizon, agents] (contiguous writes) to [agents, horizon] (minibatch indexing)
        obs_cpu = self.observations.transpose(0, 1).contiguous()
        act_cpu = self.actions.transpose(0, 1).contiguous()
        val_cpu = self.values.T.contiguous()
        lp_cpu = self.logprobs.T.contiguous()
        rew_cpu = self.rewards.T.contiguous().clamp(-1, 1)
        ter_cpu = self.terminals.T.contiguous()

        obs = obs_cpu.to(device)
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
            rew = None
            ter = None
        else:
            act = act_cpu.to(device)
            val = val_cpu.to(device)
            lp = lp_cpu.to(device)
            rew = rew_cpu.to(device)
            ter = ter_cpu.to(device)
        transfer_seconds = (
            prof.elapsed_since(P.TRAIN_MISC_TRANSFER, transfer_start)
            if diagnostics else 0.0
        )
        if diagnostics:
            prof.accum[P.TRAIN_MISC] += transfer_seconds * 1000.0
        misc_accounted_seconds = transfer_seconds
        advantage_total_seconds = 0.0
        replay_total_seconds = 0.0
        loss_opt_total_seconds = 0.0

        if use_metal_mps_train:
            train_ratio_cpu = torch.ones(self.total_agents, config['horizon'])
            advantages_cpu = torch.zeros_like(val_cpu)
            train_ratio = None
        else:
            train_ratio = torch.ones(self.total_agents, config['horizon'], device=device)
            train_ratio_cpu = None
            advantages_cpu = None

        num_minibatches = int(config['replay_ratio'] * self.batch_size / config['minibatch_size'])
        forward_calls = 0
        forward_seconds = 0.0
        for mb in range(num_minibatches):
            advantage_start = prof.timer() if diagnostics else None
            shape = val.shape
            if use_metal_mps_train:
                advantages_cpu.zero_()
                compute_puff_advantage(val_cpu, rew_cpu,
                    ter_cpu, train_ratio_cpu, advantages_cpu, config['gamma'],
                    config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
                advantages = advantages_cpu.to(device)
            else:
                advantages = torch.zeros(shape, device=device)
                advantages = compute_puff_advantage(val, rew,
                    ter, train_ratio, advantages, config['gamma'],
                    config['gae_lambda'], config['vtrace_rho_clip'], config['vtrace_c_clip'])
            advantage_seconds = (
                prof.elapsed_since(P.TRAIN_MISC_ADVANTAGE, advantage_start)
                if diagnostics else 0.0
            )
            if diagnostics:
                prof.accum[P.TRAIN_MISC] += advantage_seconds * 1000.0
                misc_accounted_seconds += advantage_seconds
                advantage_total_seconds += advantage_seconds

            replay_start = prof.timer() if diagnostics else None
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
            mb_values = val[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]
            replay_seconds = (
                prof.elapsed_since(P.TRAIN_MISC_REPLAY, replay_start)
                if diagnostics else 0.0
            )
            if diagnostics:
                prof.accum[P.TRAIN_MISC] += replay_seconds * 1000.0
                misc_accounted_seconds += replay_seconds
                replay_total_seconds += replay_seconds

            forward_start = prof.timer() if diagnostics else None
            logits, newvalue = self.policy(mb_obs)
            actions, newlogprob, entropy = sample_logits(logits, action=mb_actions)
            if diagnostics:
                forward_seconds += prof.elapsed_since(P.TRAIN_FORWARD, forward_start)
                forward_calls += 1

            loss_start = prof.timer() if diagnostics else None
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            if use_metal_mps_train:
                if mb + 1 < num_minibatches:
                    train_ratio_cpu[idx_cpu] = ratio.detach().cpu()
            else:
                train_ratio[idx] = ratio.detach()

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
            if use_metal_mps_train and mb + 1 < num_minibatches:
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
            loss_seconds = (
                prof.elapsed_since(P.TRAIN_MISC_LOSS_OPT, loss_start)
                if diagnostics else 0.0
            )
            if diagnostics:
                prof.accum[P.TRAIN_MISC] += loss_seconds * 1000.0
                misc_accounted_seconds += loss_seconds
                loss_opt_total_seconds += loss_seconds

        train_seconds = prof.elapsed_since(P.TRAIN, train_start)
        other_seconds = max(0.0, train_seconds - misc_accounted_seconds - forward_seconds)
        if diagnostics:
            prof.accum[P.TRAIN_MISC_OTHER] += other_seconds * 1000.0
            prof.accum[P.TRAIN_MISC] += other_seconds * 1000.0
            self.train_forward_calls_since_log += forward_calls
            self.train_forward_seconds_since_log += forward_seconds

        losses = {k: v.item() / num_minibatches for k, v in losses.items()}
        y_pred = val.flatten()
        y_true = advantages.flatten() + val.flatten()
        var_y = y_true.var()
        explained_var = torch.nan if var_y == 0 else (1 - (y_true - y_pred).var() / var_y).item()
        losses['explained_variance'] = explained_var
        if diagnostics:
            losses['torch_train_epoch'] = train_seconds
            losses['torch_train_transfer'] = transfer_seconds
            losses['torch_train_advantage'] = advantage_total_seconds
            losses['torch_train_replay'] = replay_total_seconds
            losses['torch_train_forward'] = forward_seconds
            losses['torch_train_loss_opt'] = loss_opt_total_seconds
            losses['torch_train_other'] = other_seconds

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
                'train_misc_transfer': perf[P.TRAIN_MISC_TRANSFER],
                'train_misc_advantage': perf[P.TRAIN_MISC_ADVANTAGE],
                'train_misc_replay': perf[P.TRAIN_MISC_REPLAY],
                'train_misc_loss_opt': perf[P.TRAIN_MISC_LOSS_OPT],
                'train_misc_other': perf[P.TRAIN_MISC_OTHER],
                'train_forward': perf[P.TRAIN_FORWARD],
                'train_forward_calls': self.train_forward_calls_since_log,
                'train_forward_mean': (
                    self.train_forward_seconds_since_log / self.train_forward_calls_since_log
                    if self.train_forward_calls_since_log else 0.0
                ),
                'metal_readback': perf[P.METAL_READBACK],
                'metal_policy_sample': perf[P.METAL_POLICY_SAMPLE],
                'metal_action_read': perf[P.METAL_ACTION_READ],
                'metal_cpu_step': perf[P.METAL_CPU_STEP],
                'metal_obs_upload': perf[P.METAL_OBS_UPLOAD],
                'metal_python_overhead': perf[P.METAL_PYTHON_OVERHEAD],
            },
            'util': (
                dict(_C.get_utilization(self.args.get('gpu_id', 0)))
                if self.gpu else
                {
                    'cpu_mem_gb': pufferlib.pufferl.current_rss_gb(),
                    'torch_num_threads': torch.get_num_threads(),
                    'omp_num_threads': int(os.environ.get('OMP_NUM_THREADS', '0') or 0),
                }
            ),
        }
        self.train_forward_calls_since_log = 0
        self.train_forward_seconds_since_log = 0.0
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
    (
        ROLLOUT,
        EVAL_GPU,
        EVAL_ENV,
        TRAIN,
        TRAIN_MISC,
        TRAIN_FORWARD,
        TRAIN_MISC_TRANSFER,
        TRAIN_MISC_ADVANTAGE,
        TRAIN_MISC_REPLAY,
        TRAIN_MISC_LOSS_OPT,
        TRAIN_MISC_OTHER,
        METAL_READBACK,
        METAL_POLICY_SAMPLE,
        METAL_ACTION_READ,
        METAL_CPU_STEP,
        METAL_OBS_UPLOAD,
        METAL_PYTHON_OVERHEAD,
        NUM,
    ) = range(18)

    def __init__(self, gpu=True, device='cpu', sync_mps=False):
        self.accum = [0.0] * Profile.NUM
        self.gpu = gpu
        self.device_type = torch.device(device).type
        self.sync_mps = sync_mps
        if gpu:
            self._events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        else:
            self._stamps = [0.0] * 4

    def _sync(self):
        if self.device_type == 'mps' and self.sync_mps:
            torch.mps.synchronize()

    def timer(self):
        if self.gpu:
            torch.cuda.synchronize()
        else:
            self._sync()
        return time.perf_counter()

    def elapsed_since(self, idx, start):
        if self.gpu:
            torch.cuda.synchronize()
        else:
            self._sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.accum[idx] += elapsed_ms
        return elapsed_ms / 1000.0

    def mark(self, idx):
        if self.gpu:
            self._events[idx].record()
        else:
            self._sync()
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

    device = _select_device()
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
