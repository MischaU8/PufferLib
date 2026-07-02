'''End-to-end parity tests for the macOS Metal rollout backend.

Drives _MetalPolicyRollout (pufferlib/torch_pufferl.py) against the reference
torch policy (pufferlib/models.py) and checks action legality, logprob/value
parity, and sampler distribution. Runnable directly or via pytest.
'''
import sys

if sys.platform != 'darwin':
    try:
        import pytest
        pytest.skip('Metal rollout backend requires macOS', allow_module_level=True)
    except ImportError:
        print('SKIP: Metal rollout backend requires macOS')
        sys.exit(0)

from types import SimpleNamespace

import numpy as np
import torch

import pufferlib.models
from pufferlib.torch_pufferl import _MetalPolicyRollout, sample_logits

VALUE_ATOL = 5e-4
LOGPROB_ATOL = 5e-4


def make_policy(obs_size, nvec, hidden, num_layers, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = pufferlib.models.Policy(
        pufferlib.models.DefaultEncoder(obs_size, hidden),
        pufferlib.models.DefaultDecoder(nvec, hidden),
        pufferlib.models.MinGRU(hidden_size=hidden, num_layers=num_layers),
    )
    policy.eval()
    return policy


def make_rollout(policy, obs_size, nvec, agents):
    pufferl = SimpleNamespace(
        gpu=0,
        _vec=SimpleNamespace(obs_size=obs_size, act_sizes=list(nvec), num_atns=len(nvec)),
        policy=policy,
        total_agents=agents,
    )
    return _MetalPolicyRollout(pufferl)


def drive_rollout(rollout, policy, obs_seq, horizon, seed, mask=None):
    '''Run the Metal rollout over obs_seq; returns per-step action copies and
    the (actions, logprobs, values) arrays from read_rollout.'''
    agents = obs_seq[0].shape[0]
    state = policy.initial_state(agents, device='cpu')
    rollout.load(policy, obs_seq[0], state)
    step_actions = []
    for t in range(horizon):
        rollout.launch_forward_sample_step(t, horizon, seed, mask)
        step_actions.append(rollout.read_actions().copy())
        if t + 1 < horizon:
            rollout.write_observations(obs_seq[t + 1])
    actions, logprobs, values = rollout.read_rollout(horizon, seed)
    return step_actions, actions, logprobs, values


def reference_step(policy, obs_t, state):
    with torch.no_grad():
        logits, values, state = policy.forward_eval(obs_t, state)
    return logits, values.reshape(-1), state


def random_mask(agents, nvec, seed):
    '''Random uint8 mask [agents, sum(nvec)] with >=1 legal action per head.'''
    gen = np.random.default_rng(seed)
    mask = (gen.random((agents, int(sum(nvec)))) < 0.6).astype(np.uint8)
    offset = 0
    for n in nvec:
        forced = gen.integers(0, n, size=agents)
        mask[np.arange(agents), offset + forced] = 1
        offset += n
    return torch.from_numpy(mask).contiguous()


def test_discrete_masked_parity():
    obs_size, nvec, hidden, num_layers = 16, (5, 3), 48, 2
    agents, horizon, seed = 64, 8, 1234
    policy = make_policy(obs_size, nvec, hidden, num_layers, seed=7)
    obs_seq = [torch.randn(agents, obs_size) for _ in range(horizon)]
    mask = random_mask(agents, nvec, seed=11)  # must stay alive across steps
    mask_bool = mask.bool()

    rollout = make_rollout(policy, obs_size, nvec, agents)
    try:
        step_actions, actions, logprobs, values = drive_rollout(
            rollout, policy, obs_seq, horizon, seed, mask)
    finally:
        rollout.close()

    # (d) per-step read_actions matches read_rollout
    for t in range(horizon):
        assert np.array_equal(step_actions[t], actions[t]), f'step {t} action mismatch'

    # (a) every sampled action is mask-legal
    offset = 0
    for head, n in enumerate(nvec):
        head_mask = mask.numpy()[:, offset:offset + n]
        legal = head_mask[np.arange(agents)[:, None], actions[:, :, head].T]
        assert legal.all(), f'illegal action sampled in head {head}'
        offset += n

    # (b)/(c) reference replay from zero state
    max_lp_diff = max_v_diff = 0.0
    state = policy.initial_state(agents, device='cpu')
    for t in range(horizon):
        logits, ref_values, state = reference_step(policy, obs_seq[t], state)
        with torch.no_grad():
            _, ref_logprob, _ = sample_logits(
                logits, action=torch.from_numpy(actions[t]), action_mask=mask_bool)
        lp_diff = np.abs(ref_logprob.numpy() - logprobs[t]).max()
        v_diff = np.abs(ref_values.numpy() - values[t]).max()
        max_lp_diff = max(max_lp_diff, float(lp_diff))
        max_v_diff = max(max_v_diff, float(v_diff))
    assert max_lp_diff < LOGPROB_ATOL, f'logprob max diff {max_lp_diff}'
    assert max_v_diff < VALUE_ATOL, f'value max diff {max_v_diff}'
    print(f'test_discrete_masked_parity: max logprob diff {max_lp_diff:.2e}, '
          f'max value diff {max_v_diff:.2e}')


def test_continuous_parity():
    obs_size, nvec, hidden, num_layers = 16, (1, 1, 1), 32, 1
    agents, horizon, seed = 64, 4, 4321
    policy = make_policy(obs_size, nvec, hidden, num_layers, seed=13)
    assert policy.decoder.is_continuous
    obs_seq = [torch.randn(agents, obs_size) for _ in range(horizon)]

    rollout = make_rollout(policy, obs_size, nvec, agents)
    try:
        step_actions, actions, logprobs, values = drive_rollout(
            rollout, policy, obs_seq, horizon, seed)
    finally:
        rollout.close()

    assert actions.dtype == np.float32
    for t in range(horizon):
        assert np.array_equal(step_actions[t], actions[t]), f'step {t} action mismatch'

    max_lp_diff = max_v_diff = 0.0
    state = policy.initial_state(agents, device='cpu')
    for t in range(horizon):
        logits, ref_values, state = reference_step(policy, obs_seq[t], state)
        with torch.no_grad():
            _, ref_logprob, _ = sample_logits(
                logits, action=torch.from_numpy(actions[t].copy()))
        lp_diff = np.abs(ref_logprob.numpy() - logprobs[t]).max()
        v_diff = np.abs(ref_values.numpy() - values[t]).max()
        max_lp_diff = max(max_lp_diff, float(lp_diff))
        max_v_diff = max(max_v_diff, float(v_diff))
    assert max_lp_diff < LOGPROB_ATOL, f'logprob max diff {max_lp_diff}'
    assert max_v_diff < VALUE_ATOL, f'value max diff {max_v_diff}'
    print(f'test_continuous_parity: max logprob diff {max_lp_diff:.2e}, '
          f'max value diff {max_v_diff:.2e}')


def test_sampler_distribution():
    obs_size, nvec, hidden, num_layers = 16, (7,), 32, 1
    agents, horizon, seed = 2048, 8, 999
    policy = make_policy(obs_size, nvec, hidden, num_layers, seed=17)
    row = torch.randn(1, obs_size)
    obs = row.expand(agents, obs_size).contiguous()
    obs_seq = [obs] * horizon

    rollout = make_rollout(policy, obs_size, nvec, agents)
    try:
        _, actions, _, _ = drive_rollout(rollout, policy, obs_seq, horizon, seed)
    finally:
        rollout.close()

    # With n=2048 samples/step and 7 categories, the empirical frequency of a
    # category with probability p has std sqrt(p(1-p)/n) <= sqrt(0.25/2048)
    # ~= 0.011 (typically ~0.008 for p~1/7). A per-step bound of 0.04 is
    # ~4-5 sigma per category, so a violation indicates a sampler bug rather
    # than noise.
    state = policy.initial_state(agents, device='cpu')
    max_freq_diff = 0.0
    for t in range(horizon):
        logits, _, state = reference_step(policy, obs_seq[t], state)
        probs = torch.softmax(logits, dim=-1)
        # identical obs + zero initial state => identical distribution per agent
        agent_spread = (probs - probs[0]).abs().max().item()
        assert agent_spread < 1e-5, f'step {t}: probs differ across agents ({agent_spread})'
        ref = probs[0].numpy()
        counts = np.bincount(actions[t, :, 0], minlength=nvec[0]).astype(np.float64)
        emp = counts / agents
        diff = np.abs(emp - ref).max()
        max_freq_diff = max(max_freq_diff, float(diff))
        assert diff < 0.04, f'step {t}: empirical vs reference diff {diff}'
    print(f'test_sampler_distribution: max |empirical - reference| {max_freq_diff:.4f}')


if __name__ == '__main__':
    test_discrete_masked_parity()
    test_continuous_parity()
    test_sampler_distribution()
    print('All Metal rollout tests passed.')
