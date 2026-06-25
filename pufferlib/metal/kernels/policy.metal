#include <metal_stdlib>

using namespace metal;

// Shape-parametric policy kernels for the PufferLib default topology
// (DefaultEncoder linear -> N-layer MinGRU -> DefaultDecoder + value head)
// with a single discrete action head. Shapes (obs/hidden/action/num_layers)
// are runtime values carried in the config struct, not compile-time constants,
// so the same kernels serve Breakout, CartPole, and other vector-obs envs.
//
// The only compile-time bound is PUFFER_MAX_HIDDEN_SIZE, used to size the
// thread-local scratch arrays in the standalone parity/microbenchmark kernels
// (Metal forbids function-constant-sized thread arrays). The production rollout
// path uses the device-memory glue kernels below, which have no such bound.
#define PUFFER_MAX_HIDDEN_SIZE 512u
#define PUFFER_MAX_HEADS 8u

struct PolicyForwardConfig {
    uint batch_size;
    uint obs_size;
    uint hidden_size;
    uint num_layers;
    uint action_size;            // discrete: sum(act_sizes); continuous: num dims
    uint num_atns;               // discrete heads, or continuous action dims
    uint act_sizes[PUFFER_MAX_HEADS];
    uint is_continuous;          // 0 = discrete heads, 1 = Gaussian continuous
};

struct PolicySampleConfig {
    uint rollout_step;
    uint rollout_horizon;
    uint seed;
    uint reserved;
};

struct PuffAdvantageConfig {
    uint num_steps;
    uint horizon;
    float gamma;
    float gae_lambda;
    float vtrace_rho_clip;
    float vtrace_c_clip;
};

static inline bool valid_config(constant PolicyForwardConfig& config) {
    if (config.obs_size == 0u || config.hidden_size == 0u ||
            config.hidden_size > PUFFER_MAX_HIDDEN_SIZE ||
            config.num_layers < 1u || config.action_size == 0u ||
            config.num_atns < 1u || config.num_atns > PUFFER_MAX_HEADS) {
        return false;
    }
    if (config.is_continuous != 0u) {
        // One Gaussian dim per action; the decoder emits num_atns means.
        return config.action_size == config.num_atns;
    }
    uint total = 0u;
    for (uint h = 0u; h < config.num_atns; ++h) {
        if (config.act_sizes[h] == 0u) {
            return false;
        }
        total += config.act_sizes[h];
    }
    return total == config.action_size;
}

static inline float puffer_sigmoid(float x) {
    return 1.0f / (1.0f + fast::exp(-x));
}

static inline float puffer_mingru_g(float x) {
    return x >= 0.0f ? x + 0.5f : puffer_sigmoid(x);
}

static inline float puffer_safe_logit(float x) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? 1.0e20f : -1.0e20f;
    }
    return clamp(x, -1.0e20f, 1.0e20f);
}

static inline uint puffer_hash_u32(uint x) {
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

static inline float puffer_uniform01(uint seed, uint step, uint batch_index, uint head) {
    uint x = seed ^ (step * 0x9e3779b9u) ^ (batch_index * 0x85ebca6bu) ^ (head * 0xc2b2ae35u);
    uint h = puffer_hash_u32(x);
    return (float((h >> 8u) & 0x00ffffffu) + 0.5f) * (1.0f / 16777216.0f);
}

// Standard normal via Box-Muller. Two independent uniforms are keyed by
// 2*dim and 2*dim+1 so each continuous action dim gets an independent draw.
static inline float puffer_standard_normal(uint seed, uint step, uint batch_index, uint dim) {
    float u1 = puffer_uniform01(seed, step, batch_index, 2u * dim);
    float u2 = puffer_uniform01(seed, step, batch_index, 2u * dim + 1u);
    return sqrt(-2.0f * fast::log(u1)) * fast::cos(2.0f * 3.14159265358979323846f * u2);
}

static inline void puffer_mingru_layer_parallel(
    threadgroup float* h,
    threadgroup float* next_h,
    uint layer,
    uint tid,
    uint batch_index,
    uint batch_size,
    uint hidden_size,
    device const float* state_in,
    device float* state_out,
    device const float* weight) {
    float hidden = 0.0f;
    float gate = 0.0f;
    float proj = 0.0f;

    for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
        float x = h[in_idx];
        hidden += weight[tid * hidden_size + in_idx] * x;
        gate += weight[(hidden_size + tid) * hidden_size + in_idx] * x;
        proj += weight[(2u * hidden_size + tid) * hidden_size + in_idx] * x;
    }

    uint state_offset = layer * batch_size * hidden_size +
        batch_index * hidden_size + tid;
    float prev = state_in[state_offset];
    float gate_sigmoid = puffer_sigmoid(gate);
    float out = prev * (1.0f - gate_sigmoid) + puffer_mingru_g(hidden) * gate_sigmoid;
    float proj_sigmoid = puffer_sigmoid(proj);
    next_h[tid] = proj_sigmoid * out + (1.0f - proj_sigmoid) * h[tid];
    state_out[state_offset] = out;
}

static inline void puffer_mingru_layer_serial(
    thread float* h,
    uint layer,
    uint batch_index,
    uint batch_size,
    uint hidden_size,
    device const float* state_in,
    device float* state_out,
    device const float* weight) {
    float next_h[PUFFER_MAX_HIDDEN_SIZE];

    for (uint out_idx = 0; out_idx < hidden_size; ++out_idx) {
        float hidden = 0.0f;
        float gate = 0.0f;
        float proj = 0.0f;

        for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
            float x = h[in_idx];
            hidden += weight[out_idx * hidden_size + in_idx] * x;
            gate += weight[(hidden_size + out_idx) * hidden_size + in_idx] * x;
            proj += weight[(2u * hidden_size + out_idx) * hidden_size + in_idx] * x;
        }

        uint state_offset = layer * batch_size * hidden_size +
            batch_index * hidden_size + out_idx;
        float prev = state_in[state_offset];
        float gate_sigmoid = puffer_sigmoid(gate);
        float out = prev * (1.0f - gate_sigmoid) + puffer_mingru_g(hidden) * gate_sigmoid;
        float proj_sigmoid = puffer_sigmoid(proj);
        next_h[out_idx] = proj_sigmoid * out + (1.0f - proj_sigmoid) * h[out_idx];
        state_out[state_offset] = out;
    }

    for (uint i = 0; i < hidden_size; ++i) {
        h[i] = next_h[i];
    }
}

// Standalone serial forward: one thread per batch element. The MinGRU layer
// weights are passed as two buffers, so this parity kernel supports the common
// 2-layer policy used by Breakout and CartPole. The production rollout path
// (host MPS matmul + glue kernels) handles num_layers generally.
kernel void puffer_policy_forward_eval_serial(
    device const float* observations [[buffer(0)]],
    device const float* state_in [[buffer(1)]],
    device const float* encoder_weight [[buffer(2)]],
    device const float* encoder_bias [[buffer(3)]],
    device const float* gru_weights [[buffer(4)]],
    device const float* decoder_weight [[buffer(5)]],
    device const float* decoder_bias [[buffer(6)]],
    device const float* value_weight [[buffer(7)]],
    device const float* value_bias [[buffer(8)]],
    device float* logits_out [[buffer(9)]],
    device float* values_out [[buffer(10)]],
    device float* state_out [[buffer(11)]],
    constant PolicyForwardConfig& config [[buffer(12)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= config.batch_size || !valid_config(config)) {
        return;
    }

    uint obs_size = config.obs_size;
    uint hidden_size = config.hidden_size;
    uint action_size = config.action_size;
    uint layer_stride = 3u * hidden_size * hidden_size;

    float h[PUFFER_MAX_HIDDEN_SIZE];

    for (uint out_idx = 0; out_idx < hidden_size; ++out_idx) {
        float acc = encoder_bias[out_idx];
        for (uint in_idx = 0; in_idx < obs_size; ++in_idx) {
            acc += encoder_weight[out_idx * obs_size + in_idx] *
                observations[gid * obs_size + in_idx];
        }
        h[out_idx] = acc;
    }

    for (uint layer = 0; layer < config.num_layers; ++layer) {
        device const float* weight = gru_weights + layer * layer_stride;
        puffer_mingru_layer_serial(h, layer, gid, config.batch_size, hidden_size,
            state_in, state_out, weight);
    }

    for (uint action = 0; action < action_size; ++action) {
        float acc = decoder_bias[action];
        for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
            acc += decoder_weight[action * hidden_size + in_idx] * h[in_idx];
        }
        logits_out[gid * action_size + action] = acc;
    }

    float value = value_bias[0];
    for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
        value += value_weight[in_idx] * h[in_idx];
    }
    values_out[gid] = value;
}

// Standalone threadgroup forward: one threadgroup per batch element, one thread
// per hidden unit. Threadgroup scratch is sized to the compile-time bound and
// indexed with the runtime hidden_size.
kernel void puffer_policy_forward_eval(
    device const float* observations [[buffer(0)]],
    device const float* state_in [[buffer(1)]],
    device const float* encoder_weight [[buffer(2)]],
    device const float* encoder_bias [[buffer(3)]],
    device const float* gru_weights [[buffer(4)]],
    device const float* decoder_weight [[buffer(5)]],
    device const float* decoder_bias [[buffer(6)]],
    device const float* value_weight [[buffer(7)]],
    device const float* value_bias [[buffer(8)]],
    device float* logits_out [[buffer(9)]],
    device float* values_out [[buffer(10)]],
    device float* state_out [[buffer(11)]],
    constant PolicyForwardConfig& config [[buffer(12)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
    uint batch_index = tgid.x;
    uint hidden_size = config.hidden_size;
    uint action_size = config.action_size;
    if (batch_index >= config.batch_size || !valid_config(config)) {
        return;
    }
    // The host dispatches max(hidden_size, action_size + 1) threads: threads
    // below hidden_size own a recurrent unit, threads below action_size own a
    // logit, and thread action_size owns the value. These sets overlap or
    // exceed each other, so gate the hidden-state work on tid < hidden_size
    // (not the whole kernel) and keep the barriers uniform across all threads.
    bool hidden_thread = tid < hidden_size;

    uint layer_stride = 3u * hidden_size * hidden_size;
    threadgroup float h[PUFFER_MAX_HIDDEN_SIZE];
    threadgroup float next_h[PUFFER_MAX_HIDDEN_SIZE];

    if (hidden_thread) {
        float acc = encoder_bias[tid];
        for (uint in_idx = 0; in_idx < config.obs_size; ++in_idx) {
            acc += encoder_weight[tid * config.obs_size + in_idx] *
                observations[batch_index * config.obs_size + in_idx];
        }
        h[tid] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint layer = 0; layer < config.num_layers; ++layer) {
        device const float* weight = gru_weights + layer * layer_stride;
        if (hidden_thread) {
            puffer_mingru_layer_parallel(
                h, next_h, layer, tid, batch_index, config.batch_size, hidden_size,
                state_in, state_out, weight);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (hidden_thread) {
            h[tid] = next_h[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < action_size) {
        float logit = decoder_bias[tid];
        for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
            logit += decoder_weight[tid * hidden_size + in_idx] * h[in_idx];
        }
        logits_out[batch_index * action_size + tid] = logit;
    } else if (tid == action_size) {
        float value = value_bias[0];
        for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
            value += value_weight[in_idx] * h[in_idx];
        }
        values_out[batch_index] = value;
    }
}

kernel void puffer_policy_encoder(
    device const float* observations [[buffer(0)]],
    device const float* encoder_weight [[buffer(1)]],
    device const float* encoder_bias [[buffer(2)]],
    device float* hidden_out [[buffer(3)]],
    constant PolicyForwardConfig& config [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint hidden_idx = gid.x;
    uint batch_idx = gid.y;
    if (batch_idx >= config.batch_size || hidden_idx >= config.hidden_size ||
            !valid_config(config)) {
        return;
    }

    uint obs_size = config.obs_size;
    float acc = encoder_bias[hidden_idx];
    for (uint in_idx = 0; in_idx < obs_size; ++in_idx) {
        acc += encoder_weight[hidden_idx * obs_size + in_idx] *
            observations[batch_idx * obs_size + in_idx];
    }
    hidden_out[batch_idx * config.hidden_size + hidden_idx] = acc;
}

kernel void puffer_policy_mingru_layer(
    device const float* hidden_in [[buffer(0)]],
    device const float* state_in [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* hidden_out [[buffer(3)]],
    device float* state_out [[buffer(4)]],
    constant uint& layer [[buffer(5)]],
    constant PolicyForwardConfig& config [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint hidden_idx = gid.x;
    uint batch_idx = gid.y;
    uint hidden_size = config.hidden_size;
    if (batch_idx >= config.batch_size || hidden_idx >= hidden_size ||
            layer >= config.num_layers || !valid_config(config)) {
        return;
    }

    float hidden = 0.0f;
    float gate = 0.0f;
    float proj = 0.0f;
    uint batch_offset = batch_idx * hidden_size;
    for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
        float x = hidden_in[batch_offset + in_idx];
        hidden += weight[hidden_idx * hidden_size + in_idx] * x;
        gate += weight[(hidden_size + hidden_idx) * hidden_size + in_idx] * x;
        proj += weight[(2u * hidden_size + hidden_idx) * hidden_size + in_idx] * x;
    }

    uint state_offset = layer * config.batch_size * hidden_size +
        batch_offset + hidden_idx;
    float prev = state_in[state_offset];
    float gate_sigmoid = puffer_sigmoid(gate);
    float out = prev * (1.0f - gate_sigmoid) + puffer_mingru_g(hidden) * gate_sigmoid;
    float proj_sigmoid = puffer_sigmoid(proj);
    hidden_out[batch_offset + hidden_idx] =
        proj_sigmoid * out + (1.0f - proj_sigmoid) * hidden_in[batch_offset + hidden_idx];
    state_out[state_offset] = out;
}

kernel void puffer_policy_add_bias(
    device float* data [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& columns [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint column = gid.x;
    uint row = gid.y;
    if (column >= columns) {
        return;
    }

    data[row * columns + column] += bias[column];
}

kernel void puffer_policy_mingru_apply(
    device const float* hidden_in [[buffer(0)]],
    device const float* linear [[buffer(1)]],
    device float* state [[buffer(2)]],
    device float* hidden_out [[buffer(3)]],
    constant uint& layer [[buffer(4)]],
    constant PolicyForwardConfig& config [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint hidden_idx = gid.x;
    uint batch_idx = gid.y;
    uint hidden_size = config.hidden_size;
    if (batch_idx >= config.batch_size || hidden_idx >= hidden_size ||
            layer >= config.num_layers || !valid_config(config)) {
        return;
    }

    uint hidden_offset = batch_idx * hidden_size + hidden_idx;
    uint linear_offset = batch_idx * (3u * hidden_size) + hidden_idx;
    float hidden = linear[linear_offset];
    float gate = linear[linear_offset + hidden_size];
    float proj = linear[linear_offset + 2u * hidden_size];

    uint state_offset = layer * config.batch_size * hidden_size + hidden_offset;
    float prev = state[state_offset];
    float gate_sigmoid = puffer_sigmoid(gate);
    float out = prev * (1.0f - gate_sigmoid) + puffer_mingru_g(hidden) * gate_sigmoid;
    float proj_sigmoid = puffer_sigmoid(proj);
    hidden_out[hidden_offset] = proj_sigmoid * out +
        (1.0f - proj_sigmoid) * hidden_in[hidden_offset];
    state[state_offset] = out;
}

kernel void puffer_policy_decoder(
    device const float* hidden [[buffer(0)]],
    device const float* decoder_weight [[buffer(1)]],
    device const float* decoder_bias [[buffer(2)]],
    device const float* value_weight [[buffer(3)]],
    device const float* value_bias [[buffer(4)]],
    device float* logits_out [[buffer(5)]],
    device float* values_out [[buffer(6)]],
    constant PolicyForwardConfig& config [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]) {
    uint output_idx = gid.x;
    uint batch_idx = gid.y;
    uint hidden_size = config.hidden_size;
    uint action_size = config.action_size;
    if (batch_idx >= config.batch_size || output_idx > action_size ||
            !valid_config(config)) {
        return;
    }

    uint batch_offset = batch_idx * hidden_size;
    if (output_idx < action_size) {
        float acc = decoder_bias[output_idx];
        for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
            acc += decoder_weight[output_idx * hidden_size + in_idx] *
                hidden[batch_offset + in_idx];
        }
        logits_out[batch_idx * action_size + output_idx] = acc;
    } else {
        float value = value_bias[0];
        for (uint in_idx = 0; in_idx < hidden_size; ++in_idx) {
            value += value_weight[in_idx] * hidden[batch_offset + in_idx];
        }
        values_out[batch_idx] = value;
    }
}

// Categorical sampler for one or more discrete action heads. The decoder emits
// concatenated logits (total length action_size = sum(act_sizes)); each head
// owns the next act_sizes[head] entries. Per head we stream its segment three
// times (max, denom, cumulative pick) with an independent head-keyed uniform,
// write its action, and accumulate the joint log-probability (the sum of the
// per-head log-softmax values), matching the torch multi-discrete convention.
// For num_atns == 1 this is bit-identical to the single-head sampler.
kernel void puffer_policy_sample_rollout(
    device const float* logits [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device int* actions_out [[buffer(2)]],
    device int* rollout_actions [[buffer(3)]],
    device float* rollout_logprobs [[buffer(4)]],
    device float* rollout_values [[buffer(5)]],
    constant PolicyForwardConfig& config [[buffer(6)]],
    constant PolicySampleConfig& sample_config [[buffer(7)]],
    device const float* decoder_logstd [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= config.batch_size || !valid_config(config) ||
            sample_config.rollout_horizon == 0u ||
            sample_config.rollout_step >= sample_config.rollout_horizon) {
        return;
    }

    uint num_atns = config.num_atns;
    uint logits_base = gid * config.action_size;
    uint rollout_offset = sample_config.rollout_step * config.batch_size + gid;
    uint actions_base = gid * num_atns;
    uint rollout_actions_base = rollout_offset * num_atns;

    // Continuous: each dim is an independent Gaussian N(mean, exp(logstd)). The
    // "logits" are the decoder means; actions are floats stored via bit-reinterpret
    // into the shared action buffers. Log-prob is the summed Gaussian log-density.
    if (config.is_continuous != 0u) {
        const float log_sqrt_2pi = 0.91893853320467274178f;
        float joint_logprob = 0.0f;
        for (uint d = 0u; d < num_atns; ++d) {
            float mean = logits[logits_base + d];
            float std = fast::exp(decoder_logstd[d]);
            float z = puffer_standard_normal(sample_config.seed, sample_config.rollout_step, gid, d);
            float action = mean + std * z;
            float zz = (action - mean) / std;
            joint_logprob += -0.5f * zz * zz - decoder_logstd[d] - log_sqrt_2pi;
            actions_out[actions_base + d] = as_type<int>(action);
            rollout_actions[rollout_actions_base + d] = as_type<int>(action);
        }
        rollout_logprobs[rollout_offset] = joint_logprob;
        rollout_values[rollout_offset] = values[gid];
        return;
    }

    float joint_logprob = 0.0f;
    uint head_offset = 0u;
    for (uint head = 0u; head < num_atns; ++head) {
        uint head_size = config.act_sizes[head];
        uint segment = logits_base + head_offset;

        float max_logit = puffer_safe_logit(logits[segment]);
        for (uint a = 1u; a < head_size; ++a) {
            max_logit = max(max_logit, puffer_safe_logit(logits[segment + a]));
        }

        float denom = 0.0f;
        for (uint a = 0u; a < head_size; ++a) {
            denom += fast::exp(puffer_safe_logit(logits[segment + a]) - max_logit);
        }

        float u = puffer_uniform01(sample_config.seed, sample_config.rollout_step, gid, head);
        int action = int(head_size) - 1;
        float selected_logit = puffer_safe_logit(logits[segment + head_size - 1u]);
        float cumulative = 0.0f;
        for (uint a = 0u; a < head_size; ++a) {
            float logit_a = puffer_safe_logit(logits[segment + a]);
            cumulative += fast::exp(logit_a - max_logit) / denom;
            if (u < cumulative) {
                action = int(a);
                selected_logit = logit_a;
                break;
            }
        }

        joint_logprob += selected_logit - (max_logit + fast::log(denom));
        actions_out[actions_base + head] = action;
        rollout_actions[rollout_actions_base + head] = action;
        head_offset += head_size;
    }

    rollout_logprobs[rollout_offset] = joint_logprob;
    rollout_values[rollout_offset] = values[gid];
}

kernel void puffer_puff_advantage(
    device const float* values [[buffer(0)]],
    device const float* rewards [[buffer(1)]],
    device const float* terminals [[buffer(2)]],
    device const float* ratio [[buffer(3)]],
    device float* advantages [[buffer(4)]],
    constant PuffAdvantageConfig& config [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= config.num_steps || config.horizon < 2u) {
        return;
    }

    uint base = gid * config.horizon;
    advantages[base + config.horizon - 1u] = 0.0f;

    float last = 0.0f;
    for (uint reverse_t = 0u; reverse_t < config.horizon - 1u; ++reverse_t) {
        uint t = config.horizon - 2u - reverse_t;
        uint current = base + t;
        uint next = current + 1u;
        float nextnonterminal = 1.0f - terminals[next];
        float importance = ratio[current];
        float rho_t = min(importance, config.vtrace_rho_clip);
        float c_t = min(importance, config.vtrace_c_clip);
        float delta = rho_t * rewards[next] +
            config.gamma * values[next] * nextnonterminal -
            values[current];
        last = delta + config.gamma * config.gae_lambda * c_t * last * nextnonterminal;
        advantages[current] = last;
    }
}
