#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PUFFER_METAL_MAX_HEADS 8

typedef struct PufferMetalPolicyForwardConfig {
    uint32_t batch_size;
    uint32_t obs_size;
    uint32_t hidden_size;
    uint32_t num_layers;
    uint32_t action_size;  // discrete: sum(act_sizes); continuous: num dims
    uint32_t num_atns;     // discrete heads, or continuous action dims
    uint32_t act_sizes[PUFFER_METAL_MAX_HEADS];
    uint32_t is_continuous;  // 0 = discrete heads, 1 = Gaussian continuous
} PufferMetalPolicyForwardConfig;

typedef struct PufferMetalPolicyForwardWeights {
    const float* encoder_weight;
    const float* encoder_bias;
    // Contiguous stack of all MinGRU layer weights, laid out
    // [num_layers][3 * hidden_size][hidden_size].
    const float* gru_weights;
    const float* decoder_weight;   // continuous: decoder_mean.weight
    const float* decoder_bias;     // continuous: decoder_mean.bias
    const float* value_weight;
    const float* value_bias;
    const float* decoder_logstd;   // continuous only ([num_atns]); NULL otherwise
} PufferMetalPolicyForwardWeights;

typedef struct PufferMetalPolicySampleConfig {
    uint32_t rollout_step;
    uint32_t rollout_horizon;
    uint32_t seed;
    uint32_t has_mask;
} PufferMetalPolicySampleConfig;

typedef struct PufferMetalPolicyContext PufferMetalPolicyContext;

int puffer_metal_policy_create(
    const char* kernel_path,
    PufferMetalPolicyContext** context_out,
    char* error,
    size_t error_len);

void puffer_metal_policy_destroy(PufferMetalPolicyContext* context);

int puffer_metal_policy_load_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const float* observations,
    const float* state_in,
    const PufferMetalPolicyForwardWeights* weights,
    char* error,
    size_t error_len);

int puffer_metal_policy_write_observations_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const float* observations,
    char* error,
    size_t error_len);

int puffer_metal_policy_forward_sample_rollout_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    const uint8_t* action_mask,
    char* error,
    size_t error_len);

int puffer_metal_policy_read_actions_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    int32_t* actions_out,
    char* error,
    size_t error_len);

int puffer_metal_policy_read_rollout_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    int32_t* actions_out,
    float* logprobs_out,
    float* values_out,
    char* error,
    size_t error_len);

#ifdef __cplusplus
}
#endif
