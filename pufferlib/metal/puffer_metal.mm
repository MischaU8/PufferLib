#include "puffer_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <thread>
#include <vector>

#include "ocean/breakout/breakout.h"
#include "src/vecenv.h"

static void set_error(char* error, size_t error_len, const char* message);
static id<MTLDevice> get_metal_device();

namespace {
// Breakout's own shape, used by the Breakout-specific native CPU-env scheduler
// (PUFFER_METAL_ROLLOUT=native) which still runs ocean/breakout's c_step.
constexpr uint32_t BREAKOUT_OBS_SIZE = 118;
constexpr uint32_t BREAKOUT_HIDDEN_SIZE = 64;
constexpr uint32_t BREAKOUT_NUM_LAYERS = 2;
constexpr uint32_t BREAKOUT_ACTION_SIZE = 3;

// Upper bound on hidden_size for the resident forward path. Must match
// PUFFER_MAX_HIDDEN_SIZE in kernels/policy.metal, which sizes the thread-local
// scratch arrays in the parity kernels.
constexpr uint32_t PUFFER_MAX_HIDDEN_SIZE = 512;

// The resident path is shape-parametric over obs/hidden/action/num_layers for
// the default PufferLib policy (encoder -> N-layer MinGRU -> decoder + value,
// single discrete action head). MinGRU weights arrive as one contiguous
// gru_weights stack of num_layers slices.
bool supported_resident_shape(const PufferMetalPolicyForwardConfig* config) {
    if (config == nullptr || config->batch_size == 0 ||
            config->obs_size == 0 || config->hidden_size == 0 ||
            config->hidden_size > PUFFER_MAX_HIDDEN_SIZE ||
            config->num_layers < 1 || config->action_size == 0 ||
            config->num_atns < 1 || config->num_atns > PUFFER_METAL_MAX_HEADS) {
        return false;
    }
    if (config->is_continuous != 0) {
        return config->action_size == config->num_atns;
    }
    uint32_t total = 0;
    for (uint32_t h = 0; h < config->num_atns; ++h) {
        if (config->act_sizes[h] == 0) {
            return false;
        }
        total += config->act_sizes[h];
    }
    return total == config->action_size;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
}

struct PufferMetalPolicyContext {
    PufferMetalPolicyContext()
        : device(nil),
          queue(nil),
          serial_pipeline(nil),
          forward_pipeline(nil),
          sample_pipeline(nil),
          advantage_pipeline(nil),
          encoder_pipeline(nil),
          mingru_pipeline(nil),
          add_bias_pipeline(nil),
          mingru_apply_pipeline(nil),
          decoder_pipeline(nil),
          encoder_matmul(nil),
          gru_matmul(nil),
          decoder_matmul(nil),
          value_matmul(nil),
          batch_size(0),
          obs_size(0),
          hidden_size(0),
          action_size(0),
          num_layers(0),
          num_atns(0),
          is_continuous(0),
          rollout_horizon(0),
          advantage_entries(0),
          observations_buffer(nil),
          state_in_buffer(nil),
          encoder_weight_buffer(nil),
          encoder_bias_buffer(nil),
          gru_weights_buffer(nil),
          decoder_weight_buffer(nil),
          decoder_bias_buffer(nil),
          value_weight_buffer(nil),
          value_bias_buffer(nil),
          decoder_logstd_buffer(nil),
          logits_buffer(nil),
          values_buffer(nil),
          actions_buffer(nil),
          rollout_actions_buffer(nil),
          rollout_logprobs_buffer(nil),
          rollout_values_buffer(nil),
          advantage_values_buffer(nil),
          advantage_rewards_buffer(nil),
          advantage_terminals_buffer(nil),
          advantage_ratio_buffer(nil),
          advantage_buffer(nil),
          hidden_a_buffer(nil),
          hidden_b_buffer(nil),
          gru_linear_buffer(nil),
          encoder_columns_buffer(nil),
          decoder_columns_buffer(nil),
          value_columns_buffer(nil),
          config_buffer(nil),
          sample_config_buffer(nil),
          advantage_config_buffer(nil),
          last_command_buffer(nil) {}

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> serial_pipeline;
    id<MTLComputePipelineState> forward_pipeline;
    id<MTLComputePipelineState> sample_pipeline;
    id<MTLComputePipelineState> advantage_pipeline;
    id<MTLComputePipelineState> encoder_pipeline;
    id<MTLComputePipelineState> mingru_pipeline;
    id<MTLComputePipelineState> add_bias_pipeline;
    id<MTLComputePipelineState> mingru_apply_pipeline;
    id<MTLComputePipelineState> decoder_pipeline;
    MPSMatrixMultiplication* encoder_matmul;
    MPSMatrixMultiplication* gru_matmul;
    MPSMatrixMultiplication* decoder_matmul;
    MPSMatrixMultiplication* value_matmul;
    uint32_t batch_size;
    uint32_t obs_size;
    uint32_t hidden_size;
    uint32_t action_size;
    uint32_t num_layers;
    uint32_t num_atns;
    uint32_t is_continuous;
    uint32_t rollout_horizon;
    uint32_t advantage_entries;
    id<MTLBuffer> observations_buffer;
    id<MTLBuffer> state_in_buffer;
    id<MTLBuffer> encoder_weight_buffer;
    id<MTLBuffer> encoder_bias_buffer;
    id<MTLBuffer> gru_weights_buffer;
    id<MTLBuffer> decoder_weight_buffer;
    id<MTLBuffer> decoder_bias_buffer;
    id<MTLBuffer> value_weight_buffer;
    id<MTLBuffer> value_bias_buffer;
    id<MTLBuffer> decoder_logstd_buffer;
    id<MTLBuffer> logits_buffer;
    id<MTLBuffer> values_buffer;
    id<MTLBuffer> actions_buffer;
    id<MTLBuffer> rollout_actions_buffer;
    id<MTLBuffer> rollout_logprobs_buffer;
    id<MTLBuffer> rollout_values_buffer;
    id<MTLBuffer> advantage_values_buffer;
    id<MTLBuffer> advantage_rewards_buffer;
    id<MTLBuffer> advantage_terminals_buffer;
    id<MTLBuffer> advantage_ratio_buffer;
    id<MTLBuffer> advantage_buffer;
    id<MTLBuffer> hidden_a_buffer;
    id<MTLBuffer> hidden_b_buffer;
    id<MTLBuffer> gru_linear_buffer;
    id<MTLBuffer> encoder_columns_buffer;
    id<MTLBuffer> decoder_columns_buffer;
    id<MTLBuffer> value_columns_buffer;
    id<MTLBuffer> config_buffer;
    id<MTLBuffer> sample_config_buffer;
    id<MTLBuffer> advantage_config_buffer;
    id<MTLCommandBuffer> last_command_buffer;
};

struct PufferMetalPolicyNativeWorker {
    uint32_t agent_start;
    uint32_t agent_count;
    uint32_t env_start;
    uint32_t env_count;

    PufferMetalPolicyNativeWorker()
        : agent_start(0),
          agent_count(0),
          env_start(0),
          env_count(0) {}
};

struct PufferMetalPolicyNativeContext {
    PufferMetalPolicyNativeConfig config;
    StaticVec* vec;
    std::string kernel_path;
    PufferMetalPolicyContext* metal;
    std::vector<PufferMetalPolicyNativeWorker> workers;
    std::vector<float> observations;
    std::vector<int32_t> actions_i32;
    std::vector<int32_t> rollout_actions_i32;
    std::vector<float> rollout_logprobs;
    std::vector<float> rollout_values;

    PufferMetalPolicyNativeContext()
        : config{},
          vec(nullptr),
          metal(nullptr) {}
};

static void set_error(char* error, size_t error_len, const char* message) {
    if (error == nullptr || error_len == 0) {
        return;
    }

    if (message == nullptr) {
        message = "unknown Metal error";
    }

    std::snprintf(error, error_len, "%s", message);
}

static void set_ns_error(char* error, size_t error_len, NSString* prefix, NSError* ns_error) {
    NSString* detail = ns_error.localizedDescription ?: @"unknown Metal error";
    NSString* message = [NSString stringWithFormat:@"%@: %@", prefix, detail];
    set_error(error, error_len, message.UTF8String);
}

static id<MTLDevice> get_metal_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (devices.count > 0) {
            device = devices[0];
        }
    }
    return device;
}

static id<MTLComputePipelineState> compile_pipeline_from_library(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* function_name,
    char* error,
    size_t error_len) {
    NSError* ns_error = nil;
    id<MTLFunction> function = [library newFunctionWithName:function_name];
    if (function == nil) {
        NSString* message = [NSString stringWithFormat:@"failed to load %@ Metal kernel", function_name];
        set_error(error, error_len, message.UTF8String);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function
                                                                                 error:&ns_error];
    if (pipeline == nil) {
        set_ns_error(error, error_len, @"failed to create Metal compute pipeline", ns_error);
        return nil;
    }
    return pipeline;
}

static MPSMatrix* make_matrix_at(id<MTLBuffer> buffer, NSUInteger offset, NSUInteger rows, NSUInteger columns) {
    MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
        matrixDescriptorWithRows:rows
                         columns:columns
                        rowBytes:columns * sizeof(float)
                        dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:buffer offset:offset descriptor:descriptor];
}

static MPSMatrix* make_matrix(id<MTLBuffer> buffer, NSUInteger rows, NSUInteger columns) {
    return make_matrix_at(buffer, 0, rows, columns);
}

static bool encode_bias(
    id<MTLCommandBuffer> command_buffer,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> data,
    id<MTLBuffer> bias,
    id<MTLBuffer> columns,
    NSUInteger column_count,
    NSUInteger row_count,
    char* error,
    size_t error_len) {
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        set_error(error, error_len, "failed to create Metal bias command encoder");
        return false;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:data offset:0 atIndex:0];
    [encoder setBuffer:bias offset:0 atIndex:1];
    [encoder setBuffer:columns offset:0 atIndex:2];
    MTLSize threads = MTLSizeMake(column_count, row_count, 1);
    MTLSize threadgroup = MTLSizeMake(std::min<NSUInteger>(column_count, 64), 4, 1);
    [encoder dispatchThreads:threads threadsPerThreadgroup:threadgroup];
    [encoder endEncoding];
    return true;
}

int puffer_metal_policy_create(
    const char* kernel_path,
    PufferMetalPolicyContext** context_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (kernel_path == nullptr || context_out == nullptr) {
            set_error(error, error_len, "invalid Breakout Metal context arguments");
            return 1;
        }

        *context_out = nullptr;

        id<MTLDevice> device = get_metal_device();
        if (device == nil) {
            set_error(error, error_len, "Metal is not supported on this machine");
            return 1;
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            set_error(error, error_len, "failed to create Metal command queue");
            return 1;
        }

        NSString* path = [NSString stringWithUTF8String:kernel_path];
        if (path == nil) {
            set_error(error, error_len, "invalid Metal kernel path");
            return 1;
        }

        NSError* ns_error = nil;
        NSString* source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&ns_error];
        if (source == nil) {
            set_ns_error(error, error_len, @"failed to read Metal kernel source", ns_error);
            return 1;
        }

        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&ns_error];
        if (library == nil) {
            set_ns_error(error, error_len, @"failed to compile Metal kernel source", ns_error);
            return 1;
        }

        id<MTLComputePipelineState> serial_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_forward_eval_serial", error, error_len);
        id<MTLComputePipelineState> forward_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_forward_eval", error, error_len);
        id<MTLComputePipelineState> sample_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_sample_rollout", error, error_len);
        id<MTLComputePipelineState> advantage_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_puff_advantage", error, error_len);
        id<MTLComputePipelineState> encoder_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_encoder", error, error_len);
        id<MTLComputePipelineState> mingru_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_mingru_layer", error, error_len);
        id<MTLComputePipelineState> add_bias_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_add_bias", error, error_len);
        id<MTLComputePipelineState> mingru_apply_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_mingru_apply", error, error_len);
        id<MTLComputePipelineState> decoder_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_decoder", error, error_len);
        if (serial_pipeline == nil || forward_pipeline == nil || sample_pipeline == nil ||
                advantage_pipeline == nil ||
                encoder_pipeline == nil || mingru_pipeline == nil ||
                add_bias_pipeline == nil || mingru_apply_pipeline == nil ||
                decoder_pipeline == nil) {
            return 1;
        }

        PufferMetalPolicyContext* context = new PufferMetalPolicyContext();
        context->device = device;
        context->queue = queue;
        context->serial_pipeline = serial_pipeline;
        context->forward_pipeline = forward_pipeline;
        context->sample_pipeline = sample_pipeline;
        context->advantage_pipeline = advantage_pipeline;
        context->encoder_pipeline = encoder_pipeline;
        context->mingru_pipeline = mingru_pipeline;
        context->add_bias_pipeline = add_bias_pipeline;
        context->mingru_apply_pipeline = mingru_apply_pipeline;
        context->decoder_pipeline = decoder_pipeline;
        *context_out = context;
        return 0;
    }
}

void puffer_metal_policy_destroy(PufferMetalPolicyContext* context) {
    delete context;
}

static PufferMetalPolicyForwardConfig native_forward_config(uint32_t agent_count) {
    PufferMetalPolicyForwardConfig config{};
    config.batch_size = agent_count;
    config.obs_size = BREAKOUT_OBS_SIZE;
    config.hidden_size = BREAKOUT_HIDDEN_SIZE;
    config.num_layers = BREAKOUT_NUM_LAYERS;
    config.action_size = BREAKOUT_ACTION_SIZE;
    config.num_atns = 1;
    config.act_sizes[0] = BREAKOUT_ACTION_SIZE;
    return config;
}

int puffer_metal_policy_native_create(
    const char* kernel_path,
    const PufferMetalPolicyNativeConfig* config,
    PufferMetalPolicyNativeContext** context_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (kernel_path == nullptr || config == nullptr || context_out == nullptr) {
            set_error(error, error_len, "invalid native Breakout Metal context arguments");
            return 1;
        }
        *context_out = nullptr;

        StaticVec* vec = reinterpret_cast<StaticVec*>(config->static_vec_ptr);
        if (vec == nullptr || vec->observations == nullptr || vec->actions == nullptr ||
                vec->rewards == nullptr || vec->terminals == nullptr || vec->envs == nullptr) {
            set_error(error, error_len, "invalid native Breakout StaticVec pointer");
            return 1;
        }
        if (config->total_agents == 0 || config->horizon == 0 ||
                vec->total_agents != static_cast<int>(config->total_agents) ||
                vec->obs_size != static_cast<int>(BREAKOUT_OBS_SIZE) ||
                vec->num_atns != 1 || vec->buffers <= 0) {
            set_error(error, error_len, "unsupported native Breakout rollout shape");
            return 1;
        }

        PufferMetalPolicyNativeContext* native = new PufferMetalPolicyNativeContext();
        native->config = *config;
        native->vec = vec;
        native->kernel_path = kernel_path;
        native->workers.resize(static_cast<size_t>(vec->buffers));
        native->observations.resize(
            static_cast<size_t>(config->total_agents) * BREAKOUT_OBS_SIZE);
        native->actions_i32.resize(config->total_agents);
        native->rollout_actions_i32.resize(
            static_cast<size_t>(config->horizon) * config->total_agents);
        native->rollout_logprobs.resize(
            static_cast<size_t>(config->horizon) * config->total_agents);
        native->rollout_values.resize(
            static_cast<size_t>(config->horizon) * config->total_agents);

        int rc = puffer_metal_policy_create(
            kernel_path, &native->metal, error, error_len);
        if (rc != 0) {
            puffer_metal_policy_native_destroy(native);
            return rc;
        }

        for (int buf = 0; buf < vec->buffers; ++buf) {
            PufferMetalPolicyNativeWorker& worker = native->workers[static_cast<size_t>(buf)];
            worker.agent_start = static_cast<uint32_t>(buf * vec->agents_per_buffer);
            worker.env_start = static_cast<uint32_t>(vec->buffer_env_starts[buf]);
            worker.env_count = static_cast<uint32_t>(vec->buffer_env_counts[buf]);
            uint32_t agent_count = 0;
            for (uint32_t e = 0; e < worker.env_count; ++e) {
                Breakout* env = &reinterpret_cast<Breakout*>(vec->envs)[worker.env_start + e];
                agent_count += static_cast<uint32_t>(env->num_agents);
            }
            worker.agent_count = agent_count;
            if (worker.agent_count == 0 ||
                    worker.agent_start + worker.agent_count > config->total_agents) {
                puffer_metal_policy_native_destroy(native);
                set_error(error, error_len, "invalid native Breakout worker slice");
                return 1;
            }
        }

        *context_out = native;
        return 0;
    }
}

void puffer_metal_policy_native_destroy(PufferMetalPolicyNativeContext* context) {
    if (context == nullptr) {
        return;
    }
    if (context->metal != nullptr) {
        puffer_metal_policy_destroy(context->metal);
        context->metal = nullptr;
    }
    delete context;
}

int puffer_metal_policy_native_rollouts(
    PufferMetalPolicyNativeContext* context,
    const PufferMetalPolicyForwardWeights* weights,
    const float* state_in,
    float* observations_out,
    float* rewards_out,
    float* terminals_out,
    float* actions_out,
    float* logprobs_out,
    float* values_out,
    uint32_t seed,
    PufferMetalPolicyNativeProfile* profile_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || weights == nullptr || state_in == nullptr ||
                observations_out == nullptr || rewards_out == nullptr ||
                terminals_out == nullptr || actions_out == nullptr ||
                logprobs_out == nullptr || values_out == nullptr ||
                profile_out == nullptr) {
            set_error(error, error_len, "invalid native Breakout rollout arguments");
            return 1;
        }

        StaticVec* vec = context->vec;
        if (vec == nullptr || vec->observations == nullptr || vec->actions == nullptr ||
                vec->rewards == nullptr || vec->terminals == nullptr || vec->envs == nullptr) {
            set_error(error, error_len, "native Breakout StaticVec is not loaded");
            return 1;
        }

        *profile_out = PufferMetalPolicyNativeProfile{};
        PufferMetalPolicyForwardConfig forward_config =
            native_forward_config(context->config.total_agents);

        float* vec_obs = reinterpret_cast<float*>(vec->observations);
        const uint32_t horizon = context->config.horizon;
        const uint32_t total_agents = context->config.total_agents;
        const size_t obs_bytes =
            static_cast<size_t>(total_agents) * BREAKOUT_OBS_SIZE * sizeof(float);
        const size_t agent_bytes = static_cast<size_t>(total_agents) * sizeof(float);

        std::memcpy(context->observations.data(), vec_obs, obs_bytes);

        int rc = puffer_metal_policy_load_resident(
            context->metal,
            &forward_config,
            context->observations.data(),
            state_in,
            weights,
            error,
            error_len);
        if (rc != 0) {
            return rc;
        }

        Breakout* envs = reinterpret_cast<Breakout*>(vec->envs);
        for (uint32_t t = 0; t < horizon; ++t) {
            const size_t rollout_agent_offset = static_cast<size_t>(t) * total_agents;
            const size_t rollout_obs_offset =
                static_cast<size_t>(t) * total_agents * BREAKOUT_OBS_SIZE;

            std::memcpy(observations_out + rollout_obs_offset, vec_obs, obs_bytes);
            std::memcpy(rewards_out + rollout_agent_offset, vec->rewards, agent_bytes);
            std::memcpy(terminals_out + rollout_agent_offset, vec->terminals, agent_bytes);

            PufferMetalPolicySampleConfig sample_config{
                t,
                horizon,
                seed,
                0,
            };

            auto start = std::chrono::steady_clock::now();
            rc = puffer_metal_policy_forward_sample_rollout_resident(
                context->metal,
                &forward_config,
                &sample_config,
                error,
                error_len);
            profile_out->policy_sample_seconds += seconds_since(start);
            if (rc != 0) {
                return rc;
            }

            start = std::chrono::steady_clock::now();
            rc = puffer_metal_policy_read_actions_resident(
                context->metal,
                &forward_config,
                context->actions_i32.data(),
                error,
                error_len);
            profile_out->action_read_seconds += seconds_since(start);
            if (rc != 0) {
                return rc;
            }

            for (uint32_t i = 0; i < total_agents; ++i) {
                vec->actions[i] = static_cast<float>(context->actions_i32[i]);
            }

            start = std::chrono::steady_clock::now();
            std::vector<std::thread> threads;
            threads.reserve(context->workers.size());
            for (PufferMetalPolicyNativeWorker& worker : context->workers) {
                threads.emplace_back([&, worker_start = worker.env_start, worker_count = worker.env_count]() {
                    for (uint32_t i = 0; i < worker_count; ++i) {
                        c_step(&envs[worker_start + i]);
                    }
                });
            }
            for (std::thread& thread : threads) {
                thread.join();
            }
            profile_out->cpu_step_seconds += seconds_since(start);

            if (t + 1 < horizon) {
                start = std::chrono::steady_clock::now();
                rc = puffer_metal_policy_write_observations_resident(
                    context->metal,
                    &forward_config,
                    vec_obs,
                    error,
                    error_len);
                profile_out->obs_upload_seconds += seconds_since(start);
                if (rc != 0) {
                    return rc;
                }
            }
        }

        PufferMetalPolicySampleConfig read_config{0, horizon, seed, 0};
        rc = puffer_metal_policy_read_rollout_resident(
            context->metal,
            &forward_config,
            &read_config,
            context->rollout_actions_i32.data(),
            context->rollout_logprobs.data(),
            context->rollout_values.data(),
            error,
            error_len);
        if (rc != 0) {
            return rc;
        }

        const size_t entries = static_cast<size_t>(horizon) * total_agents;
        for (size_t i = 0; i < entries; ++i) {
            actions_out[i] = static_cast<float>(context->rollout_actions_i32[i]);
        }
        std::memcpy(logprobs_out, context->rollout_logprobs.data(), entries * sizeof(float));
        std::memcpy(values_out, context->rollout_values.data(), entries * sizeof(float));

        return 0;
    }
}

int puffer_metal_policy_load_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const float* observations,
    const float* state_in,
    const PufferMetalPolicyForwardWeights* weights,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || observations == nullptr ||
                state_in == nullptr || weights == nullptr) {
            set_error(error, error_len, "invalid Breakout resident load arguments");
            return 1;
        }

        if (weights->encoder_weight == nullptr || weights->encoder_bias == nullptr ||
                weights->gru_weights == nullptr ||
                weights->decoder_weight == nullptr || weights->decoder_bias == nullptr ||
                weights->value_weight == nullptr || weights->value_bias == nullptr) {
            set_error(error, error_len, "invalid forward weight pointers");
            return 1;
        }

        if (!supported_resident_shape(config)) {
            set_error(error, error_len, "unsupported resident forward shape");
            return 1;
        }

        id<MTLDevice> device = context->device;

        const size_t batch = config->batch_size;
        const size_t obs_size = config->obs_size;
        const size_t hidden_size = config->hidden_size;
        const size_t action_size = config->action_size;
        const size_t num_layers = config->num_layers;
        const size_t num_atns = config->num_atns;

        const bool shape_changed =
            context->obs_size != config->obs_size ||
            context->hidden_size != config->hidden_size ||
            context->action_size != config->action_size ||
            context->num_layers != config->num_layers ||
            context->num_atns != config->num_atns ||
            context->is_continuous != config->is_continuous;

        const size_t observations_bytes = batch * obs_size * sizeof(float);
        const size_t state_bytes = num_layers * batch * hidden_size * sizeof(float);
        const size_t encoder_weight_bytes = hidden_size * obs_size * sizeof(float);
        const size_t encoder_bias_bytes = hidden_size * sizeof(float);
        const size_t gru_weights_bytes = num_layers * 3 * hidden_size * hidden_size * sizeof(float);
        const size_t decoder_weight_bytes = action_size * hidden_size * sizeof(float);
        const size_t decoder_bias_bytes = action_size * sizeof(float);
        const size_t value_weight_bytes = hidden_size * sizeof(float);
        const size_t value_bias_bytes = sizeof(float);
        const size_t logstd_bytes = num_atns * sizeof(float);
        const size_t logits_bytes = batch * action_size * sizeof(float);
        const size_t values_bytes = batch * sizeof(float);
        const size_t actions_bytes = batch * num_atns * sizeof(int32_t);
        const size_t hidden_bytes = batch * hidden_size * sizeof(float);
        const size_t gru_linear_bytes = batch * 3 * hidden_size * sizeof(float);

        if (context->encoder_weight_buffer == nil || shape_changed) {
            context->obs_size = config->obs_size;
            context->hidden_size = config->hidden_size;
            context->action_size = config->action_size;
            context->num_layers = config->num_layers;
            context->num_atns = config->num_atns;
            context->is_continuous = config->is_continuous;
            context->encoder_weight_buffer = [device newBufferWithLength:encoder_weight_bytes options:MTLResourceStorageModeShared];
            context->encoder_bias_buffer = [device newBufferWithLength:encoder_bias_bytes options:MTLResourceStorageModeShared];
            context->gru_weights_buffer = [device newBufferWithLength:gru_weights_bytes options:MTLResourceStorageModeShared];
            context->decoder_weight_buffer = [device newBufferWithLength:decoder_weight_bytes options:MTLResourceStorageModeShared];
            context->decoder_bias_buffer = [device newBufferWithLength:decoder_bias_bytes options:MTLResourceStorageModeShared];
            context->value_weight_buffer = [device newBufferWithLength:value_weight_bytes options:MTLResourceStorageModeShared];
            context->value_bias_buffer = [device newBufferWithLength:value_bias_bytes options:MTLResourceStorageModeShared];
            context->decoder_logstd_buffer = [device newBufferWithLength:logstd_bytes options:MTLResourceStorageModeShared];
            uint32_t encoder_columns = static_cast<uint32_t>(hidden_size);
            uint32_t decoder_columns = static_cast<uint32_t>(action_size);
            uint32_t value_columns = 1;
            context->encoder_columns_buffer = [device newBufferWithBytes:&encoder_columns length:sizeof(encoder_columns) options:MTLResourceStorageModeShared];
            context->decoder_columns_buffer = [device newBufferWithBytes:&decoder_columns length:sizeof(decoder_columns) options:MTLResourceStorageModeShared];
            context->value_columns_buffer = [device newBufferWithBytes:&value_columns length:sizeof(value_columns) options:MTLResourceStorageModeShared];
            context->config_buffer = [device newBufferWithLength:sizeof(*config) options:MTLResourceStorageModeShared];
            context->sample_config_buffer = [device newBufferWithLength:sizeof(PufferMetalPolicySampleConfig) options:MTLResourceStorageModeShared];
        }

        if (context->batch_size != config->batch_size || shape_changed ||
                context->observations_buffer == nil) {
            context->batch_size = config->batch_size;
            context->rollout_horizon = 0;
            context->observations_buffer = [device newBufferWithLength:observations_bytes options:MTLResourceStorageModeShared];
            context->state_in_buffer = [device newBufferWithLength:state_bytes options:MTLResourceStorageModeShared];
            context->logits_buffer = [device newBufferWithLength:logits_bytes options:MTLResourceStorageModeShared];
            context->values_buffer = [device newBufferWithLength:values_bytes options:MTLResourceStorageModeShared];
            context->actions_buffer = [device newBufferWithLength:actions_bytes options:MTLResourceStorageModeShared];
            context->rollout_actions_buffer = nil;
            context->rollout_logprobs_buffer = nil;
            context->rollout_values_buffer = nil;
            context->hidden_a_buffer = [device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
            context->hidden_b_buffer = [device newBufferWithLength:hidden_bytes options:MTLResourceStorageModeShared];
            context->gru_linear_buffer = [device newBufferWithLength:gru_linear_bytes options:MTLResourceStorageModeShared];
            context->encoder_matmul = [[MPSMatrixMultiplication alloc]
                initWithDevice:device
                 transposeLeft:NO
                transposeRight:YES
                    resultRows:batch
                 resultColumns:hidden_size
               interiorColumns:obs_size
                         alpha:1.0
                          beta:0.0];
            context->gru_matmul = [[MPSMatrixMultiplication alloc]
                initWithDevice:device
                 transposeLeft:NO
                transposeRight:YES
                    resultRows:batch
                 resultColumns:3 * hidden_size
               interiorColumns:hidden_size
                         alpha:1.0
                          beta:0.0];
            context->decoder_matmul = [[MPSMatrixMultiplication alloc]
                initWithDevice:device
                 transposeLeft:NO
                transposeRight:YES
                    resultRows:batch
                 resultColumns:action_size
               interiorColumns:hidden_size
                         alpha:1.0
                          beta:0.0];
            context->value_matmul = [[MPSMatrixMultiplication alloc]
                initWithDevice:device
                 transposeLeft:NO
                transposeRight:YES
                    resultRows:batch
                 resultColumns:1
               interiorColumns:hidden_size
                         alpha:1.0
                          beta:0.0];
        }

        id<MTLBuffer> observations_buffer = context->observations_buffer;
        id<MTLBuffer> state_in_buffer = context->state_in_buffer;
        id<MTLBuffer> encoder_weight_buffer = context->encoder_weight_buffer;
        id<MTLBuffer> encoder_bias_buffer = context->encoder_bias_buffer;
        id<MTLBuffer> gru_weights_buffer = context->gru_weights_buffer;
        id<MTLBuffer> decoder_weight_buffer = context->decoder_weight_buffer;
        id<MTLBuffer> decoder_bias_buffer = context->decoder_bias_buffer;
        id<MTLBuffer> value_weight_buffer = context->value_weight_buffer;
        id<MTLBuffer> value_bias_buffer = context->value_bias_buffer;
        id<MTLBuffer> decoder_logstd_buffer = context->decoder_logstd_buffer;
        id<MTLBuffer> logits_buffer = context->logits_buffer;
        id<MTLBuffer> values_buffer = context->values_buffer;
        id<MTLBuffer> actions_buffer = context->actions_buffer;
        id<MTLBuffer> hidden_a_buffer = context->hidden_a_buffer;
        id<MTLBuffer> hidden_b_buffer = context->hidden_b_buffer;
        id<MTLBuffer> gru_linear_buffer = context->gru_linear_buffer;
        id<MTLBuffer> encoder_columns_buffer = context->encoder_columns_buffer;
        id<MTLBuffer> decoder_columns_buffer = context->decoder_columns_buffer;
        id<MTLBuffer> value_columns_buffer = context->value_columns_buffer;
        id<MTLBuffer> config_buffer = context->config_buffer;
        id<MTLBuffer> sample_config_buffer = context->sample_config_buffer;

        if (observations_buffer == nil || state_in_buffer == nil ||
                encoder_weight_buffer == nil || encoder_bias_buffer == nil ||
                gru_weights_buffer == nil ||
                decoder_weight_buffer == nil || decoder_bias_buffer == nil ||
                value_weight_buffer == nil || value_bias_buffer == nil ||
                decoder_logstd_buffer == nil ||
                logits_buffer == nil || values_buffer == nil || actions_buffer == nil ||
                hidden_a_buffer == nil || hidden_b_buffer == nil ||
                gru_linear_buffer == nil ||
                encoder_columns_buffer == nil || decoder_columns_buffer == nil ||
                value_columns_buffer == nil ||
                config_buffer == nil || sample_config_buffer == nil) {
            set_error(error, error_len, "failed to allocate resident Metal buffers");
            return 1;
        }

        std::memcpy(observations_buffer.contents, observations, observations_bytes);
        std::memcpy(state_in_buffer.contents, state_in, state_bytes);
        std::memcpy(encoder_weight_buffer.contents, weights->encoder_weight, encoder_weight_bytes);
        std::memcpy(encoder_bias_buffer.contents, weights->encoder_bias, encoder_bias_bytes);
        std::memcpy(gru_weights_buffer.contents, weights->gru_weights, gru_weights_bytes);
        std::memcpy(decoder_weight_buffer.contents, weights->decoder_weight, decoder_weight_bytes);
        std::memcpy(decoder_bias_buffer.contents, weights->decoder_bias, decoder_bias_bytes);
        std::memcpy(value_weight_buffer.contents, weights->value_weight, value_weight_bytes);
        std::memcpy(value_bias_buffer.contents, weights->value_bias, value_bias_bytes);
        if (config->is_continuous != 0 && weights->decoder_logstd != nullptr) {
            std::memcpy(decoder_logstd_buffer.contents, weights->decoder_logstd, logstd_bytes);
        }
        std::memcpy(config_buffer.contents, config, sizeof(*config));
        return 0;
    }
}

int puffer_metal_policy_write_observations_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const float* observations,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || observations == nullptr) {
            set_error(error, error_len, "invalid Breakout observation write arguments");
            return 1;
        }

        if (context->batch_size != config->batch_size || !supported_resident_shape(config) ||
                context->obs_size != config->obs_size ||
                context->observations_buffer == nil) {
            set_error(error, error_len, "resident observation buffer is not loaded");
            return 1;
        }

        const size_t observations_bytes = static_cast<size_t>(config->batch_size) * config->obs_size * sizeof(float);
        std::memcpy(context->observations_buffer.contents, observations, observations_bytes);
        return 0;
    }
}

int puffer_metal_policy_forward_resident_async(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr) {
            set_error(error, error_len, "invalid Breakout resident forward arguments");
            return 1;
        }

        if (!supported_resident_shape(config)) {
            set_error(error, error_len, "unsupported resident forward shape");
            return 1;
        }

        id<MTLCommandQueue> queue = context->queue;
        id<MTLComputePipelineState> serial_pipeline = context->serial_pipeline;
        id<MTLComputePipelineState> forward_pipeline = context->forward_pipeline;
        id<MTLComputePipelineState> encoder_pipeline = context->encoder_pipeline;
        id<MTLComputePipelineState> mingru_pipeline = context->mingru_pipeline;
        id<MTLComputePipelineState> add_bias_pipeline = context->add_bias_pipeline;
        id<MTLComputePipelineState> mingru_apply_pipeline = context->mingru_apply_pipeline;
        id<MTLComputePipelineState> decoder_pipeline = context->decoder_pipeline;
        MPSMatrixMultiplication* encoder_matmul = context->encoder_matmul;
        MPSMatrixMultiplication* gru_matmul = context->gru_matmul;
        MPSMatrixMultiplication* decoder_matmul = context->decoder_matmul;
        MPSMatrixMultiplication* value_matmul = context->value_matmul;

        const size_t batch = config->batch_size;
        const size_t obs_size = config->obs_size;
        const size_t hidden_size = config->hidden_size;
        const size_t action_size = config->action_size;
        const size_t num_layers = config->num_layers;
        id<MTLBuffer> observations_buffer = context->observations_buffer;
        id<MTLBuffer> state_in_buffer = context->state_in_buffer;
        id<MTLBuffer> encoder_weight_buffer = context->encoder_weight_buffer;
        id<MTLBuffer> encoder_bias_buffer = context->encoder_bias_buffer;
        id<MTLBuffer> gru_weights_buffer = context->gru_weights_buffer;
        id<MTLBuffer> decoder_weight_buffer = context->decoder_weight_buffer;
        id<MTLBuffer> decoder_bias_buffer = context->decoder_bias_buffer;
        id<MTLBuffer> value_weight_buffer = context->value_weight_buffer;
        id<MTLBuffer> value_bias_buffer = context->value_bias_buffer;
        id<MTLBuffer> logits_buffer = context->logits_buffer;
        id<MTLBuffer> values_buffer = context->values_buffer;
        id<MTLBuffer> hidden_a_buffer = context->hidden_a_buffer;
        id<MTLBuffer> hidden_b_buffer = context->hidden_b_buffer;
        id<MTLBuffer> gru_linear_buffer = context->gru_linear_buffer;
        id<MTLBuffer> encoder_columns_buffer = context->encoder_columns_buffer;
        id<MTLBuffer> decoder_columns_buffer = context->decoder_columns_buffer;
        id<MTLBuffer> value_columns_buffer = context->value_columns_buffer;
        id<MTLBuffer> config_buffer = context->config_buffer;

        if (queue == nil || serial_pipeline == nil || forward_pipeline == nil ||
                encoder_pipeline == nil || mingru_pipeline == nil ||
                add_bias_pipeline == nil || mingru_apply_pipeline == nil ||
                decoder_pipeline == nil ||
                observations_buffer == nil || state_in_buffer == nil ||
                encoder_weight_buffer == nil || encoder_bias_buffer == nil ||
                gru_weights_buffer == nil ||
                decoder_weight_buffer == nil || decoder_bias_buffer == nil ||
                value_weight_buffer == nil || value_bias_buffer == nil ||
                logits_buffer == nil || values_buffer == nil ||
                hidden_a_buffer == nil || hidden_b_buffer == nil ||
                gru_linear_buffer == nil ||
                encoder_columns_buffer == nil || decoder_columns_buffer == nil ||
                value_columns_buffer == nil ||
                config_buffer == nil ||
                context->batch_size != config->batch_size) {
            set_error(error, error_len, "resident buffers are not loaded");
            return 1;
        }

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        if (command_buffer == nil) {
            set_error(error, error_len, "failed to create Metal command buffer");
            return 1;
        }

        if (batch >= 1024) {
            if (encoder_matmul == nil || gru_matmul == nil ||
                    decoder_matmul == nil || value_matmul == nil) {
                set_error(error, error_len, "Breakout MPS matmul objects are not loaded");
                return 1;
            }

            const size_t layer_stride_bytes = 3 * hidden_size * hidden_size * sizeof(float);

            MPSMatrix* observations = make_matrix(observations_buffer, batch, obs_size);
            MPSMatrix* encoder_weight = make_matrix(
                encoder_weight_buffer, hidden_size, obs_size);
            MPSMatrix* encoder_hidden = make_matrix(hidden_a_buffer, batch, hidden_size);
            [encoder_matmul encodeToCommandBuffer:command_buffer
                                       leftMatrix:observations
                                      rightMatrix:encoder_weight
                                     resultMatrix:encoder_hidden];
            if (!encode_bias(
                    command_buffer,
                    add_bias_pipeline,
                    hidden_a_buffer,
                    encoder_bias_buffer,
                    encoder_columns_buffer,
                    hidden_size,
                    batch,
                    error,
                    error_len)) {
                return 1;
            }

            // Ping-pong the MinGRU stack between hidden_a and hidden_b. Each
            // layer reads its weight slice from the contiguous gru_weights
            // buffer; the layer index is passed inline via setBytes.
            id<MTLBuffer> layer_in_buffer = hidden_a_buffer;
            id<MTLBuffer> layer_out_buffer = hidden_b_buffer;
            MTLSize hidden_threads = MTLSizeMake(hidden_size, batch, 1);
            MTLSize hidden_threadgroup = MTLSizeMake(16, 16, 1);
            for (uint32_t layer = 0; layer < num_layers; ++layer) {
                MPSMatrix* layer_input = make_matrix(layer_in_buffer, batch, hidden_size);
                MPSMatrix* gru_weight = make_matrix_at(
                    gru_weights_buffer, layer * layer_stride_bytes, 3 * hidden_size, hidden_size);
                MPSMatrix* gru_linear = make_matrix(gru_linear_buffer, batch, 3 * hidden_size);
                [gru_matmul encodeToCommandBuffer:command_buffer
                                       leftMatrix:layer_input
                                      rightMatrix:gru_weight
                                     resultMatrix:gru_linear];

                id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                if (encoder == nil) {
                    set_error(error, error_len, "failed to create Metal command encoder");
                    return 1;
                }
                [encoder setComputePipelineState:mingru_apply_pipeline];
                [encoder setBuffer:layer_in_buffer offset:0 atIndex:0];
                [encoder setBuffer:gru_linear_buffer offset:0 atIndex:1];
                [encoder setBuffer:state_in_buffer offset:0 atIndex:2];
                [encoder setBuffer:layer_out_buffer offset:0 atIndex:3];
                [encoder setBytes:&layer length:sizeof(layer) atIndex:4];
                [encoder setBuffer:config_buffer offset:0 atIndex:5];
                [encoder dispatchThreads:hidden_threads threadsPerThreadgroup:hidden_threadgroup];
                [encoder endEncoding];

                id<MTLBuffer> swap = layer_in_buffer;
                layer_in_buffer = layer_out_buffer;
                layer_out_buffer = swap;
            }

            // After the loop the final hidden activations live in layer_in_buffer.
            MPSMatrix* final_hidden = make_matrix(layer_in_buffer, batch, hidden_size);
            MPSMatrix* decoder_weight = make_matrix(
                decoder_weight_buffer, action_size, hidden_size);
            MPSMatrix* logits = make_matrix(logits_buffer, batch, action_size);
            [decoder_matmul encodeToCommandBuffer:command_buffer
                                       leftMatrix:final_hidden
                                      rightMatrix:decoder_weight
                                     resultMatrix:logits];
            if (!encode_bias(
                    command_buffer,
                    add_bias_pipeline,
                    logits_buffer,
                    decoder_bias_buffer,
                    decoder_columns_buffer,
                    action_size,
                    batch,
                    error,
                    error_len)) {
                return 1;
            }

            MPSMatrix* value_weight = make_matrix(
                value_weight_buffer, 1, hidden_size);
            MPSMatrix* values = make_matrix(values_buffer, batch, 1);
            [value_matmul encodeToCommandBuffer:command_buffer
                                     leftMatrix:final_hidden
                                    rightMatrix:value_weight
                                   resultMatrix:values];
            if (!encode_bias(
                    command_buffer,
                    add_bias_pipeline,
                    values_buffer,
                    value_bias_buffer,
                    value_columns_buffer,
                    1,
                    batch,
                    error,
                    error_len)) {
                return 1;
            }
        } else {
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            if (encoder == nil) {
                set_error(error, error_len, "failed to create Metal command encoder");
                return 1;
            }
            [encoder setComputePipelineState:forward_pipeline];
            [encoder setBuffer:observations_buffer offset:0 atIndex:0];
            [encoder setBuffer:state_in_buffer offset:0 atIndex:1];
            [encoder setBuffer:encoder_weight_buffer offset:0 atIndex:2];
            [encoder setBuffer:encoder_bias_buffer offset:0 atIndex:3];
            [encoder setBuffer:gru_weights_buffer offset:0 atIndex:4];
            [encoder setBuffer:decoder_weight_buffer offset:0 atIndex:5];
            [encoder setBuffer:decoder_bias_buffer offset:0 atIndex:6];
            [encoder setBuffer:value_weight_buffer offset:0 atIndex:7];
            [encoder setBuffer:value_bias_buffer offset:0 atIndex:8];
            [encoder setBuffer:logits_buffer offset:0 atIndex:9];
            [encoder setBuffer:values_buffer offset:0 atIndex:10];
            [encoder setBuffer:state_in_buffer offset:0 atIndex:11];
            [encoder setBuffer:config_buffer offset:0 atIndex:12];
            MTLSize threadgroups = MTLSizeMake(batch, 1, 1);
            NSUInteger tg_width = std::max<NSUInteger>(hidden_size, action_size + 1);
            tg_width = std::min<NSUInteger>(tg_width, forward_pipeline.maxTotalThreadsPerThreadgroup);
            MTLSize threadgroup = MTLSizeMake(tg_width, 1, 1);
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroup];
            [encoder endEncoding];
        }

        [command_buffer commit];
        context->last_command_buffer = command_buffer;
        return 0;
    }
}

int puffer_metal_policy_sample_rollout_resident_async(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || sample_config == nullptr) {
            set_error(error, error_len, "invalid Breakout resident sample arguments");
            return 1;
        }

        if (!supported_resident_shape(config) || sample_config->rollout_horizon == 0 ||
                sample_config->rollout_step >= sample_config->rollout_horizon) {
            set_error(error, error_len, "unsupported resident sample shape");
            return 1;
        }

        id<MTLDevice> device = context->device;
        id<MTLCommandQueue> queue = context->queue;
        id<MTLComputePipelineState> sample_pipeline = context->sample_pipeline;
        id<MTLBuffer> logits_buffer = context->logits_buffer;
        id<MTLBuffer> values_buffer = context->values_buffer;
        id<MTLBuffer> actions_buffer = context->actions_buffer;
        id<MTLBuffer> config_buffer = context->config_buffer;
        id<MTLBuffer> sample_config_buffer = context->sample_config_buffer;
        id<MTLBuffer> decoder_logstd_buffer = context->decoder_logstd_buffer;

        if (device == nil || queue == nil || sample_pipeline == nil ||
                logits_buffer == nil || values_buffer == nil ||
                actions_buffer == nil || config_buffer == nil ||
                sample_config_buffer == nil || decoder_logstd_buffer == nil ||
                context->batch_size != config->batch_size) {
            set_error(error, error_len, "Breakout resident buffers are not loaded");
            return 1;
        }

        const size_t batch = config->batch_size;
        const size_t num_atns = config->num_atns;
        const size_t horizon = sample_config->rollout_horizon;
        const size_t rollout_actions_bytes = horizon * batch * num_atns * sizeof(int32_t);
        const size_t rollout_values_bytes = horizon * batch * sizeof(float);

        if (context->rollout_horizon != sample_config->rollout_horizon ||
                context->rollout_actions_buffer == nil) {
            context->rollout_horizon = sample_config->rollout_horizon;
            context->rollout_actions_buffer = [device newBufferWithLength:rollout_actions_bytes options:MTLResourceStorageModeShared];
            context->rollout_logprobs_buffer = [device newBufferWithLength:rollout_values_bytes options:MTLResourceStorageModeShared];
            context->rollout_values_buffer = [device newBufferWithLength:rollout_values_bytes options:MTLResourceStorageModeShared];
        }

        id<MTLBuffer> rollout_actions_buffer = context->rollout_actions_buffer;
        id<MTLBuffer> rollout_logprobs_buffer = context->rollout_logprobs_buffer;
        id<MTLBuffer> rollout_values_buffer = context->rollout_values_buffer;
        if (rollout_actions_buffer == nil || rollout_logprobs_buffer == nil ||
                rollout_values_buffer == nil) {
            set_error(error, error_len, "failed to allocate Breakout rollout Metal buffers");
            return 1;
        }

        std::memcpy(sample_config_buffer.contents, sample_config, sizeof(*sample_config));

        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            set_error(error, error_len, "failed to create Metal sample command encoder");
            return 1;
        }

        [encoder setComputePipelineState:sample_pipeline];
        [encoder setBuffer:logits_buffer offset:0 atIndex:0];
        [encoder setBuffer:values_buffer offset:0 atIndex:1];
        [encoder setBuffer:actions_buffer offset:0 atIndex:2];
        [encoder setBuffer:rollout_actions_buffer offset:0 atIndex:3];
        [encoder setBuffer:rollout_logprobs_buffer offset:0 atIndex:4];
        [encoder setBuffer:rollout_values_buffer offset:0 atIndex:5];
        [encoder setBuffer:config_buffer offset:0 atIndex:6];
        [encoder setBuffer:sample_config_buffer offset:0 atIndex:7];
        [encoder setBuffer:decoder_logstd_buffer offset:0 atIndex:8];

        MTLSize threads = MTLSizeMake(batch, 1, 1);
        MTLSize threadgroup = MTLSizeMake(std::min<NSUInteger>(sample_pipeline.maxTotalThreadsPerThreadgroup, 256), 1, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [command_buffer commit];
        context->last_command_buffer = command_buffer;
        return 0;
    }
}

int puffer_metal_policy_sample_rollout_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    char* error,
    size_t error_len) {
    int rc = puffer_metal_policy_sample_rollout_resident_async(
        context, config, sample_config, error, error_len);
    if (rc != 0) {
        return rc;
    }

    return puffer_metal_policy_synchronize(context, error, error_len);
}

int puffer_metal_policy_load_sample_logits_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const float* logits,
    const float* values,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || logits == nullptr || values == nullptr) {
            set_error(error, error_len, "invalid Breakout sample logits arguments");
            return 1;
        }

        if (!supported_resident_shape(config)) {
            set_error(error, error_len, "unsupported sample logits shape");
            return 1;
        }

        if (context->batch_size != config->batch_size ||
                context->logits_buffer == nil || context->values_buffer == nil) {
            set_error(error, error_len, "Breakout resident buffers are not loaded");
            return 1;
        }

        const size_t logits_bytes =
            static_cast<size_t>(config->batch_size) * config->action_size * sizeof(float);
        const size_t values_bytes = static_cast<size_t>(config->batch_size) * sizeof(float);
        std::memcpy(context->logits_buffer.contents, logits, logits_bytes);
        std::memcpy(context->values_buffer.contents, values, values_bytes);
        return 0;
    }
}

int puffer_metal_policy_forward_sample_rollout_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    char* error,
    size_t error_len) {
    int rc = puffer_metal_policy_forward_resident_async(
        context, config, error, error_len);
    if (rc != 0) {
        return rc;
    }

    rc = puffer_metal_policy_sample_rollout_resident_async(
        context, config, sample_config, error, error_len);
    if (rc != 0) {
        return rc;
    }

    return puffer_metal_policy_synchronize(context, error, error_len);
}

int puffer_metal_policy_synchronize(
    PufferMetalPolicyContext* context,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || context->last_command_buffer == nil) {
            return 0;
        }

        id<MTLCommandBuffer> command_buffer = context->last_command_buffer;
        [command_buffer waitUntilCompleted];

        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
            set_ns_error(error, error_len, @"Metal command buffer failed", command_buffer.error);
            return 1;
        }

        context->last_command_buffer = nil;
        return 0;
    }
}

int puffer_metal_policy_forward_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    char* error,
    size_t error_len) {
    int rc = puffer_metal_policy_forward_resident_async(
        context, config, error, error_len);
    if (rc != 0) {
        return rc;
    }

    return puffer_metal_policy_synchronize(context, error, error_len);
}

int puffer_metal_policy_read_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    float* logits_out,
    float* values_out,
    float* state_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || logits_out == nullptr ||
                values_out == nullptr || state_out == nullptr) {
            set_error(error, error_len, "invalid Breakout resident read arguments");
            return 1;
        }

        if (context->batch_size != config->batch_size || !supported_resident_shape(config)) {
            set_error(error, error_len, "unsupported resident read shape");
            return 1;
        }

        const size_t hidden_size = config->hidden_size;
        const size_t action_size = config->action_size;
        const size_t num_layers = config->num_layers;
        const size_t batch = config->batch_size;
        const size_t logits_bytes = batch * action_size * sizeof(float);
        const size_t values_bytes = batch * sizeof(float);
        const size_t state_bytes = num_layers * batch * hidden_size * sizeof(float);

        if (context->logits_buffer == nil || context->values_buffer == nil ||
                context->state_in_buffer == nil) {
            set_error(error, error_len, "Breakout resident buffers are not loaded");
            return 1;
        }

        std::memcpy(logits_out, context->logits_buffer.contents, logits_bytes);
        std::memcpy(values_out, context->values_buffer.contents, values_bytes);
        std::memcpy(state_out, context->state_in_buffer.contents, state_bytes);
        return 0;
    }
}

int puffer_metal_policy_read_actions_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    int32_t* actions_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || actions_out == nullptr) {
            set_error(error, error_len, "invalid Breakout action read arguments");
            return 1;
        }

        if (context->batch_size != config->batch_size || !supported_resident_shape(config) ||
                context->actions_buffer == nil) {
            set_error(error, error_len, "unsupported resident action read shape");
            return 1;
        }

        const size_t actions_bytes = static_cast<size_t>(config->batch_size) * config->num_atns * sizeof(int32_t);
        std::memcpy(actions_out, context->actions_buffer.contents, actions_bytes);
        return 0;
    }
}

int puffer_metal_policy_read_rollout_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    int32_t* actions_out,
    float* logprobs_out,
    float* values_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || sample_config == nullptr ||
                actions_out == nullptr || logprobs_out == nullptr || values_out == nullptr) {
            set_error(error, error_len, "invalid Breakout rollout read arguments");
            return 1;
        }

        if (context->batch_size != config->batch_size || !supported_resident_shape(config) ||
                sample_config->rollout_horizon == 0 ||
                context->rollout_horizon != sample_config->rollout_horizon ||
                context->rollout_actions_buffer == nil ||
                context->rollout_logprobs_buffer == nil ||
                context->rollout_values_buffer == nil) {
            set_error(error, error_len, "unsupported Breakout rollout read shape");
            return 1;
        }

        const size_t entries = static_cast<size_t>(sample_config->rollout_horizon) * config->batch_size;
        const size_t actions_bytes = entries * config->num_atns * sizeof(int32_t);
        const size_t values_bytes = entries * sizeof(float);
        std::memcpy(actions_out, context->rollout_actions_buffer.contents, actions_bytes);
        std::memcpy(logprobs_out, context->rollout_logprobs_buffer.contents, values_bytes);
        std::memcpy(values_out, context->rollout_values_buffer.contents, values_bytes);
        return 0;
    }
}

int puffer_metal_policy_load_advantage_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPuffAdvantageConfig* config,
    const float* values,
    const float* rewards,
    const float* terminals,
    const float* ratio,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || values == nullptr ||
                rewards == nullptr || terminals == nullptr || ratio == nullptr) {
            set_error(error, error_len, "invalid Puff advantage load arguments");
            return 1;
        }

        if (config->num_steps == 0 || config->horizon < 2) {
            set_error(error, error_len, "unsupported Puff advantage shape");
            return 1;
        }

        id<MTLDevice> device = context->device;
        if (device == nil) {
            set_error(error, error_len, "Metal is not supported on this machine");
            return 1;
        }

        const size_t entries = static_cast<size_t>(config->num_steps) * config->horizon;
        const size_t bytes = entries * sizeof(float);
        if (entries > static_cast<size_t>(UINT32_MAX)) {
            set_error(error, error_len, "Puff advantage shape is too large");
            return 1;
        }

        if (context->advantage_entries != entries || context->advantage_values_buffer == nil) {
            context->advantage_entries = static_cast<uint32_t>(entries);
            context->advantage_values_buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context->advantage_rewards_buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context->advantage_terminals_buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context->advantage_ratio_buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context->advantage_buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            context->advantage_config_buffer = [device newBufferWithLength:sizeof(*config) options:MTLResourceStorageModeShared];
        }

        if (context->advantage_values_buffer == nil ||
                context->advantage_rewards_buffer == nil ||
                context->advantage_terminals_buffer == nil ||
                context->advantage_ratio_buffer == nil ||
                context->advantage_buffer == nil ||
                context->advantage_config_buffer == nil) {
            set_error(error, error_len, "failed to allocate Puff advantage Metal buffers");
            return 1;
        }

        std::memcpy(context->advantage_values_buffer.contents, values, bytes);
        std::memcpy(context->advantage_rewards_buffer.contents, rewards, bytes);
        std::memcpy(context->advantage_terminals_buffer.contents, terminals, bytes);
        std::memcpy(context->advantage_ratio_buffer.contents, ratio, bytes);
        std::memset(context->advantage_buffer.contents, 0, bytes);
        std::memcpy(context->advantage_config_buffer.contents, config, sizeof(*config));
        return 0;
    }
}

int puffer_metal_policy_compute_advantage_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPuffAdvantageConfig* config,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr) {
            set_error(error, error_len, "invalid Puff advantage compute arguments");
            return 1;
        }

        if (config->num_steps == 0 || config->horizon < 2) {
            set_error(error, error_len, "unsupported Puff advantage shape");
            return 1;
        }

        const size_t entries = static_cast<size_t>(config->num_steps) * config->horizon;
        if (context->queue == nil || context->advantage_pipeline == nil ||
                context->advantage_entries != entries ||
                context->advantage_values_buffer == nil ||
                context->advantage_rewards_buffer == nil ||
                context->advantage_terminals_buffer == nil ||
                context->advantage_ratio_buffer == nil ||
                context->advantage_buffer == nil ||
                context->advantage_config_buffer == nil) {
            set_error(error, error_len, "Puff advantage resident buffers are not loaded");
            return 1;
        }

        std::memcpy(context->advantage_config_buffer.contents, config, sizeof(*config));

        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            set_error(error, error_len, "failed to create Metal advantage command encoder");
            return 1;
        }

        [encoder setComputePipelineState:context->advantage_pipeline];
        [encoder setBuffer:context->advantage_values_buffer offset:0 atIndex:0];
        [encoder setBuffer:context->advantage_rewards_buffer offset:0 atIndex:1];
        [encoder setBuffer:context->advantage_terminals_buffer offset:0 atIndex:2];
        [encoder setBuffer:context->advantage_ratio_buffer offset:0 atIndex:3];
        [encoder setBuffer:context->advantage_buffer offset:0 atIndex:4];
        [encoder setBuffer:context->advantage_config_buffer offset:0 atIndex:5];

        MTLSize threads = MTLSizeMake(config->num_steps, 1, 1);
        MTLSize threadgroup = MTLSizeMake(std::min<NSUInteger>(
            context->advantage_pipeline.maxTotalThreadsPerThreadgroup, 256), 1, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [command_buffer commit];
        context->last_command_buffer = command_buffer;
        return puffer_metal_policy_synchronize(context, error, error_len);
    }
}

int puffer_metal_policy_read_advantage_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPuffAdvantageConfig* config,
    float* advantages_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || advantages_out == nullptr) {
            set_error(error, error_len, "invalid Puff advantage read arguments");
            return 1;
        }

        if (config->num_steps == 0 || config->horizon < 2) {
            set_error(error, error_len, "unsupported Puff advantage shape");
            return 1;
        }

        const size_t entries = static_cast<size_t>(config->num_steps) * config->horizon;
        if (context->advantage_entries != entries || context->advantage_buffer == nil) {
            set_error(error, error_len, "Puff advantage resident buffers are not loaded");
            return 1;
        }

        std::memcpy(advantages_out, context->advantage_buffer.contents, entries * sizeof(float));
        return 0;
    }
}

int puffer_metal_policy_advantage(
    PufferMetalPolicyContext* context,
    const PufferMetalPuffAdvantageConfig* config,
    const float* values,
    const float* rewards,
    const float* terminals,
    const float* ratio,
    float* advantages_out,
    char* error,
    size_t error_len) {
    int rc = puffer_metal_policy_load_advantage_resident(
        context, config, values, rewards, terminals, ratio, error, error_len);
    if (rc != 0) {
        return rc;
    }

    rc = puffer_metal_policy_compute_advantage_resident(context, config, error, error_len);
    if (rc != 0) {
        return rc;
    }

    return puffer_metal_policy_read_advantage_resident(
        context, config, advantages_out, error, error_len);
}

int puffer_metal_policy_forward_context(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const float* observations,
    const float* state_in,
    const PufferMetalPolicyForwardWeights* weights,
    float* logits_out,
    float* values_out,
    float* state_out,
    char* error,
    size_t error_len) {
    int rc = puffer_metal_policy_load_resident(
        context, config, observations, state_in, weights, error, error_len);
    if (rc != 0) {
        return rc;
    }

    rc = puffer_metal_policy_forward_resident(context, config, error, error_len);
    if (rc != 0) {
        return rc;
    }

    return puffer_metal_policy_read_resident(
        context, config, logits_out, values_out, state_out, error, error_len);
}

int puffer_metal_policy_forward(
    const char* kernel_path,
    const PufferMetalPolicyForwardConfig* config,
    const float* observations,
    const float* state_in,
    const PufferMetalPolicyForwardWeights* weights,
    float* logits_out,
    float* values_out,
    float* state_out,
    char* error,
    size_t error_len) {
    PufferMetalPolicyContext* context = nullptr;
    int rc = puffer_metal_policy_create(kernel_path, &context, error, error_len);
    if (rc != 0) {
        return rc;
    }
    rc = puffer_metal_policy_forward_context(
        context,
        config,
        observations,
        state_in,
        weights,
        logits_out,
        values_out,
        state_out,
        error,
        error_len);
    puffer_metal_policy_destroy(context);
    return rc;
}
