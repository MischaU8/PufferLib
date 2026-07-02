#include "puffer_metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

static void set_error(char* error, size_t error_len, const char* message);
static id<MTLDevice> get_metal_device();
static int forward_resident_async(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    char* error,
    size_t error_len);
static int sample_rollout_resident_async(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    const uint8_t* action_mask,
    char* error,
    size_t error_len);
static int synchronize_last_command_buffer(
    PufferMetalPolicyContext* context,
    char* error,
    size_t error_len);

namespace {
// Upper bound on hidden_size for the resident forward path. Must match
// PUFFER_MAX_HIDDEN_SIZE in kernels/policy.metal, which sizes the threadgroup
// scratch arrays in the small-batch forward kernel.
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
}

struct PufferMetalPolicyContext {
    PufferMetalPolicyContext()
        : device(nil),
          queue(nil),
          forward_pipeline(nil),
          sample_pipeline(nil),
          add_bias_pipeline(nil),
          mingru_apply_pipeline(nil),
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
          action_mask_buffer(nil),
          actions_buffer(nil),
          rollout_actions_buffer(nil),
          rollout_logprobs_buffer(nil),
          rollout_values_buffer(nil),
          hidden_a_buffer(nil),
          hidden_b_buffer(nil),
          gru_linear_buffer(nil),
          encoder_columns_buffer(nil),
          decoder_columns_buffer(nil),
          value_columns_buffer(nil),
          config_buffer(nil),
          sample_config_buffer(nil),
          last_command_buffer(nil),
          action_mask_dirty(false) {}

    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> forward_pipeline;
    id<MTLComputePipelineState> sample_pipeline;
    id<MTLComputePipelineState> add_bias_pipeline;
    id<MTLComputePipelineState> mingru_apply_pipeline;
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
    id<MTLBuffer> action_mask_buffer;
    id<MTLBuffer> actions_buffer;
    id<MTLBuffer> rollout_actions_buffer;
    id<MTLBuffer> rollout_logprobs_buffer;
    id<MTLBuffer> rollout_values_buffer;
    id<MTLBuffer> hidden_a_buffer;
    id<MTLBuffer> hidden_b_buffer;
    id<MTLBuffer> gru_linear_buffer;
    id<MTLBuffer> encoder_columns_buffer;
    id<MTLBuffer> decoder_columns_buffer;
    id<MTLBuffer> value_columns_buffer;
    id<MTLBuffer> config_buffer;
    id<MTLBuffer> sample_config_buffer;
    id<MTLCommandBuffer> last_command_buffer;
    bool action_mask_dirty;
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
            set_error(error, error_len, "invalid Metal context arguments");
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

        id<MTLComputePipelineState> forward_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_forward_eval", error, error_len);
        id<MTLComputePipelineState> sample_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_sample_rollout", error, error_len);
        id<MTLComputePipelineState> add_bias_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_add_bias", error, error_len);
        id<MTLComputePipelineState> mingru_apply_pipeline = compile_pipeline_from_library(
            device, library, @"puffer_policy_mingru_apply", error, error_len);
        if (forward_pipeline == nil || sample_pipeline == nil ||
                add_bias_pipeline == nil || mingru_apply_pipeline == nil) {
            return 1;
        }

        PufferMetalPolicyContext* context = new PufferMetalPolicyContext();
        context->device = device;
        context->queue = queue;
        context->forward_pipeline = forward_pipeline;
        context->sample_pipeline = sample_pipeline;
        context->add_bias_pipeline = add_bias_pipeline;
        context->mingru_apply_pipeline = mingru_apply_pipeline;
        *context_out = context;
        return 0;
    }
}

void puffer_metal_policy_destroy(PufferMetalPolicyContext* context) {
    delete context;
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
            set_error(error, error_len, "invalid resident load arguments");
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
        const size_t action_mask_bytes = batch * action_size * sizeof(uint8_t);
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
            context->action_mask_buffer = [device newBufferWithLength:action_mask_bytes options:MTLResourceStorageModeShared];
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
            if (context->action_mask_buffer != nil) {
                std::memset(context->action_mask_buffer.contents, 1, action_mask_bytes);
                context->action_mask_dirty = false;
            }
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
        id<MTLBuffer> action_mask_buffer = context->action_mask_buffer;
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
                logits_buffer == nil || values_buffer == nil ||
                action_mask_buffer == nil || actions_buffer == nil ||
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
            set_error(error, error_len, "invalid resident observation write arguments");
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

static int forward_resident_async(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr) {
            set_error(error, error_len, "invalid resident forward arguments");
            return 1;
        }

        if (!supported_resident_shape(config)) {
            set_error(error, error_len, "unsupported resident forward shape");
            return 1;
        }

        id<MTLCommandQueue> queue = context->queue;
        id<MTLComputePipelineState> forward_pipeline = context->forward_pipeline;
        id<MTLComputePipelineState> add_bias_pipeline = context->add_bias_pipeline;
        id<MTLComputePipelineState> mingru_apply_pipeline = context->mingru_apply_pipeline;
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

        if (queue == nil || forward_pipeline == nil ||
                add_bias_pipeline == nil || mingru_apply_pipeline == nil ||
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
                set_error(error, error_len, "MPS matmul objects are not loaded");
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

static int sample_rollout_resident_async(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    const uint8_t* action_mask,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || sample_config == nullptr) {
            set_error(error, error_len, "invalid resident sample arguments");
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
        id<MTLBuffer> action_mask_buffer = context->action_mask_buffer;
        id<MTLBuffer> actions_buffer = context->actions_buffer;
        id<MTLBuffer> config_buffer = context->config_buffer;
        id<MTLBuffer> sample_config_buffer = context->sample_config_buffer;
        id<MTLBuffer> decoder_logstd_buffer = context->decoder_logstd_buffer;

        if (device == nil || queue == nil || sample_pipeline == nil ||
                logits_buffer == nil || values_buffer == nil ||
                action_mask_buffer == nil ||
                actions_buffer == nil || config_buffer == nil ||
                sample_config_buffer == nil || decoder_logstd_buffer == nil ||
                context->batch_size != config->batch_size) {
            set_error(error, error_len, "resident buffers are not loaded");
            return 1;
        }

        const size_t batch = config->batch_size;
        const size_t action_mask_bytes = batch * config->action_size * sizeof(uint8_t);
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
            set_error(error, error_len, "failed to allocate rollout Metal buffers");
            return 1;
        }

        PufferMetalPolicySampleConfig effective_sample_config = *sample_config;
        effective_sample_config.has_mask = action_mask != nullptr ? 1u : 0u;
        std::memcpy(sample_config_buffer.contents,
            &effective_sample_config, sizeof(effective_sample_config));
        if (action_mask != nullptr) {
            std::memcpy(action_mask_buffer.contents, action_mask, action_mask_bytes);
            context->action_mask_dirty = true;
        } else if (context->action_mask_dirty) {
            std::memset(action_mask_buffer.contents, 1, action_mask_bytes);
            context->action_mask_dirty = false;
        }

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
        [encoder setBuffer:action_mask_buffer offset:0 atIndex:9];

        MTLSize threads = MTLSizeMake(batch, 1, 1);
        MTLSize threadgroup = MTLSizeMake(std::min<NSUInteger>(sample_pipeline.maxTotalThreadsPerThreadgroup, 256), 1, 1);
        [encoder dispatchThreads:threads threadsPerThreadgroup:threadgroup];
        [encoder endEncoding];

        [command_buffer commit];
        context->last_command_buffer = command_buffer;
        return 0;
    }
}

int puffer_metal_policy_forward_sample_rollout_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    const PufferMetalPolicySampleConfig* sample_config,
    const uint8_t* action_mask,
    char* error,
    size_t error_len) {
    int rc = forward_resident_async(context, config, error, error_len);
    if (rc != 0) {
        return rc;
    }

    rc = sample_rollout_resident_async(
        context, config, sample_config, action_mask, error, error_len);
    if (rc != 0) {
        return rc;
    }

    return synchronize_last_command_buffer(context, error, error_len);
}

static int synchronize_last_command_buffer(
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

int puffer_metal_policy_read_actions_resident(
    PufferMetalPolicyContext* context,
    const PufferMetalPolicyForwardConfig* config,
    int32_t* actions_out,
    char* error,
    size_t error_len) {
    @autoreleasepool {
        set_error(error, error_len, "");

        if (context == nullptr || config == nullptr || actions_out == nullptr) {
            set_error(error, error_len, "invalid resident action read arguments");
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
            set_error(error, error_len, "invalid rollout read arguments");
            return 1;
        }

        if (context->batch_size != config->batch_size || !supported_resident_shape(config) ||
                sample_config->rollout_horizon == 0 ||
                context->rollout_horizon != sample_config->rollout_horizon ||
                context->rollout_actions_buffer == nil ||
                context->rollout_logprobs_buffer == nil ||
                context->rollout_values_buffer == nil) {
            set_error(error, error_len, "unsupported rollout read shape");
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
