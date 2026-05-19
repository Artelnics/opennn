//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "configuration.h"

namespace opennn
{

void Configuration::set(Device new_device,
                        Type new_training_type,
                        Type new_inference_type)
{
    device         = new_device;
    training_type  = new_training_type;
    inference_type = new_inference_type;
    cache_valid = false;
}

#ifdef OPENNN_HAS_CUDA
static bool has_cuda_gpu()
{
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) { cudaGetLastError(); return false; }
    return count > 0;
}
static int cuda_compute_capability()
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { cudaGetLastError(); return -1; }
    return prop.major * 10 + prop.minor;
}
#else
static bool has_cuda_gpu()           { return false; }
static int  cuda_compute_capability() { return -1; }
#endif

const Configuration::Resolved& Configuration::resolve_slow() const
{
    Resolved resolved;

    {
        using enum Device;
        switch (device)
        {
        case Auto:
            resolved.device = has_cuda_gpu() ? CUDA : CPU;
            break;
        case CPU:
            resolved.device = CPU;
            break;
        case CUDA:
            if (!has_cuda_gpu())
                throw runtime_error("Configuration: CUDA requested but no GPU detected.");
            resolved.device = CUDA;
            break;
        }
    }

    const bool gpu = (resolved.device == Device::CUDA);
    const int  compute_capability = gpu ? cuda_compute_capability() : -1;
    const bool bf16_capable = gpu && (compute_capability >= 80);

    auto resolve_dtype = [&](Type requested, const char* role) -> Type
    {
        using enum Type;
        switch (requested)
        {
        case Auto: return bf16_capable ? BF16 : FP32;
        case FP32: return FP32;
        case BF16:
            if (!gpu)
                throw runtime_error(format("Configuration: BF16 {} requires CUDA.", role));
            if (!bf16_capable)
                throw runtime_error(format("Configuration: BF16 {} requires CUDA compute capability >= 8.0 (Ampere+).", role));
            return BF16;
        case INT8:
            throw runtime_error(format("Configuration: INT8 {} not yet supported (placeholder).", role));
        }
        return FP32;
    };

    resolved.training_type = resolve_dtype(training_type, "training");

    // Inference defaults to mirror training so the BF16 working copy (if any) is reused.
    resolved.inference_type = (inference_type == Type::Auto)
        ? resolved.training_type
        : resolve_dtype(inference_type, "inference");

    cached_resolved = resolved;
    cache_valid = true;
    return cached_resolved;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
