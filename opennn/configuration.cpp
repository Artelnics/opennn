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

#ifdef OPENNN_WITH_CUDA
static bool has_cuda_gpu()
{
    int count = 0;
    const cudaError_t error = cudaGetDeviceCount(&count);
    if (error != cudaSuccess) { cudaGetLastError(); return false; }
    return count > 0;
}

// BF16 Tensor Cores require Ampere+ (CC >= 80).
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

    switch (device)
    {
    case Device::Auto:
        resolved.device = has_cuda_gpu() ? Device::CUDA : Device::CPU;
        break;
    case Device::CPU:
        resolved.device = Device::CPU;
        break;
    case Device::CUDA:
        if (!has_cuda_gpu())
            throw runtime_error("Configuration: CUDA requested but no GPU detected.");
        resolved.device = Device::CUDA;
        break;
    }

    const bool gpu = (resolved.device == Device::CUDA);
    const int  compute_capability = gpu ? cuda_compute_capability() : -1;
    const bool bf16_capable = gpu && (compute_capability >= 80);

    auto resolve_dtype = [&](Type requested, const char* role) -> Type
    {
        switch (requested)
        {
        case Type::Auto: return bf16_capable ? Type::BF16 : Type::FP32;
        case Type::FP32: return Type::FP32;
        case Type::BF16:
            if (!gpu)
                throw runtime_error(string("Configuration: BF16 ") + role + " requires CUDA.");
            if (!bf16_capable)
                throw runtime_error(string("Configuration: BF16 ") + role + " requires CUDA compute capability >= 8.0 (Ampere+).");
            return Type::BF16;
        case Type::INT8:
            throw runtime_error(string("Configuration: INT8 ") + role + " not yet supported (placeholder).");
        }
        return Type::FP32;
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
