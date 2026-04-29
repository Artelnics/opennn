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

Configuration& Configuration::instance()
{
    static Configuration cfg;
    return cfg;
}

void Configuration::set(DeviceType d, TrainingPrecision tp, InferencePrecision ip)
{
    device              = d;
    training_precision  = tp;
    inference_precision = ip;
    // Invalidate the cached Resolved so a follow-up is_gpu()/resolve() re-detects
    // hardware and applies the new user request.
    cache_valid = false;
}

#ifdef OPENNN_WITH_CUDA
// True if there's at least one CUDA device available. Wraps cudaGetDeviceCount
// and clears any sticky error so a missing CUDA stack on the host doesn't poison
// later cudaGetLastError() calls.
static bool has_cuda_gpu()
{
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) { cudaGetLastError(); return false; }
    return count > 0;
}

// Compute capability of device 0. Returns major*10+minor, or -1 on failure.
// BF16 Tensor Cores require >= 80 (Ampere).
static int cuda_compute_capability()
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) { cudaGetLastError(); return -1; }
    return prop.major * 10 + prop.minor;
}
#else
static bool has_cuda_gpu()           { return false; }
static int  cuda_compute_capability(){ return -1; }
#endif

const Configuration::Resolved& Configuration::resolve() const
{
    if (cache_valid) return cached_resolved;

    Resolved r;

    // 1. Device.
    switch (device)
    {
    case DeviceType::Auto:
        r.device = has_cuda_gpu() ? DeviceType::CUDA : DeviceType::CPU;
        break;
    case DeviceType::CPU:
        r.device = DeviceType::CPU;
        break;
    case DeviceType::CUDA:
        if (!has_cuda_gpu())
            throw runtime_error("Configuration: CUDA requested but no GPU detected.");
        r.device = DeviceType::CUDA;
        break;
    }

    const bool gpu = (r.device == DeviceType::CUDA);
    const int  cc  = gpu ? cuda_compute_capability() : -1;
    const bool bf16_capable = gpu && (cc >= 80);   // Ampere+ has BF16 Tensor Cores.

    // 2. Training precision.
    switch (training_precision)
    {
    case TrainingPrecision::Auto:
        r.training_precision = bf16_capable ? TrainingPrecision::BP16 : TrainingPrecision::Float32;
        break;
    case TrainingPrecision::Float32:
        r.training_precision = TrainingPrecision::Float32;
        break;
    case TrainingPrecision::BP16:
        if (!gpu)
            throw runtime_error("Configuration: BP16 training requires CUDA.");
        if (!bf16_capable)
            throw runtime_error("Configuration: BP16 training requires CUDA compute capability >= 8.0 (Ampere+).");
        r.training_precision = TrainingPrecision::BP16;
        break;
    }

    // 3. Inference precision. Defaults to mirror the training precision so that the
    //    parameters_bf16 working copy (if any) is reused for inference without re-cast.
    switch (inference_precision)
    {
    case InferencePrecision::Auto:
        r.inference_precision = (r.training_precision == TrainingPrecision::BP16)
                                    ? InferencePrecision::BP16
                                    : InferencePrecision::Float32;
        break;
    case InferencePrecision::Float32:
        r.inference_precision = InferencePrecision::Float32;
        break;
    case InferencePrecision::BP16:
        if (!gpu)
            throw runtime_error("Configuration: BP16 inference requires CUDA.");
        if (!bf16_capable)
            throw runtime_error("Configuration: BP16 inference requires CUDA compute capability >= 8.0 (Ampere+).");
        r.inference_precision = InferencePrecision::BP16;
        break;
    case InferencePrecision::Int8:
        // Placeholder: enum exists so user code can be written against the future API,
        // but no INT8 calibration / kernels are implemented. Fail loudly.
        throw runtime_error("Configuration: INT8 inference not yet supported (placeholder).");
    }

    cached_resolved = r;
    cache_valid = true;
    return cached_resolved;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
