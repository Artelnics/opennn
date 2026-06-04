//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "configuration.h"
#include "device_backend.h"

namespace opennn
{

void Configuration::set(Device new_device,
                        Type new_training_type)
{
    device         = new_device;
    training_type  = new_training_type;
    cache_valid = false;
}

const Configuration::Resolved& Configuration::resolve_slow() const
{
    Resolved resolved;

    {
        using enum Device;
        switch (device)
        {
        case Auto:
            resolved.device = device::has_cuda_device() ? CUDA : CPU;
            break;
        case CPU:
            resolved.device = CPU;
            break;
        case CUDA:
            throw_if(!device::has_cuda_device(),
                     "Configuration: CUDA requested but no GPU detected.");
            resolved.device = CUDA;
            break;
        }
    }

    const bool gpu = (resolved.device == Device::CUDA);
    const int  compute_capability = gpu ? device::cuda_compute_capability() : -1;
    const bool bf16_capable = gpu && (compute_capability >= 80);

    auto resolve_dtype = [&](Type requested) -> Type
    {
        using enum Type;
        switch (requested)
        {
        case Auto: return bf16_capable ? BF16 : FP32;
        case FP32: return FP32;
        case BF16:
            throw_if(!gpu, "Configuration: BF16 requires CUDA.");
            throw_if(!bf16_capable, "Configuration: BF16 requires CUDA compute capability >= 8.0 (Ampere+).");
            return BF16;
        }
        return FP32;
    };

    resolved.training_type = resolve_dtype(training_type);

    cached_resolved = resolved;
    cache_valid = true;
    return cached_resolved;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
