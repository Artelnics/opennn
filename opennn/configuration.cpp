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

Configuration& Configuration::instance()
{
    static Configuration configuration;
    return configuration;
}

void Configuration::set(Device new_device,
                        Type new_training_type)
{
    const std::lock_guard<std::mutex> lock(configuration_mutex);

    device         = new_device;
    training_type  = new_training_type;
    ++generation;
}

unsigned Configuration::get_generation() const
{
    const std::lock_guard<std::mutex> lock(configuration_mutex);

    return generation;
}

Configuration::Resolved Configuration::resolve() const
{
    const std::lock_guard<std::mutex> lock(configuration_mutex);

    Resolved resolved;
    resolved.generation = generation;
    bool cuda_available = false;

    switch (device)
    {
    case Device::Auto:
        cuda_available = device::has_cuda_device();
        resolved.device = cuda_available ? Device::CUDA : Device::CPU;
        break;
    case Device::CPU:
        resolved.device = Device::CPU;
        break;
    case Device::CUDA:
        cuda_available = device::has_cuda_device();
        throw_if(!cuda_available, "Configuration: CUDA requested but no GPU detected.");
        resolved.device = Device::CUDA;
        break;
    }

    const int compute_capability = resolved.device == Device::CUDA && training_type != Type::FP32
        ? device::cuda_compute_capability()
        : -1;

    switch (training_type)
    {
    case Type::Auto:
        resolved.training_type = resolved.device == Device::CUDA && compute_capability >= 80
            ? Type::BF16
            : Type::FP32;
        break;
    case Type::FP32:
        resolved.training_type = Type::FP32;
        break;
    case Type::BF16:
        throw_if(resolved.device != Device::CUDA, "Configuration: BF16 requires CUDA.");
        throw_if(compute_capability < 80,
                 "Configuration: BF16 requires CUDA compute capability >= 8.0 (Ampere+).");
        resolved.training_type = Type::BF16;
        break;
    }

    return resolved;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
