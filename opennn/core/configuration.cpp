//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"

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
    const lock_guard<mutex> lock(configuration_mutex);

    device         = new_device;
    training_type  = new_training_type;
    ++generation;
}

unsigned Configuration::get_generation() const
{
    const lock_guard<mutex> lock(configuration_mutex);

    return generation;
}

EffectiveConfig Configuration::resolve() const
{
    const lock_guard<mutex> lock(configuration_mutex);
    return resolve_effective(device);
}

EffectiveConfig Configuration::resolve_for(const Device requested_device) const
{
    const lock_guard<mutex> lock(configuration_mutex);

    if (requested_device == Device::CPU)
        return {Device::CPU, Type::FP32, generation};

    return resolve_effective(requested_device);
}

EffectiveConfig Configuration::resolve_effective(const Device requested_device) const
{
    EffectiveConfig resolved{Device::CPU, Type::FP32, generation};

    switch (requested_device)
    {
    case Device::Auto:
        resolved.device = device::has_cuda_device() ? Device::CUDA : Device::CPU;
        break;
    case Device::CPU:
        resolved.device = Device::CPU;
        break;
    case Device::CUDA:
        throw_if(!device::has_cuda_device(),
                 "Configuration: CUDA requested but no GPU detected.");
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
    case Type::INT8:
    {
        const char* const type_name = training_type == Type::BF16 ? "BF16" : "INT8";
        throw_if(resolved.device != Device::CUDA,
                 "Configuration: {} requires CUDA.", type_name);
        throw_if(compute_capability < 80,
                 "Configuration: {} requires CUDA compute capability >= 8.0 (Ampere+).",
                 type_name);
        resolved.training_type = training_type;
        break;
    }
    }

    return resolved;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
