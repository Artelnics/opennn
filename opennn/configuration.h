//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <atomic>

#include "types.h"

namespace opennn
{

class Configuration
{
public:

    struct Resolved
    {
        Device device         = Device::CPU;
        Type   training_type  = Type::FP32;
    };

    [[nodiscard]] static Configuration& instance()
    {
        static Configuration configuration;
        return configuration;
    }

    void set(Device new_device        = Device::Auto,
             Type   new_training_type = Type::Auto);

    [[nodiscard]] Device get_device()         const { return device; }
    [[nodiscard]] Type   get_training_type()  const { return training_type; }

    [[nodiscard]] const Resolved& resolve() const
    {
        if (cache_valid)
            return cached_resolved;

        return resolve_slow();
    }

    [[nodiscard]] bool is_gpu() const { return resolve().device == Device::CUDA; }
    [[nodiscard]] bool is_cpu() const { return resolve().device == Device::CPU; }

    [[nodiscard]] bool is_bf16_training()  const { return resolve().training_type  == Type::BF16; }

private:

    Configuration() = default;

    [[nodiscard]] const Resolved& resolve_slow() const;

    Device device         = Device::Auto;
    Type   training_type  = Type::Auto;

    mutable Resolved           cached_resolved;
    mutable std::atomic<bool>  cache_valid{false};
};

[[nodiscard]] inline bool   is_gpu()            { return Configuration::instance().is_gpu(); }
[[nodiscard]] inline bool   is_cpu()            { return Configuration::instance().is_cpu(); }
[[nodiscard]] inline bool   is_bf16_training()  { return Configuration::instance().is_bf16_training(); }
[[nodiscard]] inline Device current_device()    { return Configuration::instance().resolve().device; }

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
