//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <mutex>

namespace opennn
{

enum class Device { Auto, CPU, CUDA };
enum class Type { Auto, FP32, BF16 };

class Configuration
{
public:

    struct Resolved
    {
        Device device         = Device::CPU;
        Type   training_type  = Type::FP32;
    };

    static Configuration& instance();

    void set(Device new_device        = Device::Auto,
             Type   new_training_type = Type::Auto);

    Resolved resolve() const;

    bool is_gpu() const { return resolve().device == Device::CUDA; }

private:

    Configuration() = default;

    mutable std::mutex configuration_mutex;

    Device device         = Device::Auto;
    Type   training_type  = Type::Auto;
};

inline bool   is_gpu()            { return Configuration::instance().is_gpu(); }
inline Device current_device()    { return Configuration::instance().resolve().device; }

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
