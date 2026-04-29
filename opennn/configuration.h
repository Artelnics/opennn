//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

// Singleton runtime configuration: which device the library should run on
// (CPU vs CUDA) and which precision to use for training / inference. Every
// call site that needs to ask "are we on GPU?" goes through Configuration —
// no other class duplicates this state.
//
// Lifecycle: user (optionally) calls Configuration::instance().set(...) once
// at program start; NeuralNetwork::compile() calls resolve() to convert any
// `Auto` to a concrete value based on detected hardware, and freezes the
// result inside the network.

#include "pch.h"

namespace opennn
{

// `Auto` is only valid as user input. Configuration::resolve() converts it to
// CPU or CUDA based on detected hardware before the network sees it, so any
// runtime path (Buffer, kernel dispatch) only ever observes CPU or CUDA.
enum class DeviceType { Auto, CPU, CUDA };

// Precision selectors. `Auto` is also resolved at compile() time. INT8 is a
// deliberate placeholder: Configuration::resolve() throws runtime_error if a
// user picks it — calibration + INT8 kernels are out of scope here.
enum class TrainingPrecision  { Auto, Float32, BP16 };
enum class InferencePrecision { Auto, Float32, BP16, Int8 };

class Configuration
{
public:

    struct Resolved
    {
        DeviceType         device              = DeviceType::CPU;
        TrainingPrecision  training_precision  = TrainingPrecision::Float32;
        InferencePrecision inference_precision = InferencePrecision::Float32;
    };

    static Configuration& instance();

    // Replaces all three at once. Defaulting any argument to Auto is the recommended
    // entry point — let resolve() pick. Invalidates the cached Resolved so the next
    // is_gpu()/is_cpu()/resolve() call re-detects hardware.
    void set(DeviceType         d  = DeviceType::Auto,
             TrainingPrecision  tp = TrainingPrecision::Auto,
             InferencePrecision ip = InferencePrecision::Auto);

    DeviceType         get_device()              const { return device; }
    TrainingPrecision  get_training_precision()  const { return training_precision; }
    InferencePrecision get_inference_precision() const { return inference_precision; }

    // Resolves Auto values to concrete ones by inspecting available hardware. Throws
    // runtime_error on impossible combinations (CUDA requested but no GPU; BP16 on CPU;
    // INT8 placeholder). Cached after first call.
    const Resolved& resolve() const;

    // Single source of truth for "is the active device GPU/CPU?". Resolves on first
    // call and caches.
    bool is_gpu() const { return resolve().device == DeviceType::CUDA; }
    bool is_cpu() const { return resolve().device == DeviceType::CPU; }

private:

    Configuration() = default;

    DeviceType         device              = DeviceType::Auto;
    TrainingPrecision  training_precision  = TrainingPrecision::Auto;
    InferencePrecision inference_precision = InferencePrecision::Auto;

    mutable Resolved cached_resolved;
    mutable bool     cache_valid = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
