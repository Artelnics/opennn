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

enum class ActivationDtype { Float32, BP16 };

inline ActivationDtype to_activation_dtype(TrainingPrecision precision) noexcept
{
    return precision == TrainingPrecision::BP16 ? ActivationDtype::BP16 : ActivationDtype::Float32;
}

inline ActivationDtype to_activation_dtype(InferencePrecision precision) noexcept
{
    return precision == InferencePrecision::BP16 ? ActivationDtype::BP16 : ActivationDtype::Float32;
}

inline cudnnDataType_t to_cudnn(ActivationDtype dtype) noexcept
{
    return dtype == ActivationDtype::BP16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT;
}

inline cudaDataType_t to_cuda(ActivationDtype dtype) noexcept
{
    return dtype == ActivationDtype::BP16 ? CUDA_R_16BF : CUDA_R_32F;
}

// Element size in bytes. Overloads the `dtype_bytes(cudnnDataType_t)` helper in
// tensor_utilities.h so high-level code (BackPropagation arena sizing, Buffer
// allocation) can stay on the project-internal enum.
inline Index dtype_bytes(ActivationDtype dtype) noexcept
{
    return dtype == ActivationDtype::BP16 ? Index(sizeof(__nv_bfloat16))
                                          : Index(sizeof(float));
}

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
    void set(DeviceType         new_device_type     = DeviceType::Auto,
             TrainingPrecision  new_training_precision  = TrainingPrecision::Auto,
             InferencePrecision new_inference_precision = InferencePrecision::Auto);

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

    // Reduced-precision flags. Resolved values, not raw user input — `Auto` is
    // already resolved by this point. Used by Batch / inference paths to decide
    // whether to upload inputs as BF16 (cast at H2D) instead of FP32.
    bool is_bp16_training() const  { return resolve().training_precision  == TrainingPrecision::BP16; }
    bool is_bp16_inference() const { return resolve().inference_precision == InferencePrecision::BP16; }

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
