//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

enum class DeviceType { Auto, CPU, CUDA };

enum class TrainingPrecision  { Auto, Float32, BP16 };
enum class InferencePrecision { Auto, Float32, BP16, Int8 };

enum class Type { FP32, FP16, BF16, INT8 };

inline Type to_type(TrainingPrecision precision) noexcept
{
    return precision == TrainingPrecision::BP16 ? Type::BF16 : Type::FP32;
}

inline Type to_type(InferencePrecision precision) noexcept
{
    switch (precision)
    {
        case InferencePrecision::BP16: return Type::BF16;
        case InferencePrecision::Int8: return Type::INT8;
        default:                       return Type::FP32;
    }
}

inline cudnnDataType_t to_cudnn(Type dtype) noexcept
{
    switch (dtype)
    {
        case Type::FP32: return CUDNN_DATA_FLOAT;
        case Type::FP16: return CUDNN_DATA_HALF;
        case Type::BF16: return CUDNN_DATA_BFLOAT16;
        case Type::INT8: return CUDNN_DATA_INT8;
    }
    return CUDNN_DATA_FLOAT;
}

inline cudaDataType_t to_cuda(Type dtype) noexcept
{
    switch (dtype)
    {
        case Type::FP32: return CUDA_R_32F;
        case Type::FP16: return CUDA_R_16F;
        case Type::BF16: return CUDA_R_16BF;
        case Type::INT8: return CUDA_R_8I;
    }
    return CUDA_R_32F;
}

inline Index dtype_bytes(Type dtype) noexcept
{
    switch (dtype)
    {
        case Type::FP32: return Index(sizeof(float));
        case Type::FP16: return Index(2);
        case Type::BF16: return Index(sizeof(__nv_bfloat16));
        case Type::INT8: return Index(1);
    }
    return Index(sizeof(float));
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

    void set(DeviceType         new_device_type     = DeviceType::Auto,
             TrainingPrecision  new_training_precision  = TrainingPrecision::Auto,
             InferencePrecision new_inference_precision = InferencePrecision::Auto);

    DeviceType         get_device()              const { return device; }
    TrainingPrecision  get_training_precision()  const { return training_precision; }
    InferencePrecision get_inference_precision() const { return inference_precision; }

    const Resolved& resolve() const;

    bool is_gpu() const { return resolve().device == DeviceType::CUDA; }
    bool is_cpu() const { return resolve().device == DeviceType::CPU; }

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
