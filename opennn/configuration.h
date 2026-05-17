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

enum class Device { Auto, CPU, CUDA };

enum class Type { Auto, FP32, BF16, INT8 };

template<Type T> struct TypeInfo;

template<> struct TypeInfo<Type::FP32>
{
    using type = float;
    static constexpr cudnnDataType_t cudnn = CUDNN_DATA_FLOAT;
    static constexpr cudaDataType_t  cuda  = CUDA_R_32F;
    static constexpr Index           bytes = Index(sizeof(float));
    static constexpr const char*     name  = "FP32";
};

template<> struct TypeInfo<Type::BF16>
{
    using type = bfloat16;
    static constexpr cudnnDataType_t cudnn = CUDNN_DATA_BFLOAT16;
    static constexpr cudaDataType_t  cuda  = CUDA_R_16BF;
    static constexpr Index           bytes = Index(sizeof(bfloat16));
    static constexpr const char*     name  = "BF16";
};

template<> struct TypeInfo<Type::INT8>
{
    using type = int8_t;
    static constexpr cudnnDataType_t cudnn = CUDNN_DATA_INT8;
    static constexpr cudaDataType_t  cuda  = CUDA_R_8I;
    static constexpr Index           bytes = Index(1);
    static constexpr const char*     name  = "INT8";
};

template<Type... Supported, typename F>
void visit_type(Type t, F&& f)
{
    bool matched = false;
    ((t == Supported && (f(TypeInfo<Supported>{}), matched = true, false)) || ...);
    throw_if(!matched, "visit_type: unsupported Type value");
}

template<Type... Supported, typename F>
void visit_type_pair(Type t_in, Type t_out, F&& f)
{
    visit_type<Supported...>(t_in, [&](auto in_info)
    {
        visit_type<Supported...>(t_out, [&](auto out_info)
        {
            f(in_info, out_info);
        });
    });
}

inline cudnnDataType_t to_cudnn(Type type) noexcept
{
    using enum Type;
    switch (type)
    {
        case Auto: return TypeInfo<FP32>::cudnn;
        case FP32: return TypeInfo<FP32>::cudnn;
        case BF16: return TypeInfo<BF16>::cudnn;
        case INT8: return TypeInfo<INT8>::cudnn;
    }
    return TypeInfo<FP32>::cudnn;
}

inline cudaDataType_t to_cuda(Type type) noexcept
{
    using enum Type;
    switch (type)
    {
        case Auto: return TypeInfo<FP32>::cuda;
        case FP32: return TypeInfo<FP32>::cuda;
        case BF16: return TypeInfo<BF16>::cuda;
        case INT8: return TypeInfo<INT8>::cuda;
    }
    return TypeInfo<FP32>::cuda;
}

inline Index type_bytes(Type type) noexcept
{
    using enum Type;
    switch (type)
    {
        case Auto: return TypeInfo<FP32>::bytes;
        case FP32: return TypeInfo<FP32>::bytes;
        case BF16: return TypeInfo<BF16>::bytes;
        case INT8: return TypeInfo<INT8>::bytes;
    }
    return TypeInfo<FP32>::bytes;
}

class Configuration
{
public:

    struct Resolved
    {
        Device device          = Device::CPU;
        Type   training_type  = Type::FP32;
        Type   inference_type = Type::FP32;
    };

    static Configuration& instance()
    {
        static Configuration configuration;
        return configuration;
    }

    void set(Device new_device          = Device::Auto,
             Type   new_training_type   = Type::Auto,
             Type   new_inference_type  = Type::Auto);

    Device get_device()         const { return device; }
    Type   get_training_type()  const { return training_type; }
    Type   get_inference_type() const { return inference_type; }

    const Resolved& resolve() const
    {
        if (cache_valid)
            return cached_resolved;

        return resolve_slow();
    }

    bool is_gpu() const { return resolve().device == Device::CUDA; }
    bool is_cpu() const { return resolve().device == Device::CPU; }

    bool is_bf16_training()  const { return resolve().training_type  == Type::BF16; }
    bool is_bf16_inference() const { return resolve().inference_type == Type::BF16; }

private:

    Configuration() = default;

    const Resolved& resolve_slow() const;

    Device device         = Device::Auto;
    Type   training_type  = Type::Auto;
    Type   inference_type = Type::Auto;

    mutable Resolved             cached_resolved;
    mutable atomic<bool>    cache_valid{false};
};

inline bool   is_gpu()            { return Configuration::instance().is_gpu(); }
inline bool   is_cpu()            { return Configuration::instance().is_cpu(); }
inline bool   is_bf16_training()  { return Configuration::instance().is_bf16_training(); }
inline bool   is_bf16_inference() { return Configuration::instance().is_bf16_inference(); }
inline Device current_device()    { return is_gpu() ? Device::CUDA : Device::CPU; }

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
