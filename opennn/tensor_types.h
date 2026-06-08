//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E N S O R   T Y P E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "types.h"

namespace opennn
{

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

template<Type... Supported, typename F>
void visit_type(Type t, F&& f)
{
    bool matched = false;
    ([&]
    {
        if (!matched && t == Supported)
        {
            f(TypeInfo<Supported>{});
            matched = true;
        }
    }(), ...);
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

[[nodiscard]] inline cudnnDataType_t to_cudnn(Type type) noexcept
{
    using enum Type;
    switch (type)
    {
        case Auto: return TypeInfo<FP32>::cudnn;
        case FP32: return TypeInfo<FP32>::cudnn;
        case BF16: return TypeInfo<BF16>::cudnn;
    }
    return TypeInfo<FP32>::cudnn;
}

[[nodiscard]] inline cudaDataType_t to_cuda(Type type) noexcept
{
    using enum Type;
    switch (type)
    {
        case Auto: return TypeInfo<FP32>::cuda;
        case FP32: return TypeInfo<FP32>::cuda;
        case BF16: return TypeInfo<BF16>::cuda;
    }
    return TypeInfo<FP32>::cuda;
}

[[nodiscard]] inline Index type_bytes(Type type) noexcept
{
    using enum Type;
    switch (type)
    {
        case Auto: return TypeInfo<FP32>::bytes;
        case FP32: return TypeInfo<FP32>::bytes;
        case BF16: return TypeInfo<BF16>::bytes;
    }
    return TypeInfo<FP32>::bytes;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
