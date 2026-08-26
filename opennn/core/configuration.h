//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N F I G U R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <atomic>
#include <mutex>

namespace opennn
{

enum class Device { Auto, CPU, CUDA };
enum class Type { Auto, FP32, BF16, INT8 };

// Which BLAS the CPU kernels dispatch to. Eigen is the default because it is
// what a plain build has: MKL is compiled in only when the build asks for it,
// and even then an application opts in rather than inheriting it. Read on
// every GEMM, so it is atomic rather than guarded by the configuration mutex.
enum class Blas { Eigen, Mkl };

enum class ActivationFunction { Identity, Sigmoid, Tanh, ReLU, Softmax, LeakyReLU, GELU, GELUTanh, SiLU };

inline constexpr float LEAKY_RELU_SLOPE = 0.1f;

inline Type activation_dtype(Type type) { return type == Type::INT8 ? Type::BF16 : type; }

struct EffectiveConfig
{
    Device device         = Device::CPU;
    Type   training_type  = Type::FP32;
    unsigned generation   = 0;
};

class Configuration
{
public:

    static Configuration& instance();

    void set(Device new_device        = Device::Auto,
             Type   new_training_type = Type::Auto);

    // Selecting Mkl on a build without it is not an error -- the MKL paths
    // simply are not there to dispatch to, and `blas_mkl_available()` says so.
    void set_blas(Blas new_blas);
    [[nodiscard]] Blas get_blas() const;

    [[nodiscard]] EffectiveConfig resolve() const;
    [[nodiscard]] EffectiveConfig resolve_for(Device) const;

    unsigned get_generation() const;

private:

    Configuration();

    EffectiveConfig resolve_effective(Device) const;

    mutable std::mutex configuration_mutex;

    Device device        = Device::Auto;
    Type   training_type = Type::Auto;
    unsigned generation  = 0;

    std::atomic<Blas> blas;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
