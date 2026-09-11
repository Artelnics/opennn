#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/models/models.h"
#include "../tools/startup_signal.h"
#include <cmath>

int main(int argc, char** argv)
{
    const auto entered = monotonic_ns();
    using namespace opennn;
    try {
        const bool gpu = argc > 1 && std::string(argv[1]) == "cuda";
        const bool bf16 = argc > 2 && std::string(argv[2]) == "bf16";
        set_seed(42);
        Configuration::instance().set(gpu ? Device::CUDA : Device::CPU,
                                      bf16 ? Type::BF16 : Type::FP32);
#ifdef EIGEN_USE_MKL_ALL
        Configuration::instance().set_blas(Blas::Mkl);
#else
        Configuration::instance().set_blas(Blas::Eigen);
#endif
#if FAMILY_dense
        ClassificationNetwork network({28}, {128,128}, {1}, "ReLU");
        MatrixR input(2,28); input.setConstant(0.1f);
        auto output = network.calculate_outputs(input);
#elif FAMILY_lstm
        ForecastingLstmNetwork network({8,15}, {128}, {1});
        Tensor3 input(2,8,15); input.setConstant(0.1f);
        auto output = network.calculate_outputs(input);
#elif FAMILY_cnn
        ImageClassificationNetwork network({32,32,3}, {16,32}, {10});
        Tensor4 input(2,32,32,3); input.setConstant(0.1f);
        auto output = network.calculate_outputs(input);
#elif FAMILY_transformer
        Transformer network(8,8,128,128,32,4,64,1);
        network.set_dropout_rate(0.0f);
        Tensor3 input(2,8,1), decoder(2,8,1);
        input.setConstant(1.0f); decoder.setConstant(1.0f);
        auto output = network.calculate_outputs(input, decoder);
#endif
        // calculate_outputs returns completed predictions in host memory.
        const auto ready = monotonic_ns();
        if (network.is_gpu() != gpu) throw std::runtime_error("Unexpected device fallback");
        for (Index i=0; i<output.size(); ++i)
            if (!std::isfinite(output.data()[i])) throw std::runtime_error("Nonfinite prediction");
        startup_ready(entered, ready, "opennn", gpu, bf16, network.get_parameters_number(),
                      network.get_parameters_number(), output.size(), output.data()[0]);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
