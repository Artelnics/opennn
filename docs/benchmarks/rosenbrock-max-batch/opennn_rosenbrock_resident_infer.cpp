//   Device-resident inference throughput for the shallow Rosenbrock MLP.
//
//   The PyTorch-equivalent inference loop: input lives on the GPU, weights are
//   uploaded ONCE, the ForwardPropagation (activation buffers) is built ONCE, and
//   the loop calls calculate_outputs_resident -- no per-call param re-upload, no
//   input H2D, no output D2H, no buffer (re)allocation. Compare its samples/sec
//   to the old calculate_outputs(MatrixR) path and to PyTorch.
//
//   usage: opennn_rosenbrock_resident_infer [batch] [iters] [inputs] [hidden]

#include <chrono>
#include <iostream>
#include <string>

#include "../../../opennn/standard_networks.h"
#include "../../../opennn/scaling_layer.h"
#include "../../../opennn/unscaling_layer.h"
#include "../../../opennn/bounding_layer.h"
#include "../../../opennn/forward_propagation.h"
#include "../../../opennn/device_backend.h"
#include "../../../opennn/tensor_types.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/configuration.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;

    const Index batch  = argc > 1 ? Index(std::stoll(argv[1])) : 2000;
    const Index iters  = argc > 2 ? Index(std::stoll(argv[2])) : 200;
    const Index inputs = argc > 3 ? Index(std::stoll(argv[3])) : 1000;
    const Index hidden = argc > 4 ? Index(std::stoll(argv[4])) : 1000;

    try
    {
        set_seed(0);
        const bool use_bf16 = std::getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        const char* act_env = std::getenv("OPENNN_ACT");
        const std::string act = act_env ? act_env : "Tanh";
        ApproximationNetwork network(Shape{inputs}, {hidden}, Shape{1}, act);
        static_cast<Scaling*>(network.get_first(LayerType::Scaling))->set_scalers("None");
        static_cast<Unscaling*>(network.get_first(LayerType::Unscaling))->set_scalers("None");
        static_cast<Bounding*>(network.get_first(LayerType::Bounding))->set_bounding_method("NoBounding");

        // Input on the GPU, ONCE.
        const MatrixR host_in = MatrixR::Random(batch, inputs);
        Buffer input_gpu;
        input_gpu.resize_bytes(batch * inputs * Index(sizeof(float)), Device::CUDA);
        device::copy_async(input_gpu.data, host_in.data(),
                           batch * inputs * Index(sizeof(float)),
                           device::CopyKind::HostToDevice);
        const vector<TensorView> gpu_inputs{
            TensorView(input_gpu.as<float>(), {batch, inputs}, Type::FP32, Device::CUDA)};

        // ForwardPropagation (activation buffers) built ONCE.
        ForwardPropagation forward_propagation(batch, &network);

        // Warmup: first call uploads parameters.
        network.calculate_outputs_resident(gpu_inputs, forward_propagation, /*upload=*/true);
        device::synchronize();

        const auto t0 = std::chrono::steady_clock::now();
        for (Index it = 0; it < iters; ++it)
            network.calculate_outputs_resident(gpu_inputs, forward_propagation, /*upload=*/false);
        device::synchronize();
        const auto t1 = std::chrono::steady_clock::now();

        const double per = std::chrono::duration<double>(t1 - t0).count() / double(iters);
        std::cout << "mode=resident_inference batch=" << batch << "\n";
        std::cout << "step_s=" << per << "\n";
        std::cout << "samples_per_sec=" << long(double(batch) / per) << "\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
