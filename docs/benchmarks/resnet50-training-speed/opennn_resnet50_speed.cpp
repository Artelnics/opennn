//   OpenNN GPU ResNet-50 training-speed benchmark on CIFAR-10.
//
//   Standard ResNet-50 v1.5 (stride on the 3x3 convolution, as torchvision
//   builds it) applied to 32x32x3 inputs: conv1 7x7/2 -> maxpool 3x3/2 ->
//   bottleneck stages [3,4,6,3] -> flatten (final feature map is 1x1x2048)
//   -> Dense 10 softmax. Cross-entropy, Adam, fp32. Reads the CIFAR-10 BMP
//   class folders and reports seconds/epoch and samples/second after warmup.
//
//   usage:  opennn_resnet50_speed <data_path> [epochs] [batch] [fp32|bf16]

#include <chrono>
#include <iostream>
#include <string>

#include "../../../opennn/image_dataset.h"
#include "../../../opennn/neural_network.h"
#include "../../../opennn/scaling_layer.h"
#include "../../../opennn/convolutional_layer.h"
#include "../../../opennn/pooling_layer.h"
#include "../../../opennn/addition_layer.h"
#include "../../../opennn/activation_layer.h"
#include "../../../opennn/flatten_layer.h"
#include "../../../opennn/dense_layer.h"
#include "../../../opennn/training_strategy.h"
#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/random_utilities.h"
#include "../../../opennn/configuration.h"

using namespace opennn;
using clock_type = std::chrono::steady_clock;

namespace
{

Index add_conv(NeuralNetwork& network, Index source, Index out_channels,
               Index kernel, Index stride, const std::string& activation)
{
    const Shape input_shape = network.get_layer(source)->get_output_shape();

    network.add_layer(make_unique<Convolutional>(input_shape,
                                                 Shape{kernel, kernel, input_shape[2], out_channels},
                                                 activation,
                                                 Shape{stride, stride},
                                                 kernel == 1 ? "Valid" : "Same",
                                                 true),
                      {source});
    return network.get_layers_number() - 1;
}

Index bottleneck(NeuralNetwork& network, Index input, Index mid, Index out, Index stride)
{
    const Index a = add_conv(network, input, mid, 1, 1, "ReLU");
    const Index b = add_conv(network, a, mid, 3, stride, "ReLU");
    const Index c = add_conv(network, b, out, 1, 1, "Identity");

    const Shape input_shape = network.get_layer(input)->get_output_shape();
    const Index skip = (stride != 1 || input_shape[2] != out)
        ? add_conv(network, input, out, 1, stride, "Identity")
        : input;

    network.add_layer(make_unique<Addition>(network.get_layer(c)->get_output_shape()), {c, skip});
    const Index sum = network.get_layers_number() - 1;

    network.add_layer(make_unique<Activation>(network.get_layer(sum)->get_output_shape(), "ReLU"), {sum});
    return network.get_layers_number() - 1;
}

}

int main(int argc, char* argv[])
{
    try
    {
        const std::string data_path = argc > 1 ? argv[1] : "cifar10/train";
        const Index timed_epochs = argc > 2 ? Index(std::stoll(argv[2])) : 5;
        const Index batch = argc > 3 ? Index(std::stoll(argv[3])) : 128;
        const std::string precision = argc > 4 ? argv[4] : "fp32";

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        ImageDataset dataset(data_path);
        dataset.set_sample_roles("Training");
        const Index samples = dataset.get_samples_number();

        std::cout << "samples=" << samples << " batch=" << batch
                  << " epochs=" << timed_epochs << " precision=" << precision << "\n";

        NeuralNetwork network;

        auto scaling = make_unique<Scaling>(dataset.get_shape("Input"));
        scaling->set_scalers("ImageMinMax");
        network.add_layer(move(scaling));

        Index x = add_conv(network, 0, 64, 7, 2, "ReLU");

        network.add_layer(make_unique<Pooling>(network.get_layer(x)->get_output_shape(),
                                               Shape{3, 3}, Shape{2, 2}, Shape{1, 1},
                                               "MaxPooling"),
                          {x});
        x = network.get_layers_number() - 1;

        const Index mids[4] = {64, 128, 256, 512};
        const Index outs[4] = {256, 512, 1024, 2048};
        const Index blocks[4] = {3, 4, 6, 3};

        for (int stage = 0; stage < 4; ++stage)
            for (Index block = 0; block < blocks[stage]; ++block)
                x = bottleneck(network, x, mids[stage], outs[stage],
                               (block == 0 && stage > 0) ? 2 : 1);

        network.add_layer(make_unique<Flatten>(network.get_layer(x)->get_output_shape()), {x});

        network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                     dataset.get_shape("Target"),
                                                     "Softmax"));

        network.compile();
        network.set_parameters_random();

        std::cout << "layers=" << network.get_layers_number()
                  << " parameters=" << network.get_parameters_size() << "\n";

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.get_loss()->set_regularization("NoRegularization");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);

        adam->set_maximum_epochs(2);
        training_strategy.train();

        adam->set_maximum_epochs(timed_epochs);
        const auto t0 = clock_type::now();
        const TrainingResult results = training_strategy.train();
        const auto t1 = clock_type::now();

        const double total_s = std::chrono::duration<double>(t1 - t0).count();
        const double epoch_s = total_s / double(timed_epochs);

        std::cerr << "final_training_error " << results.get_training_error() << "\n";
        std::cout << "epoch_s=" << epoch_s << "\n";
        std::cout << "samples_per_sec=" << long(double(samples) / epoch_s) << "\n";
        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
