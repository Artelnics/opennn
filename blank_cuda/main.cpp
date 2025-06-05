//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <ctime>

#include "../opennn/pch.h"
#include "../opennn/data_set.h"
#include "../opennn/neural_network.h"
#include "../opennn/training_strategy.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/image_data_set.h"
#include "../opennn/scaling_layer_4d.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/pooling_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/perceptron_layer.h"

using namespace std;
using namespace chrono;
using namespace Eigen;
using namespace opennn;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;

        #ifdef OPENNN_CUDA

        // Data set
        /*
        const Index samples_number = 2;

        const Index image_height = 3;
        const Index image_width = 3;
        const Index channels = 1;
        const Index targets = 2;

        ImageDataset data_set(samples_number, {image_height, image_width, channels}, {targets});

        data_set.set_data_random();
        data_set.set_data_ascending();

        data_set.set(Dataset::SampleUse::Training);

        data_set.print_data();
        */
        ImageDataset data_set;

        //data_set.set_data_path("C:/cifar10_bmp");
        data_set.set_data_path("../examples/mnist/data");
        //data_set.set_data_path("../examples/mnist/data_bin");

        data_set.read_bmp();

        //data_set.split_samples_random(0.8, 0.0, 0.2);
        
        //const dimensions input_dimensions = data_set.get_dimensions(Dataset::VariableUse::Input);

        // Neural network
        
        NeuralNetwork neural_network(NeuralNetwork::ModelType::ImageClassification,
            data_set.get_dimensions(Dataset::VariableUse::Input),
            { 1 },
            data_set.get_dimensions(Dataset::VariableUse::Target));
        /*
        NeuralNetwork neural_network;

        // Scaling 4D
        neural_network.add_layer(make_unique<Scaling4d>(input_dimensions));

        // --- conv1 -> pool1(dropout 0.25) ---
        {
            // Conv 3×3, 32 kernel, ReLU
            neural_network.add_layer( make_unique<Convolutional>(
                neural_network.get_output_dimensions(),
                dimensions{ 3, 3, input_dimensions[2], 32},
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
                "convolution_1")
            );
            // Pooling 2×2 stride 2 + dropout 25%
            auto pool1 = make_unique<Pooling>(
                neural_network.get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling,
                "pool1"
            );
            //pool1->set_dropout_rate(0.25f);
            neural_network.add_layer(move(pool1));
        }

        // --- conv2 -> pool2(dropout 0.25) ---
        {
            // Conv 3×3, 64 kernels, ReLU
            neural_network.add_layer(make_unique<Convolutional>(
                neural_network.get_output_dimensions(),
                dimensions{ 3, 3, 32, 64 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
                "convolution_2")
            );
            // Pooling 2×2 stride 2 + dropout 25%
            auto pool2 = make_unique<Pooling>(
                neural_network.get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling,
                "pool2"
            );
            //pool2->set_dropout_rate(0.25f);
            neural_network.add_layer(move(pool2));
        }

        // --- conv3 -> pool3(dropout 0.25) ---
        {
            // Conv 3×3, 128 kernels, ReLU
            neural_network.add_layer(make_unique<Convolutional>(
                neural_network.get_output_dimensions(),
                dimensions{ 3, 3, 64, 128 },
                Convolutional::Activation::RectifiedLinear,
                dimensions{ 1, 1 },
                Convolutional::Convolution::Valid,
                "convolution_3")
            );
            // Pooling 2×2 stride 2 + dropout 25%
            auto pool3 = make_unique<Pooling>(
                neural_network.get_output_dimensions(),
                dimensions{ 2, 2 },
                dimensions{ 2, 2 },
                dimensions{ 0, 0 },
                Pooling::PoolingMethod::MaxPooling,
                "pool3"
            );
            //pool3->set_dropout_rate(0.25f);
            neural_network.add_layer(move(pool3));
        }
        
        // Flatten
        neural_network.add_layer(make_unique<Flatten>(neural_network.get_output_dimensions()));
        
        // Perceptron layers
        neural_network.add_layer(make_unique<Dense2d>(
            neural_network.get_output_dimensions(),
            dimensions{ 512 },
            Dense2d::Activation::RectifiedLinear,
            "perceptron1")
        );
        neural_network.add_layer(make_unique<Dense2d>(
            neural_network.get_output_dimensions(),
            dimensions{ 128 },
            Dense2d::Activation::RectifiedLinear,
            "perceptron2")
        );
        
        // Probabilistic softmax
        neural_network.add_layer(make_unique<Dense2d>(
            neural_network.get_output_dimensions(),
            data_set.get_dimensions(DataSet::VariableUse::Target),
            "probabilistic")
        );*/

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_2D);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(128);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10);
        training_strategy.set_display_period(1);

        //training_strategy.perform_training();
        training_strategy.perform_training_cuda();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;

        #endif  
        cout << "Bye!" << endl;
        
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}