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
#include <time.h>

#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;
using namespace chrono;
using namespace Eigen;


int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;

        // Data set
        
        DataSet data_set("C:/Users/davidgonzalez/Documents/iris_plant_original.csv", ";", true, false);

        const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
        const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);

        data_set.print_data();

        // Neural network

        const Index neurons_number = 1;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
            { input_variables_number }, { neurons_number }, { target_variables_number });

        //neural_network.print();

        // Training strategy
        
        TrainingStrategy training_strategy(&neural_network, &data_set);
        
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        
        //TrainingResults results = training_strategy.perform_training();
        TrainingResults results = training_strategy.perform_training_cuda();

        // Testing analysis

        //const TestingAnalysis testing_analysis(&neural_network, &data_set);

        //cout << "Confusion matrix:\n" << testing_analysis.calculate_confusion() << endl;
        
        /*
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Data set

        ImageDataSet image_data_set;

        image_data_set.set_data_source_path("C:/training_mnist");

        image_data_set.read_bmp();

        image_data_set.print();

        //image_data_set.set_training();

        const Index target_variables_number = image_data_set.get_target_variables_number();

        const Tensor<Index, 1> training_samples_indices = image_data_set.get_training_samples_indices();
        const Index training_samples_number = training_samples_indices.size();

        const Tensor<Index, 1> input_variables_indices = image_data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = image_data_set.get_target_variables_indices();

        const Tensor<Index, 1> input_variables_dimensions = image_data_set.get_input_variables_dimensions();
        const Index inputs_rows_number = input_variables_dimensions[0];
        const Index input_width = input_variables_dimensions[1];
        const Index channels = input_variables_dimensions[2];

        // Convolutional layer
        // 
        //const Index kernels_rows_number = 7;
        //const Index kernels_columns_number = 7;
        //const Index kernels_channels = 1;
        //const Index kernels_number = 1;
        //const dimensions convolutional_layer_kernels_dimensions({ kernels_number, kernels_rows_number, kernels_columns_number, kernels_channels });
        //const dimensions flatten_layer_inputs_dimensions({ training_samples_number, inputs_rows_number - kernels_rows_number + 1, input_width - kernels_columns_number + 1, kernels_number });

        const dimensions pool_dimensions({ 2, 2, 1, 1 });
        const dimensions input_dimensions({ inputs_rows_number, input_width, channels });   
       
        // Neural network

        NeuralNetwork neural_network;

        ScalingLayer4D* scaling_layer = new ScalingLayer4D(input_variables_dimensions);
        neural_network.add_layer(scaling_layer);

        PoolingLayer* pooling_layer = new PoolingLayer(input_dimensions, pool_dimensions);
        //neural_network.add_layer(pooling_layer);

        const dimensions flatten_layer_inputs_dimensions({ training_samples_number, /*pooling_layer->get_outputs_rows_number(), pooling_layer->get_outputs_columns_number(), pooling_layer->get_channels_number()}*//*28,28,1 });
        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);
        neural_network.add_layer(flatten_layer);

        PerceptronLayer* perceptron_layer = new PerceptronLayer(flatten_layer->get_outputs_number(), 20);
        neural_network.add_layer(perceptron_layer);
        perceptron_layer->set_activation_function("RectifiedLinear");

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), target_variables_number);
        neural_network.add_layer(probabilistic_layer);

        cout << endl;

        neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(15);
        training_strategy.get_adaptive_moment_estimation()->set_learning_rate(0.02);
        training_strategy.set_display_period(1);

        //training_strategy.perform_training();
        training_strategy.perform_training_cuda();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &image_data_set);

        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;
        */
        
        //cout << "OpenNN. Pooling layer Example." << endl;
        /*
        ImageDataSet image_data_set;

        image_data_set.set_data_source_path("C:/test_conv");

        image_data_set.read_bmp();

        image_data_set.set_training();

        image_data_set.print();

        const Index target_variables_number = image_data_set.get_target_variables_number();

        const Tensor<Index, 1> training_samples_indices = image_data_set.get_training_samples_indices();
        const Index training_samples_number = training_samples_indices.size();

        const Tensor<Index, 1> input_variables_indices = image_data_set.get_input_variables_indices();
        const Tensor<Index, 1> target_variables_indices = image_data_set.get_target_variables_indices();

        const Tensor<Index, 1> input_variables_dimensions = image_data_set.get_input_variables_dimensions();
        const Index inputs_rows_number = input_variables_dimensions[0];
        const Index input_width = input_variables_dimensions[1];
        const Index channels = input_variables_dimensions[2];

        vector<Index> input_dimensions = { inputs_rows_number, input_width, channels };
        vector<Index> pool_dimensions = { 2, 2, 1, 1 };

        cout << "Images data:\n" << image_data_set.get_training_input_data() << endl;
        // Neural network

        NeuralNetwork neural_network;

        PoolingLayer* pooling_layer = new PoolingLayer(input_dimensions, pool_dimensions);
        pooling_layer->set_pooling_method("MaxPooling");
        pooling_layer->set_column_stride(2);
        pooling_layer->set_row_stride(2);
        neural_network.add_layer(pooling_layer);

        cout << endl;
        neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(1);
        training_strategy.set_display_period(1);

        training_strategy.perform_training_cuda();
        */
    
        /*
        const Index kernel_height = 2;
        const Index kernel_width = 2;
        const Index channels = 1;
        const Index kernels_number = 1;

        const Index pool_height = 1;
        const Index pool_width = 1;

        // Data set

        ImageDataSet image_data_set(2, 3, 3, channels, 2);

        image_data_set.set_image_data_random();

        //ImageDataSet image_data_set;

        //image_data_set.set_data_source_path("data");
        //image_data_set.set_data_source_path("C:/mnist/train");
        //image_data_set.set_data_source_path("C:/cifar10");

        //image_data_set.read_bmp();

        //image_data_set.set_training();

        image_data_set.print();

        //image_data_set.print_data();

        // Neural network

        NeuralNetwork neural_network;

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(15);
        //training_strategy.get_adaptive_moment_estimation()->set_learning_rate(type(0.02));
        training_strategy.set_display_period(1);

        training_strategy.perform_training();
        //training_strategy.perform_training_cuda();
        
        // Testing analysis
        
        const TestingAnalysis testing_analysis(&neural_network, &image_data_set);

        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;
        */
        cout << "Bye!" << endl;
        
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}