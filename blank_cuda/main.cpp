//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R O S E N B R O C K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn/opennn.h"

using namespace opennn;


int main()
{
    try
    {
        /*
        cout << "OpenNN. Iris Plant Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("data/iris_plant_original.csv", ';', true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index hidden_neurons_number = 5;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification, { input_variables_number, hidden_neurons_number, target_variables_number });

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.perform_training();
        //training_strategy.perform_training_cuda();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);
        
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        Tensor<type, 2> inputs(3, neural_network.get_inputs_number());

        inputs.setValues({ {type(5.1),type(3.5),type(1.4),type(0.2)},
                          {type(6.4),type(3.2),type(4.5),type(1.5)},
                          {type(6.3),type(2.7),type(4.9),type(1.8)} });

        const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

        cout << "\nInputs:\n" << inputs << endl;

        cout << "\nOutputs:\n" << outputs << endl;

        cout << "\nConfusion matrix:\n" << confusion << endl;
        */
        /*
        cout << "\nCreating OpenNN::DataSet object.\n" << endl;

        const Index samples_number = 4143;

        const Tensor<type, 2> tensor = loadCSVtoTensor("C:/Users/davidgonzalez/Desktop/_data_.csv");

        const Tensor<string, 1> columns_names(2049);

        bool boolean = true;

        DataSet data_set(tensor, samples_number, columns_names, boolean);

        const Index hidden_neurons_number = 1000;
        const Index hidden_layers_number = 2;

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        cout << "==================================================================" << endl;
        std::cout << "input_variables_number: " << input_variables_number << std::endl;
        std::cout << "target_variables_number: " << target_variables_number << std::endl;
        cout << "Data set dimensions: " << data_set.get_data().dimensions() << endl;
        cout << "Data set unuse samples: " << data_set.get_unused_samples_number() << endl;
        cout << "==================================================================" << endl;

        // Neural network

        const Index total_layers_number = hidden_layers_number + 2;

        Tensor<Index, 1> architecture(total_layers_number); // hidden_layers_number + input_layer + output_layer
        architecture(0) = input_variables_number;
        architecture(total_layers_number - 1) = target_variables_number;

        Index user_layer_count = 0;
        std::vector<int> neurons_per_layer = {1000,1000};

        for (Index i = 1; i < total_layers_number - 1; i++)
        {
            architecture(i) = neurons_per_layer[user_layer_count];
            user_layer_count++;
            if (user_layer_count == hidden_layers_number) break;
        }

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, architecture);

        Tensor<type, 1> lower_bounds(target_variables_number);
        lower_bounds.setZero();

        //neural_network.get_bounding_layer_pointer()->set_lower_bounds(lower_bounds);
        //neural_network.get_scaling_layer_pointer()->set_display(0);

        srand(time(NULL));
        neural_network.set_parameters_random();

        // Training Strategy
        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        //training_strategy.set_maximum_epochs_number(300);
        // training_strategy.set_display_period(1);

        cout << "==================================================================" << endl;
        //training_strategy.perform_training();
        training_strategy.perform_training_cuda();
        cout << "==================================================================" << endl;
        */
        /*
        cout << "OpenNN. Rosenbrock Example." << endl;

        // Data Set

        const Index samples_number = 100;
        const Index inputs_number = 10;
        const Index outputs_number = 1;
        const Index hidden_neurons_number = 5;

        DataSet data_set;

        data_set.generate_Rosenbrock_data(samples_number, inputs_number + outputs_number);

        data_set.set_training();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { inputs_number, hidden_neurons_number, outputs_number });

        neural_network.get_first_perceptron_layer()->set_activation_function(PerceptronLayer::ActivationFunction::HyperbolicTangent);

        PerceptronLayer* pl = static_cast<PerceptronLayer*>(neural_network.get_layers()(2));

        pl->set_activation_function(PerceptronLayer::ActivationFunction::Linear);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::L2);

        training_strategy.set_maximum_epochs_number(10);
        training_strategy.set_display_period(1);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(1000);
        training_strategy.set_maximum_time(86400);

        //training_strategy.perform_training();
        training_strategy.perform_training_cuda();

        cout << "End Rosenbrock" << endl;
        */
        
        cout << "OpenNN. National Institute of Standards and Techonology (MNIST) Example." << endl;

        // Data set

        ImageDataSet image_data_set;

        image_data_set.set_data_source_path("C:/binary_mnist");

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
        const Index inputs_columns_number = input_variables_dimensions[1];
        const Index inputs_channels_number = input_variables_dimensions[2];

        //dimensions flatten_layer_inputs_dimensions({ training_samples_number, inputs_rows_number - kernels_rows_number + 1, inputs_raw_variables_number - kernels_raw_variables_number + 1, kernels_number });
        dimensions flatten_layer_inputs_dimensions({ training_samples_number, inputs_rows_number, inputs_columns_number, inputs_channels_number });
        // Neural network
        
        NeuralNetwork neural_network;

        ScalingLayer4D* scaling_layer = new ScalingLayer4D(input_variables_dimensions);
        neural_network.add_layer(scaling_layer);

        //ConvolutionalLayer* convolutional_layer = new ConvolutionalLayer(scaling_layer->get_outputs_dimensions(), convolutional_layer_kernels_dimensions);
        //neural_network.add_layer(convolutional_layer);

        FlattenLayer* flatten_layer = new FlattenLayer(flatten_layer_inputs_dimensions);
        neural_network.add_layer(flatten_layer);

        PerceptronLayer* perceptron_layer = new PerceptronLayer(flatten_layer->get_outputs_number(), 56);
        neural_network.add_layer(perceptron_layer);

        ProbabilisticLayer* probabilistic_layer = new ProbabilisticLayer(perceptron_layer->get_neurons_number(), target_variables_number);
        neural_network.add_layer(probabilistic_layer);

        cout << endl;
        neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &image_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(5000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(10);
        training_strategy.set_display_period(1);

        training_strategy.perform_training();
        //training_strategy.perform_training_cuda();

        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &image_data_set);

        cout << "Calculating confusion...." << endl;
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;
        
        /*
        Tensor<type, 4> inputs_4d;
        
        Tensor<unsigned char,1> zero = image_data_set.read_bmp_image("../data/images/zero/0_1.bmp");
        Tensor<unsigned char,1> one = image_data_set.read_bmp_image("../data/images/one/1_1.bmp");

        vector<type> zero_int(zero.size()); ;
        vector<type> one_int(one.size());

        for(Index i = 0 ; i < zero.size() ; i++ )
        {
            zero_int[i]=(type)zero[i];
            one_int[i]=(type)one[i];
        }

        Tensor<type, 2> inputs(2, zero.size());
        Tensor<type, 2> outputs(2, neural_network.get_outputs_number());

        outputs = neural_network.calculate_outputs(inputs);

        //cout << "\nInputs:\n" << inputs << endl;

        //cout << "\nOutputs:\n" << outputs << endl;
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