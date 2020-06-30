//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com


// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;


int main(void)
{
    try
    {
        cout << "Hello Blank Application" << endl;

        cout << "OpenNN. Vaxtor Example." << endl;

      srand(static_cast<unsigned>(time(nullptr)));

      // Device

      const int n = 4;
      NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
      ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

      // Data set

      bool display_data_set = false;

      DataSet data_set("D:/vaxtor.csv",';',true);

      data_set.set_thread_pool_device(thread_pool_device);

      const Index columns_number = data_set.get_columns_number();

      data_set.set_input();
      data_set.set_column_use(columns_number-1, DataSet::VariableUse::Target);

      const Tensor<string, 1> unused_variables = data_set.unuse_constant_columns();

      const Index input_variables_number = data_set.get_input_variables_number();
      const Index target_variables_number = data_set.get_target_variables_number();
      const Index unused_variables_number = data_set.get_unused_variables_number();

      Tensor<string, 1> scaling_methods(input_variables_number);
      scaling_methods.setConstant("MinimumMaximum");

      const Tensor<Descriptives, 1> inputs_descriptives = data_set.calculate_input_variables_descriptives();
      const Tensor<Descriptives, 1> targets_descriptives = data_set.calculate_target_variables_descriptives();

      data_set.scale_inputs(scaling_methods, inputs_descriptives);

      if(display_data_set)
      {
          Tensor<DataSet::Column, 1> columns = data_set.get_columns();

          cout << "Input variables number: " << input_variables_number << endl;
          cout << "Target variables number: " << target_variables_number << endl;
          cout << "Unused variables number: " << unused_variables_number << endl;

          system("pause");
      }


      // Neural network

      bool display_neural_network = false;

      Tensor<Index, 1> architecture(3);
      architecture[0] = input_variables_number;
      architecture[1] = 150;
      architecture[2] = target_variables_number;

      NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification, architecture);
      neural_network.set_thread_pool_device(thread_pool_device);

      neural_network.set_inputs_names(data_set.get_input_variables_names());
      neural_network.set_outputs_names(data_set.get_target_variables_names());

      ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
      scaling_layer_pointer->set_descriptives(inputs_descriptives);
      scaling_layer_pointer->set_scaling_methods(scaling_methods);

      if(display_neural_network)
      {
          Tensor<Layer*, 1> layers_pointers = neural_network.get_trainable_layers_pointers();

          for(Index i = 0; i < layers_pointers.size(); i++)
          {
              cout << "Layer " << i << ": " << endl;
              cout << "   Type: " << layers_pointers(i)->get_type_string() << endl;

              if(layers_pointers(i)->get_type_string() == "Perceptron") cout << "   Activation: " << static_cast<PerceptronLayer*>(layers_pointers(i))->write_activation_function() << endl;
              if(layers_pointers(i)->get_type_string() == "Probabilistic") cout << "   Activation: " << static_cast<ProbabilisticLayer*>(layers_pointers(i))->write_activation_function() << endl;
          }

          system("pause");
      }

      neural_network.set_parameters_random();

      // Training strategy

      TrainingStrategy training_strategy(&neural_network, &data_set);
      training_strategy.set_thread_pool_device(thread_pool_device);

      training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);

      NormalizedSquaredError* normalized_squared_error_pointer = training_strategy.get_normalized_squared_error_pointer();
      normalized_squared_error_pointer->set_normalization_coefficient();

      cout << "Normalization coefficient: " << normalized_squared_error_pointer->get_normalization_coefficient() << endl;

      training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);

      training_strategy.set_maximum_time(12*60*60);

      training_strategy.set_display_period(1);

      training_strategy.set_maximum_epochs_number(2000);

      training_strategy.perform_training();

      // Testing analysis

      TestingAnalysis testing_analysis(&neural_network, &data_set);
      testing_analysis.set_thread_pool_device(thread_pool_device);

      Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
      Tensor<type, 1> multiple_classification_tests = testing_analysis.calculate_multiple_classification_tests();

      cout << "Confusion matrix: " << endl;
      cout << confusion << endl;

      cout << "Accuracy: " << multiple_classification_tests(0)*100 << endl;
      cout << "Error: " << multiple_classification_tests(1)*100 << endl;

      neural_network.save("D:/vaxtor_NN.xml");

        cout << "Bye Blank Application" << endl;

        return 0;

    }
    catch(exception& e)
    {
       cerr << e.what() << endl;
    }
  }


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
