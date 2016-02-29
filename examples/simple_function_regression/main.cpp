/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   S I M P L E   F U N C T I O N   R E G R E S S I O N   A P P L I C A T I O N                                */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// This is a function regression problem. 

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
   try
   {
      std::cout << "OpenNN. Simple Function Regression Application." << std::endl;

      srand( (unsigned)time( NULL ) );

      // Data set object

      DataSet data_set;

      data_set.set_data_file_name("../data/simplefunctionregression.dat");

      data_set.load_data();

	  Variables* variables_pointer = data_set.get_variables_pointer();

      variables_pointer->set_use(0, Variables::Input);
      variables_pointer->set_use(1, Variables::Target);

      variables_pointer->set_name(0, "x");
      variables_pointer->set_name(1, "y");

      Matrix<std::string> inputs_information = variables_pointer->arrange_inputs_information();
      Matrix<std::string> targets_information = variables_pointer->arrange_targets_information();

	  Instances* instances_pointer = data_set.get_instances_pointer();

      instances_pointer->set_training();

      Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
      Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();

      // Neural network

      NeuralNetwork neural_network(1, 15, 1);

      Inputs* inputs_pointer = neural_network.get_inputs_pointer();
      inputs_pointer->set_information(inputs_information);

      Outputs* outputs_pointer = neural_network.get_outputs_pointer();
      outputs_pointer->set_information(targets_information);

      neural_network.construct_scaling_layer();
      ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
      scaling_layer_pointer->set_statistics(inputs_statistics);
      scaling_layer_pointer->set_scaling_method(ScalingLayer::NoScaling);

      neural_network.construct_unscaling_layer();
      UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
      unscaling_layer_pointer->set_statistics(targets_statistics);
      unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);

      // Performance functional object
 
      PerformanceFunctional performance_functional(&neural_network, &data_set);

      // Training strategy

      TrainingStrategy training_strategy(&performance_functional);

      QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

      quasi_Newton_method_pointer->set_minimum_performance_increase(1.0e-3);

      TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

      // Testing analysis object

      instances_pointer->set_testing();

      TestingAnalysis testing_analysis(&neural_network, &data_set);

      TestingAnalysis::LinearRegressionResults linear_regression_results = testing_analysis.perform_linear_regression_analysis();
      
      // Save results

      scaling_layer_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);
      unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MinimumMaximum);

      data_set.save("../data/data_set.xml");

      neural_network.save("../data/neural_network.xml");
      neural_network.save_expression("../data/expression.txt");

      performance_functional.save("../data/performance_functional.xml");

      training_strategy.save("../data/training_strategy.xml");
      training_strategy_results.save("../data/training_strategy_results.dat");

      linear_regression_results.save("../data/linear_regression_analysis_results.dat");

      return(0);
   }
   catch(std::exception& e)
   {
      std::cerr << e.what() << std::endl;

      return(1);
   }
}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2015 Roberto Lopez
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
