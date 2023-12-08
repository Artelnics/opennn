//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   T E S T S   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <time.h>

// OpenNN tests includes

#include "opennn_tests.h"
#include "../opennn/opennn.h"

using namespace opennn;

const std::array test_names{
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("adaptive_moment_estimation", "adam", unique_ptr<UnitTesting>(new AdaptiveMomentEstimationTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("bounding_layer", "bl", unique_ptr<UnitTesting>(new BoundingLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("conjugate_gradient", "cg", unique_ptr<UnitTesting>(new ConjugateGradientTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("correlations", "cr", unique_ptr<UnitTesting>(new CorrelationsTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("cross_entropy_error", "cee", unique_ptr<UnitTesting>(new CrossEntropyErrorTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("convolutional_layer", "cl", unique_ptr<UnitTesting>(new ConvolutionalLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("descriptives", "dsc", unique_ptr<UnitTesting>(new StatisticsTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("data_set", "ds", unique_ptr<UnitTesting>(new DataSetTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("flatten_layer", "fl", unique_ptr<UnitTesting>(new FlattenLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("genetic_algorithm", "ga", unique_ptr<UnitTesting>(new GeneticAlgorithmTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("gradient_descent", "gd", unique_ptr<UnitTesting>(new GradientDescentTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("growing_inputs", "gi", unique_ptr<UnitTesting>(new GrowingInputsTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("growing_neurons", "gn", unique_ptr<UnitTesting>(new GrowingNeuronsTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("inputs_selection", "is", unique_ptr<UnitTesting>(new InputsSelectionTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("learning_rate_algorithm", "lra", unique_ptr<UnitTesting>(new LearningRateAlgorithmTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("levenberg_marquardt_algorithm", "lma", unique_ptr<UnitTesting>(new LevenbergMarquardtAlgorithmTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("long_short_term_memory_layer", "lstm", unique_ptr<UnitTesting>(new LongShortTermMemoryLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("mean_squared_error", "mse", unique_ptr<UnitTesting>(new MeanSquaredErrorTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("minkowski_error", "me", unique_ptr<UnitTesting>(new MinkowskiErrorTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("model_selection", "ms", unique_ptr<UnitTesting>(new ModelSelectionTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("neural_network", "nn", unique_ptr<UnitTesting>(new NeuralNetworkTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("neurons_selection", "ns", unique_ptr<UnitTesting>(new NeuronsSelectionTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("normalized_squared_error", "nse", unique_ptr<UnitTesting>(new NormalizedSquaredErrorTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("numerical_differentiation", "nd", unique_ptr<UnitTesting>(new NumericalDifferentiationTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("perceptron_layer", "pl", unique_ptr<UnitTesting>(new PerceptronLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("pooling_layer", "pll", unique_ptr<UnitTesting>(new PoolingLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("probabilistic_layer", "pbl", unique_ptr<UnitTesting>(new ProbabilisticLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("quasi_newton_method", "qnm", unique_ptr<UnitTesting>(new QuasiNewtonMethodTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("recurrent_layer", "rl", unique_ptr<UnitTesting>(new RecurrentLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("response_optimization", "ro", unique_ptr<UnitTesting>(new ResponseOptimizationTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("scaling_layer", "sl", unique_ptr<UnitTesting>(new ScalingLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("scaling", "sc", unique_ptr<UnitTesting>(new ScalingTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("statistics", "st", unique_ptr<UnitTesting>(new StatisticsTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("stochastic_gradient_descent", "sgd", unique_ptr<UnitTesting>(new StochasticGradientDescentTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("sum_squared_error", "sse", unique_ptr<UnitTesting>(new SumSquaredErrorTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("tensor_utilities", "tu", unique_ptr<UnitTesting>(new TensorUtilitiesTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("testing_analysis", "ta", unique_ptr<UnitTesting>(new TestingAnalysisTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("training_strategy", "ts", unique_ptr<UnitTesting>(new TrainingStrategyTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("unscaling_layer", "ul", unique_ptr<UnitTesting>(new UnscalingLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("weighted_squared_error", "wse", unique_ptr<UnitTesting>(new WeightedSquaredErrorTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("conv_pooling_layer", "cpl", unique_ptr<UnitTesting>(new ConvolutionalPoolingLayerTest{})),
  make_tuple<string_view, string_view, unique_ptr<UnitTesting>>("flatten_pooling_layer", "fpl", unique_ptr<UnitTesting>(new FlattenPoolingLayerTest{})),
};

UnitTesting* get_test_unit(string_view test_name)
{
  auto it = find_if(begin(test_names), end(test_names), [test_name](const auto& name_pair){
    string_view full_name = get<0>(name_pair);
    string_view acr = get<1>(name_pair);
    return test_name == full_name || acr == test_name;
  });
  if(it != end(test_names))
  {
    return get<2>(*it).get();
  }
  return nullptr;
}

int main()
{
   cout <<
   "Open Neural Networks Library. Test Suite Application.\n\n"

   "suite - run all tests\n\n"

   "Individual Tests:\n\n";
   for_each(begin(test_names), end(test_names), [](const auto& name_pair){
    string_view full_name = get<0>(name_pair);
    string_view acr = get<1>(name_pair);
    cout << full_name << " | " << acr << '\n';
   });

   string test;

   cout << "Test: ";

   cin >> test;

   // Redirect standard output to file

   //ofstream out("../data/out.txt");
   //cout.rdbuf(out.rdbuf());

   try
   {
      srand(static_cast<unsigned>(time(nullptr)));

      string message;

      Index tests_count = 0;
      Index tests_passed_count = 0;
      Index tests_failed_count = 0;
      auto perform_test = [&tests_count, &tests_passed_count, &tests_failed_count](auto& unit)
      {
        unit->run_test_case();
        tests_count += unit->get_tests_count();
        tests_passed_count += unit->get_tests_passed_count();
        tests_failed_count += unit->get_tests_failed_count();
      };
      if(test == "suite" || test == "")
      {
        for(auto& [_, __, unit_test] : test_names)
        {
          perform_test(unit_test);
        }
      }
      else
      {
        UnitTesting* unit_test = get_test_unit(test);
        if(unit_test == nullptr)
        {
          cout << "Unknown test: " << test << endl;

          return 1;
        }
        perform_test(unit_test);
      }
      cout << message << "\n"
                << "OpenNN test suite results:\n"
                << "Tests run: " << tests_count << "\n"
                << "Tests passed: " << tests_passed_count << "\n"
                << "Tests failed: " << tests_failed_count << "\n";

      if(tests_failed_count == 0)
      {
         cout << "Test OK" << endl;
      }
      else
      {
         cout << "Test NOT OK. " << tests_failed_count << " tests failed" << endl;
      }

      return 0;
   }
   catch(const exception& e)
   {
      cerr << e.what() << endl;

      return 1;
   }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
