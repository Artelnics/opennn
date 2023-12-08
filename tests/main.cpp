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

struct unit_test_storing
{
  template<typename UnitTest>
  unit_test_storing(UnitTest _) : ptr(make_unique<UnitTest>())
  {}

  UnitTesting* get_ptr() const
  {
    return ptr.get();
  }
private:
  unique_ptr<UnitTesting> ptr;
};

const std::array test_names{
  make_tuple<string_view, string_view, unit_test_storing>("adaptive_moment_estimation", "adam", unit_test_storing(AdaptiveMomentEstimationTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("bounding_layer", "bl", unit_test_storing(BoundingLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("conjugate_gradient", "cg", unit_test_storing(ConjugateGradientTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("correlations", "cr", unit_test_storing(CorrelationsTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("cross_entropy_error", "cee", unit_test_storing(CrossEntropyErrorTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("convolutional_layer", "cl", unit_test_storing(ConvolutionalLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("descriptives", "dsc", unit_test_storing(StatisticsTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("data_set", "ds", unit_test_storing(DataSetTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("flatten_layer", "fl", unit_test_storing(FlattenLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("genetic_algorithm", "ga", unit_test_storing(GeneticAlgorithmTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("gradient_descent", "gd", unit_test_storing(GradientDescentTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("growing_inputs", "gi", unit_test_storing(GrowingInputsTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("growing_neurons", "gn", unit_test_storing(GrowingNeuronsTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("inputs_selection", "is", unit_test_storing(InputsSelectionTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("learning_rate_algorithm", "lra", unit_test_storing(LearningRateAlgorithmTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("levenberg_marquardt_algorithm", "lma", unit_test_storing(LevenbergMarquardtAlgorithmTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("long_short_term_memory_layer", "lstm", unit_test_storing(LongShortTermMemoryLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("mean_squared_error", "mse", unit_test_storing(MeanSquaredErrorTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("minkowski_error", "me", unit_test_storing(MinkowskiErrorTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("model_selection", "ms", unit_test_storing(ModelSelectionTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("neural_network", "nn", unit_test_storing(NeuralNetworkTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("neurons_selection", "ns", unit_test_storing(NeuronsSelectionTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("normalized_squared_error", "nse", unit_test_storing(NormalizedSquaredErrorTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("numerical_differentiation", "nd", unit_test_storing(NumericalDifferentiationTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("perceptron_layer", "pl", unit_test_storing(PerceptronLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("pooling_layer", "pll", unit_test_storing(PoolingLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("probabilistic_layer", "pbl", unit_test_storing(ProbabilisticLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("quasi_newton_method", "qnm", unit_test_storing(QuasiNewtonMethodTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("recurrent_layer", "rl", unit_test_storing(RecurrentLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("response_optimization", "ro", unit_test_storing(ResponseOptimizationTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("scaling_layer", "sl", unit_test_storing(ScalingLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("scaling", "sc", unit_test_storing(ScalingTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("statistics", "st", unit_test_storing(StatisticsTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("stochastic_gradient_descent", "sgd", unit_test_storing(StochasticGradientDescentTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("sum_squared_error", "sse", unit_test_storing(SumSquaredErrorTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("tensor_utilities", "tu", unit_test_storing(TensorUtilitiesTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("testing_analysis", "ta", unit_test_storing(TestingAnalysisTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("training_strategy", "ts", unit_test_storing(TrainingStrategyTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("unscaling_layer", "ul", unit_test_storing(UnscalingLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("weighted_squared_error", "wse", unit_test_storing(WeightedSquaredErrorTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("conv_pooling_layer", "cpl", unit_test_storing(ConvolutionalPoolingLayerTest{})),
  make_tuple<string_view, string_view, unit_test_storing>("flatten_pooling_layer", "fpl", unit_test_storing(FlattenPoolingLayerTest{}))
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
    return get<2>(*it).get_ptr();
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
      auto perform_test = [&tests_count, &tests_passed_count, &tests_failed_count](UnitTesting* unit)
      {
        unit->run_test_case();
        tests_count += unit->get_tests_count();
        tests_passed_count += unit->get_tests_passed_count();
        tests_failed_count += unit->get_tests_failed_count();
      };
      if(test == "suite" || test == "")
      {
        for(auto& [_, __, unit_test_str] : test_names)
        {
          UnitTesting* unit_test = unit_test_str.get_ptr();
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
