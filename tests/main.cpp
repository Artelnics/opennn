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
#include "unit_testing.h"

using namespace OpenNN;

int main()
{
   cout <<
   "Open Neural Networks Library. Test Suite Application.\n"
   "bounding_layer\n"
   "conjugate_gradient\n"
   "correlations\n"
   "cross_entropy_error\n"
   "descriptives\n"
   "evolutionary_algorithm\n"
   "functions\n"
   "genetic_algorithm\n"
   "golden_section_order\n"
   "gradient_descent\n"
   "growing_inputs\n"
   "incremental_order\n"
   "instances\n"
   "inputs\n"
   "inputs_selection_algorithm\n"
   "learning_rate_algorithm\n"
   "levenberg_marquardt_algorithm\n"
   "linear_algebra\n"
   "long_short_term_memory_layer\n"
   "loss_index\n"
   "matrix\n"
   "mean_squared_error\n"
   "minkowski_error\n"
   "missing_values\n"
   "model_selection\n"
   "neural_network\n"
   "newton_method\n"
   "normalized_squared_error\n"
   "numerical_differentiation\n"
   "optimization_algorithm\n"
   "neurons_selection_algorithm\n"
   "outputs\n"
   "perceptron_layer\n"
   "probabilistic_layer\n"
   "pruning_inputs\n"
   "quasi_newton_method\n"
   "recurrent_layer\n"
   "scaling_layer\n"
   "simulated_annealing_order\n"
   "suite"
   "sum_squared_error\n"
   "testing_analysis\n"
   "training_strategy\n"
   "tensor\n"
   "unscaling_layer\n"
   "variables\n"
   "vector\n"
   "weighted_squared_error\n"
   "Write test:\n"<< endl;

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

      size_t tests_count = 0;
      size_t tests_passed_count = 0;
      size_t tests_failed_count = 0;

      if(test == "correlations" || test == "")
      {
         CorrelationsTest correlations_test;
         correlations_test.run_test_case();
         tests_count += correlations_test.get_tests_count();
         tests_passed_count += correlations_test.get_tests_passed_count();
         tests_failed_count += correlations_test.get_tests_failed_count();
      }

      else if(test == "data_set" || test == "ds")
      {
         DataSetTest data_set_test;
         data_set_test.run_test_case();
         tests_count += data_set_test.get_tests_count();
         tests_passed_count += data_set_test.get_tests_passed_count();
         tests_failed_count += data_set_test.get_tests_failed_count();
      }

      else if(test == "linear_algebra" || test == "")
      {
         MetricsTest test;
         test.run_test_case();
         tests_count += test.get_tests_count();
         tests_passed_count += test.get_tests_passed_count();
         tests_failed_count += test.get_tests_failed_count();
      }

      else if(test == "matrix" || test == "m")
      {
         MatrixTest matrix_test;
         matrix_test.run_test_case();
         tests_count += matrix_test.get_tests_count();
         tests_passed_count += matrix_test.get_tests_passed_count();
         tests_failed_count += matrix_test.get_tests_failed_count();
      }

      else if(test == "numerical_differentiation" || test == "")
      {
         NumericalDifferentiationTest test_numerical_differentiation;
         test_numerical_differentiation.run_test_case();
         tests_count += test_numerical_differentiation.get_tests_count();
         tests_passed_count += test_numerical_differentiation.get_tests_passed_count();
         tests_failed_count += test_numerical_differentiation.get_tests_failed_count();
      }

      else if(test == "perceptron_layer" || test == "pl")
      {
         PerceptronLayerTest perceptron_layer_test;
         perceptron_layer_test.run_test_case();
         tests_count += perceptron_layer_test.get_tests_count();
         tests_passed_count += perceptron_layer_test.get_tests_passed_count();
         tests_failed_count += perceptron_layer_test.get_tests_failed_count();
      }

      else if(test == "statistics" || test == "")
      {
         StatisticsTest matrix_test;
         matrix_test.run_test_case();
         tests_count += matrix_test.get_tests_count();
         tests_passed_count += matrix_test.get_tests_passed_count();
         tests_failed_count += matrix_test.get_tests_failed_count();
      }

      else if(test == "vector" || test == "v")
      {
         VectorTest vector_test;
         vector_test.run_test_case();
         tests_count += vector_test.get_tests_count();
         tests_passed_count += vector_test.get_tests_passed_count();
         tests_failed_count += vector_test.get_tests_failed_count();
      }

      else if(test == "functions" || test == "f")
      {
         FunctionsTest functions_test;
         functions_test.run_test_case();
         tests_count += functions_test.get_tests_count();
         tests_passed_count += functions_test.get_tests_passed_count();
         tests_failed_count += functions_test.get_tests_failed_count();
      }

      else if(test == "long_short_term_memory_layer" || test == "lstml")
      {
         LongShortTermMemoryLayerTest long_short_memory_layer_test;

         long_short_memory_layer_test.run_test_case();

         tests_count += long_short_memory_layer_test.get_tests_count();

         tests_passed_count += long_short_memory_layer_test.get_tests_passed_count();

         tests_failed_count += long_short_memory_layer_test.get_tests_failed_count();

      }
      else if(test == "recurrent_layer" || test == "rl")
      {
         RecurrentLayerTest recurrent_layer_test;
         recurrent_layer_test.run_test_case();
         tests_count += recurrent_layer_test.get_tests_count();
         tests_passed_count += recurrent_layer_test.get_tests_passed_count();
         tests_failed_count += recurrent_layer_test.get_tests_failed_count();
      }
      else if(test == "scaling_layer" || test == "")
      {
         ScalingLayerTest scaling_layer_test;
         scaling_layer_test.run_test_case();
         tests_count += scaling_layer_test.get_tests_count();
         tests_passed_count += scaling_layer_test.get_tests_passed_count();
         tests_failed_count += scaling_layer_test.get_tests_failed_count();
      }
      else if(test == "unscaling_layer" || test == "")
      {
         UnscalingLayerTest unscaling_layer_test;
         unscaling_layer_test.run_test_case();
         tests_count += unscaling_layer_test.get_tests_count();
         tests_passed_count += unscaling_layer_test.get_tests_passed_count();
         tests_failed_count += unscaling_layer_test.get_tests_failed_count();
      }
      else if(test == "bounding_layer" || test == "")
      {
         BoundingLayerTest bounding_layer_test;
         bounding_layer_test.run_test_case();
         tests_count += bounding_layer_test.get_tests_count();
         tests_passed_count += bounding_layer_test.get_tests_passed_count();
         tests_failed_count += bounding_layer_test.get_tests_failed_count();
      }
      else if(test == "probabilistic_layer" || test == "")
      {
         ProbabilisticLayerTest probabilistic_layer_test;
         probabilistic_layer_test.run_test_case();
         tests_count += probabilistic_layer_test.get_tests_count();
         tests_passed_count += probabilistic_layer_test.get_tests_passed_count();
         tests_failed_count += probabilistic_layer_test.get_tests_failed_count();
      }
      else if(test == "convolutional_layer" || test == "cl")
      {
         ConvolutionalLayerTest layer_test;
         layer_test.run_test_case();
         tests_count += layer_test.get_tests_count();
         tests_passed_count += layer_test.get_tests_passed_count();
         tests_failed_count += layer_test.get_tests_failed_count();
      }
      else if(test == "pooling_layer" || test == "")
      {
         PoolingLayerTest layer_test;
         layer_test.run_test_case();
         tests_count += layer_test.get_tests_count();
         tests_passed_count += layer_test.get_tests_passed_count();
         tests_failed_count += layer_test.get_tests_failed_count();
      }
      else if(test == "neural_network" || test == "nn")
      {
        NeuralNetworkTest neural_network_test;
        neural_network_test.run_test_case();
        tests_count += neural_network_test.get_tests_count();
        tests_passed_count += neural_network_test.get_tests_passed_count();
        tests_failed_count += neural_network_test.get_tests_failed_count();
      }
      else if(test == "sum_squared_error" || test == "sse" || test == "SSE")
      {
        SumSquaredErrorTest sum_squared_error_test;
        sum_squared_error_test.run_test_case();
        tests_count += sum_squared_error_test.get_tests_count();
        tests_passed_count += sum_squared_error_test.get_tests_passed_count();
        tests_failed_count += sum_squared_error_test.get_tests_failed_count();
      }
      else if(test == "mean_squared_error" || test == "mse")
      {
        MeanSquaredErrorTest mean_squared_error_test;
        mean_squared_error_test.run_test_case();
        tests_count += mean_squared_error_test.get_tests_count();
        tests_passed_count += mean_squared_error_test.get_tests_passed_count();
        tests_failed_count += mean_squared_error_test.get_tests_failed_count();
      }
      else if(test == "normalized_squared_error" || test == "nse")
      {
        NormalizedSquaredErrorTest normalized_squared_error_test;
        normalized_squared_error_test.run_test_case();
        tests_count += normalized_squared_error_test.get_tests_count();
        tests_passed_count += normalized_squared_error_test.get_tests_passed_count();
        tests_failed_count += normalized_squared_error_test.get_tests_failed_count();
      }
      else if(test == "weighted_squared_error" || test == "wse")
      {
        WeightedSquaredErrorTest weighted_squared_error_test;
        weighted_squared_error_test.run_test_case();
        tests_count += weighted_squared_error_test.get_tests_count();
        tests_passed_count += weighted_squared_error_test.get_tests_passed_count();
        tests_failed_count += weighted_squared_error_test.get_tests_failed_count();
      }
      else if(test == "minkowski_error" || test == "me")
      {
        MinkowskiErrorTest Minkowski_error_test;
        Minkowski_error_test.run_test_case();
        tests_count += Minkowski_error_test.get_tests_count();
        tests_passed_count += Minkowski_error_test.get_tests_passed_count();
        tests_failed_count += Minkowski_error_test.get_tests_failed_count();
      }
      else if(test == "cross_entropy_error" || test == "cee")
      {
        CrossEntropyErrorTest cross_entropy_error_test;
        cross_entropy_error_test.run_test_case();
        tests_count += cross_entropy_error_test.get_tests_count();
        tests_passed_count += cross_entropy_error_test.get_tests_passed_count();
        tests_failed_count += cross_entropy_error_test.get_tests_failed_count();
      }
      else if(test == "descriptives" || test == "")
      {
        StatisticsTest statistics_test;
        statistics_test.run_test_case();
        tests_count += statistics_test.get_tests_count();
        tests_passed_count += statistics_test.get_tests_passed_count();
        tests_failed_count += statistics_test.get_tests_failed_count();
      }
//      else if(test == "functions" || test == "")
//      {
//        FunctionsTest functions_test;
//        functions_test.run_test_case();
//        tests_count += functions_test.get_tests_count();
//        tests_passed_count += functions_test.get_tests_passed_count();
//        tests_failed_count += functions_test.get_tests_failed_count();
//      }
      else if(test == "learning_rate_algorithm" || test == "")
      {
        LearningRateAlgorithmTest learning_rate_algorithm_test;
        learning_rate_algorithm_test.run_test_case();
        tests_count += learning_rate_algorithm_test.get_tests_count();
        tests_passed_count += learning_rate_algorithm_test.get_tests_passed_count();
        tests_failed_count += learning_rate_algorithm_test.get_tests_failed_count();
      }
      else if(test == "gradient_descent" || test == "")
      {
        GradientDescentTest gradient_descent_test;
        gradient_descent_test.run_test_case();
        tests_count += gradient_descent_test.get_tests_count();
        tests_passed_count += gradient_descent_test.get_tests_passed_count();
        tests_failed_count += gradient_descent_test.get_tests_failed_count();
      }
      else if(test == "conjugate_gradient" || test == "")
      {
        ConjugateGradientTest conjugate_gradient_test;
        conjugate_gradient_test.run_test_case();
        tests_count += conjugate_gradient_test.get_tests_count();
        tests_passed_count += conjugate_gradient_test.get_tests_passed_count();
        tests_failed_count += conjugate_gradient_test.get_tests_failed_count();
      }
      else if(test == "quasi_newton_method" || test == "")
      {
        QuasiNewtonMethodTest quasi_Newton_method_test;
        quasi_Newton_method_test.run_test_case();
        tests_count += quasi_Newton_method_test.get_tests_count();
        tests_passed_count += quasi_Newton_method_test.get_tests_passed_count();
        tests_failed_count += quasi_Newton_method_test.get_tests_failed_count();
      }
      else if(test == "levenberg_marquardt_algorithm" || test == "")
      {
        LevenbergMarquardtAlgorithmTest Levenberg_Marquardt_algorithm_test;
        Levenberg_Marquardt_algorithm_test.run_test_case();
        tests_count += Levenberg_Marquardt_algorithm_test.get_tests_count();
        tests_passed_count += Levenberg_Marquardt_algorithm_test.get_tests_passed_count();
        tests_failed_count += Levenberg_Marquardt_algorithm_test.get_tests_failed_count();
      }
      else if(test == "stochastic_gradient_descent" || test == "")
      {
        StochasticGradientDescentTest stochastic_gradient_descent_test;
        stochastic_gradient_descent_test.run_test_case();
        tests_count += stochastic_gradient_descent_test.get_tests_count();
        tests_passed_count += stochastic_gradient_descent_test.get_tests_passed_count();
        tests_failed_count += stochastic_gradient_descent_test.get_tests_failed_count();
      }
      else if(test == "training_strategy" || test == "ts")
      {
        TrainingStrategyTest training_strategy_test;
        training_strategy_test.run_test_case();
        tests_count += training_strategy_test.get_tests_count();
        tests_passed_count += training_strategy_test.get_tests_passed_count();
        tests_failed_count += training_strategy_test.get_tests_failed_count();
      }

      else if(test == "model_selection" || test == "ms")
      {
        ModelSelectionTest model_selection_test;
        model_selection_test.run_test_case();
        tests_count += model_selection_test.get_tests_count();
        tests_passed_count += model_selection_test.get_tests_passed_count();
        tests_failed_count += model_selection_test.get_tests_failed_count();
      }

      else if(test == "neurons_selection" || test == "ns")
      {
        NeuronsSelectionTest neurons_selection_algorithm_test;
        neurons_selection_algorithm_test.run_test_case();
        tests_count += neurons_selection_algorithm_test.get_tests_count();
        tests_passed_count += neurons_selection_algorithm_test.get_tests_passed_count();
        tests_failed_count += neurons_selection_algorithm_test.get_tests_failed_count();
      }

      else if(test == "incremental_neurons" || test == "in")
      {
        IncrementalNeuronsTest incremental_order_test;
        incremental_order_test.run_test_case();
        tests_count += incremental_order_test.get_tests_count();
        tests_passed_count += incremental_order_test.get_tests_passed_count();
        tests_failed_count += incremental_order_test.get_tests_failed_count();
      }

      else if(test == "inputs_selection" || test == "is")
      {
        InputsSelectionTest inputs_selection_algorithm_test;
        inputs_selection_algorithm_test.run_test_case();
        tests_count += inputs_selection_algorithm_test.get_tests_count();
        tests_passed_count += inputs_selection_algorithm_test.get_tests_passed_count();
        tests_failed_count += inputs_selection_algorithm_test.get_tests_failed_count();
      }

      else if(test == "growing_inputs" || test == "gi")
      {
        GrowingInputsTest growing_inputs_test;
        growing_inputs_test.run_test_case();
        tests_count += growing_inputs_test.get_tests_count();
        tests_passed_count += growing_inputs_test.get_tests_passed_count();
        tests_failed_count += growing_inputs_test.get_tests_failed_count();
      }

      else if(test == "pruning_inputs" || test == "pi")
      {
        PruningInputsTest pruning_inputs_test;
        pruning_inputs_test.run_test_case();
        tests_count += pruning_inputs_test.get_tests_count();
        tests_passed_count += pruning_inputs_test.get_tests_passed_count();
        tests_failed_count += pruning_inputs_test.get_tests_failed_count();
      }

      else if(test == "genetic_algorithm" || test == "ga")
      {
        GeneticAlgorithmTest genetic_algorithm_test;
        genetic_algorithm_test.run_test_case();
        tests_count += genetic_algorithm_test.get_tests_count();
        tests_passed_count += genetic_algorithm_test.get_tests_passed_count();
        tests_failed_count += genetic_algorithm_test.get_tests_failed_count();
      }

      else if(test == "testing_analysis" || test == "ta")
      {
        TestingAnalysisTest testing_analysis_test;
        testing_analysis_test.run_test_case();
        tests_count += testing_analysis_test.get_tests_count();
        tests_passed_count += testing_analysis_test.get_tests_passed_count();
        tests_failed_count += testing_analysis_test.get_tests_failed_count();
      }

      else if(test == "tensor" || test == "tn")
      {
        TensorTest tensor_test;
        tensor_test.run_test_case();
        tests_count += tensor_test.get_tests_count();
        tests_passed_count += tensor_test.get_tests_passed_count();
        tests_failed_count += tensor_test.get_tests_failed_count();
      }

      else if(test == "suite" || test == "")
      {
          // vector

          VectorTest vector_test;
          vector_test.run_test_case();
          tests_count += vector_test.get_tests_count();
          tests_passed_count += vector_test.get_tests_passed_count();
          tests_failed_count += vector_test.get_tests_failed_count();

          // matrix

          MatrixTest matrix_test;
          matrix_test.run_test_case();
          tests_count += matrix_test.get_tests_count();
          tests_passed_count += matrix_test.get_tests_passed_count();
          tests_failed_count += matrix_test.get_tests_failed_count();

          // tensor

          TensorTest tensor_test;
          tensor_test.run_test_case();
          tests_count += tensor_test.get_tests_count();
          tests_passed_count += tensor_test.get_tests_passed_count();
          tests_failed_count += tensor_test.get_tests_failed_count();

          //functions

          FunctionsTest functions_test;
          functions_test.run_test_case();
          tests_count += functions_test.get_tests_count();
          tests_passed_count += functions_test.get_tests_passed_count();
          tests_failed_count += functions_test.get_tests_failed_count();

          // numerical differentiation

          NumericalDifferentiationTest test_numerical_differentiation;
          test_numerical_differentiation.run_test_case();
          tests_count += test_numerical_differentiation.get_tests_count();
          tests_passed_count += test_numerical_differentiation.get_tests_passed_count();
          tests_failed_count += test_numerical_differentiation.get_tests_failed_count();

          // D A T A   S E T   T E S T S

          // correlation analysis

          CorrelationsTest correlations_test;
          correlations_test.run_test_case();
          tests_count += correlations_test.get_tests_count();
          tests_passed_count += correlations_test.get_tests_passed_count();
          tests_failed_count += correlations_test.get_tests_failed_count();

          // data set

          DataSetTest data_set_test;
          data_set_test.run_test_case();
          tests_count += data_set_test.get_tests_count();
          tests_passed_count += data_set_test.get_tests_passed_count();
          tests_failed_count += data_set_test.get_tests_failed_count();

          // N E U R A L   N E T W O R K   T E S T S

          // perceptron layer

          PerceptronLayerTest perceptron_layer_test;
          perceptron_layer_test.run_test_case();
          tests_count += perceptron_layer_test.get_tests_count();
          tests_passed_count += perceptron_layer_test.get_tests_passed_count();
          tests_failed_count += perceptron_layer_test.get_tests_failed_count();

          // scaling layer

          ScalingLayerTest scaling_layer_test;
          scaling_layer_test.run_test_case();
          tests_count += scaling_layer_test.get_tests_count();
          tests_passed_count += scaling_layer_test.get_tests_passed_count();
          tests_failed_count += scaling_layer_test.get_tests_failed_count();

          // unscaling layer

          UnscalingLayerTest unscaling_layer_test;
          unscaling_layer_test.run_test_case();
          tests_count += unscaling_layer_test.get_tests_count();
          tests_passed_count += unscaling_layer_test.get_tests_passed_count();
          tests_failed_count += unscaling_layer_test.get_tests_failed_count();

          // bounding layer

          BoundingLayerTest bounding_layer_test;
          bounding_layer_test.run_test_case();
          tests_count += bounding_layer_test.get_tests_count();
          tests_passed_count += bounding_layer_test.get_tests_passed_count();
          tests_failed_count += bounding_layer_test.get_tests_failed_count();

          // probabilistic layer

          ProbabilisticLayerTest probabilistic_layer_test;
          probabilistic_layer_test.run_test_case();
          tests_count += probabilistic_layer_test.get_tests_count();
          tests_passed_count += probabilistic_layer_test.get_tests_passed_count();
          tests_failed_count += probabilistic_layer_test.get_tests_failed_count();

          // neural network

          NeuralNetworkTest neural_network_test;
          neural_network_test.run_test_case();
          tests_count += neural_network_test.get_tests_count();
          tests_passed_count += neural_network_test.get_tests_passed_count();
          tests_failed_count += neural_network_test.get_tests_failed_count();

          // L O S S   I N D E X   T E S T S

          // sum squared error

          SumSquaredErrorTest sum_squared_error_test;
          sum_squared_error_test.run_test_case();
          tests_count += sum_squared_error_test.get_tests_count();
          tests_passed_count += sum_squared_error_test.get_tests_passed_count();
          tests_failed_count += sum_squared_error_test.get_tests_failed_count();

          // mean squared error

          MeanSquaredErrorTest mean_squared_error_test;
          mean_squared_error_test.run_test_case();
          tests_count += mean_squared_error_test.get_tests_count();
          tests_passed_count += mean_squared_error_test.get_tests_passed_count();
          tests_failed_count += mean_squared_error_test.get_tests_failed_count();

          // normalized squared error

          NormalizedSquaredErrorTest normalized_squared_error_test;
          normalized_squared_error_test.run_test_case();
          tests_count += normalized_squared_error_test.get_tests_count();
          tests_passed_count += normalized_squared_error_test.get_tests_passed_count();
          tests_failed_count += normalized_squared_error_test.get_tests_failed_count();

          // minkowski error

          MinkowskiErrorTest Minkowski_error_test;
          Minkowski_error_test.run_test_case();
          tests_count += Minkowski_error_test.get_tests_count();
          tests_passed_count += Minkowski_error_test.get_tests_passed_count();
          tests_failed_count += Minkowski_error_test.get_tests_failed_count();

          // cross entropy error

          CrossEntropyErrorTest cross_entropy_error_test;
          cross_entropy_error_test.run_test_case();
          tests_count += cross_entropy_error_test.get_tests_count();
          tests_passed_count += cross_entropy_error_test.get_tests_passed_count();
          tests_failed_count += cross_entropy_error_test.get_tests_failed_count();

          // T R A I N I N G   S T R A T E G Y   T E S T S

          // learning rate algorithm

          LearningRateAlgorithmTest learning_rate_algorithm_test;
          learning_rate_algorithm_test.run_test_case();
          tests_count += learning_rate_algorithm_test.get_tests_count();
          tests_passed_count += learning_rate_algorithm_test.get_tests_passed_count();
          tests_failed_count += learning_rate_algorithm_test.get_tests_failed_count();

          // gradient descent

          GradientDescentTest gradient_descent_test;
          gradient_descent_test.run_test_case();
          tests_count += gradient_descent_test.get_tests_count();
          tests_passed_count += gradient_descent_test.get_tests_passed_count();
          tests_failed_count += gradient_descent_test.get_tests_failed_count();

          // conjugate gradient

          ConjugateGradientTest conjugate_gradient_test;
          conjugate_gradient_test.run_test_case();
          tests_count += conjugate_gradient_test.get_tests_count();
          tests_passed_count += conjugate_gradient_test.get_tests_passed_count();
          tests_failed_count += conjugate_gradient_test.get_tests_failed_count();

          // quasi newton method

          QuasiNewtonMethodTest quasi_Newton_method_test;
          quasi_Newton_method_test.run_test_case();
          tests_count += quasi_Newton_method_test.get_tests_count();
          tests_passed_count += quasi_Newton_method_test.get_tests_passed_count();
          tests_failed_count += quasi_Newton_method_test.get_tests_failed_count();

          // levenberg marquardt algorithm

          LevenbergMarquardtAlgorithmTest Levenberg_Marquardt_algorithm_test;
          Levenberg_Marquardt_algorithm_test.run_test_case();
          tests_count += Levenberg_Marquardt_algorithm_test.get_tests_count();
          tests_passed_count += Levenberg_Marquardt_algorithm_test.get_tests_passed_count();
          tests_failed_count += Levenberg_Marquardt_algorithm_test.get_tests_failed_count();


          StochasticGradientDescentTest stochastic_gradient_descent_test;
          stochastic_gradient_descent_test.run_test_case();
          tests_count += stochastic_gradient_descent_test.get_tests_count();
          tests_passed_count += stochastic_gradient_descent_test.get_tests_passed_count();
          tests_failed_count += stochastic_gradient_descent_test.get_tests_failed_count();

          // training_strategy

          TrainingStrategyTest training_strategy_test;
          training_strategy_test.run_test_case();
          tests_count += training_strategy_test.get_tests_count();
          tests_passed_count += training_strategy_test.get_tests_passed_count();
          tests_failed_count += training_strategy_test.get_tests_failed_count();

          // M O D E L   S E L E C T I O N   T E S T S

          // model selection

          ModelSelectionTest model_selection_test;
          model_selection_test.run_test_case();
          tests_count += model_selection_test.get_tests_count();
          tests_passed_count += model_selection_test.get_tests_passed_count();
          tests_failed_count += model_selection_test.get_tests_failed_count();

          // order selection algorithm

          NeuronsSelectionTest neurons_selection_algorithm_test;
          neurons_selection_algorithm_test.run_test_case();
          tests_count += neurons_selection_algorithm_test.get_tests_count();
          tests_passed_count += neurons_selection_algorithm_test.get_tests_passed_count();
          tests_failed_count += neurons_selection_algorithm_test.get_tests_failed_count();

          // incremental order

          IncrementalNeuronsTest incremental_order_test;
          incremental_order_test.run_test_case();
          tests_count += incremental_order_test.get_tests_count();
          tests_passed_count += incremental_order_test.get_tests_passed_count();
          tests_failed_count += incremental_order_test.get_tests_failed_count();

          // input selection algorithm

          InputsSelectionTest inputs_selection_algorithm_test;
          inputs_selection_algorithm_test.run_test_case();
          tests_count += inputs_selection_algorithm_test.get_tests_count();
          tests_passed_count += inputs_selection_algorithm_test.get_tests_passed_count();
          tests_failed_count += inputs_selection_algorithm_test.get_tests_failed_count();

          // growing_inputs

          GrowingInputsTest growing_inputs_test;
          growing_inputs_test.run_test_case();
          tests_count += growing_inputs_test.get_tests_count();
          tests_passed_count += growing_inputs_test.get_tests_passed_count();
          tests_failed_count += growing_inputs_test.get_tests_failed_count();

          // pruning_inputs

          PruningInputsTest pruning_inputs_test;
          pruning_inputs_test.run_test_case();
          tests_count += pruning_inputs_test.get_tests_count();
          tests_passed_count += pruning_inputs_test.get_tests_passed_count();
          tests_failed_count += pruning_inputs_test.get_tests_failed_count();

          // genetic_algorithm

          GeneticAlgorithmTest genetic_algorithm_test;
          genetic_algorithm_test.run_test_case();
          tests_count += genetic_algorithm_test.get_tests_count();
          tests_passed_count += genetic_algorithm_test.get_tests_passed_count();
          tests_failed_count += genetic_algorithm_test.get_tests_failed_count();

          // T E S T I N G   A N A L Y S I S   T E S T S

          TestingAnalysisTest testing_analysis_test;
          testing_analysis_test.run_test_case();
          tests_count += testing_analysis_test.get_tests_count();
          tests_passed_count += testing_analysis_test.get_tests_passed_count();
          tests_failed_count += testing_analysis_test.get_tests_failed_count();
      }

      else
      {
         cout << "Unknown test: " << test << endl;

         return 1;
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
   catch(exception& e)
   {
      cerr << e.what() << endl;		 

      return 1;
   }
}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
