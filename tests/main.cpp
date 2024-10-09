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

int main()
{
   cout <<
   "Open Neural Networks Library. Test Suite Application.\n\n"

   "suite - run all tests\n\n"

   "Individual Tests:\n\n"

   "adaptive_moment_estimation | adam\n"
   "bounding_layer | bl\n"
   "conjugate_gradient | cg\n"
   "correlations | cr\n"
   "cross_entropy_error | cee\n"
   "cross_entropy_error_3d | cee3d\n"
   "convulational_layer | cl\n"
   "data_set | ds\n"
   "flatten_layer | fl\n"
   "genetic_algorithm | ga\n"
   "gradient_descent | gd\n"
   "growing_inputs | gi\n"
   "growing_neurons | gn\n"
   "image_data_set | samples_id\n"
   "inputs_selection | is\n"
   "learning_rate_algorithm | lra\n"
   "levenberg_marquardt_algorithm | lma\n"
   "long_short_term_memory_layer | lstm\n"
   "mean_squared_error | mse\n"
   "minkowski_error | me\n"
   "model_selection | ms\n"
   "neural_network | nn\n"
   "neurons_selection | ns\n"
   "normalized_squared_error | nse\n"
   "perceptron_layer | pl\n"
   "pooling_layer | pll\n"
   "probabilistic_layer | pbl\n"
   "probabilistic_layer_3d | pbl3d\n"
   "quasi_newton_method | qnm\n"
   "recurrent_layer | rl\n"
   "response_optimization | ro\n"
   "scaling_layer | sl\n"
   "scaling | sc\n"
   "statistics | st\n"
   "stochastic_gradient_descent | sgd\n"
   "sum_squared_error | sse\n"
   "tensors | t\n"
   "testing_analysis | ta\n"
   "time_series_data_set | tsds\n"
   "training_strategy | ts\n"
   "transformer | tf\n"
   "unscaling_layer | ul\n"
   "weighted_squared_error | wse\n"
   "\n" << endl;


   string test;

   cout << "Test: ";

   cin >> test;

   // Redirect standard output to file

   //ofstream out("../data/out.txt");
   //cout.rdbuf(out.rdbuf());

   try
   {
      srand(unsigned(time(nullptr)));

      string message;

      Index tests_count = 0;
      Index tests_passed_count = 0;

      Index tests_failed_count = 0;

      transform(test.begin(), test.end(), test.begin(),
          [](unsigned char c){ return tolower(c); });

      if(test == "adaptive_moment_estimation" || test == "adam")
      {
         AdaptiveMomentEstimationTest adaptive_moment_estimation_test;
         adaptive_moment_estimation_test.run_test_case();
         tests_count += adaptive_moment_estimation_test.get_tests_count();
         tests_passed_count += adaptive_moment_estimation_test.get_tests_passed_count();
         tests_failed_count += adaptive_moment_estimation_test.get_tests_failed_count();
      }

      else if(test == "correlations" || test == "cr")
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

      else if(test == "perceptron_layer" || test == "pl")
      {
         PerceptronLayerTest perceptron_layer_test;
         perceptron_layer_test.run_test_case();
         tests_count += perceptron_layer_test.get_tests_count();
         tests_passed_count += perceptron_layer_test.get_tests_passed_count();
         tests_failed_count += perceptron_layer_test.get_tests_failed_count();
      }

      else if(test == "scaling" || test == "sc")
      {
         ScalingTest scaling_test;
         scaling_test.run_test_case();
         tests_count += scaling_test.get_tests_count();
         tests_passed_count += scaling_test.get_tests_passed_count();
         tests_failed_count += scaling_test.get_tests_failed_count();
      }

      else if(test == "long_short_term_memory_layer" || test == "lstm")
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

      else if(test == "scaling_layer" || test == "sl")
      {
         ScalingLayer2DTest scaling_layer_test;
         scaling_layer_test.run_test_case();
         tests_count += scaling_layer_test.get_tests_count();
         tests_passed_count += scaling_layer_test.get_tests_passed_count();
         tests_failed_count += scaling_layer_test.get_tests_failed_count();
      }

      else if(test == "unscaling_layer" || test == "ul")
      {
         UnscalingLayerTest unscaling_layer_test;
         unscaling_layer_test.run_test_case();
         tests_count += unscaling_layer_test.get_tests_count();
         tests_passed_count += unscaling_layer_test.get_tests_passed_count();
         tests_failed_count += unscaling_layer_test.get_tests_failed_count();
      }

      else if(test == "bounding_layer" || test == "bl")
      {
         BoundingLayerTest bounding_layer_test;
         bounding_layer_test.run_test_case();
         tests_count += bounding_layer_test.get_tests_count();
         tests_passed_count += bounding_layer_test.get_tests_passed_count();
         tests_failed_count += bounding_layer_test.get_tests_failed_count();
      }

      else if(test == "probabilistic_layer" || test == "pbl")
      {
         ProbabilisticLayerTest probabilistic_layer_test;
         probabilistic_layer_test.run_test_case();
         tests_count += probabilistic_layer_test.get_tests_count();
         tests_passed_count += probabilistic_layer_test.get_tests_passed_count();
         tests_failed_count += probabilistic_layer_test.get_tests_failed_count();
      }

      else if(test == "probabilistic_layer_3d" || test == "pbl3d")
      {
          ProbabilisticLayer3DTest probabilistic_layer_3d_test;
          probabilistic_layer_3d_test.run_test_case();
          tests_count += probabilistic_layer_3d_test.get_tests_count();
          tests_passed_count += probabilistic_layer_3d_test.get_tests_passed_count();
          tests_failed_count += probabilistic_layer_3d_test.get_tests_failed_count();
          }

      else if(test == "convolutional_layer" || test == "cl")
      {
         ConvolutionalLayerTest layer_test;
         layer_test.run_test_case();
         tests_count += layer_test.get_tests_count();
         tests_passed_count += layer_test.get_tests_passed_count();
         tests_failed_count += layer_test.get_tests_failed_count();
      }

      else if(test == "pooling_layer" || test == "pll")
      {
         PoolingLayerTest layer_test;
         layer_test.run_test_case();
         tests_count += layer_test.get_tests_count();
         tests_passed_count += layer_test.get_tests_passed_count();
         tests_failed_count += layer_test.get_tests_failed_count();
      }

      else if(test == "flatten_layer" || test == "fl")
      {
         FlattenLayerTest layer_test;
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

      else if(test == "sum_squared_error" || test == "sse")
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

      else if(test == "cross_entropy_error_3d" || test == "cee3d")
      {
          CrossEntropyError3DTest cross_entropy_error_3d_test;
          cross_entropy_error_3d_test.run_test_case();
          tests_count += cross_entropy_error_3d_test.get_tests_count();
          tests_passed_count += cross_entropy_error_3d_test.get_tests_passed_count();
          tests_failed_count += cross_entropy_error_3d_test.get_tests_failed_count();
      }

      else if(test == "statistics" || test == "st")
      {
        StatisticsTest statistics_test;
        statistics_test.run_test_case();
        tests_count += statistics_test.get_tests_count();
        tests_passed_count += statistics_test.get_tests_passed_count();
        tests_failed_count += statistics_test.get_tests_failed_count();
      }

      else if(test == "learning_rate_algorithm" || test == "lra")
      {
        LearningRateAlgorithmTest learning_rate_algorithm_test;
        learning_rate_algorithm_test.run_test_case();
        tests_count += learning_rate_algorithm_test.get_tests_count();
        tests_passed_count += learning_rate_algorithm_test.get_tests_passed_count();
        tests_failed_count += learning_rate_algorithm_test.get_tests_failed_count();
      }
      else if(test == "gradient_descent" || test == "gd")
      {
        GradientDescentTest gradient_descent_test;
        gradient_descent_test.run_test_case();
        tests_count += gradient_descent_test.get_tests_count();
        tests_passed_count += gradient_descent_test.get_tests_passed_count();
        tests_failed_count += gradient_descent_test.get_tests_failed_count();
      }
      else if(test == "conjugate_gradient" || test == "cg")
      {
        ConjugateGradientTest conjugate_gradient_test;
        conjugate_gradient_test.run_test_case();
        tests_count += conjugate_gradient_test.get_tests_count();
        tests_passed_count += conjugate_gradient_test.get_tests_passed_count();
        tests_failed_count += conjugate_gradient_test.get_tests_failed_count();
      }
      else if(test == "quasi_newton_method" || test == "qnm")
      {
        QuasiNewtonMethodTest quasi_Newton_method_test;
        quasi_Newton_method_test.run_test_case();
        tests_count += quasi_Newton_method_test.get_tests_count();
        tests_passed_count += quasi_Newton_method_test.get_tests_passed_count();
        tests_failed_count += quasi_Newton_method_test.get_tests_failed_count();
      }
      else if(test == "levenberg_marquardt_algorithm" || test == "lma")
      {
        LevenbergMarquardtAlgorithmTest Levenberg_Marquardt_algorithm_test;
        Levenberg_Marquardt_algorithm_test.run_test_case();
        tests_count += Levenberg_Marquardt_algorithm_test.get_tests_count();
        tests_passed_count += Levenberg_Marquardt_algorithm_test.get_tests_passed_count();
        tests_failed_count += Levenberg_Marquardt_algorithm_test.get_tests_failed_count();
      }
      else if(test == "stochastic_gradient_descent" || test == "sgd")
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

      else if(test == "growing_neurons" || test == "gn")
      {
        GrowingNeuronsTest growing_neurons_test;
        growing_neurons_test.run_test_case();
        tests_count += growing_neurons_test.get_tests_count();
        tests_passed_count += growing_neurons_test.get_tests_passed_count();
        tests_failed_count += growing_neurons_test.get_tests_failed_count();
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

      else if(test == "genetic_algorithm" || test == "ga")
      {
        GeneticAlgorithmTest genetic_algorithm_test;
        genetic_algorithm_test.run_test_case();
        tests_count += genetic_algorithm_test.get_tests_count();
        tests_passed_count += genetic_algorithm_test.get_tests_passed_count();
        tests_failed_count += genetic_algorithm_test.get_tests_failed_count();
      }

      else if(test == "tensors" || test == "t")
      {
        TensorsTest tensors_test;
        tensors_test.run_test_case();
        tests_count += tensors_test.get_tests_count();
        tests_passed_count += tensors_test.get_tests_passed_count();
        tests_failed_count += tensors_test.get_tests_failed_count();
      }

      else if(test == "testing_analysis" || test == "ta")
      {
        TestingAnalysisTest testing_analysis_test;
        testing_analysis_test.run_test_case();
        tests_count += testing_analysis_test.get_tests_count();
        tests_passed_count += testing_analysis_test.get_tests_passed_count();
        tests_failed_count += testing_analysis_test.get_tests_failed_count();
      }

      else if(test == "transformer" || test == "tf")
      {
        TransformerTest transformer_test;
        transformer_test.run_test_case();
        tests_count += transformer_test.get_tests_count();
        tests_passed_count += transformer_test.get_tests_passed_count();
        tests_failed_count += transformer_test.get_tests_failed_count();
      }

      else if(test == "response_optimization" || test == "ro")
      {
        ResponseOptimizationTest response_optimization_test;
        response_optimization_test.run_test_case();
        tests_count += response_optimization_test.get_tests_count();
        tests_passed_count += response_optimization_test.get_tests_passed_count();
        tests_failed_count += response_optimization_test.get_tests_failed_count();
      }
      else if(test == "time_series_data_set" || test == "tsds")
      {
        TimeSeriesDataSetTest time_series_data_set_test;
        time_series_data_set_test.run_test_case();
        tests_count += time_series_data_set_test.get_tests_count();
        tests_passed_count += time_series_data_set_test.get_tests_passed_count();
        tests_failed_count += time_series_data_set_test.get_tests_failed_count();
      }
      else if(test == "image_data_set" || test == "samples_id")
      {
          ImageDataSetTest image_data_set_test;
          image_data_set_test.run_test_case();
          tests_count += image_data_set_test.get_tests_count();
          tests_passed_count += image_data_set_test.get_tests_passed_count();
          tests_failed_count += image_data_set_test.get_tests_failed_count();
      }

      else if(test == "suite")
      {
          // tensors

          TensorsTest tensor_utilites_test;
          tensor_utilites_test.run_test_case();
          tests_count += tensor_utilites_test.get_tests_count();
          tests_passed_count += tensor_utilites_test.get_tests_passed_count();
          tests_failed_count += tensor_utilites_test.get_tests_failed_count();

          // D A T A   S E T   T E S T S

          // correlation analysis

          CorrelationsTest correlations_test;
          correlations_test.run_test_case();
          tests_count += correlations_test.get_tests_count();
          tests_passed_count += correlations_test.get_tests_passed_count();
          tests_failed_count += correlations_test.get_tests_failed_count();


          // statistics

          StatisticsTest statistics_test;
          statistics_test.run_test_case();
          tests_count += statistics_test.get_tests_count();
          tests_passed_count += statistics_test.get_tests_passed_count();
          tests_failed_count += statistics_test.get_tests_failed_count();

          // scaling

          ScalingTest scaling_test;
          scaling_test.run_test_case();
          tests_count += scaling_test.get_tests_count();
          tests_passed_count += scaling_test.get_tests_passed_count();
          tests_failed_count += scaling_test.get_tests_failed_count();

          // data set

          DataSetTest data_set_test;
          data_set_test.run_test_case();
          tests_count += data_set_test.get_tests_count();
          tests_passed_count += data_set_test.get_tests_passed_count();
          tests_failed_count += data_set_test.get_tests_failed_count();

          // time series data set

          TimeSeriesDataSetTest time_series_data_set_test;
          time_series_data_set_test.run_test_case();
          tests_count += time_series_data_set_test.get_tests_count();
          tests_passed_count += time_series_data_set_test.get_tests_passed_count();
          tests_failed_count += time_series_data_set_test.get_tests_failed_count();

          // image data set

          ImageDataSetTest image_data_set_test;
          image_data_set_test.run_test_case();
          tests_count += image_data_set_test.get_tests_count();
          tests_passed_count += image_data_set_test.get_tests_passed_count();
          tests_failed_count += image_data_set_test.get_tests_failed_count();


          // N E U R A L   N E T W O R K   T E S T S

          // perceptron layer

          PerceptronLayerTest perceptron_layer_test;
          perceptron_layer_test.run_test_case();
          tests_count += perceptron_layer_test.get_tests_count();
          tests_passed_count += perceptron_layer_test.get_tests_passed_count();
          tests_failed_count += perceptron_layer_test.get_tests_failed_count();

          // scaling layer

          ScalingLayer2DTest scaling_layer_test;
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

          // lstm layer

          LongShortTermMemoryLayerTest lstm_layer_test;
          lstm_layer_test.run_test_case();
          tests_count += lstm_layer_test.get_tests_count();
          tests_passed_count += lstm_layer_test.get_tests_passed_count();
          tests_failed_count += lstm_layer_test.get_tests_failed_count();

          // recurrent layer

          RecurrentLayerTest recurrent_layer_test;
          recurrent_layer_test.run_test_case();
          tests_count += recurrent_layer_test.get_tests_count();
          tests_passed_count += recurrent_layer_test.get_tests_passed_count();
          tests_failed_count += recurrent_layer_test.get_tests_failed_count();

          // convolutional layer

//          ConvolutionalLayerTest convolutional_layer_test;
//          convolutional_layer_test.run_test_case();
//          tests_count += convolutional_layer_test.get_tests_count();
//          tests_passed_count += convolutional_layer_test.get_tests_passed_count();
//          tests_failed_count += convolutional_layer_test.get_tests_failed_count();

          // recurrent layer

          PoolingLayerTest pooling_layer_test;
          pooling_layer_test.run_test_case();
          tests_count += pooling_layer_test.get_tests_count();
          tests_passed_count += pooling_layer_test.get_tests_passed_count();
          tests_failed_count += pooling_layer_test.get_tests_failed_count();

          // Flatten layer

          FlattenLayerTest flatten_layer_test;
          flatten_layer_test.run_test_case();
          tests_count += flatten_layer_test.get_tests_count();
          tests_passed_count += flatten_layer_test.get_tests_passed_count();
          tests_failed_count += flatten_layer_test.get_tests_failed_count();

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

          // weighted squared error

          WeightedSquaredErrorTest weighted_squared_error_test;
          weighted_squared_error_test.run_test_case();
          tests_count += weighted_squared_error_test.get_tests_count();
          tests_passed_count += weighted_squared_error_test.get_tests_passed_count();
          tests_failed_count += weighted_squared_error_test.get_tests_failed_count();

          // minkowski error

          MinkowskiErrorTest Minkowski_error_test;
          Minkowski_error_test.run_test_case();
          tests_count += Minkowski_error_test.get_tests_count();
          tests_passed_count += Minkowski_error_test.get_tests_passed_count();
          tests_failed_count += Minkowski_error_test.get_tests_failed_count();

          // cross-entropy error

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

          // adaptive moment estimation

          AdaptiveMomentEstimationTest adaptive_moment_estimation_test;
          adaptive_moment_estimation_test.run_test_case();
          tests_count += adaptive_moment_estimation_test.get_tests_count();
          tests_passed_count += adaptive_moment_estimation_test.get_tests_passed_count();
          tests_failed_count += adaptive_moment_estimation_test.get_tests_failed_count();

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

          // stochastic gradient descent

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

          // neurons selection algorithm

          NeuronsSelectionTest neurons_selection_algorithm_test;
          neurons_selection_algorithm_test.run_test_case();
          tests_count += neurons_selection_algorithm_test.get_tests_count();
          tests_passed_count += neurons_selection_algorithm_test.get_tests_passed_count();
          tests_failed_count += neurons_selection_algorithm_test.get_tests_failed_count();

          // growing neurons

          GrowingNeuronsTest growing_neurons_test;
          growing_neurons_test.run_test_case();
          tests_count += growing_neurons_test.get_tests_count();
          tests_passed_count += growing_neurons_test.get_tests_passed_count();
          tests_failed_count += growing_neurons_test.get_tests_failed_count();

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

          // genetic_algorithm

//          GeneticAlgorithmTest genetic_algorithm_test;
//          genetic_algorithm_test.run_test_case();
//          tests_count += genetic_algorithm_test.get_tests_count();
//          tests_passed_count += genetic_algorithm_test.get_tests_passed_count();
//          tests_failed_count += genetic_algorithm_test.get_tests_failed_count();

          // T E S T I N G   A N A L Y S I S   T E S T S

          TestingAnalysisTest testing_analysis_test;
          testing_analysis_test.run_test_case();
          tests_count += testing_analysis_test.get_tests_count();
          tests_passed_count += testing_analysis_test.get_tests_passed_count();
          tests_failed_count += testing_analysis_test.get_tests_failed_count();

          // R E S P O N S E   O P T I M I Z A T I O N   T E S T S

          ResponseOptimizationTest response_optimization_test;
          response_optimization_test.run_test_case();
          tests_count += response_optimization_test.get_tests_count();
          tests_passed_count += response_optimization_test.get_tests_passed_count();
          tests_failed_count += response_optimization_test.get_tests_failed_count();
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
   catch(const exception& e)
   {
      cerr << e.what() << endl;

      return 1;
   }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
