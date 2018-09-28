/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S U I T E   T E S T S   A P P L I C A T I O N                                                              */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/
  
// System includes

#include <iostream>
#include <time.h>

// OpenNN includes

// OpenNN tests includes

#include "opennn_tests.h"
#include "unit_testing.h"

using namespace OpenNN;

int main()
{

#ifdef __OPENNN_MPI__

    MPI_Init(NULL, NULL);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank == 0)
    {
        cout << "Tests do not work with MPI\n";
    }

    MPI_Finalize();

    return(1);

#endif


   cout <<
   "Open Neural Networks Library. Test Suite Application.\n"
   "Write test:\n"
   "variables\n"
   "instances\n"
   "missing_values\n"
   "correlation_analysis\n"
   "data_set\n"
   "plug_in\n"
   "ordinary_differential_equations\n"
   "mathematical_model\n"
   "unscaling_layer\n"
   "scaling_layer\n"
   "inputs_trending_layer\n"
   "outputs_trending_layer\n"
   "probabilistic_layer\n"
   "perceptron_layer\n"
   "perceptron\n"
   "neural_network\n"
   "multilayer_perceptron\n"
   "inputs\n"
   "outputs\n"
   "independent_parameters\n"
   "conditions_layer\n"
   "bounding_layer\n"
   "sum_squared_error\n"
   "error_term\n"
   "loss_index\n"
   "outputs_integrals\n"
   "normalized_squared_error\n"
   "weighted_squared_error\n"
   "neural_parameters_norm\n"
   "minkowski_error\n"
   "mean_squared_error\n"
   "cross_entropy_error\n"
   "training_strategy\n"
   "training_rate_algorithm\n"
   "training_algorithm\n"
   "random_search\n"
   "quasi_newton_method\n"
   "newton_method\n"
   "levenberg_marquardt_algorithm\n"
   "gradient_descent\n"
   "evolutionary_algorithm\n"
   "conjugate_gradient\n"
   "testing_analysis\n"
   "vector\n"
   "numerical_integration\n"
   "numerical_differentiation\n"
   "matrix\n"
   "sparse_matrix\n"
   "model_selection\n"
   "order_selection_algorithm\n"
   "incremental_order\n"
   "golden_section_order\n"
   "simulated_annealing_order\n"
   "inputs_selection_algorithm\n"
   "growing_inputs\n"
   "pruning_inputs\n"
   "genetic_algorithm\n"
   "suite" << endl;

   string test;

   cout << "Test: ";

   cin >> test;

   // Redirect standard output to file

   //ofstream out("../data/out.txt");
   //cout.rdbuf(out.rdbuf());

   try
   {
      srand((unsigned)time(NULL));

      string message;

      size_t tests_count = 0;
      size_t tests_passed_count = 0;
      size_t tests_failed_count = 0;

      //
      // U T I L I T I E S   T E S T S
      // 

      // Vector test

      if(test == "vector")
      {
         VectorTest vector_test;
         vector_test.run_test_case();
         message += vector_test.get_message();
         tests_count += vector_test.get_tests_count();
         tests_passed_count += vector_test.get_tests_passed_count();
         tests_failed_count += vector_test.get_tests_failed_count();
      }
      else if(test == "matrix")
      {
         MatrixTest matrix_test;
         matrix_test.run_test_case();
         message += matrix_test.get_message();
         tests_count += matrix_test.get_tests_count();
         tests_passed_count += matrix_test.get_tests_passed_count();
         tests_failed_count += matrix_test.get_tests_failed_count();
      }
      else if(test == "sparse_matrix")
      {
         SparseMatrixTest sparse_matrix_test;
         sparse_matrix_test.run_test_case();
         message += sparse_matrix_test.get_message();
         tests_count += sparse_matrix_test.get_tests_count();
         tests_passed_count += sparse_matrix_test.get_tests_passed_count();
         tests_failed_count += sparse_matrix_test.get_tests_failed_count();
      }
      else if(test == "numerical_differentiation")
      {
         NumericalDifferentiationTest test_numerical_differentiation;
         test_numerical_differentiation.run_test_case();
         message += test_numerical_differentiation.get_message();
         tests_count += test_numerical_differentiation.get_tests_count();
         tests_passed_count += test_numerical_differentiation.get_tests_passed_count();
         tests_failed_count += test_numerical_differentiation.get_tests_failed_count();
      }
      else if(test == "numerical_integration")
      {
         NumericalIntegrationTest test_numerical_integration;
         test_numerical_integration.run_test_case();
         message += test_numerical_integration.get_message();
         tests_count += test_numerical_integration.get_tests_count();
         tests_passed_count += test_numerical_integration.get_tests_passed_count();
         tests_failed_count += test_numerical_integration.get_tests_failed_count();
      }

      //
      // D A T A   S E T   T E S T S
      // 

      else if(test == "variables")
      {
         VariablesTest variables_test;
         variables_test.run_test_case();
         message += variables_test.get_message();
         tests_count += variables_test.get_tests_count();
         tests_passed_count += variables_test.get_tests_passed_count();
         tests_failed_count += variables_test.get_tests_failed_count();
      }
      else if(test == "instances")
      {
         InstancesTest instances_test;
         instances_test.run_test_case();
         message += instances_test.get_message();
         tests_count += instances_test.get_tests_count();
         tests_passed_count += instances_test.get_tests_passed_count();
         tests_failed_count += instances_test.get_tests_failed_count();
      }
      else if(test == "missing_values")
      {
         MissingValuesTest missing_values_test;
         missing_values_test.run_test_case();
         message += missing_values_test.get_message();
         tests_count += missing_values_test.get_tests_count();
         tests_passed_count += missing_values_test.get_tests_passed_count();
         tests_failed_count += missing_values_test.get_tests_failed_count();
      }
      else if(test == "correlation_analysis")
      {
         CorrelationAnalysisTest correlation_analysis_test;
         correlation_analysis_test.run_test_case();
         message += correlation_analysis_test.get_message();
         tests_count += correlation_analysis_test.get_tests_count();
         tests_passed_count += correlation_analysis_test.get_tests_passed_count();
         tests_failed_count += correlation_analysis_test.get_tests_failed_count();
      }
      else if(test == "data_set")
      {
         DataSetTest data_set_test;
         data_set_test.run_test_case();
         message += data_set_test.get_message();
         tests_count += data_set_test.get_tests_count();
         tests_passed_count += data_set_test.get_tests_passed_count();
         tests_failed_count += data_set_test.get_tests_failed_count();
      }

      //
      // M A T H E M A T I C A L   M O D E L   T E S T S
      // 

      else if(test == "mathematical_model")
      {
         MathematicalModelTest mathematical_model_test;
         mathematical_model_test.run_test_case();
         message += mathematical_model_test.get_message();
         tests_count += mathematical_model_test.get_tests_count();
         tests_passed_count += mathematical_model_test.get_tests_passed_count();
         tests_failed_count += mathematical_model_test.get_tests_failed_count();
      }
      else if(test == "ordinary_differential_equations")
      {
         OrdinaryDifferentialEquationsTest ordinary_differential_equations_test;
         ordinary_differential_equations_test.run_test_case();
         message += ordinary_differential_equations_test.get_message();
         tests_count += ordinary_differential_equations_test.get_tests_count();
         tests_passed_count += ordinary_differential_equations_test.get_tests_passed_count();
         tests_failed_count += ordinary_differential_equations_test.get_tests_failed_count();
      }
      else if(test == "plug_in")
      {
         PlugInTest plug_in_test;
         plug_in_test.run_test_case();
         message += plug_in_test.get_message();
         tests_count += plug_in_test.get_tests_count();
         tests_passed_count += plug_in_test.get_tests_passed_count();
         tests_failed_count += plug_in_test.get_tests_failed_count();
      }

      //
      // N E U R A L   N E T W O R K   T E S T S
      // 

      else if(test == "perceptron")
      {
         PerceptronTest perceptron_test;
         perceptron_test.run_test_case();
         message += perceptron_test.get_message();
         tests_count += perceptron_test.get_tests_count();
         tests_passed_count += perceptron_test.get_tests_passed_count();
         tests_failed_count += perceptron_test.get_tests_failed_count();
      }
      else if(test == "perceptron_layer")
      {
         PerceptronLayerTest perceptron_layer_test;
         perceptron_layer_test.run_test_case();
         message += perceptron_layer_test.get_message();
         tests_count += perceptron_layer_test.get_tests_count();
         tests_passed_count += perceptron_layer_test.get_tests_passed_count();
         tests_failed_count += perceptron_layer_test.get_tests_failed_count();
      }
      else if(test == "multilayer_perceptron")
      {
         MultilayerPerceptronTest multilayer_perceptron_test;
         multilayer_perceptron_test.run_test_case();
         message += multilayer_perceptron_test.get_message();
         tests_count += multilayer_perceptron_test.get_tests_count();
         tests_passed_count += multilayer_perceptron_test.get_tests_passed_count();
         tests_failed_count += multilayer_perceptron_test.get_tests_failed_count();
      }
      else if(test == "scaling_layer")
      {
         ScalingLayerTest scaling_layer_test;
         scaling_layer_test.run_test_case();
         message += scaling_layer_test.get_message();
         tests_count += scaling_layer_test.get_tests_count();
         tests_passed_count += scaling_layer_test.get_tests_passed_count();
         tests_failed_count += scaling_layer_test.get_tests_failed_count();
      }
      else if(test == "unscaling_layer")
      {
         UnscalingLayerTest unscaling_layer_test;
         unscaling_layer_test.run_test_case();
         message += unscaling_layer_test.get_message();
         tests_count += unscaling_layer_test.get_tests_count();
         tests_passed_count += unscaling_layer_test.get_tests_passed_count();
         tests_failed_count += unscaling_layer_test.get_tests_failed_count();
      }
      else if(test == "inputs_trending_layer")
      {
         InputsTrendingLayerTest inputs_trending_layer_test;
         inputs_trending_layer_test.run_test_case();
         message += inputs_trending_layer_test.get_message();
         tests_count += inputs_trending_layer_test.get_tests_count();
         tests_passed_count += inputs_trending_layer_test.get_tests_passed_count();
         tests_failed_count += inputs_trending_layer_test.get_tests_failed_count();
      }
      else if(test == "outputs_trending_layer")
      {
         OutputsTrendingLayerTest outputs_trending_layer_test;
         outputs_trending_layer_test.run_test_case();
         message += outputs_trending_layer_test.get_message();
         tests_count += outputs_trending_layer_test.get_tests_count();
         tests_passed_count += outputs_trending_layer_test.get_tests_passed_count();
         tests_failed_count += outputs_trending_layer_test.get_tests_failed_count();
      }
      else if(test == "bounding_layer")
      {
         BoundingLayerTest bounding_layer_test;
         bounding_layer_test.run_test_case();
         message += bounding_layer_test.get_message();
         tests_count += bounding_layer_test.get_tests_count();
         tests_passed_count += bounding_layer_test.get_tests_passed_count();
         tests_failed_count += bounding_layer_test.get_tests_failed_count();
      }
      else if(test == "probabilistic_layer")
      {
         ProbabilisticLayerTest probabilistic_layer_test;
         probabilistic_layer_test.run_test_case();
         message += probabilistic_layer_test.get_message();
         tests_count += probabilistic_layer_test.get_tests_count();
         tests_passed_count += probabilistic_layer_test.get_tests_passed_count();
         tests_failed_count += probabilistic_layer_test.get_tests_failed_count();
      }
      else if(test == "conditions_layer")
      {
        ConditionsLayerTest conditions_layer_test;
        conditions_layer_test.run_test_case();
        message += conditions_layer_test.get_message();
        tests_count += conditions_layer_test.get_tests_count();
        tests_passed_count += conditions_layer_test.get_tests_passed_count();
        tests_failed_count += conditions_layer_test.get_tests_failed_count();
      }
      else if(test == "inputs")
      {
        InputsTest inputs_test;
        inputs_test.run_test_case();
        message += inputs_test.get_message();
        tests_count += inputs_test.get_tests_count();
        tests_passed_count += inputs_test.get_tests_passed_count();
        tests_failed_count += inputs_test.get_tests_failed_count();
      }
      else if(test == "outputs")
      {
        OutputsTest outputs_test;
        outputs_test.run_test_case();
        message += outputs_test.get_message();
        tests_count += outputs_test.get_tests_count();
        tests_passed_count += outputs_test.get_tests_passed_count();
        tests_failed_count += outputs_test.get_tests_failed_count();
      }
      else if(test == "independent_parameters")
      {
        IndependentParametersTest independent_parameters_test;
        independent_parameters_test.run_test_case();
        message += independent_parameters_test.get_message();
        tests_count += independent_parameters_test.get_tests_count();
        tests_passed_count += independent_parameters_test.get_tests_passed_count();
        tests_failed_count += independent_parameters_test.get_tests_failed_count();
      }
      else if(test == "neural_network")
      {
        NeuralNetworkTest neural_network_test;
        neural_network_test.run_test_case();
        message += neural_network_test.get_message();
        tests_count += neural_network_test.get_tests_count();
        tests_passed_count += neural_network_test.get_tests_passed_count();
        tests_failed_count += neural_network_test.get_tests_failed_count();
      }

      //
      // P E R F O R M A N C E   F U N C T I O N A L   T E S T S
      // 

      else if(test == "error_term")
      {
        ErrorTermTest error_term_test;
        error_term_test.run_test_case();
        message += error_term_test.get_message();
        tests_count += error_term_test.get_tests_count();
        tests_passed_count += error_term_test.get_tests_passed_count();
        tests_failed_count += error_term_test.get_tests_failed_count();
      }
      else if(test == "sum_squared_error")
      {
        SumSquaredErrorTest sum_squared_error_test;
        sum_squared_error_test.run_test_case();
        message += sum_squared_error_test.get_message();
        tests_count += sum_squared_error_test.get_tests_count();
        tests_passed_count += sum_squared_error_test.get_tests_passed_count();
        tests_failed_count += sum_squared_error_test.get_tests_failed_count();
      }
      else if(test == "mean_squared_error")
      {
        MeanSquaredErrorTest mean_squared_error_test;
        mean_squared_error_test.run_test_case();
        message += mean_squared_error_test.get_message();
        tests_count += mean_squared_error_test.get_tests_count();
        tests_passed_count += mean_squared_error_test.get_tests_passed_count();
        tests_failed_count += mean_squared_error_test.get_tests_failed_count();
      }
      else if(test == "root_mean_squared_error")
      {
        RootMeanSquaredErrorTest root_mean_squared_error_test;
        root_mean_squared_error_test.run_test_case();
        message += root_mean_squared_error_test.get_message();
        tests_count += root_mean_squared_error_test.get_tests_count();
        tests_passed_count += root_mean_squared_error_test.get_tests_passed_count();
        tests_failed_count += root_mean_squared_error_test.get_tests_failed_count();
      }
      else if(test == "normalized_squared_error")
      {
        NormalizedSquaredErrorTest normalized_squared_error_test;
        normalized_squared_error_test.run_test_case();
        message += normalized_squared_error_test.get_message();
        tests_count += normalized_squared_error_test.get_tests_count();
        tests_passed_count += normalized_squared_error_test.get_tests_passed_count();
        tests_failed_count += normalized_squared_error_test.get_tests_failed_count();
      }
      else if(test == "weighted_squared_error")
      {
        WeightedSquaredErrorTest weighted_squared_error_test;
        weighted_squared_error_test.run_test_case();
        message += weighted_squared_error_test.get_message();
        tests_count += weighted_squared_error_test.get_tests_count();
        tests_passed_count += weighted_squared_error_test.get_tests_passed_count();
        tests_failed_count += weighted_squared_error_test.get_tests_failed_count();
      }
      else if(test == "minkowski_error")
      {
        MinkowskiErrorTest Minkowski_error_test;
        Minkowski_error_test.run_test_case();
        message += Minkowski_error_test.get_message();
        tests_count += Minkowski_error_test.get_tests_count();
        tests_passed_count += Minkowski_error_test.get_tests_passed_count();
        tests_failed_count += Minkowski_error_test.get_tests_failed_count();
      }
      else if(test == "cross_entropy_error")
      {
        CrossEntropyErrorTest cross_entropy_error_test;
        cross_entropy_error_test.run_test_case();
        message += cross_entropy_error_test.get_message();
        tests_count += cross_entropy_error_test.get_tests_count();
        tests_passed_count += cross_entropy_error_test.get_tests_passed_count();
        tests_failed_count += cross_entropy_error_test.get_tests_failed_count();
      }
      else if(test == "neural_parameters_norm")
      {
        NeuralParametersNormTest test_neural_parameters_norm;
        test_neural_parameters_norm.run_test_case();
        message += test_neural_parameters_norm.get_message();
        tests_count += test_neural_parameters_norm.get_tests_count();
        tests_passed_count += test_neural_parameters_norm.get_tests_passed_count();
        tests_failed_count += test_neural_parameters_norm.get_tests_failed_count();
      }
      else if(test == "outputs_integrals")
      {
        OutputsIntegralsTest test_outputs_integrals;
        test_outputs_integrals.run_test_case();
        message += test_outputs_integrals.get_message();
        tests_count += test_outputs_integrals.get_tests_count();
        tests_passed_count += test_outputs_integrals.get_tests_passed_count();
        tests_failed_count += test_outputs_integrals.get_tests_failed_count();
      }
      else if(test == "loss_index")
      {
        LossIndexTest loss_index_test;
        loss_index_test.run_test_case();
        message += loss_index_test.get_message();
        tests_count += loss_index_test.get_tests_count();
        tests_passed_count += loss_index_test.get_tests_passed_count();
        tests_failed_count += loss_index_test.get_tests_failed_count();
      }

      //
      // T R A I N I N G   S T R A T E G Y   T E S T S
      // 

      else if(test == "training_rate_algorithm")
      {
        TrainingRateAlgorithmTest training_rate_algorithm_test;
        training_rate_algorithm_test.run_test_case();
        message += training_rate_algorithm_test.get_message();
        tests_count += training_rate_algorithm_test.get_tests_count();
        tests_passed_count += training_rate_algorithm_test.get_tests_passed_count();
        tests_failed_count += training_rate_algorithm_test.get_tests_failed_count();
      }
      else if(test == "training_algorithm")
      {
        TrainingAlgorithmTest training_algorithm_test;
        training_algorithm_test.run_test_case();
        message += training_algorithm_test.get_message();
        tests_count += training_algorithm_test.get_tests_count();
        tests_passed_count += training_algorithm_test.get_tests_passed_count();
        tests_failed_count += training_algorithm_test.get_tests_failed_count();
      }
      else if(test == "random_search")
      {
        RandomSearchTest random_search_test;
        random_search_test.run_test_case();
        message += random_search_test.get_message();
        tests_count += random_search_test.get_tests_count();
        tests_passed_count += random_search_test.get_tests_passed_count();
        tests_failed_count += random_search_test.get_tests_failed_count();
      }
      else if(test == "evolutionary_algorithm")
      {
        EvolutionaryAlgorithmTest evolutionary_algorithm_test;
        evolutionary_algorithm_test.run_test_case();
        message += evolutionary_algorithm_test.get_message();
        tests_count += evolutionary_algorithm_test.get_tests_count();
        tests_passed_count += evolutionary_algorithm_test.get_tests_passed_count();
        tests_failed_count += evolutionary_algorithm_test.get_tests_failed_count();
      }
      else if(test == "gradient_descent")
      {
        GradientDescentTest gradient_descent_test;
        gradient_descent_test.run_test_case();
        message += gradient_descent_test.get_message();
        tests_count += gradient_descent_test.get_tests_count();
        tests_passed_count += gradient_descent_test.get_tests_passed_count();
        tests_failed_count += gradient_descent_test.get_tests_failed_count();
      }
      else if(test == "newton_method")
      {
        NewtonMethodTest Newton_method_test;
        Newton_method_test.run_test_case();
        message += Newton_method_test.get_message();
        tests_count += Newton_method_test.get_tests_count();
        tests_passed_count += Newton_method_test.get_tests_passed_count();
        tests_failed_count += Newton_method_test.get_tests_failed_count();
      }
      else if(test == "conjugate_gradient")
      {
        ConjugateGradientTest conjugate_gradient_test;
        conjugate_gradient_test.run_test_case();
        message += conjugate_gradient_test.get_message();
        tests_count += conjugate_gradient_test.get_tests_count();
        tests_passed_count += conjugate_gradient_test.get_tests_passed_count();
        tests_failed_count += conjugate_gradient_test.get_tests_failed_count();
      }
      else if(test == "quasi_newton_method")
      {
        QuasiNewtonMethodTest quasi_Newton_method_test;
        quasi_Newton_method_test.run_test_case();
        message += quasi_Newton_method_test.get_message();
        tests_count += quasi_Newton_method_test.get_tests_count();
        tests_passed_count += quasi_Newton_method_test.get_tests_passed_count();
        tests_failed_count += quasi_Newton_method_test.get_tests_failed_count();
      }
      else if(test == "levenberg_marquardt_algorithm")
      {
        LevenbergMarquardtAlgorithmTest Levenberg_Marquardt_algorithm_test;
        Levenberg_Marquardt_algorithm_test.run_test_case();
        message += Levenberg_Marquardt_algorithm_test.get_message();
        tests_count += Levenberg_Marquardt_algorithm_test.get_tests_count();
        tests_passed_count += Levenberg_Marquardt_algorithm_test.get_tests_passed_count();
        tests_failed_count += Levenberg_Marquardt_algorithm_test.get_tests_failed_count();
      }
      else if(test == "training_strategy")
      {
        TrainingStrategyTest training_strategy_test;
        training_strategy_test.run_test_case();
        message += training_strategy_test.get_message();
        tests_count += training_strategy_test.get_tests_count();
        tests_passed_count += training_strategy_test.get_tests_passed_count();
        tests_failed_count += training_strategy_test.get_tests_failed_count();
      }

      //
      // M O D E L   S E L E C T I O N   T E S T S
      //

      else if(test == "model_selection")
      {
        ModelSelectionTest model_selection_test;
        model_selection_test.run_test_case();
        message += model_selection_test.get_message();
        tests_count += model_selection_test.get_tests_count();
        tests_passed_count += model_selection_test.get_tests_passed_count();
        tests_failed_count += model_selection_test.get_tests_failed_count();
      }

      else if(test == "order_selection_algorithm")
      {
        OrderSelectionAlgorithmTest order_selection_algorithm_test;
        order_selection_algorithm_test.run_test_case();
        message += order_selection_algorithm_test.get_message();
        tests_count += order_selection_algorithm_test.get_tests_count();
        tests_passed_count += order_selection_algorithm_test.get_tests_passed_count();
        tests_failed_count += order_selection_algorithm_test.get_tests_failed_count();
      }

      else if(test == "incremental_order")
      {
        IncrementalOrderTest incremental_order_test;
        incremental_order_test.run_test_case();
        message += incremental_order_test.get_message();
        tests_count += incremental_order_test.get_tests_count();
        tests_passed_count += incremental_order_test.get_tests_passed_count();
        tests_failed_count += incremental_order_test.get_tests_failed_count();
      }

      else if(test == "golden_section_order")
      {
        GoldenSectionOrderTest golden_section_order_test;
        golden_section_order_test.run_test_case();
        message += golden_section_order_test.get_message();
        tests_count += golden_section_order_test.get_tests_count();
        tests_passed_count += golden_section_order_test.get_tests_passed_count();
        tests_failed_count += golden_section_order_test.get_tests_failed_count();
      }

      else if(test == "simulated_annealing_order")
      {
        SimulatedAnnealingOrderTest simulated_annealing_order_test;
        simulated_annealing_order_test.run_test_case();
        message += simulated_annealing_order_test.get_message();
        tests_count += simulated_annealing_order_test.get_tests_count();
        tests_passed_count += simulated_annealing_order_test.get_tests_passed_count();
        tests_failed_count += simulated_annealing_order_test.get_tests_failed_count();
      }

      else if(test == "inputs_selection_algorithm")
      {
        InputsSelectionAlgorithmTest inputs_selection_algorithm_test;
        inputs_selection_algorithm_test.run_test_case();
        message += inputs_selection_algorithm_test.get_message();
        tests_count += inputs_selection_algorithm_test.get_tests_count();
        tests_passed_count += inputs_selection_algorithm_test.get_tests_passed_count();
        tests_failed_count += inputs_selection_algorithm_test.get_tests_failed_count();
      }

      else if(test == "growing_inputs")
      {
        GrowingInputsTest growing_inputs_test;
        growing_inputs_test.run_test_case();
        message += growing_inputs_test.get_message();
        tests_count += growing_inputs_test.get_tests_count();
        tests_passed_count += growing_inputs_test.get_tests_passed_count();
        tests_failed_count += growing_inputs_test.get_tests_failed_count();
      }

      else if(test == "pruning_inputs")
      {
        PruningInputsTest pruning_inputs_test;
        pruning_inputs_test.run_test_case();
        message += pruning_inputs_test.get_message();
        tests_count += pruning_inputs_test.get_tests_count();
        tests_passed_count += pruning_inputs_test.get_tests_passed_count();
        tests_failed_count += pruning_inputs_test.get_tests_failed_count();
      }

      else if(test == "genetic_algorithm")
      {
        GeneticAlgorithmTest genetic_algorithm_test;
        genetic_algorithm_test.run_test_case();
        message += genetic_algorithm_test.get_message();
        tests_count += genetic_algorithm_test.get_tests_count();
        tests_passed_count += genetic_algorithm_test.get_tests_passed_count();
        tests_failed_count += genetic_algorithm_test.get_tests_failed_count();
      }

      //
      // T E S T I N G   A N A L Y S I S   T E S T S
      // 

      else if(test == "testing_analysis")
      {
        TestingAnalysisTest testing_analysis_test;
        testing_analysis_test.run_test_case();
        message += testing_analysis_test.get_message();
        tests_count += testing_analysis_test.get_tests_count();
        tests_passed_count += testing_analysis_test.get_tests_passed_count();
        tests_failed_count += testing_analysis_test.get_tests_failed_count();
      }

      else if(test == "suite")
      {
          // vector

          VectorTest vector_test;
          vector_test.run_test_case();
          message += vector_test.get_message();
          tests_count += vector_test.get_tests_count();
          tests_passed_count += vector_test.get_tests_passed_count();
          tests_failed_count += vector_test.get_tests_failed_count();

          // matrix

          MatrixTest matrix_test;
          matrix_test.run_test_case();
          message += matrix_test.get_message();
          tests_count += matrix_test.get_tests_count();
          tests_passed_count += matrix_test.get_tests_passed_count();
          tests_failed_count += matrix_test.get_tests_failed_count();

          // sparse matrix

          SparseMatrixTest sparse_matrix_test;
          sparse_matrix_test.run_test_case();
          message += sparse_matrix_test.get_message();
          tests_count += sparse_matrix_test.get_tests_count();
          tests_passed_count += sparse_matrix_test.get_tests_passed_count();
          tests_failed_count += sparse_matrix_test.get_tests_failed_count();

          // numerical differentiation

          NumericalDifferentiationTest test_numerical_differentiation;
          test_numerical_differentiation.run_test_case();
          message += test_numerical_differentiation.get_message();
          tests_count += test_numerical_differentiation.get_tests_count();
          tests_passed_count += test_numerical_differentiation.get_tests_passed_count();
          tests_failed_count += test_numerical_differentiation.get_tests_failed_count();

          // numerical integration

          NumericalIntegrationTest test_numerical_integration;
          test_numerical_integration.run_test_case();
          message += test_numerical_integration.get_message();
          tests_count += test_numerical_integration.get_tests_count();
          tests_passed_count += test_numerical_integration.get_tests_passed_count();
          tests_failed_count += test_numerical_integration.get_tests_failed_count();

          // D A T A   S E T   T E S T S

          // variables

          VariablesTest variables_test;
          variables_test.run_test_case();
          message += variables_test.get_message();
          tests_count += variables_test.get_tests_count();
          tests_passed_count += variables_test.get_tests_passed_count();
          tests_failed_count += variables_test.get_tests_failed_count();

          // instances

          InstancesTest instances_test;
          instances_test.run_test_case();
          message += instances_test.get_message();
          tests_count += instances_test.get_tests_count();
          tests_passed_count += instances_test.get_tests_passed_count();
          tests_failed_count += instances_test.get_tests_failed_count();

          // correlation analysis

          CorrelationAnalysisTest correlation_analysis_test;
          correlation_analysis_test.run_test_case();
          message += correlation_analysis_test.get_message();
          tests_count += correlation_analysis_test.get_tests_count();
          tests_passed_count += correlation_analysis_test.get_tests_passed_count();
          tests_failed_count += correlation_analysis_test.get_tests_failed_count();

          // data set

          DataSetTest data_set_test;
          data_set_test.run_test_case();
          message += data_set_test.get_message();
          tests_count += data_set_test.get_tests_count();
          tests_passed_count += data_set_test.get_tests_passed_count();
          tests_failed_count += data_set_test.get_tests_failed_count();

          // M A T H E M A T I C A L   M O D E L   T E S T S

          // mathematical model

          MathematicalModelTest mathematical_model_test;
          mathematical_model_test.run_test_case();
          message += mathematical_model_test.get_message();
          tests_count += mathematical_model_test.get_tests_count();
          tests_passed_count += mathematical_model_test.get_tests_passed_count();
          tests_failed_count += mathematical_model_test.get_tests_failed_count();

          // ordinary differential equations

          OrdinaryDifferentialEquationsTest ordinary_differential_equations_test;
          ordinary_differential_equations_test.run_test_case();
          message += ordinary_differential_equations_test.get_message();
          tests_count += ordinary_differential_equations_test.get_tests_count();
          tests_passed_count += ordinary_differential_equations_test.get_tests_passed_count();
          tests_failed_count += ordinary_differential_equations_test.get_tests_failed_count();

          // plug in

          PlugInTest plug_in_test;
          plug_in_test.run_test_case();
          message += plug_in_test.get_message();
          tests_count += plug_in_test.get_tests_count();
          tests_passed_count += plug_in_test.get_tests_passed_count();
          tests_failed_count += plug_in_test.get_tests_failed_count();

          // N E U R A L   N E T W O R K   T E S T S

          // perceptron

          PerceptronTest perceptron_test;
          perceptron_test.run_test_case();
          message += perceptron_test.get_message();
          tests_count += perceptron_test.get_tests_count();
          tests_passed_count += perceptron_test.get_tests_passed_count();
          tests_failed_count += perceptron_test.get_tests_failed_count();

          // perceptron layer

          PerceptronLayerTest perceptron_layer_test;
          perceptron_layer_test.run_test_case();
          message += perceptron_layer_test.get_message();
          tests_count += perceptron_layer_test.get_tests_count();
          tests_passed_count += perceptron_layer_test.get_tests_passed_count();
          tests_failed_count += perceptron_layer_test.get_tests_failed_count();

          // multilayer perceptron

          MultilayerPerceptronTest multilayer_perceptron_test;
          multilayer_perceptron_test.run_test_case();
          message += multilayer_perceptron_test.get_message();
          tests_count += multilayer_perceptron_test.get_tests_count();
          tests_passed_count += multilayer_perceptron_test.get_tests_passed_count();
          tests_failed_count += multilayer_perceptron_test.get_tests_failed_count();

          // scaling layer

          ScalingLayerTest scaling_layer_test;
          scaling_layer_test.run_test_case();
          message += scaling_layer_test.get_message();
          tests_count += scaling_layer_test.get_tests_count();
          tests_passed_count += scaling_layer_test.get_tests_passed_count();
          tests_failed_count += scaling_layer_test.get_tests_failed_count();

          // unscaling layer

          UnscalingLayerTest unscaling_layer_test;
          unscaling_layer_test.run_test_case();
          message += unscaling_layer_test.get_message();
          tests_count += unscaling_layer_test.get_tests_count();
          tests_passed_count += unscaling_layer_test.get_tests_passed_count();
          tests_failed_count += unscaling_layer_test.get_tests_failed_count();

          // bounding layer

          BoundingLayerTest bounding_layer_test;
          bounding_layer_test.run_test_case();
          message += bounding_layer_test.get_message();
          tests_count += bounding_layer_test.get_tests_count();
          tests_passed_count += bounding_layer_test.get_tests_passed_count();
          tests_failed_count += bounding_layer_test.get_tests_failed_count();

          // probabilistic layer

          ProbabilisticLayerTest probabilistic_layer_test;
          probabilistic_layer_test.run_test_case();
          message += probabilistic_layer_test.get_message();
          tests_count += probabilistic_layer_test.get_tests_count();
          tests_passed_count += probabilistic_layer_test.get_tests_passed_count();
          tests_failed_count += probabilistic_layer_test.get_tests_failed_count();

          // conditions layer

          ConditionsLayerTest conditions_layer_test;
          conditions_layer_test.run_test_case();
          message += conditions_layer_test.get_message();
          tests_count += conditions_layer_test.get_tests_count();
          tests_passed_count += conditions_layer_test.get_tests_passed_count();
          tests_failed_count += conditions_layer_test.get_tests_failed_count();

          // inputs

          InputsTest inputs_test;
          inputs_test.run_test_case();
          message += inputs_test.get_message();
          tests_count += inputs_test.get_tests_count();
          tests_passed_count += inputs_test.get_tests_passed_count();
          tests_failed_count += inputs_test.get_tests_failed_count();

          // outputs

          OutputsTest outputs_test;
          outputs_test.run_test_case();
          message += outputs_test.get_message();
          tests_count += outputs_test.get_tests_count();
          tests_passed_count += outputs_test.get_tests_passed_count();
          tests_failed_count += outputs_test.get_tests_failed_count();

          // independent parameters

          IndependentParametersTest independent_parameters_test;
          independent_parameters_test.run_test_case();
          message += independent_parameters_test.get_message();
          tests_count += independent_parameters_test.get_tests_count();
          tests_passed_count += independent_parameters_test.get_tests_passed_count();
          tests_failed_count += independent_parameters_test.get_tests_failed_count();

          // neural network

          NeuralNetworkTest neural_network_test;
          neural_network_test.run_test_case();
          message += neural_network_test.get_message();
          tests_count += neural_network_test.get_tests_count();
          tests_passed_count += neural_network_test.get_tests_passed_count();
          tests_failed_count += neural_network_test.get_tests_failed_count();

          // P E R F O R M A N C E   F U N C T I O N A L   T E S T S

          // error term

          ErrorTermTest error_term_test;
          error_term_test.run_test_case();
          message += error_term_test.get_message();
          tests_count += error_term_test.get_tests_count();
          tests_passed_count += error_term_test.get_tests_passed_count();
          tests_failed_count += error_term_test.get_tests_failed_count();

          // sum squared error

          SumSquaredErrorTest sum_squared_error_test;
          sum_squared_error_test.run_test_case();
          message += sum_squared_error_test.get_message();
          tests_count += sum_squared_error_test.get_tests_count();
          tests_passed_count += sum_squared_error_test.get_tests_passed_count();
          tests_failed_count += sum_squared_error_test.get_tests_failed_count();

          // mean squared error

          MeanSquaredErrorTest mean_squared_error_test;
          mean_squared_error_test.run_test_case();
          message += mean_squared_error_test.get_message();
          tests_count += mean_squared_error_test.get_tests_count();
          tests_passed_count += mean_squared_error_test.get_tests_passed_count();
          tests_failed_count += mean_squared_error_test.get_tests_failed_count();

          // root mean squared error

          RootMeanSquaredErrorTest root_mean_squared_error_test;
          root_mean_squared_error_test.run_test_case();
          message += root_mean_squared_error_test.get_message();
          tests_count += root_mean_squared_error_test.get_tests_count();
          tests_passed_count += root_mean_squared_error_test.get_tests_passed_count();
          tests_failed_count += root_mean_squared_error_test.get_tests_failed_count();

          // normalized squared error

          NormalizedSquaredErrorTest normalized_squared_error_test;
          normalized_squared_error_test.run_test_case();
          message += normalized_squared_error_test.get_message();
          tests_count += normalized_squared_error_test.get_tests_count();
          tests_passed_count += normalized_squared_error_test.get_tests_passed_count();
          tests_failed_count += normalized_squared_error_test.get_tests_failed_count();

          // minkowski error

          MinkowskiErrorTest Minkowski_error_test;
          Minkowski_error_test.run_test_case();
          message += Minkowski_error_test.get_message();
          tests_count += Minkowski_error_test.get_tests_count();
          tests_passed_count += Minkowski_error_test.get_tests_passed_count();
          tests_failed_count += Minkowski_error_test.get_tests_failed_count();

          // cross entropy error

          CrossEntropyErrorTest cross_entropy_error_test;
          cross_entropy_error_test.run_test_case();
          message += cross_entropy_error_test.get_message();
          tests_count += cross_entropy_error_test.get_tests_count();
          tests_passed_count += cross_entropy_error_test.get_tests_passed_count();
          tests_failed_count += cross_entropy_error_test.get_tests_failed_count();

          // neural parameters norm

          NeuralParametersNormTest test_neural_parameters_norm;
          test_neural_parameters_norm.run_test_case();
          message += test_neural_parameters_norm.get_message();
          tests_count += test_neural_parameters_norm.get_tests_count();
          tests_passed_count += test_neural_parameters_norm.get_tests_passed_count();
          tests_failed_count += test_neural_parameters_norm.get_tests_failed_count();

          // outputs integrals

          OutputsIntegralsTest test_outputs_integrals;
          test_outputs_integrals.run_test_case();
          message += test_outputs_integrals.get_message();
          tests_count += test_outputs_integrals.get_tests_count();
          tests_passed_count += test_outputs_integrals.get_tests_passed_count();
          tests_failed_count += test_outputs_integrals.get_tests_failed_count();

          // loss functional

          LossIndexTest loss_index_test;
          loss_index_test.run_test_case();
          message += loss_index_test.get_message();
          tests_count += loss_index_test.get_tests_count();
          tests_passed_count += loss_index_test.get_tests_passed_count();
          tests_failed_count += loss_index_test.get_tests_failed_count();

          // T R A I N I N G   S T R A T E G Y   T E S T S

          // training rate algorithm

          TrainingRateAlgorithmTest training_rate_algorithm_test;
          training_rate_algorithm_test.run_test_case();
          message += training_rate_algorithm_test.get_message();
          tests_count += training_rate_algorithm_test.get_tests_count();
          tests_passed_count += training_rate_algorithm_test.get_tests_passed_count();
          tests_failed_count += training_rate_algorithm_test.get_tests_failed_count();

          // training algorithm

          TrainingAlgorithmTest training_algorithm_test;
          training_algorithm_test.run_test_case();
          message += training_algorithm_test.get_message();
          tests_count += training_algorithm_test.get_tests_count();
          tests_passed_count += training_algorithm_test.get_tests_passed_count();
          tests_failed_count += training_algorithm_test.get_tests_failed_count();

          // random search

          RandomSearchTest random_search_test;
          random_search_test.run_test_case();
          message += random_search_test.get_message();
          tests_count += random_search_test.get_tests_count();
          tests_passed_count += random_search_test.get_tests_passed_count();
          tests_failed_count += random_search_test.get_tests_failed_count();

          // evolutionary algorithm

          EvolutionaryAlgorithmTest evolutionary_algorithm_test;
          evolutionary_algorithm_test.run_test_case();
          message += evolutionary_algorithm_test.get_message();
          tests_count += evolutionary_algorithm_test.get_tests_count();
          tests_passed_count += evolutionary_algorithm_test.get_tests_passed_count();
          tests_failed_count += evolutionary_algorithm_test.get_tests_failed_count();

          // gradient descent

          GradientDescentTest gradient_descent_test;
          gradient_descent_test.run_test_case();
          message += gradient_descent_test.get_message();
          tests_count += gradient_descent_test.get_tests_count();
          tests_passed_count += gradient_descent_test.get_tests_passed_count();
          tests_failed_count += gradient_descent_test.get_tests_failed_count();

          // newton method

          NewtonMethodTest Newton_method_test;
          Newton_method_test.run_test_case();
          message += Newton_method_test.get_message();
          tests_count += Newton_method_test.get_tests_count();
          tests_passed_count += Newton_method_test.get_tests_passed_count();
          tests_failed_count += Newton_method_test.get_tests_failed_count();

          // conjugate gradient

          ConjugateGradientTest conjugate_gradient_test;
          conjugate_gradient_test.run_test_case();
          message += conjugate_gradient_test.get_message();
          tests_count += conjugate_gradient_test.get_tests_count();
          tests_passed_count += conjugate_gradient_test.get_tests_passed_count();
          tests_failed_count += conjugate_gradient_test.get_tests_failed_count();

          // quasi newton method

          QuasiNewtonMethodTest quasi_Newton_method_test;
          quasi_Newton_method_test.run_test_case();
          message += quasi_Newton_method_test.get_message();
          tests_count += quasi_Newton_method_test.get_tests_count();
          tests_passed_count += quasi_Newton_method_test.get_tests_passed_count();
          tests_failed_count += quasi_Newton_method_test.get_tests_failed_count();

          // levenberg marquardt algorithm

          LevenbergMarquardtAlgorithmTest Levenberg_Marquardt_algorithm_test;
          Levenberg_Marquardt_algorithm_test.run_test_case();
          message += Levenberg_Marquardt_algorithm_test.get_message();
          tests_count += Levenberg_Marquardt_algorithm_test.get_tests_count();
          tests_passed_count += Levenberg_Marquardt_algorithm_test.get_tests_passed_count();
          tests_failed_count += Levenberg_Marquardt_algorithm_test.get_tests_failed_count();

          // training_strategy

          TrainingStrategyTest training_strategy_test;
          training_strategy_test.run_test_case();
          message += training_strategy_test.get_message();
          tests_count += training_strategy_test.get_tests_count();
          tests_passed_count += training_strategy_test.get_tests_passed_count();
          tests_failed_count += training_strategy_test.get_tests_failed_count();

          // M O D E L   S E L E C T I O N   T E S T S

          // model selection

          ModelSelectionTest model_selection_test;
          model_selection_test.run_test_case();
          message += model_selection_test.get_message();
          tests_count += model_selection_test.get_tests_count();
          tests_passed_count += model_selection_test.get_tests_passed_count();
          tests_failed_count += model_selection_test.get_tests_failed_count();

          // order selection algorithm

          OrderSelectionAlgorithmTest order_selection_algorithm_test;
          order_selection_algorithm_test.run_test_case();
          message += order_selection_algorithm_test.get_message();
          tests_count += order_selection_algorithm_test.get_tests_count();
          tests_passed_count += order_selection_algorithm_test.get_tests_passed_count();
          tests_failed_count += order_selection_algorithm_test.get_tests_failed_count();


          // incremental order

          IncrementalOrderTest incremental_order_test;
          incremental_order_test.run_test_case();
          message += incremental_order_test.get_message();
          tests_count += incremental_order_test.get_tests_count();
          tests_passed_count += incremental_order_test.get_tests_passed_count();
          tests_failed_count += incremental_order_test.get_tests_failed_count();


          // golden section order

          GoldenSectionOrderTest golden_section_order_test;
          golden_section_order_test.run_test_case();
          message += golden_section_order_test.get_message();
          tests_count += golden_section_order_test.get_tests_count();
          tests_passed_count += golden_section_order_test.get_tests_passed_count();
          tests_failed_count += golden_section_order_test.get_tests_failed_count();


          // simulated annealing order

          SimulatedAnnealingOrderTest simulated_annealing_order_test;
          simulated_annealing_order_test.run_test_case();
          message += simulated_annealing_order_test.get_message();
          tests_count += simulated_annealing_order_test.get_tests_count();
          tests_passed_count += simulated_annealing_order_test.get_tests_passed_count();
          tests_failed_count += simulated_annealing_order_test.get_tests_failed_count();

          // input selection algorithm

          InputsSelectionAlgorithmTest inputs_selection_algorithm_test;
          inputs_selection_algorithm_test.run_test_case();
          message += inputs_selection_algorithm_test.get_message();
          tests_count += inputs_selection_algorithm_test.get_tests_count();
          tests_passed_count += inputs_selection_algorithm_test.get_tests_passed_count();
          tests_failed_count += inputs_selection_algorithm_test.get_tests_failed_count();

          // growing_inputs

          GrowingInputsTest growing_inputs_test;
          growing_inputs_test.run_test_case();
          message += growing_inputs_test.get_message();
          tests_count += growing_inputs_test.get_tests_count();
          tests_passed_count += growing_inputs_test.get_tests_passed_count();
          tests_failed_count += growing_inputs_test.get_tests_failed_count();

          // pruning_inputs

          PruningInputsTest pruning_inputs_test;
          pruning_inputs_test.run_test_case();
          message += pruning_inputs_test.get_message();
          tests_count += pruning_inputs_test.get_tests_count();
          tests_passed_count += pruning_inputs_test.get_tests_passed_count();
          tests_failed_count += pruning_inputs_test.get_tests_failed_count();

          // genetic_algorithm

          GeneticAlgorithmTest genetic_algorithm_test;
          genetic_algorithm_test.run_test_case();
          message += genetic_algorithm_test.get_message();
          tests_count += genetic_algorithm_test.get_tests_count();
          tests_passed_count += genetic_algorithm_test.get_tests_passed_count();
          tests_failed_count += genetic_algorithm_test.get_tests_failed_count();

          // T E S T I N G   A N A L Y S I S   T E S T S

          // testing analysis

          TestingAnalysisTest testing_analysis_test;
          testing_analysis_test.run_test_case();
          message += testing_analysis_test.get_message();
          tests_count += testing_analysis_test.get_tests_count();
          tests_passed_count += testing_analysis_test.get_tests_passed_count();
          tests_failed_count += testing_analysis_test.get_tests_failed_count();
      }

      else
      {
         cout << "Unknown test: " << test << endl;

         return(1);
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

      return(0);
   }
   catch(exception& e)
   {
      cout << e.what() << endl;		 

      return(1);
   }
}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Roberto Lopez
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
