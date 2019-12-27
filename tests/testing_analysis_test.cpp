//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T E S T I N G   A N A L Y S I S   T E S T   C L A S S                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "testing_analysis_test.h"


TestingAnalysisTest::TestingAnalysisTest() : UnitTesting() 
{
}


TestingAnalysisTest::~TestingAnalysisTest()
{
}


void TestingAnalysisTest::test_constructor()
{
   cout << "test_constructor\n";


   // Neural network constructor

   NeuralNetwork nn2;
   TestingAnalysis ta2(&nn2);

   assert_true(ta2.get_neural_network_pointer() != nullptr, LOG);

   // Data set constructor

   DataSet ds3;
   TestingAnalysis ta3(&ds3);

   assert_true(ta3.get_data_set_pointer() != nullptr, LOG);

   // Neural network and data set constructor

   NeuralNetwork nn4;
   DataSet ds4;

   TestingAnalysis ta4(&nn4, &ds4);

   assert_true(ta4.get_neural_network_pointer() != nullptr, LOG);
   assert_true(ta4.get_data_set_pointer() != nullptr, LOG);

}


void TestingAnalysisTest::test_destructor()
{
   cout << "test_destructor\n";
}


void TestingAnalysisTest::test_get_neural_network_pointer()
{
   cout << "test_get_neural_network_pointer\n";

   TestingAnalysis ta;

   NeuralNetwork neural_network;

   ta.set_neural_network_pointer(&neural_network);
   
   assert_true(ta.get_neural_network_pointer() != nullptr, LOG);
}


void TestingAnalysisTest::test_get_data_set_pointer()
{
   cout << "test_get_data_set_pointer\n";

   TestingAnalysis ta;

   DataSet data_set;

   ta.set_data_set_pointer(&data_set);
   
   assert_true(ta.get_data_set_pointer() != nullptr, LOG);
}


void TestingAnalysisTest::test_get_display()
{
   cout << "test_get_display\n";
}


void TestingAnalysisTest::test_set_neural_network_pointer()
{
   cout << "test_set_neural_network_pointer\n";
}


void TestingAnalysisTest::test_set_data_set_pointer()
{
   cout << "test_set_data_set_pointer\n";
}


void TestingAnalysisTest::test_set_display()
{
   cout << "test_set_display\n";
}


void TestingAnalysisTest::test_calculate_error_data()
{
    cout << "test_calculate_error_data\n";

    NeuralNetwork neural_network;

    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector<Matrix<double>> error_data;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.initialize_data(0.0);
    data_set.set_testing();

    error_data = ta.calculate_error_data();

    assert_true(error_data.size() == 1, LOG);
    assert_true(error_data[0].get_rows_number() == 1, LOG);
    assert_true(error_data[0].get_columns_number() == 3, LOG);
    assert_true(error_data[0] == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_percentage_error_data()
{
    cout << "test_calculate_percentage_error_data\n";

    NeuralNetwork neural_network;

    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector<Vector<double>> error_data;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.initialize_data(0.0);
    data_set.set_testing();

    error_data = ta.calculate_percentage_error_data();

    assert_true(error_data.size() == 1, LOG);
    assert_true(error_data[0].size() == 1, LOG);
    assert_true(error_data[0].get_first() == 0.0, LOG);
    assert_true(error_data[0] == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_forecasting_error_data()
{}


void TestingAnalysisTest::test_calculate_absolute_errors_statistics()
{
    cout << "test_calculate_absolute_error_statistics\n";

    NeuralNetwork neural_network;

    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector<Descriptives> error_data;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
//    nn.construct_unscaling_layer();
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.set_testing();
    data_set.initialize_data(0.0);


    error_data = ta.calculate_absolute_errors_statistics();


    assert_true(error_data.size() == 1, LOG);
    assert_true(error_data[0].minimum == 0.0, LOG);
    assert_true(error_data[0].maximum == 0.0, LOG);
    assert_true(error_data[0].mean == 0.0, LOG);
    assert_true(error_data[0].standard_deviation == 0.0, LOG);

}

void TestingAnalysisTest::test_calculate_percentage_errors_statistics()
{
    cout << "test_calculate_percentage_error_statistics\n";

    NeuralNetwork neural_network;

    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector<Descriptives> error_data;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
//    nn.construct_unscaling_layer();
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.set_testing();
    data_set.initialize_data(1.0);

    error_data = ta.calculate_percentage_errors_statistics();

    assert_true(error_data.size() == 1, LOG);
    assert_true(error_data[0].standard_deviation == 0.0, LOG);
}

void TestingAnalysisTest::test_calculate_error_data_statistics()
{
    cout << "test_calculate_error_data_statistics\n";

    NeuralNetwork neural_network;
    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector< Vector<Descriptives> > error_data_statistics;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
//    nn.construct_unscaling_layer();
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.set_testing();
    data_set.initialize_data(0.0);

    error_data_statistics = ta.calculate_error_data_statistics();

    assert_true(error_data_statistics.size() == 1, LOG);
    assert_true(error_data_statistics[0].size() == 3, LOG);
    assert_true(error_data_statistics[0][0].minimum == 0.0, LOG);
    assert_true(error_data_statistics[0][0].maximum == 0.0, LOG);
    assert_true(error_data_statistics[0][0].mean == 0.0, LOG);
    assert_true(error_data_statistics[0][0].standard_deviation == 0.0, LOG);
    assert_true(error_data_statistics[0][2].minimum == 0.0, LOG);
    assert_true(error_data_statistics[0][2].maximum == 0.0, LOG);
    assert_true(error_data_statistics[0][2].mean == 0.0, LOG);
    assert_true(error_data_statistics[0][2].standard_deviation == 0.0, LOG);

}


void TestingAnalysisTest::test_print_error_data_statistics()
{
    cout << "test_print_error_data_statistics\n";

    NeuralNetwork neural_network;
    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector< Vector<Descriptives> > error_data_statistics;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
//    nn.construct_unscaling_layer();
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.set_testing();
    data_set.initialize_data(0.0);

//    ta.print_error_data_statistics();


}

void TestingAnalysisTest::test_calculate_error_data_statistics_matrices()
{
    cout << "test_calculate_error_data_statistics_matrices\n";

    NeuralNetwork neural_network;
    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector< Matrix<double> > error_data_statistics;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
//    nn.construct_unscaling_layer();
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.set_testing();
    data_set.initialize_data(0.0);

    error_data_statistics = ta.calculate_error_data_statistics_matrices();

    assert_true(error_data_statistics.size() == 1, LOG);
    assert_true(error_data_statistics[0].get_rows_number() == 2, LOG);
    assert_true(error_data_statistics[0].get_columns_number() == 4, LOG);

}


void TestingAnalysisTest::test_calculate_error_data_histograms()
{
    cout << "test_calculate_error_data_histograms\n";

    NeuralNetwork neural_network;
    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector<Histogram> error_data_histograms;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
//    nn.construct_unscaling_layer();
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 3);
    data_set.set_testing();
    data_set.initialize_data(0.0);

    error_data_histograms = ta.calculate_error_data_histograms();

    assert_true(error_data_histograms.size() == 1, LOG);
    assert_true(error_data_histograms[0].get_bins_number() == 10, LOG);
}


void TestingAnalysisTest::test_calculate_maximal_errors()
{
    cout << "test_calculate_maximal_errors\n";

    NeuralNetwork neural_network;
    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector<Vector<size_t>> error_data_maximal;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
//    nn.construct_unscaling_layer();
    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.set_testing();
    data_set.initialize_data(0.0);

    error_data_maximal = ta.calculate_maximal_errors();

    assert_true(error_data_maximal.size() == 1, LOG);
    assert_true(error_data_maximal[0] == 0 , LOG);
}


void TestingAnalysisTest::test_linear_regression()
{
   cout << "test_linear_regression\n";

   NeuralNetwork neural_network;

   DataSet data_set;

   Matrix<double> data;

   TestingAnalysis ta(&neural_network, &data_set);

   Vector<RegressionResults> linear_regression;

    // Test

   neural_network.set(NeuralNetwork::Approximation, {1, 1});

   neural_network.initialize_parameters(0.0);

   data_set.set(1,1,1);

   data_set.set_testing();

   data_set.initialize_data(0.0);

   linear_regression = ta.linear_regression();

   assert_true(linear_regression.size() == 1, LOG);
}


void TestingAnalysisTest::test_print_linear_regression_correlation()
{
   cout << "test_print_linear_regression_correlation\n";

   NeuralNetwork neural_network(NeuralNetwork::Approximation, {1,1,1});

   neural_network.initialize_parameters(0.0);

   DataSet data_set(1,1,1);

   data_set.set_testing();

   data_set.initialize_data(0.0);

   TestingAnalysis ta(&neural_network, &data_set);
//   ta.print_linear_regression_correlations();
}


void TestingAnalysisTest::test_get_linear_regression_correlations_std()
{
    cout << "test_get_linear_regression_correlations_std\n";

    NeuralNetwork neural_network(NeuralNetwork::Approximation, {1,1,1});

    neural_network.initialize_parameters(0.0);

    DataSet data_set(1,1,1);

    data_set.set_testing();

    data_set.initialize_data(0.0);

    TestingAnalysis ta(&neural_network, &data_set);

    Vector<double> correlations = ta.get_linear_regression_correlations_std();

    assert_true(correlations.size() == 1, LOG);
    assert_true(correlations[0] == 1.0 , LOG);
}

void TestingAnalysisTest::test_save_linear_regression()
{
   cout << "test_save_linear_regression\n";

   string file_name = "../data/linear_regression.dat";

   NeuralNetwork neural_network;
   DataSet data_set;

   TestingAnalysis ta(&neural_network, &data_set);
//   ta.save_linear_regression(file_name);
}


void TestingAnalysisTest::test_perform_linear_regression()
{
    cout << "test_perform_linear_regression\n";

    NeuralNetwork neural_network;
    DataSet data_set;
    TestingAnalysis ta(&neural_network, &data_set);

    Vector< TestingAnalysis::LinearRegressionAnalysis > linear_regression_analysis;

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});

    neural_network.initialize_parameters(0.0);

    data_set.set(1, 1, 1);
    data_set.set_testing();
    data_set.initialize_data(0.0);

    linear_regression_analysis = ta.perform_linear_regression_analysis();

    assert_true(linear_regression_analysis.size() == 1 , LOG);
    assert_true(linear_regression_analysis[0].targets == Vector<double>{0} , LOG);
    assert_true(linear_regression_analysis[0].correlation == 1.0 , LOG);

}

void TestingAnalysisTest::test_print_linear_regression_analysis()
{
   cout << "test_print_linear_regression_analysis\n";
}


void TestingAnalysisTest::test_save_linear_regression_analysis()
{
   cout << "test_save_linear_regression_analysis\n";
}


void TestingAnalysisTest::test_calculate_confusion()
{
   cout << "test_calculate_confusion\n";

   NeuralNetwork neural_network;
   DataSet data_set;

   TestingAnalysis ta(&neural_network, &data_set);

  // Instances* i;

   Matrix<double> actual;
   Matrix<double> predicted;

   Matrix<size_t> confusion;

   // Test

   actual.set(4, 3);
   predicted.set(4, 3);

   actual(0,0) = 1; actual(0,1) = 0; actual(0,2) = 0;
   actual(1,0) = 0; actual(1,1) = 1; actual(1,2) = 0;
   actual(2,0) = 0; actual(2,1) = 1; actual(2,2) = 0;
   actual(3,0) = 0; actual(3,1) = 0; actual(3,2) = 1;

   predicted(0,0) = 1; predicted(0,1) = 0; predicted(0,2) = 0;
   predicted(1,0) = 0; predicted(1,1) = 1; predicted(1,2) = 0;
   predicted(2,0) = 0; predicted(2,1) = 1; predicted(2,2) = 0;
   predicted(3,0) = 0; predicted(3,1) = 0; predicted(3,2) = 1;

   confusion = ta.calculate_confusion_multiple_classification(actual.to_tensor(), predicted.to_tensor());

   assert_true(confusion.calculate_sum() == 4, LOG);
   assert_true(confusion.get_diagonal().calculate_sum() == 4, LOG);
}


void TestingAnalysisTest::test_calculate_binary_classification_test()
{
   cout << "test_calculate_binary_classification_test\n";

   NeuralNetwork neural_network(NeuralNetwork::Classification, {1,1,1});

   neural_network.initialize_parameters(0.0);

   DataSet data_set(1,1,1);

   data_set.set_testing();

   data_set.initialize_data(0.0);

   TestingAnalysis ta(&neural_network, &data_set);

   Vector<double> binary = ta.calculate_binary_classification_tests();

   assert_true(binary.size() == 15 , LOG);
}


void TestingAnalysisTest::test_calculate_Wilcoxon_parameter()
{
    cout << "test_calculate_Wilcoxon_parameter\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    double wilcoxon_parameter;

    // Test

    double x = 1.5;
    double y = 2.5;

    wilcoxon_parameter = ta.calculate_Wilcoxon_parameter(x, y);

    assert_true(abs(wilcoxon_parameter) <= numeric_limits<double>::min(), LOG);

    // Test

    wilcoxon_parameter = ta.calculate_Wilcoxon_parameter(y ,x);

    assert_true(abs(wilcoxon_parameter - 1) <= numeric_limits<double>::min(), LOG);

    // Test

    x = y;

    wilcoxon_parameter = ta.calculate_Wilcoxon_parameter(x, y);

    assert_true(abs(wilcoxon_parameter - 0.5) <= numeric_limits<double>::min(), LOG);
}


void TestingAnalysisTest::test_calculate_roc_curve()
{
    cout << "test_calculate_roc_curve\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<double> targets;
    Tensor<double> outputs;

    Matrix<double> roc_curve;

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 1.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 1.0;
    outputs[3] = 1.0;

    roc_curve = ta.calculate_roc_curve(targets, outputs);

    assert_true(roc_curve.get_columns_number() == 3, LOG);
    assert_true(roc_curve.get_rows_number() == 5, LOG);

    assert_true(roc_curve(0, 0) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(0, 1) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(1, 0) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(1, 1) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(2, 0) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(2, 1) - 1.0 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(3, 0) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(3, 1) - 1.0 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(4, 0) - 1.0 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(4, 1) - 1.0 <= numeric_limits<double>::min(), LOG);

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 1.0;
    targets[2] = 1.0;
    targets[3] = 0.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.12;
    outputs[1] = 0.78;
    outputs[2] = 0.84;
    outputs[3] = 0.99;

    roc_curve = ta.calculate_roc_curve(targets, outputs);

    assert_true(roc_curve.get_columns_number() == 3, LOG);
    assert_true(roc_curve.get_rows_number() == 5, LOG);

    assert_true(roc_curve(0, 0) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(0, 1) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(1, 0) <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(1, 1) - 0.5 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(2, 0) - 0.5 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(2, 1) - 0.5 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(3, 0) - 1.0 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(3, 1) - 0.5 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(4, 0) - 1.0 <= numeric_limits<double>::min(), LOG);
    assert_true(roc_curve(4, 1) - 1.0 <= numeric_limits<double>::min(), LOG);
}


void TestingAnalysisTest::test_calculate_area_under_curve()
{
    cout << "test_calculate_area_under_curve\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<double> targets;
    Tensor<double> outputs;

    double area_under_curve;

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 1.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 1.0;
    outputs[3] = 1.0;

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve - 1.0 <= numeric_limits<double>::min(), LOG);

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 1.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.0;
    outputs[1] = 1.0;
    outputs[2] = 0.0;
    outputs[3] = 1.0;

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve == 0.5, LOG);

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 1.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.78;
    outputs[1] = 0.84;
    outputs[2] = 0.12;
    outputs[3] = 0.99;

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve == 0.5, LOG);

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 1.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 1.0;
    outputs[1] = 1.0;
    outputs[2] = 0.0;
    outputs[3] = 0.0;

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve <= numeric_limits<double>::min(), LOG);

}


void TestingAnalysisTest::test_calculate_optimal_threshold()
{
    cout << "test_calculate_optimal_threshold\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<double> targets;
    Tensor<double> outputs;

    double optimal_threshold;

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 1.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.0;
    outputs[1] = 0.0;
    outputs[2] = 1.0;
    outputs[3] = 1.0;

    optimal_threshold = ta.calculate_optimal_threshold(targets, outputs);

    assert_true(optimal_threshold == 1.0, LOG);

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 0.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 1.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 1.0;
    outputs[1] = 1.0;
    outputs[2] = 0.0;
    outputs[3] = 0.0;

    optimal_threshold = ta.calculate_optimal_threshold(targets, outputs);

    assert_true(optimal_threshold == 0.0, LOG);

    // Test

    targets.set(Vector<size_t>{5,1});

    targets[0] = 0.0;
    targets[1] = 1.0;
    targets[2] = 0.0;
    targets[3] = 1.0;
    targets[4] = 0.0;

    outputs.set(Vector<size_t>{5,1});

    outputs[0] = 0.33;
    outputs[1] = 0.14;
    outputs[2] = 0.12;
    outputs[3] = 0.62;
    outputs[4] = 0.85;

    optimal_threshold = ta.calculate_optimal_threshold(targets, outputs);

    assert_true(optimal_threshold - 0.62 <= numeric_limits<double>::min(), LOG);
}


void TestingAnalysisTest::test_calculate_cumulative_gain()
{
    cout << "test_calculate_cumulative_chart\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<double> targets;
    Tensor<double> outputs;

    Matrix<double> cumulative_gain;

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 1.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 0.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.67;
    outputs[1] = 0.98;
    outputs[2] = 0.78;
    outputs[3] = 0.45;

    cumulative_gain = ta.calculate_cumulative_gain(targets, outputs);

    assert_true(cumulative_gain.get_columns_number() == 2, LOG);
    assert_true(cumulative_gain.get_rows_number() == 21, LOG);
    assert_true(cumulative_gain(0, 0) == 0.0, LOG);
    assert_true(cumulative_gain(0, 1) == 0.0, LOG);
    assert_true(cumulative_gain(20, 0) - 1.0 < 1.0e-6, LOG);
    assert_true(cumulative_gain(20, 1) == 1.0, LOG);
 }


void TestingAnalysisTest::test_calculate_lift_chart()
{
    cout << "test_calculate_lift_chart\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<double> targets;
    Tensor<double> outputs;

    Matrix<double> cumulative_gain;

    Matrix<double> lift_chart;

    // Test

    targets.set(Vector<size_t>{4,1});

    targets[0] = 1.0;
    targets[1] = 0.0;
    targets[2] = 1.0;
    targets[3] = 0.0;

    outputs.set(Vector<size_t>{4,1});

    outputs[0] = 0.67;
    outputs[1] = 0.87;
    outputs[2] = 0.99;
    outputs[3] = 0.88;

    cumulative_gain = ta.calculate_cumulative_gain(targets, outputs);

    lift_chart = ta.calculate_lift_chart(cumulative_gain);

    assert_true(lift_chart.get_columns_number() == cumulative_gain.get_columns_number(), LOG);
    assert_true(lift_chart.get_rows_number() == cumulative_gain.get_rows_number(), LOG);
}


void TestingAnalysisTest::test_calculate_calibration_plot()
{
    cout << "test_calculate_calibration_plot\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Matrix<double> targets;
    Matrix<double> outputs;

    Matrix<double> calibration_plot;

    // Test

    targets.set(10, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 1.0;
    targets(4, 0) = 1.0;
    targets(5, 0) = 1.0;
    targets(6, 0) = 1.0;
    targets(7, 0) = 0.0;
    targets(8, 0) = 1.0;
    targets(9, 0) = 0.0;

    outputs.set(10, 1);

    outputs(0, 0) = 0.09;
    outputs(1, 0) = 0.19;
    outputs(2, 0) = 0.29;
    outputs(3, 0) = 0.39;
    outputs(4, 0) = 0.49;
    outputs(5, 0) = 0.59;
    outputs(6, 0) = 0.58;
    outputs(7, 0) = 0.79;
    outputs(8, 0) = 0.89;
    outputs(9, 0) = 0.99;

    calibration_plot = ta.calculate_calibration_plot(targets.to_tensor(), outputs.to_tensor());

    assert_true(calibration_plot.get_columns_number() == 2, LOG);
    assert_true(calibration_plot.get_rows_number() == 11, LOG);
}


void TestingAnalysisTest::test_calculate_true_positive_instances()
{
    cout << "test_calculate_true_positive_instances\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Matrix<double> targets;
    Matrix<double> outputs;

    Vector<size_t> true_positives_indices;

    // Test

    targets.set(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 1.0;

    outputs.set(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 0.0;

    const Vector<size_t> testing_indices(0, 1, 3);
    const double threshold = 0.5;

    true_positives_indices = ta.calculate_true_positive_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(true_positives_indices.size() == 1, LOG);
    assert_true(true_positives_indices[0] == 1, LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.set(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_positives_indices = ta.calculate_true_positive_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(true_positives_indices.empty(), LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 1.0;

    outputs.set(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_positives_indices = ta.calculate_true_positive_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(true_positives_indices.size() == 4, LOG);
    assert_true(true_positives_indices[0] == 0, LOG);
    assert_true(true_positives_indices[1] == 1, LOG);
    assert_true(true_positives_indices[2] == 2, LOG);
    assert_true(true_positives_indices[3] == 3, LOG);
}


void TestingAnalysisTest::test_calculate_false_positive_instances()
{
    cout << "test_calculate_false_positive_instaces\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Matrix<double> targets;
    Matrix<double> outputs;

    Vector<size_t> false_positives_indices;

    // Test

    targets.set(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 1.0;

    outputs.set(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 0.0;

    const Vector<size_t> testing_indices(0, 1, 3);
    const double threshold = 0.5;

    false_positives_indices = ta.calculate_false_positive_instances(targets.to_tensor(), outputs.to_tensor(),testing_indices, threshold);

    assert_true(false_positives_indices.size() == 1, LOG);
    assert_true(false_positives_indices[0] == 2, LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.set(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    false_positives_indices = ta.calculate_false_positive_instances(targets.to_tensor(), outputs.to_tensor(),testing_indices, threshold);

    assert_true(false_positives_indices.size() == 4, LOG);
    assert_true(false_positives_indices[0] == 0, LOG);
    assert_true(false_positives_indices[1] == 1, LOG);
    assert_true(false_positives_indices[2] == 2, LOG);
    assert_true(false_positives_indices[3] == 3, LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 1.0;

    outputs.set(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 0.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    false_positives_indices = ta.calculate_false_positive_instances(targets.to_tensor(), outputs.to_tensor(),testing_indices, threshold);

    assert_true(false_positives_indices.empty(), LOG);
}


void TestingAnalysisTest::test_calculate_false_negative_instances()
{
    cout << "test_calculate_false_negative_instances\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Matrix<double> targets;
    Matrix<double> outputs;

    Vector<size_t> false_negatives_indices;

    // Test

    targets.set(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 1.0;

    outputs.set(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 0.0;

    const Vector<size_t> testing_indices(0, 1, 3);
    const double threshold = 0.5;

    false_negatives_indices = ta.calculate_false_negative_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(false_negatives_indices.size() == 1, LOG);
    assert_true(false_negatives_indices[0] == 3, LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.set(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 0.0;
    outputs(3, 0) = 0.0;

    false_negatives_indices = ta.calculate_false_negative_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(false_negatives_indices.empty(), LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 1.0;

    outputs.set(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 0.0;
    outputs(2, 0) = 0.0;
    outputs(3, 0) = 0.0;

    false_negatives_indices = ta.calculate_false_negative_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(false_negatives_indices.size() == 4, LOG);
    assert_true(false_negatives_indices[0] == 0, LOG);
    assert_true(false_negatives_indices[1] == 1, LOG);
    assert_true(false_negatives_indices[2] == 2, LOG);
    assert_true(false_negatives_indices[3] == 3, LOG);
}


void TestingAnalysisTest::test_calculate_true_negative_instances()
{
    cout << "test_calculate_true_negative_instances\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Matrix<double> targets;
    Matrix<double> outputs;

    Vector<size_t> true_negatives_indices;

    // Test

    targets.set(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.set(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 0.0;
    outputs(2, 0) = 0.0;
    outputs(3, 0) = 0.0;

    const Vector<size_t> testing_indices(0, 1, 3);
    const double threshold = 0.5;

    true_negatives_indices = ta.calculate_true_negative_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(true_negatives_indices.size() == 4, LOG);
    assert_true(true_negatives_indices[0] == 0, LOG);
    assert_true(true_negatives_indices[1] == 1, LOG);
    assert_true(true_negatives_indices[2] == 2, LOG);
    assert_true(true_negatives_indices[3] == 3, LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 0.0;

    outputs.set(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_negatives_indices = ta.calculate_true_negative_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(true_negatives_indices.empty(), LOG);

    // Test

    targets.set(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 0.0;

    outputs.set(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_negatives_indices = ta.calculate_true_negative_instances(targets.to_tensor(), outputs.to_tensor(), testing_indices, threshold);

    assert_true(true_negatives_indices.size() == 1, LOG);
    assert_true(true_negatives_indices[0] == 0, LOG);
}


void TestingAnalysisTest::test_calculate_multiple_classification_rates()
{
    cout << "test_calculate_multiple_classification_rates\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Matrix<double> targets;
    Matrix<double> outputs;

    // Test

    targets.set(9, 3);
    outputs.set(9, 3);

    targets(0,0) = 1; targets(0,1) = 0; targets(0,2) = 0;
    targets(1,0) = 0; targets(1,1) = 1; targets(1,2) = 0;
    targets(2,0) = 0; targets(2,1) = 0; targets(2,2) = 1;
    targets(3,0) = 1; targets(3,1) = 0; targets(3,2) = 0;
    targets(4,0) = 0; targets(4,1) = 1; targets(4,2) = 0;
    targets(5,0) = 0; targets(5,1) = 0; targets(5,2) = 1;
    targets(6,0) = 1; targets(6,1) = 0; targets(6,2) = 0;
    targets(7,0) = 0; targets(7,1) = 1; targets(7,2) = 0;
    targets(8,0) = 0; targets(8,1) = 0; targets(8,2) = 1;

    outputs(0,0) = 1; outputs(0,1) = 0; outputs(0,2) = 0;
    outputs(1,0) = 0; outputs(1,1) = 1; outputs(1,2) = 0;
    outputs(2,0) = 0; outputs(2,1) = 0; outputs(2,2) = 1;
    outputs(3,0) = 0; outputs(3,1) = 1; outputs(3,2) = 0;
    outputs(4,0) = 1; outputs(4,1) = 0; outputs(4,2) = 0;
    outputs(5,0) = 0; outputs(5,1) = 1; outputs(5,2) = 0;
    outputs(6,0) = 0; outputs(6,1) = 0; outputs(6,2) = 1;
    outputs(7,0) = 0; outputs(7,0) = 0; outputs(7,2) = 1;
    outputs(8,0) = 1; outputs(8,1) = 0; outputs(8,2) = 0;

    const Vector<size_t> testing_indices(0, 1, 8);

    Matrix< Vector<size_t> > multiple_classification_rates = ta.calculate_multiple_classification_rates(targets.to_tensor(), outputs.to_tensor(), testing_indices);

    assert_true(multiple_classification_rates(0,0).size() == 1, LOG);
    assert_true(multiple_classification_rates(0,0)[0] == 0, LOG);

    assert_true(multiple_classification_rates(0,1).size() == 1, LOG);
    assert_true(multiple_classification_rates(0,1)[0] == 3, LOG);

    assert_true(multiple_classification_rates(0,2).size() == 1, LOG);
    assert_true(multiple_classification_rates(0,2)[0] == 6, LOG);

    assert_true(multiple_classification_rates(1,0).size() == 1, LOG);
    assert_true(multiple_classification_rates(1,0)[0] == 4, LOG);

    assert_true(multiple_classification_rates(1,1).size() == 1, LOG);
    assert_true(multiple_classification_rates(1,1)[0] == 1, LOG);

    assert_true(multiple_classification_rates(1,2).size() == 1, LOG);
    assert_true(multiple_classification_rates(1,2)[0] == 7, LOG);

    assert_true(multiple_classification_rates(2,0).size() == 1, LOG);
    assert_true(multiple_classification_rates(2,0)[0] == 8, LOG);

    assert_true(multiple_classification_rates(2,1).size() == 1, LOG);
    assert_true(multiple_classification_rates(2,1)[0] == 5, LOG);

    assert_true(multiple_classification_rates(2,2).size() == 1, LOG);
    assert_true(multiple_classification_rates(2,2)[0] == 2, LOG);
}


void TestingAnalysisTest::run_test_case()
{
   cout << "Running testing analysis test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods
   
   test_get_neural_network_pointer();
   test_get_data_set_pointer();
   
   test_get_display();

   // Set methods

   test_set_neural_network_pointer();
   test_set_data_set_pointer();

   test_set_display();

   // Error data methods

   test_calculate_error_data();
   test_calculate_percentage_error_data();
   test_calculate_error_data_statistics();
   test_calculate_absolute_errors_statistics();
   test_calculate_percentage_errors_statistics();
   test_calculate_error_data_statistics_matrices();
   test_print_error_data_statistics();

   test_calculate_error_data_histograms();

   test_calculate_maximal_errors();

   // Linear regression analysis methodsta


   test_linear_regression();
   test_print_linear_regression_correlation();
   test_get_linear_regression_correlations_std();
   test_save_linear_regression();

   test_print_linear_regression_analysis();
   test_save_linear_regression_analysis();

   test_perform_linear_regression();


   // Binary classification test methods

   test_calculate_binary_classification_test();

   // Confusion matrix methods

   test_calculate_confusion();

   // ROC curve methods

   test_calculate_Wilcoxon_parameter();

   test_calculate_roc_curve();
   test_calculate_area_under_curve();
   test_calculate_optimal_threshold();

   // Lift chart methods

   test_calculate_cumulative_gain();
   test_calculate_lift_chart();

   // Calibration plot

   test_calculate_calibration_plot();

   // Binary classification rates

   test_calculate_true_positive_instances();
   test_calculate_false_positive_instances();
   test_calculate_false_negative_instances();
   test_calculate_true_negative_instances();

   // Multiple classification rates

   test_calculate_multiple_classification_rates();

   cout << "End of testing analysis test case.\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the s of the GNU Lesser General Public
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
