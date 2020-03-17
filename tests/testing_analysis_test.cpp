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

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    // Test

    Tensor<type, 3> error_data= testing_analysis.calculate_error_data();

    assert_true(error_data.size() == 3, LOG);
    assert_true(error_data.dimension(0) == 1, LOG);
    assert_true(error_data.dimension(1) == 3, LOG);
    assert_true(static_cast<double>(error_data(0,0,0)) == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_percentage_error_data()
{
    cout << "test_calculate_percentage_error_data\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<type, 2> error_data;

    // Test

    error_data = testing_analysis.calculate_percentage_error_data();

    assert_true(error_data.size() == 1, LOG);
    assert_true(error_data.dimension(1) == 1, LOG);
    assert_true(static_cast<double>(error_data(0,0)) == 0.0, LOG);
}


void TestingAnalysisTest::test_calculate_forecasting_error_data()
{}


void TestingAnalysisTest::test_calculate_absolute_errors_statistics()
{
    cout << "test_calculate_absolute_error_statistics\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<Descriptives, 1> error_data = testing_analysis.calculate_absolute_errors_statistics();

    assert_true(error_data.size() == 1, LOG);
    assert_true(static_cast<double>(error_data[0].minimum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data[0].maximum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data[0].mean) == 0.0, LOG);
    assert_true(static_cast<double>(error_data[0].standard_deviation) == 0.0, LOG);

}

void TestingAnalysisTest::test_calculate_percentage_errors_statistics()
{
    cout << "test_calculate_percentage_error_statistics\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<Descriptives, 1> error_data = testing_analysis.calculate_percentage_errors_statistics();

    assert_true(error_data.size() == 1, LOG);
    assert_true(static_cast<double>(error_data[0].standard_deviation) == 0.0, LOG);
}

void TestingAnalysisTest::test_calculate_error_data_statistics()
{
    cout << "test_calculate_error_data_statistics\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<Tensor<Descriptives, 1>, 1> error_data_statistics = testing_analysis.calculate_error_data_statistics();

    assert_true(error_data_statistics.size() == 1, LOG);
    assert_true(error_data_statistics[0].size() == 3, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].minimum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].maximum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].mean) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][0].standard_deviation) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].minimum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].maximum) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].mean) == 0.0, LOG);
    assert_true(static_cast<double>(error_data_statistics[0][2].standard_deviation) == 0.0, LOG);

}


void TestingAnalysisTest::test_print_error_data_statistics()
{
    cout << "test_print_error_data_statistics\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    testing_analysis.print_error_data_statistics();


}

void TestingAnalysisTest::test_calculate_error_data_statistics_matrices()
{
    cout << "test_calculate_error_data_statistics_matrices\n";

//    // Device

//    Device device(Device::EigenSimpleThreadPool);

//    // DataSet

//    DataSet data_set;
//    data_set.set(1,2);

//    data_set.set_device_pointer(&device);

//    data_set.initialize_data(0.0);

//    data_set.set_testing();

//    // Neural Network

//    Tensor<Index, 1> architecture(2);
//    architecture.setValues({1, 1});

//    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
//    neural_network.set_parameters_constant(0.0);

//    neural_network.set_device_pointer(&device);

//    // Testing Analysis

//    TestingAnalysis testing_analysis(&neural_network, &data_set);

//    Tensor<Tensor<type, 2>, 1> error_data_statistics = testing_analysis.calculate_error_data_statistics_matrices();

//    assert_true(error_data_statistics.size() == 1, LOG);
//    assert_true(error_data_statistics[0].dimension(0) == 2, LOG);
//    assert_true(error_data_statistics[0].dimension(1) == 4, LOG);
}


void TestingAnalysisTest::test_calculate_error_data_histograms()
{
    cout << "test_calculate_error_data_histograms\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<Histogram, 1> error_data_histograms = testing_analysis.calculate_error_data_histograms();

    assert_true(error_data_histograms.size() == 1, LOG);
    assert_true(error_data_histograms[0].get_bins_number() == 10, LOG);
}


void TestingAnalysisTest::test_calculate_maximal_errors()
{
    cout << "test_calculate_maximal_errors\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(4,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(2);
    architecture.setValues({1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<Tensor<Index, 1>, 1> error_data_maximal = testing_analysis.calculate_maximal_errors(2);

    assert_true(error_data_maximal.size() == 1, LOG);
    assert_true(error_data_maximal[0](0) == 0 , LOG);

}


void TestingAnalysisTest::test_linear_regression()
{
   cout << "test_linear_regression\n";

   // Device

   Device device(Device::EigenSimpleThreadPool);

   // DataSet

   DataSet data_set;
   data_set.set(1,2);

   data_set.set_device_pointer(&device);

   data_set.initialize_data(0.0);

   data_set.set_testing();

   // Neural Network

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1, 1, 1});

   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   neural_network.set_device_pointer(&device);

   // Testing Analysis

   TestingAnalysis testing_analysis(&neural_network, &data_set);

   Tensor<RegressionResults, 1> linear_regression = testing_analysis.linear_regression();

   assert_true(linear_regression.size() == 1, LOG);
   assert_true(static_cast<double>(linear_regression(0).a) == 0.0, LOG);
   assert_true(static_cast<double>(linear_regression(0).b) == 0.0, LOG);
   assert_true(static_cast<double>(linear_regression(0).correlation) == 1.0, LOG);
}


void TestingAnalysisTest::test_print_linear_regression_correlation()
{
   cout << "test_print_linear_regression_correlation\n";

   // Device

   Device device(Device::EigenSimpleThreadPool);

   // DataSet

   DataSet data_set;
   data_set.set(1,2);

   data_set.set_device_pointer(&device);

   data_set.initialize_data(0.0);

   data_set.set_testing();

   // Neural Network

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1, 1, 1});

   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   neural_network.set_device_pointer(&device);

   // Testing Analysis

   TestingAnalysis testing_analysis(&neural_network, &data_set);

   testing_analysis.print_linear_regression_correlations();
}


void TestingAnalysisTest::test_get_linear_regression_correlations_std()
{
    cout << "test_get_linear_regression_correlations_std\n";

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(3);
    architecture.setValues({1, 1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<type, 1> correlations = testing_analysis.get_linear_regression_correlations_std();

    assert_true(correlations.size() == 1, LOG);
    assert_true(correlations(0) == 1.0 , LOG);
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

    // Device

    Device device(Device::EigenSimpleThreadPool);

    // DataSet

    DataSet data_set;
    data_set.set(1,2);

    data_set.set_device_pointer(&device);

    data_set.initialize_data(0.0);

    data_set.set_testing();

    // Neural Network

    Tensor<Index, 1> architecture(3);
    architecture.setValues({1, 1, 1});

    NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    neural_network.set_device_pointer(&device);

    // Testing Analysis

    TestingAnalysis testing_analysis(&neural_network, &data_set);

    Tensor<TestingAnalysis::LinearRegressionAnalysis, 1> linear_regression_analysis;

    // Test

    linear_regression_analysis = testing_analysis.perform_linear_regression_analysis();

    Tensor<type, 1> test(1);
    test.setValues({0});

    assert_true(linear_regression_analysis.size() == 1 , LOG);
    assert_true(linear_regression_analysis[0].targets(0) == test(0) , LOG);
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

   Tensor<type, 2> actual;
   Tensor<type, 2> predicted;

   // Test

   actual.resize(4, 3);
   predicted.resize(4, 3);

   actual(0,0) = 1; actual(0,1) = 0; actual(0,2) = 0;
   actual(1,0) = 0; actual(1,1) = 1; actual(1,2) = 0;
   actual(2,0) = 0; actual(2,1) = 1; actual(2,2) = 0;
   actual(3,0) = 0; actual(3,1) = 0; actual(3,2) = 1;

   predicted(0,0) = 1; predicted(0,1) = 0; predicted(0,2) = 0;
   predicted(1,0) = 0; predicted(1,1) = 1; predicted(1,2) = 0;
   predicted(2,0) = 0; predicted(2,1) = 1; predicted(2,2) = 0;
   predicted(3,0) = 0; predicted(3,1) = 0; predicted(3,2) = 1;

   Tensor<Index, 2> confusion = ta.calculate_confusion_multiple_classification(actual, predicted);

   Tensor<Index, 0> sum = confusion.sum();

   assert_true(sum(0) == 4, LOG);
   assert_true(confusion(0,0) == 1, LOG);
   assert_true(confusion(1,1) == 2, LOG);
   assert_true(confusion(2,2) == 1, LOG);
   assert_true(confusion(0,2) == 0, LOG);
}


void TestingAnalysisTest::test_calculate_binary_classification_test()
{
   cout << "test_calculate_binary_classification_test\n";

   // Device

   Device device(Device::EigenSimpleThreadPool);

   // DataSet

   DataSet data_set;
   data_set.set(1,2);

   data_set.set_device_pointer(&device);

   data_set.initialize_data(0.0);

   data_set.set_testing();

   // Neural Network

   Tensor<Index, 1> architecture(3);
   architecture.setValues({1, 1, 1});

   NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);
   neural_network.set_parameters_constant(0.0);

   neural_network.set_device_pointer(&device);

   // Testing Analysis

   TestingAnalysis testing_analysis(&neural_network, &data_set);

   Tensor<type, 1> binary = testing_analysis.calculate_binary_classification_tests();

   assert_true(binary.size() == 15 , LOG);
}


void TestingAnalysisTest::test_calculate_Wilcoxon_parameter()
{
    cout << "test_calculate_Wilcoxon_parameter\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    type wilcoxon_parameter;

    // Test

    type x = 1.5;
    type y = 2.5;

    wilcoxon_parameter = ta.calculate_Wilcoxon_parameter(x, y);

    assert_true(abs(wilcoxon_parameter) <= numeric_limits<type>::min(), LOG);

    // Test

    wilcoxon_parameter = ta.calculate_Wilcoxon_parameter(y ,x);

    assert_true(abs(wilcoxon_parameter - 1) <= numeric_limits<type>::min(), LOG);

    // Test

    x = y;

    wilcoxon_parameter = ta.calculate_Wilcoxon_parameter(x, y);

    assert_true(abs(wilcoxon_parameter - 0.5) <= numeric_limits<type>::min(), LOG);
}


void TestingAnalysisTest::test_calculate_roc_curve()
{
    cout << "test_calculate_roc_curve\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<type, 2> roc_curve;

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 1.0;

    outputs.resize(4,1);

    outputs(0,0) = 0.0;
    outputs(1,0) = 0.0;
    outputs(2,0) = 1.0;
    outputs(3,0) = 1.0;

    roc_curve = ta.calculate_roc_curve(targets, outputs);

    assert_true(roc_curve.dimension(1) == 3, LOG);
    assert_true(roc_curve.dimension(0) == 5, LOG);

    assert_true(roc_curve(0, 0) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(0, 1) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(1, 0) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(1, 1)-1 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(2, 0) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(2, 1) - 1.0 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(3, 0) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(3, 1) - 1.0 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(4, 0) - 1.0 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(4, 1) - 1.0 <= numeric_limits<type>::min(), LOG);

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 1.0;
    targets(2,0) = 1.0;
    targets(3,0) = 0.0;

    outputs.resize(4,1);

    outputs(0,0) = static_cast<type>(0.12);
    outputs(1,0) = static_cast<type>(0.78);
    outputs(2,0) = static_cast<type>(0.84);
    outputs(3,0) = static_cast<type>(0.99);

    roc_curve = ta.calculate_roc_curve(targets, outputs);

    assert_true(roc_curve.dimension(1) == 3, LOG);
    assert_true(roc_curve.dimension(0) == 5, LOG);

    assert_true(roc_curve(0, 0) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(0, 1) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(1, 0) <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(1, 1) - 0.5 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(2, 0) - 0.5 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(2, 1) - 0.5 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(3, 0) - 1.0 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(3, 1) - 0.5 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(4, 0) - 1.0 <= numeric_limits<type>::min(), LOG);
    assert_true(roc_curve(4, 1) - 1.0 <= numeric_limits<type>::min(), LOG);
}


void TestingAnalysisTest::test_calculate_area_under_curve()
{
    cout << "test_calculate_area_under_curve\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    type area_under_curve;

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 1.0;

    outputs.resize(4,1);

    outputs(0,0) = 0.0;
    outputs(1,0) = 0.0;
    outputs(2,0) = 1.0;
    outputs(3,0) = 1.0;

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve - 1.0 <= numeric_limits<type>::min(), LOG);

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 1.0;

    outputs.resize(4,1);

    outputs(0,0) = 0.0;
    outputs(1,0) = 1.0;
    outputs(2,0) = 0.0;
    outputs(3,0) = 1.0;

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve == 0.5, LOG);

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 1.0;

    outputs.resize(4,1);

    outputs(0,0) = static_cast<type>(0.78);
    outputs(1,0) = static_cast<type>(0.84);
    outputs(2,0) = static_cast<type>(0.12);
    outputs(3,0) = static_cast<type>(0.99);

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve == 0.5, LOG);

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 1.0;

    outputs.resize(4,1);

    outputs(0,0) = 1.0;
    outputs(1,0) = 1.0;
    outputs(2,0) = 0.0;
    outputs(3,0) = 0.0;

    area_under_curve = ta.calculate_area_under_curve(targets, outputs);

    assert_true(area_under_curve <= numeric_limits<type>::min(), LOG);

}


void TestingAnalysisTest::test_calculate_optimal_threshold()
{
    cout << "test_calculate_optimal_threshold\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    type optimal_threshold;

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 1.0;

    outputs.resize(4,1);

    outputs(0,0) = 0.0;
    outputs(1,0) = 0.0;
    outputs(2,0) = 1.0;
    outputs(3,0) = 1.0;

    optimal_threshold = ta.calculate_optimal_threshold(targets, outputs);

    assert_true(optimal_threshold == 1.0, LOG);

    // Test

    targets.resize(4,1);

    targets(0,0) = 0.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 1.0;

    outputs.resize(4,1);

    outputs(0,0) = 1.0;
    outputs(1,0) = 1.0;
    outputs(2,0) = 0.0;
    outputs(3,0) = 0.0;

    optimal_threshold = ta.calculate_optimal_threshold(targets, outputs);

    assert_true(optimal_threshold == 0.0, LOG);

    // Test

    targets.resize(5,1);

    targets(0,0) = 0.0;
    targets(1,0) = 1.0;
    targets(2,0) = 0.0;
    targets(3,0) = 1.0;
    targets(4,0) = 0.0;

    outputs.resize(5,1);

    outputs(0,0) = static_cast<type>(0.33);
    outputs(1,0) = static_cast<type>(0.14);
    outputs(2,0) = static_cast<type>(0.12);
    outputs(3,0) = static_cast<type>(0.62);
    outputs(4,0) = static_cast<type>(0.85);

    optimal_threshold = ta.calculate_optimal_threshold(targets, outputs);

    assert_true(optimal_threshold - 0.62 <= numeric_limits<type>::min(), LOG);
}


void TestingAnalysisTest::test_calculate_cumulative_gain()
{
    cout << "test_calculate_cumulative_chart\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    // Test

    targets.resize(4,1);

    targets(0,0) = 1.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 0.0;

    outputs.resize(4,1);

    outputs(0,0) = static_cast<type>(0.67);
    outputs(1,0) = static_cast<type>(0.98);
    outputs(2,0) = static_cast<type>(0.78);
    outputs(3,0) = static_cast<type>(0.45);

    Tensor<type, 2> cumulative_gain = ta.calculate_cumulative_gain(targets, outputs);

    assert_true(cumulative_gain.dimension(1) == 2, LOG);
    assert_true(cumulative_gain.dimension(0) == 21, LOG);
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

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<type, 2> cumulative_gain;

    Tensor<type, 2> lift_chart;

    // Test

    targets.resize(4,1);

    targets(0,0) = 1.0;
    targets(1,0) = 0.0;
    targets(2,0) = 1.0;
    targets(3,0) = 0.0;

    outputs.resize(4,1);

    outputs(0,0) = static_cast<type>(0.67);
    outputs(1,0) = static_cast<type>(0.87);
    outputs(2,0) = static_cast<type>(0.99);
    outputs(3,0) = static_cast<type>(0.88);

    cumulative_gain = ta.calculate_cumulative_gain(targets, outputs);

    lift_chart = ta.calculate_lift_chart(cumulative_gain);

    assert_true(lift_chart.dimension(1) == cumulative_gain.dimension(1), LOG);
    assert_true(lift_chart.dimension(0) == cumulative_gain.dimension(0), LOG);
}


void TestingAnalysisTest::test_calculate_calibration_plot()
{
    cout << "test_calculate_calibration_plot\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<type, 2> calibration_plot;

    // Test

    targets.resize(10, 1);

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

    outputs.resize(10, 1);

    outputs(0, 0) = static_cast<type>(0.09);
    outputs(1, 0) = static_cast<type>(0.19);
    outputs(2, 0) = static_cast<type>(0.29);
    outputs(3, 0) = static_cast<type>(0.39);
    outputs(4, 0) = static_cast<type>(0.49);
    outputs(5, 0) = static_cast<type>(0.59);
    outputs(6, 0) = static_cast<type>(0.58);
    outputs(7, 0) = static_cast<type>(0.79);
    outputs(8, 0) = static_cast<type>(0.89);
    outputs(9, 0) = static_cast<type>(0.99);

    calibration_plot = ta.calculate_calibration_plot(targets, outputs);

    assert_true(calibration_plot.dimension(1) == 2, LOG);
    assert_true(calibration_plot.dimension(0) == 11, LOG);
}


void TestingAnalysisTest::test_calculate_true_positive_instances()
{
    cout << "test_calculate_true_positive_instances\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<Index, 1> true_positives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 1.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 0.0;

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});

    const type threshold = 0.5;

    true_positives_indices = ta.calculate_true_positive_instances(targets, outputs, testing_indices, threshold);

    assert_true(true_positives_indices.size() == 1, LOG);
    assert_true(true_positives_indices[0] == 1, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_positives_indices = ta.calculate_true_positive_instances(targets, outputs, testing_indices, threshold);

    const Tensor<bool, 0> not_empty = true_positives_indices.any();

    assert_true(!not_empty(0), LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 1.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_positives_indices = ta.calculate_true_positive_instances(targets, outputs, testing_indices, threshold);

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

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<Index, 1> false_positives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 1.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 0.0;

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = 0.5;

    false_positives_indices = ta.calculate_false_positive_instances(targets, outputs,testing_indices, threshold);

    assert_true(false_positives_indices.size() == 1, LOG);
    assert_true(false_positives_indices[0] == 2, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    false_positives_indices = ta.calculate_false_positive_instances(targets, outputs, testing_indices, threshold);

    assert_true(false_positives_indices.size() == 4, LOG);
    assert_true(false_positives_indices[0] == 0, LOG);
    assert_true(false_positives_indices[1] == 1, LOG);
    assert_true(false_positives_indices[2] == 2, LOG);
    assert_true(false_positives_indices[3] == 3, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 1.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 0.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    false_positives_indices = ta.calculate_false_positive_instances(targets, outputs,testing_indices, threshold);

    const Tensor<bool, 0> not_empty = false_positives_indices.any();

    assert_true(!not_empty(0), LOG);

//    assert_true(false_positives_indices.empty(), LOG);
}


void TestingAnalysisTest::test_calculate_false_negative_instances()
{
    cout << "test_calculate_false_negative_instances\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<Index, 1> false_negatives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 1.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 0.0;

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = 0.5;

    false_negatives_indices = ta.calculate_false_negative_instances(targets, outputs, testing_indices, threshold);

    assert_true(false_negatives_indices.size() == 1, LOG);
    assert_true(false_negatives_indices[0] == 3, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 1.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 0.0;
    outputs(3, 0) = 0.0;

    false_negatives_indices = ta.calculate_false_negative_instances(targets, outputs, testing_indices, threshold);

//    assert_true(false_negatives_indices.empty(), LOG);

    const Tensor<bool, 0> not_empty = false_negatives_indices.any();

    assert_true(!not_empty(0), LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 1.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 0.0;
    outputs(2, 0) = 0.0;
    outputs(3, 0) = 0.0;

    false_negatives_indices = ta.calculate_false_negative_instances(targets, outputs, testing_indices, threshold);

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

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    Tensor<Index, 1> true_negatives_indices;

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 0.0;
    targets(3, 0) = 0.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 0.0;
    outputs(2, 0) = 0.0;
    outputs(3, 0) = 0.0;

    Tensor<Index, 1> testing_indices(4);
    testing_indices.setValues({0, 1, 2, 3});
    const type threshold = 0.5;

    true_negatives_indices = ta.calculate_true_negative_instances(targets, outputs, testing_indices, threshold);

    assert_true(true_negatives_indices.size() == 4, LOG);
    assert_true(true_negatives_indices[0] == 0, LOG);
    assert_true(true_negatives_indices[1] == 1, LOG);
    assert_true(true_negatives_indices[2] == 2, LOG);
    assert_true(true_negatives_indices[3] == 3, LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 1.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 0.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_negatives_indices = ta.calculate_true_negative_instances(targets, outputs, testing_indices, threshold);

    const Tensor<bool, 0> not_empty = true_negatives_indices.any();

    assert_true(!not_empty(0), LOG);

    // Test

    targets.resize(4, 1);

    targets(0, 0) = 0.0;
    targets(1, 0) = 0.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 0.0;

    outputs.resize(4, 1);

    outputs(0, 0) = 0.0;
    outputs(1, 0) = 1.0;
    outputs(2, 0) = 1.0;
    outputs(3, 0) = 1.0;

    true_negatives_indices = ta.calculate_true_negative_instances(targets, outputs, testing_indices, threshold);

    assert_true(true_negatives_indices.size() == 1, LOG);
    assert_true(true_negatives_indices[0] == 0, LOG);
}


void TestingAnalysisTest::test_calculate_multiple_classification_rates()
{
    cout << "test_calculate_multiple_classification_rates\n";
/*
    NeuralNetwork neural_network;
    DataSet data_set;

    TestingAnalysis ta(&neural_network, &data_set);

    Tensor<type, 2> targets;
    Tensor<type, 2> outputs;

    // Test

    targets.resize(9, 3);
    outputs.resize(9, 3);

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

    Tensor<Index, 1> testing_indices(9);
    testing_indices.setValues({0, 1, 2, 3, 4, 5, 6, 7, 8});

    Tensor< Tensor<Index, 1>, 2 > multiple_classification_rates = ta.calculate_multiple_classification_rates(targets, outputs, testing_indices);

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
    */
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
//   test_calculate_error_data_statistics_matrices();
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
