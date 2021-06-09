//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer_test.h"


ScalingLayerTest::ScalingLayerTest() : UnitTesting()
{
    scaling_layer.set_display(false);

}


ScalingLayerTest::~ScalingLayerTest()
{
}


void ScalingLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   ScalingLayer scaling_layer_1;

   assert_true(scaling_layer_1.get_type() == Layer::Scaling, LOG);
   assert_true(scaling_layer_1.get_descriptives().size() == 0, LOG);

   ScalingLayer scaling_layer_2(3);

   assert_true(scaling_layer_2.get_descriptives().size() == 3, LOG);
   assert_true(scaling_layer_2.get_scaling_methods().size() == 3, LOG);

   descriptives.resize(2);

   ScalingLayer scaling_layer_3(descriptives);

   assert_true(scaling_layer_3.get_descriptives().size() == 2, LOG);
}


void ScalingLayerTest::test_get_inputs_number()
{
   cout << "test_get_inputs_number\n";  

   // Test

   assert_true(scaling_layer.get_inputs_number() == 0, LOG);
   assert_true(scaling_layer.get_inputs_number() == scaling_layer.get_neurons_number(), LOG);

   // Test

   scaling_layer.set(3);

   descriptives.resize(3);
   scaling_layer.set_descriptives(descriptives);

   assert_true(scaling_layer.get_inputs_number() == 3, LOG);
   assert_true(scaling_layer.get_inputs_number() == scaling_layer.get_neurons_number(), LOG);
}


void ScalingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   // Test

   scaling_layer.set();

   assert_true(scaling_layer.get_neurons_number() == 0, LOG);
   assert_true(scaling_layer.get_neurons_number() == scaling_layer.get_inputs_number(), LOG);

   // Test

   scaling_layer.set(3);

   descriptives.resize(3);
   scaling_layer.set_descriptives(descriptives);

   assert_true(scaling_layer.get_neurons_number() == 3, LOG);
   assert_true(scaling_layer.get_neurons_number() == scaling_layer.get_inputs_number(), LOG);
}


void ScalingLayerTest::test_get_descriptives()
{
   cout << "test_get_descriptives\n";

   Descriptives item_descriptives;

   // Test
/*
   descriptives.resize(1);

   scaling_layer.set_descriptives(descriptives);

   descriptives = scaling_layer.get_descriptives();

   assert_true(abs(descriptives(0).minimum + 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(descriptives(0).maximum - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(descriptives(0).mean - 0) < numeric_limits<type>::min(), LOG);
   assert_true(abs(descriptives(0).standard_deviation - 1) < numeric_limits<type>::min(), LOG);

   // Test

   Descriptives descriptives_1(1,1,1,0);
   Descriptives descriptives_2(2,2,2,0);

   descriptives.resize(2);
   descriptives.setValues({descriptives_1, descriptives_2});

   scaling_layer.set_descriptives(descriptives);

   descriptives = scaling_layer.get_descriptives();

   assert_true(abs(descriptives(1).minimum - 2) < numeric_limits<type>::min(), LOG);
   assert_true(abs(descriptives(1).maximum - 2) < numeric_limits<type>::min(), LOG);
   assert_true(abs(descriptives(1).mean - 2) < numeric_limits<type>::min(), LOG);
   assert_true(abs(descriptives(1).standard_deviation) - 0 < numeric_limits<type>::min(), LOG);

   // Test

   descriptives.resize(1);
   scaling_layer.set(descriptives);

   item_descriptives = scaling_layer.get_descriptives(0);

   assert_true(abs(item_descriptives.minimum + 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(item_descriptives.maximum - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(item_descriptives.mean - 0) < numeric_limits<type>::min(), LOG);
   assert_true(abs(item_descriptives.standard_deviation - 1) < numeric_limits<type>::min(), LOG);

   // Test

   item_descriptives = scaling_layer.get_descriptives(0);

   assert_true(abs(item_descriptives.minimum - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(item_descriptives.maximum - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(item_descriptives.mean - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(item_descriptives.standard_deviation - 0) < numeric_limits<type>::min(), LOG);
*/
}


void ScalingLayerTest::test_get_minimums()
{
   cout << "test_get_minimums\n";

   scaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_minimums()(0) + 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_minimums()(1) + 1) < numeric_limits<type>::min(), LOG);

   // Test

   descriptives(0).minimum = 1;
   descriptives(1).minimum = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_minimums()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_minimums()(1) + 1) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_get_maximums()
{
   cout << "test_get_maximums\n";

   scaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_maximums()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_maximums()(1) - 1) < numeric_limits<type>::min(), LOG);

   // Test

   descriptives(0).maximum = 1;
   descriptives(1).maximum = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_maximums()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_maximums()(1) + 1) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_get_means()
{
   cout << "test_get_means\n";

   scaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_means()(0) - 0) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_means()(1) - 0) < numeric_limits<type>::min(), LOG);

   // Test

   descriptives(0).mean = 1;
   descriptives(1).mean = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_means()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_means()(1) + 1) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_get_standard_deviations()
{
   cout << "test_get_standard_deviations\n";

   scaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_standard_deviations()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_standard_deviations()(1) - 1) < numeric_limits<type>::min(), LOG);

   // Test

   descriptives(0).standard_deviation = 1;
   descriptives(1).standard_deviation = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_standard_deviations()(0) - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(scaling_layer.get_standard_deviations()(1) + 1) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_get_scaling_method()
{
   cout << "test_get_scaling_method\n";

   scaling_layer.set(1);

   // Test

   Scaler no_scaling = Scaler::NoScaling;

   Scaler minimum_maximum = Scaler::MinimumMaximum;

   Scaler mean_standard_deviation = Scaler::MeanStandardDeviation;

   Scaler standard_deviation = Scaler::StandardDeviation;

   scaling_layer.set_scalers(no_scaling);

   assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
   assert_true(scaling_layer.get_scaling_methods()(0) == 0, LOG);

   scaling_layer.set_scalers(minimum_maximum);

   assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MinimumMaximum, LOG);
   assert_true(scaling_layer.get_scaling_methods()(0) == 1, LOG);

   scaling_layer.set_scalers(mean_standard_deviation);

   assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MeanStandardDeviation, LOG);
   assert_true(scaling_layer.get_scaling_methods()(0) == 2, LOG);

   scaling_layer.set_scalers(standard_deviation);

   assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::StandardDeviation, LOG);
   assert_true(scaling_layer.get_scaling_methods()(0) == 3, LOG);
}


void ScalingLayerTest::test_write_scalers()
{
   cout << "test_get_scaling_method_name\n";

   scaling_layer.set(1);

   // Test

   Scaler no_scaling = Scaler::NoScaling;

   Scaler minimum_maximum = Scaler::MinimumMaximum;

   Scaler mean_standard_deviation = Scaler::MeanStandardDeviation;

   Scaler standard_deviation = Scaler::StandardDeviation;

   scaling_layer.set_scalers(no_scaling);
   assert_true(scaling_layer.write_scalers()(0) == "NoScaling", LOG);

   scaling_layer.set_scalers(minimum_maximum);
   assert_true(scaling_layer.write_scalers()(0) == "MinimumMaximum", LOG);

   scaling_layer.set_scalers(mean_standard_deviation);
   assert_true(scaling_layer.write_scalers()(0) == "MeanStandardDeviation", LOG);

   scaling_layer.set_scalers(standard_deviation);
   assert_true(scaling_layer.write_scalers()(0) == "StandardDeviation", LOG);

   // Test

   scaling_layer.set_scalers(no_scaling);
   assert_true(scaling_layer.write_scalers_text()(0) == "no scaling", LOG);

   scaling_layer.set_scalers(minimum_maximum);
   assert_true(scaling_layer.write_scalers_text()(0) == "minimum and maximum", LOG);

   scaling_layer.set_scalers(mean_standard_deviation);
   assert_true(scaling_layer.write_scalers_text()(0) == "mean and standard deviation", LOG);

   scaling_layer.set_scalers(standard_deviation);
   assert_true(scaling_layer.write_scalers_text()(0) == "standard deviation", LOG);
}


void ScalingLayerTest::test_set()
{
   cout << "test_set\n";
/*
   // Test

   scaling_layer.set();

   assert_true(scaling_layer.get_descriptives().size() == 0, LOG);

   descriptives.resize(4);
   scaling_layer.set_descriptives(descriptives);
   scaling_layer.set();

   assert_true(scaling_layer.get_descriptives().size() == 0, LOG);

   // Test

   Index new_inputs_number_ = 4;
   scaling_layer.set(new_inputs_number_);

   assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);

   // Test

   scaling_layer.set();

   Tensor<Index, 1> new_inputs_dimensions(1);
   new_inputs_dimensions.setConstant(3);
   scaling_layer.set(new_inputs_dimensions);

   assert_true(scaling_layer.get_descriptives().size()== 3, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 3, LOG);

   // Test 

   scaling_layer.set();

   descriptives.resize(0);
   scaling_layer.set(descriptives);

   assert_true(scaling_layer.get_descriptives().size()== 0, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 0, LOG);

   // Test

   descriptives.resize(4);
   scaling_layer.set(descriptives);

   assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);
*/
}


void ScalingLayerTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

   Index new_inputs_number(0);
   scaling_layer.set_inputs_number(new_inputs_number);

   assert_true(scaling_layer.get_descriptives().size()== 0, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 0, LOG);

   Index new_inputs_number_ = 4;
   scaling_layer.set_inputs_number(new_inputs_number_);

   assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);
}


void ScalingLayerTest::test_set_neurons_number()
{
   cout << "test_set_neurons_number\n";

   Index new_inputs_number(0);
   scaling_layer.set_neurons_number(new_inputs_number);

   assert_true(scaling_layer.get_descriptives().size()== 0, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 0, LOG);

   Index new_inputs_number_ = 4;
   scaling_layer.set_neurons_number(new_inputs_number_);

   assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);
}


void ScalingLayerTest::test_set_default()
{
   cout << "test_set_default\n";

   scaling_layer.set(1);

   scaling_layer.set_default();

   Tensor<Descriptives, 1> sl_descriptives = scaling_layer.get_descriptives();

   assert_true(scaling_layer.get_scaling_methods()(0) == MinimumMaximum, LOG);
   assert_true(scaling_layer.get_display(), LOG);
   assert_true(scaling_layer.get_type() == Layer::Scaling, LOG);
   assert_true(scaling_layer.get_type() == 0, LOG);
   assert_true(abs(sl_descriptives(0).minimum + 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(sl_descriptives(0).maximum - 1) < numeric_limits<type>::min(), LOG);
   assert_true(abs(sl_descriptives(0).mean - 0) < numeric_limits<type>::min(), LOG);
   assert_true(abs(sl_descriptives(0).standard_deviation - 1) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_set_descriptives()
{
   cout << "test_set_descriptives\n";  

   Descriptives item_0(1,1,1,0);
   Descriptives item_1(2,2,2,0);

   // Test
/*
   descriptives.resize(1);

   scaling_layer.set_descriptives(descriptives);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 0) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 1) < numeric_limits<type>::min(), LOG);

   // Test

   descriptives.resize(2);
   descriptives.setValues({item_0, item_1});

   scaling_layer.set_descriptives(descriptives);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 2) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 0) < numeric_limits<type>::min(), LOG);
*/
}


void ScalingLayerTest::test_set_item_descriptives()
{
   cout << "test_set_item_descriptives\n";

   Descriptives item_descriptives_1;
   Descriptives item_descriptives_2;

   // Test

   scaling_layer.set(1);

   scaling_layer.set_item_descriptives(0, item_descriptives_1);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 0) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 1) < numeric_limits<type>::min(), LOG);

   // Test

   scaling_layer.set(2);

//   item_descriptives_1.set(1,1,1,0);
//   item_descriptives_2.set(2,2,2,0);

   scaling_layer.set_item_descriptives(0, item_descriptives_1);
   scaling_layer.set_item_descriptives(1, item_descriptives_1);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 1) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 2) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 0) < numeric_limits<type>::min(), LOG);

}

void ScalingLayerTest::test_set_minimum()
{
   cout << "test_set_minimum\n";

   // Test

   scaling_layer.set(2);
   descriptives.resize(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_minimum(0, -5);
   scaling_layer.set_minimum(1, -6);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,0) + 6) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_set_maximum()
{
   cout << "test_set_maximum\n";

   scaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_maximum(0, 5);
   scaling_layer.set_maximum(1, 6);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 6) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_set_mean()
{
   cout << "test_set_mean\n";

   scaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_mean(0, 5);
   scaling_layer.set_mean(1, 6);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,2) - 6) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";

   scaling_layer.set(2);

   // Test

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_standard_deviation(0, 5);
   scaling_layer.set_standard_deviation(1, 6);

//   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 5) < numeric_limits<type>::min(), LOG);
//   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 6) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_set_scaling_method()
{
   cout << "test_set_scaling_method\n";

    scaling_layer.set(4);

    // Test

    Tensor<Scaler, 1> method_tensor_1(4);
    method_tensor_1.setValues({Scaler::NoScaling,
                                Scaler::MinimumMaximum,
                                Scaler::MeanStandardDeviation,
                                Scaler::StandardDeviation});

    scaling_layer.set_scalers(method_tensor_1);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 0, LOG);

    assert_true(scaling_layer.get_scaling_methods()(1) == Scaler::MinimumMaximum, LOG);
    assert_true(scaling_layer.get_scaling_methods()(1) == 1, LOG);

    assert_true(scaling_layer.get_scaling_methods()(2) == Scaler::MeanStandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(2) == 2, LOG);

    assert_true(scaling_layer.get_scaling_methods()(3) == Scaler::StandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(3) == 3, LOG);

    // Test

    Tensor<string, 1> method_tensor_2(4);
    method_tensor_2.setValues({"NoScaling",
                                "MinimumMaximum",
                                "MeanStandardDeviation",
                                "StandardDeviation"});

    scaling_layer.set_scalers(method_tensor_2);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 0, LOG);

    assert_true(scaling_layer.get_scaling_methods()(1) == Scaler::MinimumMaximum, LOG);
    assert_true(scaling_layer.get_scaling_methods()(1) == 1, LOG);

    assert_true(scaling_layer.get_scaling_methods()(2) == Scaler::MeanStandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(2) == 2, LOG);

    assert_true(scaling_layer.get_scaling_methods()(3) == Scaler::StandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(3) == 3, LOG);

    // Test

    string no_scaling = "NoScaling";
    string minimum_maximum = "MinimumMaximum";
    string mean_standard_deviation = "MeanStandardDeviation";
    string standard_deviation = "StandardDeviation";

    scaling_layer.set_scalers(no_scaling);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 0, LOG);

    scaling_layer.set_scalers(minimum_maximum);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MinimumMaximum, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 1, LOG);

    scaling_layer.set_scalers(mean_standard_deviation);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MeanStandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 2, LOG);

    scaling_layer.set_scalers(standard_deviation);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::StandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 3, LOG);

    // Test 

    Scaler no_scaling_4 = Scaler::NoScaling;

    Scaler minimum_maximum_4 = Scaler::MinimumMaximum;

    Scaler mean_standard_deviation_4 = Scaler::MeanStandardDeviation;

    Scaler standard_deviation_4 = Scaler::StandardDeviation;

    scaling_layer.set_scalers(no_scaling_4);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::NoScaling, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 0, LOG);

    scaling_layer.set_scalers(minimum_maximum_4);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MinimumMaximum, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 1, LOG);

    scaling_layer.set_scalers(mean_standard_deviation_4);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::MeanStandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 2, LOG);

    scaling_layer.set_scalers(standard_deviation_4);

    assert_true(scaling_layer.get_scaling_methods()(0) == Scaler::StandardDeviation, LOG);
    assert_true(scaling_layer.get_scaling_methods()(0) == 3, LOG);
}


void ScalingLayerTest::test_is_empty()
{
   cout << "test_is_empty\n";

   scaling_layer.set(1);

   assert_true(scaling_layer.is_empty(), LOG);
   assert_true(!scaling_layer.is_empty(), LOG);
}


void ScalingLayerTest::test_check_range()
{
   cout << "test_check_range\n";

   Tensor<type, 1> inputs;

   // Test

   scaling_layer.set(1);

   inputs.resize(1);
   inputs.setConstant(0.0);
   scaling_layer.check_range(inputs);

   // Test

   Tensor<Descriptives, 1> descriptives(1);
   Descriptives des(-1,1,1,0);
   descriptives.setValues({des});

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.check_range(inputs);
}


void ScalingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   Tensor<type, 2> inputs;
   Tensor<type, 2> outputs;

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(NoScaling);

   inputs.resize(1,1);
   outputs = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 1, LOG);
   assert_true(abs(outputs(0) - inputs(0)) < numeric_limits<type>::min(), LOG);

   // Test

   scaling_layer.set(3);
   scaling_layer.set_scalers(NoScaling);

   inputs.resize(1,3);
   inputs.setConstant(0);
   outputs = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 3, LOG);
   assert_true(abs(outputs(0)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(2)) < numeric_limits<type>::min(), LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(MinimumMaximum);

   inputs.resize(1,1);
   outputs = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 1, LOG);

   assert_true(abs(outputs(0) - inputs(0)) < numeric_limits<type>::min(), LOG);

   // Test

   scaling_layer.set(3);
   scaling_layer.set_scalers(MinimumMaximum);

   Tensor<type, 2> minimums_maximums(3, 4);
   minimums_maximums.setValues({{-1,2,0,0},{-2,4,0,0},{-3,6,0,0}});

   inputs.resize(1,3);
   inputs.setConstant(0);
   outputs = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 3, LOG);
   assert_true(abs(outputs(0) + static_cast<type>(0.333)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1) + static_cast<type>(0.333)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(2) + static_cast<type>(0.333)) < numeric_limits<type>::min(), LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(MeanStandardDeviation);

   inputs.resize(1,1);
   outputs = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 1, LOG);
   assert_true(abs(outputs(0) - inputs(0)) < numeric_limits<type>::min(), LOG);

   // Test

   scaling_layer.set(2);
   scaling_layer.set_scalers(MeanStandardDeviation);

   Tensor<type, 2> mean_standard_deviation(2,4);
   mean_standard_deviation.setValues({{-1,1,-1,2},{-1,1,1,4}});

   inputs.resize(1,2);
   inputs.setConstant(0);
   outputs = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);

   assert_true(abs(outputs(0) - static_cast<type>(0.5)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1) + static_cast<type>(0.25)) < numeric_limits<type>::min(), LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(StandardDeviation);

   inputs.resize(1,1);
   outputs = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 1, LOG);
   assert_true(abs(outputs(0) - inputs(0)) < numeric_limits<type>::min(), LOG);

   // Test
   scaling_layer.set(2);
   scaling_layer.set_scalers(StandardDeviation);

   Tensor<type, 2> standard_deviation(2,4);
   standard_deviation.setValues({{-1,1,-1,2},{-1,1,1,4}});

   inputs.resize(1,2);
   inputs.setConstant(1);
   outputs = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 2, LOG);
   assert_true(abs(outputs(0) - static_cast<type>(0.5)) < numeric_limits<type>::min(), LOG);
   assert_true(abs(outputs(1) - static_cast<type>(0.25)) < numeric_limits<type>::min(), LOG);
}


void ScalingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   Tensor<string, 1> inputs_names(1);
   Tensor<string, 1> outputs_names(1);

   string expression;

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(NoScaling);
   inputs_names.setValues({"x"});
   outputs_names.setValues({"y"});

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = x;\n", LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(MinimumMaximum);

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = x*(1+1)/(1-(-1))+1*(1+1)/(1+1)-1;\n", LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(MeanStandardDeviation);

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = (x-(0))/1;\n", LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(StandardDeviation);

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = x/(1);\n", LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(NoScaling);
   inputs_names.setValues({"x"});
   outputs_names.setValues({"y"});

   expression = scaling_layer.write_no_scaling_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = x;\n", LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(MinimumMaximum);

   expression = scaling_layer.write_minimum_maximum_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = 2*(x-(-1))/(1-(-1))-1;\n", LOG);

   // Test

   scaling_layer.set(1);
   scaling_layer.set_scalers(MeanStandardDeviation);

   expression = scaling_layer.write_mean_standard_deviation_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = (x-(0))/1;\n", LOG);

   // Test 

   scaling_layer.set(1);
   scaling_layer.set_scalers(StandardDeviation);

   expression = scaling_layer.write_standard_deviation_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = x/(1);\n", LOG);
}


void ScalingLayerTest::run_test_case()
{
   cout << "Running scaling layer test case...\n";

   // Constructor and destructor methods

   test_constructor();

   // Scaling layer architecture

   test_get_inputs_number();
   test_get_neurons_number();

   // Input variables descriptives

   test_get_descriptives();
   test_get_minimums();
   test_get_maximums();
   test_get_means();
   test_get_standard_deviations();

   // Variables scaling and unscaling

   test_get_scaling_method();
   test_write_scalers();

   // Set methods

   test_set();
   test_set_inputs_number();
   test_set_neurons_number();
   test_set_default();

   // Input variables descriptives

   test_set_descriptives();
   test_set_item_descriptives();
   test_set_minimum();
   test_set_maximum();
   test_set_mean();
   test_set_standard_deviation();

   // Variables scaling and unscaling

   test_set_scaling_method();

   // Input range

   test_is_empty();
   test_check_range();

   // Scaling and unscaling

   test_calculate_outputs();

   // Expression methods

   test_write_expression();

   cout << "End of scaling layer test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
//
// This library sl free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library sl distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
