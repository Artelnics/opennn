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
}


ScalingLayerTest::~ScalingLayerTest()
{
}


void ScalingLayerTest::test_constructor()
{
   cout << "test_constructor\n";

   Tensor<Descriptives, 1> descriptives;

   ScalingLayer scaling_layer_1;

   assert_true(scaling_layer_1.get_type() == Layer::Scaling, LOG);
   assert_true(scaling_layer_1.get_descriptives().size() == 0, LOG);

   ScalingLayer sl2(3);

   assert_true(sl2.get_descriptives().size() == 3, LOG);
   assert_true(sl2.get_scaling_methods().size() == 3, LOG);


   descriptives.resize(2);

   ScalingLayer sl3(descriptives);

   assert_true(sl3.get_descriptives().size() == 2, LOG);
}


void ScalingLayerTest::test_destructor()
{
   cout << "test_destructor\n";

}


void ScalingLayerTest::test_get_inputs_number()
{
   cout << "test_get_neurons_number\n";

   ScalingLayer scaling_layer;

   Tensor<Descriptives, 1> descriptives;

   // Test 0

   assert_true(scaling_layer.get_inputs_number() == 0, LOG);
   assert_true(scaling_layer.get_inputs_number() == scaling_layer.get_neurons_number(), LOG);

   // Test 1

   scaling_layer.set(3);

   descriptives.resize(3);
   scaling_layer.set_descriptives(descriptives);

   assert_true(scaling_layer.get_inputs_number() == 3, LOG);
   assert_true(scaling_layer.get_inputs_number() == scaling_layer.get_neurons_number(), LOG);
}


void ScalingLayerTest::test_get_neurons_number()
{
   cout << "test_get_neurons_number\n";

   ScalingLayer scaling_layer;

   Tensor<Descriptives, 1> descriptives;

   // Test 0

   assert_true(scaling_layer.get_neurons_number() == 0, LOG);
   assert_true(scaling_layer.get_neurons_number() == scaling_layer.get_inputs_number(), LOG);

   // Test 1

   scaling_layer.set(3);

   descriptives.resize(3);
   scaling_layer.set_descriptives(descriptives);

   assert_true(scaling_layer.get_neurons_number() == 3, LOG);
   assert_true(scaling_layer.get_neurons_number() == scaling_layer.get_inputs_number(), LOG);
}


void ScalingLayerTest::test_get_descriptives()
{
   cout << "test_get_descriptives\n";

   ScalingLayer scaling_layer;

   Tensor<Descriptives, 1> descriptives;

   // Test 0

   descriptives.resize(1);

   scaling_layer.set_descriptives(descriptives);

   descriptives = scaling_layer.get_descriptives();

   assert_true(abs(descriptives(0).minimum + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(descriptives(0).maximum - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(descriptives(0).mean - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(descriptives(0).standard_deviation - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   Descriptives descriptives_1(1,1,1,0);
   Descriptives descriptives_2(2,2,2,0);

   descriptives.resize(2);
   descriptives.setValues({descriptives_1, descriptives_2});

   scaling_layer.set_descriptives(descriptives);

   descriptives = scaling_layer.get_descriptives();

   assert_true(abs(descriptives(1).minimum - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(descriptives(1).maximum - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(descriptives(1).mean - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(descriptives(1).standard_deviation) - 0 < static_cast<type>(1e-3), LOG);

   // Test 2

   Tensor<Descriptives, 1> descriptives1(1);
   ScalingLayer sl1(descriptives1);

   Descriptives get_des_2;
   get_des_2 = sl1.get_descriptives(0);

   assert_true(abs(get_des_2.minimum + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(get_des_2.maximum - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(get_des_2.mean - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(get_des_2.standard_deviation - 1) < static_cast<type>(1e-3), LOG);

   // Test 3

   Descriptives get_des_3;
   get_des_3 = scaling_layer.get_descriptives(0);

   assert_true(abs(get_des_3.minimum - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(get_des_3.maximum - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(get_des_3.mean - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(get_des_3.standard_deviation - 0) < static_cast<type>(1e-3), LOG);

}


void ScalingLayerTest::test_get_descriptives_matrix()
{
   cout << "test_get_descriptives_matrix\n";

   // Test 0

   Tensor<Descriptives, 1> descriptives(1);

   ScalingLayer scaling_layer(descriptives);

   assert_true(scaling_layer.get_descriptives_matrix().size() == 4, LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives.resize(2);

   descriptives(0).minimum = 1;
   descriptives(0).maximum = 1;
   descriptives(0).mean = 1;
   descriptives(0).standard_deviation = 0;

   descriptives(1).minimum = 2;
   descriptives(1).maximum = 2;
   descriptives(1).mean = 2;
   descriptives(1).standard_deviation = 0;

   scaling_layer.set_descriptives(descriptives);

   assert_true(scaling_layer.get_descriptives_matrix().size() == 8, LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 0) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_get_minimums()
{
   cout << "test_get_minimums\n";

   ScalingLayer scaling_layer(2);

   // Test 0

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_minimums()(0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_minimums()(1) + 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives(0).minimum = 1;
   descriptives(1).minimum = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_minimums()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_minimums()(1) + 1) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_get_maximums()
{
   cout << "test_get_maximums\n";

   ScalingLayer scaling_layer(2);

   // Test 0

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_maximums()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_maximums()(1) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives(0).maximum = 1;
   descriptives(1).maximum = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_maximums()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_maximums()(1) + 1) < static_cast<type>(1e-3), LOG);
}

void ScalingLayerTest::test_get_means()
{
   cout << "test_get_means\n";

   ScalingLayer scaling_layer(2);

   // Test 0

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_means()(0) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_means()(1) - 0) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives(0).mean = 1;
   descriptives(1).mean = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_means()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_means()(1) + 1) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_get_standard_deviations()
{
   cout << "test_get_standard_deviations\n";

   ScalingLayer scaling_layer(2);

   // Test 0

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_standard_deviations()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_standard_deviations()(1) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives(0).standard_deviation = 1;
   descriptives(1).standard_deviation = -1;

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_standard_deviations()(0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_standard_deviations()(1) + 1) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_get_scaling_method()
{
   cout << "test_get_scaling_method\n";

   ScalingLayer scaling_layer(1);

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

   ScalingLayer scaling_layer(1);

   // Test 1

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

   // Test 2

   scaling_layer.set_scalers(no_scaling);
   assert_true(scaling_layer.write_scalers_text()(0) == "no scaling", LOG);

   scaling_layer.set_scalers(minimum_maximum);
   assert_true(scaling_layer.write_scalers_text()(0) == "minimum and maximum", LOG);

   scaling_layer.set_scalers(mean_standard_deviation);
   assert_true(scaling_layer.write_scalers_text()(0) == "mean and standard deviation", LOG);

   scaling_layer.set_scalers(standard_deviation);
   assert_true(scaling_layer.write_scalers_text()(0) == "standard deviation", LOG);
}


void ScalingLayerTest::test_get_display()
{
   cout << "test_get_display\n";

   ScalingLayer scaling_layer;

   assert_true(scaling_layer.get_display(), LOG);
}


void ScalingLayerTest::test_set()
{
   cout << "test_set\n";

   ScalingLayer scaling_layer;

   // Test 1

   scaling_layer.set();

   assert_true(scaling_layer.get_descriptives().size() == 0, LOG);

   Tensor<Descriptives, 1> descriptives(4);
   scaling_layer.set_descriptives(descriptives);
   scaling_layer.set();

   assert_true(scaling_layer.get_descriptives().size() == 0, LOG);

   // Test 2

   Index new_inputs_number_ = 4;
   scaling_layer.set(new_inputs_number_);

   assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);

   // Test 3

   scaling_layer.set();

   Tensor<Index, 1> new_inputs_dimensions(1);
   new_inputs_dimensions.setConstant(3);
   scaling_layer.set(new_inputs_dimensions);

   assert_true(scaling_layer.get_descriptives().size()== 3, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 3, LOG);

   // Test 4

   scaling_layer.set();

   Tensor<Descriptives, 1> descriptives_4;
   scaling_layer.set(descriptives_4);

   assert_true(scaling_layer.get_descriptives().size()== 0, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 0, LOG);

   Tensor<Descriptives, 1> descriptives_4_(4);
   scaling_layer.set(descriptives_4_);

   assert_true(scaling_layer.get_descriptives().size()== 4, LOG);
   assert_true(scaling_layer.get_scaling_methods().size()== 4, LOG);
}


void ScalingLayerTest::test_set_inputs_number()
{
   cout << "test_set_inputs_number\n";

   ScalingLayer scaling_layer;

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

   ScalingLayer scaling_layer;

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

   ScalingLayer scaling_layer(1);

   scaling_layer.set_default();

   Tensor<Descriptives, 1> sl_descriptives = scaling_layer.get_descriptives();

   assert_true(scaling_layer.get_scaling_methods()(0) == MinimumMaximum, LOG);
   assert_true(scaling_layer.get_display(), LOG);
   assert_true(scaling_layer.get_type() == Layer::Scaling, LOG);
   assert_true(scaling_layer.get_type() == 0, LOG);
   assert_true(abs(sl_descriptives(0).minimum + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(sl_descriptives(0).maximum - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(sl_descriptives(0).mean - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(sl_descriptives(0).standard_deviation - 1) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_set_descriptives()
{
   cout << "test_set_descriptives\n";

   ScalingLayer scaling_layer;

   Tensor<Descriptives, 1> descriptives;

   Descriptives item_0(1,1,1,0);
   Descriptives item_1(2,2,2,0);

   // Test 0

   descriptives.resize(1);

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   descriptives.resize(2);
   descriptives.setValues({item_0, item_1});

   scaling_layer.set_descriptives(descriptives);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 0) < static_cast<type>(1e-3), LOG);

}


void ScalingLayerTest::test_set_descriptives_eigen()
{
   cout << "test_set_descriptives_eigen\n";

   ScalingLayer scaling_layer(1);

   // Test 0

   Tensor<type, 2> descriptives_eigen(1,4);
   descriptives_eigen.setValues({{-1,1,0,1}});
   scaling_layer.set_descriptives_eigen(descriptives_eigen);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   ScalingLayer sl_(2);

   Tensor<type, 2> descriptives_eigen_(2,4);
   descriptives_eigen_.setValues({{1,1,1,0},{2,2,2,0}});

   sl_.set_descriptives_eigen(descriptives_eigen_);

   assert_true(abs(sl_.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(sl_.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(sl_.get_descriptives_matrix()(1,1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(sl_.get_descriptives_matrix()(1,3) - 0) < static_cast<type>(1e-3), LOG);
}

void ScalingLayerTest::test_set_item_descriptives()
{
   cout << "test_set_item_descriptives\n";

   ScalingLayer scaling_layer;

   Descriptives item_descriptives_1;
   Descriptives item_descriptives_2;

   // Test 0

   scaling_layer.set(1);

   scaling_layer.set_item_descriptives(0, item_descriptives_1);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 0) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 1) < static_cast<type>(1e-3), LOG);

   // Test 1

   scaling_layer.set(2);

//   item_descriptives_1.set(1,1,1,0);
//   item_descriptives_2.set(2,2,2,0);

   scaling_layer.set_item_descriptives(0, item_descriptives_1);
   scaling_layer.set_item_descriptives(1, item_descriptives_1);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 1) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 2) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 0) < static_cast<type>(1e-3), LOG);

}

void ScalingLayerTest::test_set_minimum()
{
   cout << "test_set_minimum\n";

   ScalingLayer scaling_layer;
   Tensor<Descriptives, 1> descriptives;

   // Test 1

   scaling_layer.set(2);
   descriptives.resize(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_minimum(0, -5);
   scaling_layer.set_minimum(1, -6);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,0) + 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,0) + 6) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_set_maximum()
{
   cout << "test_set_maximum\n";

   ScalingLayer scaling_layer(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_maximum(0, 5);
   scaling_layer.set_maximum(1, 6);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,1) - 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,1) - 6) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_set_mean()
{
   cout << "test_set_mean\n";

   ScalingLayer scaling_layer(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_mean(0, 5);
   scaling_layer.set_mean(1, 6);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,2) - 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,2) - 6) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_set_standard_deviation()
{
   cout << "test_set_standard_deviation\n";

   ScalingLayer scaling_layer(2);

   // Test 1

   Tensor<Descriptives, 1> descriptives(2);

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.set_standard_deviation(0, 5);
   scaling_layer.set_standard_deviation(1, 6);

   assert_true(abs(scaling_layer.get_descriptives_matrix()(0,3) - 5) < static_cast<type>(1e-3), LOG);
   assert_true(abs(scaling_layer.get_descriptives_matrix()(1,3) - 6) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_set_scaling_method()
{
   cout << "test_set_scaling_method\n";

    ScalingLayer scaling_layer(4);

    // Test 1

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

    // Test 2

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

    // Test 3

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

    // Test 4

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


void ScalingLayerTest::test_set_display()
{
   cout << "test_set_display\n";

   bool display_true = true;
   bool display_false = false;

   set_display(display_true);
   assert_true(get_display(), LOG);

   set_display(display_false);
   assert_true(!get_display(), LOG);
}


void ScalingLayerTest::test_is_empty()
{
   cout << "test_is_empty\n";

   ScalingLayer scaling_layer;

   ScalingLayer sl1(1);

   assert_true(scaling_layer.is_empty(), LOG);
   assert_true(!sl1.is_empty(), LOG);
}


void ScalingLayerTest::test_check_range()
{
   cout << "test_check_range\n";

   ScalingLayer scaling_layer;
   Tensor<type, 1> inputs;

   // Test 0

   scaling_layer.set(1);

   inputs.resize(1);
   inputs.setConstant(0.0);
   scaling_layer.check_range(inputs);

   // Test 1

   Tensor<Descriptives, 1> descriptives(1);
   Descriptives des(-1,1,1,0);
   descriptives.setValues({des});

   scaling_layer.set_descriptives(descriptives);

   scaling_layer.check_range(inputs);
}


void ScalingLayerTest::test_calculate_outputs()
{
   cout << "test_calculate_outputs\n";

   ScalingLayer scaling_layer;

   Tensor<type, 2> inputs;

   scaling_layer.set_display(false);

   // Test 0_0

   scaling_layer.set(1);
   scaling_layer.set_scalers(NoScaling);

   inputs.resize(1,1);
   Tensor<type, 2> outputs = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 1, LOG);
   assert_true(abs(outputs(0) - inputs(0)) < static_cast<type>(1e-3), LOG);

   // Test 0_1

   scaling_layer.set(3);
   scaling_layer.set_scalers(NoScaling);

   inputs.resize(1,3);
   inputs.setConstant(0);
   outputs = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs.dimension(0) == 1, LOG);
   assert_true(outputs.dimension(1) == 3, LOG);
   assert_true(abs(outputs(0)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(1)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs(2)) < static_cast<type>(1e-3), LOG);

   // Test 1_0

   scaling_layer.set(1);
   scaling_layer.set_scalers(MinimumMaximum);

   inputs.resize(1,1);
   Tensor<type, 2> outputs_1 = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs_1.dimension(0) == 1, LOG);
   assert_true(outputs_1.dimension(1) == 1, LOG);

   assert_true(abs(outputs_1(0) - inputs(0)) < static_cast<type>(1e-3), LOG);

   // Test 1_1

   scaling_layer.set(3);
   scaling_layer.set_scalers(MinimumMaximum);

   Tensor<type, 2> minimums_maximums(3, 4);
   minimums_maximums.setValues({{-1,2,0,0},{-2,4,0,0},{-3,6,0,0}});

   scaling_layer.set_descriptives_eigen(minimums_maximums);
   inputs.resize(1,3);
   inputs.setConstant(0);
   outputs_1 = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs_1.dimension(0) == 1, LOG);
   assert_true(outputs_1.dimension(1) == 3, LOG);
   assert_true(abs(outputs_1(0) + static_cast<type>(0.333)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs_1(1) + static_cast<type>(0.333)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs_1(2) + static_cast<type>(0.333)) < static_cast<type>(1e-3), LOG);

   // Test 2_0

   scaling_layer.set(1);
   scaling_layer.set_scalers(MeanStandardDeviation);

   inputs.resize(1,1);
   Tensor<type, 2> outputs_2 = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs_2.dimension(0) == 1, LOG);
   assert_true(outputs_2.dimension(1) == 1, LOG);
   assert_true(abs(outputs_2(0) - inputs(0)) < static_cast<type>(1e-3), LOG);

   // Test 2_1

   scaling_layer.set(2);
   scaling_layer.set_scalers(MeanStandardDeviation);

   Tensor<type, 2> mean_standard_deviation(2,4);
   mean_standard_deviation.setValues({{-1,1,-1,2},{-1,1,1,4}});

   scaling_layer.set_descriptives_eigen(mean_standard_deviation);
   inputs.resize(1,2);
   inputs.setConstant(0);
   outputs_2 = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs_2.dimension(0) == 1, LOG);
   assert_true(outputs_2.dimension(1) == 2, LOG);

   assert_true(abs(outputs_2(0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs_2(1) + static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);

   // Test 3_0

   scaling_layer.set(1);
   scaling_layer.set_scalers(StandardDeviation);

   inputs.resize(1,1);
   Tensor<type, 2> outputs_3 = scaling_layer.calculate_outputs(inputs);
   assert_true(outputs_3.dimension(0) == 1, LOG);
   assert_true(outputs_3.dimension(1) == 1, LOG);
   assert_true(abs(outputs_3(0) - inputs(0)) < static_cast<type>(1e-3), LOG);

   // Test 3_1

   scaling_layer.set(2);
   scaling_layer.set_scalers(StandardDeviation);

   Tensor<type, 2> standard_deviation(2,4);
   standard_deviation.setValues({{-1,1,-1,2},{-1,1,1,4}});

   scaling_layer.set_descriptives_eigen(standard_deviation);
   inputs.resize(1,2);
   inputs.setConstant(1);
   outputs_3 = scaling_layer.calculate_outputs(inputs);

   assert_true(outputs_3.dimension(0) == 1, LOG);
   assert_true(outputs_3.dimension(1) == 2, LOG);
   assert_true(abs(outputs_3(0) - static_cast<type>(0.5)) < static_cast<type>(1e-3), LOG);
   assert_true(abs(outputs_3(1) - static_cast<type>(0.25)) < static_cast<type>(1e-3), LOG);
}


void ScalingLayerTest::test_write_expression()
{
   cout << "test_write_expression\n";

   ScalingLayer scaling_layer;

   Tensor<string, 1> inputs_names(1);
   Tensor<string, 1> outputs_names(1);

   string expression;

   // Test 0_1

   scaling_layer.set(1);
   scaling_layer.set_scalers(NoScaling);
   inputs_names.setValues({"x"});
   outputs_names.setValues({"y"});

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = x;\n", LOG);

   // Test 0_2

   scaling_layer.set(1);
   scaling_layer.set_scalers(MinimumMaximum);

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = x*(1+1)/(1-(-1))+1*(1+1)/(1+1)-1;\n", LOG);

   // Test 0_3

   scaling_layer.set(1);
   scaling_layer.set_scalers(MeanStandardDeviation);

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = (x-(0))/1;\n", LOG);

   // Test 0_4

   scaling_layer.set(1);
   scaling_layer.set_scalers(StandardDeviation);

   expression = scaling_layer.write_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "scaled_x = x/(1);\n", LOG);

   // Test 1

   scaling_layer.set(1);
   scaling_layer.set_scalers(NoScaling);
   inputs_names.setValues({"x"});
   outputs_names.setValues({"y"});

   expression = scaling_layer.write_no_scaling_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = x;\n", LOG);

   // Test 2

   scaling_layer.set(1);
   scaling_layer.set_scalers(MinimumMaximum);

   expression = scaling_layer.write_minimum_maximum_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = 2*(x-(-1))/(1-(-1))-1;\n", LOG);

   // Test 3

   scaling_layer.set(1);
   scaling_layer.set_scalers(MeanStandardDeviation);

   expression = scaling_layer.write_mean_standard_deviation_expression(inputs_names, outputs_names);

   assert_true(!expression.empty(), LOG);
   assert_true(expression == "y = (x-(0))/1;\n", LOG);

   // Test 4

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
   test_destructor();


   // Scaling layer architecture

   test_get_inputs_number();
   test_get_neurons_number();


   // Input variables descriptives

   test_get_descriptives();
   test_get_descriptives_matrix();
   test_get_minimums();
   test_get_maximums();
   test_get_means();
   test_get_standard_deviations();


   // Variables scaling and unscaling

   test_get_scaling_method();
   test_write_scalers();


   // Display messages

   test_get_display();


   // Set methods

   test_set();
   test_set_inputs_number();
   test_set_neurons_number();
   test_set_default();


   // Input variables descriptives

   test_set_descriptives();
   test_set_descriptives_eigen();
   test_set_item_descriptives();
   test_set_minimum();
   test_set_maximum();
   test_set_mean();
   test_set_standard_deviation();


   // Variables scaling and unscaling

   test_set_scaling_method();


   // Display messages

   test_set_display();


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
