#include "pch.h"

#include "../opennn/statistics.h"
#include "../opennn/tensors.h"
#include "../opennn/unscaling_layer.h"
#include "../opennn/scaling_layer_2d.h"

using namespace opennn;

TEST(UnscalingTest, DefaultConstructor)
{
    Unscaling unscaling_layer;

    EXPECT_EQ(unscaling_layer.get_name(), "Unscaling");
    EXPECT_EQ(unscaling_layer.get_descriptives().size(), 0);
    EXPECT_EQ(unscaling_layer.get_input_dimensions(), dimensions{ 0 });
    EXPECT_EQ(unscaling_layer.get_output_dimensions(), dimensions{ 0 });
}

TEST(UnscalingTest, GeneralConstructor)
{
    Unscaling unscaling_layer({ 3 });

    EXPECT_EQ(unscaling_layer.get_input_dimensions(), dimensions{ 3 });
    EXPECT_EQ(unscaling_layer.get_output_dimensions(), dimensions{ 3 });
    EXPECT_EQ(unscaling_layer.get_name(), "Unscaling");
    EXPECT_EQ(unscaling_layer.get_descriptives().size(), 3);
}

TEST(UnscalingTest, ForwardPropagate)
{
    Index inputs_number;
    Index samples_number;
    bool is_training = true;

    // Test Unscaling Scaler::None
    {
        inputs_number = 3;
        samples_number = 1;

        Unscaling unscaling_layer_none({ inputs_number });
        unscaling_layer_none.set_scalers(Scaler::None);

        vector<Descriptives> none_descriptives(inputs_number);
        for (Index i = 0; i < inputs_number; ++i) {
            none_descriptives[i].set(type(0), type(1), type(0.5), type(1.0));
        }
        unscaling_layer_none.set_descriptives(none_descriptives);

        Tensor<type, 2> data_to_unscale(samples_number, inputs_number);
        data_to_unscale.setConstant(type(10));

        unique_ptr<LayerForwardPropagation> fw_prop_unscale =
            make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_none);
        TensorView input_pair_unscale = { data_to_unscale.data(), {{samples_number, inputs_number}} };

        unscaling_layer_none.forward_propagate({ input_pair_unscale }, fw_prop_unscale, is_training);

        TensorView output_pair_unscale = fw_prop_unscale->get_output_pair();
        Tensor<type, 2> unscaled_data = tensor_map<2>(output_pair_unscale);

        EXPECT_EQ(unscaled_data.dimension(0), samples_number);
        EXPECT_EQ(unscaled_data.dimension(1), inputs_number);
        EXPECT_NEAR(unscaled_data(0, 0), data_to_unscale(0, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(0, 1), data_to_unscale(0, 1), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(0, 2), data_to_unscale(0, 2), NUMERIC_LIMITS_MIN);
    }

    // Test Unscaling Scaler::MinimumMaximum
    {
        inputs_number = 1;
        samples_number = 3;

        Scaling2d helper_scaling_layer({ inputs_number });
        Unscaling unscaling_layer_minmax({ inputs_number });

        Tensor<type, 2> original_data(samples_number, inputs_number);
        original_data.setValues({ {type(2)}, {type(4)}, {type(6)} });

        vector<Descriptives> actual_descriptives = descriptives(original_data);

        helper_scaling_layer.set_descriptives(actual_descriptives);
        helper_scaling_layer.set_scalers(Scaler::MinimumMaximum);

        unique_ptr<LayerForwardPropagation> fw_prop_scale =
            make_unique<Scaling2dForwardPropagation>(samples_number, &helper_scaling_layer);
        TensorView input_pair_scale = { original_data.data(), {{samples_number, inputs_number}} };
        helper_scaling_layer.forward_propagate({ input_pair_scale }, fw_prop_scale, is_training);
        TensorView output_pair_scale = fw_prop_scale->get_output_pair();
        Tensor<type, 2> scaled_data = tensor_map<2>(output_pair_scale);

        // (2-2)/(6-2) = 0/4 = 0
        // (4-2)/(6-2) = 2/4 = 0.5
        // (6-2)/(6-2) = 4/4 = 1
        EXPECT_NEAR(scaled_data(0, 0), type(0.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(1, 0), type(0.5), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(2, 0), type(1.0), NUMERIC_LIMITS_MIN);

        unscaling_layer_minmax.set_descriptives(actual_descriptives);
        unscaling_layer_minmax.set_scalers(Scaler::MinimumMaximum);

        unique_ptr<LayerForwardPropagation> fw_prop_unscale =
            make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_minmax);
        TensorView input_pair_unscale = { scaled_data.data(), {{samples_number, inputs_number}} };
        unscaling_layer_minmax.forward_propagate({ input_pair_unscale }, fw_prop_unscale, is_training);
        TensorView output_pair_unscale = fw_prop_unscale->get_output_pair();
        Tensor<type, 2> unscaled_data = tensor_map<2>(output_pair_unscale);

        EXPECT_EQ(unscaled_data.dimension(0), samples_number);
        EXPECT_EQ(unscaled_data.dimension(1), inputs_number);
        EXPECT_NEAR(unscaled_data(0, 0), original_data(0, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 0), original_data(1, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(2, 0), original_data(2, 0), NUMERIC_LIMITS_MIN);
    }

    // Test Unscaling Scaler::MeanStandardDeviation
    {
        inputs_number = 2;
        samples_number = 2;

        Scaling2d helper_scaling_layer({ inputs_number });
        Unscaling unscaling_layer_msd({ inputs_number });

        Tensor<type, 2> original_data(samples_number, inputs_number);
        original_data.setValues({ {type(0), type(10)},
                                  {type(2), type(30)} });

        vector<Descriptives> actual_descriptives = descriptives(original_data);
        // Col 0: {0,2} -> mean=1, std=sqrt(2)
        // Col 1: {10,30} -> mean=20, std=sqrt(200)

        helper_scaling_layer.set_descriptives(actual_descriptives);
        helper_scaling_layer.set_scalers(Scaler::MeanStandardDeviation);

        unique_ptr<LayerForwardPropagation> fw_prop_scale =
            make_unique<Scaling2dForwardPropagation>(samples_number, &helper_scaling_layer);
        TensorView input_pair_scale = { original_data.data(), {{samples_number, inputs_number}} };
        helper_scaling_layer.forward_propagate({ input_pair_scale }, fw_prop_scale, is_training);
        TensorView output_pair_scale = fw_prop_scale->get_output_pair();
        Tensor<type, 2> scaled_data = tensor_map<2>(output_pair_scale);

        unscaling_layer_msd.set_descriptives(actual_descriptives);
        unscaling_layer_msd.set_scalers(Scaler::MeanStandardDeviation);

        unique_ptr<LayerForwardPropagation> fw_prop_unscale =
            make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_msd);
        TensorView input_pair_unscale = { scaled_data.data(), {{samples_number, inputs_number}} };
        unscaling_layer_msd.forward_propagate({ input_pair_unscale }, fw_prop_unscale, is_training);
        TensorView output_pair_unscale = fw_prop_unscale->get_output_pair();
        Tensor<type, 2> unscaled_data = tensor_map<2>(output_pair_unscale);

        EXPECT_EQ(unscaled_data.dimension(0), samples_number);
        EXPECT_EQ(unscaled_data.dimension(1), inputs_number);
        EXPECT_NEAR(unscaled_data(0, 0), original_data(0, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(0, 1), original_data(0, 1), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 0), original_data(1, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 1), original_data(1, 1), NUMERIC_LIMITS_MIN);
    }

    // Test Unscaling Scaler::StandardDeviation
    {
        inputs_number = 2;
        samples_number = 2;

        Scaling2d helper_scaling_layer({ inputs_number });
        Unscaling unscaling_layer_std({ inputs_number });

        Tensor<type, 2> original_data(samples_number, inputs_number);
        original_data.setValues({ {type(1),type(10)},
                                  {type(3),type(50)} });
        // Col 0: {1,3} -> std=sqrt( ((1-2)^2 + (3-2)^2) / 1 ) = sqrt(1+1) = sqrt(2)
        // Col 1: {10,50} -> std=sqrt( ((10-30)^2 + (50-30)^2) / 1 ) = sqrt(400+400) = sqrt(800)

        vector<Descriptives> actual_descriptives = descriptives(original_data);

        helper_scaling_layer.set_descriptives(actual_descriptives);
        helper_scaling_layer.set_scalers(Scaler::StandardDeviation);

        unique_ptr<LayerForwardPropagation> fw_prop_scale =
            make_unique<Scaling2dForwardPropagation>(samples_number, &helper_scaling_layer);
        TensorView input_pair_scale = { original_data.data(), {{samples_number, inputs_number}} };
        helper_scaling_layer.forward_propagate({ input_pair_scale }, fw_prop_scale, is_training);
        TensorView output_pair_scale = fw_prop_scale->get_output_pair();
        Tensor<type, 2> scaled_data = tensor_map<2>(output_pair_scale);

        unscaling_layer_std.set_descriptives(actual_descriptives);
        unscaling_layer_std.set_scalers(Scaler::StandardDeviation);

        unique_ptr<LayerForwardPropagation> fw_prop_unscale =
            make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_std);
        TensorView input_pair_unscale = { scaled_data.data(), {{samples_number, inputs_number}} };
        unscaling_layer_std.forward_propagate({ input_pair_unscale }, fw_prop_unscale, is_training);
        TensorView output_pair_unscale = fw_prop_unscale->get_output_pair();
        Tensor<type, 2> unscaled_data = tensor_map<2>(output_pair_unscale);

        EXPECT_EQ(unscaled_data.dimension(0), samples_number);
        EXPECT_EQ(unscaled_data.dimension(1), inputs_number);
        EXPECT_NEAR(unscaled_data(0, 0), original_data(0, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(0, 1), original_data(0, 1), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 0), original_data(1, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 1), original_data(1, 1), NUMERIC_LIMITS_MIN * 10);
    }

    // Test Unscaling Scaler::Logarithm
    {
        inputs_number = 2;
        samples_number = 2;

        Scaling2d helper_scaling_layer({ inputs_number });
        Unscaling unscaling_layer_log({ inputs_number });

        Tensor<type, 2> original_data(samples_number, inputs_number);

        original_data.setValues({ {type(1.0),      type(exp(2.0))},     // log(1)=0, log(exp(2))=2
                                  {type(exp(1.0)), type(exp(3.0))} });  // log(exp(1))=1, log(exp(3))=3

        helper_scaling_layer.set_scalers(Scaler::Logarithm);

        unique_ptr<LayerForwardPropagation> fw_prop_scale =
            make_unique<Scaling2dForwardPropagation>(samples_number, &helper_scaling_layer);
        TensorView input_pair_scale = { original_data.data(), {{samples_number, inputs_number}} };
        helper_scaling_layer.forward_propagate({ input_pair_scale }, fw_prop_scale, is_training);
        TensorView output_pair_scale = fw_prop_scale->get_output_pair();
        Tensor<type, 2> scaled_data = tensor_map<2>(output_pair_scale);

        EXPECT_NEAR(scaled_data(0, 0), type(0.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(0, 1), type(2.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(1, 0), type(1.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(1, 1), type(3.0), NUMERIC_LIMITS_MIN);

        unscaling_layer_log.set_scalers(Scaler::Logarithm);

        unique_ptr<LayerForwardPropagation> fw_prop_unscale =
            make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_log);
        TensorView input_pair_unscale = { scaled_data.data(), {{samples_number, inputs_number}} };
        unscaling_layer_log.forward_propagate({ input_pair_unscale }, fw_prop_unscale, is_training);
        TensorView output_pair_unscale = fw_prop_unscale->get_output_pair();
        Tensor<type, 2> unscaled_data = tensor_map<2>(output_pair_unscale);

        EXPECT_EQ(unscaled_data.dimension(0), samples_number);
        EXPECT_EQ(unscaled_data.dimension(1), inputs_number);
        EXPECT_NEAR(unscaled_data(0, 0), original_data(0, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(0, 1), original_data(0, 1), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 0), original_data(1, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 1), original_data(1, 1), NUMERIC_LIMITS_MIN);
    }

    // Test Unscaling Scaler::ImageMinMax
    {
        inputs_number = 2;
        samples_number = 2;

        Scaling2d helper_scaling_layer({ inputs_number });
        Unscaling unscaling_layer_img({ inputs_number });

        Tensor<type, 2> original_data(samples_number, inputs_number);
        original_data.setValues({ {type(0),     type(255)},
                                  {type(127.5), type(51)} });

        helper_scaling_layer.set_scalers(Scaler::ImageMinMax);

        unique_ptr<LayerForwardPropagation> fw_prop_scale =
            make_unique<Scaling2dForwardPropagation>(samples_number, &helper_scaling_layer);
        TensorView input_pair_scale = { original_data.data(), {{samples_number, inputs_number}} };
        helper_scaling_layer.forward_propagate({ input_pair_scale }, fw_prop_scale, is_training);
        TensorView output_pair_scale = fw_prop_scale->get_output_pair();
        Tensor<type, 2> scaled_data = tensor_map<2>(output_pair_scale);

        EXPECT_NEAR(scaled_data(0, 0), type(0.0 / 255.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(0, 1), type(255.0 / 255.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(1, 0), type(127.5 / 255.0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(scaled_data(1, 1), type(51.0 / 255.0), NUMERIC_LIMITS_MIN);

        unscaling_layer_img.set_scalers(Scaler::ImageMinMax);

        unique_ptr<LayerForwardPropagation> fw_prop_unscale =
            make_unique<UnscalingForwardPropagation>(samples_number, &unscaling_layer_img);
        TensorView input_pair_unscale = { scaled_data.data(), {{samples_number, inputs_number}} };
        unscaling_layer_img.forward_propagate({ input_pair_unscale }, fw_prop_unscale, is_training);
        TensorView output_pair_unscale = fw_prop_unscale->get_output_pair();
        Tensor<type, 2> unscaled_data = tensor_map<2>(output_pair_unscale);

        EXPECT_EQ(unscaled_data.dimension(0), samples_number);
        EXPECT_EQ(unscaled_data.dimension(1), inputs_number);

        EXPECT_NEAR(unscaled_data(0, 0), original_data(0, 0), NUMERIC_LIMITS_MIN); 
        EXPECT_NEAR(unscaled_data(0, 1), original_data(0, 1), NUMERIC_LIMITS_MIN); 
        EXPECT_NEAR(unscaled_data(1, 0), original_data(1, 0), NUMERIC_LIMITS_MIN);
        EXPECT_NEAR(unscaled_data(1, 1), original_data(1, 1), NUMERIC_LIMITS_MIN * 10);
    }
}
