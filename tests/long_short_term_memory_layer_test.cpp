#include "pch.h"

#include "../opennn/long_short_term_memory_layer.h"



TEST(LongShortTermMemoryLayerTest, DefaultConstructor)
{
    LongShortTermMemoryLayer long_short_term_memory_layer;

    EXPECT_EQ(long_short_term_memory_layer.get_input_dimensions(), dimensions{0});
//    EXPECT_EQ(long_short_term_memory_layer.get_output_dimensions(), dimensions{0});
}


TEST(LongShortTermMemoryLayerTest, GeneralConstructor)
{

//    LongShortTermMemoryLayer long_short_term_memory_layer(inputs_number, neurons_number, time_steps);

//    EXPECT_EQ(long_short_term_memory_layer.get_parameters_number(), 12);
}

/*

void LongShortTermMemoryLayerTest::test_forward_propagate()
{
    Index samples_number;
    Index inputs_number;
    Index neurons_number;

    pair<type*, dimensions> input_pairs;

    Tensor<type, 1> parameters;
    Tensor<type, 2> inputs;
    bool is_training = true;

    // Test

    samples_number = 1;
    inputs_number = 1;
    neurons_number = 1;
    time_steps = 1;

    inputs.resize(samples_number, inputs_number);
    inputs.setConstant(type(1));

    long_short_term_memory_layer.set(inputs_number, neurons_number, time_steps);
    long_short_term_memory_layer.set_activation_function(LongShortTermMemoryLayer::Activation::HyperbolicTangent);
    long_short_term_memory_layer.set_parameters_constant(type(1));

    long_short_term_layer_forward_propagation.set(samples_number, &long_short_term_memory_layer);

    input_pairs = {inputs.data(), {{samples_number, inputs_number}}};

    long_short_term_memory_layer.forward_propagate({input_pairs}, &long_short_term_layer_forward_propagation, is_training);

    EXPECT_EQ(long_short_term_layer_forward_propagation.outputs.rank() == 2);
    EXPECT_EQ(long_short_term_layer_forward_propagation.outputs.dimension(0) == 1);
    EXPECT_EQ(long_short_term_layer_forward_propagation.outputs.dimension(1) == inputs.dimension(1));

}

}
*/
