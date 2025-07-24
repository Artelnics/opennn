#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/addition_layer_3d.h"
#include "../opennn/neural_network.h"

using namespace opennn;


TEST(Addition3dTest, DefaultConstructor)
{
    Addition3d<3> addition_3d_layer;

    EXPECT_EQ(addition_3d_layer.get_output_dimensions(), dimensions({0, 0}));
}


TEST(Addition3dTest, GeneralConstructor)
{
    const Index sequence_length = get_random_index(1,10);
    const Index embedding_dimension = get_random_index(1,10);

    Addition3d<3> addition_3d_layer({sequence_length, embedding_dimension});

//    EXPECT_EQ(addition_3d_layer.get_sequence_length(), sequence_length);
//    EXPECT_EQ(addition_3d_layer.get_embedding_dimension(), embedding_dimension);
}


TEST(Addition3dTest, ForwardPropagate)
{
    const Index batch_size = get_random_index(1,10);
    const Index sequence_length = get_random_index(1,10);
    const Index embedding_dimension = get_random_index(1,10);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Addition3d<3>>(dimensions({sequence_length, embedding_dimension})));

    Tensor<type, 3> inputs_1(batch_size, sequence_length, embedding_dimension);
    inputs_1.setConstant(-2.0);

    Tensor<type, 3> inputs_2(batch_size, sequence_length, embedding_dimension);
    inputs_2.setConstant(1.0);

    const Tensor<type, 3> outputs = neural_network.calculate_outputs(inputs_1, inputs_2);

    EXPECT_EQ(outputs.dimension(0), batch_size);
    EXPECT_EQ(outputs.dimension(1), sequence_length);
    EXPECT_EQ(outputs.dimension(2), embedding_dimension);
}
