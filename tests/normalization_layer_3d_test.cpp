#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/normalization_layer_3d.h"
#include "../opennn/neural_network.h"

using namespace opennn;


TEST(Normalization3dTest, DefaultConstructor)
{
    Normalization3d normalization_3d;

    EXPECT_EQ(normalization_3d.get_input_shape(), Shape({0,0}));
    EXPECT_EQ(normalization_3d.get_output_shape(), Shape({0,0}));

}


TEST(Normalization3dTest, GeneralConstructor)
{
    const Index sequence_length = random_integer(1,10);
    const Index embedding_dimension = random_integer(1,10);

    Normalization3d normalization_3d(Shape({sequence_length, embedding_dimension}));

    EXPECT_EQ(normalization_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(normalization_3d.get_embedding_dimension(), embedding_dimension);

}


TEST(Normalization3dTest, ForwardPropagate)
{
    const Index batch_size = random_integer(1,10);
    const Index sequence_length = random_integer(1,10);
    const Index embedding_dimension = random_integer(1,10);
/*
    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Normalization3d>(shape({sequence_length, embedding_dimension})));

    Tensor3 inputs(batch_size, sequence_length, embedding_dimension);
    inputs.setRandom();

//    EXPECT_NEAR(outputs(0), type(-1.0), EPSILON);
//    EXPECT_EQ(bounding_layer.get_output_shape(), Shape{ 1 });
*/
}
