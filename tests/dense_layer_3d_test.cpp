#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/dense_layer_3d.h"
#include "../opennn/neural_network.h"

using namespace opennn;


TEST(Dense3dTest, DefaultConstructor)
{
    Dense3d dense_3d;

    EXPECT_EQ(dense_3d.get_output_dimensions(), dimensions({0, 0}));
}


TEST(Dense3dTest, GeneralConstructor)
{
    const Index sequence_length = get_random_index(1, 10);
    const Index input_embedding = get_random_index(1, 10);
    const Index output_embedding = get_random_index(1, 10);

    Dense3d dense_3d(sequence_length, input_embedding, output_embedding);

    EXPECT_EQ(dense_3d.get_sequence_length(), sequence_length);
    EXPECT_EQ(dense_3d.get_input_embedding(), input_embedding);
    EXPECT_EQ(dense_3d.get_output_embedding(), output_embedding);
}


TEST(Dense3dTest, ForwardPropagate)
{
    /*
    const Index sequence_length = get_random_index(1, 10);
    const Index input_embedding = get_random_index(1, 10);
    const Index output_embedding = get_random_index(1, 10);
    const Index batch_size = get_random_index(1, 10);

    Dense3d dense_3d(sequence_length, input_embedding, output_embedding);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<Dense3dForwardPropagation>(batch_size, &dense_3d);

    Tensor<type, 3> inputs(batch_size, sequence_length, input_embedding);
    inputs.setRandom();

    auto eigen_dimensions = inputs.dimensions();
    dimensions dims_vector(eigen_dimensions.begin(), eigen_dimensions.end());
    TensorView input_view(inputs.data(), dims_vector);
    vector<TensorView> input_views = { input_view };

    dense_3d.forward_propagate(input_views, forward_propagation, false);

    const TensorMap<Tensor<type, 2>> outputs =
        tensor_map<2>(forward_propagation->get_output_pair());

    EXPECT_EQ(outputs.dimension(0), batch_size);
    EXPECT_EQ(outputs.dimension(1), sequence_length);
    EXPECT_EQ(outputs.dimension(2), output_embedding);
    */
}
