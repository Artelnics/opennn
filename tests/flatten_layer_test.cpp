#include "pch.h"

#include "../opennn/flatten_layer.h"

class FlattenLayerTest : public ::testing::Test 
{
public:
    const Index batch_samples_number = 2;
    const Index height = 4;
    const Index width = 4;
    const Index channels = 3;

    const bool is_training = true;

    const dimensions input_dimensions = { height, width, channels };

    unique_ptr<Layer> flatten_layer;

    unique_ptr<LayerForwardPropagation> flatten_layer_forward_propagation;
    unique_ptr<LayerBackPropagation> flatten_layer_back_propagation;

    Tensor<type, 4> inputs;
    pair<type*, dimensions> input_pair;
    pair<type*, dimensions> output_pair;
    vector<pair<type*, dimensions>> input_derivatives_pair;

    void SetUp() override 
    {
        flatten_layer = make_unique<FlattenLayer>(input_dimensions);
        flatten_layer_forward_propagation = make_unique<FlattenLayerForwardPropagation>(batch_samples_number, flatten_layer.get());
        flatten_layer_back_propagation = make_unique<FlattenLayerBackPropagation>(batch_samples_number, flatten_layer.get());

        inputs.resize(batch_samples_number, height, width, channels);
        inputs.setRandom();
        input_pair = { inputs.data(), { batch_samples_number, height, width, channels } };
    }
};


TEST_F(FlattenLayerTest, Constructor) 
{
    EXPECT_EQ(flatten_layer->get_input_dimensions(), input_dimensions);
    EXPECT_EQ(flatten_layer->get_type(), Layer::Type::Flatten);
    EXPECT_EQ(flatten_layer->get_name(), "flatten_layer");
}


TEST_F(FlattenLayerTest, ForwardPropagate) 
{
    flatten_layer->forward_propagate({ input_pair }, flatten_layer_forward_propagation, true);

    output_pair = flatten_layer_forward_propagation->get_outputs_pair();

    EXPECT_EQ(output_pair.second[0], batch_samples_number);
    EXPECT_EQ(output_pair.second[1], height * width * channels);
}


TEST_F(FlattenLayerTest, BackPropagate)
{
    output_pair = flatten_layer_forward_propagation->get_outputs_pair();

    flatten_layer->back_propagate({ output_pair }, { output_pair }, flatten_layer_forward_propagation, flatten_layer_back_propagation);

    input_derivatives_pair = flatten_layer_back_propagation.get()->get_input_derivative_pairs();

    EXPECT_EQ(input_derivatives_pair[0].second[0], batch_samples_number);
    EXPECT_EQ(input_derivatives_pair[0].second[1], input_dimensions[0]);
    EXPECT_EQ(input_derivatives_pair[0].second[2], input_dimensions[1]);
    EXPECT_EQ(input_derivatives_pair[0].second[3], input_dimensions[2]);
}
