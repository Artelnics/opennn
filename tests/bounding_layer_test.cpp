#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/bounding_layer.h"
#include "../opennn/neural_network.h"

using namespace opennn;

const type tolerance = 1e-9;

TEST(BoundingTest, Constructor)
{
    Bounding bounding_layer;

    EXPECT_EQ(bounding_layer.get_output_shape(), Shape{0});
}


TEST(BoundingTest, ForwardPropagate)
{
    const Index columns_number = 3;
    const Index rows_number = 2;

    MatrixR input_data(rows_number, columns_number);
    input_data << type(-5.0), type(0.5), type(10.0),
        type(-1.0), type(0.0), type(1.0);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Bounding>(Shape{columns_number}));
    neural_network.compile();

    Bounding* layer = static_cast<Bounding*>(neural_network.get_layer(0).get());
    layer->set_bounding_method(Bounding::BoundingMethod::Bounding);
    for (Index j = 0; j < columns_number; ++j)
    {
        layer->set_lower_bound(j, type(-1.0));
        layer->set_upper_bound(j, type(1.0));
    }

    ForwardPropagation forward_propagation(rows_number, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {rows_number, columns_number}) };
    neural_network.forward_propagate(inputs, forward_propagation, false);

    TensorView output_view = forward_propagation.get_outputs();
    MatrixMap outputs(output_view.data, rows_number, columns_number);

    EXPECT_EQ(outputs.rows(), rows_number);
    EXPECT_EQ(outputs.cols(), columns_number);

    MatrixR expected_output(rows_number, columns_number);
    expected_output << type(-1.0), type(0.5), type(1.0),
        type(-1.0), type(0.0), type(1.0);

    for(Index i = 0; i < rows_number; ++i)
        for(Index j = 0; j < columns_number; ++j)
            EXPECT_NEAR(outputs(i, j), expected_output(i, j), tolerance);

    EXPECT_EQ(layer->get_output_shape(), Shape{ columns_number });
}
