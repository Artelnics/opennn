#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/layers/clamping_layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"

using namespace opennn;

const type tolerance = 1e-9;

TEST(ClampingTest, Constructor)
{
    Clamping clamping_layer;

    EXPECT_EQ(clamping_layer.get_output_shape(), Shape{0});
    EXPECT_EQ(clamping_layer.get_clamping_method(), Clamping::ClampingMethod::Clamping);
}

TEST(ClampingTest, GeneralConstructor)
{
    const Index features = 4;

    Clamping clamping_layer(Shape{features},
                            Clamping::ClampingMethod::NoClamping,
                            "my_clamping");

    EXPECT_EQ(clamping_layer.get_name(), "Clamping");
    EXPECT_EQ(clamping_layer.get_label(), "my_clamping");
    EXPECT_EQ(clamping_layer.get_input_shape(), Shape{features});
    EXPECT_EQ(clamping_layer.get_output_shape(), Shape{features});
    EXPECT_EQ(clamping_layer.get_clamping_method(), Clamping::ClampingMethod::NoClamping);
}

TEST(ClampingTest, InputShapeChangeKeepsMethodAndBounds)
{
    const Index features = 3;

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Clamping>(Shape{features},
                                                   Clamping::ClampingMethod::NoClamping,
                                                   "my_clamping"));
    neural_network.compile();

    Clamping* layer = static_cast<Clamping*>(neural_network.get_layer(0).get());

    for (Index j = 0; j < features; ++j)
    {
        layer->set_lower_bound(j, type(-1.0));
        layer->set_upper_bound(j, type(1.0));
    }

    neural_network.set_input_shape(Shape{features});

    EXPECT_EQ(layer->get_label(), "my_clamping");
    EXPECT_EQ(layer->get_clamping_method(), Clamping::ClampingMethod::NoClamping);

    for (Index j = 0; j < features; ++j)
    {
        EXPECT_NEAR(layer->get_lower_bounds()(j), type(-1.0), tolerance);
        EXPECT_NEAR(layer->get_upper_bounds()(j), type(1.0), tolerance);
    }
}

TEST(ClampingTest, RejectsSuccessors)
{
    NeuralNetwork neural_network;
    auto clamping = make_unique<Clamping>(Shape{2});
    auto dense = make_unique<opennn::Dense>(Shape{2}, Shape{1}, "Identity");

    EXPECT_FALSE(clamping->allows_successors());
    EXPECT_TRUE(dense->allows_successors());

    neural_network.add_layer(std::move(clamping));
    EXPECT_THROW(neural_network.add_layer(std::move(dense)), runtime_error);
    EXPECT_EQ(neural_network.get_layers_number(), 1);
}

TEST(ClampingTest, ForwardPropagate)
{
    const Index columns_number = 3;
    const Index rows_number = 2;

    MatrixR input_data(rows_number, columns_number);
    input_data << type(-5.0), type(0.5), type(10.0),
        type(-1.0), type(0.0), type(1.0);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Clamping>(Shape{columns_number}));
    neural_network.compile();

    Clamping* layer = static_cast<Clamping*>(neural_network.get_layer(0).get());
    layer->set_clamping_method(Clamping::ClampingMethod::Clamping);
    for (Index j = 0; j < columns_number; ++j)
    {
        layer->set_lower_bound(j, type(-1.0));
        layer->set_upper_bound(j, type(1.0));
    }

    ForwardPropagation forward_propagation(rows_number, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {rows_number, columns_number}) };
    neural_network.forward_propagate(inputs, forward_propagation, ForwardPropagationMode::Inference);

    TensorView output_view = forward_propagation.get_outputs();
    MatrixMap outputs = output_view.as_matrix();

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

TEST(ClampingTest, NoClampingModePassThrough)
{
    const Index columns_number = 3;
    const Index rows_number = 2;

    MatrixR input_data(rows_number, columns_number);
    input_data << type(-5.0), type(0.5), type(10.0),
        type(-1.0), type(0.0), type(1.0);

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<Clamping>(Shape{columns_number}));
    neural_network.compile();

    Clamping* layer = static_cast<Clamping*>(neural_network.get_layer(0).get());
    layer->set_clamping_method(Clamping::ClampingMethod::NoClamping);
    for (Index j = 0; j < columns_number; ++j)
    {
        layer->set_lower_bound(j, type(-1.0));
        layer->set_upper_bound(j, type(1.0));
    }

    ForwardPropagation forward_propagation(rows_number, &neural_network);
    vector<TensorView> inputs = { TensorView(input_data.data(), {rows_number, columns_number}) };
    neural_network.forward_propagate(inputs, forward_propagation, ForwardPropagationMode::Inference);

    TensorView output_view = forward_propagation.get_outputs();
    MatrixMap outputs = output_view.as_matrix();

    for(Index i = 0; i < rows_number; ++i)
        for(Index j = 0; j < columns_number; ++j)
            EXPECT_NEAR(outputs(i, j), input_data(i, j), tolerance);
}
