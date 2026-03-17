#include "pch.h"

#include "../opennn/tensor_utilities.h"
#include "../opennn/bounding_layer.h"

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

    Bounding bounding_layer({ columns_number });

    bounding_layer.set_bounding_method(Bounding::BoundingMethod::Bounding);
    for (Index j = 0; j < columns_number; ++j)
    {
        bounding_layer.set_lower_bound(j, type(-1.0));
        bounding_layer.set_upper_bound(j, type(1.0));
    }

    const Index rows_number = 2;

    // Inputs usando MatrixR

    MatrixR input_data(rows_number, columns_number);
    input_data << type(-5.0), type(0.5), type(10.0),
        type(-1.0), type(0.0), type(1.0);

    // Forward propagation

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<BoundingForwardPropagation>(rows_number, &bounding_layer);

    TensorView input_view(input_data.data(), {rows_number, columns_number});

    vector<TensorView> input_views = { input_view };

    bounding_layer.forward_propagate(input_views, forward_propagation, false);

    // Outputs usando MatrixMap

    MatrixMap outputs(forward_propagation->outputs.data,
                      rows_number, columns_number);

    // Verificar dimensiones

    EXPECT_EQ(outputs.rows(), rows_number);
    EXPECT_EQ(outputs.cols(), columns_number);

    // Verificar valores

    MatrixR expected_output(rows_number, columns_number);
    expected_output << type(-1.0), type(0.5), type(1.0),
        type(-1.0), type(0.0), type(1.0);

    for(Index i = 0; i < rows_number; ++i)
        for(Index j = 0; j < columns_number; ++j)
            EXPECT_NEAR(outputs(i, j), expected_output(i, j), tolerance);

    EXPECT_EQ(bounding_layer.get_output_shape(), Shape{ columns_number });
}