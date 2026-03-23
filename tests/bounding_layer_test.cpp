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

    MatrixR input_data(rows_number, columns_number);
    input_data << type(-5.0), type(0.5), type(10.0),
        type(-1.0), type(0.0), type(1.0);

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<BoundingForwardPropagation>(rows_number, &bounding_layer);

    forward_propagation->initialize();

    Tensor1 workspace(get_size(forward_propagation->get_workspace_views()));
    link(workspace.data(), forward_propagation->get_workspace_views());

    forward_propagation->inputs = { TensorView(input_data.data(), {rows_number, columns_number}) };

    bounding_layer.forward_propagate(forward_propagation, false);

    MatrixMap outputs(forward_propagation->outputs.data,
                      rows_number, columns_number);

    EXPECT_EQ(outputs.rows(), rows_number);
    EXPECT_EQ(outputs.cols(), columns_number);

    MatrixR expected_output(rows_number, columns_number);
    expected_output << type(-1.0), type(0.5), type(1.0),
        type(-1.0), type(0.0), type(1.0);

    for(Index i = 0; i < rows_number; ++i)
        for(Index j = 0; j < columns_number; ++j)
            EXPECT_NEAR(outputs(i, j), expected_output(i, j), tolerance);

    EXPECT_EQ(bounding_layer.get_output_shape(), Shape{ columns_number });
}
