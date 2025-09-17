#include "pch.h"

#include "../opennn/tensors.h"
#include "../opennn/bounding_layer.h"

using namespace opennn;

const type tolerance = 1e-9;

TEST(BoundingTest, Constructor)
{
    Bounding bounding_layer;

    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{0});
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

    Tensor<type, 2> inputs(rows_number, columns_number);

    inputs(0, 0) = -5.0;
    inputs(0, 1) = 0.5;
    inputs(0, 2) = 10.0;
    inputs(1, 0) = -1.0;
    inputs(1, 1) = 0.0;
    inputs(1, 2) = 1.0;

    unique_ptr<LayerForwardPropagation> forward_propagation =
        make_unique<BoundingForwardPropagation>(rows_number, &bounding_layer);

    auto eigen_dimensions = inputs.dimensions();
    dimensions dims_vector(eigen_dimensions.begin(), eigen_dimensions.end());

    TensorView input_view(inputs.data(), dims_vector);

    vector<TensorView> input_views = { input_view };

    bounding_layer.forward_propagate(input_views, forward_propagation, false);

    const TensorMap<Tensor<type, 2>> outputs =
        tensor_map<2>(forward_propagation->get_output_pair());

    EXPECT_EQ(outputs.dimension(0), rows_number);
    EXPECT_EQ(outputs.dimension(1), columns_number);

    EXPECT_NEAR(outputs(0, 0), type(-1.0), tolerance);
    EXPECT_NEAR(outputs(0, 1), type(0.5), tolerance);
    EXPECT_NEAR(outputs(0, 2), type(1.0), tolerance);
    EXPECT_NEAR(outputs(1, 0), type(-1.0), tolerance);
    EXPECT_NEAR(outputs(1, 1), type(0.0), tolerance);
    EXPECT_NEAR(outputs(1, 2), type(1.0), tolerance);

    EXPECT_EQ(bounding_layer.get_output_dimensions(), dimensions{ columns_number });
}
