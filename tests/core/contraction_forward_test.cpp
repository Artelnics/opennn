#include "tests/pch.h"

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"

using namespace opennn;

// The CPU dense forward has two kernels behind one door: a row-blocked schedule
// over single-threaded MKL, and Eigen's tensor contraction with the bias and the
// ReLU carried in its output kernel. Which one runs depends on the shape - the
// contraction is taken only above OPENNN_GEMM_CONTRACT_FLOPS, 8e9 by default -
// so a network small enough to be convenient never exercises it, and until this
// test existed nothing in the suite did. The batch here is chosen to be just
// over that line: 8192 * 1024 * 1024 = 8.6e9.
//
// A contraction sums in a different order than a GEMM does, so this asks for
// agreement with the definition to a tolerance, not to the bit.
TEST(ContractionForwardTest, LargeBatchDenseForwardMatchesTheDefinition)
{
    const Index batch = 8192;
    const Index inputs_number = 1024;
    const Index outputs_number = 1024;

    NeuralNetwork network;
    network.add_layer(make_unique<opennn::Dense>(Shape{inputs_number}, Shape{outputs_number}, "ReLU"));
    network.compile();
    network.set_parameters_glorot();

    const MatrixR inputs_host = MatrixR::Random(batch, inputs_number);
    const TensorView input_view(const_cast<float*>(inputs_host.data()),
                                Shape{batch, inputs_number}, Type::FP32);

    ForwardPropagation forward_propagation(batch, &network);
    network.forward_propagate({input_view}, forward_propagation, false);

    const MatrixMap outputs = forward_propagation.get_outputs().as_matrix();

    // A Dense keeps its parameters as views; take them by rank rather than by
    // position, which is not part of the contract.
    const vector<TensorView>& parameters = network.get_layer(0)->get_parameter_views();

    const TensorView* weight_view = nullptr;
    const TensorView* bias_view = nullptr;

    for (const TensorView& parameter : parameters)
    {
        if (parameter.get_shape().get_rank() == 2) weight_view = &parameter;
        if (parameter.get_shape().get_rank() == 1) bias_view = &parameter;
    }

    ASSERT_NE(weight_view, nullptr);
    ASSERT_NE(bias_view, nullptr);

    const MatrixMap weights = weight_view->as_matrix();
    const float* const bias = bias_view->as<float>();

    ASSERT_EQ(weights.rows(), inputs_number);
    ASSERT_EQ(weights.cols(), outputs_number);

    // Only a few rows: the reference is the whole point, so it is computed the
    // slow, obvious way, and the kernel is uniform across rows.
    const Index checked_rows = 8;

    for (Index row = 0; row < checked_rows; ++row)
    {
        const MatrixR reference = (inputs_host.row(row) * weights).eval();

        for (Index column = 0; column < outputs_number; ++column)
        {
            const float expected = max(0.0f, reference(0, column) + bias[column]);
            const float tolerance = 1.0e-3f * max(1.0f, abs(expected));

            ASSERT_NEAR(outputs(row, column), expected, tolerance)
                << "row " << row << " column " << column;
        }
    }
}
