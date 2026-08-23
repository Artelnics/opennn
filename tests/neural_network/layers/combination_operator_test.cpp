//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O M B I N A T I O N   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// Every Dense layer runs through this operator, and it had no direct test. The
// forward maths is covered indirectly everywhere, so these concentrate on the
// parts that are not: the parameter-slot contract, which puts the bias FIRST
// and drops it entirely when use_bias is off, and the tied-projection rule that
// decides whether the operator owns weights worth initialising.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/combination_operator.h"
#include "opennn/neural_network/forward_propagation.h"

using namespace opennn;

namespace
{

constexpr Index input_features = 3;
constexpr Index output_features = 2;


CombinationOperator make_combination(bool use_bias)
{
    CombinationOperator combination;
    combination.set(input_features, output_features);
    combination.use_bias = use_bias;

    return combination;
}

}


TEST(CombinationOperatorTest, BiasComesFirstInTheParameterSpecs)
{
    const CombinationOperator with_bias = make_combination(true);

    const vector<TensorSpec> specs = with_bias.parameter_specs();

    // Bias first, then the weight matrix. link_parameters binds positionally
    // against this order, so swapping them silently mis-binds every Dense
    // layer rather than failing anywhere obvious.
    ASSERT_EQ(specs.size(), size_t(2));
    EXPECT_EQ(specs[0].shape, (Shape{output_features}));
    EXPECT_EQ(specs[1].shape, (Shape{input_features, output_features}));
}


TEST(CombinationOperatorTest, WithoutBiasTheWeightsAreTheOnlySlot)
{
    const CombinationOperator without_bias = make_combination(false);

    const vector<TensorSpec> specs = without_bias.parameter_specs();

    ASSERT_EQ(specs.size(), size_t(1));
    EXPECT_EQ(specs[0].shape, (Shape{input_features, output_features}));
}


TEST(CombinationOperatorTest, LinkParametersBindsInSpecOrder)
{
    VectorR bias_storage = VectorR::Zero(output_features);
    MatrixR weight_storage = MatrixR::Zero(input_features, output_features);

    const TensorView bias_view(bias_storage.data(), {output_features}, Type::FP32, Device::CPU);
    const TensorView weight_view(weight_storage.data(), {input_features, output_features},
                                 Type::FP32, Device::CPU);

    CombinationOperator with_bias = make_combination(true);
    const vector<TensorView> both = {bias_view, weight_view};
    with_bias.link_parameters(both);

    EXPECT_EQ(with_bias.bias.get_data(), bias_storage.data());
    EXPECT_EQ(with_bias.weights.get_data(), weight_storage.data());

    // Without a bias the single view is the weights, not the bias.
    CombinationOperator without_bias = make_combination(false);
    const vector<TensorView> weights_only = {weight_view};
    without_bias.link_parameters(weights_only);

    EXPECT_EQ(without_bias.weights.get_data(), weight_storage.data());
    EXPECT_TRUE(without_bias.bias.empty());
}


TEST(CombinationOperatorTest, TiedProjectionsOwnNothingToInitialise)
{
    MatrixR weight_storage = MatrixR::Zero(input_features, output_features);
    const TensorView weight_view(weight_storage.data(), {input_features, output_features},
                                 Type::FP32, Device::CPU);

    CombinationOperator combination = make_combination(false);

    // Nothing linked yet: there is no buffer to write into.
    EXPECT_FALSE(combination.owns_initializable_weights());

    const vector<TensorView> weights_only = {weight_view};
    combination.link_parameters(weights_only);
    EXPECT_TRUE(combination.owns_initializable_weights());

    // A tied projection borrows its source layer's weights. Initialising them
    // here would overwrite the layer they belong to.
    combination.tied_transposed = true;
    EXPECT_FALSE(combination.owns_initializable_weights());
}


TEST(CombinationOperatorTest, GlorotInitialisationStaysInsideItsLimit)
{
    MatrixR weight_storage = MatrixR::Zero(input_features, output_features);
    VectorR bias_storage = VectorR::Constant(output_features, 7.0f);

    const TensorView weight_view(weight_storage.data(), {input_features, output_features},
                                 Type::FP32, Device::CPU);
    const TensorView bias_view(bias_storage.data(), {output_features}, Type::FP32, Device::CPU);

    CombinationOperator combination = make_combination(true);
    const vector<TensorView> views = {bias_view, weight_view};
    combination.link_parameters(views);

    combination.set_parameters_glorot();

    const float limit = sqrt(6.0f / float(input_features + output_features));

    for (Index i = 0; i < weight_storage.size(); ++i)
        EXPECT_LE(abs(weight_storage.data()[i]), limit + 1.0e-5f) << "at index " << i;

    // Glorot zeroes the bias rather than leaving whatever was there.
    for (Index i = 0; i < bias_storage.size(); ++i)
        EXPECT_FLOAT_EQ(bias_storage(i), 0.0f) << "at index " << i;
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
