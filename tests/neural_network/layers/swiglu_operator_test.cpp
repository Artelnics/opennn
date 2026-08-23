//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S W I G L U   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// swiglu_forward/backward had no direct test: they were reached only through a
// whole transformer, where a wrong gradient shows up as slightly worse training
// rather than as anything a test asserts on. These check the closed forms
// against an independent implementation and the analytic gradient against a
// central difference, plus the two optional-output paths the operator supports.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/swiglu_operator.h"

using namespace opennn;

namespace
{

// SwiGLU(gate, up) = SiLU(gate) * up, written out separately from the
// implementation so a change to both at once is what it takes to fool this.
double reference_swiglu(double gate, double up)
{
    const double sigmoid = 1.0 / (1.0 + exp(-gate));

    return gate * sigmoid * up;
}


TensorView view_of(VectorR& values)
{
    return TensorView(values.data(), {values.size()}, Type::FP32, Device::CPU);
}

}


TEST(SwiGluOperatorTest, ForwardMatchesTheClosedForm)
{
    VectorR gate(7);
    VectorR up(7);

    // Spans both signs and zero: SiLU is not symmetric and the negative tail is
    // where a sign slip hides.
    gate << -6.0f, -1.5f, -0.25f, 0.0f, 0.25f, 1.5f, 6.0f;
    up   <<  2.0f,  0.5f, -1.0f,  3.0f, -0.75f, 1.25f, -2.5f;

    VectorR output = VectorR::Constant(7, -999.0f);

    TensorView output_view = view_of(output);
    swiglu_forward(view_of(gate), view_of(up), output_view);

    for (Index i = 0; i < gate.size(); ++i)
        EXPECT_NEAR(output(i), float(reference_swiglu(gate(i), up(i))), 1.0e-5f)
            << "at index " << i;
}


TEST(SwiGluOperatorTest, GradientMatchesCentralDifferences)
{
    VectorR gate(5);
    VectorR up(5);

    gate << -2.0f, -0.5f, 0.0f, 0.75f, 3.0f;
    up   <<  1.5f,  2.0f, -1.0f, 0.5f, -2.0f;

    VectorR output_delta = VectorR::Constant(5, 1.0f);
    output_delta(1) = -2.0f;
    output_delta(3) = 0.5f;

    VectorR gate_delta = VectorR::Zero(5);
    VectorR up_delta = VectorR::Zero(5);

    TensorView gate_delta_view = view_of(gate_delta);
    TensorView up_delta_view = view_of(up_delta);

    swiglu_backward(view_of(output_delta), view_of(gate), view_of(up),
                    gate_delta_view, up_delta_view);

    constexpr double step = 1.0e-4;

    for (Index i = 0; i < gate.size(); ++i)
    {
        const double d_gate =
            (reference_swiglu(gate(i) + step, up(i)) - reference_swiglu(gate(i) - step, up(i)))
            / (2.0 * step);

        const double d_up =
            (reference_swiglu(gate(i), up(i) + step) - reference_swiglu(gate(i), up(i) - step))
            / (2.0 * step);

        EXPECT_NEAR(gate_delta(i), float(d_gate * output_delta(i)), 1.0e-3f)
            << "gate gradient at index " << i;
        EXPECT_NEAR(up_delta(i), float(d_up * output_delta(i)), 1.0e-4f)
            << "up gradient at index " << i;
    }
}


TEST(SwiGluOperatorTest, EitherGradientOutputMayBeOmitted)
{
    VectorR gate(3);
    VectorR up(3);
    gate << -1.0f, 0.5f, 2.0f;
    up   <<  2.0f, -1.0f, 0.25f;

    VectorR output_delta = VectorR::Constant(3, 1.0f);

    VectorR both_gate = VectorR::Zero(3);
    VectorR both_up = VectorR::Zero(3);

    TensorView both_gate_view = view_of(both_gate);
    TensorView both_up_view = view_of(both_up);
    swiglu_backward(view_of(output_delta), view_of(gate), view_of(up),
                    both_gate_view, both_up_view);

    // An empty view means "do not write this one"; the other must be unaffected.
    TensorView absent;

    VectorR only_up = VectorR::Zero(3);
    TensorView only_up_view = view_of(only_up);
    swiglu_backward(view_of(output_delta), view_of(gate), view_of(up),
                    absent, only_up_view);

    VectorR only_gate = VectorR::Zero(3);
    TensorView only_gate_view = view_of(only_gate);
    TensorView absent_up;
    swiglu_backward(view_of(output_delta), view_of(gate), view_of(up),
                    only_gate_view, absent_up);

    for (Index i = 0; i < gate.size(); ++i)
    {
        EXPECT_FLOAT_EQ(only_up(i), both_up(i)) << "at index " << i;
        EXPECT_FLOAT_EQ(only_gate(i), both_gate(i)) << "at index " << i;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
