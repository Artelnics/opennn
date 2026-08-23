//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M U L T I H E A D   P R O J E C T I O N   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// split_heads and concatenate_heads swap the two middle axes of a rank-4
// tensor, between (batch, sequence, heads, dim) and (batch, heads, sequence,
// dim). Pure index arithmetic, reached only through attention, and the kind of
// thing that stays plausible while being wrong: a transposed head layout still
// produces finite outputs and a network that trains, just to the wrong answer.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/multihead_projection_operator.h"

using namespace opennn;

namespace
{

constexpr Index batch = 2;
constexpr Index sequence = 3;
constexpr Index heads = 4;
constexpr Index dimension = 2;
constexpr Index total = batch * sequence * heads * dimension;


// Each element carries its own (b, s, h, d) coordinate so a misplaced value
// says where it came from rather than just being wrong.
float coordinate_code(Index b, Index s, Index h, Index d)
{
    return float(1000 * b + 100 * s + 10 * h + d);
}

}


TEST(MultiHeadProjectionOperatorTest, SplitHeadsSwapsTheMiddleAxes)
{
    VectorR source(total);

    // Source laid out as (batch, sequence, heads, dim).
    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < sequence; ++s)
            for (Index h = 0; h < heads; ++h)
                for (Index d = 0; d < dimension; ++d)
                    source(((b * sequence + s) * heads + h) * dimension + d)
                        = coordinate_code(b, s, h, d);

    VectorR destination = VectorR::Constant(total, -1.0f);

    const TensorView source_view(source.data(), {batch, sequence, heads, dimension},
                                 Type::FP32, Device::CPU);
    TensorView destination_view(destination.data(), {batch, heads, sequence, dimension},
                                Type::FP32, Device::CPU);

    split_heads(source_view, destination_view);

    // Destination is (batch, heads, sequence, dim): same element, new index.
    for (Index b = 0; b < batch; ++b)
        for (Index h = 0; h < heads; ++h)
            for (Index s = 0; s < sequence; ++s)
                for (Index d = 0; d < dimension; ++d)
                    EXPECT_FLOAT_EQ(destination(((b * heads + h) * sequence + s) * dimension + d),
                                    coordinate_code(b, s, h, d))
                        << "b=" << b << " h=" << h << " s=" << s << " d=" << d;
}


TEST(MultiHeadProjectionOperatorTest, ConcatenateHeadsInvertsSplitHeads)
{
    VectorR original(total);
    for (Index i = 0; i < total; ++i) original(i) = float(i) * 0.5f - 3.0f;

    VectorR split = VectorR::Constant(total, -1.0f);
    VectorR round_trip = VectorR::Constant(total, -1.0f);

    const TensorView original_view(original.data(), {batch, sequence, heads, dimension},
                                   Type::FP32, Device::CPU);
    TensorView split_view(split.data(), {batch, heads, sequence, dimension},
                          Type::FP32, Device::CPU);

    split_heads(original_view, split_view);

    // Back the other way: the shape handed in is the split layout, so the two
    // middle extents are swapped relative to the first call.
    const TensorView split_source(split.data(), {batch, heads, sequence, dimension},
                                  Type::FP32, Device::CPU);
    TensorView round_trip_view(round_trip.data(), {batch, sequence, heads, dimension},
                               Type::FP32, Device::CPU);

    concatenate_heads(split_source, round_trip_view);

    for (Index i = 0; i < total; ++i)
        EXPECT_FLOAT_EQ(round_trip(i), original(i)) << "at index " << i;
}


TEST(MultiHeadProjectionOperatorTest, SingleHeadIsAPlainCopy)
{
    // With one head the two layouts coincide, so nothing should move. Worth
    // pinning because it is the shape a degenerate attention configuration
    // takes, and an off-by-one in the stride would still look right at heads>1.
    constexpr Index single_total = batch * sequence * 1 * dimension;

    VectorR source(single_total);
    for (Index i = 0; i < single_total; ++i) source(i) = float(i) + 0.25f;

    VectorR destination = VectorR::Constant(single_total, -1.0f);

    const TensorView source_view(source.data(), {batch, sequence, 1, dimension},
                                 Type::FP32, Device::CPU);
    TensorView destination_view(destination.data(), {batch, 1, sequence, dimension},
                                Type::FP32, Device::CPU);

    split_heads(source_view, destination_view);

    for (Index i = 0; i < single_total; ++i)
        EXPECT_FLOAT_EQ(destination(i), source(i)) << "at index " << i;
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
