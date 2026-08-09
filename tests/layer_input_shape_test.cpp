//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   I N P U T   S H A P E   T E S T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// set_input_shape used to react five different ways to a rank a layer cannot
// represent: check_rank threw, throw_if threw, two layers returned silently
// leaving their previous shape, one validated nothing, and Embedding - which
// never overrode the empty base - ignored the call outright.
//
// Validation now lives in Layer::set_input_shape alone, so every layer answers
// the same way. These tests cover the ones that used to stay quiet, since those
// are the cases the change exists for.

#include "pch.h"

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/bounding_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/normalization_layer_3d.h"
#include "opennn/neural_network/layers/scaling_layer.h"

using namespace opennn;

namespace
{

// A shape the layer has not declared, built from a rank it rejects.
Shape shape_of_rank(Index rank)
{
    Shape shape;
    for (Index i = 0; i < rank; ++i) shape = shape.append(Shape{2});
    return shape;
}

void expect_refuses(Layer& layer, Index rank)
{
    ASSERT_FALSE(layer.accepts_input_rank(rank))
        << layer.get_name() << " was expected to reject rank " << rank;

    EXPECT_THROW(layer.set_input_shape(shape_of_rank(rank)), exception)
        << layer.get_name() << " accepted an input of rank " << rank;
}

void expect_accepts(Layer& layer, Index rank)
{
    ASSERT_TRUE(layer.accepts_input_rank(rank))
        << layer.get_name() << " was expected to accept rank " << rank;

    EXPECT_NO_THROW(layer.set_input_shape(shape_of_rank(rank)))
        << layer.get_name() << " refused an input of rank " << rank;
}

}

// Normalization3d and GroupedQueryAttention used to `return` on a rank below 2,
// leaving the caller believing the shape had been set.
TEST(LayerInputShape, NormalizationRefusesInsteadOfIgnoring)
{
    Normalization3d layer(Shape{2, 2});

    expect_refuses(layer, 1);
    expect_accepts(layer, 2);
}

// Bounding validated nothing at all.
TEST(LayerInputShape, BoundingRefusesARankItCannotRepresent)
{
    Bounding layer(Shape{2});

    expect_refuses(layer, 4);
    expect_accepts(layer, 1);
}

// Embedding never overrode set_input_shape, so the empty base swallowed it.
TEST(LayerInputShape, EmbeddingNoLongerSwallowsTheCall)
{
    Embedding layer(Shape{2, 2});

    expect_refuses(layer, 3);
}

// The layers that already threw keep behaving the same way.
TEST(LayerInputShape, LayersThatAlreadyValidatedAreUnchanged)
{
    opennn::Dense dense(Shape{2}, Shape{2}, "Linear");
    expect_accepts(dense, 1);
    expect_refuses(dense, 3);

    Scaling scaling(Shape{2});
    expect_accepts(scaling, 1);
    expect_refuses(scaling, 4);
}

// An empty shape means "not configured yet" and has always been allowed
// through; the hook must not start rejecting it.
TEST(LayerInputShape, EmptyShapeIsStillAccepted)
{
    opennn::Dense dense(Shape{2}, Shape{2}, "Linear");

    EXPECT_NO_THROW(dense.set_input_shape(Shape{}));
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
