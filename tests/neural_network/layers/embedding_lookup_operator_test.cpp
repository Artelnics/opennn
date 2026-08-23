//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E M B E D D I N G   L O O K U P   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The lookup carries three conventions nothing asserted on directly: token 0 is
// padding and produces a zero row, an out-of-range id is zeroed with a warning
// rather than read out of bounds, and the optional sqrt(d) scaling and
// positional term compose in that order. All of them are silent when wrong --
// a padded row that picks up an embedding still trains, just worse.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/embedding_lookup_operator.h"

using namespace opennn;

namespace
{

constexpr Index vocabulary_size = 5;
constexpr Index sequence_length = 3;
constexpr Index embedding_dimension = 2;
constexpr Index batch = 2;
constexpr Index tokens = batch * sequence_length;


// Row v of the table is {v, -v}, so a looked-up row names the token it came from.
MatrixR embedding_table()
{
    MatrixR weights(vocabulary_size, embedding_dimension);

    for (Index v = 0; v < vocabulary_size; ++v)
    {
        weights(v, 0) = float(v);
        weights(v, 1) = -float(v);
    }

    return weights;
}


TensorView matrix_view(MatrixR& values)
{
    return TensorView(values.data(), {values.rows(), values.cols()}, Type::FP32, Device::CPU);
}

}


TEST(EmbeddingLookupOperatorTest, LooksUpRowsAndZeroesPaddingTokens)
{
    MatrixR weights = embedding_table();

    // Token 0 is padding, and appears mid-sequence as well as at the end so a
    // "stop at the first zero" mistake would show.
    VectorR indices(tokens);
    indices << 1.0f, 0.0f, 3.0f,
               4.0f, 2.0f, 0.0f;

    MatrixR output = MatrixR::Constant(tokens, embedding_dimension, -999.0f);

    const TensorView indices_view(indices.data(), {batch, sequence_length}, Type::FP32, Device::CPU);
    TensorView output_view = matrix_view(output);
    const TensorView no_positional;

    embedding_lookup_forward(indices_view, matrix_view(weights), no_positional, output_view,
                             sequence_length, embedding_dimension, vocabulary_size,
                             false, false);

    for (Index i = 0; i < tokens; ++i)
    {
        const Index token = Index(indices(i));

        SCOPED_TRACE("token index " + to_string(i) + ", id " + to_string(token));

        EXPECT_FLOAT_EQ(output(i, 0), token == 0 ? 0.0f : float(token));
        EXPECT_FLOAT_EQ(output(i, 1), token == 0 ? 0.0f : -float(token));
    }
}


TEST(EmbeddingLookupOperatorTest, OutOfRangeTokensAreZeroedNotReadOutOfBounds)
{
    MatrixR weights = embedding_table();

    // Above the table and negative: both must be refused rather than indexed.
    VectorR indices(tokens);
    indices << 1.0f, 99.0f, 2.0f,
               -4.0f, 3.0f, 1.0f;

    MatrixR output = MatrixR::Constant(tokens, embedding_dimension, -999.0f);

    const TensorView indices_view(indices.data(), {batch, sequence_length}, Type::FP32, Device::CPU);
    TensorView output_view = matrix_view(output);
    const TensorView no_positional;

    embedding_lookup_forward(indices_view, matrix_view(weights), no_positional, output_view,
                             sequence_length, embedding_dimension, vocabulary_size,
                             false, false);

    for (Index i = 0; i < tokens; ++i)
    {
        const Index token = Index(indices(i));
        const bool in_range = token > 0 && token < vocabulary_size;

        SCOPED_TRACE("token index " + to_string(i) + ", id " + to_string(token));

        EXPECT_FLOAT_EQ(output(i, 0), in_range ? float(token) : 0.0f);
        EXPECT_FLOAT_EQ(output(i, 1), in_range ? -float(token) : 0.0f);
    }
}


TEST(EmbeddingLookupOperatorTest, ScalingIsAppliedBeforeThePositionalTerm)
{
    MatrixR weights = embedding_table();

    // Row p of the positional table is {100+p, 200+p}, distinguishable from any
    // embedding, so the two contributions can be told apart in the sum.
    MatrixR positional(sequence_length, embedding_dimension);
    for (Index p = 0; p < sequence_length; ++p)
    {
        positional(p, 0) = 100.0f + float(p);
        positional(p, 1) = 200.0f + float(p);
    }

    VectorR indices(tokens);
    indices << 1.0f, 2.0f, 3.0f,
               4.0f, 1.0f, 2.0f;

    MatrixR output = MatrixR::Constant(tokens, embedding_dimension, -999.0f);

    const TensorView indices_view(indices.data(), {batch, sequence_length}, Type::FP32, Device::CPU);
    TensorView output_view = matrix_view(output);

    embedding_lookup_forward(indices_view, matrix_view(weights), matrix_view(positional), output_view,
                             sequence_length, embedding_dimension, vocabulary_size,
                             true, true);

    const float scale = sqrt(float(embedding_dimension));

    for (Index i = 0; i < tokens; ++i)
    {
        const Index token = Index(indices(i));
        const Index position = i % sequence_length;

        SCOPED_TRACE("token index " + to_string(i));

        // scale first, then add: the other order would scale the positional
        // term too, which is a different model.
        EXPECT_NEAR(output(i, 0), float(token) * scale + positional(position, 0), 1.0e-4f);
        EXPECT_NEAR(output(i, 1), -float(token) * scale + positional(position, 1), 1.0e-4f);
    }
}


TEST(EmbeddingLookupOperatorTest, ValidLengthsCountNonPaddingTokensPerRow)
{
    VectorR indices(tokens);
    indices << 1.0f, 2.0f, 0.0f,     // two real tokens
               0.0f, 0.0f, 0.0f;     // all padding

    const TensorView indices_view(indices.data(), {batch, sequence_length}, Type::FP32, Device::CPU);

    vector<Index> valid_lengths;
    compute_token_valid_lengths(indices_view, sequence_length, valid_lengths);

    ASSERT_EQ(valid_lengths.size(), size_t(batch));
    EXPECT_EQ(valid_lengths[0], 2);
    EXPECT_EQ(valid_lengths[1], 0);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
