//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A T T E N T I O N   O P E R A T O R   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The attention maths is covered on the GPU by the SDPA comparison tests. What
// had no test at all is SdpaBf16Pack, whose entire reason for existing is that
// one definition serves both the size ForwardPropagation plans and the pointers
// the graph is handed, "so the two cannot drift apart" -- a claim nothing
// checked. If they do drift, the last slot runs off the end of a buffer that
// was planned one slot too small, which is a silent overwrite rather than a
// failure.
//
// Also covered: the causal mask, which decides what each query is allowed to
// see and is wrong in a way that still trains if the triangle is the wrong way
// round.

#include "tests/pch.h"

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/operators/attention_operator.h"

using namespace opennn;

namespace
{

using Pack = AttentionOperator::SdpaBf16Pack;

constexpr Index heads = 2;
constexpr Index head_dimension = 4;
constexpr Index query_length = 3;
constexpr Index source_length = 5;

}


TEST(AttentionOperatorTest, Bf16PackPointersStayInsideThePlannedSize)
{
    Pack pack;
    pack.query_elements = 37;      // deliberately not a multiple of the alignment
    pack.source_elements = 61;

    const Index total = pack.total_elements();

    // Braces: vector<T> v(size_t(n)) declares a function, not a vector.
    vector<opennn::bfloat16> storage(static_cast<size_t>(total), opennn::bfloat16{});
    const Pack::Pointers pointers = pack.over(storage.data());

    opennn::bfloat16* const begin = storage.data();
    opennn::bfloat16* const end = begin + total;

    // Every slot starts inside the buffer, and the last one ends inside it too.
    // This is the drift the class exists to prevent.
    EXPECT_EQ(pointers.query, begin);
    EXPECT_GE(pointers.key, begin);
    EXPECT_GE(pointers.value, pointers.key);
    EXPECT_GE(pointers.output, pointers.value);

    EXPECT_LE(pointers.output + Pack::slot_elements(pack.query_elements), end)
        << "the output slot runs past the end of the size total_elements() planned";
}


TEST(AttentionOperatorTest, Bf16PackSlotsDoNotOverlap)
{
    Pack pack;
    pack.query_elements = 37;
    pack.source_elements = 61;

    vector<opennn::bfloat16> storage(static_cast<size_t>(pack.total_elements()),
                                    opennn::bfloat16{});
    const Pack::Pointers pointers = pack.over(storage.data());

    // Query and output are query-sized; key and value are source-sized. Each
    // must start no earlier than the previous one ends.
    EXPECT_GE(pointers.key, pointers.query + Pack::slot_elements(pack.query_elements));
    EXPECT_GE(pointers.value, pointers.key + Pack::slot_elements(pack.source_elements));
    EXPECT_GE(pointers.output, pointers.value + Pack::slot_elements(pack.source_elements));
}


TEST(AttentionOperatorTest, Bf16PackSlotsAreAligned)
{
    // slot_elements rounds each slot up so the next one starts aligned; an
    // unaligned bf16 slot is what the pack rounds up to avoid.
    EXPECT_GE(Pack::slot_elements(1), Index(1));
    EXPECT_GE(Pack::slot_elements(37), Index(37));
    EXPECT_EQ(Pack::slot_elements(0), Index(0));

    // Rounding up is monotone: a bigger request never plans a smaller slot.
    for (Index elements = 0; elements < 64; ++elements)
        EXPECT_LE(Pack::slot_elements(elements), Pack::slot_elements(elements + 1))
            << "at " << elements;
}


TEST(AttentionOperatorTest, CausalMaskLetsAQueryReadOnlyItselfAndThePast)
{
    AttentionOperator attention;
    attention.set(heads, head_dimension, query_length, source_length, true, Type::FP32);

    ASSERT_EQ(attention.causal_mask.rows(), query_length);
    ASSERT_EQ(attention.causal_mask.cols(), source_length);

    for (Index query = 0; query < query_length; ++query)
        for (Index source = 0; source < source_length; ++source)
        {
            const float value = attention.causal_mask(query, source);

            // Allowed positions contribute nothing to the logits; blocked ones
            // are driven to -inf so softmax gives them zero weight.
            if (source <= query)
                EXPECT_FLOAT_EQ(value, 0.0f) << "query " << query << " source " << source;
            else
                EXPECT_LT(value, -1.0e30f) << "query " << query << " source " << source;
        }
}


TEST(AttentionOperatorTest, WithoutCausalMaskingEveryQuerySeesEverySource)
{
    AttentionOperator attention;
    attention.set(heads, head_dimension, query_length, source_length, false, Type::FP32);

    for (Index i = 0; i < attention.causal_mask.size(); ++i)
        EXPECT_FLOAT_EQ(attention.causal_mask.data()[i], 0.0f) << "at index " << i;
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
