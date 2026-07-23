#include "pch.h"

#include <vector>

#include "opennn/tensor_types.h"
#include "opennn/tensor_operations.h"

using namespace opennn;


// Tied lm_head: logits[v] = sum_h input[h] * embed[v, h] (raw, no softmax).
TEST(TiedLmHeadTest, LogitsAreInputTimesEmbeddingTransposed)
{
    const Index batch = 1, seq = 2, hidden = 4, vocab = 3;

    // input: token 0 selects embed[:,0], token 1 selects embed[:,1].
    std::vector<float> input = { 1, 0, 0, 0,   0, 1, 0, 0 };
    // embed [vocab, hidden] row-major.
    std::vector<float> embed = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };
    std::vector<float> output(size_t(seq * vocab), 0.0f);

    TensorView in(input.data(), {batch, seq, hidden});
    TensorView e(embed.data(), {vocab, hidden});
    TensorView out(output.data(), {batch, seq, vocab});
    tied_lm_head_forward(in, e, out);

    // seq 0 -> embed[:,0] = {1,5,9};  seq 1 -> embed[:,1] = {2,6,10}.
    EXPECT_NEAR(output[0], 1.0f, 1.0e-5f);
    EXPECT_NEAR(output[1], 5.0f, 1.0e-5f);
    EXPECT_NEAR(output[2], 9.0f, 1.0e-5f);
    EXPECT_NEAR(output[3], 2.0f, 1.0e-5f);
    EXPECT_NEAR(output[4], 6.0f, 1.0e-5f);
    EXPECT_NEAR(output[5], 10.0f, 1.0e-5f);
}
