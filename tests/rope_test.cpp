#include "pch.h"

#include <cmath>
#include <vector>

#include "opennn/tensor_types.h"
#include "opennn/tensor_operations.h"
#include "opennn/random_utilities.h"

using namespace opennn;


// RoPE with base 10000, head_dim 4 (full rotary). Position 0 leaves the input
// unchanged (angle 0); position 1 rotates pair (0,2) by inv_freq_0 = 1 rad and
// pair (1,3) by inv_freq_1 = 0.01 rad.
TEST(RopeTest, ForwardMatchesHandComputed)
{
    const Index batch = 1, seq = 2, head_dim = 4, rotary_dim = 4;
    const Index model_dim = head_dim;   // one head
    const float base = 10000.0f;

    std::vector<float> cos(size_t(seq * rotary_dim));
    std::vector<float> sin(size_t(seq * rotary_dim));
    TensorView cos_view(cos.data(), {seq, rotary_dim});
    TensorView sin_view(sin.data(), {seq, rotary_dim});
    rotary_build_tables(cos_view, sin_view, seq, rotary_dim, base);

    std::vector<float> input  = {1, 2, 3, 4,   1, 0, 0, 0};   // [batch, seq, model_dim]
    std::vector<float> output(input.size(), 0.0f);
    TensorView in_view(input.data(), {batch, seq, model_dim});
    TensorView out_view(output.data(), {batch, seq, model_dim});

    rotary_forward(in_view, cos_view, sin_view, out_view, head_dim, rotary_dim, 0);

    // Position 0: identity.
    EXPECT_NEAR(output[0], 1.0f, 1.0e-5f);
    EXPECT_NEAR(output[1], 2.0f, 1.0e-5f);
    EXPECT_NEAR(output[2], 3.0f, 1.0e-5f);
    EXPECT_NEAR(output[3], 4.0f, 1.0e-5f);

    // Position 1 with x = [1,0,0,0] -> [cos1, 0, sin1, 0].
    EXPECT_NEAR(output[4], std::cos(1.0f), 1.0e-5f);
    EXPECT_NEAR(output[5], 0.0f,           1.0e-5f);
    EXPECT_NEAR(output[6], std::sin(1.0f), 1.0e-5f);
    EXPECT_NEAR(output[7], 0.0f,           1.0e-5f);
}


// A rotation preserves the L2 norm of each head's rotary block.
TEST(RopeTest, PreservesNorm)
{
    const Index batch = 2, seq = 5, num_heads = 3, head_dim = 8, rotary_dim = 8;
    const Index model_dim = num_heads * head_dim;
    const float base = 1.0e6f;

    std::vector<float> cos(size_t(seq * rotary_dim)), sin(size_t(seq * rotary_dim));
    TensorView cos_view(cos.data(), {seq, rotary_dim});
    TensorView sin_view(sin.data(), {seq, rotary_dim});
    rotary_build_tables(cos_view, sin_view, seq, rotary_dim, base);

    const size_t total = size_t(batch * seq * model_dim);
    std::vector<float> input(total), output(total, 0.0f);
    for (size_t i = 0; i < total; ++i)
        input[i] = std::sin(0.017f * float(i)) + 0.3f * std::cos(0.004f * float(i));

    TensorView in_view(input.data(), {batch, seq, model_dim});
    TensorView out_view(output.data(), {batch, seq, model_dim});
    rotary_forward(in_view, cos_view, sin_view, out_view, head_dim, rotary_dim, 0);

    const Index rows = batch * seq;
    for (Index row = 0; row < rows; ++row)
        for (Index h = 0; h < num_heads; ++h)
        {
            double norm_in = 0.0, norm_out = 0.0;
            const Index base_i = row * model_dim + h * head_dim;
            for (Index j = 0; j < head_dim; ++j)
            {
                norm_in  += double(input[size_t(base_i + j)])  * input[size_t(base_i + j)];
                norm_out += double(output[size_t(base_i + j)]) * output[size_t(base_i + j)];
            }
            EXPECT_NEAR(std::sqrt(norm_out), std::sqrt(norm_in), 1.0e-4f);
        }
}


// RoPE is an orthogonal map, so the backward (its transpose) is the inverse:
// rotary_backward(rotary_forward(x)) == x.
TEST(RopeTest, BackwardIsInverseRotation)
{
    const Index batch = 2, seq = 4, num_heads = 2, head_dim = 6, rotary_dim = 6;
    const Index model_dim = num_heads * head_dim;
    const float base = 1.0e6f;

    std::vector<float> cos(size_t(seq * rotary_dim)), sin(size_t(seq * rotary_dim));
    TensorView cos_view(cos.data(), {seq, rotary_dim});
    TensorView sin_view(sin.data(), {seq, rotary_dim});
    rotary_build_tables(cos_view, sin_view, seq, rotary_dim, base);

    const size_t total = size_t(batch * seq * model_dim);
    std::vector<float> input(total), rotated(total, 0.0f), recovered(total, 0.0f);
    for (size_t i = 0; i < total; ++i)
        input[i] = std::cos(0.011f * float(i)) - 0.5f * std::sin(0.003f * float(i));

    TensorView in_view(input.data(), {batch, seq, model_dim});
    TensorView rot_view(rotated.data(), {batch, seq, model_dim});
    TensorView rec_view(recovered.data(), {batch, seq, model_dim});

    rotary_forward(in_view, cos_view, sin_view, rot_view, head_dim, rotary_dim, 0);
    rotary_backward(rot_view, cos_view, sin_view, rec_view, head_dim, rotary_dim, 0);

    double max_abs = 0.0;
    for (size_t i = 0; i < total; ++i)
        max_abs = std::max(max_abs, std::abs(double(recovered[i]) - double(input[i])));
    EXPECT_LT(max_abs, 1.0e-4);
}
