#include "pch.h"

#include <cmath>
#include <vector>

#include "opennn/tensor_types.h"
#include "opennn/tensor_operations.h"

using namespace opennn;


// QK-Norm normalizes each head's head_dim vector independently. With a unit
// weight, every head's output has mean-square ~= 1 regardless of its input scale.
TEST(QKNormTest, PerHeadUnitRootMeanSquare)
{
    const Index heads = 2, head_dim = 4;
    const float eps = 1.0e-6f;

    std::vector<float> weight(size_t(head_dim), 1.0f);
    // Head 0 small magnitude, head 1 large: each must still normalize to unit RMS.
    std::vector<float> input = { 0.1f, 0.2f, 0.3f, 0.4f,   10.0f, -20.0f, 30.0f, -40.0f };
    std::vector<float> output(input.size(), 0.0f);

    TensorView in(input.data(), {1, 1, heads * head_dim});
    TensorView w(weight.data(), {head_dim});
    TensorView out(output.data(), {1, 1, heads * head_dim});
    qk_norm_forward(in, w, out, head_dim, eps);

    for (Index h = 0; h < heads; ++h)
    {
        double mean_square = 0.0;
        for (Index d = 0; d < head_dim; ++d)
        {
            const float y = output[size_t(h * head_dim + d)];
            mean_square += double(y) * y;
        }
        mean_square /= double(head_dim);
        EXPECT_NEAR(mean_square, 1.0, 1.0e-4);
    }
}


// The per-channel weight scales the normalized output: y = weight * x / rms.
TEST(QKNormTest, WeightScalesOutput)
{
    const Index head_dim = 4;
    const float eps = 1.0e-6f;

    std::vector<float> weight = { 2.0f, 0.5f, 1.0f, 3.0f };
    std::vector<float> input  = { 1.0f, 2.0f, 3.0f, 4.0f };
    std::vector<float> output(input.size(), 0.0f);

    TensorView in(input.data(), {1, 1, head_dim});
    TensorView w(weight.data(), {head_dim});
    TensorView out(output.data(), {1, 1, head_dim});
    qk_norm_forward(in, w, out, head_dim, eps);

    // rms = sqrt(mean(x^2)) = sqrt(7.5); y_d = weight_d * x_d / rms.
    const float rms = std::sqrt(7.5f);
    for (Index d = 0; d < head_dim; ++d)
        EXPECT_NEAR(output[size_t(d)], weight[size_t(d)] * input[size_t(d)] / rms, 1.0e-5f);
}
