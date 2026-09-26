#include "tests/pch.h"

#include <vector>

#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/tensor_operations.h"

using namespace opennn;

TEST(LinearForwardTransposedTest, LogitsAreInputTimesEmbeddingTransposed)
{
    const Index batch = 1, seq = 2, hidden = 4, vocab = 3;

    vector<float> input = { 1, 0, 0, 0,   0, 1, 0, 0 };

    vector<float> embed = {
        1,  2,  3,  4,
        5,  6,  7,  8,
        9, 10, 11, 12
    };
    vector<float> output(size_t(seq * vocab), 0.0f);

    TensorView in(input.data(), {batch, seq, hidden});
    TensorView e(embed.data(), {vocab, hidden});
    TensorView out(output.data(), {batch, seq, vocab});
    linear_forward_transposed(in, e, out);

    EXPECT_NEAR(output[0], 1.0f, 1.0e-5f);
    EXPECT_NEAR(output[1], 5.0f, 1.0e-5f);
    EXPECT_NEAR(output[2], 9.0f, 1.0e-5f);
    EXPECT_NEAR(output[3], 2.0f, 1.0e-5f);
    EXPECT_NEAR(output[4], 6.0f, 1.0e-5f);
    EXPECT_NEAR(output[5], 10.0f, 1.0e-5f);
}

#ifdef OPENNN_HAS_CUDA
namespace
{
template<typename T>
void check_padded_lt_invocations(DeviceDataType dtype, BlasOperation trans_a,
                                 BlasOperation trans_b, LinearEpilogue epilogue)
{
    constexpr int m = 8, n = 5, k = 16, ldd = m + 8;
    const int lda = (trans_a == CUBLAS_OP_N ? m : k) + 8;
    const int ldb = ((trans_b == CUBLAS_OP_N ? k : n) + 7) / 8 * 8 + 8;
    vector<T> a(lda * (trans_a == CUBLAS_OP_N ? k : m), T(7.0f));
    vector<T> b(ldb * (trans_b == CUBLAS_OP_N ? n : k), T(9.0f));
    vector<T> c(ldd * n, T(11.0f)), initial_d(ldd * n, T(13.0f)), bias(m);
    const auto a_index = [&](int row, int inner) {
        return trans_a == CUBLAS_OP_N ? row + inner * lda : inner + row * lda;
    };
    const auto b_index = [&](int inner, int column) {
        return trans_b == CUBLAS_OP_N ? inner + column * ldb : column + inner * ldb;
    };
    for (int row = 0; row < m; ++row)
        for (int inner = 0; inner < k; ++inner)
            a[a_index(row, inner)] = T(float((row * 3 + inner) % 9 - 4) / 8.0f);
    for (int column = 0; column < n; ++column)
    {
        for (int inner = 0; inner < k; ++inner)
            b[b_index(inner, column)] = T(float((inner * 5 + column) % 7 - 3) / 8.0f);
        for (int row = 0; row < m; ++row)
        {
            c[row + column * ldd] = T(float(row + column + 1) / 4.0f);
            initial_d[row + column * ldd] = T(float(row - column - 2) / 4.0f);
        }
    }

    Buffer device_a, device_b, device_c, device_d, device_bias;
    const DeviceStream stream = device::get_compute_stream();
    const auto upload = [&](Buffer& buffer, const vector<T>& values) {
        const Index bytes = Index(values.size() * sizeof(T));
        buffer.resize_bytes(bytes, Device::CUDA);
        device::copy_async(buffer.data(), values.data(), bytes, device::CopyKind::HostToDevice, stream);
    };
    upload(device_a, a);
    upload(device_b, b);
    upload(device_c, c);

    // The first call tunes with D also serving as C. Later calls reuse that
    // plan while changing the scalars, addend and bias; none belong to its key.
    for (int pass = 0; pass < 3; ++pass)
    {
        SCOPED_TRACE(format("dtype={}, transA={}, transB={}, epilogue={}, pass={}",
                            int(dtype), int(trans_a), int(trans_b), int(epilogue), pass));
        const bool in_place = pass != 1;
        const float alpha = pass == 0 ? 0.5f : pass == 1 ? 1.25f : -0.5f;
        const float beta = pass == 0 ? 0.25f : pass == 1 ? -0.5f : 1.0f;
        for (int row = 0; row < m; ++row) bias[row] = T(float(row - pass * 3) / 8.0f);
        upload(device_d, initial_d);
        upload(device_bias, bias);
        ASSERT_NO_THROW(run_lt_matmul_cached(
            m, n, k, trans_a, trans_b, epilogue,
            device_a.data(), device_b.data(), device_d.data(),
            epilogue == LinearEpilogue::Default ? nullptr : device_bias.data(), dtype, dtype, dtype,
            nullptr, in_place ? nullptr : device_c.data(), alpha, beta, lda, ldb, ldd));

        vector<T> actual(initial_d.size());
        device::copy_async(actual.data(), device_d.data(), Index(actual.size() * sizeof(T)),
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);
        for (int column = 0; column < n; ++column)
            for (int row = 0; row < ldd; ++row)
            {
                const int index = row + column * ldd;
                float expected = float(initial_d[index]);
                if (row < m)
                {
                    float product = 0.0f;
                    for (int inner = 0; inner < k; ++inner)
                        product += float(a[a_index(row, inner)]) * float(b[b_index(inner, column)]);
                    expected = alpha * product + beta * float(in_place ? initial_d[index] : c[index]);
                    if (epilogue == LinearEpilogue::ReluBias)
                        expected = max(0.0f, expected + float(bias[row]));
                    expected = float(T(expected));
                }
                EXPECT_NEAR(float(actual[index]), expected, dtype == CUDA_R_32F ? 1.0e-5f : 0.015625f)
                    << "row=" << row << ", column=" << column;
            }
    }
}
}

TEST(LinearForwardTransposedTest, CudaLtPreservesPaddedOperandsAndCallTimeValues)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "CUDA device unavailable.";
    for (const auto trans_a : {CUBLAS_OP_N, CUBLAS_OP_T})
        for (const auto trans_b : {CUBLAS_OP_N, CUBLAS_OP_T})
            for (const auto epilogue : {LinearEpilogue::Default, LinearEpilogue::ReluBias})
            {
                check_padded_lt_invocations<float>(CUDA_R_32F, trans_a, trans_b, epilogue);
                if (device::cuda_compute_capability() >= 80)
                    check_padded_lt_invocations<opennn::bfloat16>(CUDA_R_16BF, trans_a, trans_b, epilogue);
            }
}
#endif
