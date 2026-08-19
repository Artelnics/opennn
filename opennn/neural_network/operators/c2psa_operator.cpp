//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   O P E R A T O R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/operators/c2psa_operator.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "opennn/neural_network/operators/kernel_c2psa.cuh"
#endif

namespace opennn
{

using Eigen::Dynamic;
using MatF = Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor>;
using MapC = Eigen::Map<const MatF>;
using MapM = Eigen::Map<MatF>;

void C2PSAOperator::set(Index new_h, Index new_w, Index new_channels)
{
    h        = new_h;
    w        = new_w;
    channels = new_channels;
}

vector<TensorSpec> C2PSAOperator::parameter_specs() const
{
    if (channels == 0) return {};
    const Index half_c = channels / 2;
    return {
        {{half_c, half_c}, compute_dtype},
        {{half_c, half_c}, compute_dtype},
        {{half_c, half_c}, compute_dtype},
        {{channels, channels}, compute_dtype},
    };
}

void C2PSAOperator::link_parameters(span<const TensorView> views)
{
    link_views(views, {&Wq, &Wk, &Wv, &Wout});
}

void C2PSAOperator::link_gradients(span<const TensorView> views)
{
    link_views(views, {&dWq, &dWk, &dWv, &dWout});
}

void C2PSAOperator::set_parameters_random()
{
    set_parameters_glorot();
}

void C2PSAOperator::set_parameters_glorot()
{
    if (Wq.empty()) return;
    const Index half_c = channels / 2;
    const float lqkv   = glorot_limit(half_c, half_c);
    const float lout   = glorot_limit(channels, channels);
    set_random_uniform(Wq  .as_vector(), -lqkv, lqkv);
    set_random_uniform(Wk  .as_vector(), -lqkv, lqkv);
    set_random_uniform(Wv  .as_vector(), -lqkv, lqkv);
    set_random_uniform(Wout.as_vector(), -lout, lout);
}

void C2PSAOperator::forward_propagate(ForwardPropagation& fp, size_t layer, bool)
{
    const TensorView& x = get_input(fp, layer);
    TensorView& output  = get_output(fp, layer);

    const Index B      = x.get_shape()[0];
    const Index tokens = Index(h) * Index(w);
    const Index C      = channels;
    const Index half_c = C / 2;
    const float scale  = 1.0f / sqrtf(float(half_c));

#ifdef OPENNN_HAS_CUDA
    if (x.is_cuda())
    {
        const int BT     = int(B * tokens);
        const int H      = int(half_c);
        const int T      = int(tokens);
        const int C_int  = int(C);
        const cudaDataType_t dtype = x.cuda_dtype();
        void* xa_gpu   = fp.slots[layer][1].get_data();
        void* Q_gpu    = fp.slots[layer][2].get_data();
        void* K_gpu    = fp.slots[layer][3].get_data();
        void* Attn_gpu = fp.slots[layer][4].get_data();
        void* V_gpu    = fp.slots[layer][5].get_data();
        void* cat_gpu  = fp.slots[layer][6].get_data();
        void* out_gpu  = output.get_data();

        void* attn_v_gpu = fp.slots[layer][forward_scratch_slot].get_data();

        c2psa_split_cuda(x.get_data(), xa_gpu, cat_gpu, BT, C_int, H, dtype);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, BT, H,
            Wq.get_data(), dtype, H, 0LL,
            xa_gpu,  dtype, H, 0LL,
            Q_gpu,   dtype, H, 0LL, 1);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, BT, H,
            Wk.get_data(), dtype, H, 0LL,
            xa_gpu,  dtype, H, 0LL,
            K_gpu,   dtype, H, 0LL, 1);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, BT, H,
            Wv.get_data(), dtype, H, 0LL,
            xa_gpu,  dtype, H, 0LL,
            V_gpu,   dtype, H, 0LL, 1);

        gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, H,
            K_gpu,    dtype, H, (long long)T * H,
            Q_gpu,    dtype, H, (long long)T * H,
            Attn_gpu, dtype, T, (long long)T * T,
            int(B), scale);

        c2psa_row_softmax_cuda(Attn_gpu, BT, T, dtype);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, T, T,
            V_gpu,       dtype, H, (long long)T * H,
            Attn_gpu,    dtype, T, (long long)T * T,
            attn_v_gpu,  dtype, H, (long long)T * H,
            int(B));

        c2psa_fill_cat_left_cuda(attn_v_gpu, cat_gpu, BT, C_int, H, dtype);

        return gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
                   C_int, BT, C_int,
                   Wout.get_data(), dtype, C_int, 0LL,
                   cat_gpu,   dtype, C_int, 0LL,
                   out_gpu,   dtype, C_int, 0LL, 1);
    }
#endif

    const float* x_ptr = x.as<float>();
    float* xa  = fp.slots[layer][1].as<float>();
    float* Q   = fp.slots[layer][2].as<float>();
    float* K   = fp.slots[layer][3].as<float>();
    float* A   = fp.slots[layer][4].as<float>();
    float* V   = fp.slots[layer][5].as<float>();
    float* cat = fp.slots[layer][6].as<float>();

    const MatrixMap Wq_m   = Wq.as_matrix();
    const MatrixMap Wk_m   = Wk.as_matrix();
    const MatrixMap Wv_m   = Wv.as_matrix();
    const MatrixMap Wout_m = Wout.as_matrix();

    for (Index b = 0; b < B; ++b)
    {
        MapC x_b (x_ptr + b * tokens * C,     tokens, C);
        MapM xa_b(xa   + b * tokens * half_c, tokens, half_c);
        xa_b = x_b.leftCols(half_c);

        MapM Q_b(Q + b * tokens * half_c, tokens, half_c);
        MapM K_b(K + b * tokens * half_c, tokens, half_c);
        MapM V_b(V + b * tokens * half_c, tokens, half_c);
        Q_b.noalias() = xa_b * Wq_m;
        K_b.noalias() = xa_b * Wk_m;
        V_b.noalias() = xa_b * Wv_m;

        float* A_b_ptr = A + b * tokens * tokens;
        MapM   A_b(A_b_ptr, tokens, tokens);
        A_b.noalias() = Q_b * K_b.transpose() * scale;

        for (Index i = 0; i < tokens; ++i)
        {
            Eigen::Map<Eigen::ArrayXf> row(A_b_ptr + i * tokens, tokens);
            row -= row.maxCoeff();
            row  = row.exp();
            row /= row.sum();
        }

        MapM cat_b(cat + b * tokens * C, tokens, C);
        cat_b.leftCols(half_c).noalias() = A_b * V_b;
        cat_b.rightCols(half_c) = x_b.rightCols(half_c);
    }

    MapC cat_m(cat, B * tokens, C);
    MatrixMap out_m = output.as_flat_matrix();
    out_m.noalias() = cat_m * Wout_m;
}

void C2PSAOperator::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    const TensorView& x    = get_input(fp, layer);
    const Index B          = x.get_shape()[0];
    const Index tokens     = Index(h) * Index(w);
    const Index C          = channels;
    const Index half_c     = C / 2;
    const float scale      = 1.0f / sqrtf(float(half_c));

    const TensorView& delta_out = get_output_delta(bp, layer);
    TensorView&       delta_in  = get_input_delta(bp, layer);

#ifdef OPENNN_HAS_CUDA
    if (x.is_cuda())
    {
        const int BT     = int(B * tokens);
        const int H      = int(half_c);
        const int T      = int(tokens);
        const int C_int  = int(C);
        const cudaDataType_t dtype = x.cuda_dtype();
        const Index esz  = (dtype == CUDA_R_32F) ? sizeof(float) : sizeof(uint16_t);

        const void* xa_gpu   = fp.slots[layer][1].get_data();
        const void* Q_gpu    = fp.slots[layer][2].get_data();
        const void* K_gpu    = fp.slots[layer][3].get_data();
        const void* Attn_gpu = fp.slots[layer][4].get_data();
        const void* V_gpu    = fp.slots[layer][5].get_data();
        const void* cat_gpu  = fp.slots[layer][6].get_data();

        uint8_t* scratch = static_cast<uint8_t*>(
            bp.slots[layer][backward_scratch_slot].get_data());
        void* compact_d_ao = scratch;
        void* d_cat_gpu    = scratch + (size_t)BT * H    * esz;
        void* d_A_gpu      = scratch + (size_t)BT * (H + C_int) * esz;
        void* dQ_gpu       = scratch + (size_t)BT * (H + C_int + T) * esz;
        void* dK_gpu       = scratch + (size_t)BT * (H + C_int + T + H) * esz;
        void* dV_gpu       = scratch + (size_t)BT * (H + C_int + T + H * 2) * esz;
        void* d_xa_gpu     = scratch + (size_t)BT * (H + C_int + T + H * 3) * esz;

        const void* d_out_gpu = delta_out.get_data();
        void*       din_gpu   = delta_in.get_data();

        gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
            C_int, BT, C_int,
            Wout.get_data(), dtype, C_int, 0LL,
            d_out_gpu, dtype, C_int, 0LL,
            d_cat_gpu, dtype, C_int, 0LL, 1);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            C_int, C_int, BT,
            d_out_gpu, dtype, C_int, 0LL,
            cat_gpu,   dtype, C_int, 0LL,
            dWout.get_data(), dtype, C_int, 0LL,
            1, 1.0f, 1.0f);

        c2psa_gather_left_cuda(d_cat_gpu, compact_d_ao, BT, C_int, H, dtype);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, T, T,
            compact_d_ao, dtype, H, (long long)T * H,
            Attn_gpu,     dtype, T, (long long)T * T,
            dV_gpu,       dtype, H, (long long)T * H,
            int(B));

        gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, H,
            V_gpu,        dtype, H, (long long)T * H,
            compact_d_ao, dtype, H, (long long)T * H,
            d_A_gpu,      dtype, T, (long long)T * T,
            int(B));

        c2psa_softmax_bwd_cuda(Attn_gpu, d_A_gpu, scale, BT, T, dtype);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, T, T,
            K_gpu,   dtype, H, (long long)T * H,
            d_A_gpu, dtype, T, (long long)T * T,
            dQ_gpu,  dtype, H, (long long)T * H,
            int(B));

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, T, T,
            Q_gpu,   dtype, H, (long long)T * H,
            d_A_gpu, dtype, T, (long long)T * T,
            dK_gpu,  dtype, H, (long long)T * H,
            int(B));

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, H, BT,
            dQ_gpu,   dtype, H, 0LL,
            xa_gpu,   dtype, H, 0LL,
            dWq.get_data(), dtype, H, 0LL,
            1, 1.0f, 1.0f);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, H, BT,
            dK_gpu,   dtype, H, 0LL,
            xa_gpu,   dtype, H, 0LL,
            dWk.get_data(), dtype, H, 0LL,
            1, 1.0f, 1.0f);

        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, H, BT,
            dV_gpu,   dtype, H, 0LL,
            xa_gpu,   dtype, H, 0LL,
            dWv.get_data(), dtype, H, 0LL,
            1, 1.0f, 1.0f);

        if (din_gpu)
        {

            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                H, BT, H,
                Wq.get_data(), dtype, H, 0LL,
                dQ_gpu,  dtype, H, 0LL,
                d_xa_gpu, dtype, H, 0LL,
                1, 1.0f, 0.0f);
            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                H, BT, H,
                Wk.get_data(), dtype, H, 0LL,
                dK_gpu,  dtype, H, 0LL,
                d_xa_gpu, dtype, H, 0LL,
                1, 1.0f, 1.0f);
            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                H, BT, H,
                Wv.get_data(), dtype, H, 0LL,
                dV_gpu,  dtype, H, 0LL,
                d_xa_gpu, dtype, H, 0LL,
                1, 1.0f, 1.0f);

            c2psa_scatter_dx_cuda(d_xa_gpu, d_cat_gpu, din_gpu, BT, C_int, H, dtype);
        }
        return;
    }
#endif

    const float* xa  = fp.slots[layer][1].as<float>();
    const float* Q   = fp.slots[layer][2].as<float>();
    const float* K   = fp.slots[layer][3].as<float>();
    const float* A   = fp.slots[layer][4].as<float>();
    const float* V   = fp.slots[layer][5].as<float>();
    const float* cat = fp.slots[layer][6].as<float>();

    const float* dout = delta_out.as<float>();
    float*       din  = delta_in.get_data() ? delta_in.as<float>() : nullptr;

    const MatrixMap Wq_m   = Wq.as_matrix();
    const MatrixMap Wk_m   = Wk.as_matrix();
    const MatrixMap Wv_m   = Wv.as_matrix();
    const MatrixMap Wout_m = Wout.as_matrix();
    MatrixMap dWq_m        = dWq.as_matrix();
    MatrixMap dWk_m        = dWk.as_matrix();
    MatrixMap dWv_m        = dWv.as_matrix();
    MatrixMap dWout_m      = dWout.as_matrix();

    MapC cat_m (cat,  B * tokens, C);
    MapC dout_m(dout, B * tokens, C);

    dWout_m.noalias() += cat_m.transpose() * dout_m;

    MatF d_concat(B * tokens, C);
    d_concat.noalias() = dout_m * Wout_m.transpose();

    for (Index b = 0; b < B; ++b)
    {
        MapC Q_b (Q  + b * tokens * half_c, tokens, half_c);
        MapC K_b (K  + b * tokens * half_c, tokens, half_c);
        MapC V_b (V  + b * tokens * half_c, tokens, half_c);
        MapC A_b (A  + b * tokens * tokens, tokens, tokens);
        MapC xa_b(xa + b * tokens * half_c, tokens, half_c);

        MatF d_ao = d_concat.block(b * tokens, 0, tokens, half_c);

        MatF dV(tokens, half_c);
        dV.noalias() = A_b.transpose() * d_ao;

        MatF dA(tokens, tokens);
        dA.noalias() = d_ao * V_b.transpose();

        for (Index i = 0; i < tokens; ++i)
        {
            const float dot = A_b.row(i).dot(dA.row(i));
            dA.row(i) = (A_b.row(i).array() * (dA.row(i).array() - dot)).matrix();
        }
        dA *= scale;

        MatF dQ(tokens, half_c), dK(tokens, half_c);
        dQ.noalias() = dA            * K_b;
        dK.noalias() = dA.transpose() * Q_b;

        dWq_m.noalias() += xa_b.transpose() * dQ;
        dWk_m.noalias() += xa_b.transpose() * dK;
        dWv_m.noalias() += xa_b.transpose() * dV;

        if (din)
        {
            MapM din_m(din, B * tokens, C);
            din_m.block(b * tokens, 0, tokens, half_c).noalias() =
                dQ * Wq_m.transpose() + dK * Wk_m.transpose() + dV * Wv_m.transpose();
            din_m.block(b * tokens, half_c, tokens, half_c) =
                d_concat.block(b * tokens, half_c, tokens, half_c);
        }
    }
}

} // namespace opennn

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
