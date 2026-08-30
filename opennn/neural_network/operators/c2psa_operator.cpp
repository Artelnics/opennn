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

// MatrixR was character-for-character core's MatrixR. The maps stay separate from
// core's MatrixMap, which is 64-byte aligned: these views start at arbitrary
// per-batch offsets inside the slot buffers and so have to be unaligned.
using ConstBlockMap = Eigen::Map<const MatrixR>;
using BlockMap      = Eigen::Map<MatrixR>;

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

vector<Operator::ParameterSlot> C2PSAOperator::parameter_slots()
{
    return {{&Wq, &dWq}, {&Wk, &dWk}, {&Wv, &dWv}, {&Wout, &dWout}};
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

void C2PSAOperator::forward_propagate(ForwardPropagation& fp, size_t layer, ForwardPropagationMode)
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
        void* xa_gpu   = fp.slots[layer][Split].get_data();
        void* Q_gpu    = fp.slots[layer][Query].get_data();
        void* K_gpu    = fp.slots[layer][Key].get_data();
        void* Attn_gpu = fp.slots[layer][AttentionWeights].get_data();
        void* V_gpu    = fp.slots[layer][Value].get_data();
        void* cat_gpu  = fp.slots[layer][Concatenated].get_data();
        void* out_gpu  = output.get_data();

        throw_if(!forward_scratch_slot,
                 "C2PSAOperator: forward scratch slot was not planned.");
        void* attn_v_gpu = fp.slots[layer][*forward_scratch_slot].get_data();

        c2psa_split_cuda(x.get_data(), xa_gpu, cat_gpu, BT, C_int, H, dtype);

        // One half-width square weight applied to every token. The three
        // projections differ only in the weight and the destination.
        const auto project = [&](const void* weight, void* destination)
        {
            gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
                H, BT, H,
                weight,      dtype, H, 0LL,
                xa_gpu,      dtype, H, 0LL,
                destination, dtype, H, 0LL, 1);
        };

        project(Wq.get_data(), Q_gpu);
        project(Wk.get_data(), K_gpu);
        project(Wv.get_data(), V_gpu);

        gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, H,
            K_gpu,    dtype, H, (long long)T * H,
            Q_gpu,    dtype, H, (long long)T * H,
            Attn_gpu, dtype, T, (long long)T * T,
            int(B), scale);

        softmax(fp.slots[layer][AttentionWeights]);

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
    float* xa  = fp.slots[layer][Split].as<float>();
    float* Q   = fp.slots[layer][Query].as<float>();
    float* K   = fp.slots[layer][Key].as<float>();
    float* A   = fp.slots[layer][AttentionWeights].as<float>();
    float* V   = fp.slots[layer][Value].as<float>();
    float* cat = fp.slots[layer][Concatenated].as<float>();

    const MatrixMap Wq_m   = Wq.as_matrix();
    const MatrixMap Wk_m   = Wk.as_matrix();
    const MatrixMap Wv_m   = Wv.as_matrix();
    const MatrixMap Wout_m = Wout.as_matrix();

    for (Index b = 0; b < B; ++b)
    {
        ConstBlockMap x_b (x_ptr + b * tokens * C,     tokens, C);
        BlockMap xa_b(xa   + b * tokens * half_c, tokens, half_c);
        xa_b = x_b.leftCols(half_c);

        BlockMap Q_b(Q + b * tokens * half_c, tokens, half_c);
        BlockMap K_b(K + b * tokens * half_c, tokens, half_c);
        BlockMap V_b(V + b * tokens * half_c, tokens, half_c);
        Q_b.noalias() = xa_b * Wq_m;
        K_b.noalias() = xa_b * Wk_m;
        V_b.noalias() = xa_b * Wv_m;

        float* A_b_ptr = A + b * tokens * tokens;
        BlockMap   A_b(A_b_ptr, tokens, tokens);
        A_b.noalias() = Q_b * K_b.transpose() * scale;

        // The GPU path normalizes through the shared softmax; this was an
        // inlined row loop doing the same max-subtract, exp, divide.
        TensorView scores(A_b_ptr, Shape{tokens, tokens}, x.get_type(), Device::CPU);
        softmax(scores);

        BlockMap cat_b(cat + b * tokens * C, tokens, C);
        cat_b.leftCols(half_c).noalias() = A_b * V_b;
        cat_b.rightCols(half_c) = x_b.rightCols(half_c);
    }

    ConstBlockMap cat_m(cat, B * tokens, C);
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

        const void* xa_gpu   = fp.slots[layer][Split].get_data();
        const void* Q_gpu    = fp.slots[layer][Query].get_data();
        const void* K_gpu    = fp.slots[layer][Key].get_data();
        const void* Attn_gpu = fp.slots[layer][AttentionWeights].get_data();
        const void* V_gpu    = fp.slots[layer][Value].get_data();
        const void* cat_gpu  = fp.slots[layer][Concatenated].get_data();

        throw_if(!backward_scratch_slot,
                 "C2PSAOperator: backward scratch slot was not planned.");
        uint8_t* scratch = static_cast<uint8_t*>(
            bp.slots[layer][*backward_scratch_slot].get_data());
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

        TensorView attention_delta(d_A_gpu, Shape{B, tokens, tokens},
                                   x.get_type(), Device::CUDA);
        softmax_backward(fp.slots[layer][AttentionWeights], attention_delta, scale);

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

        // Each weight gradient accumulates its head delta against the shared
        // split input; the three calls differ only in those two pointers.
        const auto accumulate_weight_gradient = [&](const void* head_delta, void* weight_gradient)
        {
            gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
                H, H, BT,
                head_delta,      dtype, H, 0LL,
                xa_gpu,          dtype, H, 0LL,
                weight_gradient, dtype, H, 0LL,
                1, 1.0f, 1.0f);
        };

        accumulate_weight_gradient(dQ_gpu, dWq.get_data());
        accumulate_weight_gradient(dK_gpu, dWk.get_data());
        accumulate_weight_gradient(dV_gpu, dWv.get_data());

        if (din_gpu)
        {
            // The three head deltas fold back through their weights into one
            // accumulator, so only the first call overwrites it.
            const auto accumulate_split_delta = [&](const void* weight, const void* head_delta, float beta)
            {
                gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                    H, BT, H,
                    weight,     dtype, H, 0LL,
                    head_delta, dtype, H, 0LL,
                    d_xa_gpu,   dtype, H, 0LL,
                    1, 1.0f, beta);
            };

            accumulate_split_delta(Wq.get_data(), dQ_gpu, 0.0f);
            accumulate_split_delta(Wk.get_data(), dK_gpu, 1.0f);
            accumulate_split_delta(Wv.get_data(), dV_gpu, 1.0f);

            c2psa_scatter_dx_cuda(d_xa_gpu, d_cat_gpu, din_gpu, BT, C_int, H, dtype);
        }
        return;
    }
#endif

    const float* xa  = fp.slots[layer][Split].as<float>();
    const float* Q   = fp.slots[layer][Query].as<float>();
    const float* K   = fp.slots[layer][Key].as<float>();
    const float* A   = fp.slots[layer][AttentionWeights].as<float>();
    const float* V   = fp.slots[layer][Value].as<float>();
    const float* cat = fp.slots[layer][Concatenated].as<float>();

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

    ConstBlockMap cat_m (cat,  B * tokens, C);
    ConstBlockMap dout_m(dout, B * tokens, C);

    dWout_m.noalias() += cat_m.transpose() * dout_m;

    MatrixR d_concat(B * tokens, C);
    d_concat.noalias() = dout_m * Wout_m.transpose();

    // These were declared inside the loop: five heap allocations per image on
    // every backward pass, all of them the same shape every time round.
    MatrixR d_ao(tokens, half_c);
    MatrixR dV(tokens, half_c);
    MatrixR dA(tokens, tokens);
    MatrixR dQ(tokens, half_c);
    MatrixR dK(tokens, half_c);

    for (Index b = 0; b < B; ++b)
    {
        ConstBlockMap Q_b (Q  + b * tokens * half_c, tokens, half_c);
        ConstBlockMap K_b (K  + b * tokens * half_c, tokens, half_c);
        ConstBlockMap V_b (V  + b * tokens * half_c, tokens, half_c);
        ConstBlockMap A_b (A  + b * tokens * tokens, tokens, tokens);
        ConstBlockMap xa_b(xa + b * tokens * half_c, tokens, half_c);

        d_ao = d_concat.block(b * tokens, 0, tokens, half_c);

        dV.noalias() = A_b.transpose() * d_ao;

        dA.noalias() = d_ao * V_b.transpose();

        for (Index i = 0; i < tokens; ++i)
        {
            const float dot = A_b.row(i).dot(dA.row(i));
            dA.row(i) = (A_b.row(i).array() * (dA.row(i).array() - dot)).matrix();
        }
        dA *= scale;

        dQ.noalias() = dA            * K_b;
        dK.noalias() = dA.transpose() * Q_b;

        dWq_m.noalias() += xa_b.transpose() * dQ;
        dWk_m.noalias() += xa_b.transpose() * dK;
        dWv_m.noalias() += xa_b.transpose() * dV;

        if (din)
        {
            BlockMap din_m(din, B * tokens, C);
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
