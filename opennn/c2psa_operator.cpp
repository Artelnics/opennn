//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   O P E R A T O R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "c2psa_operator.h"
#include "device_backend.h"
#include "random_utilities.h"
#include "forward_propagation.h"
#include "back_propagation.h"

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

namespace opennn
{

// Declared in kernel_c2psa.cu
#ifdef OPENNN_HAS_CUDA
void c2psa_split_cuda(const void* x, void* xa, void* cat, int BT, int C, int H, cudaDataType_t dtype);
void c2psa_fill_cat_left_cuda(const void* attn_v, void* cat, int BT, int C, int H, cudaDataType_t dtype);
void c2psa_row_softmax_cuda(void* A, int rows, int T, cudaDataType_t dtype);
void c2psa_softmax_bwd_cuda(const void* A, void* dA, float scale, int rows, int T, cudaDataType_t dtype);
void c2psa_scatter_dx_cuda(const void* d_xa, const void* d_cat, void* din, int BT, int C, int H, cudaDataType_t dtype);
#endif

using Eigen::Dynamic;
using MatF = Eigen::Matrix<float, Dynamic, Dynamic, Eigen::RowMajor>;
using MapC = Eigen::Map<const MatF>;
using MapM = Eigen::Map<MatF>;

void C2PSAOperator::set(Index new_h, Index new_w, Index new_channels)
{
    h        = new_h;
    w        = new_w;
    channels = new_channels;
    output_slots = {7};   // 7 specs → forward_slots[layer][7] is the output
}

vector<TensorSpec> C2PSAOperator::parameter_specs() const
{
    if (channels == 0) return {};
    const Index half_c = channels / 2;
    return {
        {{half_c, half_c}, compute_dtype},     // Wq
        {{half_c, half_c}, compute_dtype},     // Wk
        {{half_c, half_c}, compute_dtype},     // Wv
        {{channels, channels}, compute_dtype}, // Wout
    };
}

void C2PSAOperator::link_parameters(span<const TensorView> views)
{
    if (views.size() < 4) return;
    Wq   = views[0];
    Wk   = views[1];
    Wv   = views[2];
    Wout = views[3];
}

void C2PSAOperator::link_gradients(span<const TensorView> views)
{
    if (views.size() < 4) return;
    dWq   = views[0];
    dWk   = views[1];
    dWv   = views[2];
    dWout = views[3];
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
    TensorView& output  = get_output(fp, layer);   // forward_slots[layer][7]

    const Index B      = x.shape[0];
    const Index tokens = Index(h) * Index(w);
    const Index C      = channels;
    const Index half_c = C / 2;
    const float scale  = 1.0f / sqrtf(float(half_c));

#ifdef OPENNN_HAS_CUDA
    if (x.device == Device::CUDA)
    {
        const int BT     = int(B * tokens);
        const int H      = int(half_c);
        const int T      = int(tokens);
        const int C_int  = int(C);
        const cudaDataType_t dtype = x.cuda_dtype();
        const Index esz  = (dtype == CUDA_R_32F) ? sizeof(float) : sizeof(uint16_t);

        void* xa_gpu   = fp.forward_slots[layer][1].data;
        void* Q_gpu    = fp.forward_slots[layer][2].data;
        void* K_gpu    = fp.forward_slots[layer][3].data;
        void* Attn_gpu = fp.forward_slots[layer][4].data;
        void* V_gpu    = fp.forward_slots[layer][5].data;
        void* cat_gpu  = fp.forward_slots[layer][6].data;
        void* out_gpu  = output.data;

        // attn_v scratch [BT, H] at offset 0; backward temps follow.
        const Index scratch_needed = (Index(BT) * H
                                    + Index(BT) * C_int
                                    + Index(BT) * T
                                    + Index(BT) * H * 4) * esz;
        gpu_scratch.resize_bytes(scratch_needed, Device::CUDA);
        void* attn_v_gpu = gpu_scratch.data;

        // 1. Split x → xa (left half), fill cat right half (identity path)
        c2psa_split_cuda(x.data, xa_gpu, cat_gpu, BT, C_int, H, dtype);

        // 2. Q = xa @ Wq  [BT,H] = [BT,H] @ [H,H]
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, BT, H,
            Wq.data, dtype, H, 0LL,
            xa_gpu,  dtype, H, 0LL,
            Q_gpu,   dtype, H, 0LL, 1);

        // 3. K = xa @ Wk
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, BT, H,
            Wk.data, dtype, H, 0LL,
            xa_gpu,  dtype, H, 0LL,
            K_gpu,   dtype, H, 0LL, 1);

        // 4. V = xa @ Wv
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, BT, H,
            Wv.data, dtype, H, 0LL,
            xa_gpu,  dtype, H, 0LL,
            V_gpu,   dtype, H, 0LL, 1);

        // 5. Attn[b] = Q[b] @ K[b]^T * scale  → [B,T,T]
        gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, H,
            K_gpu,    dtype, H, (long long)T * H,
            Q_gpu,    dtype, H, (long long)T * H,
            Attn_gpu, dtype, T, (long long)T * T,
            int(B), scale);

        // 6. Row-wise softmax on Attn[B*T, T]
        c2psa_row_softmax_cuda(Attn_gpu, BT, T, dtype);

        // 7. attn_v[b] = Attn[b] @ V[b]  → scratch [BT,H]
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, T, T,
            V_gpu,       dtype, H, (long long)T * H,
            Attn_gpu,    dtype, T, (long long)T * T,
            attn_v_gpu,  dtype, H, (long long)T * H,
            int(B));

        // 8. Merge attn_v into left half of cat
        c2psa_fill_cat_left_cuda(attn_v_gpu, cat_gpu, BT, C_int, H, dtype);

        // 9. output = cat @ Wout  [BT,C] = [BT,C] @ [C,C]
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            C_int, BT, C_int,
            Wout.data, dtype, C_int, 0LL,
            cat_gpu,   dtype, C_int, 0LL,
            out_gpu,   dtype, C_int, 0LL, 1);
        return;
    }
#endif

    // CPU path: compute directly in the live forward slots.
    const float* x_ptr = x.as<float>();
    float* xa  = fp.forward_slots[layer][1].as<float>();
    float* Q   = fp.forward_slots[layer][2].as<float>();
    float* K   = fp.forward_slots[layer][3].as<float>();
    float* A   = fp.forward_slots[layer][4].as<float>();
    float* V   = fp.forward_slots[layer][5].as<float>();
    float* cat = fp.forward_slots[layer][6].as<float>();

    MapC Wq_m  (Wq  .as<float>(), half_c, half_c);
    MapC Wk_m  (Wk  .as<float>(), half_c, half_c);
    MapC Wv_m  (Wv  .as<float>(), half_c, half_c);
    MapC Wout_m(Wout.as<float>(), C, C);

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
    MapM out_m(output.as<float>(), B * tokens, C);
    out_m.noalias() = cat_m * Wout_m;
}

void C2PSAOperator::back_propagate(ForwardPropagation& fp, BackPropagation& bp, size_t layer) const
{
    const TensorView& x    = get_input(fp, layer);
    const Index B          = x.shape[0];
    const Index tokens     = Index(h) * Index(w);
    const Index C          = channels;
    const Index half_c     = C / 2;
    const float scale      = 1.0f / sqrtf(float(half_c));

    const TensorView& delta_out = get_output_delta(bp, layer);
    TensorView&       delta_in  = get_input_delta(bp, layer);

#ifdef OPENNN_HAS_CUDA
    if (x.device == Device::CUDA)
    {
        const int BT     = int(B * tokens);
        const int H      = int(half_c);
        const int T      = int(tokens);
        const int C_int  = int(C);
        const cudaDataType_t dtype = x.cuda_dtype();
        const Index esz  = (dtype == CUDA_R_32F) ? sizeof(float) : sizeof(uint16_t);

        // Forward slots still hold Q, K, Attn, V, cat from the forward pass.
        const void* xa_gpu   = fp.forward_slots[layer][1].data;
        const void* Q_gpu    = fp.forward_slots[layer][2].data;
        const void* K_gpu    = fp.forward_slots[layer][3].data;
        const void* Attn_gpu = fp.forward_slots[layer][4].data;
        const void* V_gpu    = fp.forward_slots[layer][5].data;
        const void* cat_gpu  = fp.forward_slots[layer][6].data;

        // Scratch layout: [attn_v BT*H] [d_cat BT*C] [d_A BT*T] [dQ BT*H] [dK BT*H] [dV BT*H] [d_xa BT*H]
        // (Offsets in elements; multiply by esz for byte offsets.)
        uint8_t* scratch = static_cast<uint8_t*>(gpu_scratch.data);
        void* compact_d_ao = scratch;                                          // [BT, H]  (reuse attn_v slot)
        void* d_cat_gpu    = scratch + (size_t)BT * H    * esz;
        void* d_A_gpu      = scratch + (size_t)BT * (H + C_int) * esz;
        void* dQ_gpu       = scratch + (size_t)BT * (H + C_int + T) * esz;
        void* dK_gpu       = scratch + (size_t)BT * (H + C_int + T + H) * esz;
        void* dV_gpu       = scratch + (size_t)BT * (H + C_int + T + H * 2) * esz;
        void* d_xa_gpu     = scratch + (size_t)BT * (H + C_int + T + H * 3) * esz;

        const void* d_out_gpu = delta_out.data;
        void*       din_gpu   = delta_in.data;

        // 1. d_cat = d_out @ Wout^T  [BT,C] = [BT,C] @ [C,C]^T
        gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
            C_int, BT, C_int,
            Wout.data, dtype, C_int, 0LL,
            d_out_gpu, dtype, C_int, 0LL,
            d_cat_gpu, dtype, C_int, 0LL, 1);

        // 2. dWout += cat^T @ d_out  [C,C] += [BT,C]^T @ [BT,C]  (beta=1 to accumulate)
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            C_int, C_int, BT,
            d_out_gpu, dtype, C_int, 0LL,
            cat_gpu,   dtype, C_int, 0LL,
            dWout.data, dtype, C_int, 0LL,
            1, 1.0f, 1.0f);

        // d_cat has stride C; gemms below need compact [BT,H] view of its left half.
        // c2psa_split_cuda extracts d_cat[:,0:H] → compact_d_ao. The right-half write
        // into d_cat is a no-op here (same pointer), which is fine.
        c2psa_split_cuda(d_cat_gpu, compact_d_ao, d_cat_gpu, BT, C_int, H, dtype);

        // 3. dV[b] += Attn[b]^T @ d_ao[b]  [T,H] += [T,T]^T @ [T,H]
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, T, T,
            compact_d_ao, dtype, H, (long long)T * H,
            Attn_gpu,     dtype, T, (long long)T * T,
            dV_gpu,       dtype, H, (long long)T * H,
            int(B));

        // 4. d_A[b] = d_ao[b] @ V[b]^T  [T,T] = [T,H] @ [T,H]^T
        gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, H,
            V_gpu,        dtype, H, (long long)T * H,
            compact_d_ao, dtype, H, (long long)T * H,
            d_A_gpu,      dtype, T, (long long)T * T,
            int(B));

        // 5. Softmax backward + scale in-place on d_A[B*T, T]
        c2psa_softmax_bwd_cuda(Attn_gpu, d_A_gpu, scale, BT, T, dtype);

        // 6. dQ[b] = d_A_scaled[b] @ K[b]  [T,H] = [T,T] @ [T,H]
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_N,
            H, T, T,
            K_gpu,   dtype, H, (long long)T * H,
            d_A_gpu, dtype, T, (long long)T * T,
            dQ_gpu,  dtype, H, (long long)T * H,
            int(B));

        // 7. dK[b] = d_A_scaled[b]^T @ Q[b]  [T,H] = [T,T]^T @ [T,H]
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, T, T,
            Q_gpu,   dtype, H, (long long)T * H,
            d_A_gpu, dtype, T, (long long)T * T,
            dK_gpu,  dtype, H, (long long)T * H,
            int(B));

        // 8. dWq += xa^T @ dQ  [H,H] += [BT,H]^T @ [BT,H]  (beta=1)
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, H, BT,
            dQ_gpu,   dtype, H, 0LL,
            xa_gpu,   dtype, H, 0LL,
            dWq.data, dtype, H, 0LL,
            1, 1.0f, 1.0f);

        // 9. dWk += xa^T @ dK  (beta=1)
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, H, BT,
            dK_gpu,   dtype, H, 0LL,
            xa_gpu,   dtype, H, 0LL,
            dWk.data, dtype, H, 0LL,
            1, 1.0f, 1.0f);

        // 10. dWv += xa^T @ dV  (beta=1)
        gemm_strided_batched_cuda(CUBLAS_OP_N, CUBLAS_OP_T,
            H, H, BT,
            dV_gpu,   dtype, H, 0LL,
            xa_gpu,   dtype, H, 0LL,
            dWv.data, dtype, H, 0LL,
            1, 1.0f, 1.0f);

        if (din_gpu)
        {
            // 11. d_xa = dQ @ Wq^T + dK @ Wk^T + dV @ Wv^T  [BT,H]
            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                H, BT, H,
                Wq.data, dtype, H, 0LL,
                dQ_gpu,  dtype, H, 0LL,
                d_xa_gpu, dtype, H, 0LL,
                1, 1.0f, 0.0f);
            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                H, BT, H,
                Wk.data, dtype, H, 0LL,
                dK_gpu,  dtype, H, 0LL,
                d_xa_gpu, dtype, H, 0LL,
                1, 1.0f, 1.0f);
            gemm_strided_batched_cuda(CUBLAS_OP_T, CUBLAS_OP_N,
                H, BT, H,
                Wv.data, dtype, H, 0LL,
                dV_gpu,  dtype, H, 0LL,
                d_xa_gpu, dtype, H, 0LL,
                1, 1.0f, 1.0f);

            // 12. Scatter: din[:, :H] = d_xa, din[:, H:] = d_cat[:, H:]
            c2psa_scatter_dx_cuda(d_xa_gpu, d_cat_gpu, din_gpu, BT, C_int, H, dtype);
        }
        return;
    }
#endif

    // CPU path: read the live forward slots and accumulate into the live gradients.
    const float* xa  = fp.forward_slots[layer][1].as<float>();
    const float* Q   = fp.forward_slots[layer][2].as<float>();
    const float* K   = fp.forward_slots[layer][3].as<float>();
    const float* A   = fp.forward_slots[layer][4].as<float>();
    const float* V   = fp.forward_slots[layer][5].as<float>();
    const float* cat = fp.forward_slots[layer][6].as<float>();

    const float* dout = delta_out.as<float>();
    float*       din  = delta_in.data ? delta_in.as<float>() : nullptr;

    MapC Wq_m  (Wq  .as<float>(), half_c, half_c);
    MapC Wk_m  (Wk  .as<float>(), half_c, half_c);
    MapC Wv_m  (Wv  .as<float>(), half_c, half_c);
    MapC Wout_m(Wout.as<float>(), C, C);
    MapM dWq_m  (dWq  .as<float>(), half_c, half_c);
    MapM dWk_m  (dWk  .as<float>(), half_c, half_c);
    MapM dWv_m  (dWv  .as<float>(), half_c, half_c);
    MapM dWout_m(dWout.as<float>(), C, C);

    MapC cat_m (cat,  B * tokens, C);
    MapC dout_m(dout, B * tokens, C);

    // dL/dWout += concat.T @ delta_out
    dWout_m.noalias() += cat_m.transpose() * dout_m;

    // delta_concat = delta_out @ Wout.T
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

        // Softmax backward: dA_ij = A_ij * (dA_ij - sum_k A_ik * dA_ik)
        for (Index i = 0; i < tokens; ++i)
        {
            float dot = 0.0f;
            for (Index j = 0; j < tokens; ++j)
                dot += A_b(i, j) * dA(i, j);
            for (Index j = 0; j < tokens; ++j)
                dA(i, j) = A_b(i, j) * (dA(i, j) - dot);
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
