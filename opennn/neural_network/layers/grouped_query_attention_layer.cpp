//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O U P E D   Q U E R Y   A T T E N T I O N   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/layers/grouped_query_attention_layer.h"

#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "opennn/core/tensor_operations.h"
#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/registry.h"

#ifdef OPENNN_HAS_CUDA
#include "opennn/core/device_backend.h"
#include "opennn/core/cuda/cudnn_frontend_utilities.h"
#include "opennn/core/cuda/kernel_attention.cuh"
#include "opennn/core/cuda/kernel_normalization.cuh"
#endif

namespace opennn
{

// Defined below under OPENNN_HAS_CUDA.
static void grouped_attention_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, float, Index, float*, const int*);
static void qk_norm_gpu(const TensorView&, const TensorView&, TensorView&, Index, float);
static void rope_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index);
static void rope_backward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index);

void rotary_build_tables(TensorView& cos_table, TensorView& sin_table,
                         Index sequence_length, Index rotary_dim, float base)
{
    float* cos_data = cos_table.as<float>();
    float* sin_data = sin_table.as<float>();
    const Index half = rotary_dim / 2;

    #pragma omp parallel for schedule(static)
    for (Index pos = 0; pos < sequence_length; ++pos)
        for (Index i = 0; i < half; ++i)
        {
            const float inv_freq = 1.0f / powf(base, (2.0f * float(i)) / float(rotary_dim));
            const float angle    = float(pos) * inv_freq;
            const float c = cosf(angle);
            const float s = sinf(angle);
            cos_data[pos * rotary_dim + i]        = c;
            cos_data[pos * rotary_dim + i + half] = c;
            sin_data[pos * rotary_dim + i]        = s;
            sin_data[pos * rotary_dim + i + half] = s;
        }
}

static void rotary_forward_cpu(const TensorView& input, const TensorView& cos_table, const TensorView& sin_table,
                        TensorView& output, Index head_dim, Index rotary_dim, Index position_offset)
{
    const Index seq       = input.shape[1];
    const Index model_dim = input.shape.back();
    const Index num_heads = model_dim / head_dim;
    const Index rows      = input.size() / model_dim;
    const Index half      = rotary_dim / 2;

    const float* in       = input.as<float>();
    float* out            = output.as<float>();
    const float* cos_data = cos_table.as<float>();
    const float* sin_data = sin_table.as<float>();

    const bool parallel = rows * model_dim >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index row = 0; row < rows; ++row)
    {
        const Index pos = (row % seq) + position_offset;
        const float* cr = cos_data + pos * rotary_dim;
        const float* sr = sin_data + pos * rotary_dim;

        for (Index h = 0; h < num_heads; ++h)
        {
            const Index base = row * model_dim + h * head_dim;

            for (Index j = 0; j < rotary_dim; ++j)
            {
                const float rotated = (j < half) ? -in[base + j + half] : in[base + j - half];
                out[base + j] = in[base + j] * cr[j] + rotated * sr[j];
            }
            for (Index j = rotary_dim; j < head_dim; ++j)
                out[base + j] = in[base + j];
        }
    }
}

static void rotary_backward_cpu(const TensorView& output_delta, const TensorView& cos_table, const TensorView& sin_table,
                         TensorView& input_delta, Index head_dim, Index rotary_dim, Index position_offset)
{
    const Index seq       = output_delta.shape[1];
    const Index model_dim = output_delta.shape.back();
    const Index num_heads = model_dim / head_dim;
    const Index rows      = output_delta.size() / model_dim;
    const Index half      = rotary_dim / 2;

    const float* dout     = output_delta.as<float>();
    float* din            = input_delta.as<float>();
    const float* cos_data = cos_table.as<float>();
    const float* sin_data = sin_table.as<float>();

    const bool parallel = rows * model_dim >= 65536;

    #pragma omp parallel for schedule(static) if(parallel)
    for (Index row = 0; row < rows; ++row)
    {
        const Index pos = (row % seq) + position_offset;
        const float* cr = cos_data + pos * rotary_dim;
        const float* sr = sin_data + pos * rotary_dim;

        for (Index h = 0; h < num_heads; ++h)
        {
            const Index base = row * model_dim + h * head_dim;

            for (Index j = 0; j < rotary_dim; ++j)
            {
                const float rotated = (j < half) ? -dout[base + j + half] : dout[base + j - half];
                din[base + j] = dout[base + j] * cr[j] - rotated * sr[j];
            }
            for (Index j = rotary_dim; j < head_dim; ++j)
                din[base + j] = dout[base + j];
        }
    }
}

void rotary_forward(const TensorView& input, const TensorView& cos_table, const TensorView& sin_table,
                    TensorView& output, Index head_dim, Index rotary_dim, Index position_offset)
{
    if (input.is_cuda()) { rope_forward_gpu(input, cos_table, sin_table, output, head_dim, rotary_dim, position_offset); return; }
    rotary_forward_cpu(input, cos_table, sin_table, output, head_dim, rotary_dim, position_offset);
}

void rotary_backward(const TensorView& output_delta, const TensorView& cos_table, const TensorView& sin_table,
                     TensorView& input_delta, Index head_dim, Index rotary_dim, Index position_offset)
{
    if (output_delta.is_cuda()) { rope_backward_gpu(output_delta, cos_table, sin_table, input_delta, head_dim, rotary_dim, position_offset); return; }
    rotary_backward_cpu(output_delta, cos_table, sin_table, input_delta, head_dim, rotary_dim, position_offset);
}

void grouped_attention_forward(const TensorView& query, const TensorView& key, const TensorView& value,
                               TensorView& output, Index n_query_heads, Index n_kv_heads, Index head_dim,
                               bool causal, float scale, Index query_position_offset,
                               float* decode_partials, const int* position_device)
{
    if (query.is_cuda()) {
        grouped_attention_gpu(query, key, value, output, n_query_heads, n_kv_heads, head_dim,
                              causal, scale, query_position_offset, decode_partials, position_device);
        return;
    }

    const Index batch     = query.shape[0];
    const Index query_seq = query.shape[1];
    const Index key_seq   = key.shape[1];
    const Index group     = n_query_heads / n_kv_heads;

    const float* Q = query.as<float>();
    const float* K = key.as<float>();
    const float* V = value.as<float>();
    float* O       = output.as<float>();

    const auto q_off = [&](Index b, Index t, Index h) {
        return ((b * query_seq + t) * n_query_heads + h) * head_dim;
    };
    const auto kv_off = [&](Index b, Index t, Index h) {
        return ((b * key_seq + t) * n_kv_heads + h) * head_dim;
    };

    const auto calculate_weights = [&](Index b, Index i, Index hq, Index hkv,
                                       Index valid, vector<float>& scores) {
        const Map<const VectorR> q_map(Q + q_off(b, i, hq), head_dim);

        float max_score = NEG_INFINITY;
        for (Index j = 0; j < valid; ++j)
        {
            const float dot =
                q_map.dot(Map<const VectorR>(K + kv_off(b, j, hkv), head_dim)) * scale;
            scores[size_t(j)] = dot;
            max_score = max(max_score, dot);
        }

        Map<Array<float, Dynamic, 1>> score_map(scores.data(), valid);
        score_map = (score_map - max_score).exp();
        return 1.0f / score_map.sum();
    };

    const auto write_output = [&](Index b, Index i, Index hq, Index hkv,
                                  Index valid, const vector<float>& scores, float inv_sum) {
        Map<VectorR> o_map(O + q_off(b, i, hq), head_dim);
        o_map.setZero();
        for (Index j = 0; j < valid; ++j)
            o_map += (scores[size_t(j)] * inv_sum)
                   * Map<const VectorR>(V + kv_off(b, j, hkv), head_dim);
    };

    const auto attend_head = [&](Index b, Index hq) {
        const Index hkv = hq / group;

        thread_local vector<float> scores;
        if (scores.size() < size_t(key_seq))
            scores.resize(size_t(key_seq));

        for (Index i = 0; i < query_seq; ++i)
        {
            const Index valid = causal ? min(query_position_offset + i + 1, key_seq) : key_seq;
            const float inv_sum = calculate_weights(b, i, hq, hkv, valid, scores);
            write_output(b, i, hq, hkv, valid, scores, inv_sum);
        }
    };

    const Index heads_count = batch * n_query_heads;

    #pragma omp parallel for schedule(static)
    for (Index head = 0; head < heads_count; ++head)
        attend_head(head / n_query_heads, head % n_query_heads);
}

void qk_norm_forward(const TensorView& input, const TensorView& weight, TensorView& output,
                     Index head_dim, float epsilon)
{
    if (input.is_cuda()) { qk_norm_gpu(input, weight, output, head_dim, epsilon); return; }

    const Index rows  = input.size() / head_dim;
    const float inv_D = 1.0f / to_type(head_dim);

    const float* x = input.as<float>();
    float* o       = output.as<float>();
    const float* w = weight.as<float>();

    #pragma omp parallel for schedule(static)
    for (Index r = 0; r < rows; ++r)
    {
        const float* x_row = x + r * head_dim;
        float* o_row       = o + r * head_dim;

        const Map<const Array<float, Dynamic, 1>> x_map(x_row, head_dim);
        const float mean_square = x_map.square().sum() * inv_D;
        const float inverse     = 1.0f / sqrt(mean_square + epsilon);

        Map<Array<float, Dynamic, 1>>(o_row, head_dim) =
            Map<const Array<float, Dynamic, 1>>(w, head_dim) * x_map * inverse;
    }
}

#ifdef OPENNN_HAS_CUDA

static void rope_forward_gpu(const TensorView& input, const TensorView& cos_table, const TensorView& sin_table,
                             TensorView& output, Index head_dim, Index rotary_dim, Index position_offset)
{
    const int seq       = to_int(input.shape[1]);
    const int model_dim = to_int(input.shape.back());
    const int rows      = to_int(input.size() / input.shape.back());

    output.dispatch([&]<typename T>() {
        rope_forward_cuda<T>(rows, seq, model_dim, to_int(head_dim), to_int(rotary_dim), to_int(position_offset),
                             input.as<T>(), output.as<T>(),
                             cos_table.as<float>(), sin_table.as<float>());
    });
}

static void rope_backward_gpu(const TensorView& output_delta, const TensorView& cos_table, const TensorView& sin_table,
                              TensorView& input_delta, Index head_dim, Index rotary_dim, Index position_offset)
{
    const int seq       = to_int(output_delta.shape[1]);
    const int model_dim = to_int(output_delta.shape.back());
    const int rows      = to_int(output_delta.size() / output_delta.shape.back());

    input_delta.dispatch([&]<typename T>() {
        rope_backward_cuda<T>(rows, seq, model_dim, to_int(head_dim), to_int(rotary_dim), to_int(position_offset),
                              output_delta.as<T>(), input_delta.as<T>(),
                              cos_table.as<float>(), sin_table.as<float>());
    });
}

Index grouped_attention_decode_scratch_floats(Index n_query_heads, Index head_dim)
{
    return n_query_heads * GROUPED_ATTENTION_DECODE_SPLITS * (head_dim + 2);
}

static cublasHandle_t grouped_attention_cublas()
{
    thread_local Buffer cublas_workspace{Device::CUDA};
    thread_local cublasHandle_t handle = nullptr;
    if (!handle)
    {
        constexpr Index workspace_bytes = Index(4) << 20;
        cublas_workspace.grow_to(workspace_bytes);
        CHECK_CUBLAS(cublasCreate(&handle));
        CHECK_CUBLAS(cublasSetStream(handle, device::get_compute_stream()));
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        CHECK_CUBLAS(cublasSetWorkspace(handle, cublas_workspace.as<char>(), size_t(workspace_bytes)));
    }
    return handle;
}

static void grouped_attention_gemm(cublasOperation_t transa, cublasOperation_t transb,
                                   int m, int n, int k, float alpha,
                                   const void* A, cudaDataType_t a_type, int lda, long long stride_a,
                                   const void* B, cudaDataType_t b_type, int ldb, long long stride_b,
                                   void* C, cudaDataType_t c_type, int ldc, long long stride_c,
                                   int batch_count)
{
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(grouped_attention_cublas(),
                                            transa, transb, m, n, k,
                                            &alpha,
                                            A, a_type, lda, stride_a,
                                            B, b_type, ldb, stride_b,
                                            &beta,
                                            C, c_type, ldc, stride_c,
                                            batch_count,
                                             CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}

template<typename T>
static void zero_grouped_attention_value_tail(T* values, int batch_heads,
                                              int key_seq, int valid_key_seq, int head_dim)
{
    if (valid_key_seq >= key_seq)
        return;

    const size_t tail_bytes = size_t(key_seq - valid_key_seq) * head_dim * sizeof(T);
    for (int i = 0; i < batch_heads; ++i)
        CHECK_CUDA(cudaMemsetAsync(values + (Index(i) * key_seq + valid_key_seq) * head_dim,
                                   0, tail_bytes, device::get_compute_stream()));
}

template<typename T>
static bool grouped_attention_gemm_gpu(const int batch, const int query_seq, const int key_seq,
                                       const int n_query_heads, const int n_kv_heads, const int head_dim,
                                       const float scale, const int query_position_offset, const bool causal,
                                       const T* Q, const T* K, const T* V, T* O)
{
    const int group = n_kv_heads > 0 ? n_query_heads / n_kv_heads : 0;
    if (group < 1 || group * n_kv_heads != n_query_heads || key_seq <= 0 || head_dim <= 0)
        return false;

    constexpr bool is_fp32 = is_same_v<T, float>;
    const cudaDataType_t dtype = is_fp32 ? CUDA_R_32F : CUDA_R_16BF;

    const Index q_elems  = Index(query_seq) * n_query_heads * head_dim;
    const Index kv_elems = Index(key_seq) * n_kv_heads * head_dim;
    const Index s_elems  = Index(n_query_heads) * query_seq * key_seq;

    const Index per_batch_bytes = s_elems * Index(sizeof(float))
                                + (is_fp32 ? 0 : s_elems * Index(sizeof(T)))
                                + 2 * (q_elems + kv_elems) * Index(sizeof(T));

    constexpr Index budget_bytes = Index(256) << 20;
    const int chunk = to_int(max(Index(1), min(Index(batch), budget_bytes / per_batch_bytes)));

    auto aligned = [](Index bytes) { return (bytes + 15) & ~Index(15); };
    const Index scores_bytes = aligned(Index(chunk) * s_elems * Index(sizeof(float)));
    const Index probs_bytes  = is_fp32 ? 0 : aligned(Index(chunk) * s_elems * Index(sizeof(T)));
    const Index q_bytes      = aligned(Index(chunk) * q_elems * Index(sizeof(T)));
    const Index kv_bytes     = aligned(Index(chunk) * kv_elems * Index(sizeof(T)));

    thread_local Buffer workspace{Device::CUDA};

    try
    {
        workspace.grow_to(scores_bytes + probs_bytes + 2 * q_bytes + 2 * kv_bytes);

        char* base    = workspace.as<char>();
        float* scores = reinterpret_cast<float*>(base);
        T* probs      = is_fp32 ? reinterpret_cast<T*>(scores)
                                : reinterpret_cast<T*>(base + scores_bytes);
        T* Qt         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes);
        T* Ot         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes + q_bytes);
        T* Kt         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes + 2 * q_bytes);
        T* Vt         = reinterpret_cast<T*>(base + scores_bytes + probs_bytes + 2 * q_bytes + kv_bytes);

        const int mq = group * query_seq;
        const int valid_key_seq = causal ? min(query_position_offset + query_seq, key_seq) : key_seq;

        for (int b0 = 0; b0 < batch; b0 += chunk)
        {
            const int bc = min(chunk, batch - b0);
            const int batch_count = bc * n_kv_heads;

            split_heads_cuda<T>(Index(bc) * q_elems, Q + Index(b0) * q_elems, Qt,
                                query_seq, n_query_heads, head_dim);
            split_heads_cuda<T>(Index(bc) * kv_elems, K + Index(b0) * kv_elems, Kt,
                                key_seq, n_kv_heads, head_dim);
            split_heads_cuda<T>(Index(bc) * kv_elems, V + Index(b0) * kv_elems, Vt,
                                key_seq, n_kv_heads, head_dim);

            zero_grouped_attention_value_tail(Vt, bc * n_kv_heads,
                                              key_seq, valid_key_seq, head_dim);

            grouped_attention_gemm(CUBLAS_OP_T, CUBLAS_OP_N, key_seq, mq, head_dim, scale,
                                   Kt, dtype, head_dim, Index(key_seq) * head_dim,
                                   Qt, dtype, head_dim, Index(mq) * head_dim,
                                   scores, CUDA_R_32F, key_seq, Index(mq) * key_seq,
                                   batch_count);

            grouped_attention_softmax_cuda<T>(bc * n_query_heads * query_seq, query_seq, key_seq,
                                              query_position_offset, causal, scores, probs);

            grouped_attention_gemm(CUBLAS_OP_N, CUBLAS_OP_N, head_dim, mq, key_seq, 1.0f,
                                   Vt, dtype, head_dim, Index(key_seq) * head_dim,
                                   probs, dtype, key_seq, Index(mq) * key_seq,
                                   Ot, dtype, head_dim, Index(mq) * head_dim,
                                   batch_count);

            merge_heads_cuda<T>(Index(bc) * q_elems, Ot, O + Index(b0) * q_elems,
                                query_seq, n_query_heads, head_dim);
        }
    }
    catch (...)
    {
        device::reset_last_error();
        return false;
    }

    return true;
}

// cuDNN's fused attention, which replaces the path above rather than accelerating
// it: it never forms the batch*query_heads*query_seq*key_seq score matrix, and it
// runs on the tensor cores that grouped_attention_cublas() turns off. Grouped
// shapes are native to it — K and V simply carry fewer heads than Q. Measured on
// sm_120 at batch 8, 16:4 heads, head_dim 64: 0.88 ms against 9.03 ms for the
// materialized path at sequence 2048, with no workspace against 2 GiB of scores.
struct GroupedAttentionSdpaCache
{
    struct Key
    {
        int batch = 0, query_seq = 0, key_seq = 0;
        int query_heads = 0, kv_heads = 0, head_dim = 0;
        bool causal = false;

        bool operator==(const Key&) const = default;
    };

    struct KeyHash
    {
        size_t operator()(const Key& key) const
        {
            size_t hash = 1469598103934665603ull;
            for (const int field : {key.batch, key.query_seq, key.key_seq,
                                    key.query_heads, key.kv_heads, key.head_dim, int(key.causal)})
                hash = (hash ^ size_t(field)) * 1099511628211ull;
            return hash;
        }
    };

    struct Entry
    {
        shared_ptr<cudnn_frontend::graph::Graph> graph;
        shared_ptr<cudnn_frontend::graph::Tensor_attributes> Q, K, V, O;
        int64_t workspace_bytes = 0;
    };

    unordered_map<Key, Entry, KeyHash> entries;
    bool disabled = false;
};

static unique_ptr<GroupedAttentionSdpaCache> grouped_attention_sdpa_cache;

template<typename T>
static bool grouped_attention_sdpa_gpu(const int batch, const int query_seq, const int key_seq,
                                       const int n_query_heads, const int n_kv_heads, const int head_dim,
                                       const float scale, const int query_position_offset, const bool causal,
                                       const T* Q, const T* K, const T* V, T* O)
{
    // cuDNN's fused attention is BF16-only, so FP32 keeps the path below.
    if constexpr (!is_same_v<T, bfloat16>)
        return false;
    else
    {
        // Escape hatch for A/B-ing the fused path against the materialized one.
        static const bool disabled = env_flag_enabled("OPENNN_GQA_DISABLE_SDPA");
        if (disabled) return false;

        // The fused mask places query i at absolute position i, so a decode offset
        // would have it mask against the wrong positions.
        if (query_position_offset != 0) return false;
        if (n_kv_heads <= 0 || n_query_heads % n_kv_heads != 0) return false;

        const GroupedAttentionSdpaCache::Key key{batch, query_seq, key_seq,
                                                 n_query_heads, n_kv_heads, head_dim, causal};

        return cudnn_frontend::run_frontend(grouped_attention_sdpa_cache, "GroupedQueryAttention",
                                            [&](GroupedAttentionSdpaCache& cache)
        {
            auto& entry = cache.entries[key];

            if (!entry.graph)
            {
                const auto graph = cudnn_frontend::new_graph(Type::BF16);

                const int64_t batch_size = batch;
                const int64_t depth      = head_dim;

                const auto dims = [&](int64_t heads, int64_t seq)
                    { return vector<int64_t>{batch_size, heads, seq, depth}; };

                // Q, K and V arrive as [batch][seq][heads][dim], so the head stride is
                // just depth and the sequence stride steps over every head. Addressing
                // that layout directly is what makes the split_heads/merge_heads copies
                // the materialized path needs unnecessary here.
                const auto strides = [&](int64_t heads, int64_t seq)
                    { return vector<int64_t>{seq * heads * depth, depth, heads * depth, 1}; };

                const auto bshd = [&](const char* name, int64_t heads, int64_t seq)
                {
                    return graph->tensor(cudnn_frontend::graph::Tensor_attributes()
                                         .set_name(name)
                                         .set_dim(dims(heads, seq))
                                         .set_stride(strides(heads, seq)));
                };

                entry.Q = bshd("Q", n_query_heads, query_seq);
                entry.K = bshd("K", n_kv_heads,    key_seq);
                entry.V = bshd("V", n_kv_heads,    key_seq);

                // Stats are only produced for training, which this layer does not do.
                auto [out, stats] = graph->sdpa(entry.Q, entry.K, entry.V,
                                                cudnn_frontend::graph::SDPA_attributes()
                                                .set_name("gqa_flash_fwd")
                                                .set_generate_stats(false)
                                                .set_causal_mask(causal)
                                                .set_attn_scale(scale));
                (void)stats;

                out->set_output(true)
                   .set_dim(dims(n_query_heads, query_seq))
                   .set_stride(strides(n_query_heads, query_seq));
                entry.O = out;

                cudnn_frontend::finalize_attention(*graph, "gqa sdpa fwd");
                cudnn_frontend::check_status(graph->get_workspace_size(entry.workspace_bytes),
                                             "gqa sdpa get_workspace_size");
                entry.graph = graph;
            }

            unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
            tensors[entry.Q] = const_cast<T*>(Q);
            tensors[entry.K] = const_cast<T*>(K);
            tensors[entry.V] = const_cast<T*>(V);
            tensors[entry.O] = O;

            cudnn_frontend::execute_graph(*entry.graph, tensors,
                                          cudnn_frontend::shared_workspace(entry.workspace_bytes),
                                          "gqa sdpa execute", string());
        });
    }
}

static void grouped_attention_gpu(const TensorView& query, const TensorView& key, const TensorView& value,
                                  TensorView& output, Index n_query_heads, Index n_kv_heads, Index head_dim,
                                  bool causal, float scale, Index query_position_offset,
                                  float* decode_partials, const int* kv_length_device)
{
    const int batch     = to_int(query.shape[0]);
    const int query_seq = to_int(query.shape[1]);
    const int key_seq   = to_int(key.shape[1]);
    const int group     = to_int(n_kv_heads) > 0 ? to_int(n_query_heads / n_kv_heads) : 0;

    const bool decode = batch == 1 && query_seq == 1 && causal && decode_partials
                     && grouped_attention_decode_supported(to_int(head_dim), group);

    output.dispatch([&]<typename T>() {

        // Variable key lengths would need the padding-mask plumbing the fused path
        // does not carry here, so those shapes stay on the materialized path.
        if (batch * query_seq * to_int(n_query_heads) > 0 && !decode && !kv_length_device
            && grouped_attention_sdpa_gpu<T>(batch, query_seq, key_seq,
                                             to_int(n_query_heads), to_int(n_kv_heads), to_int(head_dim),
                                             scale, to_int(query_position_offset), causal,
                                             query.as<T>(), key.as<T>(), value.as<T>(), output.as<T>()))
            return;

        if (batch * query_seq * to_int(n_query_heads) > 0 && !decode
            && grouped_attention_gemm_gpu<T>(batch, query_seq, key_seq,
                                             to_int(n_query_heads), to_int(n_kv_heads), to_int(head_dim),
                                             scale, to_int(query_position_offset), causal,
                                             query.as<T>(), key.as<T>(), value.as<T>(), output.as<T>()))
            return;

        grouped_attention_cuda<T>(batch, query_seq, key_seq, to_int(n_query_heads), to_int(n_kv_heads),
                                  to_int(head_dim), scale, to_int(query_position_offset), causal,
                                  kv_length_device, decode_partials,
                                  query.as<T>(), key.as<T>(), value.as<T>(), output.as<T>());
    });
}

void qk_rope_cache_append(const TensorView& qkv_row, const TensorView& q_norm_weight,
                          const TensorView& k_norm_weight, const TensorView& cos_table,
                          const TensorView& sin_table, TensorView& q_out,
                          TensorView& key_cache, TensorView& value_cache,
                          Index n_query_heads, Index n_kv_heads, Index head_dim,
                          float epsilon, const int* position_device)
{
    throw_if(!qkv_row.is_cuda() || !position_device, "qk_rope_cache_append: GPU tensors and a device position are required.");

    q_out.dispatch([&]<typename T>() {
        qk_rope_cache_append_cuda<T>(to_int(n_query_heads), to_int(n_kv_heads), to_int(head_dim),
                                     epsilon, position_device, qkv_row.as<T>(),
                                     q_norm_weight.empty() ? nullptr : q_norm_weight.as<float>(),
                                     k_norm_weight.empty() ? nullptr : k_norm_weight.as<float>(),
                                     cos_table.as<float>(), sin_table.as<float>(),
                                     q_out.as<T>(), key_cache.as<T>(), value_cache.as<T>());
    });
}

static void qk_norm_gpu(const TensorView& input, const TensorView& weight, TensorView& output,
                        Index head_dim, float epsilon)
{
    const int rows = to_int(input.size() / head_dim);
    output.dispatch([&]<typename T>() {
        rmsnorm_forward_cuda<T>(rows, to_int(head_dim), input.as<T>(), output.as<T>(),
                                nullptr, weight.as<float>(), epsilon);
    });
}

#else

Index grouped_attention_decode_scratch_floats(Index, Index)
{
    return 0;
}

void qk_rope_cache_append(const TensorView&, const TensorView&, const TensorView&, const TensorView&,
                          const TensorView&, TensorView&, TensorView&, TensorView&,
                          Index, Index, Index, float, const int*)
{
    throw runtime_error("qk_rope_cache_append: CUDA support not compiled in.");
}

static void grouped_attention_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index, bool, float, Index, float*, const int*) { throw runtime_error("grouped_attention_gpu: CUDA support not compiled in."); }
static void qk_norm_gpu(const TensorView&, const TensorView&, TensorView&, Index, float) { throw runtime_error("qk_norm_gpu: CUDA support not compiled in."); }
static void rope_forward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index) { throw runtime_error("rope_forward_gpu: CUDA support not compiled in."); }
static void rope_backward_gpu(const TensorView&, const TensorView&, const TensorView&, TensorView&, Index, Index, Index) { throw runtime_error("rope_backward_gpu: CUDA support not compiled in."); }

#endif


void GroupedQueryAttentionOperator::set(Index new_sequence_length, Index new_hidden,
                                        Index new_q_heads, Index new_kv_heads, Index new_head_dim,
                                        float new_rope_theta, float new_rms_epsilon, bool new_use_qk_norm)
{
    sequence_length = new_sequence_length;
    hidden          = new_hidden;
    q_heads         = new_q_heads;
    kv_heads        = new_kv_heads;
    head_dim        = new_head_dim;
    rope_theta      = new_rope_theta;
    rms_epsilon     = new_rms_epsilon;
    use_qk_norm     = new_use_qk_norm;

}

vector<TensorSpec> GroupedQueryAttentionOperator::parameter_specs() const
{

    vector<TensorSpec> specs = {
        {Shape{q_dim(),  hidden},   weights_dtype},
        {Shape{kv_dim(), hidden},   weights_dtype},
        {Shape{kv_dim(), hidden},   weights_dtype},
        {Shape{hidden,   q_dim()},  weights_dtype},
    };

    if (use_qk_norm)
    {
        specs.push_back({Shape{head_dim}, Type::FP32});
        specs.push_back({Shape{head_dim}, Type::FP32});
    }

    return specs;
}

vector<Operator::SlotQuantization> GroupedQueryAttentionOperator::parameter_quantization() const
{
    return {{q_dim(), 0}, {kv_dim(), 0}, {kv_dim(), 0}, {hidden, 0}};
}

void GroupedQueryAttentionOperator::link_parameters(span<const TensorView> views)
{
    if (!link_views(views, {&q_proj, &k_proj, &v_proj, &o_proj})) return;

    const Index elem = Index(type_bytes(q_proj.type));
    qkv_fused = q_proj.type == k_proj.type && k_proj.type == v_proj.type
        && static_cast<const char*>(k_proj.data) == static_cast<const char*>(q_proj.data) + q_proj.size() * elem
        && static_cast<const char*>(v_proj.data) == static_cast<const char*>(k_proj.data) + k_proj.size() * elem;

    if (use_qk_norm && views.size() >= 6)
    {
        q_norm = views[4];
        k_norm = views[5];
    }
    else
    {
        q_norm = {};
        k_norm = {};
    }
}

void GroupedQueryAttentionOperator::link_parameter_scales(span<const TensorView> views)
{
    if (views.size() < 4) return;
    q_scale = views[0];
    k_scale = views[1];
    v_scale = views[2];
    o_scale = views[3];

    const bool scales_fused = q_scale.data && k_scale.data && v_scale.data
        && k_scale.as<const float>() == q_scale.as<const float>() + q_scale.size()
        && v_scale.as<const float>() == k_scale.as<const float>() + k_scale.size();

    qkv_scale = scales_fused
        ? TensorView(q_scale.data, Shape{q_dim() + 2 * kv_dim()}, Type::FP32, q_scale.device)
        : TensorView{};

    if (q_proj.is_int8() && !scales_fused)
        qkv_fused = false;
}

void GroupedQueryAttentionOperator::set_parameters_random()
{

    if (q_norm.data) q_norm.as_vector().setOnes();
    if (k_norm.data) k_norm.as_vector().setOnes();
}

void GroupedQueryAttentionOperator::back_propagate(ForwardPropagation&, BackPropagation&, size_t) const
{

    throw runtime_error("GroupedQueryAttention is inference-only: back-propagation is not implemented.");
}

namespace
{

struct GroupedAttentionCpuScratch
{
    vector<float> cos, sin;
    vector<float> q, k, v, qr, kr, attn;
    Index table_len = -1, head_dim = 0;
    float theta = 0.0f;

    void build_tables(Index new_table_len, Index new_head_dim, float new_theta)
    {
        if (table_len == new_table_len && head_dim == new_head_dim && theta == new_theta) return;
        cos.resize(size_t(new_table_len) * new_head_dim);
        sin.resize(size_t(new_table_len) * new_head_dim);
        TensorView cos_v(cos.data(), {new_table_len, new_head_dim});
        TensorView sin_v(sin.data(), {new_table_len, new_head_dim});
        rotary_build_tables(cos_v, sin_v, new_table_len, new_head_dim, new_theta);
        table_len = new_table_len; head_dim = new_head_dim; theta = new_theta;
    }
};

GroupedAttentionCpuScratch& gqa_cpu_scratch()
{
    thread_local GroupedAttentionCpuScratch scratch;
    return scratch;
}

float* grown(vector<float>& buffer, size_t n)
{
    if (buffer.size() < n) buffer.resize(n);
    return buffer.data();
}

}

void GroupedQueryAttentionOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool  )
{
    TensorView& input  = get_input(forward_propagation, layer);
    TensorView& output = get_output(forward_propagation, layer);

    const Index batch = forward_propagation.batch_size;

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        forward_gpu(input, output, batch, forward_propagation.past_length,
                    forward_propagation.get_sequence_capacity(),
                    static_cast<const int*>(forward_propagation.position_device.data));
        return;
    }
#endif

    const Index seq   = input.shape[1];
    const Index qd    = q_dim();
    const Index kd    = kv_dim();
    const float scale = 1.0f / sqrt(float(head_dim));

    const Index table_len = sequence_length;
    throw_if(seq < 1 || forward_propagation.past_length < 0
             || forward_propagation.past_length + seq > table_len,
             "GroupedQueryAttentionOperator: query [{}, {}) exceeds the "
             "{}-token KV cache.",
             forward_propagation.past_length,
             forward_propagation.past_length + seq, table_len);
    auto& scratch = gqa_cpu_scratch();
    scratch.build_tables(table_len, head_dim, rope_theta);
    TensorView cos_v(scratch.cos.data(), {table_len, head_dim}), sin_v(scratch.sin.data(), {table_len, head_dim});

    float* x_all = input.as<float>();
    float* o_all = output.as<float>();

    if (batch == 1)
    {
        const Index past  = forward_propagation.past_length;
        const Index total = past + seq;

        const Index capacity_bytes = table_len * kd * Index(sizeof(float));
        if (cache_capacity != table_len || kv_key.device_type != Device::CPU)
        {
            kv_key.resize_bytes(capacity_bytes, Device::CPU);
            kv_value.resize_bytes(capacity_bytes, Device::CPU);
            cache_capacity = table_len;
        }
        float* kcache = kv_key.as<float>();
        float* vcache = kv_value.as<float>();

        float* q    = grown(scratch.q,    size_t(seq) * qd);
        float* k    = grown(scratch.k,    size_t(seq) * kd);
        float* qr   = grown(scratch.qr,   size_t(seq) * qd);
        float* attn = grown(scratch.attn, size_t(seq) * qd);

        TensorView x_b(x_all, {1, seq, hidden});
        TensorView q_v(q, {1, seq, qd}), k_v(k, {1, seq, kd});
        TensorView v_slot(vcache + size_t(past) * kd, {1, seq, kd});
        TensorView k_slot(kcache + size_t(past) * kd, {1, seq, kd});

        linear_forward_transposed(x_b, q_proj, q_v);
        linear_forward_transposed(x_b, k_proj, k_v);
        linear_forward_transposed(x_b, v_proj, v_slot);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        TensorView qr_v(qr, {1, seq, qd});
        rotary_forward(q_v, cos_v, sin_v, qr_v,   head_dim, head_dim, past);
        rotary_forward(k_v, cos_v, sin_v, k_slot, head_dim, head_dim, past);

        TensorView key_all(kcache, {1, total, kd}), val_all(vcache, {1, total, kd});
        TensorView attn_v(attn, {1, seq, qd});
        grouped_attention_forward(qr_v, key_all, val_all, attn_v, q_heads, kv_heads, head_dim, true, scale, past);

        TensorView o_b(o_all, {1, seq, hidden});
        linear_forward_transposed(attn_v, o_proj, o_b);
        return;
    }

    throw_if(forward_propagation.past_length != 0,
             "GroupedQueryAttentionOperator: KV-cache decoding requires batch size 1.");

    float* q    = grown(scratch.q,    size_t(seq) * qd);
    float* k    = grown(scratch.k,    size_t(seq) * kd);
    float* v    = grown(scratch.v,    size_t(seq) * kd);
    float* qr   = grown(scratch.qr,   size_t(seq) * qd);
    float* kr   = grown(scratch.kr,   size_t(seq) * kd);
    float* attn = grown(scratch.attn, size_t(seq) * qd);

    for (Index b = 0; b < batch; ++b)
    {
        TensorView x_b(x_all + size_t(b) * seq * hidden, {1, seq, hidden});
        TensorView q_v(q, {1, seq, qd}), k_v(k, {1, seq, kd}), v_v(v, {1, seq, kd});

        linear_forward_transposed(x_b, q_proj, q_v);
        linear_forward_transposed(x_b, k_proj, k_v);
        linear_forward_transposed(x_b, v_proj, v_v);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        TensorView qr_v(qr, {1, seq, qd}), kr_v(kr, {1, seq, kd});
        rotary_forward(q_v, cos_v, sin_v, qr_v, head_dim, head_dim, 0);
        rotary_forward(k_v, cos_v, sin_v, kr_v, head_dim, head_dim, 0);

        TensorView attn_v(attn, {1, seq, qd});
        grouped_attention_forward(qr_v, kr_v, v_v, attn_v, q_heads, kv_heads, head_dim, true, scale, 0);

        TensorView o_b(o_all + size_t(b) * seq * hidden, {1, seq, hidden});
        linear_forward_transposed(attn_v, o_proj, o_b);
    }
}

#ifdef OPENNN_HAS_CUDA

namespace
{

struct GroupedAttentionScratch
{
    Buffer cos{Device::CUDA}, sin{Device::CUDA};
    Buffer q{Device::CUDA}, k{Device::CUDA}, v{Device::CUDA};
    Buffer qr{Device::CUDA}, kr{Device::CUDA}, attn{Device::CUDA};
    Buffer qkv{Device::CUDA}, partials{Device::CUDA};
    Index sequence = -1;
    Index query_capacity = 0;
    Index q_dim = 0, kv_dim = 0, head_dim = 0;
    float theta = 0.0f;
    Type dtype = Type::FP32;
};

GroupedAttentionScratch& gqa_scratch(Index sequence, Index q_dim, Index kv_dim,
                                     Index head_dim, float theta, Type dtype)
{
    thread_local map<tuple<Index, Index, Index, Index, float, int>,
                     GroupedAttentionScratch> scratches;
    return scratches[{sequence, q_dim, kv_dim, head_dim, theta, int(dtype)}];
}

struct GroupedAttentionSDPA
{
    shared_ptr<cudnn_frontend::graph::Graph> graph;
    shared_ptr<cudnn_frontend::graph::Tensor_attributes> Q, K, V, O, SeqQ, SeqKV;
    unordered_map<shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> tensors;
    void* workspace = nullptr;
    int32_t* seq_device = nullptr;
    int32_t* seq_pinned = nullptr;
    Index max_q = 0, max_kv = 0;
    Index q_heads = 0, kv_heads = 0, head_dim = 0;
    bool failed = false;

    ~GroupedAttentionSDPA()
    {
        device::deallocate(Device::CUDA, workspace, 0);
        device::deallocate(Device::CUDA, seq_device, 0);
        if (seq_pinned) device::deallocate_pinned_host(seq_pinned);
    }
};

GroupedAttentionSDPA& gqa_sdpa(Index max_q, Index max_kv,
                               Index q_heads, Index kv_heads, Index head_dim)
{
    thread_local map<tuple<Index, Index, Index, Index, Index>,
                     GroupedAttentionSDPA> graphs;
    return graphs[{max_q, max_kv, q_heads, kv_heads, head_dim}];
}

shared_ptr<cudnn_frontend::graph::Tensor_attributes>
gqa_bshd_tensor(cudnn_frontend::graph::Graph& graph, const char* name,
                int64_t heads, int64_t max_seq, int64_t head_dim)
{
    return graph.tensor(cudnn_frontend::graph::Tensor_attributes()
                        .set_name(name)
                        .set_dim   ({1, heads, max_seq, head_dim})
                        .set_stride({heads * max_seq * head_dim, head_dim, heads * head_dim, 1}));
}

void gqa_sdpa_build(GroupedAttentionSDPA& s, Index max_q, Index max_kv,
                    Index q_heads, Index kv_heads, Index head_dim, float scale)
{
    auto graph = cudnn_frontend::new_graph(Type::BF16);

    s.Q = gqa_bshd_tensor(*graph, "Q", q_heads,  max_q,  head_dim);
    s.K = gqa_bshd_tensor(*graph, "K", kv_heads, max_kv, head_dim);
    s.V = gqa_bshd_tensor(*graph, "V", kv_heads, max_kv, head_dim);

    s.SeqQ  = cudnn_frontend::seq_len_scalar(*graph, "SeqQ");
    s.SeqKV = cudnn_frontend::seq_len_scalar(*graph, "SeqKV");

    auto options = cudnn_frontend::graph::SDPA_attributes()
                   .set_name("gqa_prefill")
                   .set_generate_stats(false)
                   .set_padding_mask(true)
                   .set_seq_len_q(s.SeqQ)
                   .set_seq_len_kv(s.SeqKV)
                   .set_causal_mask_bottom_right(true)
                   .set_attn_scale(scale);

    auto [O, stats] = graph->sdpa(s.Q, s.K, s.V, options);
    (void)stats;
    O->set_output(true)
      .set_dim   ({1, q_heads, max_q, head_dim})
      .set_stride({q_heads * max_q * head_dim, head_dim, q_heads * head_dim, 1});
    s.O = O;

    cudnn_frontend::finalize_attention(*graph, "gqa sdpa");

    const int64_t workspace_bytes = graph->get_workspace_size();
    device::deallocate(Device::CUDA, s.workspace, 0);
    s.workspace = workspace_bytes > 0 ? device::allocate(Device::CUDA, Index(workspace_bytes)) : nullptr;

    if (!s.seq_device) s.seq_device = static_cast<int32_t*>(device::allocate(Device::CUDA, Index(2 * sizeof(int32_t))));
    if (!s.seq_pinned) s.seq_pinned = static_cast<int32_t*>(device::allocate_pinned_host(Index(2 * sizeof(int32_t))));

    s.graph = std::move(graph);
    s.tensors.clear();
    s.tensors.reserve(6);
    s.max_q = max_q;
    s.max_kv = max_kv;
    s.q_heads = q_heads;
    s.kv_heads = kv_heads;
    s.head_dim = head_dim;
}

}

void GroupedQueryAttentionOperator::forward_gpu(TensorView& input, TensorView& output, Index batch, Index past,
                                                Index query_capacity,
                                                const int* position_device)
{
    const Index seq = input.shape[1];
    const Index qd  = q_dim();
    const Index kd  = kv_dim();
    const float scale = 1.0f / sqrt(float(head_dim));
    cudaStream_t stream = device::get_compute_stream();

    const Type  act  = input.type;
    const Index elem = Index(type_bytes(act));

    const Index table_len = sequence_length;
    throw_if(seq < 1 || query_capacity < seq,
             "GroupedQueryAttentionOperator: query length {} exceeds its "
             "temporary capacity {}.", seq, query_capacity);
    throw_if(past < 0 || past + seq > table_len,
             "GroupedQueryAttentionOperator: query [{}, {}) exceeds the "
             "{}-token KV cache.", past, past + seq, table_len);
    auto& s = gqa_scratch(table_len, qd, kd, head_dim, rope_theta, act);
    {
        const bool geometry_changed =
            s.sequence != table_len || s.dtype != act
            || s.q_dim != qd || s.kv_dim != kd
            || s.head_dim != head_dim || s.theta != rope_theta;
        if (geometry_changed)
        {
            vector<float> cos_h(size_t(table_len) * head_dim), sin_h(size_t(table_len) * head_dim);
            { TensorView cv(cos_h.data(), {table_len, head_dim}), sv(sin_h.data(), {table_len, head_dim});
              rotary_build_tables(cv, sv, table_len, head_dim, rope_theta); }

            auto upload = [&](const vector<float>& host) {
                Buffer b(Device::CPU);
                b.resize_bytes(Index(host.size()) * Index(sizeof(float)), Device::CPU);
                memcpy(b.data, host.data(), host.size() * sizeof(float));
                b.migrate_to(Device::CUDA, stream);
                return b;
            };
            s.cos = upload(cos_h);
            s.sin = upload(sin_h);
            s.query_capacity = 0;
            s.partials.resize_bytes(grouped_attention_decode_scratch_floats(q_heads, head_dim)
                                    * Index(sizeof(float)), Device::CUDA);
            s.sequence = table_len;
            s.q_dim = qd; s.kv_dim = kd; s.head_dim = head_dim;
            s.theta = rope_theta;
            s.dtype = act;
        }

        if (s.query_capacity < query_capacity)
        {
            s.q.grow_to(query_capacity * qd * elem);
            s.k.grow_to(query_capacity * kd * elem);
            s.v.grow_to(query_capacity * kd * elem);
            s.qr.grow_to(query_capacity * qd * elem);
            s.kr.grow_to(query_capacity * kd * elem);
            s.attn.grow_to(query_capacity * qd * elem);
            s.qkv.grow_to((qd + 2 * kd) * elem);
            s.query_capacity = query_capacity;
        }

        if (cache_capacity != table_len || cache_dtype != act || kv_key.device_type != Device::CUDA)
        {
            kv_key.resize_bytes(table_len * kd * elem, Device::CUDA);
            kv_value.resize_bytes(table_len * kd * elem, Device::CUDA);
            cache_capacity = table_len;
            cache_dtype = act;
        }
    }

    TensorView cos_v(s.cos.data, {table_len, head_dim}, Type::FP32, Device::CUDA);
    TensorView sin_v(s.sin.data, {table_len, head_dim}, Type::FP32, Device::CUDA);

    if (batch == 1)
    {
        const Index total = past + seq;
        TensorView x_b(input.data,  {1, seq, hidden}, act, Device::CUDA);
        TensorView o_b(output.data, {1, seq, hidden}, act, Device::CUDA);
        TensorView q_v(s.q.data,  {1, seq, qd}, act, Device::CUDA);
        TensorView k_v(s.k.data,  {1, seq, kd}, act, Device::CUDA);
        TensorView qr_v(s.qr.data, {1, seq, qd}, act, Device::CUDA);
        TensorView attn_v(s.attn.data, {1, seq, qd}, act, Device::CUDA);

        char* v_at = static_cast<char*>(kv_value.data) + size_t(past) * kd * elem;
        char* k_at = static_cast<char*>(kv_key.data)   + size_t(past) * kd * elem;
        TensorView v_slot(v_at, {1, seq, kd}, act, Device::CUDA);
        TensorView k_slot(k_at, {1, seq, kd}, act, Device::CUDA);

        if (seq == 1 && qkv_fused && position_device && use_qk_norm)
        {
            TensorView qkv_row(s.qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            {
                TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
                linear_forward_transposed(x_b, qkv_w, qkv_row, qkv_scale);
            }

            TensorView key_cache(kv_key.data,   {1, table_len, kd}, act, Device::CUDA);
            TensorView val_cache(kv_value.data, {1, table_len, kd}, act, Device::CUDA);
            {
                qk_rope_cache_append(qkv_row, q_norm, k_norm, cos_v, sin_v, qr_v, key_cache, val_cache,
                                     q_heads, kv_heads, head_dim, rms_epsilon, position_device);
            }
            {
                grouped_attention_forward(qr_v, key_cache, val_cache, attn_v, q_heads, kv_heads, head_dim,
                                          true, scale, past,
                                          static_cast<float*>(s.partials.data), position_device);
            }
            {
                linear_forward_transposed(attn_v, o_proj, o_b, o_scale);
            }
            return;
        }

        if (seq == 1 && qkv_fused)
        {
            TensorView qkv_row(s.qkv.data, {1, 1, qd + 2 * kd}, act, Device::CUDA);
            TensorView qkv_w(q_proj.data, {qd + 2 * kd, hidden}, q_proj.type, Device::CUDA);
            linear_forward_transposed(x_b, qkv_w, qkv_row, qkv_scale);
            q_v = TensorView(s.qkv.data, {1, 1, qd}, act, Device::CUDA);
            k_v = TensorView(static_cast<char*>(s.qkv.data) + size_t(qd) * elem, {1, 1, kd}, act, Device::CUDA);
            device::copy_async(v_at, static_cast<char*>(s.qkv.data) + size_t(qd + kd) * elem,
                               kd * elem, device::CopyKind::DeviceToDevice, stream);
        }
        else
        {
            linear_forward_transposed(x_b, q_proj, q_v, q_scale);
            linear_forward_transposed(x_b, k_proj, k_v, k_scale);
            linear_forward_transposed(x_b, v_proj, v_slot, v_scale);
        }

        {
            if (use_qk_norm)
            {
                qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
                qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
            }

            rotary_forward(q_v, cos_v, sin_v, qr_v,   head_dim, head_dim, past);
            rotary_forward(k_v, cos_v, sin_v, k_slot, head_dim, head_dim, past);
        }

        auto& sdpa = gqa_sdpa(query_capacity, table_len,
                              q_heads, kv_heads, head_dim);
        if (seq > 1 && act == Type::BF16 && !sdpa.failed)
        {
            if (!sdpa.graph || sdpa.max_q != query_capacity
                || sdpa.max_kv != table_len || sdpa.q_heads != q_heads
                || sdpa.kv_heads != kv_heads || sdpa.head_dim != head_dim)
            {
                try
                {
                    gqa_sdpa_build(sdpa, query_capacity, table_len,
                                   q_heads, kv_heads, head_dim, scale);
                }
                catch (const exception& e)
                {
                    sdpa.failed = true;
                    cerr << "GroupedQueryAttention: cuDNN flash-attention prefill unavailable ("
                         << e.what() << "); using the generic kernel.\n";
                }
            }

            if (!sdpa.failed)
            {
                {
                    sdpa.seq_pinned[0] = int32_t(seq);
                    sdpa.seq_pinned[1] = int32_t(total);
                    device::copy_async(sdpa.seq_device, sdpa.seq_pinned, Index(2 * sizeof(int32_t)),
                                       device::CopyKind::HostToDevice, stream);

                    sdpa.tensors[sdpa.Q]     = s.qr.data;
                    sdpa.tensors[sdpa.K]     = kv_key.data;
                    sdpa.tensors[sdpa.V]     = kv_value.data;
                    sdpa.tensors[sdpa.O]     = s.attn.data;
                    sdpa.tensors[sdpa.SeqQ]  = sdpa.seq_device;
                    sdpa.tensors[sdpa.SeqKV] = sdpa.seq_device + 1;
                    cudnn_frontend::check_status(
                        sdpa.graph->execute(Backend::get_cudnn_handle(), sdpa.tensors, sdpa.workspace),
                        "gqa sdpa execute");
                }
                {
                    linear_forward_transposed(attn_v, o_proj, o_b, o_scale);
                }
                return;
            }
        }

        TensorView key_all(kv_key.data,   {1, total, kd}, act, Device::CUDA);
        TensorView val_all(kv_value.data, {1, total, kd}, act, Device::CUDA);
        {
            grouped_attention_forward(qr_v, key_all, val_all, attn_v, q_heads, kv_heads, head_dim, true, scale, past,
                                      static_cast<float*>(s.partials.data));
        }
        {
            linear_forward_transposed(attn_v, o_proj, o_b, o_scale);
        }
        return;
    }

    throw_if(past != 0, "GroupedQueryAttentionOperator: KV-cache decoding requires batch size 1.");

    TensorView q_v (s.q.data,    {1, seq, qd}, act, Device::CUDA);
    TensorView k_v (s.k.data,    {1, seq, kd}, act, Device::CUDA);
    TensorView v_v (s.v.data,    {1, seq, kd}, act, Device::CUDA);
    TensorView qr_v(s.qr.data,   {1, seq, qd}, act, Device::CUDA);
    TensorView kr_v(s.kr.data,   {1, seq, kd}, act, Device::CUDA);
    TensorView attn_v(s.attn.data, {1, seq, qd}, act, Device::CUDA);

    for (Index b = 0; b < batch; ++b)
    {
        char* in_b  = static_cast<char*>(input.data)  + size_t(b) * seq * hidden * elem;
        char* out_b = static_cast<char*>(output.data) + size_t(b) * seq * hidden * elem;
        TensorView x_b(in_b,  {1, seq, hidden}, act, Device::CUDA);
        TensorView o_b(out_b, {1, seq, hidden}, act, Device::CUDA);

        linear_forward_transposed(x_b, q_proj, q_v, q_scale);
        linear_forward_transposed(x_b, k_proj, k_v, k_scale);
        linear_forward_transposed(x_b, v_proj, v_v, v_scale);

        if (use_qk_norm)
        {
            qk_norm_forward(q_v, q_norm, q_v, head_dim, rms_epsilon);
            qk_norm_forward(k_v, k_norm, k_v, head_dim, rms_epsilon);
        }

        rotary_forward(q_v, cos_v, sin_v, qr_v, head_dim, head_dim, 0);
        rotary_forward(k_v, cos_v, sin_v, kr_v, head_dim, head_dim, 0);

        grouped_attention_forward(qr_v, kr_v, v_v, attn_v, q_heads, kv_heads, head_dim, true, scale, 0);

        linear_forward_transposed(attn_v, o_proj, o_b, o_scale);
    }
}

#endif

GroupedQueryAttention::GroupedQueryAttention(const Shape& new_input_shape,
                                             Index new_q_heads, Index new_kv_heads, Index new_head_dim,
                                             float new_rope_theta, float new_rms_epsilon,
                                             bool new_use_qk_norm,
                                             const string& new_name)
    : Layer(LayerType::GroupedQueryAttention)
{
    operators = {&attention};

    set(new_input_shape, new_q_heads, new_kv_heads, new_head_dim,
        new_rope_theta, new_rms_epsilon, new_use_qk_norm, new_name);
}

void GroupedQueryAttention::set(const Shape& new_input_shape,
                                Index new_q_heads, Index new_kv_heads, Index new_head_dim,
                                float new_rope_theta, float new_rms_epsilon,
                                bool new_use_qk_norm,
                                const string& new_label)
{
    sequence_length = new_input_shape.dim_or_zero(0);
    hidden          = new_input_shape.dim_or_zero(1);
    q_heads         = new_q_heads;
    kv_heads        = new_kv_heads;
    head_dim        = new_head_dim;
    rope_theta      = new_rope_theta;
    rms_epsilon     = new_rms_epsilon;
    use_qk_norm     = new_use_qk_norm;

    set_label(new_label);

    attention.set(sequence_length, hidden, q_heads, kv_heads, head_dim,
                  rope_theta, rms_epsilon, use_qk_norm);
}

void GroupedQueryAttention::apply_input_shape(const Shape& new_input_shape)
{
    if (new_input_shape.rank < 2) return;
    set({new_input_shape[0], new_input_shape[1]},
        q_heads, kv_heads, head_dim, rope_theta, rms_epsilon, use_qk_norm, label);
}

void GroupedQueryAttention::read_JSON_body(const Json* element)
{
    const Shape new_input_shape = string_to_shape(read_json_string(element, "InputDimensions"));
    const Index new_q_heads  = read_json_index(element, "QueryHeads");
    const Index new_kv_heads = read_json_index(element, "KeyValueHeads");
    const Index new_head_dim = read_json_index(element, "HeadDim");
    const float new_rope_theta  = read_json_float(element, "RopeTheta");
    const float new_rms_epsilon = read_json_float(element, "RmsEpsilon");

    const bool new_use_qk_norm = element->has("QKNorm") ? read_json_bool(element, "QKNorm") : true;

    set(new_input_shape, new_q_heads, new_kv_heads, new_head_dim,
        new_rope_theta, new_rms_epsilon, new_use_qk_norm, get_label());
}

void GroupedQueryAttention::write_JSON_body(JsonWriter& printer) const
{
    write_json(printer, {
        {"QueryHeads",    q_heads},
        {"KeyValueHeads", kv_heads},
        {"HeadDim",       head_dim},
        {"RopeTheta",     rope_theta},
        {"RmsEpsilon",    rms_epsilon},
        {"QKNorm",        use_qk_norm}
    });
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
