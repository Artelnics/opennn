// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/device_backend.h"
#include "opennn/core/profiler.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/cuda/kernel_cast.cuh"
#include "opennn/core/cuda/matmul_cudnn.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <utility>

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

static cublasLtEpilogue_t native_epilogue(LinearEpilogue);

namespace
{
    // Tiles whose shape is known, so that a candidate kernel's data movement
    // can be compared: a tile of rows x columns reads (1/rows + 1/columns)
    // bytes through L2 and shared memory per multiply-add, and board power
    // follows that ratio closely. Measured on an RTX 5070 Ti, one
    // 1024x8192x1024 bf16 GEMM with bias and ReLU:
    //
    //     64x64   0.0312 -> 265 W      128x240  0.0120 -> 179 W
    //     64x640  0.0172 -> 189 W      128x320  0.0109 -> 172 W
    //     80x512  0.0145 -> 182 W      256x160  0.0102 -> 169 W
    //
    // The arithmetic is identical in every one of them; the 96 W is traffic.
    // Listed lowest-traffic first, which is the order they are tried in --
    // from probed_tiles_begin on; see below.
    struct LtTile { int id; int rows; int columns; };

    constexpr LtTile lt_known_tiles[] = {
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_256x256, 256, 256}, {CUBLASLT_MATMUL_TILE_192x256, 192, 256},
        {CUBLASLT_MATMUL_TILE_256x192, 256, 192}, {CUBLASLT_MATMUL_TILE_256x160, 256, 160},
        {CUBLASLT_MATMUL_TILE_128x320, 128, 320},
#endif
        {CUBLASLT_MATMUL_TILE_192x128, 192, 128},
        {CUBLASLT_MATMUL_TILE_128x256, 128, 256}, {CUBLASLT_MATMUL_TILE_256x128, 256, 128},
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_128x240, 128, 240}, {CUBLASLT_MATMUL_TILE_256x96,  256,  96},
#endif
        {CUBLASLT_MATMUL_TILE_128x192, 128, 192}, {CUBLASLT_MATMUL_TILE_128x160, 128, 160},
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_80x512,   80, 512},
#endif
        {CUBLASLT_MATMUL_TILE_128x128, 128, 128},
#if CUBLAS_VER_MAJOR >= 13
        {CUBLASLT_MATMUL_TILE_64x640,   64, 640},
#endif
        {CUBLASLT_MATMUL_TILE_128x64,  128,  64},
        {CUBLASLT_MATMUL_TILE_64x128,   64, 128}, {CUBLASLT_MATMUL_TILE_64x64,    64,  64},
    };

    // With cuBLAS 13 or newer, the first three rows -- 256x256, 192x256 and
    // 256x192 -- are priced but never probed: AlgoCheck refused every one of
    // them in all 14,708 measured configurations, and the candidate scan below
    // has no early exit. Older cuBLAS headers do not define the wider tiles;
    // every tile present in their reduced table remains eligible for probing.
    constexpr size_t probed_tiles_begin = CUBLAS_VER_MAJOR >= 13 ? 3 : 0;

    // Unknown tiles report infinite traffic: they are never preferred over a
    // tile whose cost is known, only kept when they are the fastest.
    float tile_traffic(int tile_id, int splitk_number = 1)
    {
        for (const LtTile& tile : lt_known_tiles)
            if (tile.id == tile_id)
                // A split-k kernel writes fp32 partials for every split and
                // reads them all back to reduce; that traffic is invisible to
                // 1/rows + 1/columns, so charge one extra pass per split
                // rather than let a split-k kernel look cheap. It can still be
                // selected on time, only never preferred on traffic.
                return float(max(splitk_number, 1))
                     * (1.0f / float(tile.rows) + 1.0f / float(tile.columns));
        return numeric_limits<float>::infinity();
    }

    // Board power is close to linear in traffic over the tiles above -- 169 W
    // at 0.0102, 265 W at 0.0312 -- so a two-point fit is enough to compare
    // candidates by energy instead of by time alone. It overstates the
    // interior points by at most 6% (0.0172 models at 201 W against 189 W
    // measured), always in the direction of the thirstier tile. An unknown
    // tile has infinite traffic and so is priced at the top of the range,
    // the same worst-case assumption tile_traffic already makes about it.
    float tile_power_watts(float traffic)
    {
        constexpr float lowest_traffic  = 0.0102f, lowest_watts  = 169.0f;
        constexpr float highest_traffic = 0.0312f, highest_watts = 265.0f;
        constexpr float watts_per_traffic =
            (highest_watts - lowest_watts) / (highest_traffic - lowest_traffic);

        return lowest_watts
             + watts_per_traffic * (clamp(traffic, lowest_traffic, highest_traffic) - lowest_traffic);
    }

    // Where a candidate kernel comes from. This matters to the selection rule
    // below and nowhere else: tile_traffic and tile_power_watts are a model of
    // cuBLASLt's nvjet tiles, fitted on cuBLASLt's nvjet tiles, and they have
    // nothing to say about an engine from another library.
    enum class MatmulSource { CublasLt, Cudnn };

    struct MatmulCandidate
    {
        cublasLtMatmulAlgo_t algorithm{};
        size_t               workspace_bytes = 0;
        float                traffic = numeric_limits<float>::infinity();
        MatmulSource         source = MatmulSource::CublasLt;
        int                  cudnn_candidate = -1;
    };

    // Everything a plan's descriptor and layouts are built from, and nothing
    // else. alpha never enters -- it is a call-time pointer that changes no
    // kernel -- and beta enters only as beta_is_zero, because a nonzero beta
    // makes the kernel read C, which changes which algorithm is valid and
    // which is fastest, while its value does not.
    //
    // Every field added here canonicalises to what the call sites already
    // implied: dtype_b equals dtype_a unless the operands genuinely differ,
    // and the three leading dimensions are stored as 0 when derived from m, n
    // and k. That is not tidiness. get_matmul_plan throws on a miss and
    // optimizer.cpp holds cuda_matmul_plan_creation_forbidden across the whole
    // epoch loop, so a field that split one of today's keys in two would take
    // out training in steady state, not merely cost a plan.
    struct MatmulPlanKey
    {
        int m;
        int n;
        int k;
        int transA;
        int transB;
        int epilogue;
        int dtype_a;
        int dtype_b;
        int out_dtype;
        int lda;
        int ldb;
        int ldd;
        int beta_is_zero;
        int tf32;   // the compute type of an fp32 plan follows allow_tf32() at build time

        bool operator==(const MatmulPlanKey&) const noexcept = default;
    };

    struct MatmulPlanKeyHash
    {
        size_t operator()(const MatmulPlanKey& key) const noexcept
        {
            return hash_combine(key.m, key.n, key.k,
                                key.transA, key.transB, key.epilogue,
                                key.dtype_a, key.dtype_b, key.out_dtype,
                                key.lda, key.ldb, key.ldd,
                                key.beta_is_zero, key.tf32);
        }
    };

    matmul::cudnn::Problem make_cudnn_problem(const MatmulPlanKey& key)
    {
        return {
            key.m, key.n, key.k,
            cublasOperation_t(key.transA), cublasOperation_t(key.transB),
            cublasLtEpilogue_t(key.epilogue),
            cudaDataType_t(key.dtype_a), cudaDataType_t(key.dtype_b),
            cudaDataType_t(key.out_dtype),
            key.lda, key.ldb, key.ldd, key.beta_is_zero != 0
        };
    }

    struct MatmulPlan
    {
        MatmulPlanKey        key{};
        cublasLtMatmulDesc_t   matmul_descriptor = nullptr;
        cublasLtMatrixLayout_t a_matrix_layout = nullptr;
        cublasLtMatrixLayout_t b_matrix_layout = nullptr;
        cublasLtMatrixLayout_t output_matrix_layout = nullptr;
        cublasLtMatmulAlgo_t   algorithm{};
        bool                   has_algorithm = false;
        size_t                 workspace_bytes = 0;

        // The cuDNN engine set for this same shape, when there is one, and the
        // configuration the tuner chose out of it. These are an OVERLAY on the
        // cuBLASLt fields above, never a replacement: algorithm and
        // workspace_bytes always hold a usable cuBLASLt kernel, so "fall back
        // to cuBLASLt" is a one-line branch at the call site and stays true
        // even if a cuDNN execution fails at run time, years from now, on a
        // driver nobody here has seen.
        matmul::cudnn::Plan*    cudnn_plan = nullptr;
        int                    cudnn_candidate = -1;
        size_t                 cudnn_workspace_bytes = 0;

        vector<MatmulCandidate> candidates;
        bool                   tuned = true;

        MatmulPlan() = default;
        MatmulPlan(const MatmulPlan&) = delete;
        MatmulPlan& operator=(const MatmulPlan&) = delete;
        MatmulPlan& operator=(MatmulPlan&&) = delete;
        MatmulPlan(MatmulPlan&& other) noexcept
        {
            swap(key, other.key);
            swap(matmul_descriptor, other.matmul_descriptor);
            swap(a_matrix_layout, other.a_matrix_layout);
            swap(b_matrix_layout, other.b_matrix_layout);
            swap(output_matrix_layout, other.output_matrix_layout);
            swap(algorithm, other.algorithm);
            swap(has_algorithm, other.has_algorithm);
            swap(workspace_bytes, other.workspace_bytes);
            swap(cudnn_plan, other.cudnn_plan);
            swap(cudnn_candidate, other.cudnn_candidate);
            swap(cudnn_workspace_bytes, other.cudnn_workspace_bytes);
            swap(candidates, other.candidates);
            swap(tuned, other.tuned);
        }

        ~MatmulPlan()
        {
            matmul::cudnn::destroy(cudnn_plan);
            cublasLtMatrixLayoutDestroy(output_matrix_layout);
            cublasLtMatrixLayoutDestroy(b_matrix_layout);
            cublasLtMatrixLayoutDestroy(a_matrix_layout);
            cublasLtMatmulDescDestroy(matmul_descriptor);
        }

        // Called when the tuner has decided against cuDNN, or could not tune
        // at all. The graph and its built plans are the only thing on this
        // path that holds device memory of its own, and the benchmark reports
        // peak memory, so an unused engine set is released rather than parked.
        void release_cudnn() noexcept
        {
            matmul::cudnn::destroy(cudnn_plan);
            cudnn_plan = nullptr;
            cudnn_candidate = -1;
            cudnn_workspace_bytes = 0;
        }
    };

    struct CudaMatmulThreadState
    {
        using LaneWorkspaces = std::array<Buffer, static_cast<size_t>(device::GraphWorkspaceKind::Count)>;
        std::array<LaneWorkspaces, device::MAX_LANES> workspaces =
            make_lanes(make_index_sequence<device::MAX_LANES>{});

        unordered_map<MatmulPlanKey, MatmulPlan, MatmulPlanKeyHash> matmul_plans;

        template<size_t... I>
        static LaneWorkspaces make_workspaces(index_sequence<I...>)
        {
            return {((void)I, Buffer{Device::CUDA})...};
        }
        template<size_t... L>
        static std::array<LaneWorkspaces, sizeof...(L)> make_lanes(index_sequence<L...>)
        {
            return {((void)L, make_workspaces(make_index_sequence<static_cast<size_t>(device::GraphWorkspaceKind::Count)>{}))...};
        }
    };

    CudaMatmulThreadState& thread_state()
    {
        thread_local CudaMatmulThreadState state;
        return state;
    }

    constexpr size_t cublas_lt_workspace_search_bytes = 32ull * 1024 * 1024;
    constexpr size_t cublas_lt_plan_cache_capacity = 1024;

    // autotune_matmul_plan picks the fastest of up to eight heuristic candidates
    // by timing each of them, and the timings overlap: on Qwen3-4B's
    // gate projection at decode (9728x1x2560 bf16) the split-K=1 kernels swing
    // 2.3x between processes while the winner moves 2%, so 30% of processes
    // under GPU contention picked a different kernel, and three of the eight
    // candidates round bf16 differently. Same operands, three output bit
    // patterns, and a greedy decode that disagreed with itself across runs.
    //
    // OPENNN_LT_DETERMINISTIC=1 takes the heuristic's first candidate for
    // every shape and never times anything -- 40 of 40 contended processes
    // gave one hash. It costs 26% on that GEMM (0.109 ms against 0.086), so
    // it is for tests and CI gates, not for the numbers that get published.
    bool lt_deterministic_selection()
    {
        static const bool deterministic = env_flag_enabled("OPENNN_LT_DETERMINISTIC", false);
        return deterministic;
    }

    // The default keeps the tuner and makes its verdict outlive the process:
    // the first process to see a shape times the candidates and writes the
    // winner below the temp directory, every later one loads it and skips the
    // timing. Mirrors the cuDNN plan cache in cudnn_frontend_utilities.h --
    // OPENNN_LT_PLAN_CACHE=0 turns it off, OPENNN_LT_PLAN_CACHE_DIR moves it.
    // The directory names the card, its architecture and the cuBLASLt build,
    // because a serialised cublasLtMatmulAlgo_t is only promised to mean the
    // same thing under the library that produced it.
    bool lt_plan_cache_enabled()
    {
        static const bool enabled = env_flag_enabled("OPENNN_LT_PLAN_CACHE", true);
        return enabled;
    }

    const filesystem::path& lt_plan_cache_path()
    {
        static const filesystem::path directory = []
        {
            const char* override_path = getenv("OPENNN_LT_PLAN_CACHE_DIR");

            filesystem::path root;

            if (override_path && *override_path)
                root = filesystem::path(override_path);
            else
            {
                error_code error;
                const filesystem::path temporary = filesystem::temp_directory_path(error);
                if (error) return filesystem::path{};
                root = temporary / "opennn-lt-plans";
            }

            cudaDeviceProp properties{};
            if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess)
            {
                cudaGetLastError();
                return filesystem::path{};
            }

            string card(properties.name);
            for (char& character : card)
                if (!isalnum(static_cast<unsigned char>(character))) character = '-';

            // cuDNN is in the name too: a cached plan may name a cuDNN matmul
            // engine by its position in that library's heuristic list.
            return root / format("{}-sm{}{}-cublaslt{}-cudnn{}", card, properties.major, properties.minor,
                                 cublasLtGetVersion(), CUDNN_VERSION);
        }();

        return directory;
    }

    // One cached plan: the key it was tuned for, every knob that steered the
    // tuner, and the winner. The knobs are stored and compared, not merely
    // hashed into the file name, so a name collision cannot hand a plan tuned
    // under one OPENNN_LT_TILE_TOLERANCE to a process running another.
    struct MatmulPlanCacheRecord
    {
        uint32_t magic = 0x4c50544fu;   // "OTPL"
        uint32_t version = 3;
        MatmulPlanKey key{};

        // Every knob the tuner reads, in the order it reads them: the cuBLASLt
        // candidate set and tie-break, then the cross-source rule, then what
        // matmul::cudnn::create() admits into the candidate list at all.
        long long candidates = 0;
        long long tile_tolerance = 0;
        long long traffic_budget = 0;
        long long cross_source_gain = 0;
        long long anchor_on_fastest = 0;
        long long cudnn_enabled = 0;
        long long cudnn_workspace_mb = 0;
        long long cudnn_min_gflop = 0;
        long long cudnn_min_dim = 0;
        long long cudnn_candidates = 0;
        uint64_t workspace_search_bytes = 0;

        // The winner: always a cuBLASLt kernel, plus the cuDNN engine the
        // tuner preferred over it when it did, as the same overlay the plan
        // carries. The engine is named by its index in cuDNN's own
        // enumeration, which is a stable identity for one cuDNN build on one
        // card -- both are in the directory name -- and nothing else; the
        // position it held in the tuner's candidate list is kept for the
        // record only. Version 3 added the enumeration index: rebuilding the
        // whole set to find the winner again cost a Qwen3-4B session about
        // twelve seconds of warm-up.
        cublasLtMatmulAlgo_t algorithm{};
        uint64_t workspace_bytes = 0;
        int cudnn_candidate = -1;
        uint64_t cudnn_workspace_bytes = 0;
        int64_t cudnn_plan_index = -1;

        bool same_tuning(const MatmulPlanCacheRecord& other) const noexcept
        {
            return magic == other.magic && version == other.version && key == other.key
                && candidates == other.candidates && tile_tolerance == other.tile_tolerance
                && traffic_budget == other.traffic_budget
                && cross_source_gain == other.cross_source_gain
                && anchor_on_fastest == other.anchor_on_fastest
                && cudnn_enabled == other.cudnn_enabled
                && cudnn_workspace_mb == other.cudnn_workspace_mb
                && cudnn_min_gflop == other.cudnn_min_gflop
                && cudnn_min_dim == other.cudnn_min_dim
                && cudnn_candidates == other.cudnn_candidates
                && workspace_search_bytes == other.workspace_search_bytes;
        }
    };

    static_assert(is_trivially_copyable_v<MatmulPlanCacheRecord>,
                  "MatmulPlanCacheRecord is written to disk as raw bytes.");

    MatmulPlanCacheRecord matmul_plan_cache_record(const MatmulPlanKey& key)
    {
        MatmulPlanCacheRecord record;
        record.key = key;
        record.candidates = clamp(env_int_or("OPENNN_LT_AUTOTUNE_CANDIDATES", 8), 1LL, 32LL);
        record.tile_tolerance = clamp(env_int_or("OPENNN_LT_TILE_TOLERANCE", 10), 0LL, 100LL);
        record.traffic_budget = clamp(env_int_or("OPENNN_LT_TRAFFIC_BUDGET", 120), 1LL, 10000LL);
        record.cross_source_gain = clamp(env_int_or("OPENNN_MATMUL_CROSS_SOURCE_GAIN", 2), 0LL, 1000LL);
        record.anchor_on_fastest = env_flag_enabled("OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST", false);
        // The same defaults and clamps as matmul_cudnn.cpp applies to them.
        record.cudnn_enabled = env_flag_enabled("OPENNN_CUDNN_MATMUL", true);
        record.cudnn_workspace_mb = clamp(env_int_or("OPENNN_CUDNN_MATMUL_WORKSPACE_MB", 32), 0LL, 4096LL);
        record.cudnn_min_gflop = clamp(env_int_or("OPENNN_CUDNN_MATMUL_MIN_GFLOP", 8), 0LL, 1000000LL);
        record.cudnn_min_dim = clamp(env_int_or("OPENNN_CUDNN_MATMUL_MIN_DIM", 128), 1LL, 1000000LL);
        record.cudnn_candidates = clamp(env_int_or("OPENNN_CUDNN_MATMUL_CANDIDATES", 0), 0LL, 4096LL);
        record.workspace_search_bytes = cublas_lt_workspace_search_bytes;
        return record;
    }

    filesystem::path matmul_plan_cache_file(const MatmulPlanCacheRecord& record)
    {
        // The record version is part of the name, so builds that write
        // different versions keep separate files instead of each rejecting
        // and re-tuning the other's on every launch.
        size_t name = MatmulPlanKeyHash{}(record.key);
        for (const long long knob : {static_cast<long long>(record.version),
                                     record.candidates, record.tile_tolerance, record.traffic_budget,
                                     record.cross_source_gain, record.anchor_on_fastest,
                                     record.cudnn_enabled, record.cudnn_workspace_mb,
                                     record.cudnn_min_gflop, record.cudnn_min_dim, record.cudnn_candidates,
                                     static_cast<long long>(record.workspace_search_bytes)})
            name ^= std::hash<long long>{}(knob) + 0x9e3779b9u + (name << 6) + (name >> 2);

        return lt_plan_cache_path() / format("{:016x}.ltplan", name);
    }

    // The plan already has its descriptor and layouts, which is what the
    // library needs to say whether the stored algorithm still applies. A file
    // from another driver, a truncated write, or an algorithm the check
    // refuses all fall through to the tuner rather than into the matmul.
    bool load_cached_matmul_plan(MatmulPlan& plan)
    {
        if (!lt_plan_cache_enabled() || lt_plan_cache_path().empty()) return false;

        PROFILE_SCOPE_HOST("lt:plan_cache_load");

        const MatmulPlanCacheRecord expected = matmul_plan_cache_record(plan.key);
        const filesystem::path file = matmul_plan_cache_file(expected);

        error_code failed;
        if (!filesystem::exists(file, failed) || failed) return false;

        ifstream stream(file, ios::binary);
        MatmulPlanCacheRecord record;
        if (!stream.read(reinterpret_cast<char*>(&record), streamsize(sizeof(record)))) return false;
        if (!record.same_tuning(expected)) return false;

        cublasLtMatmulHeuristicResult_t check{};
        if (cublasLtMatmulAlgoCheck(device::get_cublas_lt_handle(),
                                    plan.matmul_descriptor,
                                    plan.a_matrix_layout,
                                    plan.b_matrix_layout,
                                    plan.output_matrix_layout,
                                    plan.output_matrix_layout,
                                    &record.algorithm, &check) != CUBLAS_STATUS_SUCCESS
            || check.state != CUBLAS_STATUS_SUCCESS
            || check.workspaceSize > cublas_lt_workspace_search_bytes)
        {
            device::reset_last_error();
            return false;
        }

        plan.algorithm = record.algorithm;
        plan.has_algorithm = true;
        plan.workspace_bytes = check.workspaceSize;
        plan.tuned = true;

        // The tuner preferred a cuDNN engine: rebuild that one engine, which
        // becomes the plan's only cuDNN candidate. If cuDNN no longer offers
        // it -- which the directory name says cannot happen, but the check is
        // cheaper than the argument -- the cuBLASLt kernel just loaded serves
        // instead, exactly as run_lt_matmul_cached would fall back at run time.
        if (record.cudnn_candidate >= 0 && record.cudnn_plan_index >= 0)
        {
            {
                PROFILE_SCOPE_HOST("lt:plan_cache_cudnn_rebuild");
                plan.cudnn_plan = matmul::cudnn::create(
                    make_cudnn_problem(plan.key), record.cudnn_plan_index);
            }
            if (plan.cudnn_plan && matmul::cudnn::candidate_count(plan.cudnn_plan) == 1)
            {
                plan.cudnn_candidate = 0;
                plan.cudnn_workspace_bytes =
                    matmul::cudnn::candidate(plan.cudnn_plan, 0).workspace_bytes;
            }
            else
            {
                plan.release_cudnn();
            }
        }

        return true;
    }

    void store_cached_matmul_plan(const MatmulPlan& plan)
    {
        if (!lt_plan_cache_enabled() || lt_plan_cache_path().empty()) return;

        MatmulPlanCacheRecord record = matmul_plan_cache_record(plan.key);
        record.algorithm = plan.algorithm;
        record.workspace_bytes = plan.workspace_bytes;
        record.cudnn_candidate = plan.cudnn_candidate;
        record.cudnn_workspace_bytes = plan.cudnn_workspace_bytes;
        record.cudnn_plan_index = matmul::cudnn::candidate(
            plan.cudnn_plan, plan.cudnn_candidate).plan_index;

        error_code failed;
        filesystem::create_directories(lt_plan_cache_path(), failed);
        if (failed) return;

        static atomic<uint64_t> sequence{0};

        const filesystem::path file = matmul_plan_cache_file(record);
        const filesystem::path pending = file.string()
            + format(".{:x}-{}.tmp", std::hash<thread::id>{}(this_thread::get_id()), sequence++);

        {
            ofstream stream(pending, ios::binary | ios::trunc);
            if (!stream) return;
            stream.write(reinterpret_cast<const char*>(&record), streamsize(sizeof(record)));
            if (!stream) { filesystem::remove(pending, failed); return; }
        }

        filesystem::rename(pending, file, failed);
        if (failed) filesystem::remove(pending, failed);
    }

    cublasComputeType_t matmul_compute_type(cudaDataType_t a_type,
                                            cudaDataType_t b_type = CUDA_R_32F)
    {
        if (a_type == CUDA_R_16BF || b_type == CUDA_R_16BF) return CUBLAS_COMPUTE_32F_FAST_16BF;
        return device::allow_tf32() ? CUBLAS_COMPUTE_DTYPE : CUBLAS_COMPUTE_32F;
    }

    // Only used to size the tuner's scratch destination, so an unrecognised
    // type is rounded up rather than guessed: too large wastes a few bytes of
    // a block that is already megabytes, too small is a write past its end.
    size_t matmul_dtype_bytes(cudaDataType_t type)
    {
        if (type == CUDA_R_8I) return 1;
        if (type == CUDA_R_16F || type == CUDA_R_16BF) return 2;
        return 4;
    }

    void* thread_workspace(device::GraphWorkspaceKind kind, Index minimum_bytes)
    {
        if (device::active_lane() == 0)
            if (const optional<void*> graph_workspace =
                    device::graph_workspace_override(kind, minimum_bytes))
                return *graph_workspace;

        Buffer& buffer = thread_state().workspaces[static_cast<size_t>(device::active_lane())][static_cast<size_t>(kind)];
        if (minimum_bytes > buffer.byte_size() && buffer.data())
        {
            throw_if(device::cuda_allocation_growth_forbidden(),
                     "workspace growth forbidden (warmup incomplete).");
            device::synchronize(device::get_compute_stream());
        }
        const Index before = buffer.byte_size();
        void* pointer = buffer.ensure<uint8_t>(minimum_bytes);
        if (buffer.byte_size() > before)
            memory_debug::record(string("workspace.") + device::graph_workspace_labels[static_cast<size_t>(kind)],
                                 device::graph_workspace_labels[static_cast<size_t>(kind)],
                                 buffer.byte_size() - before, "high_water");
        return pointer;
    }

    vector<int> cublas_algorithm_ids(
        const vector<cublasLtMatmulHeuristicResult_t>& heuristics)
    {
        vector<int> ids;
        for(const auto& heuristic : heuristics)
        {
            int id = 0;
            size_t written = 0;
            if(cublasLtMatmulAlgoConfigGetAttribute(
                   &heuristic.algo, CUBLASLT_ALGO_CONFIG_ID,
                   &id, sizeof(id), &written) == CUBLAS_STATUS_SUCCESS
               && ranges::find(ids, id) == ids.end())
                ids.push_back(id);
        }
        return ids;
    }

    vector<int> cublas_algorithm_stages(cublasLtMatmulAlgo_t& algorithm)
    {
        size_t bytes = 0;
        vector<int> stages;
        if(cublasLtMatmulAlgoCapGetAttribute(
               &algorithm, CUBLASLT_ALGO_CAP_STAGES_IDS,
               nullptr, 0, &bytes) == CUBLAS_STATUS_SUCCESS && bytes > 0)
        {
            stages.resize(bytes / sizeof(int));
            cublasLtMatmulAlgoCapGetAttribute(
                &algorithm, CUBLASLT_ALGO_CAP_STAGES_IDS,
                stages.data(), bytes, &bytes);
        }
        if(stages.empty()) stages.push_back(CUBLASLT_MATMUL_STAGES_UNDEFINED);
        return stages;
    }

    void configure_algorithm(cublasLtMatmulAlgo_t& algorithm,
                             cublasLtMatmulAlgoConfigAttributes_t attribute,
                             int value)
    {
        cublasLtMatmulAlgoConfigSetAttribute(
            &algorithm, attribute, &value, sizeof(value));
    }

    bool append_checked_candidate(MatmulPlan& plan,
                                  cublasLtMatmulAlgo_t& algorithm,
                                  float traffic)
    {
        cublasLtMatmulHeuristicResult_t check{};
        if(cublasLtMatmulAlgoCheck(device::get_cublas_lt_handle(),
                                   plan.matmul_descriptor,
                                   plan.a_matrix_layout,
                                   plan.b_matrix_layout,
                                   plan.output_matrix_layout,
                                   plan.output_matrix_layout,
                                   &algorithm, &check) != CUBLAS_STATUS_SUCCESS
           || check.state != CUBLAS_STATUS_SUCCESS
           || check.workspaceSize > cublas_lt_workspace_search_bytes)
        {
            device::reset_last_error();
            return false;
        }
        plan.candidates.push_back({algorithm, check.workspaceSize, traffic});
        return true;
    }

    // cuBLASLt's heuristic returns a handful of candidates and ranks them by
    // expected speed, so the low-traffic kernels never appear: for the dense
    // benchmark's 1024x8192x1024 GEMM it offers eight, of which the fastest
    // (64x64) draws 96 W more than a 256x160 that is 4% slower and that the
    // heuristic does not offer at all. This adds the tiles of
    // `lt_known_tiles` as extra candidates -- lowest-traffic first, over every
    // stage and custom option each algorithm advertises -- so that
    // `autotune_matmul_plan` has something to choose between. A check costs under
    // 2 us and only the survivors are ever timed.
    void add_wide_tile_candidates(
        MatmulPlan& plan,
        const vector<cublasLtMatmulHeuristicResult_t>& heuristics)
    {
        const int m = plan.key.m;
        const int n = plan.key.n;
        const auto dtype_a = cudaDataType_t(plan.key.dtype_a);
        const auto dtype_b = cudaDataType_t(plan.key.dtype_b);
        const auto out_dtype = cudaDataType_t(plan.key.out_dtype);
        // Every candidate kept here costs four timed matmuls in
        // autotune_matmul_plan, so the overall cap stays where it was and the
        // wider option sweep is paid for out of it: 96 / 16 is six tiles,
        // each given four options over four stages instead of one option
        // over sixteen. Budget is charged on candidates kept, not on checks
        // attempted, so a tile the driver refuses outright costs nothing and
        // the next one gets the slots. Six is enough because the six probed
        // rows -- 256x160 through 128x240 -- already contain every tile at
        // or under the default OPENNN_LT_TRAFFIC_BUDGET of 0.0120, and a
        // tile above the budget can only ever be picked for being fastest,
        // which is what the driver's own heuristic already searches for.
        constexpr size_t most_added_candidates = 96;
        constexpr size_t most_candidates_per_tile = 16;
        constexpr int most_options_per_stage = 4;
        constexpr int most_custom_options = 256;

        const vector<int> algorithm_ids = cublas_algorithm_ids(heuristics);

        const size_t before = plan.candidates.size();
        size_t tile_before = before;

        const auto candidate_budget_spent = [&]
        {
            return plan.candidates.size() - before >= most_added_candidates
                || plan.candidates.size() - tile_before >= most_candidates_per_tile;
        };

        size_t tile_index = 0;
        for (const LtTile& tile : lt_known_tiles)
        {
            if (tile_index++ < probed_tiles_begin) continue;
            if (plan.candidates.size() - before >= most_added_candidates) break;

            // A tile larger than the product is priced for an output that is
            // not there -- 1/rows + 1/columns charged over rows x columns of
            // which the shape has fewer -- so the traffic the tie-break reads
            // for it is a fiction. Skip it before the sweep rather than after,
            // because the budgets above count candidates kept, not checks
            // attempted: a shape no wide tile fits otherwise pays the whole
            // enumeration (13,460 configurations in 0.86 s on this card) and
            // keeps nothing. An attention QK^T at n = 64 is exactly that.
            if (tile.rows > m || tile.columns > n) continue;

            tile_before = plan.candidates.size();

            for (const int id : algorithm_ids)
            {
                if (candidate_budget_spent()) break;

                cublasLtMatmulAlgo_t algorithm{};
                if (cublasLtMatmulAlgoInit(device::get_cublas_lt_handle(),
                                           matmul_compute_type(dtype_a, dtype_b), CUDA_R_32F,
                                           dtype_a, dtype_b, out_dtype, out_dtype,
                                           id, &algorithm) != CUBLAS_STATUS_SUCCESS)
                {
                    device::reset_last_error();
                    continue;
                }

                vector<int> stages = cublas_algorithm_stages(algorithm);

                int custom_maximum = 0;
                size_t written = 0;
                cublasLtMatmulAlgoCapGetAttribute(&algorithm, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,
                                                  &custom_maximum, sizeof(custom_maximum), &written);
                custom_maximum = min(custom_maximum, most_custom_options);

                configure_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, tile.id);
                configure_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 1);
                configure_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                    CUBLASLT_REDUCTION_SCHEME_NONE);
                configure_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0);

                // Sweep the custom options rather than keep the first that
                // checks out. They do not "differ by well under a percent" as
                // this once claimed: over the timed configurations the first
                // valid option is 55.0% slower than the best one of the same
                // tile on 128x160, 31.1% on 128x192, 30.0% on 128x240 and
                // 29.9% on 256x96. Which one wins has to be timed, so keep
                // them all -- up to most_options_per_stage -- and let
                // autotune_matmul_plan decide.
                for (const int stage : stages)
                {
                    if (candidate_budget_spent()) break;

                    configure_algorithm(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID, stage);

                    int kept_options = 0;
                    for (int custom = 0;
                         custom <= custom_maximum
                         && kept_options < most_options_per_stage
                         && !candidate_budget_spent();
                         ++custom)
                    {
                        configure_algorithm(
                            algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom);
                        const float traffic =
                            1.0f / float(tile.rows) + 1.0f / float(tile.columns);
                        if(append_checked_candidate(plan, algorithm, traffic)) ++kept_options;
                    }
                }
            }
        }
    }

    // cuDNN ships a matmul engine set of its own, and for the dense
    // benchmark's 8192x1024x1024 bf16 hidden layer it contains a kernel
    // cuBLASLt does not expose: 189.6 us at 235 W against cuBLASLt's choice
    // between 201.2 us at 169 W and 192.4 us at 265 W. An exhaustive sweep of
    // all 13,460 cuBLASLt configurations for that shape has nothing
    // comparable. matmul::cudnn::create() declines everything it has no
    // measured reason to serve -- see the guards there -- so on most shapes
    // this is one function call that returns nullptr and costs nothing.
    //
    // The candidates are appended to the SAME list the cuBLASLt algorithms
    // are in, so there is one timing loop and one selection rule rather than
    // two competing ones. They carry infinite traffic, which is not a
    // pessimistic estimate but an admission: a cuDNN engine has no tile id,
    // so tile_traffic cannot price it at all.
    void add_cudnn_candidates(MatmulPlan& plan)
    {
        plan.cudnn_plan = matmul::cudnn::create(make_cudnn_problem(plan.key));
        if (!plan.cudnn_plan) return;

        const int count = matmul::cudnn::candidate_count(plan.cudnn_plan);
        for (int candidate = 0; candidate < count; ++candidate)
        {
            const auto info = matmul::cudnn::candidate(plan.cudnn_plan, candidate);
            MatmulCandidate entry;
            entry.workspace_bytes = info.workspace_bytes;
            entry.traffic = numeric_limits<float>::infinity();
            entry.source = MatmulSource::Cudnn;
            entry.cudnn_candidate = candidate;
            plan.candidates.push_back(entry);
        }
    }

    void create_matmul_descriptors(MatmulPlan& plan)
    {
        const MatmulPlanKey& key = plan.key;
        const auto trans_a = cublasOperation_t(key.transA);
        const auto trans_b = cublasOperation_t(key.transB);
        const auto epilogue = cublasLtEpilogue_t(key.epilogue);
        const auto dtype_a = cudaDataType_t(key.dtype_a);
        const auto dtype_b = cudaDataType_t(key.dtype_b);
        const auto out_dtype = cudaDataType_t(key.out_dtype);
        CHECK_CUBLAS(cublasLtMatmulDescCreate(
            &plan.matmul_descriptor, matmul_compute_type(dtype_a, dtype_b), CUDA_R_32F));

        const auto set = [&](cublasLtMatmulDescAttributes_t attribute, const auto& value) {
            CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                plan.matmul_descriptor, attribute, &value, sizeof(value)));
        };
        set(CUBLASLT_MATMUL_DESC_TRANSA, trans_a);
        set(CUBLASLT_MATMUL_DESC_TRANSB, trans_b);
        set(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
        set(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_dtype);

        if(is_one_of(epilogue, CUBLASLT_EPILOGUE_RELU_AUX_BIAS,
                     CUBLASLT_EPILOGUE_DRELU))
            throw_if(key.m % 128 != 0,
                     "cuBLASLt ReLU bitmask epilogue requires m % 128 == 0, got {}.",
                     key.m);
        if(is_one_of(epilogue, CUBLASLT_EPILOGUE_GELU_AUX_BIAS,
                     CUBLASLT_EPILOGUE_RELU_AUX_BIAS, CUBLASLT_EPILOGUE_DRELU))
        {
            const int64_t aux_ld = key.m;
            set(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, aux_ld);
        }

        const int a_rows = trans_a == CUBLAS_OP_N ? key.m : key.k;
        const int a_cols = trans_a == CUBLAS_OP_N ? key.k : key.m;
        const int b_rows = trans_b == CUBLAS_OP_N ? key.k : key.n;
        const int b_cols = trans_b == CUBLAS_OP_N ? key.n : key.k;
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
            &plan.a_matrix_layout, dtype_a, a_rows, a_cols, key.lda ? key.lda : a_rows));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
            &plan.b_matrix_layout, dtype_b, b_rows, b_cols, key.ldb ? key.ldb : b_rows));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(
            &plan.output_matrix_layout, out_dtype, key.m, key.n, key.ldd ? key.ldd : key.m));
    }

    vector<cublasLtMatmulHeuristicResult_t> collect_cublas_candidates(MatmulPlan& plan)
    {
        cublasLtMatmulPreference_t preference = nullptr;
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&preference));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &cublas_lt_workspace_search_bytes, sizeof(cublas_lt_workspace_search_bytes)));

        const int requested = int(clamp(
            env_int_or("OPENNN_LT_AUTOTUNE_CANDIDATES", 8), 1LL, 32LL));
        vector<cublasLtMatmulHeuristicResult_t> results(static_cast<size_t>(requested));
        int returned = 0;
        const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            device::get_cublas_lt_handle(), plan.matmul_descriptor,
            plan.a_matrix_layout, plan.b_matrix_layout,
            plan.output_matrix_layout, plan.output_matrix_layout,
            preference, requested, results.data(), &returned);
        cublasLtMatmulPreferenceDestroy(preference);
        CHECK_CUBLAS(status);

        results.resize(size_t(max(returned, 0)));
        erase_if(results, [](const auto& result) {
            return result.state != CUBLAS_STATUS_SUCCESS;
        });
        for(const auto& result : results)
        {
            int tile = CUBLASLT_MATMUL_TILE_UNDEFINED;
            int split_k = 1;
            size_t written = 0;
            cublasLtMatmulAlgoConfigGetAttribute(
                &result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                &tile, sizeof(tile), &written);
            cublasLtMatmulAlgoConfigGetAttribute(
                &result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                &split_k, sizeof(split_k), &written);
            plan.candidates.push_back(
                {result.algo, result.workspaceSize, tile_traffic(tile, split_k)});
        }
        return results;
    }

    MatmulPlan& get_matmul_plan(const MatmulPlanKey& key)
    {
        auto& plans = thread_state().matmul_plans;
        auto it = plans.find(key);
        if (it != plans.end()) return it->second;

        throw_if(device::cuda_matmul_plan_creation_forbidden(),
                 "matmul plan forbidden (warmup incomplete).");

        detail::make_bounded_cache_room(plans, cublas_lt_plan_cache_capacity);

        MatmulPlan plan;
        plan.key = key;
        create_matmul_descriptors(plan);

        // A winner another process already timed is taken as it stands, so
        // every process on this card runs the same kernel for this shape and
        // the heuristic query below is not even made.
        if (!lt_deterministic_selection() && load_cached_matmul_plan(plan))
            return plans.emplace(key, std::move(plan)).first->second;

        const vector<cublasLtMatmulHeuristicResult_t> heuristics =
            collect_cublas_candidates(plan);

        // The wide tiles exist to give the tuner something to time; with the
        // tuner off they would only be enumerated to be discarded.
        if (!lt_deterministic_selection())
            add_wide_tile_candidates(plan, heuristics);

        // The untuned default is the heuristic's first cuBLASLt algorithm,
        // exactly as before. cuDNN candidates are appended after it and are
        // never the default: cuDNN's own heuristic puts a 226 us engine first
        // for the shape this path was added for, where the best configuration
        // runs at 189.6, so an untimed cuDNN pick would be a downgrade. They
        // are only ever selected by autotune_matmul_plan, on measured time.
        if (!plan.candidates.empty())
        {
            plan.algorithm = plan.candidates.front().algorithm;
            plan.has_algorithm = true;
            plan.workspace_bytes = plan.candidates.front().workspace_bytes;
        }

        // cuDNN engines, like the wide tiles, are only ever chosen on measured
        // time, so with the tuner off there is nothing for them to be.
        if (!lt_deterministic_selection())
            add_cudnn_candidates(plan);

        plan.tuned = plan.candidates.size() <= 1 || lt_deterministic_selection();
        if (plan.tuned)
        {
            plan.release_cudnn();
            plan.candidates.clear();
        }

        return plans.emplace(key, std::move(plan)).first->second;
    }

    // --- verifying a candidate from another library -----------------------
    //
    // A cuBLASLt candidate is the same kernel family answering the same
    // descriptor, so the tuner has never had to ask whether a candidate is
    // CORRECT, only which is fastest. A cuDNN candidate is a different
    // library reading the same memory through strides this file derived by
    // hand, and a stride derived wrongly does not fail: it returns a
    // confidently wrong tensor. So every cuDNN candidate is checked against
    // the cuBLASLt answer on the caller's own data before it is timed, and a
    // candidate that disagrees is dropped rather than ranked.
    //
    // The check reads three windows of the destination -- head, middle and
    // tail -- rather than all of it. That is not a sample of a random
    // process: an operand read through a wrong stride, or a bias broadcast
    // along the wrong axis, is wrong in nearly every element, so a few
    // thousand elements settle it, while the windows at the two ends are
    // where a tail-handling bug would hide. It costs three 8 KB copies per
    // candidate and no device memory at all, because it reuses the
    // destination the tuner is already writing into -- run_lt_matmul_cached
    // overwrites it with the real call the moment the tuner returns.
    constexpr size_t verification_window_elements = 4096;

    float bf16_bits_to_float(uint16_t bits)
    {
        const uint32_t widened = uint32_t(bits) << 16;
        float value = 0.0f;
        memcpy(&value, &widened, sizeof(value));
        return value;
    }

    bool read_verification_windows(const void* source, size_t elements,
                                   vector<uint16_t>& windows, DeviceStream stream)
    {
        if (elements == 0) return false;

        const size_t window = min(elements, verification_window_elements);
        const size_t offsets[3] = {0, (elements - window) / 2, elements - window};

        windows.resize(window * 3);

        for (size_t index = 0; index < 3; ++index)
            if (cudaMemcpyAsync(windows.data() + index * window,
                                static_cast<const uint16_t*>(source) + offsets[index],
                                window * sizeof(uint16_t),
                                cudaMemcpyDeviceToHost, stream) != cudaSuccess)
            {
                device::reset_last_error();
                return false;
            }

        if (cudaStreamSynchronize(stream) != cudaSuccess)
        {
            device::reset_last_error();
            return false;
        }

        return true;
    }

    bool verification_windows_agree(const vector<uint16_t>& reference,
                                    const vector<uint16_t>& candidate)
    {
        if (reference.empty() || reference.size() != candidate.size()) return false;

        double sum_of_squares = 0.0;
        for (const uint16_t bits : reference)
        {
            const double value = bf16_bits_to_float(bits);
            sum_of_squares += value * value;
        }
        const double root_mean_square = sqrt(sum_of_squares / double(reference.size()));

        // bf16 carries about three decimal digits, so 2e-2 is the gate the
        // probe used against the same reference. Dividing by the tensor's own
        // RMS rather than by each element keeps a sample that landed on a
        // ReLU zero from manufacturing a failure.
        const double floor_value = max(1e-6, 0.01 * root_mean_square);

        for (size_t index = 0; index < reference.size(); ++index)
        {
            const double want = bf16_bits_to_float(reference[index]);
            const double got  = bf16_bits_to_float(candidate[index]);
            if (fabs(want - got) / max(floor_value, fabs(want)) > 2e-2) return false;
        }

        return true;
    }

    struct MatmulInvocation
    {
        const void* a;
        const void* b;
        const void* c;
        void* d;
        const void* bias;
        float alpha;
        float beta;
        size_t destination_bytes;
        DeviceStream stream;
    };

    struct CandidateRunner
    {
        MatmulPlan& plan;
        const void* a;
        const void* b;
        const void* c;
        void* d;
        const void* bias;
        float alpha;
        float beta;
        void* workspace;
        void* destination;
        DeviceStream stream;

        cublasStatus_t launch_lt(size_t index) const
        {
            const MatmulCandidate& candidate = plan.candidates[index];
            return cublasLtMatmul(device::get_cublas_lt_handle(), plan.matmul_descriptor,
                                  &alpha, a, plan.a_matrix_layout, b, plan.b_matrix_layout,
                                  &beta, c, plan.output_matrix_layout,
                                  destination, plan.output_matrix_layout,
                                  &candidate.algorithm, workspace,
                                  candidate.workspace_bytes, stream);
        }

        bool launch_cudnn(size_t index) const
        {
            return matmul::cudnn::run(plan.cudnn_plan,
                                     plan.candidates[index].cudnn_candidate,
                                     a, b, bias, d, workspace);
        }
    };

    struct TimingResult
    {
        size_t index;
        float milliseconds;
    };

    template<typename Launch>
    optional<float> time_candidate(Launch&& launch,
                                   const device::CudaEvent& start,
                                   const device::CudaEvent& stop,
                                   DeviceStream stream,
                                   int timed_runs)
    {
        if(!launch()) return nullopt;
        device::record_event(start.get(), stream);
        bool ok = true;
        for(int run = 0; run < timed_runs && ok; ++run) ok = launch();
        device::record_event(stop.get(), stream);
        device::synchronize_event(stop.get());
        if(!ok) return nullopt;

        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()));
        return milliseconds;
    }

    TimingResult time_lt_candidates(const CandidateRunner& call,
                                        vector<float>& times,
                                        const device::CudaEvent& start,
                                        const device::CudaEvent& stop,
                                        int timed_runs)
    {
        TimingResult best{call.plan.candidates.size(), numeric_limits<float>::infinity()};
        for(size_t index = 0; index < call.plan.candidates.size(); ++index)
        {
            if(call.plan.candidates[index].source != MatmulSource::CublasLt) continue;
            const optional<float> milliseconds = time_candidate(
                [&] { return call.launch_lt(index) == CUBLAS_STATUS_SUCCESS; },
                start, stop, call.stream, timed_runs);
            if(!milliseconds)
            {
                device::reset_last_error();
                continue;
            }
            times[index] = *milliseconds;
            if(*milliseconds < best.milliseconds) best = {index, *milliseconds};
        }
        return best;
    }

    TimingResult time_cudnn_candidates(const CandidateRunner& call,
                                           vector<float>& times,
                                           TimingResult best,
                                           size_t reference_index,
                                           size_t destination_bytes,
                                           bool destination_is_operand,
                                           const device::CudaEvent& start,
                                           const device::CudaEvent& stop,
                                           int timed_runs)
    {
        if(!call.plan.cudnn_plan) return best;

        const size_t elements = destination_bytes / sizeof(uint16_t);
        vector<uint16_t> reference;
        vector<uint16_t> candidate;
        const bool comparable = reference_index < call.plan.candidates.size()
            && !destination_is_operand && elements > 0
            && call.launch_lt(reference_index) == CUBLAS_STATUS_SUCCESS
            && read_verification_windows(call.d, elements, reference, call.stream);
        if(!comparable) device::reset_last_error();

        for(size_t index = 0; comparable && index < call.plan.candidates.size(); ++index)
        {
            const MatmulCandidate& option = call.plan.candidates[index];
            if(option.source != MatmulSource::Cudnn || !call.launch_cudnn(index)) continue;
            if(cudaStreamSynchronize(call.stream) != cudaSuccess)
            {
                device::reset_last_error();
                continue;
            }
            if(!read_verification_windows(call.d, elements, candidate, call.stream)) continue;
            if(!verification_windows_agree(reference, candidate))
            {
                logging::warning() << "cudnn matmul: candidate "
                    << matmul::cudnn::candidate(
                           call.plan.cudnn_plan, option.cudnn_candidate).name
                    << " disagreed with cuBLASLt and was dropped.\n";
                continue;
            }

            const optional<float> milliseconds = time_candidate(
                [&] { return call.launch_cudnn(index); },
                start, stop, call.stream, timed_runs);
            if(!milliseconds)
            {
                device::reset_last_error();
                continue;
            }
            times[index] = *milliseconds;
            if(*milliseconds < best.milliseconds) best = {index, *milliseconds};
        }
        return best;
    }

    size_t select_energy_aware_lt(const MatmulPlan& plan,
                                  span<const float> times,
                                  TimingResult fastest,
                                  float tolerance,
                                  float traffic_budget)
    {
        size_t best = fastest.index;
        if(tolerance <= 0.0f || best >= plan.candidates.size()) return best;

        const float allowed_ms = fastest.milliseconds * (1.0f + tolerance);
        const float allowed_energy = fastest.milliseconds
                                   * tile_power_watts(plan.candidates[best].traffic);
        float least_traffic = plan.candidates[best].traffic;
        for(size_t index = 0; index < plan.candidates.size(); ++index)
        {
            const MatmulCandidate& candidate = plan.candidates[index];
            if(candidate.source != MatmulSource::CublasLt
               || candidate.traffic > traffic_budget
               || candidate.traffic > least_traffic
               || times[index] > allowed_ms
               || times[index] * tile_power_watts(candidate.traffic) >= allowed_energy)
                continue;

            constexpr float resolvable = 0.02f;
            if(candidate.traffic == least_traffic
               && times[index] >= times[best] * (1.0f - resolvable))
                continue;
            least_traffic = candidate.traffic;
            best = index;
        }
        return best;
    }

    size_t select_cross_source(const MatmulPlan& plan,
                               span<const float> times,
                               size_t best_lt,
                               TimingResult fastest_lt,
                               TimingResult fastest_any,
                               float tolerance)
    {
        if(tolerance <= 0.0f) return fastest_any.index;
        if(fastest_any.index >= plan.candidates.size()
           || plan.candidates[fastest_any.index].source != MatmulSource::Cudnn
           || best_lt >= plan.candidates.size())
            return best_lt;

        const float gain = float(clamp(
            env_int_or("OPENNN_MATMUL_CROSS_SOURCE_GAIN", 2), 0LL, 1000LL)) / 100.0f;
        const bool anchor_fastest = env_flag_enabled(
            "OPENNN_MATMUL_CROSS_SOURCE_ANCHOR_FASTEST", false);
        const float anchor = anchor_fastest ? fastest_lt.milliseconds : times[best_lt];
        return fastest_any.milliseconds * (1.0f + gain) < anchor
             ? fastest_any.index : best_lt;
    }

    void autotune_matmul_plan(MatmulPlan& plan, const MatmulInvocation& invocation)
    {
        PROFILE_SCOPE_HOST("lt:autotune");
        if(plan.candidates.size() <= 1)
        {
            plan.release_cudnn();
            plan.candidates.clear();
            plan.tuned = true;
            return;
        }

        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        if(cudaStreamIsCapturing(invocation.stream, &capture) != cudaSuccess
           || capture != cudaStreamCaptureStatusNone)
            return device::reset_last_error();
        plan.tuned = true;

        // A cuDNN graph has no alpha node, so only alpha == 1 is comparable.
        if(plan.cudnn_plan && invocation.alpha != 1.0f)
        {
            plan.release_cudnn();
            erase_if(plan.candidates, [](const MatmulCandidate& candidate) {
                return candidate.source == MatmulSource::Cudnn;
            });
            if(plan.candidates.size() <= 1)
            {
                plan.candidates.clear();
                return;
            }
        }

        if(device::lanes_available() > 1) device::synchronize();
        size_t workspace_bytes = 0;
        for(const MatmulCandidate& candidate : plan.candidates)
            workspace_bytes = max(workspace_bytes, candidate.workspace_bytes);

        // When D is also C, tune into scratch to avoid repeated beta accumulation.
        const bool destination_is_operand =
            invocation.beta != 0.0f && invocation.c == invocation.d;
        constexpr size_t alignment = 256;
        const size_t destination_offset =
            (workspace_bytes + alignment - 1) / alignment * alignment;
        void* workspace = nullptr;
        try
        {
            workspace = ensure_shared_scratch(destination_is_operand
                      ? destination_offset + invocation.destination_bytes : workspace_bytes);
        }
        catch(const exception&)
        {
            plan.release_cudnn();
            plan.candidates.clear();
            return;
        }

        void* destination = destination_is_operand
                          ? static_cast<char*>(workspace) + destination_offset : invocation.d;
        CandidateRunner call{plan, invocation.a, invocation.b, invocation.c,
                             invocation.d, invocation.bias, invocation.alpha,
                             invocation.beta, workspace, destination, invocation.stream};
        const device::CudaEvent start(cudaEventDefault), stop(cudaEventDefault);
        constexpr int timed_runs = 7;
        vector<float> times(plan.candidates.size(), numeric_limits<float>::infinity());

        const TimingResult fastest_lt =
            time_lt_candidates(call, times, start, stop, timed_runs);
        const TimingResult fastest_any = time_cudnn_candidates(
            call, times, fastest_lt, fastest_lt.index, invocation.destination_bytes,
            destination_is_operand, start, stop, timed_runs);

        // The energy model is valid only within cuBLASLt. Cross-library
        // selection uses measured time because cuDNN does not expose tile traffic.
        const float tolerance = float(clamp(
            env_int_or("OPENNN_LT_TILE_TOLERANCE", 10), 0LL, 100LL)) / 100.0f;
        const float traffic_budget = float(clamp(
            env_int_or("OPENNN_LT_TRAFFIC_BUDGET", 120), 1LL, 10000LL)) / 10000.0f;
        const size_t best_lt = select_energy_aware_lt(
            plan, times, fastest_lt, tolerance, traffic_budget);

        if(best_lt < plan.candidates.size())
        {
            plan.algorithm = plan.candidates[best_lt].algorithm;
            plan.workspace_bytes = plan.candidates[best_lt].workspace_bytes;
            plan.has_algorithm = true;
        }

        const size_t chosen = select_cross_source(
            plan, times, best_lt, fastest_lt, fastest_any, tolerance);
        if(chosen < plan.candidates.size()
           && plan.candidates[chosen].source == MatmulSource::Cudnn)
        {
            plan.cudnn_candidate = plan.candidates[chosen].cudnn_candidate;
            plan.cudnn_workspace_bytes = plan.candidates[chosen].workspace_bytes;
            if(matmul::cudnn::verbose())
                logging::warning() << "cudnn matmul: chose "
                    << matmul::cudnn::candidate(
                           plan.cudnn_plan, plan.cudnn_candidate).name
                    << " at " << times[chosen] * 1000.0f / timed_runs
                    << " us against cuBLASLt's "
                    << (best_lt < plan.candidates.size()
                        ? times[best_lt] * 1000.0f / timed_runs : 0.0f)
                    << " us\n";
        }
        else
            plan.release_cudnn();

        plan.candidates.clear();
        if(fastest_lt.milliseconds < numeric_limits<float>::infinity())
            store_cached_matmul_plan(plan);
    }
}

void* ensure_workspace_bytes(device::GraphWorkspaceKind kind, Index bytes)
{
    return thread_workspace(kind, bytes);
}

void release_thread_workspaces()
{
    device::synchronize(device::get_compute_stream());
    for (auto& lane : thread_state().workspaces)
        for (Buffer& buffer : lane)
            buffer.resize_bytes(0, Device::CUDA);
}

string device::lt_plan_cache_directory() noexcept
{
    try
    {
        if (!lt_plan_cache_enabled()) return {};
        return lt_plan_cache_path().string();
    }
    catch (const exception&)
    {
        return {};
    }
}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type)
{
    if (input.get_type() == target_type) return input.get_data();

    if (input.is_fp32() && target_type == Type::BF16)
    {
        bfloat16* dst = ensure_workspace<bfloat16>(device::GraphWorkspaceKind::Bf16Input, input.size());
        cast_fp32_to_bf16(input.size(), input.as<float>(), dst);
        return dst;
    }

    if (input.is_bf16() && target_type == Type::FP32)
    {
        float* dst = ensure_bf16_to_fp32_workspace(input.size());
        cast_bf16_to_fp32(input.size(), input.as<bfloat16>(), dst);
        return dst;
    }

    throw runtime_error("data_for_gemm_dtype: unsupported type pair");
}

const void* bias_for_gemm_bf16(const TensorView& bias)
{

    bfloat16* dst = ensure_bf16_gradient_workspace(bias.size());
    cast_fp32_to_bf16(bias.size(), bias.as<float>(), dst);
    return dst;
}

static cublasLtEpilogue_t native_epilogue(const LinearEpilogue epilogue)
{
    switch(epilogue)
    {
    case LinearEpilogue::Default:      return CUBLASLT_EPILOGUE_DEFAULT;
    case LinearEpilogue::Relu:         return CUBLASLT_EPILOGUE_RELU;
    case LinearEpilogue::Bias:         return CUBLASLT_EPILOGUE_BIAS;
    case LinearEpilogue::ReluBias:     return CUBLASLT_EPILOGUE_RELU_BIAS;
    case LinearEpilogue::ReluAuxBias:  return CUBLASLT_EPILOGUE_RELU_AUX_BIAS;
    case LinearEpilogue::DRelu:        return CUBLASLT_EPILOGUE_DRELU;
    case LinearEpilogue::GeluAuxBias:  return CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
    case LinearEpilogue::BiasGradient: return CUBLASLT_EPILOGUE_BGRADA;
    }
    return CUBLASLT_EPILOGUE_DEFAULT;
}

void run_lt_matmul_cached(
    int m, int n, int k,
    BlasOperation transA,
    BlasOperation transB,
    LinearEpilogue epilogue,
    const void* a_data, const void* b_data, void* d_data,
    const void* bias_pointer,
    DeviceDataType dtype_a,
    DeviceDataType dtype_b,
    DeviceDataType out_dtype,
    const void* aux_pointer,
    const void* addend,
    float alpha,
    float beta,
    int lda, int ldb, int ldd)
{
    // beta used to be derived from this pointer, so the combination could not
    // be asked for; now that it is a parameter, an addend at beta 0 is an
    // operand the kernel reads and discards. Say so rather than compute the
    // wrong sum quietly.
    throw_if(addend && beta == 0.0f,
             "run_lt_matmul_cached: an addend with beta 0 is a discarded operand.");

    const MatmulPlanKey key{
        m, n, k, int(transA), int(transB), int(native_epilogue(epilogue)),
        int(dtype_a), int(dtype_b), int(out_dtype), lda, ldb, ldd,
        int(beta == 0.0f), int(device::allow_tf32())
    };
    MatmulPlan& plan = get_matmul_plan(key);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));

    if (aux_pointer)
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, &aux_pointer, sizeof(aux_pointer)));

    // C is the addend when there is one and the destination otherwise; at
    // beta 0 the kernel reads neither, which is why aliasing it to D was
    // always safe and still is.
    const void* const c_data = addend ? addend : d_data;

    if (!plan.tuned)
        autotune_matmul_plan(plan, {
            a_data, b_data, c_data, d_data, bias_pointer, alpha, beta,
            size_t(ldd ? ldd : m) * size_t(n) * matmul_dtype_bytes(out_dtype),
            device::get_compute_stream()
        });

    // The cuDNN overlay, when the tuner chose one. Three things are checked
    // here rather than trusted from the plan, because none of them is part of
    // MatmulPlanKey and so none of them is guaranteed to match the call that
    // tuned this plan: the cuDNN graph has no alpha node, no beta node and no
    // second output, so alpha must be 1, the epilogue must not be writing an
    // auxiliary tensor, and beta is already pinned by the key. A plan reached
    // with any other combination silently falls through to cuBLASLt below,
    // which is the same kernel it would have run yesterday.
    //
    // matmul::cudnn::run returns false instead of throwing, so an engine that
    // stops working -- a driver change, a workspace that could not be raised
    // -- costs one wasted launch attempt and then the cuBLASLt kernel, not an
    // exception in the middle of a training step.
    if (plan.cudnn_plan && plan.cudnn_candidate >= 0
        && alpha == 1.0f && beta == 0.0f && !aux_pointer)
    {
        void* cudnn_workspace = nullptr;
        bool have_workspace = true;
        try
        {
            cudnn_workspace = ensure_shared_scratch(plan.cudnn_workspace_bytes);
        }
        catch (const exception&)
        {
            have_workspace = false;
        }

        if (have_workspace
            && matmul::cudnn::run(plan.cudnn_plan, plan.cudnn_candidate,
                                 a_data, b_data, bias_pointer, d_data, cudnn_workspace))
            return;

        // It declined once; it will decline again. Give the shape back to
        // cuBLASLt for the life of the plan rather than pay the failed launch
        // on every forward pass.
        plan.release_cudnn();
    }

    CHECK_CUBLAS(cublasLtMatmul(device::get_cublas_lt_handle(),
                                plan.matmul_descriptor,
                                &alpha,
                                a_data, plan.a_matrix_layout,
                                b_data, plan.b_matrix_layout,
                                &beta,
                                c_data, plan.output_matrix_layout,
                                d_data, plan.output_matrix_layout,
                                plan.has_algorithm ? &plan.algorithm : nullptr,
                                ensure_shared_scratch(plan.workspace_bytes),
                                plan.workspace_bytes,
                                device::get_compute_stream()));
}

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void* A, cudaDataType_t Atype, int lda, long long stride_a,
                               const void* B, cudaDataType_t Btype, int ldb, long long stride_b,
                               void* C, cudaDataType_t Ctype, int ldc, long long stride_c,
                               int batch_count,
                               float alpha, float beta)
{
    const cublasComputeType_t compute = matmul_compute_type(Atype, Btype);
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(device::get_cublas_handle(),
                                            transa, transb,
                                            m, n, k,
                                            &alpha,
                                            A, Atype, lda, stride_a,
                                            B, Btype, ldb, stride_b,
                                            &beta,
                                            C, Ctype, ldc, stride_c,
                                            batch_count,
                                            compute,
                                            CUBLAS_GEMM_DEFAULT));
}

}

#else

namespace opennn
{

void* ensure_workspace_bytes(device::GraphWorkspaceKind, Index) OPENNN_CUDA_STUB_BODY(ensure_workspace_bytes)

void release_thread_workspaces() OPENNN_CUDA_STUB_BODY(release_thread_workspaces)

string device::lt_plan_cache_directory() noexcept { return {}; }

const void* data_for_gemm_dtype(const TensorView&, Type) OPENNN_CUDA_STUB_BODY(data_for_gemm_dtype)

const void* bias_for_gemm_bf16(const TensorView&) OPENNN_CUDA_STUB_BODY(bias_for_gemm_bf16)

void run_lt_matmul_cached(int, int, int,
                          BlasOperation,
                          BlasOperation,
                          LinearEpilogue,
                          const void*, const void*, void*,
                          const void*,
                          DeviceDataType,
                          DeviceDataType,
                          DeviceDataType,
                          const void*,
                          const void*,
                          float,
                          float,
                          int, int, int) OPENNN_CUDA_STUB_BODY(run_lt_matmul_cached)

void gemm_strided_batched_cuda(cublasOperation_t, cublasOperation_t,
                               int, int, int,
                               const void*, cudaDataType_t, int, long long,
                               const void*, cudaDataType_t, int, long long,
                               void*, cudaDataType_t, int, long long,
                               int,
                               float, float) OPENNN_CUDA_STUB_BODY(gemm_strided_batched_cuda)

}

#endif
