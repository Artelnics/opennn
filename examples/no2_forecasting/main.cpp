//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R E C A S T I N G   B E N C H M A R K :   R E C U R R E N T   v s   L S T M
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../../opennn/time_series_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/loss.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/configuration.h"
#include "../../opennn/adaptive_moment_estimation.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/long_short_term_memory_layer.h"
#include "../../opennn/recurrent_layer.h"
#include "../../opennn/random_utilities.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace opennn;
using Clock = std::chrono::steady_clock;

namespace
{
const string DATA_DIR = "../data/";

constexpr int     SEED_COUNT = 1;
constexpr unsigned SEEDS[SEED_COUNT] = {42};

string format_seconds(double s)
{
    int total = int(s);
    const int h = total / 3600; total %= 3600;
    const int m = total / 60;   const int sec = total % 60;
    std::ostringstream os;
    if (h > 0) os << h << "h";
    if (h > 0 || m > 0) os << (h > 0 && m < 10 ? "0" : "") << m << "m";
    os << (sec < 10 && (h > 0 || m > 0) ? "0" : "") << sec << "s";
    return os.str();
}

struct ScenarioProgress
{
    int done  = 0;
    int total = 0;
    Clock::time_point started;

    void start(int total_runs)
    {
        done    = 0;
        total   = total_runs;
        started = Clock::now();
        draw();
    }

    void tick()
    {
        ++done;
        draw();
    }

    void finish()
    {
        std::cout << "\n";
        std::cout.flush();
    }

    void draw() const
    {
        constexpr int W = 30;
        const float frac = total ? float(done) / float(total) : 0.0f;
        const int filled = int(frac * W + 0.5f);
        const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
        const double eta = (done > 0)
            ? elapsed * double(total - done) / double(done)
            : 0.0;

        std::cout << "\r    [";
        for (int i = 0; i < W; ++i)
            std::cout << (i < filled ? '#' : '.');
        std::cout << "] " << done << "/" << total
                  << "  elapsed=" << format_seconds(elapsed)
                  << "  ETA=" << format_seconds(eta) << "   ";
        std::cout.flush();
    }
};

ScenarioProgress g_bar;

struct Scenario
{
    string  id;
    string  description;
    string  csv_file;
    string  separator;
    Index   past;
    Index   future;
    bool    multi_target;
    vector<Index> extra_targets;
    Shape   hidden;
    float   learning_rate;
    Index   batch_size;
    Index   max_epochs;
    Index   patience;   // per-scenario early-stop patience
};

struct RunResult
{
    Index   params = 0;
    Index   epochs = 0;
    float   train_err = std::numeric_limits<float>::quiet_NaN();
    float   val_err   = std::numeric_limits<float>::quiet_NaN();
    float   test_rmse = std::numeric_limits<float>::quiet_NaN();
    float   test_rmse_rel = std::numeric_limits<float>::quiet_NaN();
    double  seconds = 0.0;
    bool    restored_best = false;
    string  notes;
};

struct AggregatedResult
{
    string net;
    float  test_rmse_mean = std::numeric_limits<float>::quiet_NaN();
    float  test_rmse_std  = std::numeric_limits<float>::quiet_NaN();
    float  test_rmse_best = std::numeric_limits<float>::quiet_NaN();
    float  val_err_mean   = std::numeric_limits<float>::quiet_NaN();
    float  test_rmse_rel_mean = std::numeric_limits<float>::quiet_NaN();
    double time_mean = 0.0;
    Index  epochs_mean = 0;
    Index  params = 0;
    int    successful_runs = 0;
};

unique_ptr<TimeSeriesDataset> load_dataset(const Scenario& s)
{
    auto ds = make_unique<TimeSeriesDataset>(DATA_DIR + s.csv_file,
                                             s.separator,
                                             /*has_header=*/false,
                                             /*has_sample_ids=*/false);
    for (Index col : s.extra_targets)
        ds->set_variable_role(col, "InputTarget");

    ds->set_past_time_steps(s.past);
    ds->set_future_time_steps(s.future);
    ds->set_multi_target(s.multi_target);
    return ds;
}

RunResult train_one(NeuralNetwork* nn,
                    TimeSeriesDataset* ds,
                    const Scenario& s)
{
    RunResult r;
    try
    {
        TrainingStrategy strategy(nn, ds);
        strategy.set_loss("MeanSquaredError");
        strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = static_cast<AdaptiveMomentEstimation*>(strategy.get_optimization_algorithm());
        adam->set_learning_rate(s.learning_rate);
        adam->set_batch_size(s.batch_size);
        adam->set_maximum_epochs(s.max_epochs);
        adam->set_maximum_validation_failures(s.patience);
        adam->set_display(false);

        const auto t0 = Clock::now();
        const TrainingResults results = strategy.train();
        const auto t1 = Clock::now();

        r.params        = nn->get_parameters_number();
        r.epochs        = results.get_epochs_number();
        r.train_err     = results.get_training_error();
        r.val_err       = results.get_validation_error();
        r.seconds       = std::chrono::duration<double>(t1 - t0).count();
        r.restored_best = results.restored_best_parameters;

        try
        {
            TestingAnalysis ta(nn, ds);
            const VectorR errs = ta.calculate_errors("Testing");
            if (errs.size() >= 3) r.test_rmse = errs(2);
        }
        catch (const std::exception& e) { r.notes = e.what(); }

        if (std::isfinite(r.test_rmse))
        {
            const vector<Index> testing_idx = ds->get_sample_indices("Testing");
            const vector<Index> target_idx  = ds->get_feature_indices("Target");
            const Index target_width = ds->get_target_shape().size();
            if (!testing_idx.empty() && target_width > 0)
            {
                MatrixR targets(testing_idx.size(), target_width);
                ds->fill_targets(testing_idx, target_idx, targets.data(),
                                 /*is_training=*/false, /*parallelize=*/true);
                const float range = targets.maxCoeff() - targets.minCoeff();
                if (range > 0.0f) r.test_rmse_rel = r.test_rmse / range;
            }
        }
    }
    catch (const std::exception& e)
    {
        r.notes = string("EXCEPTION: ") + e.what();
    }
    return r;
}

template <typename Builder>
AggregatedResult run_multi_seed(const Scenario& s,
                                const string& net_label,
                                Builder build)
{
    AggregatedResult agg;
    agg.net = net_label;

    vector<float>  rmse_vals, rmse_rel_vals, val_vals;
    vector<double> time_vals;
    vector<Index>  epoch_vals;

    for (unsigned seed : SEEDS)
    {
        set_seed(seed);
        auto ds = load_dataset(s);
        auto nn = build(*ds);
        const RunResult r = train_one(nn.get(), ds.get(), s);

        g_bar.tick();

        if (std::isfinite(r.test_rmse) && std::isfinite(r.val_err))
        {
            rmse_vals.push_back(r.test_rmse);
            val_vals.push_back(r.val_err);
            time_vals.push_back(r.seconds);
            epoch_vals.push_back(r.epochs);
            if (std::isfinite(r.test_rmse_rel)) rmse_rel_vals.push_back(r.test_rmse_rel);
            agg.params = r.params;
        }
    }

    if (rmse_vals.empty()) return agg;

    auto mean = [](const auto& v) -> float {
        return float(std::accumulate(v.begin(), v.end(), 0.0) / v.size());
    };
    auto stddev = [&](const vector<float>& v, float m) -> float {
        if (v.size() < 2) return 0.0f;
        double acc = 0.0;
        for (float x : v) acc += double(x - m) * double(x - m);
        return float(std::sqrt(acc / (v.size() - 1)));
    };

    agg.successful_runs    = int(rmse_vals.size());
    agg.test_rmse_mean     = mean(rmse_vals);
    agg.test_rmse_std      = stddev(rmse_vals, agg.test_rmse_mean);
    agg.test_rmse_best     = *std::min_element(rmse_vals.begin(), rmse_vals.end());
    agg.val_err_mean       = mean(val_vals);
    if (!rmse_rel_vals.empty()) agg.test_rmse_rel_mean = mean(rmse_rel_vals);
    agg.time_mean          = std::accumulate(time_vals.begin(), time_vals.end(), 0.0) / time_vals.size();
    agg.epochs_mean        = Index(std::accumulate(epoch_vals.begin(), epoch_vals.end(), Index(0)) / Index(epoch_vals.size()));
    return agg;
}

void print_agg(const AggregatedResult& a)
{
    std::cout << "    " << std::left << std::setw(10) << a.net
              << "  params=" << std::right << std::setw(6) << a.params
              << "  ep_mean=" << std::setw(4) << a.epochs_mean
              << "  val_mean=" << std::scientific << std::setprecision(3) << a.val_err_mean
              << "  test_rmse=" << a.test_rmse_mean
              << " ± " << a.test_rmse_std
              << "  best=" << a.test_rmse_best;
    if (std::isfinite(a.test_rmse_rel_mean))
        std::cout << "  rmse%=" << std::fixed << std::setprecision(2)
                  << (100.0f * a.test_rmse_rel_mean);
    std::cout << "  time=" << std::fixed << std::setprecision(2) << a.time_mean << "s\n";
}

struct ScenarioVerdict
{
    string id;
    AggregatedResult rec;
    AggregatedResult lstm;
    string winner = "n/a";   // best test_rmse: "Recurrent" / "LSTM" / "n/a"
};

namespace
{
auto build_recurrent(const Scenario& s)
{
    return [&s](TimeSeriesDataset& ds) {
        return std::make_unique<ForecastingNetwork>(
            ds.get_input_shape(), s.hidden, ds.get_target_shape());
    };
}

auto build_lstm(const Scenario& s)
{
    return [&s](TimeSeriesDataset& ds) {
        return std::make_unique<ForecastingLstmNetwork>(
            ds.get_input_shape(), s.hidden, ds.get_target_shape());
    };
}

string pick_winner(const AggregatedResult& a, const AggregatedResult& b,
                   const string& a_name, const string& b_name)
{
    if (!std::isfinite(a.test_rmse_mean) || !std::isfinite(b.test_rmse_mean))
        return "n/a";
    return (a.test_rmse_mean <= b.test_rmse_mean) ? a_name : b_name;
}
} // namespace

ScenarioVerdict run_scenario(const Scenario& s)
{
    std::cout << "\n=== " << s.id << "  " << s.description << " ===\n";
    std::cout << "    dataset=" << s.csv_file
              << "  past="    << s.past
              << "  future="  << s.future
              << "  hidden_layers=" << s.hidden.rank
              << "  epochs<=" << s.max_epochs
              << "  patience=" << s.patience
              << "  seeds="   << SEED_COUNT
              << "  lr="      << s.learning_rate << "\n";

    g_bar.start(/*total_runs=*/ 2 * SEED_COUNT);

    auto rec_agg  = run_multi_seed(s, "Recurrent", build_recurrent(s));
    auto lstm_agg = run_multi_seed(s, "LSTM",      build_lstm(s));

    g_bar.finish();

    print_agg(rec_agg);
    print_agg(lstm_agg);

    ScenarioVerdict v;
    v.id     = s.id;
    v.rec    = rec_agg;
    v.lstm   = lstm_agg;
    v.winner = pick_winner(rec_agg, lstm_agg, "Recurrent", "LSTM");

    std::cout << "    winner: " << v.winner;
    if (std::isfinite(rec_agg.test_rmse_mean) && std::isfinite(lstm_agg.test_rmse_mean)
        && rec_agg.test_rmse_mean > 0.0f)
    {
        const float delta_pct = 100.0f *
            (rec_agg.test_rmse_mean - lstm_agg.test_rmse_mean) / rec_agg.test_rmse_mean;
        std::cout << "  (LSTM test_rmse vs Recurrent: "
                  << std::fixed << std::setprecision(1) << delta_pct << "%)";
    }
    std::cout << "\n";
    return v;
}

// Per-phase recap (one device at a time).
void print_phase_summary(const vector<ScenarioVerdict>& vs, const string& phase)
{
    std::cout << "\n\n";
    std::cout << "===============================================================\n";
    std::cout << "      P H A S E   S U M M A R Y   :   " << phase << "\n";
    std::cout << "===============================================================\n";

    std::cout << std::left << std::setw(8) << "Scen"
              << std::right
              << std::setw(14) << "Rec rmse"
              << std::setw(14) << "LSTM rmse"
              << std::setw(11) << "Rec(s)"
              << std::setw(11) << "LSTM(s)"
              << std::setw(13) << "Winner"
              << "\n";
    std::cout << std::string(71, '-') << "\n";

    int lstm_wins = 0, total = 0;
    for (const auto& v : vs)
    {
        std::cout << std::left << std::setw(8) << v.id
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(14) << v.rec.test_rmse_mean
                  << std::setw(14) << v.lstm.test_rmse_mean
                  << std::fixed << std::setprecision(2)
                  << std::setw(11) << v.rec.time_mean
                  << std::setw(11) << v.lstm.time_mean
                  << std::setw(13) << v.winner
                  << "\n";
        if (v.winner == "LSTM") ++lstm_wins;
        if (v.winner != "n/a")  ++total;
    }
    std::cout << "\nLSTM wins (" << phase << "): " << lstm_wins << " / " << total << "\n";
}

// CPU vs GPU comparison across the two phases.
void print_combined_summary(const vector<ScenarioVerdict>& cpu_vs,
                            const vector<ScenarioVerdict>& gpu_vs)
{
    auto speedup = [](double cpu_s, double gpu_s) -> float {
        if (cpu_s <= 0.0 || gpu_s <= 0.0) return std::numeric_limits<float>::quiet_NaN();
        return float(cpu_s / gpu_s);
    };

    std::cout << "\n\n";
    std::cout << "===============================================================================================\n";
    std::cout << "      C P U   v s   G P U   :  test_rmse + speedup\n";
    std::cout << "===============================================================================================\n";

    std::cout << std::left << std::setw(7) << "Scen"
              << std::right
              << std::setw(13) << "Rec CPU s"
              << std::setw(13) << "Rec GPU s"
              << std::setw(9)  << "Rec×"
              << std::setw(13) << "LSTM CPU s"
              << std::setw(13) << "LSTM GPU s"
              << std::setw(9)  << "LSTM×"
              << "\n";
    std::cout << std::string(77, '-') << "\n";

    const size_t n = std::min(cpu_vs.size(), gpu_vs.size());
    for (size_t i = 0; i < n; ++i)
    {
        const auto& c = cpu_vs[i];
        const auto& g = gpu_vs[i];
        std::cout << std::left << std::setw(7) << c.id
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(13) << c.rec.time_mean
                  << std::setw(13) << g.rec.time_mean
                  << std::setw(8)  << speedup(c.rec.time_mean,  g.rec.time_mean) << "x"
                  << std::setw(13) << c.lstm.time_mean
                  << std::setw(13) << g.lstm.time_mean
                  << std::setw(8)  << speedup(c.lstm.time_mean, g.lstm.time_mean) << "x"
                  << "\n";
    }
    std::cout << "\nSpeedup = CPU mean time / GPU mean time. Recurrent uses custom CUDA\n"
                 "kernels + cuBLAS; LSTM uses cudnnRNNForward (cellMode = CUDNN_LSTM).\n";

    // Accuracy side-by-side (CPU rmse vs GPU rmse for each architecture).
    std::cout << "\n";
    std::cout << "===============================================================================================\n";
    std::cout << "      A C C U R A C Y   ( test_rmse mean over " << SEED_COUNT
              << " seed" << (SEED_COUNT > 1 ? "s" : "") << " )\n";
    std::cout << "===============================================================================================\n";

    std::cout << std::left << std::setw(7) << "Scen"
              << std::right
              << std::setw(14) << "Rec CPU"
              << std::setw(14) << "LSTM CPU"
              << std::setw(14) << "Rec GPU"
              << std::setw(14) << "LSTM GPU"
              << std::setw(13) << "CPU winner"
              << std::setw(13) << "GPU winner"
              << "\n";
    std::cout << std::string(89, '-') << "\n";

    int lstm_wins_cpu = 0, lstm_wins_gpu = 0, total = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const auto& c = cpu_vs[i];
        const auto& g = gpu_vs[i];
        std::cout << std::left << std::setw(7) << c.id
                  << std::right << std::scientific << std::setprecision(3)
                  << std::setw(14) << c.rec.test_rmse_mean
                  << std::setw(14) << c.lstm.test_rmse_mean
                  << std::setw(14) << g.rec.test_rmse_mean
                  << std::setw(14) << g.lstm.test_rmse_mean
                  << std::setw(13) << c.winner
                  << std::setw(13) << g.winner
                  << "\n";
        if (c.winner == "LSTM") ++lstm_wins_cpu;
        if (g.winner == "LSTM") ++lstm_wins_gpu;
        if (c.winner != "n/a")  ++total;
    }

    std::cout << "\n";
    std::cout << "LSTM wins on CPU: " << lstm_wins_cpu << " / " << total << "\n";
    std::cout << "LSTM wins on GPU: " << lstm_wins_gpu << " / " << total << "\n";
}

const vector<Scenario>& scenarios()
{
    static const vector<Scenario> v = {
        // -------- LIGHT (original S1-S4) --------
        {"S1", "Sine, past=5, future=1",
            "funcion_seno_inputTarget.csv", ",",
            5, 1, false, {},
            Shape{8}, 0.01f, 64, /*ep*/300, /*pat*/25},

        {"S2", "Twopendulum, past=5, future=1",
            "twopendulum.csv", ";",
            5, 1, false, {},
            Shape{16}, 0.01f, 64, 300, 25},

        {"S3", "Twopendulum, past=5, future=3",
            "twopendulum.csv", ";",
            5, 3, true, {},
            Shape{24}, 0.005f, 64, 300, 25},

        {"S4", "Sine, past=5, future=5 (multi-target)",
            "funcion_seno_inputTarget.csv", ",",
            5, 5, true, {},
            Shape{16}, 0.01f, 64, 300, 25},

        // -------- MEDIUM: longer windows on twopendulum --------
        // {"S5", "Twopendulum, past=10, future=1, hidden=32",
        //     "twopendulum.csv", ";",
        //     10, 1, false, {},
        //     Shape{32}, 0.005f, 64, 250, 30},

        // {"S6", "Twopendulum, past=20, future=5 (multi), hidden=48",
        //     "twopendulum.csv", ";",
        //     20, 5, true, {},
        //     Shape{48}, 0.003f, 128, 200, 35},

        // -------- HIGH: Pendulum.csv (99k rows) --------
        // {"S7", "Pendulum, past=15, future=1, hidden=32",
        //     "Pendulum.csv", ",",
        //     15, 1, false, {},
        //     Shape{32}, 0.005f, 128, 150, 30},

        // {"S8", "Pendulum, past=30, future=3 (multi), stacked {48,32}",
        //     "Pendulum.csv", ",",
        //     30, 3, true, {7, 8},
        //     Shape{48, 32}, 0.003f, 128, 120, 30},

        // -------- VERY HIGH: long sequences + deep stacked LSTM --------
        // {"S9", "Pendulum, past=60, future=5 (multi), stacked {64,64}",
        //     "Pendulum.csv", ",",
        //     60, 5, true, {7, 8},
        //     Shape{64, 64}, 0.002f, 128, 80, 30},

        // {"S10", "Pendulum, past=100, future=10 (multi), deep {128,64,32}",
        //     "Pendulum.csv", ",",
        //     100, 10, true, {7, 8},
        //     Shape{128, 64, 32}, 0.001f, 128, 60, 30},

        // -------- EXTREME: maximum stress for the LSTM ----------
        // {"S11", "Pendulum, past=150, future=15 (multi), very deep {256,128,96,48}",
        //     "Pendulum.csv", ",",
        //     150, 15, true, {7, 8},
        //     Shape{256, 128, 96, 48}, 0.0005f, 128, 50, 25},
    };
    return v;
}

} // namespace

#ifdef OPENNN_HAS_CUDA
static int cudnn_rnn_smoke_test()
{
    const int B = 64;
    const int T = 5;
    const int F = 2;
    const int H = 16;

    std::cerr << "[smoke] cudnnGetVersion=" << cudnnGetVersion()
              << "  cudartVersion=" << cudnnGetCudartVersion() << "\n";

    cudnnHandle_t handle = nullptr;
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) {
        std::cerr << "[smoke] cudnnCreate FAILED\n";
        return 1;
    }

    cudnnRNNDescriptor_t rnn = nullptr;
    cudnnCreateRNNDescriptor(&rnn);

    cudnnDropoutDescriptor_t drop = nullptr;
    cudnnCreateDropoutDescriptor(&drop);
    size_t drop_states_bytes = 0;
    cudnnDropoutGetStatesSize(handle, &drop_states_bytes);
    void* drop_states = nullptr;
    cudaMalloc(&drop_states, drop_states_bytes);
    cudnnSetDropoutDescriptor(drop, handle, 0.0f, drop_states, drop_states_bytes, 0ULL);

    cudnnStatus_t st = cudnnSetRNNDescriptor_v8(
        rnn,
        CUDNN_RNN_ALGO_STANDARD,
        CUDNN_LSTM,
        CUDNN_RNN_SINGLE_INP_BIAS,
        CUDNN_UNIDIRECTIONAL,
        CUDNN_LINEAR_INPUT,
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_FLOAT,
        CUDNN_DEFAULT_MATH,
        F, H, H, 1, drop, CUDNN_RNN_PADDED_IO_ENABLED);
    std::cerr << "[smoke] SetRNNDescriptor_v8: " << cudnnGetErrorString(st) << "\n";
    if (st != CUDNN_STATUS_SUCCESS) return 2;

    // Data descriptors
    cudnnRNNDataDescriptor_t xDesc = nullptr, yDesc = nullptr;
    cudnnCreateRNNDataDescriptor(&xDesc);
    cudnnCreateRNNDataDescriptor(&yDesc);

    std::vector<int> seqLen(B, T);

    st = cudnnSetRNNDataDescriptor(
        xDesc, CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
        T, B, F, seqLen.data(), nullptr);
    std::cerr << "[smoke] SetRNNDataDescriptor x: " << cudnnGetErrorString(st) << "\n";
    if (st != CUDNN_STATUS_SUCCESS) return 3;

    st = cudnnSetRNNDataDescriptor(
        yDesc, CUDNN_DATA_FLOAT,
        CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED,
        T, B, H, seqLen.data(), nullptr);
    std::cerr << "[smoke] SetRNNDataDescriptor y: " << cudnnGetErrorString(st) << "\n";
    if (st != CUDNN_STATUS_SUCCESS) return 4;

    // h/c descriptors
    cudnnTensorDescriptor_t hDesc = nullptr, cDesc = nullptr;
    cudnnCreateTensorDescriptor(&hDesc);
    cudnnCreateTensorDescriptor(&cDesc);
    int dimA[3]   = {1, B, H};
    int strA[3]   = {B*H, H, 1};
    cudnnSetTensorNdDescriptor(hDesc, CUDNN_DATA_FLOAT, 3, dimA, strA);
    cudnnSetTensorNdDescriptor(cDesc, CUDNN_DATA_FLOAT, 3, dimA, strA);

    // Buffer sizes
    size_t weight_bytes = 0, work_bytes = 0, reserve_bytes = 0;
    cudnnGetRNNWeightSpaceSize(handle, rnn, &weight_bytes);
    cudnnGetRNNTempSpaceSizes(handle, rnn, CUDNN_FWD_MODE_TRAINING, xDesc,
                              &work_bytes, &reserve_bytes);
    std::cerr << "[smoke] weight=" << weight_bytes
              << " work=" << work_bytes
              << " reserve=" << reserve_bytes << "\n";

    // Device buffers
    void *x = nullptr, *y = nullptr, *w = nullptr, *ws = nullptr, *rs = nullptr;
    int32_t* devSeq = nullptr;
    cudaMalloc(&x,  size_t(B)*T*F*sizeof(float));
    cudaMalloc(&y,  size_t(B)*T*H*sizeof(float));
    cudaMalloc(&w,  weight_bytes);
    cudaMalloc(&ws, work_bytes);
    cudaMalloc(&rs, reserve_bytes);
    cudaMalloc(&devSeq, size_t(B)*sizeof(int32_t));

    // Init: zero everything so weights are valid floats
    cudaMemset(x, 0, size_t(B)*T*F*sizeof(float));
    cudaMemset(w, 0, weight_bytes);
    cudaMemset(ws, 0, work_bytes);
    cudaMemset(rs, 0, reserve_bytes);
    std::vector<int32_t> seqLen32(B, T);
    cudaMemcpy(devSeq, seqLen32.data(), B*sizeof(int32_t), cudaMemcpyHostToDevice);

    // Flush any pending CUDA error from the memsets/memcpys above.
    cudaDeviceSynchronize();
    cudaError_t pending = cudaGetLastError();
    std::cerr << "[smoke] pending CUDA err before forward: "
              << cudaGetErrorString(pending) << "\n";

    // Pre-build RNN execution plan for this batch size.
    st = cudnnBuildRNNDynamic(handle, rnn, B);
    std::cerr << "[smoke] BuildRNNDynamic (B=" << B << "): "
              << cudnnGetErrorString(st) << "\n";

    std::cerr << "[smoke] >>> cudnnRNNForward (TRAINING)\n";
    st = cudnnRNNForward(
        handle, rnn, CUDNN_FWD_MODE_TRAINING,
        devSeq,
        xDesc, x,
        yDesc, y,
        hDesc, nullptr, nullptr,
        cDesc, nullptr, nullptr,
        weight_bytes, w,
        work_bytes, ws,
        reserve_bytes, rs);
    std::cerr << "[smoke] <<< cudnnRNNForward TRAINING: " << cudnnGetErrorString(st)
              << " (status=" << int(st) << ")\n";
    if (st != CUDNN_STATUS_SUCCESS) {
        char last[2048] = {0};
        cudnnGetLastErrorString(last, sizeof(last));
        std::cerr << "[smoke] last cuDNN error message: "
                  << (last[0] ? last : "(empty)") << "\n";
    }

    // Try INFERENCE explicitly (no reserveSpace).
    std::cerr << "[smoke] >>> cudnnRNNForward (INFERENCE)\n";
    st = cudnnRNNForward(
        handle, rnn, CUDNN_FWD_MODE_INFERENCE,
        devSeq,
        xDesc, x,
        yDesc, y,
        hDesc, nullptr, nullptr,
        cDesc, nullptr, nullptr,
        weight_bytes, w,
        work_bytes, ws,
        0, nullptr);
    std::cerr << "[smoke] <<< cudnnRNNForward INFERENCE: " << cudnnGetErrorString(st)
              << " (status=" << int(st) << ")\n";
    if (st != CUDNN_STATUS_SUCCESS) return 10;

    cudaError_t cerr = cudaStreamSynchronize(0);
    std::cerr << "[smoke] streamSync: " << cudaGetErrorString(cerr) << "\n";
    if (cerr != cudaSuccess) return 11;

    std::cerr << "[smoke] SUCCESS - cuDNN-RNN works on this system.\n";

    // Cleanup
    cudaFree(x); cudaFree(y); cudaFree(w); cudaFree(ws); cudaFree(rs);
    cudaFree(devSeq); cudaFree(drop_states);
    cudnnDestroyTensorDescriptor(hDesc);
    cudnnDestroyTensorDescriptor(cDesc);
    cudnnDestroyRNNDataDescriptor(xDesc);
    cudnnDestroyRNNDataDescriptor(yDesc);
    cudnnDestroyDropoutDescriptor(drop);
    cudnnDestroyRNNDescriptor(rnn);
    cudnnDestroy(handle);
    return 0;
}
#endif

int main()
{
    try
    {
        std::cout << "OpenNN - Recurrent vs LSTM forecasting benchmark "
                  << "(" << SEED_COUNT << " seed" << (SEED_COUNT > 1 ? "s" : "")
                  << " per scenario)\n";
        std::cout << "Flow: phase 1 runs every scenario on GPU; when GPU is done,\n"
                     "      phase 2 reruns the same scenarios on CPU.\n";

        std::cout << "\n#################  PHASE 1 / 2  :  G P U  #################\n";
        Configuration::instance().set(Device::CUDA, Type::FP32);
        vector<ScenarioVerdict> gpu_verdicts;
        for (const auto& s : scenarios())
            gpu_verdicts.push_back(run_scenario(s));
        print_phase_summary(gpu_verdicts, "GPU");

        std::cout << "\n#################  PHASE 2 / 2  :  C P U  #################\n";
        Configuration::instance().set(Device::CPU, Type::FP32);
        vector<ScenarioVerdict> cpu_verdicts;
        for (const auto& s : scenarios())
            cpu_verdicts.push_back(run_scenario(s));
        print_phase_summary(cpu_verdicts, "CPU");

        print_combined_summary(cpu_verdicts, gpu_verdicts);
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
