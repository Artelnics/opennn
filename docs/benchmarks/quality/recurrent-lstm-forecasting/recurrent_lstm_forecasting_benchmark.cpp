//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E C U R R E N T   V S   L S T M   F O R E C A S T I N G   B E N C H M A R K
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/time_series_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/standard_networks.h"
#include "opennn/loss.h"
#include "opennn/training_strategy.h"
#include "opennn/configuration.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/testing_analysis.h"
#include "opennn/random_utilities.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace opennn;
using Clock = chrono::steady_clock;

namespace
{

const string DATA_FILE = "beijing_pm25_forecasting.csv";

constexpr int      SEED_COUNT = 5;
constexpr unsigned SEEDS[SEED_COUNT] = {0, 1, 2, 3, 4};

constexpr float RMSE_HALF_TO_STD = 1.41421356237309515f;

string forecasting_data_dir()
{
    const char* env = getenv("OPENNN_FORECASTING_DATA_DIR");
    if (env && env[0] != '\0')
    {
        string path(env);
        if (!path.empty() && path.back() != '/' && path.back() != '\\')
            path += '/';
        return path;
    }
    return "../data/";
}

string forecasting_phase()
{
    const char* env = getenv("OPENNN_FORECASTING_PHASE");
    string phase = (env && env[0] != '\0') ? string(env) : string();
    transform(phase.begin(), phase.end(), phase.begin(),
                   [](unsigned char c) { return char(tolower(c)); });
    return phase;
}

float forecasting_clip_norm()
{
    const char* env = getenv("OPENNN_FORECASTING_CLIP");
    return (env && env[0] != '\0') ? float(atof(env)) : 0.0f;
}

bool forecasting_pytorch_init()
{
    const char* env = getenv("OPENNN_FORECASTING_INIT");
    return !(env && string(env) == "keras");
}

bool forecasting_cuda_graph()
{
    const char* env = getenv("OPENNN_FORECASTING_GRAPH");
    return !(env && string(env) == "0");
}

int forecasting_seed_count()
{
    const char* env = getenv("OPENNN_FORECASTING_SEEDS");
    if (!env || env[0] == '\0') return SEED_COUNT;
    const int n = atoi(env);
    return (n >= 1 && n <= SEED_COUNT) ? n : SEED_COUNT;
}

bool scenario_selected(const string& id)
{
    const char* env = getenv("OPENNN_FORECASTING_SCENARIOS");
    if (!env || env[0] == '\0') return true;
    const string list(env);
    size_t pos = 0;
    while (pos < list.size())
    {
        size_t comma = list.find(',', pos);
        if (comma == string::npos) comma = list.size();
        if (list.substr(pos, comma - pos) == id) return true;
        pos = comma + 1;
    }
    return false;
}

string format_seconds(double s)
{
    int total = int(s);
    const int h = total / 3600; total %= 3600;
    const int m = total / 60;   const int sec = total % 60;
    ostringstream os;
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
        cout << "\n";
        cout.flush();
    }

    void draw() const
    {
        constexpr int W = 30;
        const float frac = total ? float(done) / float(total) : 0.0f;
        const int filled = int(frac * W + 0.5f);
        const double elapsed = chrono::duration<double>(Clock::now() - started).count();
        const double eta = (done > 0) ? elapsed * (total - done) / done : 0.0;

        cout << "\r    [" << string(filled, '#') << string(W - filled, '.')
                  << "] " << done << "/" << total
                  << "  elapsed=" << format_seconds(elapsed)
                  << "  ETA=" << format_seconds(eta) << "   ";
        cout.flush();
    }
};

ScenarioProgress g_bar;

struct Scenario
{
    string  id;
    string  description;
    Index   past;
    Index   future;
    bool    multi_target;
    Shape   hidden;
    float   learning_rate;
    Index   batch_size;
    Index   max_epochs;
    Index   patience;
};

const vector<Scenario>& scenarios()
{
    static const vector<Scenario> beijing = {
        {"B1", "Beijing PM2.5, past=24h, future=1h",
            24, 1, false, Shape{32}, 0.003f, 128, 120, 20},

        {"B2", "Beijing PM2.5, past=48h, future=1h",
            48, 1, false, Shape{48}, 0.003f, 128, 100, 20},

        {"B3", "Beijing PM2.5, past=72h, future=24h",
            72, 24, true, Shape{64}, 0.002f, 128, 80, 20},

        {"B4", "Beijing PM2.5, past=168h, future=24h",
            168, 24, true, Shape{64}, 0.001f, 128, 60, 15},
    };

    return beijing;
}

struct RunResult
{
    Index   params = 0;
    Index   epochs = 0;
    float   train_err = numeric_limits<float>::quiet_NaN();
    float   val_err   = numeric_limits<float>::quiet_NaN();
    float   test_rmse = numeric_limits<float>::quiet_NaN();
    float   test_rmse_native = numeric_limits<float>::quiet_NaN();
    float   test_rmse_rel = numeric_limits<float>::quiet_NaN();
    double  seconds = 0.0;
    bool    restored_best = false;
    string  notes;
};

struct AggregatedResult
{
    string net;
    float  test_rmse_mean = numeric_limits<float>::quiet_NaN();
    float  test_rmse_std  = numeric_limits<float>::quiet_NaN();
    float  test_rmse_best = numeric_limits<float>::quiet_NaN();
    float  test_rmse_native_mean = numeric_limits<float>::quiet_NaN();
    float  val_err_mean   = numeric_limits<float>::quiet_NaN();
    float  test_rmse_rel_mean = numeric_limits<float>::quiet_NaN();
    double time_mean = 0.0;
    Index  epochs_mean = 0;
    Index  params = 0;
    int    successful_runs = 0;
};

unique_ptr<TimeSeriesDataset> load_dataset(const Scenario& s)
{
    auto ds = make_unique<TimeSeriesDataset>(forecasting_data_dir() + DATA_FILE,
                                             ",",
                                                            true,
                                                                false);
    ds->set_past_time_steps(s.past);
    ds->set_future_time_steps(s.future);
    ds->set_multi_target(s.multi_target);
    ds->set_storage_mode(Dataset::StorageMode::GPUPersistantData);
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
        adam->set_gradient_clip_norm(forecasting_clip_norm());
        adam->set_cuda_graph(forecasting_cuda_graph());
        adam->set_display(false);

        const auto t0 = Clock::now();
        const TrainingResult results = strategy.train();
        const auto t1 = Clock::now();

        r.params        = nn->get_parameters_number();
        r.epochs        = results.get_epochs_number();
        r.train_err     = results.get_training_error();
        r.val_err       = results.get_validation_error();
        r.seconds       = chrono::duration<double>(t1 - t0).count();
        r.restored_best = results.restored_best_parameters;

        const Index target_width = ds->get_target_shape().size();

        try
        {
            TestingAnalysis ta(nn, ds);
            const VectorR errs = ta.calculate_errors("Testing");
            if (errs.size() >= 3 && target_width > 0)
            {
                r.test_rmse_native = errs(2);
                r.test_rmse = errs(2) * RMSE_HALF_TO_STD / sqrt(float(target_width));
            }
        }
        catch (const exception& e) { r.notes = e.what(); }

        if (isfinite(r.test_rmse))
        {
            const vector<Index> testing_idx = ds->get_sample_indices("Testing");
            const vector<Index> target_idx  = ds->get_feature_indices("Target");
            if (!testing_idx.empty() && target_width > 0)
            {
                MatrixR targets(testing_idx.size(), target_width);
                ds->fill_targets(testing_idx, target_idx, targets.data(),
                                                 false,                 true);
                const float range = targets.maxCoeff() - targets.minCoeff();
                if (range > 0.0f) r.test_rmse_rel = r.test_rmse / range;
            }
        }
    }
    catch (const exception& e)
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

    vector<float>  rmse_vals, rmse_native_vals, rmse_rel_vals, val_vals;
    vector<double> time_vals;
    vector<Index>  epoch_vals;

    const int seed_count = forecasting_seed_count();
    for (int s_i = 0; s_i < seed_count; ++s_i)
    {
        const unsigned seed = SEEDS[s_i];
        set_seed(seed);
        auto ds = load_dataset(s);
        auto nn = build(*ds);
        if (forecasting_pytorch_init()) nn->set_parameters_pytorch();
        const RunResult r = train_one(nn.get(), ds.get(), s);

        if (!r.notes.empty())
            cout << "\n    [" << net_label << " seed " << seed << "] " << r.notes << "\n";

        g_bar.tick();

        if (isfinite(r.test_rmse) && isfinite(r.val_err))
        {
            rmse_vals.push_back(r.test_rmse);
            val_vals.push_back(r.val_err);
            time_vals.push_back(r.seconds);
            epoch_vals.push_back(r.epochs);
            if (isfinite(r.test_rmse_native)) rmse_native_vals.push_back(r.test_rmse_native);
            if (isfinite(r.test_rmse_rel)) rmse_rel_vals.push_back(r.test_rmse_rel);
            agg.params = r.params;
        }
    }

    if (rmse_vals.empty()) return agg;

    auto mean = [](const auto& v) -> float {
        return float(accumulate(v.begin(), v.end(), 0.0) / v.size());
    };
    auto stddev = [&](const vector<float>& v, float m) -> float {
        if (v.size() < 2) return 0.0f;
        double acc = 0.0;
        for (float x : v) acc += double(x - m) * double(x - m);
        return float(sqrt(acc / (v.size() - 1)));
    };

    agg.successful_runs    = int(rmse_vals.size());
    agg.test_rmse_mean     = mean(rmse_vals);
    agg.test_rmse_std      = stddev(rmse_vals, agg.test_rmse_mean);
    agg.test_rmse_best     = *min_element(rmse_vals.begin(), rmse_vals.end());
    if (!rmse_native_vals.empty()) agg.test_rmse_native_mean = mean(rmse_native_vals);
    agg.val_err_mean       = mean(val_vals);
    if (!rmse_rel_vals.empty()) agg.test_rmse_rel_mean = mean(rmse_rel_vals);
    agg.time_mean          = accumulate(time_vals.begin(), time_vals.end(), 0.0) / time_vals.size();
    agg.epochs_mean        = Index(accumulate(epoch_vals.begin(), epoch_vals.end(), Index(0)) / Index(epoch_vals.size()));
    return agg;
}

void print_agg(const AggregatedResult& a)
{
    cout << "    " << left << setw(10) << a.net
              << "  params=" << right << setw(6) << a.params
              << "  ep_mean=" << setw(4) << a.epochs_mean
              << "  val_mean=" << scientific << setprecision(3) << a.val_err_mean
              << "  test_rmse=" << a.test_rmse_mean
              << " +/- " << a.test_rmse_std
              << "  best=" << a.test_rmse_best;
    if (isfinite(a.test_rmse_rel_mean))
        cout << "  rmse%=" << fixed << setprecision(2)
                  << (100.0f * a.test_rmse_rel_mean);
    cout << "  time=" << fixed << setprecision(2) << a.time_mean << "s\n";
}

void print_metric_line(const string& phase,
                       const string& scenario_id,
                       const AggregatedResult& a,
                       const string& winner)
{
    ostringstream os;
    os << "METRIC"
       << " phase=" << phase
       << " scenario=" << scenario_id
       << " net=" << a.net
       << " params=" << a.params
       << " epochs_mean=" << a.epochs_mean
       << " successful_runs=" << a.successful_runs
       << " val_err_mean=" << setprecision(9) << a.val_err_mean
       << " test_rmse_mean=" << setprecision(9) << a.test_rmse_mean
       << " test_rmse_std=" << setprecision(9) << a.test_rmse_std
       << " test_rmse_best=" << setprecision(9) << a.test_rmse_best
       << " test_rmse_native_halfconv_mean=" << setprecision(9) << a.test_rmse_native_mean
       << " test_rmse_rel_mean=" << setprecision(9) << a.test_rmse_rel_mean
       << " time_s_mean=" << setprecision(9) << a.time_mean
       << " winner=" << winner;

    cout << os.str() << "\n";
}

void print_speedup_metric(const string& scenario_id,
                          const string& net,
                          double cpu_s,
                          double gpu_s,
                          float speedup)
{
    ostringstream os;
    os << "SPEEDUP"
       << " scenario=" << scenario_id
       << " net=" << net
       << " cpu_time_s=" << setprecision(9) << cpu_s
       << " gpu_time_s=" << setprecision(9) << gpu_s
       << " cpu_over_gpu=" << setprecision(9) << speedup;

    cout << os.str() << "\n";
}

struct ScenarioVerdict
{
    string id;
    AggregatedResult rec;
    AggregatedResult lstm;
    string winner = "n/a";
};

auto build_recurrent(const Scenario& s)
{
    return [&s](TimeSeriesDataset& ds) {
        return make_unique<ForecastingNetwork>(
            ds.get_input_shape(), s.hidden, ds.get_target_shape());
    };
}

auto build_lstm(const Scenario& s)
{
    return [&s](TimeSeriesDataset& ds) {
        return make_unique<ForecastingLstmNetwork>(
            ds.get_input_shape(), s.hidden, ds.get_target_shape());
    };
}

string pick_winner(const AggregatedResult& a, const AggregatedResult& b,
                   const string& a_name, const string& b_name)
{
    if (!isfinite(a.test_rmse_mean) || !isfinite(b.test_rmse_mean))
        return "n/a";
    return (a.test_rmse_mean <= b.test_rmse_mean) ? a_name : b_name;
}

ScenarioVerdict run_scenario(const Scenario& s)
{
    cout << "\n=== " << s.id << "  " << s.description << " ===\n";
    cout << "    dataset=" << DATA_FILE
              << "  past="    << s.past
              << "  future="  << s.future
              << "  hidden_layers=" << s.hidden.rank
              << "  epochs<=" << s.max_epochs
              << "  patience=" << s.patience
              << "  seeds="   << forecasting_seed_count()
              << "  lr="      << s.learning_rate << "\n";

    g_bar.start(                2 * forecasting_seed_count());

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

    cout << "    winner: " << v.winner;
    if (isfinite(rec_agg.test_rmse_mean) && isfinite(lstm_agg.test_rmse_mean)
        && rec_agg.test_rmse_mean > 0.0f)
    {
        const float delta_pct = 100.0f *
            (rec_agg.test_rmse_mean - lstm_agg.test_rmse_mean) / rec_agg.test_rmse_mean;
        cout << "  (LSTM test_rmse vs Recurrent: "
                  << fixed << setprecision(1) << delta_pct << "%)";
    }
    cout << "\n";
    return v;
}

void print_phase_summary(const vector<ScenarioVerdict>& vs, const string& phase)
{
    cout << "\n\n";
    cout << "===============================================================\n";
    cout << "      P H A S E   S U M M A R Y   :   " << phase << "\n";
    cout << "===============================================================\n";

    cout << left << setw(8) << "Scen"
              << right
              << setw(14) << "Rec rmse"
              << setw(14) << "LSTM rmse"
              << setw(11) << "Rec(s)"
              << setw(11) << "LSTM(s)"
              << setw(13) << "Winner"
              << "\n";
    cout << string(71, '-') << "\n";

    int lstm_wins = 0, total = 0;
    for (const auto& v : vs)
    {
        cout << left << setw(8) << v.id
                  << right << scientific << setprecision(3)
                  << setw(14) << v.rec.test_rmse_mean
                  << setw(14) << v.lstm.test_rmse_mean
                  << fixed << setprecision(2)
                  << setw(11) << v.rec.time_mean
                  << setw(11) << v.lstm.time_mean
                  << setw(13) << v.winner
                  << "\n";
        if (v.winner == "LSTM") ++lstm_wins;
        if (v.winner != "n/a")  ++total;
    }
    cout << "\nLSTM wins (" << phase << "): " << lstm_wins << " / " << total << "\n";

    for (const auto& v : vs)
    {
        print_metric_line(phase, v.id, v.rec, v.winner);
        print_metric_line(phase, v.id, v.lstm, v.winner);
    }
}

void print_combined_summary(const vector<ScenarioVerdict>& cpu_vs,
                            const vector<ScenarioVerdict>& gpu_vs)
{
    auto speedup = [](double cpu_s, double gpu_s) -> float {
        if (cpu_s <= 0.0 || gpu_s <= 0.0) return numeric_limits<float>::quiet_NaN();
        return float(cpu_s / gpu_s);
    };

    cout << "\n\n";
    cout << "===============================================================================================\n";
    cout << "      C P U   v s   G P U   :  test_rmse + speedup\n";
    cout << "===============================================================================================\n";

    cout << left << setw(7) << "Scen"
              << right
              << setw(13) << "Rec CPU s"
              << setw(13) << "Rec GPU s"
              << setw(9)  << "Rec x"
              << setw(13) << "LSTM CPU s"
              << setw(13) << "LSTM GPU s"
              << setw(9)  << "LSTM x"
              << "\n";
    cout << string(77, '-') << "\n";

    const size_t n = min(cpu_vs.size(), gpu_vs.size());
    for (size_t i = 0; i < n; ++i)
    {
        const auto& c = cpu_vs[i];
        const auto& g = gpu_vs[i];
        cout << left << setw(7) << c.id
                  << right << fixed << setprecision(2)
                  << setw(13) << c.rec.time_mean
                  << setw(13) << g.rec.time_mean
                  << setw(8)  << speedup(c.rec.time_mean,  g.rec.time_mean) << "x"
                  << setw(13) << c.lstm.time_mean
                  << setw(13) << g.lstm.time_mean
                  << setw(8)  << speedup(c.lstm.time_mean, g.lstm.time_mean) << "x"
                  << "\n";
        print_speedup_metric(c.id, "Recurrent", c.rec.time_mean, g.rec.time_mean,
                             speedup(c.rec.time_mean, g.rec.time_mean));
        print_speedup_metric(c.id, "LSTM", c.lstm.time_mean, g.lstm.time_mean,
                             speedup(c.lstm.time_mean, g.lstm.time_mean));
    }
    cout << "\nSpeedup = CPU mean time / GPU mean time. Both networks use cuDNN RNN\n"
                 "on GPU (CUDNN_RNN_TANH / CUDNN_LSTM).\n";

    cout << "\n";
    cout << "===============================================================================================\n";
    cout << "      A C C U R A C Y   ( test_rmse mean over " << forecasting_seed_count()
              << " seed" << (forecasting_seed_count() > 1 ? "s" : "") << " )\n";
    cout << "===============================================================================================\n";

    cout << left << setw(7) << "Scen"
              << right
              << setw(14) << "Rec CPU"
              << setw(14) << "LSTM CPU"
              << setw(14) << "Rec GPU"
              << setw(14) << "LSTM GPU"
              << setw(13) << "CPU winner"
              << setw(13) << "GPU winner"
              << "\n";
    cout << string(89, '-') << "\n";

    int lstm_wins_cpu = 0, lstm_wins_gpu = 0, total = 0;
    for (size_t i = 0; i < n; ++i)
    {
        const auto& c = cpu_vs[i];
        const auto& g = gpu_vs[i];
        cout << left << setw(7) << c.id
                  << right << scientific << setprecision(3)
                  << setw(14) << c.rec.test_rmse_mean
                  << setw(14) << c.lstm.test_rmse_mean
                  << setw(14) << g.rec.test_rmse_mean
                  << setw(14) << g.lstm.test_rmse_mean
                  << setw(13) << c.winner
                  << setw(13) << g.winner
                  << "\n";
        if (c.winner == "LSTM") ++lstm_wins_cpu;
        if (g.winner == "LSTM") ++lstm_wins_gpu;
        if (c.winner != "n/a")  ++total;
    }

    cout << "\n";
    cout << "LSTM wins on CPU: " << lstm_wins_cpu << " / " << total << "\n";
    cout << "LSTM wins on GPU: " << lstm_wins_gpu << " / " << total << "\n";
}

}

int main()
{
    try
    {
        const string phase = forecasting_phase();
        const bool run_gpu = phase.empty() || phase == "gpu";
        const bool run_cpu = phase.empty() || phase == "cpu";

        cout << "OpenNN - Recurrent vs LSTM forecasting benchmark "
                  << "(" << forecasting_seed_count() << " seed"
                  << (forecasting_seed_count() > 1 ? "s" : "") << " per scenario)\n";
        cout << "Dataset: UCI Beijing PM2.5  data_dir=" << forecasting_data_dir() << "\n";
        cout << "Flow: phase 1 runs every scenario on GPU; when GPU is done,\n"
                     "      phase 2 reruns the same scenarios on CPU.\n";
        if (!phase.empty())
            cout << "OPENNN_FORECASTING_PHASE=" << phase
                      << " -> running only that phase.\n";

        vector<ScenarioVerdict> gpu_verdicts;
        if (run_gpu)
        {
            cout << "\n#################  PHASE 1 / 2  :  G P U  #################\n";
            Configuration::instance().set(Device::CUDA, Type::FP32);
            for (const auto& s : scenarios())
                if (scenario_selected(s.id))
                    gpu_verdicts.push_back(run_scenario(s));
            print_phase_summary(gpu_verdicts, "GPU");
        }

        vector<ScenarioVerdict> cpu_verdicts;
        if (run_cpu)
        {
            cout << "\n#################  PHASE 2 / 2  :  C P U  #################\n";
            Configuration::instance().set(Device::CPU, Type::FP32);
            for (const auto& s : scenarios())
                if (scenario_selected(s.id))
                    cpu_verdicts.push_back(run_scenario(s));
            print_phase_summary(cpu_verdicts, "CPU");
        }

        if (run_gpu && run_cpu)
            print_combined_summary(cpu_verdicts, gpu_verdicts);
        return 0;
    }
    catch (const exception& e)
    {
        cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
