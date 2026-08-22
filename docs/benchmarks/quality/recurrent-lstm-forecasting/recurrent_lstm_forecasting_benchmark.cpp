// OpenNN Recurrent-vs-LSTM forecasting benchmark: one main function, written
// the way a user writes an OpenNN forecasting application, plus a timer. The
// measurement protocol (5 seeds, mean +/- sample std, best) and the JSON
// artifact live in run_forecasting.py, which parses the METRIC and SPEEDUP
// lines only.
//
//   recurrent_lstm_forecasting_benchmark        (no arguments; knobs are env vars)
//
// Phase 1 trains every scenario on GPU, phase 2 reruns the same scenarios on
// CPU. Each scenario trains ForecastingNetwork (Recurrent) and
// ForecastingLstmNetwork (LSTM) over the seeds and prints one METRIC line per
// network; when both phases ran, one SPEEDUP line per scenario and network
// pairs their mean training times.
//
// Environment knobs (A/B levers, all optional):
//   OPENNN_FORECASTING_DATA_DIR    directory holding beijing_pm25_forecasting.csv (default ../data/)
//   OPENNN_FORECASTING_PHASE       gpu | cpu            (default: both, GPU first)
//   OPENNN_FORECASTING_SCENARIOS   comma list of ids    (default: all of B1..B4)
//   OPENNN_FORECASTING_SEEDS       1..5 seeds           (default 5, seeds 0..4)
//   OPENNN_FORECASTING_BATCH_SIZES comma list of sizes  (default 128)
//   OPENNN_FORECASTING_EPOCHS      fixed matched-work epochs (default: quality protocol)
//   OPENNN_FORECASTING_CLIP        Adam gradient clip norm (default 0 = off)
//   OPENNN_FORECASTING_INIT        keras -> Keras-style init; otherwise set_parameters_pytorch()
//   OPENNN_FORECASTING_GRAPH       0 -> CUDA graph off  (default on)
//   OPENNN_FORECASTING_PRECISION   fp32 | bf16          (default fp32; GPU only)
//   OPENNN_FORECASTING_CPU_THREADS positive CPU thread count (default: OpenNN configuration)

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/time_series_dataset.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/testing_analysis/testing_analysis.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;
int main()
{
    try
    {
        const char* env = getenv("OPENNN_FORECASTING_DATA_DIR");
        string data_dir = (env && env[0]) ? env : "../data/";
        if (data_dir.back() != '/' && data_dir.back() != '\\') data_dir += '/';
        const string data_file = data_dir + "beijing_pm25_forecasting.csv";

        env = getenv("OPENNN_FORECASTING_PHASE");
        string phase = (env && env[0]) ? env : "";
        for (char& c : phase) c = char(tolower(static_cast<unsigned char>(c)));
        const bool run_gpu = phase.empty() || phase == "gpu";
        const bool run_cpu = phase.empty() || phase == "cpu";

        env = getenv("OPENNN_FORECASTING_CLIP");
        const float clip_norm = (env && env[0]) ? float(atof(env)) : 0.0f;

        env = getenv("OPENNN_FORECASTING_INIT");
        const bool pytorch_init = !(env && string(env) == "keras");

        env = getenv("OPENNN_FORECASTING_GRAPH");
        const bool cuda_graph = !(env && string(env) == "0");

        env = getenv("OPENNN_FORECASTING_PRECISION");
        string precision = (env && env[0]) ? env : "fp32";
        for (char& c : precision) c = char(tolower(static_cast<unsigned char>(c)));
        throw_if(precision != "fp32" && precision != "bf16",
                 "OPENNN_FORECASTING_PRECISION must be fp32 or bf16.");
        const bool gpu_bf16 = precision == "bf16";

        env = getenv("OPENNN_FORECASTING_CPU_THREADS");
        const int cpu_threads = (env && env[0]) ? atoi(env) : 0;
        throw_if(cpu_threads < 0,
                 "OPENNN_FORECASTING_CPU_THREADS must be non-negative.");

        env = getenv("OPENNN_FORECASTING_SEEDS");
        const int requested_seeds = (env && env[0]) ? atoi(env) : 0;
        const int seed_count = (requested_seeds >= 1 && requested_seeds <= 5) ? requested_seeds : 5;

        env = getenv("OPENNN_FORECASTING_EPOCHS");
        const Index fixed_epochs = (env && env[0]) ? Index(atoll(env)) : 0;

        env = getenv("OPENNN_FORECASTING_SCENARIOS");
        vector<string> selected_ids;                        // empty: every scenario
        stringstream selected_text(env ? env : "");
        for (string item; getline(selected_text, item, ',');)
            selected_ids.push_back(item);

        const auto selected = [&](const string& id)
        {
            return selected_ids.empty()
                || find(selected_ids.begin(), selected_ids.end(), id) != selected_ids.end();
        };

        struct Scenario
        {
            string id, description;
            Index  past, future;
            bool   multi_target;
            Shape  hidden;
            float  learning_rate;
            Index  batch_size, max_epochs, patience;
        };

        const vector<Scenario> base_scenarios = {
            {"B1", "Beijing PM2.5, past=24h, future=1h",    24,  1, false, Shape{32}, 0.003f, 128, 120, 20},
            {"B2", "Beijing PM2.5, past=48h, future=1h",    48,  1, false, Shape{48}, 0.003f, 128, 100, 20},
            {"B3", "Beijing PM2.5, past=72h, future=24h",   72, 24, true,  Shape{64}, 0.002f, 128,  80, 20},
            {"B4", "Beijing PM2.5, past=168h, future=24h", 168, 24, true,  Shape{64}, 0.001f, 128,  60, 15},
        };

        env = getenv("OPENNN_FORECASTING_BATCH_SIZES");
        vector<Index> requested_batch_sizes;
        stringstream batch_text(env ? env : "");
        for (string item; getline(batch_text, item, ',');)
        {
            const Index value = Index(atoll(item.c_str()));
            if (value > 0) requested_batch_sizes.push_back(value);
        }

        vector<Scenario> scenarios;
        for (const Scenario& base : base_scenarios)
        {
            const vector<Index> batch_sizes = requested_batch_sizes.empty()
                ? vector<Index>{base.batch_size} : requested_batch_sizes;
            for (const Index batch_size : batch_sizes)
            {
                Scenario scenario = base;
                scenario.batch_size = batch_size;
                if (fixed_epochs > 0)
                {
                    scenario.max_epochs = fixed_epochs;
                    scenario.patience = fixed_epochs + 1;
                }
                scenarios.push_back(move(scenario));
            }
        }

        struct Aggregate
        {
            Index  params = 0, epochs_mean = 0;
            int    successful_runs = 0;
            float  val_err_mean = numeric_limits<float>::quiet_NaN();
            float  test_rmse_mean = numeric_limits<float>::quiet_NaN();
            float  test_rmse_std = numeric_limits<float>::quiet_NaN();
            float  test_rmse_best = numeric_limits<float>::quiet_NaN();
            float  test_rmse_native_mean = numeric_limits<float>::quiet_NaN();
            float  test_rmse_rel_mean = numeric_limits<float>::quiet_NaN();
            double time_mean = 0.0;
            double samples_per_sec_mean = 0.0;
            Index  train_windows = 0;
        };

        // TestingAnalysis errs(2) is sqrt(sum_sq / 2N) over all N x W outputs;
        // the headline test_rmse is the standard per-element sqrt(sum_sq / NW).
        constexpr float rmse_half_to_std = 1.41421356237309515f;

        const char* const phase_name[2] = {"GPU", "CPU"};
        const char* const net_name[2] = {"Recurrent", "LSTM"};
        vector<double> time_mean[2];        // per phase, scenario-major / net-minor, for SPEEDUP

        cout << "OpenNN - Recurrent vs LSTM forecasting benchmark ("
             << seed_count << " seed" << (seed_count > 1 ? "s" : "") << " per scenario)\n"
             << "Dataset: UCI Beijing PM2.5  data_dir=" << data_dir << "\n"
             << "Flow: phase 1 runs every scenario on GPU; when GPU is done,\n"
                "      phase 2 reruns the same scenarios on CPU.\n";
        if (!phase.empty())
            cout << "OPENNN_FORECASTING_PHASE=" << phase << " -> running only that phase.\n";

        for (int p = 0; p < 2; ++p)
        {
            if (p == 0 ? !run_gpu : !run_cpu) continue;

            cout << "\n#################  PHASE " << phase_name[p] << "  #################\n";
            Configuration::instance().set(p == 0 ? Device::CUDA : Device::CPU,
                                          p == 0 && gpu_bf16 ? Type::BF16 : Type::FP32);
            if (p == 1 && cpu_threads > 0)
                set_threads_number(cpu_threads);

            for (const Scenario& s : scenarios)
            {
                if (!selected(s.id)) continue;

                cout << "\n=== " << s.id << "  " << s.description << " ===\n"
                     << "    past=" << s.past << "  future=" << s.future
                     << "  hidden_layers=" << s.hidden.get_rank()
                     << "  batch=" << s.batch_size
                     << "  epochs<=" << s.max_epochs << "  patience=" << s.patience
                     << "  seeds=" << seed_count << "  lr=" << s.learning_rate << "\n" << flush;

                Aggregate aggregate[2];

                for (int net = 0; net < 2; ++net)
                {
                    vector<float>  rmse_values, rmse_native_values, rmse_rel_values, val_values;
                    vector<double> time_values, throughput_values;
                    vector<Index>  epoch_values;
                    Aggregate& a = aggregate[net];

                    for (int seed = 0; seed < seed_count; ++seed)
                    {
                        set_seed(unsigned(seed));

                        TimeSeriesDataset dataset(data_file, ",", true, false);
                        dataset.set_past_time_steps(s.past);
                        dataset.set_future_time_steps(s.future);
                        dataset.set_multi_target(s.multi_target);
                        dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
                        const Index training_windows = Index(dataset.get_sample_indices("Training").size());

                        unique_ptr<NeuralNetwork> network;
                        if (net == 0)
                            network = make_unique<ForecastingNetwork>(
                                dataset.get_input_shape(), s.hidden, dataset.get_target_shape());
                        else
                            network = make_unique<ForecastingLstmNetwork>(
                                dataset.get_input_shape(), s.hidden, dataset.get_target_shape());

                        if (pytorch_init) network->set_parameters_pytorch();

                        float  test_rmse = numeric_limits<float>::quiet_NaN();
                        float  test_rmse_native = numeric_limits<float>::quiet_NaN();
                        float  test_rmse_rel = numeric_limits<float>::quiet_NaN();
                        float  val_err = numeric_limits<float>::quiet_NaN();
                        double seconds = 0.0;
                        Index  params = 0, epochs = 0;
                        string notes;

                        try
                        {
                            TrainingStrategy training_strategy(network.get(), &dataset);
                            training_strategy.set_loss("MeanSquaredError");
                            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

                            auto* adam = static_cast<AdaptiveMomentEstimation*>(
                                training_strategy.get_optimization_algorithm());
                            adam->set_learning_rate(s.learning_rate);
                            adam->set_batch_size(s.batch_size);
                            adam->set_maximum_epochs(s.max_epochs);
                            adam->set_maximum_validation_failures(s.patience);
                            if (fixed_epochs > 0)
                            {
                                adam->set_validation_period(s.max_epochs);
                                adam->set_restore_best(false);
                            }
                            adam->set_gradient_clip_norm(clip_norm);
                            adam->set_cuda_graph(cuda_graph);
                            adam->set_display(false);

                            // Match PyTorch's timed region: its model and all train/validation
                            // tensors are already resident on CUDA before the timer starts.
                            // Optimizer::train() deliberately accepts an already-resident
                            // network/dataset, so stage the same one-time transfers here.
                            if (p == 0)
                            {
                                network->copy_parameters_device();
                                network->copy_states_device();
                                dataset.enable_device_residency();
                                device::synchronize(device::get_compute_stream());
                            }

                            const TrainingResult results = training_strategy.train();

                            params  = network->get_parameters_number();
                            epochs  = results.get_epochs_number();
                            val_err = results.get_validation_error();
                            seconds = results.training_seconds;

                            const Index target_width = dataset.get_target_shape().size();

                            try
                            {
                                TestingAnalysis testing_analysis(network.get(), &dataset);
                                const VectorR errors = testing_analysis.calculate_errors("Testing");
                                if (errors.size() >= 3 && target_width > 0)
                                {
                                    test_rmse_native = errors(2);
                                    test_rmse = errors(2) * rmse_half_to_std / sqrt(float(target_width));
                                }
                            }
                            catch (const exception& e) { notes = e.what(); }

                            if (isfinite(test_rmse))
                            {
                                const vector<Index> testing_indices = dataset.get_sample_indices("Testing");
                                const vector<Index> target_indices = dataset.get_feature_indices("Target");
                                if (!testing_indices.empty() && target_width > 0)
                                {
                                    MatrixR targets(testing_indices.size(), target_width);
                                    dataset.fill_targets(testing_indices, target_indices, targets.data(),
                                                         FillMode::Inference);
                                    const float range = targets.maxCoeff() - targets.minCoeff();
                                    if (range > 0.0f) test_rmse_rel = test_rmse / range;
                                }
                            }
                        }
                        catch (const exception& e) { notes = string("EXCEPTION: ") + e.what(); }

                        if (!notes.empty())
                            cout << "    [" << net_name[net] << " seed " << seed << "] " << notes << "\n" << flush;

                        if (isfinite(test_rmse) && isfinite(val_err))
                        {
                            cout << "METRIC engine=opennn phase=" << phase_name[p]
                                 << " scenario=" << s.id << " net=" << net_name[net]
                                 << " batch_size=" << s.batch_size << " seed=" << seed
                                 << " params=" << params << " epochs=" << epochs
                                 << " test_rmse=" << test_rmse << " time_s=" << seconds
                                 << " samples_per_sec="
                                 << (seconds > 0.0 ? double(training_windows * epochs) / seconds : 0.0)
                                 << " train_windows=" << training_windows
                                 << " device=" << (p == 0 ? "cuda" : "cpu")
                                 << " precision=" << (p == 0 && gpu_bf16 ? "bf16" : "fp32") << '\n';
                            rmse_values.push_back(test_rmse);
                            val_values.push_back(val_err);
                            time_values.push_back(seconds);
                            if (seconds > 0.0)
                                throughput_values.push_back(double(training_windows * epochs) / seconds);
                            epoch_values.push_back(epochs);
                            if (isfinite(test_rmse_native)) rmse_native_values.push_back(test_rmse_native);
                            if (isfinite(test_rmse_rel)) rmse_rel_values.push_back(test_rmse_rel);
                            a.params = params;
                            a.train_windows = training_windows;
                        }
                    }

                    if (!rmse_values.empty())
                    {
                        const auto mean = [](const auto& values)
                        {
                            double sum = 0.0;
                            for (const auto value : values) sum += value;
                            return sum / values.size();
                        };

                        a.successful_runs = int(rmse_values.size());
                        a.test_rmse_mean = float(mean(rmse_values));
                        a.test_rmse_best = *min_element(rmse_values.begin(), rmse_values.end());
                        a.val_err_mean = float(mean(val_values));
                        if (!rmse_native_values.empty()) a.test_rmse_native_mean = float(mean(rmse_native_values));
                        if (!rmse_rel_values.empty()) a.test_rmse_rel_mean = float(mean(rmse_rel_values));
                        a.time_mean = mean(time_values);
                        if (!throughput_values.empty()) a.samples_per_sec_mean = mean(throughput_values);

                        double squares = 0.0;                   // sample std over the seeds
                        for (const float value : rmse_values)
                            squares += double(value - a.test_rmse_mean) * double(value - a.test_rmse_mean);
                        a.test_rmse_std = rmse_values.size() < 2
                            ? 0.0f : float(sqrt(squares / (rmse_values.size() - 1)));

                        Index total_epochs = 0;
                        for (const Index value : epoch_values) total_epochs += value;
                        a.epochs_mean = total_epochs / Index(epoch_values.size());
                    }

                    time_mean[p].push_back(a.time_mean);
                }

                const string winner =
                    !isfinite(aggregate[0].test_rmse_mean) || !isfinite(aggregate[1].test_rmse_mean) ? "n/a"
                    : aggregate[0].test_rmse_mean <= aggregate[1].test_rmse_mean ? "Recurrent" : "LSTM";

                for (int net = 0; net < 2; ++net)
                {
                    const Aggregate& a = aggregate[net];
                    ostringstream line;
                    line << "METRIC phase=" << phase_name[p] << " scenario=" << s.id << " net=" << net_name[net]
                         << " batch_size=" << s.batch_size
                         << " params=" << a.params
                         << " epochs_mean=" << a.epochs_mean
                         << " successful_runs=" << a.successful_runs
                         << setprecision(9)
                         << " val_err_mean=" << a.val_err_mean
                         << " test_rmse_mean=" << a.test_rmse_mean
                         << " test_rmse_std=" << a.test_rmse_std
                         << " test_rmse_best=" << a.test_rmse_best
                         << " test_rmse_native_halfconv_mean=" << a.test_rmse_native_mean
                         << " test_rmse_rel_mean=" << a.test_rmse_rel_mean
                         << " time_s_mean=" << a.time_mean
                         << " samples_per_sec_mean=" << a.samples_per_sec_mean
                         << " train_windows=" << a.train_windows
                         << " device=" << (p == 0 ? "cuda" : "cpu")
                         << " precision=" << (p == 0 && gpu_bf16 ? "bf16" : "fp32")
                         << " winner=" << winner;
                    cout << line.str() << "\n" << flush;
                }
            }
        }

        if (run_gpu && run_cpu)
        {
            cout << "\n";
            size_t i = 0;
            for (const Scenario& s : scenarios)
            {
                if (!selected(s.id)) continue;

                for (int net = 0; net < 2; ++net, ++i)
                {
                    const double cpu_s = time_mean[1][i];
                    const double gpu_s = time_mean[0][i];
                    const float speedup = (cpu_s <= 0.0 || gpu_s <= 0.0)
                        ? numeric_limits<float>::quiet_NaN() : float(cpu_s / gpu_s);

                    ostringstream line;
                    line << "SPEEDUP scenario=" << s.id << " net=" << net_name[net]
                         << " batch_size=" << s.batch_size << setprecision(9)
                         << " cpu_time_s=" << cpu_s << " gpu_time_s=" << gpu_s << " cpu_over_gpu=" << speedup;
                    cout << line.str() << "\n";
                }
            }
            cout << flush;
        }

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
