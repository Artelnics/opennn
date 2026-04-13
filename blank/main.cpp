//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   K O H L E   M O D E L   E X P E R I M E N T S
//
//   Phase 1: Model selection (optimal neuron count)
//   Phase 2: Multiple runs with random initializations
//   All reproducible via master seed

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

#include "../opennn/opennn.h"

using namespace opennn;

// ============================================================
// Configuration
// ============================================================

const Index MASTER_SEED = 20260410;
const Index NUM_RUNS = 10;
const type TRAIN_RATIO = type(0.6);
const type VALIDATION_RATIO = type(0.2);
const type TEST_RATIO = type(0.2);

// Model selection
const Index MIN_NEURONS = 1;
const Index MAX_NEURONS = 20;
const Index SELECTION_TRIALS = 3;
const Index MAX_SELECTION_FAILURES = 3;

// ============================================================
// Utility
// ============================================================

struct Stats
{
    double mean;
    double std_dev;
    double min_val;
    double max_val;
};

Stats compute_stats(const vector<double>& v)
{
    Stats s;
    double sum = 0;
    s.min_val = 1e30; s.max_val = -1e30;
    for(auto x : v) { sum += x; s.min_val = min(s.min_val, x); s.max_val = max(s.max_val, x); }
    s.mean = sum / v.size();
    double sq_sum = 0;
    for(auto x : v) sq_sum += (x - s.mean) * (x - s.mean);
    s.std_dev = (v.size() > 1) ? sqrt(sq_sum / (v.size() - 1)) : 0;
    return s;
}


// ============================================================
// Train a single model and return R2/RMSE on test
// ============================================================

struct RunResult
{
    type r2_train;
    type r2_test;
    type rmse_train;
    type rmse_test;
    type val_error;
};

RunResult train_and_evaluate(const string& dataset_path,
                             Index neurons,
                             Index seed,
                             bool verbose = false)
{
    set_seed(seed);

    Dataset dataset(dataset_path, ";", true, false);
    dataset.split_samples_random(TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO);

    const Shape input_shape = dataset.get_input_shape();
    const Shape target_shape = dataset.get_target_shape();

    ApproximationNetwork neural_network(input_shape, {neurons}, target_shape);

    Bounding* bounding = (Bounding*)neural_network.get_first("Bounding");
    if(bounding) bounding->set_bounding_method("NoBounding");

    neural_network.set_parameters_random();

    TrainingStrategy training_strategy(&neural_network, &dataset);
    training_strategy.get_loss()->set_regularization_method("L2");
    training_strategy.get_loss()->set_regularization_weight(type(0.001));
    training_strategy.get_optimization_algorithm()->set_display(false);

    TrainingResults training_results = training_strategy.train();

    TestingAnalysis testing_analysis(&neural_network, &dataset);

    const auto [targets_train, outputs_train] = testing_analysis.get_targets_and_outputs("Training");
    const auto [targets_test, outputs_test] = testing_analysis.get_targets_and_outputs("Testing");

    VectorR errors_train = testing_analysis.calculate_errors(targets_train, outputs_train);
    VectorR errors_test = testing_analysis.calculate_errors(targets_test, outputs_test);

    RunResult result;
    result.r2_train = testing_analysis.calculate_determination(
        targets_train.col(0), outputs_train.col(0));
    result.r2_test = testing_analysis.calculate_determination(
        targets_test.col(0), outputs_test.col(0));
    result.rmse_train = errors_train(2);
    result.rmse_test = errors_test(2);
    result.val_error = training_results.get_validation_error();

    if(verbose)
        cout << "  Train R2: " << result.r2_train
             << "  Test R2: " << result.r2_test
             << "  Test RMSE: " << result.rmse_test << endl;

    return result;
}


// ============================================================
// Phase 1: Manual model selection
// ============================================================

Index perform_model_selection(const string& dataset_path,
                              const string& label,
                              ofstream& report)
{
    cout << "\n========================================================" << endl;
    cout << "PHASE 1 - Model Selection: " << label << endl;
    cout << "========================================================" << endl;

    Index optimal_neurons = MIN_NEURONS;
    type best_val_error = type(1e30);
    Index validation_failures = 0;

    report << label << "\n";
    report << string(60, '-') << "\n";
    report << "Neuron range: " << MIN_NEURONS << " to " << MAX_NEURONS << "\n";
    report << "Trials per neuron count: " << SELECTION_TRIALS << "\n";
    report << "Max validation failures: " << MAX_SELECTION_FAILURES << "\n\n";
    report << "  Neurons   Best Val Error   Best Test R2\n";
    report << "  ------------------------------------------\n";

    for(Index n = MIN_NEURONS; n <= MAX_NEURONS; n++)
    {
        cout << "\nTesting " << n << " neurons..." << endl;

        type best_val_this_n = type(1e30);
        type best_r2_this_n = type(0);

        for(Index trial = 0; trial < SELECTION_TRIALS; trial++)
        {
            Index trial_seed = MASTER_SEED + n * 1000 + trial * 31;

            RunResult result = train_and_evaluate(dataset_path, n, trial_seed);

            cout << "  Trial " << trial + 1
                 << ": val_error=" << result.val_error
                 << " test_R2=" << result.r2_test << endl;

            if(result.val_error < best_val_this_n)
            {
                best_val_this_n = result.val_error;
                best_r2_this_n = result.r2_test;
            }
        }

        report << "  " << setw(7) << n
               << "   " << setw(13) << best_val_this_n
               << "   " << fixed << setprecision(4) << setw(12) << best_r2_this_n << "\n";

        if(best_val_this_n < best_val_error)
        {
            best_val_error = best_val_this_n;
            optimal_neurons = n;
            validation_failures = 0;
        }
        else
        {
            validation_failures++;
        }

        if(validation_failures >= MAX_SELECTION_FAILURES)
        {
            cout << "Stopping: " << MAX_SELECTION_FAILURES
                 << " consecutive failures." << endl;
            break;
        }
    }

    report << "\n  ==> Optimal neurons: " << optimal_neurons << "\n";
    report << "  ==> Best validation error: " << best_val_error << "\n\n\n";

    cout << "RESULT: Optimal neurons = " << optimal_neurons << endl;

    return optimal_neurons;
}


// ============================================================
// Phase 2: Multiple runs
// ============================================================

void perform_multiple_runs(const string& dataset_path,
                           const string& label,
                           Index neurons,
                           ofstream& report)
{
    cout << "\n========================================================" << endl;
    cout << "PHASE 2 - Multiple Runs: " << label
         << " (" << neurons << " neurons)" << endl;
    cout << "========================================================" << endl;

    vector<double> train_r2_v, test_r2_v, train_rmse_v, test_rmse_v;

    report << label << " (" << neurons << " neurons, " << NUM_RUNS << " runs)\n";
    report << string(60, '-') << "\n\n";
    report << "  Run   Seed        Train R2    Test R2     Train RMSE   Test RMSE\n";
    report << "  --------------------------------------------------------------------\n";

    for(Index run = 0; run < NUM_RUNS; run++)
    {
        Index run_seed = MASTER_SEED + run * 7919;

        cout << "\n--- Run " << run + 1 << "/" << NUM_RUNS
             << " (seed=" << run_seed << ") ---" << endl;

        RunResult result = train_and_evaluate(dataset_path, neurons, run_seed, true);

        train_r2_v.push_back(static_cast<double>(result.r2_train));
        test_r2_v.push_back(static_cast<double>(result.r2_test));
        train_rmse_v.push_back(static_cast<double>(result.rmse_train));
        test_rmse_v.push_back(static_cast<double>(result.rmse_test));

        report << "  " << setw(3) << run + 1
               << "   " << setw(10) << run_seed
               << "   " << fixed << setprecision(4) << setw(8) << result.r2_train
               << "   " << setw(8) << result.r2_test
               << "   " << setprecision(1) << setw(10) << result.rmse_train
               << "   " << setw(10) << result.rmse_test << "\n";
    }

    Stats r2_train_s = compute_stats(train_r2_v);
    Stats r2_test_s = compute_stats(test_r2_v);
    Stats rmse_train_s = compute_stats(train_rmse_v);
    Stats rmse_test_s = compute_stats(test_rmse_v);

    report << "\n  SUMMARY (mean +/- std)\n";
    report << "  ----------------------\n";
    report << fixed << setprecision(4);
    report << "  Train R2:    " << r2_train_s.mean << " +/- " << r2_train_s.std_dev
           << "  [" << r2_train_s.min_val << " - " << r2_train_s.max_val << "]\n";
    report << "  Test  R2:    " << r2_test_s.mean << " +/- " << r2_test_s.std_dev
           << "  [" << r2_test_s.min_val << " - " << r2_test_s.max_val << "]\n";
    report << setprecision(1);
    report << "  Train RMSE:  " << rmse_train_s.mean << " +/- " << rmse_train_s.std_dev
           << "  [" << rmse_train_s.min_val << " - " << rmse_train_s.max_val << "]\n";
    report << "  Test  RMSE:  " << rmse_test_s.mean << " +/- " << rmse_test_s.std_dev
           << "  [" << rmse_test_s.min_val << " - " << rmse_test_s.max_val << "]\n\n\n";
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        cout << "========================================================" << endl;
        cout << "EAF Energy Prediction - Rigorous Experiments" << endl;
        cout << "Master seed: " << MASTER_SEED << endl;
        cout << "========================================================" << endl;

        const string data_dir = "../data/";
        const string no_scrap_path = data_dir + "kohle_no_scrap.csv";
        const string with_scrap_path = data_dir + "kohle_with_scrap.csv";

        ofstream report("../data/experiment_results.txt");

        report << "================================================================\n";
        report << " EAF ENERGY PREDICTION - RIGOROUS EXPERIMENTS\n";
        report << "================================================================\n\n";
        report << " Master seed:       " << MASTER_SEED << "\n";
        report << " Dataset:           4144 heats\n";
        report << " Split:             60% train / 20% validation / 20% test\n";
        report << " Regularization:    L2 (weight=0.001)\n";
        report << " Activation:        HyperbolicTangent (hidden), Linear (output)\n";
        report << " Optimizer:         Quasi-Newton (default)\n";
        report << " Number of runs:    " << NUM_RUNS << "\n";
        report << " Run seeds:         MASTER_SEED + i*7919 (i=0.."
               << NUM_RUNS-1 << ")\n\n";

        // ====================================================
        // PHASE 1: Model Selection
        // ====================================================

        report << "================================================================\n";
        report << " PHASE 1: MODEL SELECTION (manual growing neurons)\n";
        report << "================================================================\n\n";

        Index optimal_no_scrap = perform_model_selection(
            no_scrap_path, "Dataset A: No scrap composition (8 inputs)", report);

        Index optimal_with_scrap = perform_model_selection(
            with_scrap_path, "Dataset B: With scrap composition (19 inputs)", report);

        report << "  OPTIMAL NEURON SUMMARY:\n";
        report << "  No scrap:   " << optimal_no_scrap << " neurons\n";
        report << "  With scrap: " << optimal_with_scrap << " neurons\n\n";

        // ====================================================
        // PHASE 2: Multiple Runs
        // ====================================================

        report << "================================================================\n";
        report << " PHASE 2: MULTIPLE RUNS (" << NUM_RUNS << " random initializations)\n";
        report << "================================================================\n\n";

        perform_multiple_runs(no_scrap_path,
            "Dataset A: No scrap", optimal_no_scrap, report);

        perform_multiple_runs(with_scrap_path,
            "Dataset B: With scrap", optimal_with_scrap, report);

        // ====================================================
        // Reproducibility
        // ====================================================

        report << "================================================================\n";
        report << " REPRODUCIBILITY\n";
        report << "================================================================\n\n";
        report << " All experiments are fully reproducible by setting:\n";
        report << "   Master seed = " << MASTER_SEED << "\n\n";
        report << " Phase 1 (model selection):\n";
        report << "   seed = MASTER_SEED + neurons*1000 + trial*31\n\n";
        report << " Phase 2 (multiple runs):\n";
        report << "   seed = MASTER_SEED + run_index * 7919\n";
        for(Index i = 0; i < NUM_RUNS; i++)
            report << "   Run " << setw(2) << i+1 << ": seed = " << MASTER_SEED + i*7919 << "\n";
        report << "\n OpenNN set_seed() is called before each individual run,\n";
        report << " ensuring deterministic parameter initialization and\n";
        report << " data splitting for every experiment.\n";

        report.close();

        cout << "\n========================================================" << endl;
        cout << "All experiments completed." << endl;
        cout << "Results saved to data/experiment_results.txt" << endl;
        cout << "========================================================" << endl;

        return 0;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
