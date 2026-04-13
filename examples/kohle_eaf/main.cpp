//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   K O H L E   E A F   E X P E R I M E N T S
//
//   Phase 1: Model selection (GrowingNeurons)
//   Phase 2: Multiple runs with random initializations (mean +/- std)
//   All reproducible via master seed

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>

#include "../../opennn/opennn.h"

using namespace opennn;

// ============================================================
// Configuration
// ============================================================

const Index MASTER_SEED = 20260410;
const Index NUM_RUNS = 10;
const type TRAIN_RATIO = type(0.6);
const type VALIDATION_RATIO = type(0.2);
const type TEST_RATIO = type(0.2);

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
// Phase 1: Model Selection using GrowingNeurons
// ============================================================

Index run_model_selection(const string& dataset_path,
                          const string& label,
                          ofstream& report)
{
    cout << "\n========================================================" << endl;
    cout << "PHASE 1 - Model Selection: " << label << endl;
    cout << "========================================================" << endl;

    set_seed(MASTER_SEED);

    Dataset dataset(dataset_path, ";", true, false);
    dataset.split_samples_random(TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO);

    const Shape input_shape = dataset.get_input_shape();
    const Shape target_shape = dataset.get_target_shape();
    cout << "Inputs: " << input_shape[0] << endl;

    // Initial network with 1 neuron (GrowingNeurons will resize)
    ApproximationNetwork neural_network(input_shape, {1}, target_shape);

    Bounding* bounding = (Bounding*)neural_network.get_first("Bounding");
    if(bounding) bounding->set_bounding_method("NoBounding");

    // Training strategy
    TrainingStrategy training_strategy(&neural_network, &dataset);
    training_strategy.get_loss()->set_regularization_method("L2");
    training_strategy.get_loss()->set_regularization_weight(type(0.001));
    training_strategy.get_optimization_algorithm()->set_display(false);

    // Model selection via GrowingNeurons (direct instantiation)
    GrowingNeurons growing_neurons(&training_strategy);

    growing_neurons.set_minimum_neurons(1);
    growing_neurons.set_maximum_neurons(20);
    growing_neurons.set_neurons_increment(1);
    growing_neurons.set_trials_number(3);
    growing_neurons.set_maximum_validation_failures(3);
    growing_neurons.set_display(true);

    cout << "Starting neuron selection..." << endl;
    NeuronsSelectionResults results = growing_neurons.perform_neurons_selection();

    Index optimal = results.optimal_neurons_number;

    cout << "\n==> Optimal neurons: " << optimal << endl;

    // Write report
    report << label << "\n" << string(60, '-') << "\n";
    report << "Neuron range: 1 to 20 (step 1, 3 trials each)\n\n";

    report << "  Neurons   Train Error   Validation Error\n";
    report << "  ------------------------------------------\n";
    for(Index i = 0; i < results.neurons_number_history.size(); i++)
    {
        report << "  " << setw(7) << results.neurons_number_history(i)
               << "   " << setw(12) << results.training_error_history(i)
               << "   " << setw(16) << results.validation_error_history(i) << "\n";
    }
    report << "\n  ==> Optimal neurons: " << optimal << "\n";
    report << "  ==> Best validation error: " << results.optimum_validation_error << "\n";
    report << "  Elapsed: " << results.elapsed_time << "\n\n\n";

    return optimal;
}


// ============================================================
// Phase 2: Multiple runs with different seeds
// ============================================================

void run_multiple_experiments(const string& dataset_path,
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
        set_seed(run_seed);

        cout << "\n--- Run " << run + 1 << "/" << NUM_RUNS
             << " (seed=" << run_seed << ") ---" << endl;

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

        type r2_train = testing_analysis.calculate_determination(
            targets_train.col(0), outputs_train.col(0));
        type r2_test = testing_analysis.calculate_determination(
            targets_test.col(0), outputs_test.col(0));

        train_r2_v.push_back(static_cast<double>(r2_train));
        test_r2_v.push_back(static_cast<double>(r2_test));
        train_rmse_v.push_back(static_cast<double>(errors_train(2)));
        test_rmse_v.push_back(static_cast<double>(errors_test(2)));

        cout << "  Train R2=" << r2_train << "  Test R2=" << r2_test
             << "  RMSE=" << errors_test(2) << endl;

        report << "  " << setw(3) << run + 1
               << "   " << setw(10) << run_seed
               << "   " << fixed << setprecision(4) << setw(8) << r2_train
               << "   " << setw(8) << r2_test
               << "   " << setprecision(1) << setw(10) << errors_train(2)
               << "   " << setw(10) << errors_test(2) << "\n";
    }

    Stats r2_train_s = compute_stats(train_r2_v);
    Stats r2_test_s = compute_stats(test_r2_v);
    Stats rmse_train_s = compute_stats(train_rmse_v);
    Stats rmse_test_s = compute_stats(test_rmse_v);

    report << "\n  SUMMARY (mean +/- std)\n  ----------------------\n";
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

        const string data_dir = "data/";
        const string no_scrap = data_dir + "kohle_no_scrap.csv";
        const string with_scrap = data_dir + "kohle_with_scrap.csv";

        ofstream report("data/experiment_results.txt");

        report << "================================================================\n";
        report << " EAF ENERGY PREDICTION - RIGOROUS EXPERIMENTS\n";
        report << "================================================================\n\n";
        report << " Master seed:       " << MASTER_SEED << "\n";
        report << " Dataset:           4144 heats\n";
        report << " Split:             60% train / 20% validation / 20% test\n";
        report << " Regularization:    L2 (weight=0.001)\n";
        report << " Activation:        HyperbolicTangent (hidden), Linear (output)\n";
        report << " Optimizer:         Quasi-Newton (default)\n";
        report << " Multiple runs:     " << NUM_RUNS << "\n\n";

        // Phase 1
        report << "================================================================\n";
        report << " PHASE 1: MODEL SELECTION (GrowingNeurons)\n";
        report << "================================================================\n\n";

        Index opt_a = run_model_selection(no_scrap,
            "Dataset A: No scrap composition (8 inputs)", report);

        Index opt_b = run_model_selection(with_scrap,
            "Dataset B: With scrap composition (19 inputs)", report);

        report << "  OPTIMAL NEURONS:\n";
        report << "    No scrap:   " << opt_a << "\n";
        report << "    With scrap: " << opt_b << "\n\n";

        // Phase 2
        report << "================================================================\n";
        report << " PHASE 2: MULTIPLE RUNS (" << NUM_RUNS << " initializations)\n";
        report << "================================================================\n\n";

        run_multiple_experiments(no_scrap, "Dataset A: No scrap", opt_a, report);
        run_multiple_experiments(with_scrap, "Dataset B: With scrap", opt_b, report);

        // Reproducibility
        report << "================================================================\n";
        report << " REPRODUCIBILITY\n";
        report << "================================================================\n\n";
        report << " Master seed = " << MASTER_SEED << "\n";
        report << " Phase 1: set_seed(MASTER_SEED)\n";
        report << " Phase 2: set_seed(MASTER_SEED + run*7919)\n";
        for(Index i = 0; i < NUM_RUNS; i++)
            report << "   Run " << setw(2) << i+1 << ": seed=" << MASTER_SEED+i*7919 << "\n";

        report.close();

        cout << "\nResults saved to data/experiment_results.txt" << endl;
        return 0;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}
