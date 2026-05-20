//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R E C A S T I N G   P R O J E C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com
//
//   This example exercises ForecastingNetwork (Scaling → Recurrent → Dense →
//   Unscaling) across the four supported scenarios:
//     case 1 — 1 future step,  1 variable
//     case 2 — 1 future step,  N variables  (N target columns predicted at t+1)
//     case 3 — K future steps, 1 variable   (K consecutive points of one column)
//     case 4 — K future steps, N variables  (K points × N columns)
//
//   For each case we report whether the training error drops below a loose
//   threshold so it is obvious at a glance if anything regresses.

#include <cstring>
#include <iostream>
#include <iomanip>
#include <string>

#include "../../opennn/time_series_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/loss.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/configuration.h"
#include "../../opennn/adaptive_moment_estimation.h"

using namespace opennn;

namespace
{
const string DATA_DIR = "/home/adriangonzalez/Documents/opennn/examples/forecasting/data/";

struct CaseResult
{
    string label;
    Index   parameters_number = 0;
    Index   epochs_run = 0;
    float   training_error = 0.0f;
    float   validation_error = 0.0f;
    float   threshold = 0.05f;
};

void print_header(const string& title)
{
    cout << "\n=============================================\n"
         << title << "\n"
         << "=============================================\n";
}

CaseResult train(const string& label,
                 TimeSeriesDataset& dataset,
                 const Shape& complexity,
                 Index epochs,
                 float threshold)
{
    print_header(label);

    cout << "Raw rows:          " << dataset.get_samples_number()    << "\n"
         << "Past time steps:   " << dataset.get_past_time_steps()   << "\n"
         << "Future time steps: " << dataset.get_future_time_steps() << "\n"
         << "Multi-target:      " << (dataset.get_multi_target() ? "yes" : "no") << "\n"
         << "Input shape:       " << dataset.get_input_shape()        << "\n"
         << "Target shape:      " << dataset.get_target_shape()       << "\n";

    ForecastingNetwork nn(dataset.get_input_shape(),
                          complexity,
                          dataset.get_target_shape());

    cout << "Network params:    " << nn.get_parameters_number() << "\n";

    TrainingStrategy strategy(&nn, &dataset);
    strategy.set_loss("MeanSquaredError");

    auto* adam = static_cast<AdaptiveMomentEstimation*>(strategy.get_optimization_algorithm());
    adam->set_maximum_epochs(epochs);
    adam->set_learning_rate(0.01f);
    adam->set_batch_size(64);
    adam->set_display_period(epochs);  // only print at end to reduce noise

    cout << "Training...\n";
    TrainingResults results = strategy.train();

    CaseResult cr;
    cr.label             = label;
    cr.parameters_number = nn.get_parameters_number();
    cr.epochs_run        = results.get_epochs_number();
    cr.training_error    = results.get_training_error();
    cr.validation_error  = results.get_validation_error();
    cr.threshold         = threshold;

    cout << "Training error:    " << cr.training_error    << "\n"
         << "Validation error:  " << cr.validation_error  << "\n"
         << "Verdict:           "
         << (cr.training_error < threshold ? "[OK]" : "[WARN]")
         << " (threshold " << threshold << ")\n";

    return cr;
}
}

int main()
{
    try
    {
        cout << "OpenNN. Forecasting — multi-case validation suite." << endl;

        Configuration::instance().set(Device::CPU, Type::FP32, Type::FP32);

        vector<CaseResult> results;

        // ---------------------------------------------------------------
        // Case 1 — 1 future step, 1 variable.
        // funcion_seno_inputTarget.csv: single sine column.
        // ---------------------------------------------------------------
        {
            TimeSeriesDataset ds(DATA_DIR + "funcion_seno_inputTarget.csv",
                                 ",", /*has_header=*/false, /*has_sample_ids=*/false);
            ds.set_past_time_steps(5);
            ds.set_future_time_steps(1);
            ds.set_multi_target(false);

            results.push_back(train("CASE 1 — 1 step, 1 variable",
                                    ds, /*complexity=*/{10}, /*epochs=*/300,
                                    /*threshold=*/0.05f));
        }

        // ---------------------------------------------------------------
        // Case 2 — 1 future step, N variables (multivariate target).
        // twopendulum.csv: two coupled pendulum angles. Both columns are
        // marked InputTarget by the dataset's default forecasting role
        // assignment, so the network predicts both at t + future_steps.
        // ---------------------------------------------------------------
        {
            TimeSeriesDataset ds(DATA_DIR + "twopendulum.csv",
                                 ";", /*has_header=*/false, /*has_sample_ids=*/false);
            ds.set_past_time_steps(5);
            ds.set_future_time_steps(1);
            ds.set_multi_target(false);

            results.push_back(train("CASE 2 — 1 step, N variables",
                                    ds, {15}, 300, 0.1f));
        }

        // ---------------------------------------------------------------
        // Case 3 — K future steps, 1 variable.
        // funcion_seno_inputTarget.csv: predict next 5 points of the sine.
        // ---------------------------------------------------------------
        {
            TimeSeriesDataset ds(DATA_DIR + "funcion_seno_inputTarget.csv",
                                 ",", false, false);
            ds.set_past_time_steps(5);
            ds.set_future_time_steps(5);
            ds.set_multi_target(true);

            results.push_back(train("CASE 3 — K steps, 1 variable",
                                    ds, {15}, 300, 0.1f));
        }

        // ---------------------------------------------------------------
        // Case 4 — K future steps, N variables (the full multi-step
        // multi-variate case). twopendulum.csv with multi_target=true.
        // Target shape becomes {K * N} = {5 * N_target_columns}.
        // ---------------------------------------------------------------
        {
            TimeSeriesDataset ds(DATA_DIR + "twopendulum.csv",
                                 ";", false, false);
            ds.set_past_time_steps(5);
            ds.set_future_time_steps(5);
            ds.set_multi_target(true);

            results.push_back(train("CASE 4 — K steps, N variables",
                                    ds, {20}, 300, 0.15f));
        }

        // ---------------------------------------------------------------
        // Summary
        // ---------------------------------------------------------------
        print_header("SUMMARY");
        cout << left
             << setw(36) << "Case"
             << right
             << setw(10) << "Params"
             << setw(8)  << "Epochs"
             << setw(14) << "Train err"
             << setw(14) << "Val err"
             << setw(10) << "Result"
             << "\n";
        cout << string(92, '-') << "\n";

        bool all_ok = true;
        for (const auto& r : results)
        {
            const bool ok = r.training_error < r.threshold;
            all_ok = all_ok && ok;
            cout << left
                 << setw(36) << r.label
                 << right
                 << setw(10) << r.parameters_number
                 << setw(8)  << r.epochs_run
                 << setw(14) << scientific << setprecision(3) << r.training_error
                 << setw(14) << r.validation_error
                 << setw(10) << (ok ? "[OK]" : "[WARN]")
                 << "\n";
        }
        cout << "\nOverall: " << (all_ok ? "ALL OK" : "SOME WARNINGS") << "\n";

        return all_ok ? 0 : 2;
    }
    catch (const exception& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or any later version.
