//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N P U T S   S E L E C T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <random>
#include <set>
#include <limits>
#include <cmath>

#include "dataset.h"
#include "tabular_dataset.h"
#include "time_series_dataset.h"
#include "neural_network.h"
#include "training_strategy.h"
#include "optimizer.h"
#include "inputs_selection.h"

namespace opennn
{

InputsSelection::InputsSelection(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}

vector<vector<Index>> InputsSelection::build_fold_partition() const
{
    Dataset* dataset = training_strategy->get_dataset();
    const Index k = max<Index>(folds_number, Index(1));

    // Development pool = Training + Validation (Testing/None excluded).
    vector<Index> development = dataset->get_sample_indices("Training");
    const vector<Index> validation = dataset->get_sample_indices("Validation");
    development.insert(development.end(), validation.begin(), validation.end());

    vector<vector<Index>> folds(static_cast<size_t>(k));

    auto deal_blocks = [&folds, k](const vector<Index>& items)      // contiguous, order-preserving
    {
        const Index n = ssize(items);
        for (Index f = 0; f < k; ++f)
            for (Index j = f * n / k; j < (f + 1) * n / k; ++j)
                folds[size_t(f)].push_back(items[j]);
    };

    auto deal_round_robin = [&folds, k](const vector<Index>& items)  // spreads a class evenly
    {
        for (size_t i = 0; i < items.size(); ++i)
            folds[i % size_t(k)].push_back(items[i]);
    };

    // Time series: sequential blocks preserving order -- shuffling would leak autocorrelated
    // neighbours across the train/validation boundary.
    if (dynamic_cast<TimeSeriesDataset*>(dataset))
    {
        sort(development.begin(), development.end());
        deal_blocks(development);
        return folds;
    }

    // Otherwise stratify by class (threshold = target mean, robust to scaling) using a LOCAL seeded
    // RNG (independent of the training RNG) so every fold keeps the class balance and the partition
    // is reproducible. Falls back to a plain shuffled split when the target is not a single column.
    mt19937 rng(static_cast<unsigned>(folds_seed));
    const vector<Index> target_features = dataset->get_feature_indices("Target");

    if (target_features.size() == 1 && !development.empty())
    {
        const MatrixR& data = dataset->get_data();
        const Index tcol = target_features[0];

        double target_sum = 0.0;
        for (const Index s : development) target_sum += double(data(s, tcol));
        const float threshold = float(target_sum / double(development.size()));

        vector<Index> positives, negatives;
        for (const Index s : development)
            (data(s, tcol) > threshold ? positives : negatives).push_back(s);

        shuffle(positives.begin(), positives.end(), rng);
        shuffle(negatives.begin(), negatives.end(), rng);
        deal_round_robin(positives);
        deal_round_robin(negatives);
    }
    else
    {
        shuffle(development.begin(), development.end(), rng);
        deal_blocks(development);
    }

    return folds;
}

InputsSelection::FoldEvaluation InputsSelection::evaluate_folds(const vector<vector<Index>>& fold_partition) const
{
    Dataset* dataset = training_strategy->get_dataset();
    NeuralNetwork* neural_network = training_strategy->get_loss()->get_neural_network();
    const Index k = ssize(fold_partition);

    vector<Index> development;
    for (const vector<Index>& fold : fold_partition)
        development.insert(development.end(), fold.begin(), fold.end());

    float validation_error_sum = 0.0f;
    float training_error_sum = 0.0f;
    Index epochs_sum = 0;

    for (Index f = 0; f < k; ++f)
    {
        const vector<Index>& validation_indices = fold_partition[size_t(f)];
        const std::set<Index> validation_set(validation_indices.begin(), validation_indices.end());

        vector<Index> training_indices;
        training_indices.reserve(development.size());
        for (const Index s : development)
            if (!validation_set.count(s)) training_indices.push_back(s);

        // Transient fold split: never mutates the user's persistent roles (restored on scope exit).
        FoldScope scope(*dataset, training_indices, validation_indices);

        neural_network->set_parameters_random();
        const TrainingResult training_results = training_strategy->train();

        float validation_error = training_results.get_validation_error();
        float training_error = training_results.get_training_error();
        if (!isfinite(validation_error)) validation_error = numeric_limits<float>::max();
        if (!isfinite(training_error))   training_error   = numeric_limits<float>::max();

        // Best epoch of this fold (validation-based early stopping restored it), else epochs run.
        // +1 turns the 0-based epoch index into an epoch count.
        const Index fold_epochs = training_results.restored_best_parameters
            ? training_results.restored_epoch + 1
            : training_results.get_epochs_number();

        validation_error_sum += validation_error;
        training_error_sum += training_error;
        epochs_sum += max<Index>(fold_epochs, Index(1));
    }

    const Index divisor = k > 0 ? k : 1;

    FoldEvaluation evaluation;
    evaluation.validation_error = validation_error_sum / float(divisor);
    evaluation.training_error = training_error_sum / float(divisor);
    evaluation.epochs = max<Index>(epochs_sum / divisor, Index(1));
    return evaluation;
}

void InputsSelection::refit_final_model_on_development() const
{
    Dataset* dataset = training_strategy->get_dataset();
    NeuralNetwork* neural_network = training_strategy->get_loss()->get_neural_network();
    Optimizer* optimizer = training_strategy->get_optimization_algorithm();

    // Epoch budget = mean best epoch from the cross-validation of the (already configured) final
    // subset. There is no validation set below, so this bounds the fit in place of early stopping.
    const Index final_epochs = evaluate_folds(build_fold_partition()).epochs;

    // Development pool = Training + Validation; the final model trains on all of it, no validation.
    vector<Index> development = dataset->get_sample_indices("Training");
    const vector<Index> validation = dataset->get_sample_indices("Validation");
    development.insert(development.end(), validation.begin(), validation.end());

    const Index saved_epochs = optimizer->get_maximum_epochs();
    optimizer->set_maximum_epochs(final_epochs);

    {
        FoldScope scope(*dataset, development, {});   // all development as Training, no Validation
        neural_network->set_parameters_random();
        training_strategy->train();
    }

    optimizer->set_maximum_epochs(saved_epochs);
}

void InputsSelection::configure_neural_network_inputs(NeuralNetwork* neural_network, Dataset* dataset, Index input_features_number)
{
    const TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(dataset);

    const Shape input_shape = time_series_dataset
        ? Shape{ time_series_dataset->get_past_time_steps(), input_features_number }
        : Shape{ input_features_number };
    neural_network->set_input_shape(input_shape);
    dataset->set_shape("Input", input_shape);

    if (time_series_dataset)
    {
        const Index past_time_steps = time_series_dataset->get_past_time_steps();
        const vector<string> base_names = dataset->get_variable_names("Input");

        vector<string> final_feature_names;
        final_feature_names.reserve(base_names.size() * past_time_steps);

        for (const string& base_name : base_names)
            for (Index j = 0; j < past_time_steps; ++j)
                final_feature_names.push_back(format("{}_lag{}", base_name.empty() ? "variable" : base_name, j));

        neural_network->set_input_names(final_feature_names);
    }
    else
    {
        neural_network->set_input_variables(dataset->get_variables("Input"));
    }

    neural_network->compile();
}

InputsSelectionResult::InputsSelectionResult(const Index maximum_epochs)
{
    set(maximum_epochs);
}

Index InputsSelectionResult::get_epochs_number() const
{
    return training_error_history.size();
}

void InputsSelectionResult::set(const Index maximum_epochs)
{
    training_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    validation_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    mean_validation_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    mean_training_error_history = VectorR::Constant(maximum_epochs, -1.0f);
}

void InputsSelectionResult::resize_history(const Index new_size)
{
    training_error_history.conservativeResize(new_size);
    validation_error_history.conservativeResize(new_size);
    mean_training_error_history.conservativeResize(new_size);
    mean_validation_error_history.conservativeResize(new_size);
}

void InputsSelectionResult::print() const
{
    cout << "\n"
         << "Input Selection Results" << "\n"
         << "Optimal inputs number: " << optimal_input_variable_names.size() << "\n"
         << "Inputs: " << "\n";

    for (const string& name : optimal_input_variable_names)
        cout << "   " << name << "\n";

    cout << "Optimum training error: " << optimum_training_error << "\n"
         << "Optimum validation error: " << optimum_validation_error << "\n";
}

void InputsSelection::save(const filesystem::path& file_name) const
{
    save_json_file(file_name, *this);
}

void InputsSelection::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
