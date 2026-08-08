//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <map>

#include "dataset.h"
#include "tabular_dataset.h"
#include "time_series_dataset.h"
#include "growing_inputs.h"
#include "correlations.h"
#include "optimizer.h"
#include "training_strategy.h"
#include "cross_validation.h"
#include "selection_utilities.h"

namespace opennn
{

static vector<pair<Index, Index>> input_feature_ids(const Dataset* dataset)
{
    const vector<Variable>& variables = dataset->get_variables();

    vector<pair<Index, Index>> ids;

    for (const Index variable_index : dataset->get_variable_indices(VariableRole::Input))
        for (Index feature = 0; feature < variables[variable_index].get_feature_count(); ++feature)
            ids.emplace_back(variable_index, feature);

    return ids;
}

static vector<Index> map_feature_rows(const vector<pair<Index, Index>>& old_ids,
                                      const vector<pair<Index, Index>>& new_ids)
{
    map<pair<Index, Index>, Index> old_rows;
    for (Index row = 0; row < ssize(old_ids); ++row)
        old_rows.emplace(old_ids[row], row);

    vector<Index> row_map(new_ids.size(), -1);
    for (Index row = 0; row < ssize(new_ids); ++row)
        if (const auto it = old_rows.find(new_ids[row]); it != old_rows.end())
            row_map[row] = it->second;

    return row_map;
}

GrowingInputs::GrowingInputs(TrainingStrategy* new_training_strategy)
    : InputsSelection(new_training_strategy)
{
    set_default();
}

void GrowingInputs::set_default()
{
    name = "GrowingInputs";

    maximum_validation_failures = 100;
    minimum_inputs_number = 1;
    trials_number = 3;
    maximum_epochs = 1000;
    maximum_time = 3600.0f;

    maximum_inputs_number = (training_strategy && training_strategy->get_neural_network() && training_strategy->get_dataset())
        ? training_strategy->get_dataset()->get_variables_number(VariableRole::Input)
        : 50;
}

void GrowingInputs::set_maximum_inputs_number(const Index new_maximum_inputs_number)
{
    const Index inputs_number = training_strategy->get_dataset()->get_variables_number(VariableRole::Input);

    maximum_inputs_number = (inputs_number == 0)
                                ? new_maximum_inputs_number
                                : min(new_maximum_inputs_number, inputs_number);
}

InputsSelectionResult GrowingInputs::perform_input_selection()
{

    Dataset* dataset = training_strategy->get_dataset();
    const Index original_input_variables_number = dataset->get_variables_number(VariableRole::Input);

    if (dataset->has_nan())
        dataset->scrub_missing_values();

    if (display) cout << "Performing growing input selection...\n";

    InputsSelectionResult input_selection_results(original_input_variables_number);

    training_strategy->get_optimization_algorithm()->set_display(false);

    float previous_validation_error = MAX;

    const vector<Index> target_variable_indices = dataset->get_variable_indices(VariableRole::Target);
    const vector<Index> time_variable_indices = dataset->get_variable_indices(VariableRole::Time);
    const vector<string> variable_names = dataset->get_variable_names();

    if (display) cout << "Calculating correlations...\n";

    const auto* correlations_dataset = dynamic_cast<const TabularDataset*>(dataset);
    throw_if(!correlations_dataset, "Expected TabularDataset.");

    const VectorR total_correlations =
        get_correlation_values(correlations_dataset->calculate_input_target_variable_pearson_correlations()).col(0).array().abs();

    vector<Index> correlation_indices(original_input_variables_number);
    iota(correlation_indices.begin(), correlation_indices.end(), 0);

    ranges::sort(correlation_indices, greater<>{},
                 [&total_correlations](Index index) { return total_correlations[index]; });

    const vector<Index> input_variable_indices = dataset->get_variable_indices(VariableRole::Input);

    VectorI correlations_rank_descending(input_variable_indices.size());

    ranges::transform(correlation_indices,
                      correlations_rank_descending.data(),
                      [&input_variable_indices](Index correlation_index) { return input_variable_indices[correlation_index]; });

    dataset->set_input_variables_unused();

    Index variable_index = 0;

    NeuralNetwork* neural_network = training_strategy->get_neural_network();

    Index validation_failures = 0;

    time_t beginning_time;
    float elapsed_time = 0.0f;
    time(&beginning_time);

    Index epoch = 0;

    const vector<vector<Index>> fold_partition =
        folds_number > 1 ? build_fold_partition(training_strategy, folds_number) : vector<vector<Index>>{};

    ParameterSnapshot warm_snapshot;
    ParameterSnapshot candidate_snapshot;
    vector<pair<Index, Index>> warm_feature_ids;

    while (!input_selection_results.stopping_condition)
    {
        if (variable_index >= correlations_rank_descending.size())
        {
            if (display) cout << "\nAll the variables has been used.\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
            continue;
        }

        const Index current_variable_index = correlations_rank_descending[variable_index];
        const VariableRole current_use = dataset->get_variables()[current_variable_index].role;

        dataset->set_variable_role(current_variable_index,
            current_use == VariableRole::InputTarget ? "InputTarget" : "Input");

        const Index input_variables_number = dataset->get_variables_number(VariableRole::Input);
        const Index input_features_number = dataset->get_features_number(VariableRole::Input);

        if (input_variables_number < minimum_inputs_number)
        {
            ++variable_index;
            continue;
        }

        configure_neural_network_inputs(neural_network, dataset, input_features_number);

        const string& candidate_name = variable_names[current_variable_index];

        if (display)
            cout << "\nTrying to add \"" << candidate_name << "\"  ->  "
                 << input_variables_number << " inputs\n";

        const vector<Index> warm_row_map = warm_start && !warm_snapshot.empty() && folds_number == 1
            ? map_feature_rows(warm_feature_ids, input_feature_ids(dataset))
            : vector<Index>{};

        const CandidateEvaluation candidate_evaluation = evaluate_candidate(
            training_strategy, neural_network, folds_number, fold_partition, trials_number, false,
            [&](Index trial, float training_error, float validation_error, bool improved)
            {
                if (improved && warm_start)
                    candidate_snapshot = capture_parameter_snapshot(neural_network);

                if (improved && validation_error < input_selection_results.optimum_validation_error)
                {
                    input_selection_results.optimal_input_variables_indices = dataset->get_variable_indices(VariableRole::Input);
                    input_selection_results.optimal_input_variable_names = dataset->get_variable_names(VariableRole::Input);
                    neural_network->copy_parameters_host();
                    input_selection_results.optimal_parameters =
                        Eigen::Map<const VectorR>(neural_network->get_parameters_data(),
                                                  neural_network->get_parameters_size());
                    input_selection_results.optimum_training_error = training_error;
                    input_selection_results.optimum_validation_error = validation_error;
                }

                if (display)
                    cout << (trials_number > 1 ? "   Trial " + to_string(trial + 1) + ": " : "   ")
                         << "training error " << training_error
                         << ", validation error " << validation_error << "\n";
            },
            [&](Index trial)
            {
                neural_network->set_parameters_random();

                if (trial == 0 && !warm_row_map.empty())
                    seed_parameters_from_snapshot(neural_network, warm_snapshot, warm_row_map);
            });

        const float minimum_training_error = candidate_evaluation.training_error;
        const float minimum_validation_error = candidate_evaluation.validation_error;

        if (folds_number > 1)
        {
            if (minimum_validation_error < input_selection_results.optimum_validation_error)
            {
                input_selection_results.optimal_input_variables_indices = dataset->get_variable_indices(VariableRole::Input);
                input_selection_results.optimal_input_variable_names = dataset->get_variable_names(VariableRole::Input);
                input_selection_results.optimal_parameters = VectorR();
                input_selection_results.optimum_training_error = minimum_training_error;
                input_selection_results.optimum_validation_error = minimum_validation_error;
            }

            if (display)
                cout << "   " << folds_number << "-fold CV validation error " << minimum_validation_error << "\n";
        }

        if (previous_validation_error < minimum_validation_error)
        {
            ++validation_failures;

            if (display)
                cout << "   Rejected: validation error " << minimum_validation_error
                     << " did not beat the best so far (" << previous_validation_error
                     << "). Removing \"" << candidate_name << "\". Validation failures: "
                     << validation_failures << "/" << maximum_validation_failures << "\n";

            dataset->set_variable_role(current_variable_index,
                dataset->get_variables()[current_variable_index].role == VariableRole::InputTarget ? "Target" : "None");

            candidate_snapshot = {};
        }
        else
        {
            previous_validation_error = minimum_validation_error;

            if (warm_start && !candidate_snapshot.empty())
            {
                warm_snapshot = move(candidate_snapshot);
                candidate_snapshot = {};
                warm_feature_ids = input_feature_ids(dataset);
            }

            input_selection_results.training_error_history(epoch) = minimum_training_error;
            input_selection_results.validation_error_history(epoch) = minimum_validation_error;

            ++epoch;

            if (display)
                cout << "   Accepted. Epoch " << epoch << ": " << input_variables_number
                     << " inputs kept, best validation error " << minimum_validation_error << "\n"
                     << "   Inputs: " << dataset->get_variable_names(VariableRole::Input);
        }

        ++variable_index;
        elapsed_time = get_elapsed_time(beginning_time);

        const Index current_inputs = dataset->get_variables_number(VariableRole::Input);

        input_selection_results.stopping_condition = first_stopping_condition<StoppingCondition>(display,
        {
            {elapsed_time >= maximum_time, StoppingCondition::MaximumTime,
             format("Epoch {}\nMaximum time reached: {}\n", epoch, get_time(elapsed_time))},
            {input_selection_results.optimum_validation_error <= validation_error_goal, StoppingCondition::ValidationErrorGoal,
             format("\nValidation error goal reached: {:g}\n", input_selection_results.optimum_validation_error)},
            {epoch >= maximum_epochs, StoppingCondition::MaximumEpochs,
             "\nMaximum number of epochs reached.\n"},
            {validation_failures >= maximum_validation_failures, StoppingCondition::MaximumValidationFailures,
             format("\nMaximum validation failures ({}) reached.\n", validation_failures)},
            {current_inputs >= maximum_inputs_number, StoppingCondition::MaximumInputs,
             format("\nMaximum inputs ({}) reached.\n", current_inputs)}
        });
    }

    input_selection_results.elapsed_time = get_time(elapsed_time);
    input_selection_results.resize_history(epoch);

    dataset->set_variable_indices(input_selection_results.optimal_input_variables_indices,
        target_variable_indices);

    const Index optimal_processed_variables_number = dataset->get_features_number(VariableRole::Input);

    if (dynamic_cast<TimeSeriesDataset*>(dataset) && time_variable_indices.size() == 1)
        dataset->set_variable_role(time_variable_indices[0], "Time");

    configure_neural_network_inputs(neural_network, dataset, optimal_processed_variables_number);

    const InputScaling input_scaling = capture_input_scaling(dataset);

    set_maximum_inputs_number(dataset->get_variables_number(VariableRole::Input));

    apply_input_scaling(neural_network, input_scaling);

    finalize_selected_model(training_strategy, neural_network,
                            input_selection_results.optimal_parameters, folds_number, display, "inputs");

    if (display) input_selection_results.print();

    return input_selection_results;
}

void GrowingInputs::to_JSON(JsonWriter& printer) const
{
    printer.open_element("GrowingInputs");

    write_json(printer, {
        {"TrialsNumber", trials_number},
        {"WarmStart", warm_start},
        {"ValidationErrorGoal", validation_error_goal},
        {"MaximumValidationFailures", maximum_validation_failures},
        {"MinimumInputsNumber", minimum_inputs_number},
        {"MaximumInputsNumber", maximum_inputs_number},
        {"MaximumEpochsNumber", maximum_epochs},
        {"MaximumTime", maximum_time},
        {"FoldsNumber", folds_number}
    });

    printer.close_element();
}

void GrowingInputs::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "GrowingInputs");

    set_trials_number(read_json_index(root_element, "TrialsNumber"));
    set_validation_error_goal(read_json_float_alias(root_element, "ValidationErrorGoal", "SelectionErrorGoal"));
    set_maximum_epochs(read_json_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_json_float(root_element, "MaximumTime"));
    set_minimum_inputs_number(read_json_index(root_element, "MinimumInputsNumber"));
    set_maximum_inputs_number(read_json_index(root_element, "MaximumInputsNumber"));
    set_maximum_validation_failures(read_json_index_alias(root_element, "MaximumValidationFailures", "MaximumSelectionFailures"));

    if (root_element->has("FoldsNumber"))
        set_folds_number(read_json_index(root_element, "FoldsNumber"));

    if (root_element->has("WarmStart"))
        set_warm_start(read_json_bool(root_element, "WarmStart"));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
