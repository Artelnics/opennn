//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dataset.h"
#include "tabular_dataset.h"
#include "time_series_dataset.h"
#include "growing_inputs.h"
#include "correlations.h"
#include "scaling_layer.h"
#include "optimizer.h"
#include "training_strategy.h"
#include "cross_validation.h"

namespace opennn
{

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
        ? training_strategy->get_dataset()->get_variables_number("Input")
        : 50;
}

void GrowingInputs::set_maximum_inputs_number(const Index new_maximum_inputs_number)
{
    const Index inputs_number = training_strategy->get_dataset()->get_variables_number("Input");

    maximum_inputs_number = (inputs_number == 0)
                                ? new_maximum_inputs_number
                                : min(new_maximum_inputs_number, inputs_number);
}

InputsSelectionResult GrowingInputs::perform_input_selection()
{

    Dataset* dataset = training_strategy->get_dataset();
    const Index original_input_variables_number = dataset->get_variables_number("Input");

    if (dataset->has_nan())
        dataset->scrub_missing_values();

    if (display) cout << "Performing growing input selection...\n";

    InputsSelectionResult input_selection_results(original_input_variables_number);


    training_strategy->get_optimization_algorithm()->set_display(false);

    float previous_validation_error = MAX;

    const vector<Index> target_variable_indices = dataset->get_variable_indices("Target");
    const vector<Index> time_variable_indices = dataset->get_variable_indices("Time");
    const vector<string> variable_names = dataset->get_variable_names();

    if (display) cout << "Calculating correlations...\n";

    const VectorR total_correlations =
        get_correlation_values(dataset->calculate_input_target_variable_pearson_correlations()).col(0).array().abs();

    vector<Index> correlation_indices(original_input_variables_number);
    iota(correlation_indices.begin(), correlation_indices.end(), 0);

    ranges::sort(correlation_indices,
                 [&](Index i, Index j) {return total_correlations[i] > total_correlations[j]; });

    const vector<Index> input_variable_indices = dataset->get_variable_indices("Input");

    VectorI correlations_rank_descending(input_variable_indices.size());

    ranges::transform(correlation_indices,
                      correlations_rank_descending.data(),
                      [&input_variable_indices](Index correlation_index) { return input_variable_indices[correlation_index]; });

    dataset->set_input_variables_unused();

    Index variable_index = 0;


    NeuralNetwork* neural_network = training_strategy->get_neural_network();


    Index validation_failures = 0;
    TrainingResult training_results;


    time_t beginning_time;
    time_t current_time;
    float elapsed_time = 0.0f;
    time(&beginning_time);

    Index epoch = 0;

    // k-fold CV partition (folds_number > 1): built ONCE so every candidate subset is scored on the
    // same folds. Empty when folds_number == 1 (legacy single Training/Validation-split scoring).
    const vector<vector<Index>> fold_partition =
        folds_number > 1 ? build_fold_partition(training_strategy, folds_number, folds_seed) : vector<vector<Index>>{};

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

        const Index input_variables_number = dataset->get_variables_number("Input");
        const Index input_features_number = dataset->get_features_number("Input");

        if (input_variables_number < minimum_inputs_number)
        {
            ++variable_index;
            continue;
        }

        configure_neural_network_inputs(neural_network, dataset, input_features_number);

        // Growing inputs is a greedy forward selection: at each step it takes the
        // next candidate input (by descending correlation) and keeps it ONLY if it
        // lowers the validation error. Rejected candidates are removed and the next
        // one is tried, so several candidates can be evaluated at the same input
        // count. The epoch counter therefore tracks *accepted* inputs, not attempts.
        const string& candidate_name = variable_names[current_variable_index];

        if (display)
            cout << "\nTrying to add \"" << candidate_name << "\"  ->  "
                 << input_variables_number << " inputs\n";

        float minimum_training_error = MAX;
        float minimum_validation_error = MAX;

        if (folds_number > 1)
        {
            // k-fold CV score: robust, less overfittable than a single validation split. It trains k
            // transient models and keeps none, so the optimal-parameters snapshot is left empty and
            // the final model is refit on all development after selection (see below).
            const FoldEvaluation evaluation = evaluate_folds(training_strategy, fold_partition);
            minimum_training_error = evaluation.training_error;
            minimum_validation_error = evaluation.validation_error;

            if (minimum_validation_error < input_selection_results.optimum_validation_error)
            {
                input_selection_results.optimal_input_variables_indices = dataset->get_variable_indices("Input");
                input_selection_results.optimal_input_variable_names = dataset->get_variable_names("Input");
                input_selection_results.optimal_parameters = VectorR();
                input_selection_results.optimum_training_error = minimum_training_error;
                input_selection_results.optimum_validation_error = minimum_validation_error;
            }

            if (display)
                cout << "   " << folds_number << "-fold CV validation error " << minimum_validation_error << "\n";
        }
        else
        {
            for (Index j = 0; j < trials_number; ++j)
            {
                neural_network->set_parameters_random();
                training_results = training_strategy->train();

                const float training_error = training_results.get_training_error();
                const float validation_error = training_results.get_validation_error();

                if (validation_error < minimum_validation_error)
                {
                    minimum_training_error = training_error;
                    minimum_validation_error = validation_error;

                    if (minimum_validation_error < input_selection_results.optimum_validation_error)
                    {
                        input_selection_results.optimal_input_variables_indices = dataset->get_variable_indices("Input");
                        input_selection_results.optimal_input_variable_names = dataset->get_variable_names("Input");
                        neural_network->copy_parameters_host();
                        input_selection_results.optimal_parameters =
                            Eigen::Map<const VectorR>(neural_network->get_parameters_data(),
                                                      neural_network->get_parameters_size());
                        input_selection_results.optimum_training_error = training_error;
                        input_selection_results.optimum_validation_error = validation_error;
                    }
                }

                if (display)
                    cout << (trials_number > 1 ? "   Trial " + to_string(j + 1) + ": " : "   ")
                         << "training error " << training_error
                         << ", validation error " << validation_error << "\n";
            }
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
        }
        else
        {
            previous_validation_error = minimum_validation_error;

            input_selection_results.training_error_history(epoch) = minimum_training_error;
            input_selection_results.validation_error_history(epoch) = minimum_validation_error;

            ++epoch;

            if (display)
                cout << "   Accepted. Epoch " << epoch << ": " << input_variables_number
                     << " inputs kept, best validation error " << minimum_validation_error << "\n"
                     << "   Inputs: " << dataset->get_variable_names("Input");
        }

        ++variable_index;
        time(&current_time);
        elapsed_time = float(difftime(current_time, beginning_time));


        if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum time reached: " << get_time(elapsed_time) << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumTime;
        }
        else if (input_selection_results.optimum_validation_error <= validation_error_goal)
        {
            if (display) cout << "\nValidation error goal reached: " << input_selection_results.optimum_validation_error << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::ValidationErrorGoal;
        }
        else if (epoch >= maximum_epochs)
        {
            if (display) cout << "\nMaximum number of epochs reached." << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumEpochs;
        }
        else if (validation_failures >= maximum_validation_failures)
        {
            if (display) cout << "\nMaximum validation failures (" << validation_failures << ") reached." << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumValidationFailures;
        }
        else if (const Index current_inputs = dataset->get_variables_number("Input");
                 current_inputs >= maximum_inputs_number)
        {
            if (display) cout << "\nMaximum inputs (" << current_inputs << ") reached." << "\n";
            input_selection_results.stopping_condition = InputsSelection::StoppingCondition::MaximumInputs;
        }
    }

    input_selection_results.elapsed_time = get_time(elapsed_time);
    input_selection_results.resize_history(epoch);


    dataset->set_variable_indices(input_selection_results.optimal_input_variables_indices,
        target_variable_indices);

    const Index optimal_processed_variables_number = dataset->get_features_number("Input");

    if (dynamic_cast<TimeSeriesDataset*>(dataset) && time_variable_indices.size() == 1)
        dataset->set_variable_role(time_variable_indices[0], "Time");

    configure_neural_network_inputs(neural_network, dataset, optimal_processed_variables_number);

    auto* tabular_dataset = dynamic_cast<TabularDataset*>(dataset);
    const vector<string> input_variable_scalers = tabular_dataset ? tabular_dataset->get_feature_scalers("Input") : vector<string>{};
    const vector<Descriptives> input_variable_descriptives = dataset->calculate_feature_descriptives("Input");

    set_maximum_inputs_number(dataset->get_variables_number("Input"));

    if (auto* scaling_layer = dynamic_cast<Scaling*>(neural_network->get_first(LayerType::Scaling)))
    {
        scaling_layer->set_descriptives(input_variable_descriptives);
        scaling_layer->set_scalers(input_variable_scalers);
    }

    if (input_selection_results.optimal_parameters.size() == neural_network->get_parameters_size())
    {
        neural_network->set_parameters(input_selection_results.optimal_parameters);
    }
    else if (folds_number > 1)
    {
        // k-fold CV path: refit the final model on ALL development samples (Training + Validation),
        // using the epoch budget the CV of the selected subset found best.
        if (display) cout << "Refitting the final model on all development samples.\n";
        refit_final_model_on_development(training_strategy, folds_number, folds_seed);
    }
    else
    {
        // No snapshot with folds=1 (changed parameter layout): refit on the user's split.
        if (display) cout << "Refitting the final model on the selected inputs.\n";
        neural_network->set_parameters_random();
        training_strategy->train();
    }

    if (display) input_selection_results.print();

    return input_selection_results;
}

void GrowingInputs::to_JSON(JsonWriter& printer) const
{
    printer.open_element("GrowingInputs");

    write_json(printer, {
        {"TrialsNumber", to_string(trials_number)},
        {"ValidationErrorGoal", to_string(validation_error_goal)},
        {"MaximumValidationFailures", to_string(maximum_validation_failures)},
        {"MinimumInputsNumber", to_string(minimum_inputs_number)},
        {"MaximumInputsNumber", to_string(maximum_inputs_number)},
        {"MaximumEpochsNumber", to_string(maximum_epochs)},
        {"MaximumTime", to_string(maximum_time)},
        {"FoldsNumber", to_string(folds_number)}
    });

    printer.close_element();
}

void GrowingInputs::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "GrowingInputs");

    set_trials_number(read_json_index(root_element, "TrialsNumber"));
    set_validation_error_goal(read_json_float(root_element,
        root_element->has("ValidationErrorGoal") ? "ValidationErrorGoal" : "SelectionErrorGoal"));
    set_maximum_epochs(read_json_index(root_element, "MaximumEpochsNumber"));
    set_maximum_time(read_json_float(root_element, "MaximumTime"));
    set_minimum_inputs_number(read_json_index(root_element, "MinimumInputsNumber"));
    set_maximum_inputs_number(read_json_index(root_element, "MaximumInputsNumber"));
    set_maximum_validation_failures(read_json_index(root_element,
        root_element->has("MaximumValidationFailures") ? "MaximumValidationFailures" : "MaximumSelectionFailures"));

    // Backward compatible: projects saved before k-fold CV have no FoldsNumber -> keep legacy folds=1.
    if (root_element->has("FoldsNumber"))
        set_folds_number(read_json_index(root_element, "FoldsNumber"));
}

REGISTER(InputsSelection, GrowingInputs, "GrowingInputs");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
