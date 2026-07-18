//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "neural_network.h"
#include "optimizer.h"
#include "training_strategy.h"
#include "growing_neurons.h"
#include "string_utilities.h"
#include "cross_validation.h"

namespace opennn
{

GrowingNeurons::GrowingNeurons(TrainingStrategy* new_training_strategy)
    : NeuronSelection(new_training_strategy)
{
    set_default();
}

void GrowingNeurons::set_default()
{
    name = "GrowingNeurons";

    minimum_neurons = 1;
    maximum_neurons = 10;
    trials_number = 3;
    neurons_increment = 1;
    maximum_validation_failures = 100;
    maximum_time = 3600.0f;
}

void GrowingNeurons::set_neurons_increment(const Index new_neurons_increment)
{
    neurons_increment = new_neurons_increment;
}

NeuronsSelectionResult GrowingNeurons::perform_neurons_selection()
{
    NeuronsSelectionResult neuron_selection_results(maximum_epochs);

    if (display) cout << "Performing growing neuron selection...\n";


    NeuralNetwork* neural_network = training_strategy->get_neural_network();

    const Index last_trainable_layer_index = neural_network->get_last_trainable_layer_index();

    throw_if(last_trainable_layer_index < 1,
             "GrowingNeurons requires a layer before the last trainable layer to resize.");

    Index neurons_number = 0;


    float previous_validation_error = MAX;


    Index validation_failures = 0;

    time_t beginning_time;
    time_t current_time;

    float elapsed_time = 0.0f;

    TrainingResult training_results;

    time(&beginning_time);

    // k-fold CV partition (folds_number > 1): built ONCE so every neuron count is scored on the same
    // folds. Empty when folds_number == 1 (legacy single Training/Validation-split scoring).
    const vector<vector<Index>> fold_partition =
        folds_number > 1 ? build_fold_partition(training_strategy, folds_number, folds_seed) : vector<vector<Index>>{};

    for (Index epoch = 0; epoch < maximum_epochs; ++epoch)
    {
        if (display) cout << "\nGrowing neurons epoch: " << epoch << "\n";


        neurons_number = minimum_neurons + epoch*neurons_increment;

        const Shape neurons_shape = { neurons_number };
        neural_network->get_layer(last_trainable_layer_index - 1)->set_output_shape(neurons_shape);
        neural_network->get_layer(last_trainable_layer_index)->set_input_shape(neurons_shape);

        neural_network->compile();


        neuron_selection_results.neurons_number_history(epoch) = neurons_number;


        float minimum_training_error = MAX;
        float minimum_validation_error = MAX;

        if (folds_number > 1)
        {
            // k-fold CV score for this neuron count. It trains k transient models and keeps none, so
            // the optimal-parameters snapshot is left empty and the final model is refit on all
            // development after selection (see below).
            const FoldEvaluation evaluation = evaluate_folds(training_strategy, fold_partition);
            minimum_training_error = evaluation.training_error;
            minimum_validation_error = evaluation.validation_error;

            neuron_selection_results.training_error_history(epoch) = minimum_training_error;
            neuron_selection_results.validation_error_history(epoch) = minimum_validation_error;

            if (minimum_validation_error < neuron_selection_results.optimum_validation_error)
            {
                neuron_selection_results.optimal_neurons_number = neurons_number;
                neuron_selection_results.optimal_parameters = VectorR();
                neuron_selection_results.optimum_training_error = minimum_training_error;
                neuron_selection_results.optimum_validation_error = minimum_validation_error;
            }

            if (display)
                cout << "Neurons: " << neurons_number << ", " << folds_number
                     << "-fold CV validation error " << minimum_validation_error << "\n";
        }
        else
        {
            for (Index trial = 0; trial < trials_number; ++trial)
            {
                neural_network->set_parameters_random();

                training_results = training_strategy->train();

                const float training_error = training_results.get_training_error();

                const float validation_error = training_results.validation_error_history.size() > 0
                    ? training_results.validation_error_history.minCoeff()
                    : training_results.get_validation_error();

                if (display)
                    cout << "Trial: " << trial+1 << "\n"
                         << "Training error: " << training_error << "\n"
                         << "Validation error: " << validation_error << "\n";

                if (validation_error < minimum_validation_error)
                {
                    minimum_training_error = training_error;
                    minimum_validation_error = validation_error;

                    neuron_selection_results.training_error_history(epoch) = minimum_training_error;
                    neuron_selection_results.validation_error_history(epoch) = minimum_validation_error;

                    if (minimum_validation_error < neuron_selection_results.optimum_validation_error)
                    {
                        neuron_selection_results.optimal_neurons_number = neurons_number;
                        neural_network->copy_parameters_host();
                        neuron_selection_results.optimal_parameters =
                            Eigen::Map<const VectorR>(neural_network->get_parameters_data(),
                                                      neural_network->get_parameters_size());
                        neuron_selection_results.optimum_training_error = minimum_training_error;
                        neuron_selection_results.optimum_validation_error = minimum_validation_error;
                    }
                }
            }
        }

        if (display)
            cout << "Neurons number: " << neurons_number << "\n"
                 << "Training error: " << minimum_training_error << "\n"
                 << "Validation error: " << minimum_validation_error << "\n"
                 << "Elapsed time: " << get_time(elapsed_time) << "\n";

        if (previous_validation_error < minimum_validation_error)
            ++validation_failures;
        else
            previous_validation_error = minimum_validation_error;

        time(&current_time);

        elapsed_time = float(difftime(current_time,beginning_time));


        if (elapsed_time >= maximum_time)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum time reached: " << get_time(elapsed_time) << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumTime;
        }
        else if (minimum_validation_error <= validation_error_goal)
        {
            if (display) cout << "Epoch " << epoch << "\nValidation error goal reached: " << minimum_validation_error << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::ValidationErrorGoal;
        }
        else if (validation_failures >= maximum_validation_failures)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum validation failures reached: " << validation_failures << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumValidationFailures;
        }
        else if (neurons_number >= maximum_neurons)
        {
            if (display) cout << "Epoch " << epoch << "\nMaximum number of neurons reached: " << neurons_number << "\n";
            neuron_selection_results.stopping_condition = GrowingNeurons::StoppingCondition::MaximumNeurons;
        }

        if (neuron_selection_results.stopping_condition)
        {
            neuron_selection_results.elapsed_time = get_time(elapsed_time);

            neuron_selection_results.resize_history(epoch+1);

            break;
        }
    }


    if (display)
        cout << "Parameters number: " << neuron_selection_results.optimal_parameters.size() << "\n";

    const Shape optimal_shape = { neuron_selection_results.optimal_neurons_number };
    neural_network->get_layer(last_trainable_layer_index - 1)->set_output_shape(optimal_shape);
    neural_network->get_layer(last_trainable_layer_index)->set_input_shape(optimal_shape);

    neural_network->compile();

    if (neuron_selection_results.optimal_parameters.size() == neural_network->get_parameters_size())
    {
        neural_network->set_parameters(neuron_selection_results.optimal_parameters);
    }
    else if (folds_number > 1)
    {
        // k-fold CV path: refit the final model on ALL development samples with the selected neuron
        // count, using the epoch budget the CV of that architecture found best.
        if (display) cout << "Refitting the final model on all development samples.\n";
        refit_final_model_on_development(training_strategy, folds_number, folds_seed);
    }
    else if (display)
        cout << "Warning: no optimal parameter snapshot captured; keeping current weights.\n";

    if (display) neuron_selection_results.print();

    return neuron_selection_results;
}

void GrowingNeurons::to_JSON(JsonWriter& printer) const
{
    printer.open_element("GrowingNeurons");

    write_json(printer, {
        {"MinimumNeurons", to_string(minimum_neurons)},
        {"MaximumNeurons", to_string(maximum_neurons)},
        {"NeuronsIncrement", to_string(neurons_increment)},
        {"TrialsNumber", to_string(trials_number)},
        {"ValidationErrorGoal", to_string(validation_error_goal)},
        {"MaximumValidationFailures", to_string(maximum_validation_failures)},
        {"MaximumTime", to_string(maximum_time)},
        {"FoldsNumber", to_string(folds_number)}
    });

    printer.close_element();
}

void GrowingNeurons::from_JSON(const JsonDocument& document)
{
    const Json* root_element = get_json_root(document, "GrowingNeurons");

    set_minimum_neurons(read_json_index(root_element, "MinimumNeurons"));
    set_maximum_neurons(read_json_index(root_element, "MaximumNeurons"));
    set_neurons_increment(read_json_index(root_element, "NeuronsIncrement"));
    set_trials_number(read_json_index(root_element, "TrialsNumber"));
    set_validation_error_goal(read_json_float(root_element,
        root_element->has("ValidationErrorGoal") ? "ValidationErrorGoal" : "SelectionErrorGoal"));
    set_maximum_validation_failures(read_json_index(root_element,
        root_element->has("MaximumValidationFailures") ? "MaximumValidationFailures" : "MaximumSelectionFailures"));
    set_maximum_time(read_json_float(root_element, "MaximumTime"));

    // Backward compatible: projects saved before k-fold CV have no FoldsNumber -> keep legacy folds=1.
    if (root_element->has("FoldsNumber"))
        set_folds_number(read_json_index(root_element, "FoldsNumber"));
}

REGISTER(NeuronSelection, GrowingNeurons, "GrowingNeurons");

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
