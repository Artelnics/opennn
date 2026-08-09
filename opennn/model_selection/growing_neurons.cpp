//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/optimizer.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/model_selection/growing_neurons.h"
#include "opennn/core/string_utilities.h"
#include "opennn/model_selection/cross_validation.h"
#include "opennn/model_selection/selection_utilities.h"

namespace opennn
{

namespace
{

// Neuron selection builds a rank-1 shape and hands it to the last trainable
// layer. Only layers that accept rank 1 can honour that. Layers which cannot
// used to react in three different ways - a throw, or silently keeping their
// previous shape while selection went on to report results for a network it
// never actually changed. Ask first, and say plainly what is wrong.
void require_grows_by_neurons(const Layer& layer)
{
    throw_if(!layer.accepts_input_rank(1),
             "GrowingNeurons: the last trainable layer is {}, which does not accept a "
             "rank-1 input shape. Neuron selection needs a Dense-like layer there.",
             layer.get_name());
}

}


GrowingNeurons::GrowingNeurons(TrainingStrategy* new_training_strategy)
{
    set(new_training_strategy);
}

void GrowingNeurons::set(TrainingStrategy* new_training_strategy)
{
    training_strategy = new_training_strategy;

    set_default();
}

void GrowingNeurons::set_default()
{
    minimum_neurons = 1;
    maximum_neurons = 10;
    trials_number = 3;
    neurons_increment = 1;
    validation_error_goal = 0.0f;
    maximum_epochs = 1000;
    maximum_validation_failures = 100;
    maximum_time = 3600.0f;
    display = true;

    const NeuralNetwork* neural_network = training_strategy
        ? training_strategy->get_neural_network()
        : nullptr;

    if (!neural_network) return;

    maximum_neurons = 2 * (neural_network->get_inputs_number() + neural_network->get_outputs_number());
    trials_number = 1;
}

void GrowingNeurons::save(const filesystem::path& file_name) const
{
    save_json_file(file_name, *this);
}

void GrowingNeurons::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
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

    float elapsed_time = 0.0f;

    time(&beginning_time);

    const vector<vector<Index>> fold_partition =
        folds_number > 1 ? build_fold_partition(training_strategy, folds_number) : vector<vector<Index>>{};

    ParameterSnapshot warm_snapshot;
    ParameterSnapshot candidate_snapshot;

    for (Index epoch = 0; epoch < maximum_epochs; ++epoch)
    {
        if (display) cout << "\nGrowing neurons epoch: " << epoch << "\n";

        neurons_number = minimum_neurons + epoch*neurons_increment;

        const Shape neurons_shape = { neurons_number };
        require_grows_by_neurons(*neural_network->get_layer(last_trainable_layer_index));

        neural_network->get_layer(last_trainable_layer_index - 1)->set_output_shape(neurons_shape);
        neural_network->get_layer(last_trainable_layer_index)->set_input_shape(neurons_shape);

        neural_network->compile();

        neuron_selection_results.neurons_number_history(epoch) = neurons_number;

        const CandidateEvaluation candidate_evaluation = evaluate_candidate(
            training_strategy, neural_network, folds_number, fold_partition, trials_number, true,
            [&](Index trial, float training_error, float validation_error, bool improved)
            {
                if (display)
                    cout << "Trial: " << trial+1 << "\n"
                         << "Training error: " << training_error << "\n"
                         << "Validation error: " << validation_error << "\n";

                if (improved)
                {
                    if (warm_start)
                        candidate_snapshot = capture_parameter_snapshot(neural_network);

                    neuron_selection_results.training_error_history(epoch) = training_error;
                    neuron_selection_results.validation_error_history(epoch) = validation_error;

                    if (validation_error < neuron_selection_results.optimum_validation_error)
                    {
                        neuron_selection_results.optimal_neurons_number = neurons_number;
                        neural_network->copy_parameters_host();
                        neuron_selection_results.optimal_parameters = neural_network->get_parameters_map();
                        neuron_selection_results.optimum_training_error = training_error;
                        neuron_selection_results.optimum_validation_error = validation_error;
                    }
                }
            },
            [&](Index trial)
            {
                neural_network->set_parameters_random();

                if (trial == 0 && warm_start && !warm_snapshot.empty())
                    seed_parameters_from_snapshot(neural_network, warm_snapshot);
            });

        if (warm_start && !candidate_snapshot.empty())
        {
            warm_snapshot = move(candidate_snapshot);
            candidate_snapshot = {};
        }

        const float minimum_training_error = candidate_evaluation.training_error;
        const float minimum_validation_error = candidate_evaluation.validation_error;

        if (folds_number > 1)
        {
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

        elapsed_time = get_elapsed_time(beginning_time);

        if (display)
            cout << "Neurons number: " << neurons_number << "\n"
                 << "Training error: " << minimum_training_error << "\n"
                 << "Validation error: " << minimum_validation_error << "\n"
                 << "Elapsed time: " << get_time(elapsed_time) << "\n";

        if (previous_validation_error < minimum_validation_error)
            ++validation_failures;
        else
            previous_validation_error = minimum_validation_error;

        neuron_selection_results.stopping_condition = first_stopping_condition<StoppingCondition>(display,
        {
            {elapsed_time >= maximum_time, StoppingCondition::MaximumTime,
             format("Epoch {}\nMaximum time reached: {}\n", epoch, get_time(elapsed_time))},
            {minimum_validation_error <= validation_error_goal, StoppingCondition::ValidationErrorGoal,
             format("Epoch {}\nValidation error goal reached: {:g}\n", epoch, minimum_validation_error)},
            {validation_failures >= maximum_validation_failures, StoppingCondition::MaximumValidationFailures,
             format("Epoch {}\nMaximum validation failures reached: {}\n", epoch, validation_failures)},
            {neurons_number >= maximum_neurons, StoppingCondition::MaximumNeurons,
             format("Epoch {}\nMaximum number of neurons reached: {}\n", epoch, neurons_number)}
        });

        if (neuron_selection_results.stopping_condition)
        {
            neuron_selection_results.elapsed_time = get_time(elapsed_time);

            neuron_selection_results.resize_history(epoch+1);

            break;
        }
    }

    if (display)
        cout << "Parameters number: " << neuron_selection_results.optimal_parameters.size() << "\n";

    require_grows_by_neurons(*neural_network->get_layer(last_trainable_layer_index));

    const Shape optimal_shape = { neuron_selection_results.optimal_neurons_number };
    neural_network->get_layer(last_trainable_layer_index - 1)->set_output_shape(optimal_shape);
    neural_network->get_layer(last_trainable_layer_index)->set_input_shape(optimal_shape);

    neural_network->compile();

    finalize_selected_model(training_strategy, neural_network,
                            neuron_selection_results.optimal_parameters, folds_number, display, "neurons");

    if (display) neuron_selection_results.print();

    return neuron_selection_results;
}

void GrowingNeurons::to_JSON(JsonWriter& printer) const
{
    printer.open_element("GrowingNeurons");

    write_json(printer, {
        {"MinimumNeurons", minimum_neurons},
        {"MaximumNeurons", maximum_neurons},
        {"NeuronsIncrement", neurons_increment},
        {"TrialsNumber", trials_number},
        {"WarmStart", warm_start},
        {"ValidationErrorGoal", validation_error_goal},
        {"MaximumValidationFailures", maximum_validation_failures},
        {"MaximumTime", maximum_time},
        {"FoldsNumber", folds_number}
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
    set_validation_error_goal(read_json_float_alias(root_element, "ValidationErrorGoal", "SelectionErrorGoal"));
    set_maximum_validation_failures(read_json_index_alias(root_element, "MaximumValidationFailures", "MaximumSelectionFailures"));
    set_maximum_time(read_json_float(root_element, "MaximumTime"));

    if (root_element->has("FoldsNumber"))
        set_folds_number(read_json_index(root_element, "FoldsNumber"));

    if (root_element->has("WarmStart"))
        set_warm_start(read_json_bool(root_element, "WarmStart"));
}

NeuronsSelectionResult::NeuronsSelectionResult(const Index maximum_epochs)
{
    neurons_number_history = VectorI::Zero(maximum_epochs);

    training_error_history = VectorR::Constant(maximum_epochs, -1.0f);
    validation_error_history = VectorR::Constant(maximum_epochs, -1.0f);

    optimum_training_error = MAX;
    optimum_validation_error = MAX;
}

void NeuronsSelectionResult::resize_history(const Index new_size)
{
    neurons_number_history.conservativeResize(new_size);
    training_error_history.conservativeResize(new_size);
    validation_error_history.conservativeResize(new_size);
}

void NeuronsSelectionResult::print() const
{
    cout << "\n"
         << "Neuron Selection Results" << "\n"
         << "Optimal neurons number: " << optimal_neurons_number << "\n"
         << "Optimum training error: " << optimum_training_error << "\n"
         << "Optimum validation error: " << optimum_validation_error << "\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
