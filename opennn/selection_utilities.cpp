//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S E L E C T I O N   U T I L I T I E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "json.h"
#include "neural_network.h"
#include "training_result.h"
#include "training_strategy.h"
#include "cross_validation.h"
#include "selection_utilities.h"

namespace opennn
{

CandidateEvaluation evaluate_candidate(TrainingStrategy* training_strategy,
                                       NeuralNetwork* neural_network,
                                       const Index folds_number,
                                       const vector<vector<Index>>& fold_partition,
                                       const Index trials_number,
                                       const bool use_validation_history_minimum,
                                       const function<void(Index, float, float, bool)>& on_trial)
{
    CandidateEvaluation evaluation;

    if (folds_number > 1)
    {
        const FoldEvaluation fold_evaluation = evaluate_folds(training_strategy, fold_partition);
        evaluation.training_error = fold_evaluation.training_error;
        evaluation.validation_error = fold_evaluation.validation_error;

        return evaluation;
    }

    for (Index trial = 0; trial < trials_number; ++trial)
    {
        neural_network->set_parameters_random();

        const TrainingResult training_results = training_strategy->train();

        const float training_error = training_results.get_training_error();

        const float validation_error =
            use_validation_history_minimum && training_results.validation_error_history.size() > 0
                ? training_results.validation_error_history.minCoeff()
                : training_results.get_validation_error();

        const bool improved = validation_error < evaluation.validation_error;

        if (improved)
        {
            evaluation.training_error = training_error;
            evaluation.validation_error = validation_error;
        }

        on_trial(trial, training_error, validation_error, improved);
    }

    return evaluation;
}

void finalize_selected_model(TrainingStrategy* training_strategy,
                             NeuralNetwork* neural_network,
                             const VectorR& optimal_parameters,
                             const Index folds_number,
                             const bool display,
                             const char* selected_label)
{
    if (optimal_parameters.size() == neural_network->get_parameters_size())
    {
        neural_network->set_parameters(optimal_parameters);
    }
    else if (folds_number > 1)
    {
        if (display) cout << "Refitting the final model on all development samples.\n";
        refit_final_model_on_development(training_strategy, folds_number);
    }
    else
    {
        if (display) cout << "Refitting the final model on the selected " << selected_label << ".\n";
        neural_network->set_parameters_random();
        training_strategy->train();
    }
}

float read_json_float_alias(const Json* root, const string_view primary, const string_view legacy)
{
    return read_json_float(root, root->has(primary) ? primary : legacy);
}

long long read_json_index_alias(const Json* root, const string_view primary, const string_view legacy)
{
    return read_json_index(root, root->has(primary) ? primary : legacy);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
