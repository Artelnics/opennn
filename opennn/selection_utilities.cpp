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
                                       const function<void(Index, float, float, bool)>& on_trial,
                                       const function<void(Index)>& initialize_trial)
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
        initialize_trial ? initialize_trial(trial) : neural_network->set_parameters_random();

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

ParameterSnapshot capture_parameter_snapshot(NeuralNetwork* neural_network)
{
    neural_network->copy_parameters_host();

    const auto& layers = neural_network->get_layers();

    ParameterSnapshot snapshot;
    snapshot.layers.resize(layers.size());

    for (size_t i = 0; i < layers.size(); ++i)
    {
        const Layer::TiedWeight tie = layers[i]->get_tied_weight();
        const vector<TensorView>& views = layers[i]->get_parameter_views();

        auto& blocks = snapshot.layers[i];
        blocks.resize(views.size());

        for (size_t v = 0; v < views.size(); ++v)
        {
            if (views[v].empty() || (tie.source && v == tie.spec_index)) continue;

            blocks[v].shape = views[v].shape;
            blocks[v].values.assign(views[v].as_float(), views[v].as_float() + views[v].size());
        }
    }

    return snapshot;
}

void seed_parameters_from_snapshot(NeuralNetwork* neural_network,
                                   const ParameterSnapshot& snapshot,
                                   const vector<Index>& input_row_map)
{
    if (snapshot.empty()) return;

    const bool was_on_device = neural_network->get_parameters_device() == Device::CUDA;
    if (was_on_device) neural_network->copy_parameters_host();

    const auto& layers = neural_network->get_layers();
    const Index first_trainable = neural_network->get_first_trainable_layer_index();
    const size_t common_layers = min(layers.size(), snapshot.layers.size());

    for (size_t i = 0; i < common_layers; ++i)
    {
        const Layer::TiedWeight tie = layers[i]->get_tied_weight();
        const vector<TensorView>& views = layers[i]->get_parameter_views();
        const auto& blocks = snapshot.layers[i];
        const size_t common_views = min(views.size(), blocks.size());

        for (size_t v = 0; v < common_views; ++v)
        {
            const ParameterSnapshot::Block& block = blocks[v];
            const TensorView& view = views[v];

            if (block.values.empty() || view.empty() || (tie.source && v == tie.spec_index))
                continue;

            if (view.shape == block.shape)
            {
                copy(block.values.begin(), block.values.end(), view.as_float());
                continue;
            }

            if (view.shape.rank == 1 && block.shape.rank == 1)
            {
                const Index count = min(view.size(), Index(block.values.size()));
                copy_n(block.values.begin(), count, view.as_float());
                continue;
            }

            if (view.shape.rank != 2 || block.shape.rank != 2) continue;

            MatrixMap destination = view.as_matrix();
            const Eigen::Map<const MatrixR> source(block.values.data(),
                                                   block.shape[0], block.shape[1]);

            const Index columns = min(destination.cols(), source.cols());

            if (Index(i) != first_trainable
                || ssize(input_row_map) != destination.rows())
            {
                const Index rows = min(destination.rows(), source.rows());
                destination.topLeftCorner(rows, columns) = source.topLeftCorner(rows, columns);
                continue;
            }

            for (Index row = 0; row < destination.rows(); ++row)
                if (input_row_map[row] >= 0 && input_row_map[row] < source.rows())
                    destination.row(row).head(columns) = source.row(input_row_map[row]).head(columns);
        }
    }

    if (was_on_device) neural_network->copy_parameters_device();
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
