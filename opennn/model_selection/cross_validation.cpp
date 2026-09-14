// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/model_selection/cross_validation.h"

#include <cmath>
#include <limits>
#include <random>
#include <set>

#include "opennn/dataset/dataset.h"
#include "opennn/network/network.h"
#include "opennn/training/optimizer.h"
#include "opennn/training/training.h"

namespace opennn
{

vector<vector<Index>> build_fold_partition(Training* training, Index folds_number, Index folds_seed)
{
    throw_if(!training || !training->get_dataset(),
             "Cross-validation requires a training configuration with a dataset.");
    Dataset* dataset = training->get_dataset();
    const Index k = folds_number;

    vector<Index> development = dataset->get_sample_indices(SampleRole::Training);
    const vector<Index> validation = dataset->get_sample_indices(SampleRole::Validation);
    development.insert(development.end(), validation.begin(), validation.end());

    throw_if(k < 2 || k > ssize(development),
             "Cross-validation requires between 2 and {} nonempty folds; requested {}.",
             development.size(), k);

    vector<vector<Index>> folds(static_cast<size_t>(k));

    auto deal_blocks = [&folds, k](const vector<Index>& items)
    {
        const Index n = ssize(items);
        for (Index f = 0; f < k; ++f)
            for (Index j = f * n / k; j < (f + 1) * n / k; ++j)
                folds[size_t(f)].push_back(items[j]);
    };

    size_t next_fold = 0;
    auto deal_round_robin = [&folds, k, &next_fold](const vector<Index>& items)
    {
        for (const Index sample : items)
        {
            folds[next_fold].push_back(sample);
            next_fold = (next_fold + 1) % size_t(k);
        }
    };

    if (dataset->sample_order_matters())
    {
        ranges::sort(development);
        deal_blocks(development);
        return folds;
    }

    mt19937 rng(static_cast<unsigned>(folds_seed));
    const vector<Index> target_features = dataset->get_feature_indices(VariableRole::Target);

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

FoldEvaluation evaluate_folds(Training* training, const vector<vector<Index>>& fold_partition)
{
    throw_if(!training || !training->get_dataset() || !training->get_network()
             || !training->get_optimization_algorithm(),
             "Cross-validation requires a dataset, network and optimizer.");
    Dataset* dataset = training->get_dataset();
    Network* network = training->get_network();
    const Index k = ssize(fold_partition);
    throw_if(k < 2, "Cross-validation requires at least two nonempty folds.");

    vector<Index> eligible = dataset->get_sample_indices(SampleRole::Training);
    const vector<Index> validation = dataset->get_sample_indices(SampleRole::Validation);
    eligible.insert(eligible.end(), validation.begin(), validation.end());
    const std::set<Index> eligible_set(eligible.begin(), eligible.end());
    std::set<Index> seen;

    vector<Index> development;
    for (const vector<Index>& fold : fold_partition)
    {
        throw_if(fold.empty(), "Cross-validation folds cannot be empty.");
        for (const Index sample : fold)
        {
            throw_if(!eligible_set.contains(sample),
                     "Cross-validation sample {} is not a training or validation sample.", sample);
            throw_if(!seen.insert(sample).second,
                     "Cross-validation sample {} appears more than once.", sample);
            development.push_back(sample);
        }
    }
    throw_if(seen.size() != eligible_set.size(),
             "Cross-validation folds must cover every training and validation sample.");

    double validation_error_sum = 0.0;
    double training_error_sum = 0.0;
    bool valid_validation_errors = true;
    bool valid_training_errors = true;
    Index epochs_sum = 0;

    for (Index f = 0; f < k; ++f)
    {
        const vector<Index>& validation_indices = fold_partition[size_t(f)];
        const std::set<Index> validation_set(validation_indices.begin(), validation_indices.end());

        vector<Index> training_indices;
        training_indices.reserve(development.size());
        ranges::copy_if(development, back_inserter(training_indices),
                        [&validation_set](const Index sample) { return !validation_set.contains(sample); });

        FoldScope scope(*dataset, training_indices, validation_indices);

        network->set_parameters_random();
        const TrainingResult training_results = training->train();

        const float validation_error = training_results.get_validation_error();
        const float training_error = training_results.get_training_error();
        valid_validation_errors = valid_validation_errors && isfinite(validation_error);
        valid_training_errors = valid_training_errors && isfinite(training_error);

        const Index fold_epochs = training_results.restored_epoch
            ? *training_results.restored_epoch + 1
            : training_results.get_epochs_number();

        if (isfinite(validation_error)) validation_error_sum += double(validation_error);
        if (isfinite(training_error)) training_error_sum += double(training_error);
        epochs_sum += max<Index>(fold_epochs, Index(1));
    }

    FoldEvaluation evaluation;
    evaluation.validation_error = valid_validation_errors ? float(validation_error_sum / double(k)) : MAX;
    evaluation.training_error = valid_training_errors ? float(training_error_sum / double(k)) : MAX;
    evaluation.epochs = max<Index>(epochs_sum / k, Index(1));
    return evaluation;
}

void refit_final_model_on_development(Training* training, Index folds_number, Index folds_seed)
{
    Dataset* dataset = training->get_dataset();
    Network* network = training->get_loss()->get_network();
    Optimizer* optimizer = training->get_optimization_algorithm();

    const Index final_epochs = evaluate_folds(training, build_fold_partition(training, folds_number, folds_seed)).epochs;

    vector<Index> development = dataset->get_sample_indices(SampleRole::Training);
    const vector<Index> validation = dataset->get_sample_indices(SampleRole::Validation);
    development.insert(development.end(), validation.begin(), validation.end());

    const Index saved_epochs = optimizer->get_maximum_epochs();
    optimizer->set_maximum_epochs(final_epochs);

    ScopeExit epochs_cleanup([optimizer, saved_epochs] { optimizer->set_maximum_epochs(saved_epochs); });

    FoldScope scope(*dataset, development, {});
    network->set_parameters_random();
    training->train();
}

}
