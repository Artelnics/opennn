//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   V A L I D A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"

namespace opennn
{

class TrainingStrategy;

// Shared k-fold cross-validation used to score candidates during model selection (input selection
// and neuron selection). The folds partition the Training+Validation pool; scoring installs a
// transient FoldScope per fold, so the dataset's persistent sample roles are never mutated.

// Result of scoring the currently-configured model by k-fold CV: the errors averaged over folds plus
// the mean best epoch (the epoch budget for refitting the final model without a validation set).
struct FoldEvaluation
{
    float validation_error = 0.0f;
    float training_error = 0.0f;
    Index epochs = 0;
};

// Build the k-fold partition of the Training+Validation pool ONCE per selection run. Returns k index
// lists (the validation set of each fold). Stratified by target class for classification, sequential
// contiguous blocks for time series. Deterministic given folds_seed.
vector<vector<Index>> build_fold_partition(TrainingStrategy* training_strategy,
                                           Index folds_number,
                                           Index folds_seed = 0);

// Score the currently-configured model by k-fold CV over the given partition.
FoldEvaluation evaluate_folds(TrainingStrategy* training_strategy,
                              const vector<vector<Index>>& fold_partition);

// After a CV-based selection, retrain the currently-configured final model on ALL development samples
// (Training + Validation, no held-out validation), for the epoch budget the CV of the selected
// configuration found best. The dataset's persistent roles are restored on return.
void refit_final_model_on_development(TrainingStrategy* training_strategy,
                                      Index folds_number,
                                      Index folds_seed = 0);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
