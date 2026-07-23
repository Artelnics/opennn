//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C R O S S   V A L I D A T I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

class TrainingStrategy;


struct FoldEvaluation
{
    float validation_error = 0.0f;
    float training_error = 0.0f;
    Index epochs = 0;
};

vector<vector<Index>> build_fold_partition(TrainingStrategy* training_strategy,
                                           Index folds_number,
                                           Index folds_seed = 0);

FoldEvaluation evaluate_folds(TrainingStrategy* training_strategy,
                              const vector<vector<Index>>& fold_partition);

void refit_final_model_on_development(TrainingStrategy* training_strategy,
                                      Index folds_number,
                                      Index folds_seed = 0);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
