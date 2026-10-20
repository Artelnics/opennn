// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{

class Training;

struct FoldEvaluation
{
    float validation_error = 0.0f;
    float training_error = 0.0f;
    Index epochs = 0;
};

vector<vector<Index>> build_fold_partition(Training* training,
                                           Index folds_number,
                                           Index folds_seed = 0);

FoldEvaluation evaluate_folds(Training* training,
                              const vector<vector<Index>>& fold_partition);

void refit_final_model_on_development(Training* training,
                                      Index folds_number,
                                      Index folds_seed = 0);

}
