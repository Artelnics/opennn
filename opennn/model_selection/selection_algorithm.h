//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S E L E C T I O N   A L G O R I T H M   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{

class TrainingStrategy;

// What every selection algorithm needs regardless of what it selects: the
// training strategy it drives, how many trials and folds each candidate gets,
// and the four stopping conditions. InputsSelection and GrowingNeurons each
// declared this block for themselves, setters included, and the two copies had
// to be kept in step by hand.
//
// The setter names and signatures are the ones both classes already exposed --
// Neural Designer calls several of them -- so this changes where they are
// declared and nothing else.
class SelectionAlgorithm
{
public:

    explicit SelectionAlgorithm(TrainingStrategy* new_training_strategy = nullptr)
        : training_strategy(new_training_strategy)
    {
    }

    virtual ~SelectionAlgorithm() = default;

    const TrainingStrategy* get_training_strategy() const noexcept { return training_strategy; }

    void set(TrainingStrategy* new_training_strategy) { training_strategy = new_training_strategy; }

    void set_trials_number(const Index new_trials_number) { trials_number = new_trials_number; }

    void set_display(bool new_display) { display = new_display; }

    void set_validation_error_goal(const float new_validation_error_goal) { validation_error_goal = new_validation_error_goal; }
    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_validation_failures(const Index new_maximum_validation_failures) { maximum_validation_failures = new_maximum_validation_failures; }
    void set_maximum_time(const float new_maximum_time) { maximum_time = new_maximum_time; }

    // Never below one: a zero here would divide by zero in the fold loop.
    void set_folds_number(const Index new_folds_number) { folds_number = max<Index>(new_folds_number, Index(1)); }

protected:

    TrainingStrategy* training_strategy = nullptr;

    Index trials_number = 1;

    Index folds_number = 1;

    bool display = true;

    float validation_error_goal = 0;

    Index maximum_epochs = 10;

    Index maximum_validation_failures = 100;

    float maximum_time = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
