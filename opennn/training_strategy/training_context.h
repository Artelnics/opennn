//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   C O N T E X T   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"

namespace opennn
{

class Loss;

// Everything one batch size needs to train: the activations, the deltas planned
// into the same arena as those activations, and the gradient.
//
// Building the two halves in the right order is the whole point. The delta
// lifetimes have to be known before the forward arena is planned, because they
// are planned into it; the deltas can only be bound once that arena exists. Left
// to the caller, that is three statements that have to appear in one order, and
// a training run needs the sequence twice - once for whole batches, once for the
// remainder.
//
// A context built over another one lays its arena and its gradient on top of the
// other's rather than allocating a second set. That is what the remainder batch
// does: it is smaller than a whole batch and runs only after every whole batch
// has been consumed and its update applied, so the memory is free by then. Both
// sharing paths are checked rather than hoped for.
struct TrainingContext
{
    TrainingContext(Index batch_size, Loss&, bool inputs_pre_scaled = false,
                    TrainingContext* share_memory_with = nullptr);

    TrainingContext(const TrainingContext&) = delete;
    TrainingContext& operator=(const TrainingContext&) = delete;

    bool shares_memory() const noexcept { return !forward.arena.owns_memory(); }

    // Declared in this order: the deltas bind into the arena the forward pass
    // owns, so the forward half has to be alive first.
    ForwardPropagation forward;
    BackPropagation backward;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
