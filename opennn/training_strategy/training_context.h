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

struct TrainingContext
{
    TrainingContext(Index batch_size, Loss&, bool inputs_pre_scaled = false,
                    TrainingContext* share_memory_with = nullptr,
                    bool joint_gradient_arena = false);

    TrainingContext(const TrainingContext&) = delete;
    TrainingContext& operator=(const TrainingContext&) = delete;

    bool shares_memory() const noexcept { return !forward.arena.owns_memory(); }

    ForwardPropagation forward;
    BackPropagation backward;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
