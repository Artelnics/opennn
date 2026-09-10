// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/forward_propagation.h"
#include "opennn/network/back_propagation.h"

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
