// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/training/training_context.h"
#include "opennn/training/loss.h"
#include "opennn/network/network.h"
#include "opennn/network/training_arena_plan.h"

namespace opennn
{

TrainingContext::TrainingContext(const Index batch_size, Loss& loss,
                                 const bool inputs_pre_scaled,
                                 TrainingContext* share_memory_with,
                                 const bool joint_gradient_arena)
{
    Network* const network = loss.get_network();
    shared_memory = share_memory_with != nullptr;

    throw_if(!network, "TrainingContext: the loss has no neural network.");
    throw_if(share_memory_with == this, "TrainingContext: a context cannot share with itself.");

    const bool use_joint_gradient_arena =
        joint_gradient_arena
        || (share_memory_with
            && share_memory_with->backward.has_joint_gradient_arena());

    TrainingArenaPlan arena_plan(batch_size, loss, use_joint_gradient_arena);

    forward.set(batch_size, network,
                share_memory_with ? &share_memory_with->forward.arena : nullptr,
                inputs_pre_scaled, arena_plan);

    throw_if(share_memory_with && forward.arena.owns_memory(),
             "TrainingContext: {} samples did not fit in the arena of the {}-sample "
             "context offered and allocated one of their own, which the steady-state "
             "allocation guard forbids.",
             batch_size, share_memory_with->forward.batch_size);

    backward.set(batch_size, loss, &forward.arena,
        share_memory_with && !use_joint_gradient_arena
            ? &share_memory_with->backward.gradient
            : nullptr,
        arena_plan);
}

}
