// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/training/training_context.h"
#include "opennn/training/loss.h"
#include "opennn/network/network.h"

namespace opennn
{

TrainingContext::TrainingContext(const Index batch_size, Loss& loss,
                                 const bool inputs_pre_scaled,
                                 TrainingContext* share_memory_with,
                                 const bool joint_gradient_arena)
{
    Network* const network = loss.get_network();

    throw_if(!network, "TrainingContext: the loss has no neural network.");
    throw_if(share_memory_with == this, "TrainingContext: a context cannot share with itself.");

    const bool use_joint_gradient_arena =
        joint_gradient_arena
        || (share_memory_with
            && share_memory_with->backward.has_joint_gradient_arena());

    const vector<MemoryPoolEntry> delta_lifetimes =
        BackPropagation::make_co_planned_lifetimes(loss, batch_size);

    const vector<MemoryPoolEntry> gradient_lifetimes =
        use_joint_gradient_arena
        ? BackPropagation::make_gradient_co_planned_lifetimes(loss)
        : vector<MemoryPoolEntry>{};

    vector<MemoryPoolEntry> joint_lifetimes;
    joint_lifetimes.reserve(delta_lifetimes.size()
                            + gradient_lifetimes.size());
    joint_lifetimes.insert(joint_lifetimes.end(),
                           delta_lifetimes.begin(),
                           delta_lifetimes.end());
    joint_lifetimes.insert(joint_lifetimes.end(),
                           gradient_lifetimes.begin(),
                           gradient_lifetimes.end());

    forward.set(batch_size,
                network,
                share_memory_with ? &share_memory_with->forward.arena : nullptr,
                ForwardPropagationMode::Training,
                InferenceShapePolicy{},
                inputs_pre_scaled,
                joint_lifetimes,
                use_joint_gradient_arena);

    throw_if(share_memory_with && forward.arena.owns_memory(),
             "TrainingContext: {} samples did not fit in the arena of the {}-sample "
             "context offered and allocated one of their own, which the steady-state "
             "allocation guard forbids.",
             batch_size, share_memory_with->forward.batch_size);

    const span<const Index> joint_offsets(forward.co_planned_offsets);
    const span<const Index> delta_offsets =
        joint_offsets.first(delta_lifetimes.size());
    const span<const Index> gradient_offsets =
        joint_offsets.subspan(delta_lifetimes.size());

    backward.set(
        batch_size,
        loss,
        &forward.arena,
        delta_offsets,
        share_memory_with && !use_joint_gradient_arena
            ? &share_memory_with->backward.gradient
            : nullptr,
        gradient_offsets);
}

}
