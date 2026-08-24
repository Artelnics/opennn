//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   C O N T E X T   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/training_context.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/neural_network/neural_network.h"

namespace opennn
{

TrainingContext::TrainingContext(const Index batch_size, Loss& loss,
                                 const bool inputs_pre_scaled,
                                 TrainingContext* share_memory_with)
{
    NeuralNetwork* const neural_network = loss.get_neural_network();

    throw_if(!neural_network, "TrainingContext: the loss has no neural network.");
    throw_if(share_memory_with == this, "TrainingContext: a context cannot share with itself.");

    const vector<MemoryPoolEntry> delta_lifetimes =
        BackPropagation::make_co_planned_lifetimes(loss, batch_size);

    forward.set(batch_size,
                neural_network,
                share_memory_with ? &share_memory_with->forward.arena : nullptr,
                ForwardPropagationMode::Training,
                InferenceShapePolicy{},
                inputs_pre_scaled,
                delta_lifetimes);

    throw_if(share_memory_with && forward.arena.owns_memory(),
             "TrainingContext: {} samples did not fit in the arena of the {}-sample "
             "context offered and allocated one of their own, which the steady-state "
             "allocation guard forbids.",
             batch_size, share_memory_with->forward.batch_size);

    backward.set(batch_size,
                 loss,
                 &forward.arena,
                 forward.co_planned_offsets,
                 share_memory_with ? &share_memory_with->backward.gradient : nullptr);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
