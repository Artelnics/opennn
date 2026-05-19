//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "neuron_selection.h"

namespace opennn
{

struct GrowingNeuronsResults;

/// @brief Selects the optimal hidden neuron count by incrementally growing the number of neurons.
class GrowingNeurons final : public NeuronSelection
{

public:

    /// @brief Constructs the algorithm bound to an optional training strategy.
    GrowingNeurons(TrainingStrategy* = nullptr);

    /// @brief Restores default search bounds and stopping criteria.
    void set_default();

    /// @brief Sets the step size used when growing the number of neurons between trials.
    void set_neurons_increment(const Index);

    /// @brief Runs the neuron growing procedure until the stopping criterion is met.
    /// @return Selection results including the optimal neuron count and error history.
    NeuronsSelectionResults perform_neurons_selection() override;

    /// @brief Loads algorithm configuration from a JSON document.
    void from_JSON(const JsonDocument&) override;

    /// @brief Writes algorithm configuration to a JSON writer.
    void to_JSON(JsonWriter&) const override;

private:

   Index neurons_increment = 0;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
