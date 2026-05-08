//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file growing_neurons.h
 * @brief Declares the GrowingNeurons hidden-layer-size selection method.
 */

#pragma once

#include "neuron_selection.h"

namespace opennn
{

struct GrowingNeuronsResults;

/**
 * @class GrowingNeurons
 * @brief Forward-selection of hidden-layer size.
 *
 * Starts from the configured minimum hidden-layer size and iteratively
 * grows the layer by a fixed increment as long as the validation error
 * keeps improving, up to the configured maximum.
 */
class GrowingNeurons final : public NeuronSelection
{

public:

    /**
     * @brief Constructs the selector.
     * @param training_strategy Training strategy used to evaluate candidate sizes.
     */
    GrowingNeurons(TrainingStrategy* training_strategy = nullptr);

    /** @brief Resets all hyperparameters to their default values. */
    void set_default();

    /**
     * @brief Sets the increment applied to the hidden-layer size at each step.
     *
     * Receives the number of neurons added at each iteration.
     */
    void set_neurons_increment(const Index);

    /**
     * @brief Runs the forward-selection algorithm.
     * @return Best-of-run hidden-layer size and supporting statistics.
     */
    NeuronsSelectionResults perform_neurons_selection() override;

    /**
     * @brief Loads selector hyperparameters from a parsed JSON document.
     */
    void from_JSON(const JsonDocument&) override;

    /**
     * @brief Writes selector hyperparameters to a streaming JSON writer.
     */
    void to_JSON(JsonWriter&) const override;

private:

   /** @brief Number of neurons added at each forward-selection step. */
   Index neurons_increment = 0;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
