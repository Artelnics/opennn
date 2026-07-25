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

class GrowingNeurons final : public NeuronSelection
{

public:

    explicit GrowingNeurons(TrainingStrategy* = nullptr);

    void set_default();

    void set_neurons_increment(const Index);

    void set_warm_start(bool new_warm_start) { warm_start = new_warm_start; }
    bool get_warm_start() const noexcept { return warm_start; }

    NeuronsSelectionResult perform_neurons_selection() override;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    Index neurons_increment = 0;

    bool warm_start = true;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
