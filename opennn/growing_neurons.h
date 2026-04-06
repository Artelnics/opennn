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

class GrowingNeurons final : public NeuronSelection
{

public:

    GrowingNeurons(TrainingStrategy* = nullptr);

    void set_default();

    void set_neurons_increment(const Index);

    NeuronsSelectionResults perform_neurons_selection() override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

   Index neurons_increment = 0;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
