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

    GrowingNeurons(const TrainingStrategy* = nullptr);

    Index get_neurons_increment() const;

    Index get_maximum_validation_failures() const;

    void set_default();

    void set_neurons_increment(const Index);

    void set_maximum_validation_failures(const Index);

    NeuronsSelectionResults perform_neurons_selection() override;

    Tensor<string, 2> to_string_matrix() const override;
    
    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

   Index neurons_increment = 0;

   Index maximum_validation_failures = 0;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
