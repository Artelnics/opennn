//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GROWINGNEURONS_H
#define GROWINGNEURONS_H

#include "training_strategy.h"
#include "neurons_selection.h"

namespace opennn
{

struct GrowingNeuronsResults;

class GrowingNeurons : public NeuronsSelection
{

public:

    // Constructors

    explicit GrowingNeurons();

    explicit GrowingNeurons(TrainingStrategy*);

    // Get

    const Index& get_step() const;

    const Index& get_maximum_selection_failures() const;

    // Set

    void set_default();

    void set_neurons_increment(const Index&);

    void set_maximum_selection_failures(const Index&);

    // Neurons selection

    NeuronsSelectionResults perform_neurons_selection() final;

    // Serialization

    Tensor<string, 2> to_string_matrix() const;
    
    void from_XML(const tinyxml2::XMLDocument&);

    void to_XML(tinyxml2::XMLPrinter&) const;    

    void save(const string&) const;
    void load(const string&);

private:

   Index neurons_increment;

   Index maximum_selection_failures;

};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
