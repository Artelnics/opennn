//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GROWINGNEURONS_H
#define GROWINGNEURONS_H

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "training_strategy.h"
#include "neurons_selection.h"
#include "config.h"

namespace OpenNN
{

/// This concrete class represents an growing neurons algorithm for the NeuronsSelection as part of the ModelSelection[1] class.

/// [1] Neural Designer "Model Selection Algorithms in Predictive Analytics."
/// \ref https://www.neuraldesigner.com/blog/model-selection

class GrowingNeurons : public NeuronsSelection
{

public:

    // Constructors

    explicit GrowingNeurons();

    explicit GrowingNeurons(TrainingStrategy*);

    // Destructor

    virtual ~GrowingNeurons();

    /// This structure contains the training results for the growing neurons method.

    struct GrowingNeuronsResults : public NeuronsSelection::Results
    {
        /// Default constructor.

        explicit GrowingNeuronsResults() : NeuronsSelection::Results()
        {
        }

        /// Destructor.

        virtual ~GrowingNeuronsResults()
        {
        }
    };

    // Get methods

    const Index& get_step() const;

    const Index& get_maximum_selection_failures() const;

    // Set methods

    void set_default();

    void set_step(const Index&);

    void set_maximum_selection_failures(const Index&);

    // Order selection methods

    GrowingNeuronsResults* perform_neurons_selection();

    // Serialization methods

    Tensor<string, 2> to_string_matrix() const;
    
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;    

    void save(const string&) const;
    void load(const string&);

private:

   /// Number of neurons added at each iteration.

   Index step;

   /// Maximum number of epochs at which the selection error increases.

   Index maximum_selection_failures;

};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
