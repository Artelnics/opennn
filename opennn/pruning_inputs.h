/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R U N I N G   I N P U T S   C L A S S   H E A D E R                                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __PRUNINGINPUTS_H__
#define __PRUNINGINPUTS_H__

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

#include "training_strategy.h"

#include "inputs_selection_algorithm.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a pruning algorithm for the inputs selection of a neural network.
///

class PruningInputs : public InputsSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit PruningInputs();

    // TRAINING STRATEGY CONSTRUCTOR

    explicit PruningInputs(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit PruningInputs(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit PruningInputs(const string&);

    // DESTRUCTOR

    virtual ~PruningInputs();


    // STRUCTURES

    ///
    /// This structure contains the training results for the pruning inputs method.
    ///

    struct PruningInputsResults : public InputsSelectionAlgorithm::InputsSelectionResults
    {
        /// Default constructor.

        explicit PruningInputsResults() : InputsSelectionAlgorithm::InputsSelectionResults()
        {
        }

        /// Destructor.

        virtual ~PruningInputsResults()
        {
        }

    };

    // METHODS

    // Get methods

    const size_t& get_minimum_inputs_number() const;

    const size_t& get_maximum_inputs_number() const;

    const size_t& get_maximum_selection_failures() const;

    // Set methods

    void set_default();

    void set_minimum_inputs_number(const size_t&);

    void set_maximum_inputs_number(const size_t&);

    void set_maximum_selection_failures(const size_t&);

    // Order selection methods

    PruningInputsResults* perform_inputs_selection();

    // Serialization methods

    Matrix<string> to_string_matrix() const;

    tinyxml2::XMLDocument* to_XML() const;
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;
    //void read_XML(   );

    void save(const string&) const;
    void load(const string&);

private:

    // STOPPING CRITERIA

    /// Minimum number of inputs in the neural network.

    size_t minimum_inputs_number;

    /// Maximum number of inputs in the neural network.

    size_t maximum_inputs_number;

    /// Maximum number of iterations at which the selection error increases.

    size_t maximum_selection_failures;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
