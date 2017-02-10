/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   G R O W I N G   I N P U T S   C L A S S   H E A D E R                                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __GROWINGINPUTS_H__
#define __GROWINGINPUTS_H__

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

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a growing algorithm for the inputs selection of a neural network.
///

class GrowingInputs : public InputsSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit GrowingInputs(void);

    // TRAINING STRATEGY CONSTRUCTOR

    explicit GrowingInputs(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit GrowingInputs(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit GrowingInputs(const std::string&);

    // DESTRUCTOR

    virtual ~GrowingInputs(void);

    // STRUCTURES

    ///
    /// This structure contains the training results for the growing inputs method.
    ///

    struct GrowingInputsResults : public InputsSelectionAlgorithm::InputsSelectionResults
    {
        /// Default constructor.

        explicit GrowingInputsResults(void) : InputsSelectionAlgorithm::InputsSelectionResults()
        {
        }

        /// Destructor.

        virtual ~GrowingInputsResults(void)
        {
        }

    };

    // METHODS

    // Get methods

    const size_t& get_maximum_inputs_number(void) const;

    const size_t& get_maximum_selection_failures(void) const;

    // Set methods

    void set_default(void);

    void set_maximum_inputs_number(const size_t&);

    void set_maximum_selection_failures(const size_t&);

    // Order selection methods

    GrowingInputsResults* perform_inputs_selection(void);

    // Serialization methods

    Matrix<std::string> to_string_matrix(void) const;

    tinyxml2::XMLDocument* to_XML(void) const;
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;
    // void read_XML(   );

    void save(const std::string&) const;
    void load(const std::string&);

private:

    // STOPPING CRITERIA

    /// Maximum number of inputs in the neural network.

    size_t maximum_inputs_number;

    /// Maximum number of iterations at which the selection loss increases.

    size_t maximum_selection_failures;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
