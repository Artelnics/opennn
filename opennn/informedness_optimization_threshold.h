/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N F O R M E D N E S S   O P T I M I Z A T I O N   T H R E S H O L D   C L A S S   H E A D E R            */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __INFORMEDNESSOPTIMIZATIONTHRESHOLD_H__
#define __INFORMEDNESSOPTIMIZATIONTHRESHOLD_H__

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

#include "threshold_selection_algorithm.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a incremental algorithm for the order selection of a neural network.
///

class InformednessOptimizationThreshold : public ThresholdSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit InformednessOptimizationThreshold(void);

    // TRAINING STRATEGY CONSTRUCTOR

    explicit InformednessOptimizationThreshold(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit InformednessOptimizationThreshold(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit InformednessOptimizationThreshold(const std::string&);

    // DESTRUCTOR

    virtual ~InformednessOptimizationThreshold(void);


    // STRUCTURES

    ///
    /// This structure contains the training results for the incremental order method.
    ///

    struct InformednessOptimizationThresholdResults : public ThresholdSelectionAlgorithm::ThresholdSelectionResults
    {
        /// Default constructor.

        explicit InformednessOptimizationThresholdResults(void) : ThresholdSelectionAlgorithm::ThresholdSelectionResults()
        {
        }

        /// Destructor.

        virtual ~InformednessOptimizationThresholdResults(void)
        {
        }


    };

    // METHODS

    // Get methods

    const double& get_step(void) const;

    const size_t& get_maximum_selection_failures(void) const;

    // Set methods

    void set_default(void);

    void set_step(const double&);

    void set_maximum_selection_failures(const size_t&);

    // Order selection methods

    InformednessOptimizationThresholdResults* perform_threshold_selection(void);

    // Serialization methods

    Matrix<std::string> to_string_matrix(void) const;

    tinyxml2::XMLDocument* to_XML(void) const;

    void from_XML(const tinyxml2::XMLDocument&);

    void save(const std::string&) const;
    void load(const std::string&);

private:

    /// Difference in the thresholds between two consecutive iterations.

    double step;

   // STOPPING CRITERIA

   /// Maximum number of iterations at which the selection performance increases.

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
