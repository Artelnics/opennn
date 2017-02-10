/********************************************************************************************************************/
/*                                                                                                                  */
/*   OpenNN: Open Neural Networks Library                                                                           */
/*   www.opennn.net                                                                                                 */
/*                                                                                                                  */
/*   K A P P A   C O E F F I C I E N T   O P T I M I Z A T I O N   T H R E S H O L D   C L A S S   H E A D E R      */
/*                                                                                                                  */
/*   Fernando Gomez                                                                                                 */
/*   Artelnics - Making intelligent use of data                                                                     */
/*   fernandogomez@artelnics.com                                                                                    */
/*                                                                                                                  */
/********************************************************************************************************************/

#ifndef __KAPPACOEFFICIENTOPTIMIZATIONTHRESHOLD_H__
#define __KAPPACOEFFICIENTOPTIMIZATIONTHRESHOLD_H__

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
/// This concrete class represents a kappa coefficient optimization for the threshold selection of a neural network.
///

class KappaCoefficientOptimizationThreshold : public ThresholdSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit KappaCoefficientOptimizationThreshold(void);

    // TRAINING STRATEGY CONSTRUCTOR

    explicit KappaCoefficientOptimizationThreshold(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit KappaCoefficientOptimizationThreshold(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit KappaCoefficientOptimizationThreshold(const std::string&);

    // DESTRUCTOR

    virtual ~KappaCoefficientOptimizationThreshold(void);


    // STRUCTURES

    ///
    /// This structure contains the selection results for the kappa coefficient optimization.
    ///

    struct KappaCoefficientOptimizationThresholdResults : public ThresholdSelectionAlgorithm::ThresholdSelectionResults
    {
        /// Default constructor.

        explicit KappaCoefficientOptimizationThresholdResults(void) : ThresholdSelectionAlgorithm::ThresholdSelectionResults()
        {
        }

        /// Destructor.

        virtual ~KappaCoefficientOptimizationThresholdResults(void)
        {
        }


    };

    // METHODS

    // Get methods

    const double& get_minimum_threshold(void) const;

    const double& get_maximum_threshold(void) const;

    const double& get_step(void) const;

    // Set methods

    void set_default(void);

    void set_minimum_threshold(const double&);

    void set_maximum_threshold(const double&);

    void set_step(const double&);

    // Order selection methods

    KappaCoefficientOptimizationThresholdResults* perform_threshold_selection(void);

    // Serialization methods

    Matrix<std::string> to_string_matrix(void) const;

    tinyxml2::XMLDocument* to_XML(void) const;
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;
    // void read_XML(   );

    void save(const std::string&) const;
    void load(const std::string&);

private:

    /// Minimum threshold to be evaluated.

    double minimum_threshold;

    /// Maximum threshold to be evaluated.

    double maximum_threshold;

    /// Difference in the thresholds between two consecutive iterations.

    double step;

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
