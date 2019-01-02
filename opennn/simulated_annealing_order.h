/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S I M U L A T E D   A N N E A L I N G   O R D E R   C L A S S   H E A D E R                                */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __SIMULATEDANNEALING_H__
#define __SIMULATEDANNEALING_H__

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

#include "order_selection_algorithm.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a simulated annealing algorithm for the order selection of a neural network.
///

class SimulatedAnnealingOrder : public OrderSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit SimulatedAnnealingOrder();

    // TRAINING STRATEGY CONSTRUCTOR

    explicit SimulatedAnnealingOrder(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit SimulatedAnnealingOrder(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit SimulatedAnnealingOrder(const string&);

    // DESTRUCTOR

    virtual ~SimulatedAnnealingOrder();


    // STRUCTURES

    ///
    /// This structure contains the training results for the simulated annealing order method.
    ///

    struct SimulatedAnnealingOrderResults : public OrderSelectionAlgorithm::OrderSelectionResults
    {
        /// Default constructor.

        explicit SimulatedAnnealingOrderResults() : OrderSelectionAlgorithm::OrderSelectionResults()
        {
        }

        /// Destructor.

        virtual ~SimulatedAnnealingOrderResults()
        {
        }

    };

    // METHODS

    // Get methods

    const double& get_cooling_rate() const;

    const double& get_minimum_temperature() const;

    // Set methods

    void set_default();

    void set_cooling_rate(const double&);

    void set_minimum_temperature(const double&);

    // Order selection methods

    size_t get_optimal_selection_error_index() const;

    SimulatedAnnealingOrderResults* perform_order_selection();

    // Serialization methods

    Matrix<string> to_string_matrix() const;

    tinyxml2::XMLDocument* to_XML() const;
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;
    //void read_XML(   );

    void save(const string&) const;
    void load(const string&);

private:

   // MEMBERS

   /// Temperature reduction factor for the simulated annealing.

   double cooling_rate;

   // STOPPING CRITERIA

   /// Minimum temperature reached in the simulated annealing algorithm.

   double minimum_temperature;

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
