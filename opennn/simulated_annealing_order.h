/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   S I M U L A T E D   A N N E A L I N G   O R D E R   C L A S S   H E A D E R                                */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
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

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

///
/// This concrete class represents a simulated annealing algorithm for the order selection of a neural network.
///

class SimulatedAnnealingOrder : public OrderSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit SimulatedAnnealingOrder(void);

    // TRAINING STRATEGY CONSTRUCTOR

    explicit SimulatedAnnealingOrder(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit SimulatedAnnealingOrder(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit SimulatedAnnealingOrder(const std::string&);

    // DESTRUCTOR

    virtual ~SimulatedAnnealingOrder(void);


    // STRUCTURES

    ///
    /// This structure contains the training results for the simulated annealing order method.
    ///

    struct SimulatedAnnealingOrderResults : public OrderSelectionAlgorithm::OrderSelectionResults
    {
        /// Default constructor.

        explicit SimulatedAnnealingOrderResults(void) : OrderSelectionAlgorithm::OrderSelectionResults()
        {
        }

        /// Destructor.

        virtual ~SimulatedAnnealingOrderResults(void)
        {
        }

    };

    // METHODS

    // Get methods

    const double& get_cooling_rate(void) const;

    const size_t& get_maximum_generalization_failures(void) const;
    const double& get_minimum_temperature(void) const;

    // Set methods

    void set_default(void);

    void set_cooling_rate(const double&);

    void set_maximum_generalization_failures(const size_t&);
    void set_minimum_temperature(const double&);

    // Order selection methods

    SimulatedAnnealingOrderResults* perform_order_selection(void);

    // Serialization methods

    tinyxml2::XMLDocument* to_XML(void) const;

    void from_XML(const tinyxml2::XMLDocument&);

    void save(const std::string&) const;
    void load(const std::string&);

private:

   // MEMBERS

   /// Temperature reduction factor for the simulated annealing.

   double cooling_rate;

   // STOPPING CRITERIA

   /// Maximum number of iterations at which the generalization performance increases.

   size_t maximum_generalization_failures;

   /// Minimum temperature reached in the simulated annealing algorithm.

   double minimum_temperature;

};

}

#endif
