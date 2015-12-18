/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   G O L D E N   S E C T I O N   O R D E R   C L A S S   H E A D E R                                          */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __GOLDENSECTIONORDER_H__
#define __GOLDENSECTIONORDER_H__

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
/// This concrete class represents a golden section algorithm for the order selection of a neural network.
///

class GoldenSectionOrder : public OrderSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit GoldenSectionOrder(void);

    // TRAINING STRATEGY CONSTRUCTOR

    explicit GoldenSectionOrder(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit GoldenSectionOrder(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit GoldenSectionOrder(const std::string&);

    // DESTRUCTOR

    virtual ~GoldenSectionOrder(void);


    // STRUCTURES

    ///
    /// This structure contains the training results for the golden section order method.
    ///

    struct GoldenSectionOrderResults : public OrderSelectionAlgorithm::OrderSelectionResults
    {
        /// Default constructor.

        explicit GoldenSectionOrderResults(void) : OrderSelectionAlgorithm::OrderSelectionResults()
        {
        }

        /// Destructor.

        virtual ~GoldenSectionOrderResults(void)
        {
        }

    };

    // Order selection methods

    GoldenSectionOrderResults* perform_order_selection(void);

    // Serialization methods

    tinyxml2::XMLDocument* to_XML(void) const;

    void from_XML(const tinyxml2::XMLDocument&);

    void save(const std::string&) const;
    void load(const std::string&);


};

}

#endif
