/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
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

    const size_t& get_maximum_generalization_failures(void) const;

    // Set methods

    void set_default(void);

    void set_maximum_generalization_failures(const size_t&);

    // Order selection methods

    GrowingInputsResults* perform_inputs_selection(void);

    // Serialization methods

    tinyxml2::XMLDocument* to_XML(void) const;

    void from_XML(const tinyxml2::XMLDocument&);

    void save(const std::string&) const;
    void load(const std::string&);

private:

   // STOPPING CRITERIA

   /// Maximum number of iterations at which the generalization performance increases.

   size_t maximum_generalization_failures;
};

}

#endif
