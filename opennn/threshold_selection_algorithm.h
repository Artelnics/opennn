/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T H R E S H O L D   S E L E C T I O N   A L G O R I T H M   C L A S S   H E A D E R                        */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __THRESHOLDELECTIONALGORITHM_H__
#define __THRESHOLDELECTIONALGORITHM_H__

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

#include "training_strategy.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This abstract class represents the concept of order selection algorithm for a neural network.
/// Any derived class must implement the perform_order_selection(void) method.

class ThresholdSelectionAlgorithm
{
public:

    // DEFAULT CONSTRUCTOR

    explicit ThresholdSelectionAlgorithm(void);

    // TRAINING STRATEGY CONSTRUCTOR
  /// ownership not passed
    explicit ThresholdSelectionAlgorithm(TrainingStrategy*);

    // FILE CONSTRUCTOR

    explicit ThresholdSelectionAlgorithm(const std::string&);

    // XML CONSTRUCTOR

    explicit ThresholdSelectionAlgorithm(const tinyxml2::XMLDocument&);


    // DESTRUCTOR

    virtual ~ThresholdSelectionAlgorithm(void);

    // ENUMERATIONS

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{PerfectConfusionMatrix, AlgorithmFinished};

    // STRUCTURES

    ///
    /// This structure contains the results from the order selection.
    ///

    struct ThresholdSelectionResults
    {
       explicit ThresholdSelectionResults(void)
       {

       }

       virtual ~ThresholdSelectionResults(void)
       {

       }

       std::string write_stopping_condition(void) const;

       std::string to_string(void) const;

       /// Threshold of the different neural networks.

       Vector<double> threshold_data;

       /// Parameters of the different neural networks.

       Vector< Vector<double> > binary_classification_test_data;

       /// Value to optimize in the algorithm.

       Vector<double> function_data;

       /// Value of optimum threshold.

       double final_threshold;

       /// Value of the value to optimize with the optimum threshold.

       double final_function_value;

       /// Number of iterations to perform the threshold selection.

       size_t iterations_number;

       /// Stopping condition of the algorithm.

       StoppingCondition stopping_condition;
    };

    // METHODS

    // Get methods
  /// ownership not passed
    TrainingStrategy* get_training_strategy_pointer(void) const;

    bool has_training_strategy(void) const;

    const bool& get_reserve_binary_classification_tests_data(void) const;

    const bool& get_reserve_function_data(void) const;

    const bool& get_display(void) const;

    // Set methods
  /// ownership not passed

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default(void);

    void set_reserve_binary_classification_tests_data(const bool&);

    void set_reserve_function_data(const bool&);

    void set_display(const bool&);

    // Errors calculation methods

    Matrix<size_t> calculate_confusion(const double&) const;

    Vector<double> calculate_binary_classification_test(const Matrix<size_t>&) const;

    // threshold selection methods

    void check(void) const;

    /// Performs the threshold selection for a neural network.
  /// ownership passed - use delete to destroy

    virtual ThresholdSelectionResults* perform_threshold_selection(void) = 0;

protected:

    // MEMBERS

    /// Pointer to a training strategy object.

    TrainingStrategy* training_strategy_pointer;

    // Threshold selection results

    /// True if the values of all binary classification tests are to be reserved.

    bool reserve_binary_classification_tests_data;

    /// True if the function values to be optimized are to be reserved.

    bool reserve_function_data;

    /// Display messages to screen.

    bool display;
};
}

#endif // __THRESHOLDELECTIONALGORITHM_H__

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
