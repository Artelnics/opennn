/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T H R E S H O L D   S E L E C T I O N   A L G O R I T H M   C L A S S   H E A D E R                        */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
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

#include "tinyxml2.h"

namespace OpenNN
{

/// This abstract class represents the concept of order selection algorithm for a neural network.
/// Any derived class must implement the perform_order_selection() method.

class ThresholdSelectionAlgorithm
{
public:

    // DEFAULT CONSTRUCTOR

    explicit ThresholdSelectionAlgorithm();

    // TRAINING STRATEGY CONSTRUCTOR

    explicit ThresholdSelectionAlgorithm(TrainingStrategy*);

    // FILE CONSTRUCTOR

    explicit ThresholdSelectionAlgorithm(const string&);

    // XML CONSTRUCTOR

    explicit ThresholdSelectionAlgorithm(const tinyxml2::XMLDocument&);


    // DESTRUCTOR

    virtual ~ThresholdSelectionAlgorithm();

    // ENUMERATIONS

    /// Enumeration of all possibles condition of stop for the algorithms.

    enum StoppingCondition{PerfectConfusionMatrix, AlgorithmFinished};

    // STRUCTURES

    ///
    /// This structure contains the results from the order selection.
    ///

    struct ThresholdSelectionResults
    {
       explicit ThresholdSelectionResults()
       {

       }

       virtual ~ThresholdSelectionResults()
       {

       }

       string write_stopping_condition() const;

       string object_to_string() const;

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

    TrainingStrategy* get_training_strategy_pointer() const;

    bool has_training_strategy() const;

    const bool& get_reserve_binary_classification_tests_data() const;

    const bool& get_reserve_function_data() const;

    const bool& get_display() const;

    // Set methods

    void set_training_strategy_pointer(TrainingStrategy*);

    void set_default();

    void set_reserve_binary_classification_tests_data(const bool&);

    void set_reserve_function_data(const bool&);

    void set_display(const bool&);

    // Errors calculation methods

    Matrix<size_t> calculate_confusion(const double&) const;

    Vector<double> calculate_binary_classification_test(const Matrix<size_t>&) const;

    // threshold selection methods

    void check() const;

    /// Performs the threshold selection for a neural network.

    virtual ThresholdSelectionResults* perform_threshold_selection() = 0;

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
