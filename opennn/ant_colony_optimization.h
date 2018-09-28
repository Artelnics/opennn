/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   A N T   C O L O N Y   O P T I M I Z A T I O N   C L A S S   H E A D E R                                    */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __ANTCOLONYOPTIMIZATION_H__
#define __ANTCOLONYOPTIMIZATION_H__

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
/// This concrete class represents a incremental algorithm for the order selection of a neural network.
///

class AntColonyOptimization : public OrderSelectionAlgorithm
{
public:
    // DEFAULT CONSTRUCTOR

    explicit AntColonyOptimization();

    // TRAINING STRATEGY CONSTRUCTOR
  /// ownership not passed
    explicit AntColonyOptimization(TrainingStrategy*);

    // XML CONSTRUCTOR

    explicit AntColonyOptimization(const tinyxml2::XMLDocument&);

    // FILE CONSTRUCTOR

    explicit AntColonyOptimization(const string&);

    // DESTRUCTOR

    virtual ~AntColonyOptimization();


    // STRUCTURES

    ///
    /// This structure contains the training results for the incremental order method.
    ///

    struct AntColonyOptimizationResults : public OrderSelectionAlgorithm::OrderSelectionResults
    {
        /// Default constructor.

        explicit AntColonyOptimizationResults() : OrderSelectionAlgorithm::OrderSelectionResults()
        {
        }

        /// Destructor.

        virtual ~AntColonyOptimizationResults()
        {
        }
    };

    // METHODS

    // Get methods

    const size_t& get_maximum_selection_failures() const;

    // Set methods

    void set_default();

    void set_maximum_selection_failures(const size_t&);

    // Model evaluation methods

    Vector<double> perform_minimum_model_evaluation(const Vector<size_t>&);
    Vector<double> perform_maximum_model_evaluation(const Vector<size_t>&);
    Vector<double> perform_mean_model_evaluation(const Vector<size_t>&);

    Vector<double> perform_model_evaluation(const Vector<size_t>&);

    // Order selection methods

    void chose_paths();
    void evaluate_ants();

    AntColonyOptimizationResults* perform_order_selection();

    // Serialization methods

    Matrix<string> to_string_matrix() const;

    tinyxml2::XMLDocument* to_XML() const;
    void from_XML(const tinyxml2::XMLDocument&);

    void write_XML(tinyxml2::XMLPrinter&) const;
    // void read_XML(   );


    void save(const string&) const;
    void load(const string&);

private:

   // MEMBERS

    /// Maximum number of hidden layers.

    size_t maximum_layers;

    /// Activation function of the hidden layers.

    Perceptron::ActivationFunction default_activation_function;

   /// Number of ants that will be evaluated in the algorithm.

   size_t ants_number;

   /// Pheromone trail for every arc of the path.

   Vector< Matrix<double> > pheromone_trail;

   /// Ratio of the evaporation of the pheromone trail in each iteration.

   double evaporation_rate;

   /// Parameter to scale the actualization of the pheromone trail of the best ants.

   double scaling_parameter;

   /// Architecture of that represents the path of each ant.

   Matrix<size_t> architectures;

   /// Losses of the models selected for each ant.

   Vector<double> model_loss;

   /// Architecture of all the neural networks trained.

   Vector< Vector<size_t> > architecture_history;

   // STOPPING CRITERIA

   /// Maximum number of iterations at which the selection loss increases.

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
