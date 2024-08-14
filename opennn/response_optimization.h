//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R   
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef RESPONSEOPTIMIZATION_H
#define RESPONSEOPTIMIZATION_H

// System includes

#include <string>

// OpenNN includes

#include "config.h"
#include "neural_network.h"
#include "data_set.h"

namespace opennn
{

struct ResponseOptimizationResults;

/// This class optimizes model response identify the combinations of variable settings jointly optimize a set of responses.

///
/// This tool is adequate when you need to know the behaviour of a multiple variables on a response
/// and satisfy the requirements of the architecture.

class ResponseOptimization
{

public:

   // DEFAULT CONSTRUCTOR

    explicit ResponseOptimization();

    explicit ResponseOptimization(NeuralNetwork*);

    explicit ResponseOptimization(NeuralNetwork*,DataSet*);


    ///Enumeration of the available conditions for response optimization.

   enum class Condition{None, Between, EqualTo, LessEqualTo, GreaterEqualTo, Minimum, Maximum};

   // Get methods

   Tensor<Condition, 1> get_inputs_conditions() const;
   Tensor<Condition, 1> get_outputs_conditions() const;
   Index get_evaluations_number() const;

   Tensor<type, 1> get_inputs_minimums() const;
   Tensor<type, 1> get_inputs_maximums() const;
   Tensor<type, 1> get_outputs_minimums() const;
   Tensor<type, 1> get_outputs_maximums() const;

   // Set methods

   void set(NeuralNetwork*);

   void set_evaluations_number(const Index&);

   void set_input_condition(const string&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());
   void set_output_condition(const string&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());

   void set_input_condition(const Index&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());
   void set_output_condition(const Index&, const Condition&, const Tensor<type, 1>& = Tensor<type, 1>());

   void set_inputs_outputs_conditions(const Tensor<string, 1>&, const Tensor<string, 1>&, const Tensor<type, 1>& = Tensor<type, 1>());

   Tensor<Condition, 1> get_conditions(const Tensor<string, 1>&) const;
   Tensor<Tensor<type, 1>, 1> get_values_conditions(const Tensor<Condition, 1>&, const Tensor<type, 1>&) const;

   Tensor<type, 2> calculate_inputs() const;

   Tensor<type, 2> calculate_envelope(const Tensor<type, 2>&, const Tensor<type, 2>&) const;

   ResponseOptimizationResults* perform_optimization() const;

private:

    NeuralNetwork* neural_network = nullptr;

    DataSet* data_set = nullptr;

    Tensor<Condition, 1> inputs_conditions;
    Tensor<Condition, 1> outputs_conditions;
    Tensor<Condition, 1> conditions;

    Tensor<type, 1> inputs_minimums;
    Tensor<type, 1> inputs_maximums;

    Tensor<type, 1> outputs_minimums;
    Tensor<type, 1> outputs_maximums;

    Index evaluations_number = 1000;
};


///
/// This structure returns the results obtained in the optimization, e.g. optimum inputs number, etc.
///

struct ResponseOptimizationResults
{
    /// Default constructor.

    explicit ResponseOptimizationResults(NeuralNetwork* new_neural_network)
    {
        neural_network = new_neural_network;
    }

    virtual ~ResponseOptimizationResults(){}

    DataSet* data_set = nullptr;

    NeuralNetwork* neural_network = nullptr;

    Tensor<type, 1> optimal_variables;

    void print() const;
};


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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

