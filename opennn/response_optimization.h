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

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "neural_network.h"

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

    NeuralNetwork* neural_network_pointer = nullptr;

    DataSet* data_set_pointer = nullptr;

    Tensor<Condition, 1> inputs_conditions;
    Tensor<Condition, 1> outputs_conditions;
    Tensor<Condition, 1> conditions;

    Tensor<type, 1> inputs_minimums;
    Tensor<type, 1> inputs_maximums;

    Tensor<type, 1> outputs_minimums;
    Tensor<type, 1> outputs_maximums;

    Index evaluations_number = 1000;

    type calculate_random_uniform(const type&, const type&) const;

};


///
/// This structure returns the results obtained in the optimization, e.g. optimum inputs number, etc.
///

struct ResponseOptimizationResults
{
    /// Default constructor.

    explicit ResponseOptimizationResults(NeuralNetwork* new_neural_network_pointer)
    {
        neural_network_pointer = new_neural_network_pointer;
    }

    virtual ~ResponseOptimizationResults(){}

    DataSet* data_set = nullptr;

    NeuralNetwork* neural_network_pointer = nullptr;

    Tensor<type, 1> optimal_variables;

    void print() const
    {
        const Index inputs_number = neural_network_pointer->get_inputs_number();
        const Index outputs_number = neural_network_pointer->get_outputs_number();

        const Tensor<string, 1> inputs_names = neural_network_pointer->get_inputs_names();
        const Tensor<string, 1> outputs_names = neural_network_pointer->get_outputs_names();

        cout << "\nResponse optimization results: " << endl;
        if(optimal_variables.size() == 0)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ResponseOptimization class.\n"
                   << "void ResponseOptimizationResults::print() method.\n"
                   << "Optimal variables vector is empty.\n";

            throw invalid_argument(buffer.str());
        }

        for(Index i = 0; i < inputs_number; i++)
        {
            cout << inputs_names[i] << ": " << optimal_variables[i] << endl;
        }

        for(Index i = 0; i < outputs_number; i++)
        {
            cout << outputs_names[i] << ": " << optimal_variables[inputs_number+i] << endl;
        }
    }
};


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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

