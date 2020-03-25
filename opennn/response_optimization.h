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

#include "vector.h"
#include "matrix.h"
#include "neural_network.h"



#include "tinyxml2.h"

namespace OpenNN
{

/// This class is used to optimize model response identify the combinations of variable settings jointly optimize a set of responses.

///
/// This tool is adequate when you need to know the behaviour of a multiple variables on a response
/// and satisfy the requirements of the architecture.

class ResponseOptimization
{

public:

   // DEFAULT CONSTRUCTOR

    explicit ResponseOptimization();

    explicit ResponseOptimization(NeuralNetwork*);

    void set_evaluations_number(const size_t&);

   

   virtual ~ResponseOptimization();

    ///Enumeration of available conditions for response optimization.

   enum Condition{Between, EqualTo, LessEqualTo, GreaterEqualTo, Minimum, Maximum};

   ///
   /// This structure returns the results obtained in the optimization, e.g. optimum inputs number, etc.
   ///

   struct Results
   {
       /// Default constructor.

       explicit Results(NeuralNetwork* new_neural_network_pointer)
       {
           neural_network_pointer = new_neural_network_pointer;
       }

       virtual ~Results(){}

       NeuralNetwork* neural_network_pointer = nullptr;

       Vector<double> optimal_variables;

       double optimum_objective;

       void print() const
       {
           const size_t inputs_number = neural_network_pointer->get_inputs_number();
           const size_t outputs_number = neural_network_pointer->get_outputs_number();

           const Vector<string> inputs_names = neural_network_pointer->get_inputs_names();
           const Vector<string> outputs_names = neural_network_pointer->get_outputs_names();

           for(size_t i = 0; i < inputs_number; i++)
           {
               cout << inputs_names[i] << ": " << optimal_variables[i] << endl;
           }

           for(size_t i = 0; i < outputs_number; i++)
           {
               cout << outputs_names[i] << " " << optimal_variables[inputs_number+i] << endl;
           }

           cout << "Objective: " << optimum_objective << endl;
       }
   };


   // Get methods

   Vector<Condition> get_inputs_conditions();
   Vector<Condition> get_outputs_conditions();

   Vector<double> get_inputs_minimums();
   Vector<double> get_inputs_maximums();
   Vector<double> get_outputs_minimums();
   Vector<double> get_outputs_maximums();

   // Set methods

   void set_input_condition(const string&, const Condition&, const Vector<double>& = Vector<double>());
   void set_output_condition(const string&, const Condition&, const Vector<double>& = Vector<double>());

   void set_input_condition(const size_t&, const Condition&, const Vector<double>& = Vector<double>());
   void set_output_condition(const size_t&, const Condition&, const Vector<double>& = Vector<double>());

   void set_inputs_outputs_conditions(const Vector<string>&, const Vector<string>&, const Vector<double>& = Vector<double>());

   Vector<Condition> get_conditions(const Vector<string>&) const;
   Vector<Vector<double>> get_values_conditions(const Vector<Condition>&, const Vector<double>&) const;

   Tensor<double> calculate_inputs() const;

   Matrix<double> calculate_envelope(const Tensor<double>&, const Tensor<double>&) const;

   Results* perform_optimization() const;

private:

    NeuralNetwork* neural_network_pointer = nullptr;

    Vector<Condition> inputs_conditions;
    Vector<Condition> outputs_conditions;

    Vector<double> inputs_minimums;
    Vector<double> inputs_maximums;

    Vector<double> outputs_minimums;
    Vector<double> outputs_maximums;

    size_t evaluations_number = 1000;

    double calculate_random_uniform(const double&, const double&) const;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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

