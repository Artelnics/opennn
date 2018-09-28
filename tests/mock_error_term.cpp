/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M O C K   P E R F O R M A N C E   T E R M   C L A S S                                                      */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// System includes

#include <iostream>
#include <fstream>
#include <cmath>


// Unit testing includes

#include "mock_error_term.h"


using namespace OpenNN;


// DEFAULT CONSTRUCTOR

MockErrorTerm::MockErrorTerm() : ErrorTerm()
{
    set_default();
}


// GENERAL CONSTRUCTOR

MockErrorTerm::MockErrorTerm(NeuralNetwork* new_neural_network_pointer)
: ErrorTerm(new_neural_network_pointer)
{
    set_default();
}


MockErrorTerm::MockErrorTerm(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
: ErrorTerm(new_neural_network_pointer, new_data_set_pointer)
{
    set_default();
}


// DESTRUCTOR

/// Destructor.

MockErrorTerm::~MockErrorTerm() 
{
}


// METHODS

const MockErrorTerm::Expression& MockErrorTerm::get_expression() const
{
    return(expression);
}


void MockErrorTerm::set_expression(const MockErrorTerm::Expression& new_expression)
{
    expression = new_expression;
}


void MockErrorTerm::set_default()
{
    expression = SumSquaredParameters;
}

/*
Vector<double> MockErrorTerm::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target) const
{
    const Vector<double> output_gradient = (output-target)*2.0;

    return(output_gradient);
}


Matrix<double> MockErrorTerm::calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const
{
    const size_t outputs_number = neural_network_pointer->get_multilayer_perceptron_pointer()->get_outputs_number();

    Matrix<double> output_Hessian(outputs_number, outputs_number);
    output_Hessian.initialize_diagonal(2.0);

    return(output_Hessian);
}
*/

double MockErrorTerm::calculate_sum_squared_parameters() const
{
    const Vector<double> parameters = neural_network_pointer->arrange_parameters();

    return((parameters*parameters).calculate_sum());
}


Vector<double> MockErrorTerm::calculate_sum_squared_parameters_gradient() const
{
    const Vector<double> parameters = neural_network_pointer->arrange_parameters();

    return(parameters*2.0);
}


Matrix<double> MockErrorTerm::calculate_sum_squared_parameters_Hessian() const
{
    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Matrix<double> Hessian(parameters_number, parameters_number);

    Hessian.set_diagonal(2.0);

    return(Hessian);
}


Vector<double> MockErrorTerm::calculate_sum_squared_parameters_terms() const
{
    const Vector<double> parameters = neural_network_pointer->arrange_parameters();

    return(parameters);
}


Matrix<double> MockErrorTerm::calculate_sum_squared_parameters_terms_Jacobian() const
{
    const Vector<double> parameters = neural_network_pointer->arrange_parameters();
    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Matrix<double> Jacobian;

    Jacobian.set_identity(parameters_number);

    return(Jacobian);
}


double MockErrorTerm::calculate_sum_squared_parameters(const Vector<double>& parameters) const
{
    return((parameters*parameters).calculate_sum());
}


Vector<double> MockErrorTerm::calculate_sum_squared_parameters_gradient(const Vector<double>& parameters) const
{
    return(parameters*2.0);
}


Matrix<double> MockErrorTerm::calculate_sum_squared_parameters_Hessian(const Vector<double>&) const
{
    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    Matrix<double> Hessian(parameters_number, parameters_number, 0.0);
    Hessian.set_diagonal(2.0);

    return(Hessian);
}


double MockErrorTerm::calculate_output_integral_integrand(const double& x) const
{
   const Vector<double> input(1, x);

   const Vector<double> output = neural_network_pointer->calculate_outputs(input);

   return(output[0]*output[0]);   
}


double MockErrorTerm::calculate_output_integral() const
{
//   double trapezoid_integral = 
//   numerical_integration.calculate_trapezoid_integral(*this, 
//   &MockErrorTerm::calculate_evaluation_integrand,
//   0.0, 1.0, 101);

//   return(trapezoid_integral);

	return(0.0);
}


// @todo

Vector<double> MockErrorTerm::calculate_output_integral_integrand_gradient(const double& x) const
{
   const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

   const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

   const Vector<double> input(1, x);

   const Vector< Vector< Vector<double> > > first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(input);

   const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
   const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

   const Vector<double>& output = layers_activation[layers_number-1];

   const Vector<double> output_gradient(1, 2.0*output[0]);

   const Vector< Vector<double> > layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);

   const Vector<double> objective_gradient_integrand;// = calculate_point_objective_gradient(input, layers_activation, layers_delta);

   return(objective_gradient_integrand);
}


// @todo

Vector<double> MockErrorTerm::calculate_output_integral_gradient() const
{
//    Vector<double> trapezoid_integral = 
//    numerical_integration.calculate_trapezoid_integral(*this, 
//    &MockErrorTerm::calculate_gradient_integrand,
//    0.0, 1.0, 101);

//    return(trapezoid_integral);

   Vector<double> v;

   return(v);
}


// @todo

Matrix<double> MockErrorTerm::calculate_output_integral_integrand_Hessian(const double&) const
{
   Matrix<double> objective_Hessian_integrand;

   return(objective_Hessian_integrand);
}


// @todo

Matrix<double> MockErrorTerm::calculate_output_integral_Hessian() const
{
   Matrix<double> objective_Hessian;

   return(objective_Hessian);
}


// @todo

double MockErrorTerm::calculate_output_integral(const Vector<double>&) const
{
/*
   // Control sentence (if debug)

   #ifndef NDEBUG

   const size_t size = parameters.size();

   const size_t parameters_number = neural_network_pointer->count_parameters_number();

   if(size != parameters_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Error: MockErrorTerm class." << endl
             << "double calculate_output_integral(const Vector<double>&) const method." << endl
             << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ")." << endl;

      throw logic_error(buffer.str());
   }

   #endif
*/
   return(0.0);
}


Vector<double> MockErrorTerm::calculate_output_integral_gradient(const Vector<double>&) const
{
    Vector<double> gradient;

    return(gradient);
}


Matrix<double> MockErrorTerm::calculate_output_integral_Hessian(const Vector<double>&) const
{
    Matrix<double> Hessian;

    return(Hessian);
}


double MockErrorTerm::calculate_error() const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters());
       }
       break;

       case OutputIntegral:
       {
          return(calculate_output_integral());
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "double calculate_error() const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }
}

Vector<double> MockErrorTerm::calculate_gradient() const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters_gradient());
       }
       break;

       case OutputIntegral:
       {
          return(calculate_output_integral_gradient());
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "Vector<double> calculate_gradient() const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }

}


Matrix<double> MockErrorTerm::calculate_Hessian() const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters_Hessian());
       }
       break;

       case OutputIntegral:
       {
          return(calculate_output_integral_Hessian());
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "Matrix<double> calculate_Hessian() const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }
}


Vector<double> MockErrorTerm::calculate_terms() const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters_terms());
       }
       break;

       case OutputIntegral:
       {
            ostringstream buffer;

            buffer << "OpenNN Exception: MockLossIndex class\n"
                   << "Vector<double> calculate_terms() const method.\n"
                   << "The terms function is not defined for the output integral expression.\n";

            throw logic_error(buffer.str());
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "Vector<double> calculate_terms() const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }
}


Matrix<double> MockErrorTerm::calculate_terms_Jacobian() const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters_terms_Jacobian());
       }
       break;

       case OutputIntegral:
       {
            ostringstream buffer;

            buffer << "OpenNN Exception: MockLossIndex class\n"
                   << "Matrix<double> calculate_terms_Jacobian() const method.\n"
                   << "The terms function is not defined for the output integral expression.\n";

            throw logic_error(buffer.str());
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "Matrix<double> calculate_terms_Jacobian() const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }
}


double MockErrorTerm::calculate_error(const Vector<double>& parameters) const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters(parameters));
       }
       break;

       case OutputIntegral:
       {
          return(calculate_output_integral(parameters));
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "double calculate_error(const Vector<double>&) const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }
}


Vector<double> MockErrorTerm::calculate_gradient(const Vector<double>& parameters) const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters_gradient(parameters));
       }
       break;

       case OutputIntegral:
       {
          return(calculate_output_integral_gradient(parameters));
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "Vector<double> calculate_gradient(const Vector<double>) const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }
}


Matrix<double> MockErrorTerm::calculate_Hessian(const Vector<double>& parameters) const
{
    switch(expression)
    {
       case SumSquaredParameters:
       {
          return(calculate_sum_squared_parameters_Hessian(parameters));
       }
       break;

       case OutputIntegral:
       {
          return(calculate_output_integral_Hessian(parameters));
       }
       break;

       default:
       {
          ostringstream buffer;

          buffer << "OpenNN Exception: MockLossIndex class\n"
                 << "Vector<double> calculate_Hessian(const Vector<double>&) const method.\n"
                 << "Unknown expression.\n";

          throw logic_error(buffer.str());
       }
       break;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Roberto Lopez
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

