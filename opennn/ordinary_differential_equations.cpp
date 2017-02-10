/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O R D I N A R Y   D I F F E R E N T I A L   E Q U A T I O N S   C L A S S                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "ordinary_differential_equations.h"

namespace OpenNN
{

// GENERAL CONSTRUCTOR

/// Default constructor.
/// It constructs a default ordinary differential equations object. 

OrdinaryDifferentialEquations::OrdinaryDifferentialEquations(void) : MathematicalModel()
{                                            
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor.
/// It creates a ordinary differential equations mathematical modeel and loads its members from a TinyXML document.
/// @param ordinary_differential_equations_document XML document containing the member data.

OrdinaryDifferentialEquations::OrdinaryDifferentialEquations(const tinyxml2::XMLDocument& ordinary_differential_equations_document)
: MathematicalModel(ordinary_differential_equations_document)
{                
   set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/// It creates an ordinary differential equations mathematical model and loads its members from a XML file. 
/// @param file_name Name of XML file. 

OrdinaryDifferentialEquations::OrdinaryDifferentialEquations(const std::string& file_name)
: MathematicalModel(file_name)
{                
   set_default();
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates an ordinary differential equations mathematical model object and copies its members from another object. 
/// @param other_ordinary_differential_equations Ordinary differential equations object to be copied. 

OrdinaryDifferentialEquations::OrdinaryDifferentialEquations(const OrdinaryDifferentialEquations& other_ordinary_differential_equations) 
 : MathematicalModel()
{
   set(other_ordinary_differential_equations);
}


// DESTRUCTOR

/// Destructor. 
/// It does not delete any object.  

OrdinaryDifferentialEquations::~OrdinaryDifferentialEquations(void)
{ 
}


// ASSIGNMENT OPERATOR

// OrdinaryDifferentialEquations& operator = (const OrdinaryDifferentialEquations&) method

/// Assignment operator. 
/// It assigns to this object the members of an existing ordinary differential equations object.
/// @param other_ordinary_differential_equations Ordinary differential equations object to be assigned.

OrdinaryDifferentialEquations& OrdinaryDifferentialEquations::operator = (const OrdinaryDifferentialEquations& other_ordinary_differential_equations)
{
   if(this != &other_ordinary_differential_equations) 
   {
      initial_independent_variable = other_ordinary_differential_equations.initial_independent_variable;
      final_independent_variable = other_ordinary_differential_equations.final_independent_variable;
      initial_dependent_variables = other_ordinary_differential_equations.initial_dependent_variables;
   }

   return(*this);
}


// EQUAL TO OPERATOR

// bool operator == (const OrdinaryDifferentialEquations&) const method

/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @param other_ordinary_differential_equations Object to be compared with.

bool OrdinaryDifferentialEquations::operator == (const OrdinaryDifferentialEquations& other_ordinary_differential_equations) const
{
   if(initial_independent_variable == other_ordinary_differential_equations.initial_independent_variable
   && final_independent_variable == other_ordinary_differential_equations.final_independent_variable
   && initial_dependent_variables == other_ordinary_differential_equations.initial_dependent_variables)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


// METHODS

// const double& get_initial_independent_variable(void) const method

/// Returns the initial value of the independent variable. 

const double& OrdinaryDifferentialEquations::get_initial_independent_variable(void) const
{
   return(initial_independent_variable);
}


// const double& get_final_independent_variable(void) const method

/// Returns the final value of the independent variable. 

const double& OrdinaryDifferentialEquations::get_final_independent_variable(void) const
{
   return(final_independent_variable);
}


// const Vector<double>& get_initial_dependent_variables(void) const method

/// Returns the initial values of the independent variables. 

const Vector<double>& OrdinaryDifferentialEquations::get_initial_dependent_variables(void) const
{
   return(initial_dependent_variables);
}


// const double& get_initial_dependent_variable(const size_t&) const method

/// Returns the initial value of a single independent variable.
/// @param i Index of dependent variable. 

const double& OrdinaryDifferentialEquations::get_initial_dependent_variable(const size_t& i) const
{
   return(initial_dependent_variables[i]);
}


// const SolutionMethod& get_solution_method(void) const method

/// Returns the numerical method to be used for integration of the ordinary differential equation. 

const OrdinaryDifferentialEquations::SolutionMethod& OrdinaryDifferentialEquations::get_solution_method(void) const
{
   return(solution_method);
}


// std::string write_solution_method(void) const method

/// Returns a string with the name of the numerical method to be used for integration of the ordinary differential equation. 

std::string OrdinaryDifferentialEquations::write_solution_method(void) const
{
   if(solution_method == RungeKutta)
   {
      return("RungeKutta");
   }
   else if(solution_method == RungeKuttaFehlberg)
   {
      return("RungeKuttaFehlberg");
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: OrdinaryDifferentialEquations class.\n"
             << "std::string write_solution_method(void) const method.\n"
             << "Unknown solution method.\n";
 
	  throw std::logic_error(buffer.str());
   }
}


// const size_t& get_points_number(void) const method

/// Returns the number of integration points in the Runge-Kutta method. 

const size_t& OrdinaryDifferentialEquations::get_points_number(void) const
{
   return(points_number);
}


// const double& get_tolerance(void) const method

/// Returns the tolerance in the Runge-Kutta-Fehlberg method. 

const double& OrdinaryDifferentialEquations::get_tolerance(void) const
{
   return(tolerance);
}


// const size_t& get_initial_size(void) const method

/// Returns the initial size to be reserved for the solutions in the Runge-Kutta-Fehlberg method. 

const size_t& OrdinaryDifferentialEquations::get_initial_size(void) const
{
   return(initial_size);
}


// const size_t& get_warning_size(void) const method

/// Returns a warning size for the solutions in the Runge-Kutta-Fehlberg method. 

const size_t& OrdinaryDifferentialEquations::get_warning_size(void) const
{
   return(warning_size);
}


// const size_t& get_error_size(void) const method

/// Returns an error size for the solutions in the Runge-Kutta-Fehlberg method. 

const size_t& OrdinaryDifferentialEquations::get_error_size(void) const
{
   return(error_size);
}


// void set(const OrdinaryDifferentialEquations&) method

/// Sets the members of this object to be the members of another object of the same class. 
/// @param other_ordinary_differential_equations Object to be copied. 

void OrdinaryDifferentialEquations::set(const OrdinaryDifferentialEquations& other_ordinary_differential_equations)
{
   independent_variables_number = other_ordinary_differential_equations.independent_variables_number;
   dependent_variables_number = other_ordinary_differential_equations.dependent_variables_number;

   initial_independent_variable = other_ordinary_differential_equations.initial_independent_variable;
   final_independent_variable = other_ordinary_differential_equations.final_independent_variable;

   initial_dependent_variables = other_ordinary_differential_equations.initial_dependent_variables;

   solution_method = other_ordinary_differential_equations.solution_method;

   points_number = other_ordinary_differential_equations.points_number;

   tolerance = other_ordinary_differential_equations.tolerance;

   initial_size = other_ordinary_differential_equations.initial_size;
   warning_size = other_ordinary_differential_equations.warning_size;
   error_size = other_ordinary_differential_equations.error_size;

   display = other_ordinary_differential_equations.display;
}


// void set_initial_independent_variable(const double&) method

/// Sets the initial value for the independent variable. 
/// @param new_initial_independent_variable Initial value for the independent variable. 

void OrdinaryDifferentialEquations::set_initial_independent_variable(const double& new_initial_independent_variable)
{
   initial_independent_variable = new_initial_independent_variable;
}


// void set_final_independent_variable(const double&) method

/// Sets the final value for the independent variable. 
/// @param new_final_independent_variable Final value for the independent variable. 

void OrdinaryDifferentialEquations::set_final_independent_variable(const double& new_final_independent_variable)
{
	final_independent_variable = new_final_independent_variable;
}


// void set_initial_dependent_variables(const Vector<double>&) const method

/// Sets the initial values for the dependent variables. 
/// @param new_initial_dependent_variables Initial values for the dependent variables. 

void OrdinaryDifferentialEquations::set_initial_dependent_variables(const Vector<double>& new_initial_dependent_variables) 
{
   initial_dependent_variables = new_initial_dependent_variables;
}


// void set_initial_dependent_variable(const size_t&, const double&) const method

/// Sets the initial value for a single dependent variable.
/// @param i Index of dependent variable. 
/// @param new_initial_dependent_variable Initial value for the corresponding dependent variable. 

void OrdinaryDifferentialEquations::set_initial_dependent_variable(const size_t& i, const double& new_initial_dependent_variable)
{
	initial_dependent_variables[i] = new_initial_dependent_variable;
}


// void set_solution_method(const SolutionMethod&) method

/// Sets the numerical method for calculating the solution to the ordinary differential equations.
/// Available methos include the Runge-Kutta and Runge-Kutta-Fehlber methods.
/// @param new_solution_method Solution method. 

void OrdinaryDifferentialEquations::set_solution_method(const SolutionMethod& new_solution_method)
{
   solution_method = new_solution_method;
}


// void set_solution_method(const std::string&) method

/// Sets the numerical method for calculating the solution to the ordinary differential equations.
/// Available methos include the Runge-Kutta and Runge-Kutta-Fehlber methods.
/// @param new_solution_method String with the name of the solution method ("RungeKutta" or "RungeKuttaFehlberg"). 

void OrdinaryDifferentialEquations::set_solution_method(const std::string& new_solution_method)
{
   if(new_solution_method == "RungeKutta")
   {
      set_solution_method(RungeKutta);
   }
   else if(new_solution_method == "RungeKuttaFehlberg")
   {
      set_solution_method(RungeKuttaFehlberg);
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: OrdinaryDifferentialEquations class.\n"
             << "void set_solution_method(const std::string&) method.\n"
			 << "Unknown solution method: " << new_solution_method << ".\n";

	  throw std::logic_error(buffer.str());
   }
}


// void set_points_number(const size_t&) method

/// Sets the number of points in the Runge-Kutta method. 
/// @param new_points_number Number of points. 

void OrdinaryDifferentialEquations::set_points_number(const size_t& new_points_number)
{
   points_number = new_points_number;
}


// void set_tolerance(const double&) method

/// Sets the tolerance in the Runge-Kutta-Fehlberg method. 
/// @param new_tolerance Tolerance value. 

void OrdinaryDifferentialEquations::set_tolerance(const double& new_tolerance)
{
   tolerance = new_tolerance;
}


// void set_initial_size(const size_t&) method

/// Sets the initial size for the solution in the Runge-Kutta-Fehlberg method. 
/// @param new_initial_size Initial solution size. 

void OrdinaryDifferentialEquations::set_initial_size(const size_t& new_initial_size)
{
   initial_size = new_initial_size;
}


// void set_warning_size(const size_t&) method

/// Sets a warning size for the solution in the Runge-Kutta-Fehlberg method. 
/// @param new_warning_size Warning solution size. 

void OrdinaryDifferentialEquations::set_warning_size(const size_t& new_warning_size)
{
   warning_size = new_warning_size;
}


// void set_error_size(const size_t&) method

/// Sets the error size for the solution in the Runge-Kutta-Fehlberg method. 
/// @param new_error_size Error solution size. 

void OrdinaryDifferentialEquations::set_error_size(const size_t& new_error_size)
{
   error_size = new_error_size;
}


// void set_default(void) method

/// Sets the following default values:
/// <ul>
/// <li> Independent variables number: 1.
/// <li> Dependent variables number: 0.
/// <li> Solution method: Runge-Kutta-Fehlberg. 
/// <li> Points number: 101.
/// <li> Tolerance: 1e-6
/// <li> Initial size: 1e3.
/// <li> Warning size: 1e6.
/// <li> Error size: 1e9.
/// <li> Display: True.
/// </ul>

void OrdinaryDifferentialEquations::set_default(void)
{
   independent_variables_number = 1;
   dependent_variables_number = 0;

   solution_method = RungeKuttaFehlberg;

   points_number = 101;

   tolerance = 1.0e-6;

   initial_size = (size_t)1.0e3;
   warning_size = (size_t)1.0e6;
   error_size = (size_t)1.0e9;

   display = true;

}


// Matrix<double> calculate_Runge_Kutta_solution(const NeuralNetwork& neural_network, const double& xi, const double& xf, const Vector<double>& yi) const method

/// This method calculate the numerical solution of the ordinary differential equations model using the Runge-Kutta method.
/// @param neural_network Neural network which represents the external inputs to the mathematical model. 

Matrix<double> OrdinaryDifferentialEquations::calculate_Runge_Kutta_solution(const NeuralNetwork& neural_network) const
{
   const size_t variables_number = count_variables_number();

   const double h = (final_independent_variable - initial_independent_variable)/(points_number-1.0);      

   // Fourth order Runge-Kutta coefficients

   Vector< Vector<double> > c(4);

   Matrix<double> solution(points_number, variables_number);

   Vector<double> variables(variables_number);

   // Initial variables

   solution(0,0) = initial_independent_variable;

   for(size_t j = 0; j < dependent_variables_number; j++)
   {
      solution(0,1+j) = initial_dependent_variables[j];
   }

   // Main loop

   for(size_t i = 0; i < points_number-1; i++)
   {
      solution(i+1,0) = solution(i,0) + h;

      // First coefficient 

      variables[0] = solution(i,0); 

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(i,1+j);
      }   

      c[0] = calculate_dependent_variables_dots(neural_network, variables);

      // Second coefficient 

      variables[0] = solution(i,0) + h/2.0; 

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(i,1+j) + h*c[0][j]/2.0;
      }   

      c[1] = calculate_dependent_variables_dots(neural_network, variables);

      // Third coefficient

      variables[0] = solution(i,0) + h/2.0; 
               
      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(i,1+j) + h*c[1][j]/2.0;
      }
           
      c[2] = calculate_dependent_variables_dots(neural_network, variables);

      // Fourth coefficient

      variables[0] = solution(i+1,0);

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(i,1+j) + h*c[2][j]/2.0;
      }
            
      c[3] = calculate_dependent_variables_dots(neural_network, variables);

      // Dependent variables

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         solution(i+1,1+j) = solution(i,1+j) + h*(c[0][j] + 2.0*c[1][j] + 2.0*c[2][j] + c[3][j])/6.0;
      }
   }

   solution(points_number-1,0) = final_independent_variable;

   return(solution);
}


// Matrix<double> calculate_Runge_Kutta_final_solution(const NeuralNetwork& neural_network) const method

/// This method calculate the solution at the final independent variable of the ordinary differential equations model using the Runge-Kutta method.
/// @param neural_network Neural network which represents the external inputs to the mathematical model. 

Vector<double> OrdinaryDifferentialEquations::calculate_Runge_Kutta_final_solution(const NeuralNetwork& neural_network) const
{
   const size_t variables_number = count_variables_number();

   const double h = (final_independent_variable - initial_independent_variable)/(points_number-1.0);      

   // Fourth order Runge-Kutta coefficients

   Vector< Vector<double> > c(4);

   Vector<double> final_solution(variables_number);

   Vector<double> variables(variables_number);

   // Initial variables

   final_solution[0] = initial_independent_variable;

   for(size_t j = 0; j < dependent_variables_number; j++)
   {
      final_solution[0] = initial_dependent_variables[j];
   }

   // Main loop

   for(size_t i = 0; i < points_number-1; i++)
   {
      final_solution[0] = final_solution[0] + h;

      // First coefficient 

      variables[0] = final_solution[0]; 

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j]; 
      }   

      c[0] = calculate_dependent_variables_dots(neural_network, variables);

      // Second coefficient 

      variables[0] = final_solution[0] + h/2.0; 

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j] + h*c[0][j]/2.0; 
      }   

      c[1] = calculate_dependent_variables_dots(neural_network, variables);

      // Third coefficient

      variables[0] = final_solution[0] + h/2.0; 
               
      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j] + h*c[1][j]/2.0; 
      }
           
      c[2] = calculate_dependent_variables_dots(neural_network, variables);

      // Fourth coefficient

      variables[0] = final_solution[0]; 

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j] + h*c[2][j]/2.0; 
      }
            
      c[3] = calculate_dependent_variables_dots(neural_network, variables);

      // Dependent variables

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         final_solution[1+j] = final_solution[1+j] + h*(c[0][j] + 2.0*c[1][j] + 2.0*c[2][j] + c[3][j])/6.0;
      }
   }

   return(final_solution);
}


// Matrix<double> calculate_Runge_Kutta_Fehlberg_solution(const NeuralNetwork&) const method

/// This method calculate the numerical solution of the ordinary differential equations model using the Runge-Kutta method.
/// @param neural_network Neural network which represents the external inputs to the mathematical model. 

Matrix<double> OrdinaryDifferentialEquations::calculate_Runge_Kutta_Fehlberg_solution(const NeuralNetwork& neural_network) const
{
   const double epsilon = 1.0e-12;//std::numeric_limits<double>::epsilon();

   const double a2 = 1.0/5.0;
   const double a3 = 3.0/10.0;
   const double a4 = 3.0/5.0;
   const double a5 = 1.0;
   const double a6 = 7.0/8.0;

   const double b21 = 1.0/5.0;   
   const double b31 = 3.0/40.0;
   const double b32 = 9.0/40.0;   
   const double b41 = 3.0/10.0;
   const double b42 = -9.0/10.0;   
   const double b43 = 6.0/5.0; 
   const double b51 = -11.0/54.0;
   const double b52 = 5.0/2.0;
   const double b53 = -70.0/27.0;
   const double b54 = 35.0/27.0; 
   const double b61 = 1631.0/55296.0;
   const double b62 = 175.0/512.0;   
   const double b63 = 575.0/13824.0;   
   const double b64 = 44275.0/110592.0;   
   const double b65 = 253.0/4096.0;

   const double c41 = 37.0/378.0;
   const double c42 = 0.0;
   const double c43 = 250.0/621.0;
   const double c44 = 125.0/594.0;
   const double c45 = 0.0;
   const double c46 = 512.0/1771.0;

   const double c51 = 2825.0/27648.0;  
   const double c52 = 0.0;  
   const double c53 = 18575.0/48384.0;   
   const double c54 = 13525.0/55296.0;
   const double c55 = 277.0/14336.0;
   const double c56 = 1.0/4.0;

   const double d1 = c41 - c51;
   const double d2 = c42 - c52;
   const double d3 = c43 - c53;
   const double d4 = c44 - c54;
   const double d5 = c45 - c55;
   const double d6 = c46 - c56;

   const size_t variables_number = count_variables_number();

   size_t size = initial_size;

   Vector<double> errors(dependent_variables_number, 0.0);

   double error = 0.0;

   Vector< Vector<double> > c(6);

   double hmin = 0.0;
   double h = (final_independent_variable - initial_independent_variable)*1.0e-3;

   Vector<double> variables(variables_number);

   Matrix<double> solution(size, variables_number);

   size_t point_index = 0;

   // Initial values

   solution(0,0) = initial_independent_variable;

   for(size_t j = 0; j < dependent_variables_number; j++)
   {
      solution(0,1+j) = initial_dependent_variables[j];
   } 

   // Main loop

   while(solution(point_index,0) < final_independent_variable)
   {   
      // Set smallest allowable stepsize

      hmin = 32.0*epsilon*fabs(solution(point_index,0));

      if(h < hmin)
      {
         if(display)
         {
            std::cout << "OpenNN Warning: OrdinaryDifferentialEquations class.\n" 
                      << "calculate_Runge_Kutta_Fehlberg_solution() method.\n" 
                      << "Step size is less than smallest allowable." << std::endl;
         }

         h = hmin;
      }

      if(solution(point_index,0) + h > final_independent_variable)
      {        
         h = final_independent_variable - solution(point_index,0);
      }

      // First coefficient

      variables[0] = solution(point_index,0);

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(point_index,1+j);
      }

      c[0] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Second coefficient

      variables[0] = solution(point_index,0) + a2*h;

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(point_index,1+j)+ b21*c[0][j];
      }

      c[1] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Third coefficient

      variables[0] = solution(point_index,0) + a3*h;

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] =  solution(point_index,1+j) + b31*c[0][j] + b32*c[1][j];
      }

      c[2] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Fourth coefficient

      variables[0] = solution(point_index,0) + a4*h;

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(point_index,1+j) + b41*c[0][j] + b42*c[1][j] + b43*c[2][j];
      }

      c[3] = calculate_dependent_variables_dots(neural_network, variables)*h; 

      // Fifth coefficient

      variables[0] = solution(point_index,0) + a5*h;

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(point_index,1+j) + b51*c[0][j] + b52*c[1][j] + b53*c[2][j] + b54*c[3][j];
      }
      
      c[4] = calculate_dependent_variables_dots(neural_network, variables)*h;
            
      // Sixth coefficient

      variables[0] = solution(point_index,0) + a6*h;
            
      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = solution(point_index,1+j) +  b61*c[0][j] + b62*c[1][j] + b63*c[2][j] + b64*c[3][j] + b65*c[4][j];
      }
      
      c[5] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Error estimate

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         errors[j] = fabs(d1*c[0][j] + d2*c[1][j] + d3*c[2][j] + d4*c[3][j] + d5*c[4][j] + d6*c[5][j]);         
      }

      error = errors.calculate_maximum();
   
      if(error <= tolerance)
      {
         solution(point_index+1,0) = solution(point_index,0) + h;

         for(size_t j = 0; j < dependent_variables_number; j++)
         {
            solution(point_index+1,1+j) = solution(point_index,1+j) + c51*c[0][j] + c52*c[1][j] + c53*c[2][j] + c54*c[3][j] + c55*c[4][j] + c56*c[5][j];
         }

         point_index++;

         if(error != 0.0)
         {
            h *= 0.9*pow(fabs(tolerance/error), 0.2);
         }

         if(point_index >= size)
         {
            size *= 2;

            if(display && size > warning_size)
            {
               std::cout << "OpenNN Warning: OrdinaryDifferentialEquations class." << std::endl
                         << "calculate_Runge_Kutta_Fehlberg_solution() method." << std::endl
                         << "Solution size is " << size << std::endl;
            }
            else if(size > error_size)
            {
               std::ostringstream buffer;

               buffer << "OpenNN Exception: OrdinaryDifferentialEquations class." << std::endl
                      << "calculate_Runge_Kutta_Fehlberg_solution() method." << std::endl
                      << "Solution size is bigger than greatest allowable." << std::endl;
                            
               throw std::logic_error(buffer.str());          
            }

            // @todo Substitute resize method

            //solution.resize(size, variables_number);
         }
      }
      else
      {
         h *= 0.9*pow(fabs(tolerance/error), 0.25);
      }
   } // end while loop   

   // @todo Substitute resize method

   //solution.resize(point_index+1, variables_number);

   return(solution);
}


// Vector<double> calculate_Runge_Kutta_Fehlberg_final_solution(const NeuralNetwork&) const method

/// This method calculate the numerical solution of the ordinary differential equations model using the Runge-Kutta-Fehlberg method.
/// @param neural_network Neural network which represents the external inputs to the mathematical model. 

Vector<double> OrdinaryDifferentialEquations::calculate_Runge_Kutta_Fehlberg_final_solution(const NeuralNetwork& neural_network) const
{
   const double epsilon = 1.0e-12;//std::numeric_limits<double>::epsilon();

   const double a2 = 1.0/5.0;
   const double a3 = 3.0/10.0;
   const double a4 = 3.0/5.0;
   const double a5 = 1.0;
   const double a6 = 7.0/8.0;

   const double b21 = 1.0/5.0;   
   const double b31 = 3.0/40.0;
   const double b32 = 9.0/40.0;   
   const double b41 = 3.0/10.0;
   const double b42 = -9.0/10.0;   
   const double b43 = 6.0/5.0; 
   const double b51 = -11.0/54.0;
   const double b52 = 5.0/2.0;
   const double b53 = -70.0/27.0;
   const double b54 = 35.0/27.0; 
   const double b61 = 1631.0/55296.0;
   const double b62 = 175.0/512.0;   
   const double b63 = 575.0/13824.0;   
   const double b64 = 44275.0/110592.0;   
   const double b65 = 253.0/4096.0;

   const double c41 = 37.0/378.0;
   const double c42 = 0.0;
   const double c43 = 250.0/621.0;
   const double c44 = 125.0/594.0;
   const double c45 = 0.0;
   const double c46 = 512.0/1771.0;

   const double c51 = 2825.0/27648.0;  
   const double c52 = 0.0;  
   const double c53 = 18575.0/48384.0;   
   const double c54 = 13525.0/55296.0;
   const double c55 = 277.0/14336.0;
   const double c56 = 1.0/4.0;

   const double d1 = c41 - c51;
   const double d2 = c42 - c52;
   const double d3 = c43 - c53;
   const double d4 = c44 - c54;
   const double d5 = c45 - c55;
   const double d6 = c46 - c56;

   const size_t variables_number = count_variables_number();

   Vector<double> errors(dependent_variables_number, 0.0);

   double error = 0.0;

   Vector< Vector<double> > c(6);

   double hmin = 0.0;
   double h = (final_independent_variable - initial_independent_variable)*1.0e-3;

   Vector<double> variables(variables_number);

   Vector<double> final_solution(variables_number);

   // Initial values

   final_solution[0] = initial_independent_variable;

   for(size_t j = 0; j < dependent_variables_number; j++)
   {
      final_solution[1+j] = initial_dependent_variables[j];
   } 

   // Main loop

   while(final_solution[0] < final_independent_variable)
   {   
      // Set smallest allowable stepsize

      hmin = 32.0*epsilon*fabs(final_solution[0]);

      if(h < hmin)
      {
         if(display)
         {
            std::cout << "OpenNN Warning: OrdinaryDifferentialEquations class.\n" 
                      << "calculate_Runge_Kutta_Fehlberg_solution() method.\n" 
                      << "Step size is less than smallest allowable." << std::endl;
         }

         h = hmin;
      }

      if(final_solution[0] + h > final_independent_variable)
      {        
         h = final_independent_variable - final_solution[0];
      }

      // First coefficient

      variables[0] = final_solution[0]; 

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j];
      }

      c[0] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Second coefficient

      variables[0] = final_solution[0] + a2*h;

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j]+ b21*c[0][j];
      }

      c[1] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Third coefficient

      variables[0] = final_solution[0] + a3*h;

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] =  final_solution[1+j] + b31*c[0][j] + b32*c[1][j]; 
      }

      c[2] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Fourth coefficient

      variables[0] = final_solution[0] + a4*h; 

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j] + b41*c[0][j] + b42*c[1][j] + b43*c[2][j];
      }

      c[3] = calculate_dependent_variables_dots(neural_network, variables)*h; 

      // Fifth coefficient

      variables[0] = final_solution[0] + a5*h;

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j] + b51*c[0][j] + b52*c[1][j] + b53*c[2][j] + b54*c[3][j];
      }
      
      c[4] = calculate_dependent_variables_dots(neural_network, variables)*h;
            
      // Sixth coefficient

      variables[0] = final_solution[0] + a6*h; 
            
      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         variables[1+j] = final_solution[1+j] +  b61*c[0][j] + b62*c[1][j] + b63*c[2][j] + b64*c[3][j] + b65*c[4][j];
      }
      
      c[5] = calculate_dependent_variables_dots(neural_network, variables)*h;

      // Error estimate

      for(size_t j = 0; j < dependent_variables_number; j++)
      {
         errors[j] = fabs(d1*c[0][j] + d2*c[1][j] + d3*c[2][j] + d4*c[3][j] + d5*c[4][j] + d6*c[5][j]);         
      }

      error = errors.calculate_maximum();
   
      if(error <= tolerance)
      {
         final_solution[0] +=  h;

         for(size_t j = 0; j < dependent_variables_number; j++)
         {
            final_solution[1+j] += c51*c[0][j] + c52*c[1][j] + c53*c[2][j] + c54*c[3][j] + c55*c[4][j] + c56*c[5][j];   
         }

         if(error != 0.0)
         {
            h *= 0.9*pow(fabs(tolerance/error), 0.2);
         }
      }
      else
      {
         h *= 0.9*pow(fabs(tolerance/error), 0.25);
      }
   } // end while loop   


   return(final_solution);
}


// Matrix<double> calculate_solutions(const NeuralNetwork&) const method

Matrix<double> OrdinaryDifferentialEquations::calculate_solutions(const NeuralNetwork& neural_network) const
{
   switch(solution_method)
   {
      case RungeKutta:
      {
         return(calculate_Runge_Kutta_solution(neural_network));
      }            
      break;

      case RungeKuttaFehlberg:
      {
         return(calculate_Runge_Kutta_Fehlberg_solution(neural_network));
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Error: OrdinaryDifferentialEquations class\n"
                << "Vector<double> calculate_solutions(const NeuralNetwork&) const method.\n"               
                << "Unknown solution method.\n";

	     throw std::logic_error(buffer.str());
      }
      break;
   }
}


// Vector<double> calculate_final_solutions(const NeuralNetwork&) const method

Vector<double> OrdinaryDifferentialEquations::calculate_final_solutions(const NeuralNetwork& neural_network) const
{
   switch(solution_method)
   {
      case RungeKutta:
      {
         return(calculate_Runge_Kutta_final_solution(neural_network));
      }            
      break;

      case RungeKuttaFehlberg:
      {
         return(calculate_Runge_Kutta_Fehlberg_final_solution(neural_network));
      }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: ScalingLayer class\n"
                << "Vector<double> calculate_final_solutions(const NeuralNetwork&) const method.\n"               
                << "Unknown solution method.\n";

	     throw std::logic_error(buffer.str());
      }
      break;
   }
}


// std::string to_string(void) const method

/// Returns a string representation of the current ordinary differential equations object. 

std::string OrdinaryDifferentialEquations::to_string(void) const
{
   std::ostringstream buffer; 

   buffer << "Mathematical model\n"
          << "Independent variables number: " << independent_variables_number << "\n" 
          << "Dependent variables number: " << dependent_variables_number << "\n"
          << "Initial independent variable: " << initial_independent_variable << "\n"
          << "Final independent variable: " << final_independent_variable << "\n"
          << "Initial dependent variables: " <<  initial_dependent_variables << "\n"
          << "Solution method: " << write_solution_method() << "\n"
          << "Points number: " << points_number << "\n"
          << "Tolerance" << tolerance << "\n"
          << "Initial size: " << initial_size << "\n"
          << "Warning size: " << warning_size << "\n"
          << "Error size: " << error_size << "\n"
          << "Display: " << display << std::endl;

   return(buffer.str());
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the ordinary differential equations object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this document. 

tinyxml2::XMLDocument* OrdinaryDifferentialEquations::to_XML(void) const
{    
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   std::ostringstream buffer;

   tinyxml2::XMLElement* ordinary_differential_equations_element = document->NewElement("OrdinaryDifferentialEquations");

   document->InsertFirstChild(ordinary_differential_equations_element);

   // Independent variables number 
   {
      tinyxml2::XMLElement* element = document->NewElement("IndependentVariablesNumber");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << independent_variables_number;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Dependent variables number 
   {
      tinyxml2::XMLElement* element = document->NewElement("DependentVariablesNumber");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << dependent_variables_number;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Initial independent variable
   {
      tinyxml2::XMLElement* element = document->NewElement("InitialIndependentVariable");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << initial_independent_variable;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Final independent variable
   {
      tinyxml2::XMLElement* element = document->NewElement("FinalIndependentVariable");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << final_independent_variable;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Initial dependent variables 
   {
      tinyxml2::XMLElement* element = document->NewElement("InitialDependentVariables");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << initial_dependent_variables;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Solution method
   {
      tinyxml2::XMLElement* element = document->NewElement("SolutionMethod");
      ordinary_differential_equations_element->LinkEndChild(element);

      tinyxml2::XMLText* text = document->NewText(write_solution_method().c_str());
      element->LinkEndChild(text);
   }

   // Points number
   {
      tinyxml2::XMLElement* element = document->NewElement("PointsNumber");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << points_number;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Tolerance
   {
      tinyxml2::XMLElement* element = document->NewElement("Tolerance");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << tolerance;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Initial size
   {
      tinyxml2::XMLElement* element = document->NewElement("InitialSize");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << initial_size;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Warning size
   {
      tinyxml2::XMLElement* element = document->NewElement("WarningSize");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << warning_size;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Error size
   {
      tinyxml2::XMLElement* element = document->NewElement("ErrorSize");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << error_size;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Display 
   {
      tinyxml2::XMLElement* element = document->NewElement("Display");
      ordinary_differential_equations_element->LinkEndChild(element);

      buffer.str("");
      buffer << display;

      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   return(document);   
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the ordinary differential equations object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void OrdinaryDifferentialEquations::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("OrdinaryDifferentialEquations");

    // Independent variables number

    file_stream.OpenElement("IndependentVariablesNumber");

    buffer.str("");
    buffer << independent_variables_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Dependent variables number

    file_stream.OpenElement("DependentVariablesNumber");

    buffer.str("");
    buffer << dependent_variables_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Initial independent variable

    file_stream.OpenElement("InitialIndependentVariable");

    buffer.str("");
    buffer << initial_independent_variable;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Final independent variable

    file_stream.OpenElement("FinalIndependentVariable");

    buffer.str("");
    buffer << final_independent_variable;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Initial dependent variables

    file_stream.OpenElement("InitialDependentVariables");

    buffer.str("");
    buffer << initial_dependent_variables;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Solution method

    file_stream.OpenElement("SolutionMethod");

    file_stream.PushText(write_solution_method().c_str());

    file_stream.CloseElement();

    // Points number

    file_stream.OpenElement("PointsNumber");

    buffer.str("");
    buffer << points_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Tolerance

    file_stream.OpenElement("Tolerance");

    buffer.str("");
    buffer << tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Initial size

    file_stream.OpenElement("InitialSize");

    buffer.str("");
    buffer << initial_size;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Warning size

    file_stream.OpenElement("WarningSize");

    buffer.str("");
    buffer << warning_size;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Error size

    file_stream.OpenElement("ErrorSize");

    buffer.str("");
    buffer << error_size;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Display

    file_stream.OpenElement("Display");

    buffer.str("");
    buffer << display;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this ordinary differential equations object.
/// @param document TinyXML document containing the member data.

void OrdinaryDifferentialEquations::from_XML(const tinyxml2::XMLDocument& document)
{
  // Dependent variables number
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("DependentVariablesNumber");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_dependent_variables_number(atoi(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Initial independent variable
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("InitialIndependentVariable");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_initial_independent_variable(atof(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Final independent variable
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("FinalIndependentVariable");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_final_independent_variable(atof(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Initial dependent variables
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("InitialDependentVariables");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              Vector<double> new_initial_dependent_variables;
              new_initial_dependent_variables.parse(text);

              set_initial_dependent_variables(new_initial_dependent_variables);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Solution method
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("SolutionMethod");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              std::string new_solution_method(text);

              set_solution_method(new_solution_method);
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Points number
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("PointsNumber");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_points_number(atoi(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Tolerance
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("Tolerance");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_tolerance(atof(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Initial size
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("InitialSize");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_initial_size(atoi(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Warning size
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("WarningSize");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_warning_size(atoi(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Error size
  {
     const tinyxml2::XMLElement* element = document.FirstChildElement("ErrorSize");

     if(element)
     {
        const char* text = element->GetText();

        if(text)
        {
           try
           {
              set_error_size(atoi(text));
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }

  // Display
  {
     const tinyxml2::XMLElement* display_element = document.FirstChildElement("Display");

     if(display_element)
     {
        const char* display_text = display_element->GetText();

        if(display_text)
        {
           try
           {
              std::string display_string(display_text);

              set_display(display_string != "0");
           }
           catch(const std::logic_error& e)
           {
              std::cout << e.what() << std::endl;
           }
        }
     }
  }
}


// void save_data(const NeuralNetwork&, const std::string&) const method

/// Saves the solution data of an ordinary differential equation to a data file.
/// @param neural_network Neural network which represents the external inputs to the mathematical model.
/// @param file_name Name of solution data file.

void OrdinaryDifferentialEquations::save_data(const NeuralNetwork& neural_network, const std::string& file_name) const
{
   const Matrix<double> solution = calculate_Runge_Kutta_solution(neural_network);

   solution.save(file_name);
}

}


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
