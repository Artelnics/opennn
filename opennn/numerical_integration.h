/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N U M E R I C A L   I N T E G R A T I O N   C L A S S   H E A D E R                                        */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __NUMERICALINTEGRATION_H__
#define __NUMERICALINTEGRATION_H__

// System includes

#include<iostream>

// OpenNN includes

#include "vector.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"


namespace OpenNN
{

/// This class contains methods for numerical integration of functions. 
/// In particular it implements the trapezoid method and the Simpson's method.

class NumericalIntegration 
{
   public:

   // CONSTRUCTOR

   explicit NumericalIntegration(void);


   // DESTRUCTOR

   virtual ~NumericalIntegration(void);

   /// Enumeration of available methods for numerical integration.

   enum NumericalIntegrationMethod{TrapezoidMethod, SimpsonMethod};

   // METHODS

   const NumericalIntegrationMethod& get_numerical_integration_method(void) const;
   std::string write_numerical_integration_method(void) const;
   
   const bool& get_display(void) const;

   void set(const NumericalIntegration&);

   void set_numerical_integration_method(const NumericalIntegrationMethod&);
   void set_numerical_integration_method(const std::string&);

   void set_display(const bool&);

   void set_default(void);
 
   // Integration of pairs of data (x,y)

   double calculate_trapezoid_integral(const Vector<double>&, const Vector<double>&) const;
   double calculate_Simpson_integral(const Vector<double>&, const Vector<double>&) const;

   double calculate_integral(const Vector<double>&, const Vector<double>&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;   
   void from_XML(const tinyxml2::XMLDocument&);

   void write_XML(tinyxml2::XMLPrinter&) const;
   // void read_XML(   );

   // Integration of class member methods

   // template<class T> double calculate_trapezoid_integral
   // (const T&, double (T::*f)(const double&) const, const double& a, const double& b, const size_t& n) method

   /// This method evaluates the integral of a function given as a class
   /// member method using the composite trapezoid rule with n intervals.
   /// @param t : Object constructor containing the member method to integrate.  
   /// @param f: Pointer to the member method.
   /// @param a: Lower integration limit. 
   /// @param b: Upper integration limit. 
   /// @param n: Number of intervals. 

   template<class T> 
   static double calculate_trapezoid_integral(const T& t,
   double (T::*f)(const double&) const , const double& a, const double& b, const size_t& n)
   {
      // Integration step

      const double h = (b-a)/(n-1.0);

      // Sum

      double sum = (t.*f)(a)/2.0;

      for(size_t i = 1; i < n-1; i++)
      {
         sum += (t.*f)(a + i*h);
      }

      sum += (t.*f)(b)/2.0;

      // Trapezoidal rule

      return(h*sum);
   }


   // Vector<double> calculate_trapezoid_integral
   // (const T&, Vector<double> (T::*f)(const double&) const, const double, const double, const size_t&) const method

   /// This method evaluates the integral of a vector function given as a class
   /// member method using the composite trapezoid rule with n intervals.
   /// @param t : Object constructor containing the member method to integrate.  
   /// @param f: Pointer to the member method.
   /// @param a: Lower integration limit. 
   /// @param b: Upper integration limit. 
   /// @param n: Number of intervals. 

   template<class T> 
   static Vector<double> calculate_trapezoid_integral(const T& t, Vector<double> (T::*f)(const double&) const , const double& a, const double& b, const size_t& n)
   {
      // Integration step

      const double h = (b-a)/(n-1.0);

      // Sum

      Vector<double> sum = (t.*f)(a)/2.0;

      for(size_t i = 1; i < n-1; i++)
      {
         sum += (t.*f)(a + i*h);
      }

      sum += (t.*f)(b)/2.0;

      // Trapezoidal rule

      return(sum*h);
   }


   // template<class T> double calculate_Simpson_integral
   // (const T&, double (T::*f)(double), const double&, const double&, const size_t&) const method

   /// This method evaluates the integral of a function given as a class
   /// member method using the composite Simpsons rule with n intervals. 
   /// @param t : Object constructor containing the member method to integrate.  
   /// @param f: Pointer to the member method.
   /// @param a: Lower integration limit. 
   /// @param b: Upper integration limit. 
   /// @param n: Number of intervals. 

   template<class T> 
   static double calculate_Simpson_integral(const T& t, double (T::*f)(const double&) const , const double& a, const double& b, const size_t& n)
   {
      const double h = (b-a)/(n-1.0);

      double sum = (t.*f)(a)/3.0;

      for(size_t i = 1; i < n-1; i++)
      {
         if(i%2 != 0) // odd
         {
            sum += 4.0*(t.*f)(a + i*h)/3.0;
         }
         else // even
         {
            sum += 2.0*(t.*f)(a + i*h)/3.0;       
         }
      }

      sum += (t.*f)(b)/3.0;

      return(h*sum);
   }


   // Vector<double> calculate_Simpson_integral
   // (const T&, Vector<double> (T::*f)(const double&) const , const double&, const double&, const size_t&) const method

   /// This method evaluates the integral of a vector function given as a class
   /// member method using the composite Simpsons rule with n intervals. 
   /// @param t : Object constructor containing the member method to integrate.  
   /// @param f: Pointer to the member method.
   /// @param a: Lower integration limit. 
   /// @param b: Upper integration limit. 
   /// @param n: Number of intervals. 

   template<class T> 
   static Vector<double> calculate_Simpson_integral(const T& t, Vector<double> (T::*f)(const double&) const , const double& a, const double& b, const size_t& n)
   {
      const double h = (b-a)/(n-1.0);

      Vector<double> sum = (t.*f)(a)/3.0;

      for(size_t i = 1; i < n-1; i++)
      {
         if(i%2 != 0) // odd
         {
            sum += (t.*f)(a + i*h)*4.0/3.0;
         }
         else // even
         {
            sum += (t.*f)(a + i*h)*2.0/3.0;       
         }
      }

      sum += (t.*f)(b)/3.0;

      // Simpson's rule

      return(sum*h);
   }


   // template<class T> double calculate_integral
   // (const T&, double (T::*f)(const double&) const, const double&, const double&, const size_t&) const method

   /// This method evaluates the integral of a function given as a class member method. 
   /// @param t : Object constructor containing the member method to integrate.  
   /*/// @param f: Pointer to the member method.*/
   /// @param a: Lower integration limit. 
   /// @param b: Upper integration limit. 
   /// @param n: Number of intervals. 

   template<class T> 
   double calculate_integral(const T& t,/* double (T::*)(const double&) const , */const double& a, const double& b, const size_t& n) const
   {
      switch(numerical_integration_method)
      {
         case TrapezoidMethod:
         {
            return(calculate_trapezoid_integral(t, &T::f, a, b, n));
	     }
	     break;

         case SimpsonMethod:
         {
            return(calculate_Simpson_integral(t, &T::f, a, b, n));
 	     }
	     break;

         default:
         {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: NumericalIntegration class.\n"
                   << "double calculate_integral(const T&, double (T::*f)(const double&) const, const double&, const double&, const size_t&) const method.\n"
                   << "Unknown numerical integration method.\n";
 
            throw std::logic_error(buffer.str());
	     }
	     break;
      }
   }


   // Vector<double> calculate_integral
   // (const T&, Vector<double> (T::*f)(const double&) const, const double&, const double&, const size_t&) const method

   /// This method evaluates the integral of a vector function given as a class member method with n intervals. 
   /// @param t : Object constructor containing the member method to integrate.  
   /*/// @param f: Pointer to the member method.*/
   /// @param a: Lower integration limit. 
   /// @param b: Upper integration limit. 
   /// @param n: Number of intervals. 

   template<class T> 
   Vector<double> calculate_integral(const T& t,/* Vector<double> (T::*)(const double&) const , */const double& a, const double& b, const size_t& n) const
   {
      switch(numerical_integration_method)
      {
         case TrapezoidMethod:
         {
            return(calculate_trapezoid_integral(t, &T::f, a, b, n));
	     }
	     break;

         case SimpsonMethod:
         {
            return(calculate_Simpson_integral(t, &T::f, a, b, n));
 	     }
	     break;

         default:
         {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: NumericalIntegration class.\n"
                   << "double calculate_integral(const T&, Vector<double> (T::*f)(const double&) const, const double&, const double&, const size_t&) const method.\n"
                   << "Unknown numerical integration method.\n";
 
            throw std::logic_error(buffer.str());
	     }
	     break;
      }
   }


private:

   /// Numerical integration method variable. 

   NumericalIntegrationMethod numerical_integration_method;

   /// Flag for displaying warning messages from this class. 

   bool display;

};

}

#endif


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
