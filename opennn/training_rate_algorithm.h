/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   R A T E   A L G O R I T H M   C L A S S   H E A D E R                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __TRAININGRATEALGORITHM_H__
#define __TRAININGRATEALGORITHM_H__

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "neural_network.h"
#include "loss_index.h"

// TinyXml includes

#include "../tinyxml2/tinyxml2.h"

namespace OpenNN
{

/// This class is used by many different training algorithms to calculate the training rate given a training direction. 
/// It implements the golden section method and the Brent's methods. 

class TrainingRateAlgorithm
{

public:

   // ENUMERATIONS

   /// Available training operators for obtaining the perform_training rate.

   enum TrainingRateMethod{Fixed, GoldenSection, BrentMethod};

   // DEFAULT CONSTRUCTOR

   explicit TrainingRateAlgorithm(void);

   // GENERAL CONSTRUCTOR

   explicit TrainingRateAlgorithm(LossIndex*);

   // XML CONSTRUCTOR

   explicit TrainingRateAlgorithm(const tinyxml2::XMLDocument&);

   // DESTRUCTOR

   virtual ~TrainingRateAlgorithm(void);

   ///
   /// Defines a set of three points (A, U, B) for bracketing a directional minimum.
   ///

   struct Triplet
   {
       /// Default constructor.

       Triplet(void)
       {
           A.set(2, 0.0);
           U.set(2, 0.0);
           B.set(2, 0.0);
       }

       /// Destructor.

       virtual ~Triplet(void)
       {
       }

       /// Equal to operator.
       /// It compares this triplet with another triplet.
       /// It returns true if both triplets have the same points A, U and B, and false otherwise.
       /// @ param other_triplet Triplet to be compared with.

       inline bool operator == (const Triplet& other_triplet) const
       {
          if(A == other_triplet.A
          && U == other_triplet.U
          && B == other_triplet.B)
          {
             return(true);
          }
          else
          {
             return(false);
          }
       }

       /// Returns true if the length of the interval (A,B) is zero,
       /// and false otherwise.

       inline bool has_length_zero(void) const
       {
           if(A[0] == B[0])
           {
              return(true);
           }
           else
           {
              return(false);
           }
       }

       /// Returns true if the interval (A,B) is constant,
       /// and false otherwise.

       inline bool is_constant(void) const
       {
           if(A[1] == B[1])
           {
              return(true);
           }
           else
           {
              return(false);
           }
       }

       /// Writes a string with the values of A, U and B.

       inline std::string to_string(void) const
       {
           std::ostringstream buffer;

           buffer << "A = (" << A[0] << "," << A[1] << ")\n"
                  << "U = (" << U[0] << "," << U[1] << ")\n"
                  << "B = (" << B[0] << "," << B[1] << ")" << std::endl;

           return(buffer.str());
       }

       /// Prints the triplet points to the standard output.

       inline void print(void) const
       {
           std::cout << to_string();
       }

       /// Checks that the points A, U and B define a minimum.
       /// That is, a < u < b, fa > fu and fu < fb.
       /// If some of that conditions is not satisfied, an exception is thrown.

       inline void check(void) const
       {
           std::ostringstream buffer;

           if(A[0] > U[0] || U[0] > B[0])
           {
              buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
                     << "void check(void) const method.\n"
                     << "Uncorrect triplet:\n"
                     << to_string();

              throw std::logic_error(buffer.str());
           }

           if(A[1] < U[1] || U[1] > B[1])
           {
              buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
                     << "void check(void) const method.\n"
                     << "Triplet does not satisfy minimum condition:\n"
                     << to_string();

              throw std::logic_error(buffer.str());
           }
       }

       /// Left point of the triplet.

       Vector<double> A;

       /// Interior point of the triplet.

       Vector<double> U;

       /// Right point of the triplet.

       Vector<double> B;
   };


   // METHODS

   // Get methods

   LossIndex* get_loss_index_pointer(void) const;

   bool has_loss_index(void) const;

   // Training operators

   const TrainingRateMethod& get_training_rate_method(void) const;
   std::string write_training_rate_method(void) const;

   // Training parameters

   const double& get_bracketing_factor(void) const;   
   const double& get_training_rate_tolerance(void) const;

   const double& get_warning_training_rate(void) const;

   const double& get_error_training_rate(void) const;
  
   // Utilities
   
   const bool& get_display(void) const;
  
   // Set methods

   void set(void);
   void set(LossIndex*);

   void set_loss_index_pointer(LossIndex*);

   // Training operators

   void set_training_rate_method(const TrainingRateMethod&);
   void set_training_rate_method(const std::string&);

   // Training parameters

   void set_bracketing_factor(const double&);   
   void set_training_rate_tolerance(const double&);

   void set_warning_training_rate(const double&);

   void set_error_training_rate(const double&);

   // Utilities

   void set_display(const bool&);

   virtual void set_default(void);

   // Training rate method

   double calculate_golden_section_training_rate(const Triplet&) const;
   double calculate_Brent_method_training_rate(const Triplet&) const;

   Triplet calculate_bracketing_triplet(const double&, const Vector<double>&, const double&) const;

   Vector<double> calculate_fixed_directional_point(const double&, const Vector<double>&, const double&) const;
   Vector<double> calculate_golden_section_directional_point(const double&, const Vector<double>&, const double&) const;
   Vector<double> calculate_Brent_method_directional_point(const double&, const Vector<double>&, const double&) const;

   Vector<double> calculate_directional_point(const double&, const Vector<double>&, const double&) const;

   // Serialization methods

   tinyxml2::XMLDocument* to_XML(void) const;   
   void from_XML(const tinyxml2::XMLDocument&);   

   void write_XML(tinyxml2::XMLPrinter&) const;
   //void read_XML(   );

protected:

   // FIELDS

   /// Pointer to an external loss functional object.

   LossIndex* loss_index_pointer;


   // TRAINING OPERATORS

   /// Variable containing the actual method used to obtain a suitable perform_training rate. 

   TrainingRateMethod training_rate_method;

   /// Increase factor when bracketing a minimum.

   double bracketing_factor;

   /// Maximum interval length for the training rate.

   double training_rate_tolerance;

   /// Big training rate value at which the algorithm displays a warning. 

   double warning_training_rate;

   /// Big training rate value at which the algorithm throws an exception. 

   double error_training_rate;

   // UTILITIES

   /// Display messages to screen.

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

