//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEARNINGRATEALGORITHM_H
#define LEARNINGRATEALGORITHM_H

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <omp.h>

// OpenNN includes

#include "config.h"
#include "neural_network.h"
#include "loss_index.h"
#include "optimization_algorithm.h"

namespace OpenNN
{

/// A learning rate that is adjusted according to an algorithm during training to minimize training time.

///
/// This class is used by many different optimization algorithms to calculate the learning rate given a training direction.
///
/// It implements the golden section method and the Brent's method.

class LearningRateAlgorithm
{

public:

   // Enumerations

   /// Available training operators for obtaining the perform_training rate.

   enum LearningRateMethod{GoldenSection, BrentMethod};

   // Constructors

   explicit LearningRateAlgorithm();

   explicit LearningRateAlgorithm(LossIndex*);

   // Destructor

   virtual ~LearningRateAlgorithm();

   /// Defines a set of three points (A, U, B) for bracketing a directional minimum.

   struct Triplet
   {
       /// Default constructor.

       Triplet()
       {
           A = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
           U = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
           B = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
       }

       /// Destructor.

       virtual ~Triplet()
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
             return true;
          }
          else
          {
             return false;
          }
       }

       inline type get_length() const
       {
           return abs(B.first - A.first);
       }


       inline pair<type,type> minimum() const
       {
           Tensor<type, 1> losses(3);

           losses.setValues({A.second, U.second, B.second});

           const Index minimal_index = OpenNN::minimal_index(losses);

           if(minimal_index == 0) return A;
           else if(minimal_index == 1) return U;
           else return B;
       }

       /// Returns true if the length of the interval(A,B) is zero,
       /// and false otherwise.

       inline bool has_length_zero() const
       {
           /*
           if(abs(A.first - B.first) < numeric_limits<type>::min())
           {
              return true;
           }
           else
           {
              return false;
           }
           */
           return false;
       }

       /// Returns true if the interval(A,B) is constant,
       /// and false otherwise.

       inline bool is_constant() const
       {
           /*if(abs(A.second - B.second) < numeric_limits<type>::min())
           {
              return true;
           }
           else
           {
              return false;
           }*/
           return false;
       }

       /// Writes a string with the values of A, U and B.

       inline string struct_to_string() const
       {
           ostringstream buffer;

           buffer << "A = (" << A.first << "," << A.second << ")\n"
                  << "U = (" << U.first << "," << U.second << ")\n"
                  << "B = (" << B.first << "," << B.second << ")" << endl;

           return buffer.str();
       }

       /// Prints the triplet points to the standard output.

       inline void print() const
       {
           cout << struct_to_string();
           cout << "Lenght: " << get_length() << endl;
       }

       /// Checks that the points A, U and B define a minimum.
       /// That is, a < u < b, fa > fu and fu < fb.
       /// If some of that conditions is not satisfied, an exception is thrown.

       inline void check() const
       {
           ostringstream buffer;

           if(U.first < A.first)
           {
              buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                     << "void check() const method.\n"
                     << "U is less than A:\n"
                     << struct_to_string();

              throw logic_error(buffer.str());
           }

           if(U.first > B.first)
           {
              buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                     << "void check() const method.\n"
                     << "U is greater than A:\n"
                     << struct_to_string();

              throw logic_error(buffer.str());
           }

           if(U.second >= A.second)
           {
              buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                     << "void check() const method.\n"
                     << "fU is equal or greater than fA:\n"
                     << struct_to_string();

              throw logic_error(buffer.str());
           }

           if(U.second >= B.second)
           {
              buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                     << "void check() const method.\n"
                     << "fU is equal or greater than fB:\n"
                     << struct_to_string();

              throw logic_error(buffer.str());
           }
       }

       /// Left point of the triplet.

       pair<type, type> A;

       /// Interior point of the triplet.

       pair<type, type> U;

       /// Right point of the triplet.

       pair<type, type> B;
   };

   // Get methods

   LossIndex* get_loss_index_pointer() const;

   bool has_loss_index() const;

   // Training operators

   const LearningRateMethod& get_learning_rate_method() const;
   string write_learning_rate_method() const;

   // Training parameters

   const type& get_learning_rate_tolerance() const;

   // Utilities
   
   const bool& get_display() const;
  
   // Set methods

   void set();
   void set(LossIndex*);

   void set_loss_index_pointer(LossIndex*);
   void set_threads_number(const int&);

   // Training operators

   void set_learning_rate_method(const LearningRateMethod&);
   void set_learning_rate_method(const string&);

   // Training parameters

   void set_learning_rate_tolerance(const type&);

   // Utilities

   void set_display(const bool&);

   void set_default();

   // Learning rate methods

   type calculate_golden_section_learning_rate(const Triplet&) const;
   type calculate_Brent_method_learning_rate(const Triplet&) const;

   Triplet calculate_bracketing_triplet(const DataSet::Batch&,
                                        NeuralNetwork::ForwardPropagation&,
                                        LossIndex::BackPropagation&,
                                        OptimizationAlgorithm::OptimizationData&) const;

   pair<type, type> calculate_directional_point(const DataSet::Batch&,
                                                NeuralNetwork::ForwardPropagation&,
                                                LossIndex::BackPropagation&,
                                                OptimizationAlgorithm::OptimizationData&) const;

   // Serialization methods

      
   void from_XML(const tinyxml2::XMLDocument&);   

   void write_XML(tinyxml2::XMLPrinter&) const;

protected:

   // FIELDS

   /// Pointer to an external loss index object.

   LossIndex* loss_index_pointer = nullptr;

   // TRAINING OPERATORS

   /// Variable containing the actual method used to obtain a suitable perform_training rate. 

   LearningRateMethod learning_rate_method;

   /// Maximum interval length for the learning rate.

   type learning_rate_tolerance = static_cast<type>(1.0e-3);

   type loss_tolerance = static_cast<type>(1.0e-3);

   // UTILITIES

   /// Display messages to screen.

   bool display = true;

   const type golden_ratio = static_cast<type>(1.618);

   NonBlockingThreadPool* non_blocking_thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device = nullptr;

   bool is_zero(const Tensor<type, 1>& tensor) const
   {
       const Index size = tensor.size();

       for(Index i = 0; i < size; i++)
       {
           if(abs(tensor[i]) > numeric_limits<type>::min()) return false;
       }

       return true;
   }
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
