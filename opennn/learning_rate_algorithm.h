//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEARNINGRATEALGORITHM_H
#define LEARNINGRATEALGORITHM_H

#include <iostream>
#include <limits>
#include <cmath>
#include <cstdlib>

#include "config.h"
#include "neural_network.h"
#include "loss_index.h"
#include "optimization_algorithm.h"
#include "statistics.h"

namespace opennn
{

class LearningRateAlgorithm
{

public:

   enum class LearningRateMethod{GoldenSection, BrentMethod};

   explicit LearningRateAlgorithm(LossIndex* = nullptr);

   virtual ~LearningRateAlgorithm();

   struct Triplet
   {
       Triplet()
       {
           A = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
           U = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
           B = make_pair(numeric_limits<type>::max(), numeric_limits<type>::max());
       }

       inline bool operator == (const Triplet& other_triplet) const
       {
          if(A == other_triplet.A
          && U == other_triplet.U
          && B == other_triplet.B)
             return true;
          else
             return false;
       }

       inline type get_length() const
       {
           return abs(B.first - A.first);
       }


       inline pair<type, type> minimum() const
       {
           Tensor<type, 1> losses(3);

           losses.setValues({A.second, U.second, B.second});

           const Index minimal_index = opennn::minimal_index(losses);

           if(minimal_index == 0) return A;
           else if(minimal_index == 1) return U;
           else return B;
       }


       inline string struct_to_string() const
       {
           ostringstream buffer;

           buffer << "A = (" << A.first << "," << A.second << ")\n"
                  << "U = (" << U.first << "," << U.second << ")\n"
                  << "B = (" << B.first << "," << B.second << ")" << endl;

           return buffer.str();
       }


       inline void print() const
       {
           cout << struct_to_string();
           cout << "Lenght: " << get_length() << endl;
       }


       inline void check() const
       {
           ostringstream buffer;

           if(U.first < A.first)
              throw runtime_error("U is less than A:\n" + struct_to_string());

           if(U.first > B.first)
              throw runtime_error("U is greater than B:\n" + struct_to_string());

           if(U.second >= A.second)
              throw runtime_error("fU is equal or greater than fA:\n" + struct_to_string());

           if(U.second >= B.second)
              throw runtime_error("fU is equal or greater than fB:\n" + struct_to_string());
       }

       pair<type, type> A;

       pair<type, type> U;

       pair<type, type> B;
   };

   // Get

   LossIndex* get_loss_index() const;

   bool has_loss_index() const;

   // Training operators

   const LearningRateMethod& get_learning_rate_method() const;
   string write_learning_rate_method() const;

   // Training parameters

   const type& get_learning_rate_tolerance() const;

   // Utilities
   
   const bool& get_display() const;
  
   // Set

   void set(LossIndex* = nullptr);

   void set_loss_index(LossIndex*);
   void set_threads_number(const int&);

   // Training operators

   void set_learning_rate_method(const LearningRateMethod&);
   void set_learning_rate_method(const string&);

   // Training parameters

   void set_learning_rate_tolerance(const type&);

   // Utilities

   void set_display(const bool&);

   void set_default();

   // Learning rate

   type calculate_golden_section_learning_rate(const Triplet&) const;
   type calculate_Brent_method_learning_rate(const Triplet&) const;

   Triplet calculate_bracketing_triplet(const Batch&,
                                        ForwardPropagation&,
                                        BackPropagation&,
                                        OptimizationAlgorithmData&) const;

   pair<type, type> calculate_directional_point(const Batch&,
                                                ForwardPropagation&,
                                                BackPropagation&,
                                                OptimizationAlgorithmData&) const;

   // Serialization
      
   void from_XML(const tinyxml2::XMLDocument&);   

   void to_XML(tinyxml2::XMLPrinter&) const;

protected:

   // FIELDS

   LossIndex* loss_index = nullptr;

   // TRAINING OPERATORS

   LearningRateMethod learning_rate_method;

   type learning_rate_tolerance;

   type loss_tolerance;

   // UTILITIES

   bool display = true;

   const type golden_ratio = type(1.618);

   ThreadPool* thread_pool = nullptr;
   ThreadPoolDevice* thread_pool_device = nullptr;
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
