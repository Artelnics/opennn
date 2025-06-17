//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LEARNINGRATEALGORITHM_H
#define LEARNINGRATEALGORITHM_H

#include "tinyxml2.h"

using namespace tinyxml2;

namespace opennn
{

class LossIndex;
class OptimizationAlgorithm;

struct Batch;
struct ForwardPropagation;
struct BackPropagation;
struct OptimizationAlgorithmData;

class LearningRateAlgorithm
{

public:

   enum class LearningRateMethod{GoldenSection, BrentMethod};

   LearningRateAlgorithm(LossIndex* = nullptr);

   ~LearningRateAlgorithm()
   {
       if(thread_pool != nullptr)
           thread_pool.reset();
       if(thread_pool_device != nullptr)
           thread_pool_device.reset();
   }

   struct Triplet
   {
       Triplet();

       bool operator == (const Triplet& other_triplet) const
       {
           return (A == other_triplet.A && U == other_triplet.U && B == other_triplet.B);
       }

       type get_length() const;

       pair<type, type> minimum() const;

       string struct_to_string() const;

       void print() const;

       void check() const;

       pair<type, type> A;

       pair<type, type> U;

       pair<type, type> B;
   };

   LossIndex* get_loss_index() const;

   bool has_loss_index() const;

   const LearningRateMethod& get_learning_rate_method() const;
   string write_learning_rate_method() const;

   const type& get_learning_rate_tolerance() const;
   
   const bool& get_display() const;
  
   void set(LossIndex* = nullptr);

   void set_loss_index(LossIndex*);
   void set_threads_number(const int&);

   void set_learning_rate_method(const LearningRateMethod&);
   void set_learning_rate_method(const string&);

   void set_learning_rate_tolerance(const type&);

   void set_display(const bool&);

   void set_default();

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
      
   void from_XML(const XMLDocument&);   

   void to_XML(XMLPrinter&) const;

private:

   LossIndex* loss_index = nullptr;

   LearningRateMethod learning_rate_method;

   type learning_rate_tolerance;

   type loss_tolerance;

   bool display = true;

   const type golden_ratio = type(1.618);

   unique_ptr<ThreadPool> thread_pool = nullptr;
   unique_ptr<ThreadPoolDevice> thread_pool_device = nullptr;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
