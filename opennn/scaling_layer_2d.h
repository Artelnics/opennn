//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYER2D_H
#define SCALINGLAYER2D_H

#include "layer.h"
#include "statistics.h"
#include "scaling.h"

namespace opennn
{

class Scaling2d : public Layer
{

public:

   Scaling2d(const dimensions& = {0});

   dimensions get_input_dimensions() const override;
   dimensions get_output_dimensions() const override;

   vector<Descriptives> get_descriptives() const;
   Descriptives get_descriptives(const Index&) const;

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;
   Tensor<type, 1> get_means() const;
   Tensor<type, 1> get_standard_deviations() const;

   vector<Scaler> get_scaling_methods() const;

   vector<string> write_scalers() const;
   vector<string> write_scalers_text() const;

   void set(const dimensions& = {0});

   void set_input_dimensions(const dimensions&) override;
   void set_output_dimensions(const dimensions&) override;

   void set_descriptives(const vector<Descriptives>&);

   void set_min_max_range(const type& min, const type& max);

   void set_scalers(const vector<Scaler>&);
   void set_scalers(const vector<string>&);
   void set_scalers(const Scaler&);
   void set_scalers(const string&);

   bool is_empty() const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) override;

   void calculate_outputs(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>& );

   string write_no_scaling_expression(const vector<string>&, const vector<string>&) const;

   string write_minimum_maximum_expression(const vector<string>&, const vector<string>&) const;

   string write_mean_standard_deviation_expression(const vector<string>&, const vector<string>&) const;

   string write_standard_deviation_expression(const vector<string>&, const vector<string>&) const;

   string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

   void print() const override;

   void from_XML(const XMLDocument&) override;
   void to_XML(XMLPrinter&) const override;

private:

   vector<Descriptives> descriptives;

   vector<Scaler> scalers;

   type min_range;
   type max_range;
};


struct Scaling2dForwardPropagation : LayerForwardPropagation
{
    Scaling2dForwardPropagation(const Index& = 0, Layer* = nullptr);
       
    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 2> outputs;
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
