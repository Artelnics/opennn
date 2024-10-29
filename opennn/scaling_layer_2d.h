//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYER2D_H
#define SCALINGLAYER2D_H

#include <iostream>
#include <string>

#include "scaling.h"
#include "layer.h"
#include "layer_forward_propagation.h"

namespace opennn
{

class ScalingLayer2D : public Layer
{

public:

   explicit ScalingLayer2D(const dimensions& = {0});

   dimensions get_output_dimensions() const;

   Index get_inputs_number() const final;
   dimensions get_input_dimensions() const;
   Index get_neurons_number() const final;

   Tensor<Descriptives, 1> get_descriptives() const;
   Descriptives get_descriptives(const Index&) const;

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;
   Tensor<type, 1> get_means() const;
   Tensor<type, 1> get_standard_deviations() const;

   Tensor<Scaler, 1> get_scaling_methods() const;

   Tensor<string, 1> write_scalers() const;
   Tensor<string, 1> write_scalers_text() const;

   const bool& get_display() const;

   void set(const dimensions& = {0});

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   void set_default();

   void set_descriptives(const Tensor<Descriptives, 1>&);
   void set_item_descriptives(const Index&, const Descriptives&);

   void set_minimum(const Index&, const type&);
   void set_maximum(const Index&, const type&);
   void set_mean(const Index&, const type&);
   void set_standard_deviation(const Index&, const type&);

   void set_min_max_range(const type& min, const type& max);

   void set_scalers(const Tensor<Scaler, 1>&);
   void set_scalers(const Tensor<string, 1>&);

   void set_scaler(const Index&, const Scaler&);
   void set_scaler(const Index&, const string&);
   void set_scalers(const Scaler&);
   void set_scalers(const string&);

   void set_display(const bool&);

   bool is_empty() const;

   void forward_propagate(const vector<pair<type*, dimensions>>&,
                          unique_ptr<LayerForwardPropagation>&,
                          const bool&) final;

   string write_no_scaling_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_minimum_maximum_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_mean_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   void print() const;

   virtual void from_XML(const tinyxml2::XMLDocument&) final;

   void to_XML(tinyxml2::XMLPrinter&) const final;

protected:

   dimensions input_dimensions;

   Tensor<Descriptives, 1> descriptives;

   Tensor<Scaler, 1> scalers;

   type min_range;
   type max_range;

   bool display = true;
};


struct ScalingLayer2DForwardPropagation : LayerForwardPropagation
{
    explicit ScalingLayer2DForwardPropagation(const Index& new_batch_samples_number = 0, Layer* new_layer = nullptr)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }
       
    pair<type*, dimensions> get_outputs_pair() const final;

    void set(const Index& = 0, Layer* = nullptr) final;

    void print() const
    {
        cout << "Outputs:" << endl
             << outputs << endl;
    }

    Tensor<type, 2> outputs;
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
