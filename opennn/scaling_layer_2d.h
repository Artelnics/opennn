//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef SCALINGLAYER2D_H
#define SCALINGLAYER2D_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "scaling.h"
#include "layer.h"
#include "layer_forward_propagation.h"

namespace opennn
{

/// This class represents a layer of scaling neurons.
/// Scaling layers are included in the definition of a neural network.
/// They are used to normalize variables so they are in an appropriate range for computer processing.

class ScalingLayer2D : public Layer
{

public:

   // Constructors

   explicit ScalingLayer2D();

   explicit ScalingLayer2D(const Index&);
   explicit ScalingLayer2D(const Tensor<Index, 1>&);

   explicit ScalingLayer2D(const Tensor<Descriptives, 1>&);

   // Get methods


   Tensor<Index, 1> get_outputs_dimensions() const;

   Index get_inputs_number() const final;
   Tensor<Index, 1> get_inputs_dimensions() const;
   Index get_neurons_number() const final;

   // Inputs descriptives

   Tensor<Descriptives, 1> get_descriptives() const;
   Descriptives get_descriptives(const Index&) const;

   Tensor<type, 1> get_minimums() const;
   Tensor<type, 1> get_maximums() const;
   Tensor<type, 1> get_means() const;
   Tensor<type, 1> get_standard_deviations() const;

   // Variables scaling and unscaling

   Tensor<Scaler, 1> get_scaling_methods() const;

   Tensor<string, 1> write_scalers() const;
   Tensor<string, 1> write_scalers_text() const;

   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&);
   void set(const Tensor<Index, 1>&);
   void set(const Tensor<Descriptives, 1>&);
   void set(const Tensor<Descriptives, 1>&, const Tensor<Scaler, 1>&);
   void set(const tinyxml2::XMLDocument&);

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;

   void set_default();

   // Descriptives

   void set_descriptives(const Tensor<Descriptives, 1>&);
   void set_item_descriptives(const Index&, const Descriptives&);

   void set_minimum(const Index&, const type&);
   void set_maximum(const Index&, const type&);
   void set_mean(const Index&, const type&);
   void set_standard_deviation(const Index&, const type&);

   void set_min_max_range(const type& min, const type& max);

   // Scaling method

   void set_scalers(const Tensor<Scaler, 1>&);
   void set_scalers(const Tensor<string, 1>&);

   void set_scaler(const Index&, const Scaler&);
   void set_scalers(const Scaler&);
   void set_scalers(const string&);

   // Display messages

   void set_display(const bool&);

   // Check methods

   bool is_empty() const;

   void check_range(const Tensor<type, 1>&) const;

   void forward_propagate(const pair<type*, dimensions>&, LayerForwardPropagation*, const bool&) final;

   // Expression methods

   string write_no_scaling_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_minimum_maximum_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_mean_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_standard_deviation_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const;

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;

   // Serialization methods

   void print() const;

   virtual void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   Tensor<Index, 1> inputs_dimensions;

   /// Descriptives of input variables.

   Tensor<Descriptives, 1> descriptives;

   /// Vector of scaling methods for each variable.

   Tensor<Scaler, 1> scalers;

   /// Min and max range for minmaxscaling

   type min_range;
   type max_range;

   /// Display warning messages to screen.

   bool display = true;

};

struct ScalingLayer2DForwardPropagation : LayerForwardPropagation
{
    // Constructor

    explicit ScalingLayer2DForwardPropagation() : LayerForwardPropagation()
    {
    }


    virtual ~ScalingLayer2DForwardPropagation()
    {
    }

    // Constructor

    explicit ScalingLayer2DForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    pair<type*, dimensions> get_outputs() const final
    {
        const Index neurons_number = layer_pointer->get_neurons_number();

        return pair<type*, dimensions>(outputs_data, {{batch_samples_number, neurons_number}});
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer) final
    {
        layer_pointer = new_layer_pointer;

        const Index neurons_number = layer_pointer->get_neurons_number();

        batch_samples_number = new_batch_samples_number;

        outputs.resize(batch_samples_number, neurons_number);

        outputs_data = outputs.data();
    }


    void print() const
    {
        cout << "Outputs:" << endl;
        cout << outputs << endl;
    }

    Tensor<type, 2> outputs;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
