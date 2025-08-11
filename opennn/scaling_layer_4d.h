//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   4 D   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef Scaling4d_H
#define Scaling4d_H

#include "layer.h"

namespace opennn
{

class Scaling4d : public Layer
{

public:

    Scaling4d(const dimensions& = {0, 0, 0, 0});

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    void set(const dimensions& = { 0, 0, 0, 0 });

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

    void forward_propagate_cuda(const vector<float*>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                const bool&) override;

#endif

private:

    dimensions input_dimensions;

    type min_range;
    type max_range;

};


struct Scaling4dForwardPropagation : LayerForwardPropagation
{
    Scaling4dForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_output_pair() const override;

    void set(const Index& = 0, Layer* = nullptr) override;

    void print() const override;

    Tensor<type, 4> outputs;
};

#ifdef OPENNN_CUDA

struct Scaling4dForwardPropagationCuda : public LayerForwardPropagationCuda
{
    Scaling4dForwardPropagationCuda(const Index & = 0, Layer* = nullptr);

    void set(const Index & = 0, Layer* = nullptr);

    void print() const override;

    void free() override;

    type* scalar_device = nullptr;
};

#endif

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
