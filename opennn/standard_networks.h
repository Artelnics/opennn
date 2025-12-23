//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   STANDARD   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef STANDARDNETWORKS_H
#define STANDARDNETWORKS_H

#include "neural_network.h"

namespace opennn
{

class ApproximationNetwork : public NeuralNetwork
{

public:

    ApproximationNetwork(const dimensions& input_dimensions,
                         const dimensions& complexity_dimensions,
                         const dimensions& output_dimensions);
};


class ClassificationNetwork : public NeuralNetwork
{

public:

    ClassificationNetwork(const dimensions& input_dimensions,
                          const dimensions& complexity_dimensions,
                          const dimensions& output_dimensions);
};


class ForecastingNetwork : public NeuralNetwork
{

public:

    ForecastingNetwork(const dimensions& input_dimensions,
                       const dimensions& complexity_dimensions,
                       const dimensions& output_dimensions);
};


class AutoAssociationNetwork : public NeuralNetwork
{

public:

    AutoAssociationNetwork(const dimensions& input_dimensions,
                           const dimensions& complexity_dimensions,
                           const dimensions& output_dimensions);
};


class ImageClassificationNetwork : public NeuralNetwork
{

public:

    ImageClassificationNetwork(const dimensions& input_dimensions,
                               const dimensions& complexity_dimensions,
                               const dimensions& output_dimensions);
};


class SimpleResNet : public NeuralNetwork
{

public:

    SimpleResNet(const dimensions& input_dimensions,
                 const std::vector<Index>& blocks_per_stage,
                 const dimensions& initial_filters,
                 const dimensions& output_dimensions);

    void print_dim(const dimensions& dims) const;
};


class TextClassificationNetwork : public NeuralNetwork
{

public:

    TextClassificationNetwork(const dimensions& input_dimensions,
                              const dimensions& complexity_dimensions,
                              const dimensions& output_dimensions,
                              const vector<string>& new_input_vocabulary = vector<string>());

    Tensor<type, 2> calculate_outputs(const Tensor<string, 1>& input_documents) const;

private:

    vector<string> input_vocabulary;
};

} // namespace opennn

#endif // STANDARDNETWORKS_H

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
