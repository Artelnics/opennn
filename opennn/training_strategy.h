//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R           
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRAININGSTRATEGY_H
#define TRAININGSTRATEGY_H

#include "loss_index.h"
#include "optimization_algorithm.h"

namespace opennn
{

//class NeuralNetwork;
class LossIndex;
class OptimizationAlgorithm;

struct TrainingResults;

class TrainingStrategy
{

public:

    TrainingStrategy(const NeuralNetwork* = nullptr, const Dataset* = nullptr);

    Dataset* get_dataset();
    NeuralNetwork* get_neural_network() const;

    LossIndex* get_loss_index() const;
    OptimizationAlgorithm* get_optimization_algorithm() const;

    bool has_neural_network() const;
    bool has_dataset() const;

    // Set

    void set(const NeuralNetwork* = nullptr, const Dataset* = nullptr);
    void set_default();

    void set_dataset(const Dataset*);
    void set_neural_network(const NeuralNetwork*);

    void set_loss_index(const string&);
    void set_optimization_algorithm(const string&);

    TrainingResults perform_training();

#ifdef OPENNN_CUDA
    TrainingResults perform_training_cuda();
#endif

    // Check

    void fix_forecasting();

    // Serialization

    void print() const;

    void from_XML(const XMLDocument&);
    void to_XML(XMLPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    Dataset* dataset = nullptr;

    NeuralNetwork* neural_network = nullptr;

    unique_ptr<LossIndex> loss_index;

    unique_ptr<OptimizationAlgorithm> optimization_algorithm;
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
