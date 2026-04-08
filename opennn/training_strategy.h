//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "loss.h"
#include "optimizer.h"

namespace opennn
{

class Loss;
class Optimizer;

struct TrainingResults;

class TrainingStrategy
{

public:

    TrainingStrategy(NeuralNetwork* = nullptr, Dataset* = nullptr);

    Dataset* get_dataset() const { return dataset; }
    NeuralNetwork* get_neural_network() const { return neural_network; }

    Loss* get_loss() const { return loss.get(); }
    Optimizer* get_optimization_algorithm() const { return optimizer.get(); }

    bool has_neural_network() const { return neural_network; }
    bool has_dataset() const { return dataset; }

    // Set

    void set(NeuralNetwork* = nullptr, Dataset* = nullptr);
    void set_default();

    void set_dataset(Dataset* ds) { dataset = ds; }
    void set_neural_network(NeuralNetwork* nn) { neural_network = nn; }

    void set_loss(const string&);
    void set_optimization_algorithm(const string&);

    TrainingResults train();

#ifdef CUDA
    TrainingResults train_cuda();
#endif

    // Serialization

    void from_XML(const XMLDocument&);
    void to_XML(XMLPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    void fix_forecasting();

    Dataset* dataset = nullptr;

    NeuralNetwork* neural_network = nullptr;

    unique_ptr<Loss> loss;

    unique_ptr<Optimizer> optimizer;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
