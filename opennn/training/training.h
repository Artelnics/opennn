//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/training/loss.h"
#include "opennn/training/optimizer.h"

namespace opennn
{

class Training
{

public:

    explicit Training(Network* = nullptr, Dataset* = nullptr);

    const Dataset* get_dataset() const noexcept { return dataset; }
    Dataset* get_dataset() { return dataset; }

    const Network* get_network() const noexcept { return network; }
    Network* get_network() { return network; }

    const Loss* get_loss() const noexcept { return loss.get(); }
    Loss* get_loss() { return loss.get(); }

    const Optimizer* get_optimization_algorithm() const noexcept { return optimizer.get(); }
    Optimizer* get_optimization_algorithm() { return optimizer.get(); }
    void set(Network* = nullptr, Dataset* = nullptr);
    void set_default();

    void set_dataset(Dataset*);
    void set_network(Network*);

    void set_loss(const string&);
    void set_optimization_algorithm(const string&);

    TrainingResult train();
    void from_JSON(const JsonDocument&);
    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    Dataset* dataset = nullptr;

    Network* network = nullptr;

    unique_ptr<Loss> loss;

    unique_ptr<Optimizer> optimizer;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
