//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T O C H A S T I C   G R A D I E N T   D E S C E N T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "optimizer.h"

namespace opennn
{

struct BackPropagation;

class StochasticGradientDescent final : public Optimizer
{

public:

    enum DataSlot { ParameterUpdate, LastParameterUpdate };

    StochasticGradientDescent(Loss* = nullptr);

    void set_default();

    void set_batch_size(const Index);

    Index get_samples_number() const;

    void set_initial_learning_rate(const type);
    void set_initial_decay(const type);
    void set_momentum(const type);
    void set_nesterov(bool);

    void update_parameters(BackPropagation&, OptimizerData&, type) const;

    TrainingResults train() override;

    void from_XML(const XmlDocument&) override;

    void to_XML(XmlPrinter&) const override;

private:

    type initial_learning_rate;

    type initial_decay;

    type momentum = type(0);

    bool nesterov = false;

    Index batch_size = 1000;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
