//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M I N I M A L   I N F E R E N C E   E X A M P L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <memory>

#include "opennn/core/configuration.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/dense_layer.h"

int main()
{
    try
    {
        using namespace opennn;

        Configuration::instance().set(Device::CPU, Type::FP32);

        // Two inputs, one output, and fixed weights for a reproducible demo.
        // A trained application would learn or load these parameters instead.
        Network network;
        network.add_layer(std::make_unique<Dense>(Shape{2}, Shape{1}, "Identity"), {-1});
        network.compile();
        network.get_parameters_map().setConstant(0.5f);

        MatrixR inputs(1, 2);
        inputs << 1.0f, 2.0f;
        const MatrixR outputs = network.calculate_outputs(inputs);

        std::cout << "Prediction: " << outputs(0, 0) << '\n';
        return outputs.allFinite() ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
