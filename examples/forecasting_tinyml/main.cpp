//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R E C A S T I N G   T I N Y M L   E X P O R T   G E N E R A T O R
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com
//
//   Builds small forecasting networks (LSTM and simple recurrent), exports
//   them to C (expression and CEmbedded backends) and writes reference
//   input/output vectors for the TinyML parity pipeline
//   (examples/forecasting_tinyml/run_forecasting_test.sh).
//
//   Training quality is irrelevant here: the networks keep their Glorot
//   initialization. What the pipeline verifies is that the exported C code
//   reproduces OpenNN outputs exactly on another target (emulated MCU).

#include "../../opennn/standard_networks.h"
#include "../../opennn/scaling_layer.h"
#include "../../opennn/unscaling_layer.h"
#include "../../opennn/model_expression.h"
#include "../../opennn/configuration.h"
#include "../../opennn/variable.h"

#include <cmath>
#include <fstream>
#include <iostream>

using namespace opennn;

static constexpr Index TIME_STEPS = 6;
static constexpr Index FEATURES = 2;
static constexpr Index HIDDEN = 6;
static constexpr Index REFERENCE_ROWS = 8;

static void configure_network(NeuralNetwork& network)
{
    network.set_input_variables(vector<Variable>(FEATURES));
    network.set_output_variables(vector<Variable>(1));

    auto* scaling = static_cast<Scaling*>(network.get_first(LayerType::Scaling));
    scaling->set_scalers("MeanStandardDeviation");
    scaling->set_descriptives({Descriptives(0.0f, 10.0f, 5.0f, 2.0f),
                               Descriptives(-1.0f, 1.0f, 0.0f, 0.5f)});

    auto* unscaling = static_cast<Unscaling*>(network.get_first(LayerType::Unscaling));
    unscaling->set_scalers("MeanStandardDeviation");
    unscaling->set_descriptives({Descriptives(0.0f, 10.0f, 3.0f, 1.5f)});
}

static void export_model(NeuralNetwork& network, const string& stem)
{
    const ModelExpression model_expression(&network);
    model_expression.save(stem + "_model.c", ModelExpression::ProgrammingLanguage::C);
    model_expression.save(stem + "_model_tables.c", ModelExpression::ProgrammingLanguage::CEmbedded);

    // Deterministic synthetic windows in the scale of the configured descriptives

    Tensor3 inputs(REFERENCE_ROWS, TIME_STEPS, FEATURES);

    for (Index row = 0; row < REFERENCE_ROWS; ++row)
        for (Index t = 0; t < TIME_STEPS; ++t)
        {
            inputs(row, t, 0) = 5.0f + 2.0f * sin(0.4f * float(t) + 0.9f * float(row));
            inputs(row, t, 1) = 0.5f * cos(0.7f * float(t) + 0.3f * float(row));
        }

    const MatrixR outputs = network.calculate_outputs(inputs);

    ofstream reference_file(stem + "_reference.csv");
    reference_file.precision(9);

    for (Index row = 0; row < REFERENCE_ROWS; ++row)
    {
        for (Index t = 0; t < TIME_STEPS; ++t)
            for (Index f = 0; f < FEATURES; ++f)
                reference_file << inputs(row, t, f) << ";";

        for (Index j = 0; j < outputs.cols(); ++j)
            reference_file << outputs(row, j) << (j + 1 < outputs.cols() ? ";" : "\n");
    }

    cout << "Exported " << stem << "_model.c, " << stem << "_model_tables.c and "
         << stem << "_reference.csv" << endl;
}

int main()
{
    try
    {
        cout << "OpenNN. Forecasting TinyML export generator." << endl;

        Configuration::instance().set(Device::CPU, Type::FP32);

        {
            ForecastingLstmNetwork network({TIME_STEPS, FEATURES}, {HIDDEN}, {1});
            configure_network(network);
            export_model(network, "lstm");
        }

        {
            ForecastingNetwork network({TIME_STEPS, FEATURES}, {HIDDEN}, {1});
            configure_network(network);
            export_model(network, "rnn");
        }

        return 0;
    }
    catch (const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
