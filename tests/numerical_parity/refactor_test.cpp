// Numerical parity test — DEV-REFACTOR branch version
#include "../opennn/pch.h"
#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"
#include "../opennn/dense_layer.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/loss.h"
#include "../opennn/statistics.h"

#include <fstream>
#include <iomanip>
#include <random>

using namespace opennn;
using namespace std;

int main()
{
    ofstream out("refactor_outputs.txt");
    out << fixed << setprecision(10);

    // Test 1: Single Dense layer, constant params, Linear
    {
        const Index samples = 4, inputs_n = 3, outputs_n = 2;

        NeuralNetwork network;
        network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_n}, Shape{outputs_n}, "Linear"));
        network.compile();
        VectorMap(network.get_parameters_data(), network.get_parameters_size()).setConstant(type(0.1));

        MatrixR input_data(samples, inputs_n);
        for(Index i = 0; i < samples; i++)
            for(Index j = 0; j < inputs_n; j++)
                input_data(i, j) = type((i * inputs_n + j) + 1);

        MatrixR outputs = network.calculate_outputs(input_data);

        out << "# Test1_Dense_Constant" << endl;
        out << "layers " << network.get_layers_number() << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.rows(); i++)
            for(Index j = 0; j < outputs.cols(); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    // Test 2: Single Dense layer, Sigmoid
    {
        const Index samples = 4, inputs_n = 2, outputs_n = 2;

        NeuralNetwork network;
        network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_n}, Shape{outputs_n}, "Sigmoid"));
        network.compile();
        VectorMap(network.get_parameters_data(), network.get_parameters_size()).setConstant(type(0.5));

        MatrixR input_data(samples, inputs_n);
        input_data << 0.1f, 0.2f, 0.5f, 0.6f, -0.3f, 0.8f, 1.0f, -0.5f;

        MatrixR outputs = network.calculate_outputs(input_data);

        out << "# Test2_Dense_Sigmoid" << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.rows(); i++)
            for(Index j = 0; j < outputs.cols(); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    // Test 3: Two Dense layers (ReLU + Linear)
    {
        const Index samples = 3, inputs_n = 2, hidden_n = 3, outputs_n = 1;

        NeuralNetwork network;
        network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_n}, Shape{hidden_n}, "RectifiedLinear"));
        network.add_layer(make_unique<opennn::Dense<2>>(Shape{hidden_n}, Shape{outputs_n}, "Linear"));
        network.compile();
        VectorMap(network.get_parameters_data(), network.get_parameters_size()).setConstant(type(0.2));

        MatrixR input_data(samples, inputs_n);
        input_data << 1.0f, 2.0f, -1.0f, 0.5f, 0.0f, 0.0f;

        MatrixR outputs = network.calculate_outputs(input_data);

        out << "# Test3_TwoDense_ReLU_Linear" << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.rows(); i++)
            for(Index j = 0; j < outputs.cols(); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    // Test 4: MSE error
    {
        const Index samples = 6, inputs_n = 2, outputs_n = 1;

        Dataset dataset(samples, {inputs_n}, {outputs_n});
        MatrixR data(samples, inputs_n + outputs_n);
        data << 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
                -1.0f, -2.0f, -3.0f,
                0.5f, 0.5f, 0.5f,
                -0.5f, 1.5f, 0.0f;
        dataset.set_data(data);
        dataset.set_sample_roles("Training");

        NeuralNetwork network;
        network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_n}, Shape{outputs_n}, "Linear"));
        network.compile();
        VectorMap(network.get_parameters_data(), network.get_parameters_size()).setConstant(type(0.1));

        Loss loss(&network, &dataset);
        loss.set_error(Loss::Error::MeanSquaredError);
        const type error = loss.calculate_numerical_error();

        out << "# Test4_MSE_Error" << endl;
        out << "error " << error << endl;
    }

    // Test 5: Scaling layer with real descriptives from dataset
    {
        const Index samples = 6, inputs_n = 2, outputs_n = 1;

        Dataset dataset(samples, {inputs_n}, {outputs_n});
        MatrixR data(samples, inputs_n + outputs_n);
        data << 1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f,
                -1.0f, -2.0f, -3.0f,
                0.5f, 0.5f, 0.5f,
                -0.5f, 1.5f, 0.0f;
        dataset.set_data(data);
        dataset.set_sample_roles("Training");

        // Build Scaling + Dense, compile, then set descriptives
        NeuralNetwork network;
        network.add_layer(make_unique<Scaling<2>>(Shape{inputs_n}));
        network.add_layer(make_unique<opennn::Dense<2>>(Shape{inputs_n}, Shape{outputs_n}, "Linear"));
        network.compile();
        VectorMap(network.get_parameters_data(), network.get_parameters_size()).setConstant(type(0.1));

        // Set descriptives after compile (states are now linked)
        MatrixR input_data = dataset.get_feature_data("Input");
        vector<Descriptives> desc = descriptives(input_data);

        Scaling<2>* scaling = static_cast<Scaling<2>*>(network.get_layer(0).get());
        scaling->set_descriptives(desc);
        scaling->set_scalers("MeanStandardDeviation");

        MatrixR outputs = network.calculate_outputs(input_data);

        out << "# Test5_Scaling_Descriptives" << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.rows(); i++)
            for(Index j = 0; j < outputs.cols(); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    out.close();
    cout << "Refactor outputs written to refactor_outputs.txt" << endl;
    return 0;
}
