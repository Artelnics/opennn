// Numerical parity test — MASTER branch version
#include "../opennn/pch.h"
#include "../opennn/dataset.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"
#include "../opennn/dense_layer.h"
#include "../opennn/mean_squared_error.h"
#include "../opennn/scaling_layer_2d.h"

#include <fstream>
#include <iomanip>
#include <random>

using namespace opennn;
using namespace std;

int main()
{
    ofstream out("master_outputs.txt");
    out << fixed << setprecision(10);

    // Test 1: Single Dense layer, constant params, Linear
    {
        const Index samples = 4, inputs_n = 3, outputs_n = 2;

        NeuralNetwork network;
        network.add_layer(make_unique<Dense2d>(dimensions({inputs_n}), dimensions({outputs_n}), "Linear"));

        Tensor<type, 1> params(network.get_parameters_number());
        params.setConstant(type(0.1));
        network.set_parameters(params);

        Tensor<type, 2> input_data(samples, inputs_n);
        for(Index i = 0; i < samples; i++)
            for(Index j = 0; j < inputs_n; j++)
                input_data(i, j) = type((i * inputs_n + j) + 1);

        Tensor<type, 2> outputs = network.calculate_outputs<2, 2>(input_data);

        out << "# Test1_Dense_Constant" << endl;
        out << "layers " << network.get_layers_number() << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.dimension(0); i++)
            for(Index j = 0; j < outputs.dimension(1); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    // Test 2: Single Dense layer, Logistic/Sigmoid
    {
        const Index samples = 4, inputs_n = 2, outputs_n = 2;

        NeuralNetwork network;
        network.add_layer(make_unique<Dense2d>(dimensions({inputs_n}), dimensions({outputs_n}), "Logistic"));

        Tensor<type, 1> params(network.get_parameters_number());
        params.setConstant(type(0.5));
        network.set_parameters(params);

        Tensor<type, 2> input_data(samples, inputs_n);
        input_data.setValues({{0.1f, 0.2f}, {0.5f, 0.6f}, {-0.3f, 0.8f}, {1.0f, -0.5f}});

        Tensor<type, 2> outputs = network.calculate_outputs<2, 2>(input_data);

        out << "# Test2_Dense_Sigmoid" << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.dimension(0); i++)
            for(Index j = 0; j < outputs.dimension(1); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    // Test 3: Two Dense layers (ReLU + Linear)
    {
        const Index samples = 3, inputs_n = 2, hidden_n = 3, outputs_n = 1;

        NeuralNetwork network;
        network.add_layer(make_unique<Dense2d>(dimensions({inputs_n}), dimensions({hidden_n}), "RectifiedLinear"));
        network.add_layer(make_unique<Dense2d>(dimensions({hidden_n}), dimensions({outputs_n}), "Linear"));

        Tensor<type, 1> params(network.get_parameters_number());
        params.setConstant(type(0.2));
        network.set_parameters(params);

        Tensor<type, 2> input_data(samples, inputs_n);
        input_data.setValues({{1.0f, 2.0f}, {-1.0f, 0.5f}, {0.0f, 0.0f}});

        Tensor<type, 2> outputs = network.calculate_outputs<2, 2>(input_data);

        out << "# Test3_TwoDense_ReLU_Linear" << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.dimension(0); i++)
            for(Index j = 0; j < outputs.dimension(1); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    // Test 4: MSE error
    {
        const Index samples = 6, inputs_n = 2, outputs_n = 1;

        Dataset dataset(samples, {inputs_n}, {outputs_n});
        Tensor<type, 2> data(samples, inputs_n + outputs_n);
        data.setValues({
            {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f},
            {-1.0f, -2.0f, -3.0f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 1.5f, 0.0f}
        });
        dataset.set_data(data);
        dataset.set_sample_uses("Training");

        NeuralNetwork network;
        network.add_layer(make_unique<Dense2d>(dimensions({inputs_n}), dimensions({outputs_n}), "Linear"));

        Tensor<type, 1> params(network.get_parameters_number());
        params.setConstant(type(0.1));
        network.set_parameters(params);

        MeanSquaredError mse(&network, &dataset);
        const type error = mse.calculate_numerical_error();

        out << "# Test4_MSE_Error" << endl;
        out << "error " << error << endl;
    }

    // Test 5: Scaling layer with real descriptives from dataset
    {
        const Index samples = 6, inputs_n = 2, outputs_n = 1;

        Dataset dataset(samples, {inputs_n}, {outputs_n});
        Tensor<type, 2> data(samples, inputs_n + outputs_n);
        data.setValues({
            {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f},
            {-1.0f, -2.0f, -3.0f}, {0.5f, 0.5f, 0.5f}, {-0.5f, 1.5f, 0.0f}
        });
        dataset.set_data(data);
        dataset.set_sample_uses("Training");

        // Build Scaling + Dense manually, set descriptives from data
        NeuralNetwork network;
        network.add_layer(make_unique<Scaling2d>(dimensions({inputs_n})));

        // Set descriptives from the dataset
        Scaling2d* scaling = static_cast<Scaling2d*>(network.get_layer(0).get());
        scaling->set_descriptives(dataset.calculate_variable_descriptives("Input"));
        scaling->set_scalers("MeanStandardDeviation");

        network.add_layer(make_unique<Dense2d>(dimensions({inputs_n}), dimensions({outputs_n}), "Linear"));

        Tensor<type, 1> params(network.get_parameters_number());
        params.setConstant(type(0.1));
        network.set_parameters(params);

        Tensor<type, 2> input_data(samples, inputs_n);
        for(Index i = 0; i < samples; i++)
            for(Index j = 0; j < inputs_n; j++)
                input_data(i, j) = data(i, j);

        Tensor<type, 2> outputs = network.calculate_outputs<2, 2>(input_data);

        out << "# Test5_Scaling_Descriptives" << endl;
        out << "params " << network.get_parameters_number() << endl;
        for(Index i = 0; i < outputs.dimension(0); i++)
            for(Index j = 0; j < outputs.dimension(1); j++)
                out << "output " << i << " " << j << " " << outputs(i, j) << endl;
    }

    out.close();
    cout << "Master outputs written to master_outputs.txt" << endl;
    return 0;
}
