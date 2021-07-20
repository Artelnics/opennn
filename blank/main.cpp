//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R O S E N B R O C K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include "iostream"
#include <stdio.h>
#include <vector>

using namespace std;

class LSTMNetwork
{

public:

    LSTMNetwork()
    {
        hidden_states.resize(3);
        cell_states.resize(3);
    }

    vector<vector<float>> neural_network_batch(const vector<vector<float>>& inputs)
    {
        vector<vector<float>> outputs(inputs.size());

        for(size_t i; i < inputs.size(); i++)
        {
            outputs[i] = neural_network(inputs[i]);
        }

        return outputs;
    }


private:

    vector<float> hidden_states;
    vector<float> cell_states;


vector<float> scaling_layer(const vector<float>& inputs)
{
    vector<float> outputs(4);

    outputs[0] = (inputs[0]-0.0004061360087)/1.249300003;
    outputs[1] = (inputs[1]-0.0005194490077)/1.868620038;
    outputs[2] = (inputs[2]-0.0003262569953)/1.24921;
    outputs[3] = (inputs[3]-0.0006226960104)/1.868729949;

    return outputs;
}

vector<float> long_short_term_memory_layer(const vector<float>& inputs)
{
    vector<float> forget_gate_combinations(3);

    forget_gate_combinations[0] = 0.00662842 +  inputs[0] * (-0.0356323) +  inputs[0] * (-0.13208) +  inputs[0] * (0.0435913) +  inputs[0] * (0) + 0 * (0.0859009) + 0 * (-0.0552612) + 0 * (-0.100378);
    forget_gate_combinations[1] = -0.0422485 +  inputs[1] * (-0.138525) +  inputs[1] * (-0.0629028) +  inputs[1] * (-0.112109) +  inputs[1] * (0) + 0 * (0.0526734) + 0 * (0.0942383) + 0 * (-0.024585);
    forget_gate_combinations[2] = 0.107666 +  inputs[2] * (-0.125842) +  inputs[2] * (0.0830444) +  inputs[2] * (0.154199) +  inputs[2] * (-8.40488e+33) + 0 * (-0.107385) + 0 * (-0.136914) + 0 * (-0.0520996);

    vector<float> forget_gate_activations(3);

    forget_gate_activations[0] = 1.0/(1.0 + exp(-forget_gate_combinations[0]));
    forget_gate_activations[1] = 1.0/(1.0 + exp(-forget_gate_combinations[1]));
    forget_gate_activations[2] = 1.0/(1.0 + exp(-forget_gate_combinations[2]));

    vector<float> input_gate_combinations(3);

    input_gate_combinations[0] = -0.0872925 + inputs[0] * (-0.0521607) + inputs[0] * (-0.169617) + inputs[0] * (0.112903) + inputs[0] * (0) + 0 * (0.162695) + 0 * (0.130347) + 0 * (0.0553101);
    input_gate_combinations[1] = -0.0607178 + inputs[1] * (-0.0802368) + inputs[1] * (-0.125732) + inputs[1] * (-0.0570923) + inputs[1] * (0) + 0 * (-0.073938) + 0 * (-0.0362061) + 0 * (-0.16853);
    input_gate_combinations[2] = -0.0837524 + inputs[2] * (-0.0556519) + inputs[2] * (-0.0436279) + inputs[2] * (-0.153601) + inputs[2] * (-9.86522e+33) + 0 * (-0.000769049) + 0 * (0.146973) + 0 * (-0.0249512);

    vector<float> input_gate_activations(3);

    input_gate_activations[0] = 1.0/(1.0 + exp(-input_gate_combinations[0]));
    input_gate_activations[1] = 1.0/(1.0 + exp(-input_gate_combinations[1]));
    input_gate_activations[2] = 1.0/(1.0 + exp(-input_gate_combinations[2]));

    vector<float> state_gate_combinations(3);

    state_gate_combinations[0] = 0.0697388 + inputs[0] * (0.0526489) + inputs[0] * (0.103845) + inputs[0] * (0.189282) + inputs[0] * (0) + 0 * (-0.198645) + 0 * (0.175232) + 0 * (0.0690918);
    state_gate_combinations[1] = -0.00729981 + inputs[1] * (-0.0740112) + inputs[1] * (0.19845) + inputs[1] * (0.145337) + inputs[1] * (0) + 0 * (0.131519) + 0 * (-0.0788574) + 0 * (-0.11864);
    state_gate_combinations[2] = 0.16394 + inputs[2] * (0.0973389) + inputs[2] * (0.171216) + inputs[2] * (-0.146985) + inputs[2] * (-7.9181e+33) + 0 * (0.172986) + 0 * (0.0276123) + 0 * (0.176514);

    vector<float> state_gate_activations(3);

    state_gate_activations[0] = 1.0/(1.0 + exp(-state_gate_combinations[0]));
    state_gate_activations[1] = 1.0/(1.0 + exp(-state_gate_combinations[1]));
    state_gate_activations[2] = 1.0/(1.0 + exp(-state_gate_combinations[2]));

    vector<float> output_gate_combinations(3);

    output_gate_combinations[0] = -0.0415039 + inputs[0] * (-0.105566) + inputs[0] * (-0.00318603) + inputs[0] * (0.14303) + inputs[0] * (0) + 0 * (0.0691894) + 0 * (-0.0498169) + 0 * (-0.0579834);
    output_gate_combinations[1] = -0.136121 + inputs[1] * (0.0291504) + inputs[1] * (0.0518799) + inputs[1] * (0.173914) + inputs[1] * (0) + 0 * (-0.0653564) + 0 * (-0.0651855) + 0 * (-0.0469116);
    output_gate_combinations[2] = -0.16095 + inputs[2] * (-0.10166) + inputs[2] * (-0.102417) + inputs[2] * (-0.131104) + inputs[2] * (-7.10687e+33) + 0 * (-0.126001) + 0 * (-0.146362) + 0 * (-0.140942);

    vector<float> output_gate_activations(3);

    output_gate_activations[0] = 1.0/(1.0 + exp(-output_gate_combinations[0]));
    output_gate_activations[1] = 1.0/(1.0 + exp(-output_gate_combinations[1]));
    output_gate_activations[2] = 1.0/(1.0 + exp(-output_gate_combinations[2]));

    cell_states[0] = forget_gate_activations[0] * cell_states[0] + input_gate_activations[0] * state_gate_activations[0];
    cell_states[1] = forget_gate_activations[1] * cell_states[1] + input_gate_activations[1] * state_gate_activations[1];
    cell_states[2] = forget_gate_activations[2] * cell_states[2] + input_gate_activations[2] * state_gate_activations[2];

    vector<float> cell_state_activations(3);

    cell_state_activations[0] = 1.0/(1.0 + exp(-cell_states[0]));
    cell_state_activations[1] = 1.0/(1.0 + exp(-cell_states[1]));
    cell_state_activations[2] = 1.0/(1.0 + exp(-cell_states[2]));

    hidden_states[0] = output_gate_activations[0] * cell_state_activations[0];
    hidden_states[1] = output_gate_activations[1] * cell_state_activations[1];
    hidden_states[2] = output_gate_activations[2] * cell_state_activations[2];

    vector<float> long_short_term_memory_output(3);

    long_short_term_memory_output[0] = hidden_states[0];
    long_short_term_memory_output[1] = hidden_states[1];
    long_short_term_memory_output[2] = hidden_states[2];

    return long_short_term_memory_output;
}

vector<float> perceptron_layer_1(const vector<float>& inputs)
{
    vector<float> combinations(2);

    combinations[0] = -0.179761 +0.0939819*inputs[0] -0.12627*inputs[1] -0.13501*inputs[2];
    combinations[1] = -0.190613 +0.00655517*inputs[0] -0.0846436*inputs[1] +0.149927*inputs[2];

    vector<float> activations(2);

    activations[0] = combinations[0];
    activations[1] = combinations[1];

    return activations;
}

vector<float> unscaling_layer(const vector<float>& inputs)
{
    vector<float> outputs(2);

    outputs[0] = inputs[0]*1.249029994+0.0001127939977;
    outputs[1] = inputs[1]*1.86868+0.0005692829727;

    return outputs;
}

vector<float> bounding_layer(const vector<float>& inputs)
{
    vector<float> outputs(2);

    outputs[0] = inputs[0];
    outputs[1] = inputs[1];

    return outputs;
}

vector<float> neural_network(const vector<float>& inputs)
{
    vector<float> outputs;

    outputs = scaling_layer(inputs);
    outputs = long_short_term_memory_layer(outputs);
    outputs = perceptron_layer_1(outputs);
    outputs = unscaling_layer(outputs);
    outputs = bounding_layer(outputs);

    return outputs;
}

};


int main()
{
    LSTMNetwork lstmNetwork;

    vector<vector<float>> inputs(2);
    inputs[0] = {5,3,3,1};
    inputs[1] = {1,2,3,4};

    cout << "inputs: " << endl;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        for(size_t j = 0; j < inputs[i].size(); j++)
        {
            cout << inputs[i][j] << "," ;
        }
        cout << endl;
    }

    vector<vector<float>> outputs = lstmNetwork.neural_network_batch(inputs);

    cout << "outputs: " << endl;

    for(size_t i = 0; i < outputs.size(); i++)
    {
        for(size_t j = 0; j < outputs[i].size(); j++)
        {
            cout << outputs[i][j] << ",";
        }
        cout << endl;
    }

    return 0;

}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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
