// Artificial Intelligence Techniques SL	
// artelnics@artelnics.com	
// 
// Your model has been exported to this file.
// You can manage it with the 'neural network' method.	
// Example:
// 
// 	vector<float> sample(n);	
// 	sample[0] = 1;	
// 	sample[1] = 2;	
// 	sample[n] = 10;	
// 	vector<float> outputs = neural_network(sample);
// 
// Notice that only one sample is allowed as input. DataSetBatch of inputs are not yet implement,	
// however you can loop through neural network function in order to get multiple outputs.	

#include <vector>

using namespace std;

vector<float> scaling_layer(const vector<float>& inputs)
{
	vector<float> outputs(4);

	outputs[0] = (inputs[0]-5.843333244)/0.828066051;
	outputs[1] = (inputs[1]-3.057333231)/0.4358663261;
	outputs[2] = (inputs[2]-3.757999897)/1.765298247;
	outputs[3] = (inputs[3]-1.19933331)/0.762237668;

	return outputs;
}

vector<float> perceptron_layer_1(const vector<float>& inputs)
{
	vector<float> combinations(3);

	combinations[0] = 1.66413 +0.485914*inputs[0] +0.0471202*inputs[1] -1.99069*inputs[2] -1.24612*inputs[3];
	combinations[1] = -0.40017 +0.911162*inputs[0] -1.47913*inputs[1] +0.48726*inputs[2] +0.258728*inputs[3];
	combinations[2] = 0.812448 +0.277402*inputs[0] -0.456121*inputs[1] +0.817191*inputs[2] +0.791688*inputs[3];

	vector<float> activations(3);

	activations[0] = tanh(combinations[0]);
	activations[1] = tanh(combinations[1]);
	activations[2] = tanh(combinations[2]);

	return activations;
}

vector<float> probabilistic_layer(const vector<float>& inputs)
{
	vector<float> combinations(3);

	combinations[0] = 0.179285 +0.814947*inputs[0] -1.20329*inputs[1] -1.69145*inputs[2];
	combinations[1] = -0.0153737 +1.91541*inputs[0] -0.331712*inputs[1] +1.76223*inputs[2];
	combinations[2] = -0.164746 -2.24823*inputs[0] +1.65161*inputs[1] +0.134469*inputs[2];

	vector<float> activations(3);

	float sum = 0;

	sum = exp(combinations[0]) + exp(combinations[1]) + exp(combinations[2]);

	activations[0] = exp(combinations[0])/sum;
	activations[1] = exp(combinations[1])/sum;
	activations[2] = exp(combinations[2])/sum;

	return activations;
}

vector<float> neural_network(const vector<float>& inputs)
{
	vector<float> outputs;

	outputs = scaling_layer(inputs);
	outputs = perceptron_layer_1(outputs);
	outputs = probabilistic_layer(outputs);

	return outputs;
}
int main(){return 0;}
