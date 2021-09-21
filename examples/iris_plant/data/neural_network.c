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

	combinations[0] = -0.502284 -0.262351*inputs[0] +0.425987*inputs[1] -0.668783*inputs[2] -0.624135*inputs[3];
	combinations[1] = -2.08545 -0.130762*inputs[0] -0.576229*inputs[1] +1.99443*inputs[2] +1.41635*inputs[3];
	combinations[2] = 0.462892 +0.260135*inputs[0] -0.430484*inputs[1] +0.656358*inputs[2] +0.604672*inputs[3];

	vector<float> activations(3);

	activations[0] = tanh(combinations[0]);
	activations[1] = tanh(combinations[1]);
	activations[2] = tanh(combinations[2]);

	return activations;
}

vector<float> probabilistic_layer(const vector<float>& inputs)
{
	vector<float> combinations(3);

	combinations[0] = 0.265959 +1.30834*inputs[0] -0.596008*inputs[1] -1.33945*inputs[2];
	combinations[1] = -0.391851 -0.966543*inputs[0] -2.11191*inputs[1] +0.905591*inputs[2];
	combinations[2] = 0.125989 -0.459693*inputs[0] +2.39002*inputs[1] +0.559947*inputs[2];

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
