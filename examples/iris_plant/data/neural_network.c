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
// however you can loop through neural network function to get multiple outputs.	

#include <vector>

using namespace std;

vector<float> scaling_layer(const vector<float>& inputs)
{
	vector<float> outputs(4);

	outputs[0] = (inputs[0]-5.843333244)/0.8280661106;
	outputs[1] = (inputs[1]-3.057333231)/0.4358662963;
	outputs[2] = (inputs[2]-3.757999897)/1.765298247;
	outputs[3] = (inputs[3]-1.19933331)/0.762237668;

	return outputs;
}

vector<float> perceptron_layer_1(const vector<float>& inputs)
{
	vector<float> combinations(3);

	combinations[0] = 0.171232 +0.127784*inputs[0] +0.156159*inputs[1] -0.0116542*inputs[2] -0.192001*inputs[3];
	combinations[1] = -0.100386 +0.0459516*inputs[0] -0.127511*inputs[1] +0.158615*inputs[2] -0.187598*inputs[3];
	combinations[2] = -0.0988517 +0.158462*inputs[0] -0.161392*inputs[1] +0.0213799*inputs[2] +0.125907*inputs[3];

	vector<float> activations(3);

	activations[0] = tanh(combinations[0]);
	activations[1] = tanh(combinations[1]);
	activations[2] = tanh(combinations[2]);

	return activations;
}

vector<float> probabilistic_layer(const vector<float>& inputs)
{
	vector<float> combinations(3);

	combinations[0] = 0.0739868 -0.172685*inputs[0] -0.10595*inputs[1] +0.00743201*inputs[2];
	combinations[1] = 0.0420353 +0.0467598*inputs[0] +0.189647*inputs[1] +0.191713*inputs[2];
	combinations[2] = -0.0777208 +0.145562*inputs[0] +0.170601*inputs[1] -0.0659442*inputs[2];

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
