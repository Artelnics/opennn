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
	vector<float> combinations(10);

	combinations[0] = 0.395957 -0.125231*inputs[0] +0.326583*inputs[1] -0.388694*inputs[2] -0.389443*inputs[3];
	combinations[1] = -0.374394 -0.420275*inputs[0] +0.36244*inputs[1] -0.533691*inputs[2] -0.274081*inputs[3];
	combinations[2] = -0.52096 -0.0361089*inputs[0] +0.0703434*inputs[1] +0.549457*inputs[2] +0.420585*inputs[3];
	combinations[3] = 0.522812 +0.146044*inputs[0] +0.062207*inputs[1] -0.411823*inputs[2] -0.750093*inputs[3];
	combinations[4] = 0.364621 +0.504047*inputs[0] -0.437882*inputs[1] +0.291139*inputs[2] +0.340977*inputs[3];
	combinations[5] = 0.384518 +0.541423*inputs[0] -0.367351*inputs[1] +0.498336*inputs[2] +0.172826*inputs[3];
	combinations[6] = -0.600615 +0.0453112*inputs[0] -0.128768*inputs[1] +0.396013*inputs[2] +0.604464*inputs[3];
	combinations[7] = 0.412966 +0.0497757*inputs[0] +0.19464*inputs[1] -0.500716*inputs[2] -0.42613*inputs[3];
	combinations[8] = 0.549809 +0.54521*inputs[0] -0.34135*inputs[1] +0.36204*inputs[2] +0.325535*inputs[3];
	combinations[9] = 0.401373 +0.0593159*inputs[0] +0.046146*inputs[1] -0.53164*inputs[2] -0.39144*inputs[3];

	vector<float> activations(10);

	activations[0] = tanh(combinations[0]);
	activations[1] = tanh(combinations[1]);
	activations[2] = tanh(combinations[2]);
	activations[3] = tanh(combinations[3]);
	activations[4] = tanh(combinations[4]);
	activations[5] = tanh(combinations[5]);
	activations[6] = tanh(combinations[6]);
	activations[7] = tanh(combinations[7]);
	activations[8] = tanh(combinations[8]);
	activations[9] = tanh(combinations[9]);

	return activations;
}

vector<float> probabilistic_layer(const vector<float>& inputs)
{
	vector<float> combinations(3);

	combinations[0] = -0.122392 +0.45217*inputs[0] +0.346158*inputs[1] -0.409535*inputs[2] +0.427543*inputs[3] -0.399237*inputs[4] -0.354133*inputs[5] -0.433496*inputs[6] +0.391537*inputs[7] -0.503915*inputs[8] +0.454471*inputs[9];
	combinations[1] = 0.0803996 +0.308758*inputs[0] -0.64641*inputs[1] -0.816637*inputs[2] +0.532274*inputs[3] +0.488053*inputs[4] +0.684737*inputs[5] -0.628385*inputs[6] +0.246522*inputs[7] +0.673086*inputs[8] +0.456811*inputs[9];
	combinations[2] = -0.0582235 -0.621868*inputs[0] -0.441797*inputs[1] +0.620912*inputs[2] -0.547483*inputs[3] +0.43949*inputs[4] +0.351803*inputs[5] +0.624807*inputs[6] -0.736667*inputs[7] +0.231193*inputs[8] -0.585228*inputs[9];

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
