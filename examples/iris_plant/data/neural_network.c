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
	vector<float> outputs(6);

	outputs[0] = inputs[0]*2.415266275-14.11320591;
	outputs[1] = inputs[1]*4.588562965-14.02876568;
	outputs[2] = inputs[2]*1.132953048-4.257637501;
	outputs[3] = inputs[3]*4.228474617-1.409491658;
	outputs[4] = inputs[4]*4.228474617-1.409491658;
	outputs[5] = inputs[5]*4.228474617-1.409491658;

	return outputs;
}

vector<float> Perceptron_layer_1(const vector<float>& inputs)
{
	vector<float> combinations(3);

	combinations[0] = 0.808051 -0.137501*inputs[0] +0.233128*inputs[1] +0.875576*inputs[2] -0.432395*inputs[3] +0.296181*inputs[4] +0.164928*inputs[5];
	combinations[1] = -0.901862 +0.00684303*inputs[0] +0.0279776*inputs[1] -0.0950193*inputs[2] +0.47241*inputs[3] -0.196155*inputs[4] -0.268482*inputs[5];
	combinations[2] = -0.738978 +0.0462208*inputs[0] -0.0988235*inputs[1] -0.477717*inputs[2] +0.532404*inputs[3] -0.293018*inputs[4] -0.250419*inputs[5];

	vector<float> activations(3);

	activations[0] = tanh(combinations[0]);
	activations[1] = tanh(combinations[1]);
	activations[2] = tanh(combinations[2]);

	return activations;
}

vector<float> probabilistic_layer(const vector<float>& inputs)
{
	vector<float> combinations(1);

	combinations[0] = 2.23115 +2.17447*inputs[0] -1.96864*inputs[1] -2.04471*inputs[2];

	vector<float> activations(1);

	activations[0] = 1.0/(1.0 + exp(-combinations[0]));

	return activations;
}

vector<float> neural_network(const vector<float>& inputs)
{
	vector<float> outputs;

	outputs = scaling_layer(inputs);
	outputs = Perceptron_layer_1(outputs);
	outputs = probabilistic_layer(outputs);

	return outputs;
}
int main(){return 0;}
