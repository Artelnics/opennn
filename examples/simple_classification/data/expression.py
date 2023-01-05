'''
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.	
Example:

	model = NeuralNetwork()	
	sample = [input_1, input_2, input_3, input_4, ...]	
	outputs = model.calculate_output(sample)

	Inputs Names: 	
	1 )x1
	2 )x2

You can predict with a batch of samples using calculate_batch_output method	
IMPORTANT: input batch must be <class 'numpy.ndarray'> type	
Example_1:	
	model = NeuralNetwork()	
	input_batch = np.array([[1, 2], [4, 5]], np.int32)	
	outputs = model.calculate_batch_output(input_batch)
Example_2:	
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})	
	outputs = model.calculate_batch_output(input_batch.values)
'''

import numpy as np

class NeuralNetwork:
 
	def __init__(self):
 
		self.parameters_number = 41
 
	def scaling_layer(self,inputs):

		outputs = [None] * 2

		outputs[0] = (inputs[0]-0.4971455932)/0.1903580129
		outputs[1] = (inputs[1]-0.4880183339)/0.1982035935

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 10

		combinations[0] = 0.00238452 -0.364511*inputs[0] +0.450243*inputs[1] 
		combinations[1] = -0.00456767 +0.366057*inputs[0] -0.45232*inputs[1] 
		combinations[2] = 0.000673228 +0.362297*inputs[0] -0.447277*inputs[1] 
		combinations[3] = 0.00403211 -0.365687*inputs[0] +0.451815*inputs[1] 
		combinations[4] = -0.00673104 +0.367579*inputs[0] -0.454366*inputs[1] 
		combinations[5] = -0.00398696 +0.365649*inputs[0] -0.451769*inputs[1] 
		combinations[6] = -0.00223028 +0.364388*inputs[0] -0.450079*inputs[1] 
		combinations[7] = 0.003009 -0.364955*inputs[0] +0.450832*inputs[1] 
		combinations[8] = 0.00516576 -0.366472*inputs[0] +0.452885*inputs[1] 
		combinations[9] = 0.0120367 -0.371196*inputs[0] +0.459323*inputs[1] 
		
		activations = [None] * 10

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])
		activations[6] = np.tanh(combinations[6])
		activations[7] = np.tanh(combinations[7])
		activations[8] = np.tanh(combinations[8])
		activations[9] = np.tanh(combinations[9])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 1

		combinations[0] = -0.465869 +0.726045*inputs[0] -0.729723*inputs[1] -0.720411*inputs[2] +0.728907*inputs[3] -0.733134*inputs[4] -0.728764*inputs[5] -0.725726*inputs[6] +0.727096*inputs[7] +0.73067*inputs[8] +0.740099*inputs[9] 
		
		activations = [None] * 1

		activations[0] = 1.0/(1.0 + np.exp(-combinations[0]));

		return activations;


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

		output_probabilistic_layer = self.probabilistic_layer(output_perceptron_layer_1)

		return output_probabilistic_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

			output_probabilistic_layer = self.probabilistic_layer(output_perceptron_layer_1)

			output = np.append(output,output_probabilistic_layer, axis=0)

		return output
