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
		outputs[1] = (inputs[1]-0.4880183339)/0.1982036084

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 10

		combinations[0] = 0.296804 -0.954437*inputs[0] +0.885734*inputs[1] 
		combinations[1] = 0.3031 -0.971211*inputs[0] +0.901309*inputs[1] 
		combinations[2] = -0.277971 +0.904248*inputs[0] -0.839228*inputs[1] 
		combinations[3] = 0.318156 -1.01146*inputs[0] +0.938595*inputs[1] 
		combinations[4] = 0.289379 -0.934634*inputs[0] +0.867385*inputs[1] 
		combinations[5] = 0.313975 -1.00031*inputs[0] +0.92816*inputs[1] 
		combinations[6] = 0.315903 -1.00549*inputs[0] +0.932938*inputs[1] 
		combinations[7] = 0.268993 -0.88033*inputs[0] +0.817097*inputs[1] 
		combinations[8] = -0.311001 +0.992423*inputs[0] -0.920825*inputs[1] 
		combinations[9] = 0.28485 -0.922566*inputs[0] +0.856198*inputs[1] 
		
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

		combinations[0] = 0.11943 +1.35157*inputs[0] +1.37691*inputs[1] -1.27634*inputs[2] +1.43747*inputs[3] +1.32187*inputs[4] +1.42081*inputs[5] +1.42853*inputs[6] +1.24052*inputs[7] -1.40855*inputs[8] +1.30378*inputs[9] 
		
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
