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

		combinations[0] = -0.0553626 -0.284121*inputs[0] +0.393168*inputs[1] 
		combinations[1] = 0.0500527 +0.287347*inputs[0] -0.396931*inputs[1] 
		combinations[2] = 0.0541212 +0.285623*inputs[0] -0.396737*inputs[1] 
		combinations[3] = -0.0535371 -0.285073*inputs[0] +0.394165*inputs[1] 
		combinations[4] = -0.0539547 -0.284463*inputs[0] +0.394741*inputs[1] 
		combinations[5] = -0.055159 -0.284111*inputs[0] +0.394182*inputs[1] 
		combinations[6] = -0.05399 -0.284568*inputs[0] +0.394337*inputs[1] 
		combinations[7] = -0.0559819 -0.283431*inputs[0] +0.394646*inputs[1] 
		combinations[8] = 0.0538082 +0.283855*inputs[0] -0.394204*inputs[1] 
		combinations[9] = -0.0564466 -0.284477*inputs[0] +0.393685*inputs[1] 
		
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

		combinations[0] = -0.248673 +0.604046*inputs[0] -0.607294*inputs[1] -0.6078*inputs[2] +0.607067*inputs[3] +0.605086*inputs[4] +0.604312*inputs[5] +0.604541*inputs[6] +0.605493*inputs[7] -0.605263*inputs[8] +0.606772*inputs[9] 
		
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
