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

		outputs[0] = inputs[0]*10.50651836-5.223269463
		outputs[1] = inputs[1]*10.09063339-4.924414158

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 10

		combinations[0] = 0.0146402 -0.0574556*inputs[0] +0.482391*inputs[1] 
		combinations[1] = -0.00501259 +0.0573209*inputs[0] -0.481549*inputs[1] 
		combinations[2] = -0.0124921 +0.0565344*inputs[0] -0.482869*inputs[1] 
		combinations[3] = -0.00290161 +0.0535286*inputs[0] -0.480699*inputs[1] 
		combinations[4] = -0.009853 +0.0562651*inputs[0] -0.482262*inputs[1] 
		combinations[5] = -0.50577 +2.6992*inputs[0] +3.40169*inputs[1] 
		combinations[6] = 2.32576 +0.00369966*inputs[0] -2.51684*inputs[1] 
		combinations[7] = 1.37366 -0.325119*inputs[0] +2.31626*inputs[1] 
		combinations[8] = 0.012314 -0.0458432*inputs[0] +0.476505*inputs[1] 
		combinations[9] = 0.0108689 -0.0591511*inputs[0] +0.484173*inputs[1] 
		
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

		combinations[0] = -0.0587929 +0.683608*inputs[0] -0.681948*inputs[1] -0.690227*inputs[2] -0.689325*inputs[3] -0.689073*inputs[4] -4.63996*inputs[5] -3.21611*inputs[6] +2.80742*inputs[7] +0.680546*inputs[8] +0.692442*inputs[9] 
		
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
