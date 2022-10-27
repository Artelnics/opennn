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
	1 )Passengers_lag_1
	2 )Passengers_lag_0

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

		outputs[0] = (inputs[0]-278.4577332)/119.7667694
		outputs[1] = (inputs[1]-280.4154968)/119.2977219

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 10

		combinations[0] = -0.136524 -0.18759*inputs[0] +0.263137*inputs[1] 
		combinations[1] = -0.602783 -0.256299*inputs[0] -0.556332*inputs[1] 
		combinations[2] = 0.203363 +0.219422*inputs[0] -0.309509*inputs[1] 
		combinations[3] = -0.442103 -0.204843*inputs[0] -0.377829*inputs[1] 
		combinations[4] = -0.181952 +0.0189888*inputs[0] -0.27534*inputs[1] 
		combinations[5] = -0.286766 -0.151463*inputs[0] -0.208432*inputs[1] 
		combinations[6] = -0.0751686 -0.192901*inputs[0] +0.268802*inputs[1] 
		combinations[7] = 0.193406 +0.282964*inputs[0] -0.421903*inputs[1] 
		combinations[8] = -0.0823634 -0.18008*inputs[0] +0.259467*inputs[1] 
		combinations[9] = 0.158052 +0.264705*inputs[0] -0.381678*inputs[1] 
		
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


	def perceptron_layer_2(self,inputs):

		combinations = [None] * 1

		combinations[0] = -0.256116 +0.35846*inputs[0] -0.549723*inputs[1] -0.34595*inputs[2] -0.429958*inputs[3] -0.165212*inputs[4] -0.0895895*inputs[5] +0.317934*inputs[6] -0.354081*inputs[7] +0.420682*inputs[8] -0.51796*inputs[9] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*119.1759262+282.62677

		return outputs


	def bounding_layer(self,inputs):

		outputs = [None] * 1

		if inputs[0] < -3.40282e+38:

			outputs[0] = -3.40282e+38

		elif inputs[0] >3.40282e+38:

			outputs[0] = 3.40282e+38

		else:

			outputs[0] = inputs[0]


		return outputs


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

		output_perceptron_layer_2 = self.perceptron_layer_2(output_perceptron_layer_1)

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_2)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

			output_perceptron_layer_2 = self.perceptron_layer_2(output_perceptron_layer_1)

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_2)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
