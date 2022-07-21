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

		combinations[0] = 0.14122 -0.182705*inputs[0] -0.206823*inputs[1] 
		combinations[1] = -0.164227 +0.221125*inputs[0] +0.0921721*inputs[1] 
		combinations[2] = -0.123684 -0.0911117*inputs[0] -0.0462765*inputs[1] 
		combinations[3] = -0.0082476 +0.154298*inputs[0] +0.00951754*inputs[1] 
		combinations[4] = 0.101263 +0.114064*inputs[0] +0.287481*inputs[1] 
		combinations[5] = 0.125717 +0.132671*inputs[0] +0.230353*inputs[1] 
		combinations[6] = -0.134883 -0.504073*inputs[0] -0.428432*inputs[1] 
		combinations[7] = -0.0479741 +0.0972213*inputs[0] +0.491557*inputs[1] 
		combinations[8] = -0.0235664 +0.136482*inputs[0] -0.0892608*inputs[1] 
		combinations[9] = -0.0368696 -0.190251*inputs[0] -0.0165799*inputs[1] 
		
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

		combinations[0] = -0.159178 -0.0859112*inputs[0] +0.0105446*inputs[1] -0.210578*inputs[2] +0.0689023*inputs[3] +0.365389*inputs[4] +0.427273*inputs[5] -0.654416*inputs[6] +0.369509*inputs[7] -0.00577036*inputs[8] -0.119732*inputs[9] 
		
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
