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

		outputs[0] = inputs[0]*0.003861003788-1.401544333
		outputs[1] = inputs[1]*0.003861003788-1.401544333

		return outputs;


	def perceptron_layer(self,inputs):

		combinations = [None] * 10

		combinations[0] = 0.0330065 +0.0830493*inputs[0] +0.296817*inputs[1] 
		combinations[1] = -0.0333454 -0.0827217*inputs[0] -0.297391*inputs[1] 
		combinations[2] = 0.0336491 +0.0830849*inputs[0] +0.297368*inputs[1] 
		combinations[3] = -0.0338421 -0.0827444*inputs[0] -0.297307*inputs[1] 
		combinations[4] = -0.0335228 -0.0826635*inputs[0] -0.297969*inputs[1] 
		combinations[5] = -0.0332087 -0.0827688*inputs[0] -0.296673*inputs[1] 
		combinations[6] = 0.0335995 +0.0825544*inputs[0] +0.297259*inputs[1] 
		combinations[7] = -0.0339583 -0.0824925*inputs[0] -0.297393*inputs[1] 
		combinations[8] = -0.0333522 -0.0826598*inputs[0] -0.29676*inputs[1] 
		combinations[9] = -0.0332117 -0.0825865*inputs[0] -0.296763*inputs[1] 
		
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


	def perceptron_layer(self,inputs):

		combinations = [None] * 1

		combinations[0] = -0.0761012 +0.332532*inputs[0] -0.332558*inputs[1] +0.332942*inputs[2] -0.332951*inputs[3] -0.333413*inputs[4] -0.332074*inputs[5] +0.332432*inputs[6] -0.332834*inputs[7] -0.331948*inputs[8] -0.331905*inputs[9] 
		
		activations = [None] * 1

		activations[0] = np.tanh(combinations[0])

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*259+363

		return outputs


	def bounding_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]

		return outputs


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer = self.perceptron_layer(output_scaling_layer)

		output_perceptron_layer = self.perceptron_layer(output_perceptron_layer)

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer = self.perceptron_layer(output_scaling_layer)

			output_perceptron_layer = self.perceptron_layer(output_perceptron_layer)

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
