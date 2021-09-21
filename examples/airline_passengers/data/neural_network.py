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
 
		self.parameters_number = 1233
 
	def scaling_layer(self,inputs):

		outputs = [None] * 2

		outputs[0] = inputs[0]*0.003861003788-1.401544333
		outputs[1] = inputs[1]*0.003861003788-1.401544333

		return outputs;



	def perceptron_layer_1(self,inputs):

		combinations = [None] * 1

		combinations[0] = -0.111467 +4.21092e-05*inputs[0] +0.00139274*inputs[1] -3.85441e-05*inputs[2] -5.92739e-05*inputs[3] +0.000619816*inputs[4] -3.99875e-05*inputs[5] +2.77004e-05*inputs[6] -0.00011394*inputs[7] +4.44926e-05*inputs[8] -2.68639e-05*inputs[9] -8.08119e-05*inputs[10] -0.00049337*inputs[11] +0.800999*inputs[12] +0.00083867*inputs[13] -0.000216696*inputs[14] -0.000405913*inputs[15] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

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

		output_long_short_term_memory_layer = self.long_short_term_memory_layer(output_scaling_layer)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_long_short_term_memory_layer)

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_1)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_long_short_term_memory_layer = self.long_short_term_memory_layer(output_scaling_layer)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_long_short_term_memory_layer)

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_1)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
