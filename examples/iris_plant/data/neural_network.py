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
	1 )sepal_lenght
	2 )sepal_width
	3 )petal_lenght
	4 )petal_width

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
 
		self.parameters_number = 27
 
	def scaling_layer(self,inputs):

		outputs = [None] * 4

		outputs[0] = (inputs[0]-5.843333244)/0.8280661106
		outputs[1] = (inputs[1]-3.057333231)/0.4358662963
		outputs[2] = (inputs[2]-3.757999897)/1.765298247
		outputs[3] = (inputs[3]-1.19933331)/0.762237668

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 3

		combinations[0] = -0.648513 -0.310869*inputs[0] +0.46917*inputs[1] -0.8179*inputs[2] -0.712404*inputs[3] 
		combinations[1] = -0.645249 -0.301602*inputs[0] +0.468052*inputs[1] -0.819502*inputs[2] -0.717822*inputs[3] 
		combinations[2] = -2.53433 -0.717074*inputs[0] -0.212808*inputs[1] +3.04323*inputs[2] +1.531*inputs[3] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 3

		combinations[0] = 0.440813 +1.91102*inputs[0] +1.90659*inputs[1] -0.784815*inputs[2] 
		combinations[1] = -0.758312 -1.31256*inputs[0] -1.311*inputs[1] -2.96961*inputs[2] 
		combinations[2] = 0.316884 -0.59882*inputs[0] -0.595287*inputs[1] +3.75488*inputs[2] 
		
		activations = [None] * 3

		sum_ = 0;

		sum_ = 	np.exp(combinations[0]) + 	np.exp(combinations[1]) + 	np.exp(combinations[2]);

		activations[0] = np.exp(combinations[0])/sum_;
		activations[1] = np.exp(combinations[1])/sum_;
		activations[2] = np.exp(combinations[2])/sum_;

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
