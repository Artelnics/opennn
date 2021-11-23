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

		outputs[0] = inputs[0]*2.415266275-14.11320591
		outputs[1] = inputs[1]*4.588562965-14.02876568
		outputs[2] = inputs[2]*1.132953048-4.257637501
		outputs[3] = inputs[3]*2.623853445-3.146874905

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 3

		combinations[0] = 1.66413 +0.485914*inputs[0] +0.0471202*inputs[1] -1.99069*inputs[2] -1.24612*inputs[3] 
		combinations[1] = -0.40017 +0.911162*inputs[0] -1.47913*inputs[1] +0.48726*inputs[2] +0.258728*inputs[3] 
		combinations[2] = 0.812448 +0.277402*inputs[0] -0.456121*inputs[1] +0.817191*inputs[2] +0.791688*inputs[3] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 3

		combinations[0] = 0.179285 +0.814947*inputs[0] -1.20329*inputs[1] -1.69145*inputs[2] 
		combinations[1] = -0.0153737 +1.91541*inputs[0] -0.331712*inputs[1] +1.76223*inputs[2] 
		combinations[2] = -0.164746 -2.24823*inputs[0] +1.65161*inputs[1] +0.134469*inputs[2] 
		
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
