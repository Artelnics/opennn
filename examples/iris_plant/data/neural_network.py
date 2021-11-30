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

		outputs[0] = (inputs[0]-5.843333244)/0.828066051
		outputs[1] = (inputs[1]-3.057333231)/0.4358663261
		outputs[2] = (inputs[2]-3.757999897)/1.765298247
		outputs[3] = (inputs[3]-1.19933331)/0.762237668

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 3

		combinations[0] = -0.633906 -0.405145*inputs[0] +0.674149*inputs[1] -0.669616*inputs[2] -0.604541*inputs[3] 
		combinations[1] = -0.635636 -0.407228*inputs[0] +0.672023*inputs[1] -0.669132*inputs[2] -0.602476*inputs[3] 
		combinations[2] = 2.26119 -0.144742*inputs[0] -0.295567*inputs[1] -2.38942*inputs[2] -1.58967*inputs[3] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 3

		combinations[0] = 0.413436 +1.88506*inputs[0] +1.88341*inputs[1] +0.77632*inputs[2] 
		combinations[1] = -0.71565 -1.33322*inputs[0] -1.33367*inputs[1] +2.69755*inputs[2] 
		combinations[2] = 0.302323 -0.551563*inputs[0] -0.549589*inputs[1] -3.47387*inputs[2] 
		
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
