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
	1 )input_1

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
 
		self.parameters_number = 37
 
	def scaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*2-0

		return outputs;


	def Perceptron_layer_1(self,inputs):

		combinations = [None] * 12

		combinations[0] = 0.00579174 +0.0703753*inputs[0] 
		combinations[1] = -0.00923719 -0.150621*inputs[0] 
		combinations[2] = 0.0036113 +0.0173306*inputs[0] 
		combinations[3] = -0.00432969 -0.0580116*inputs[0] 
		combinations[4] = 0.00480544 -0.0272374*inputs[0] 
		combinations[5] = -0.00427574 -0.0654304*inputs[0] 
		combinations[6] = 0.000137226 +0.0634266*inputs[0] 
		combinations[7] = -0.0148643 -0.204804*inputs[0] 
		combinations[8] = -0.000624239 +0.0348438*inputs[0] 
		combinations[9] = 0.00494637 -0.087008*inputs[0] 
		combinations[10] = 0.00402091 +0.0771491*inputs[0] 
		combinations[11] = 0.00865105 +0.11452*inputs[0] 
		
		activations = [None] * 12

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
		activations[10] = np.tanh(combinations[10])
		activations[11] = np.tanh(combinations[11])

		return activations;


	def Perceptron_layer_2(self,inputs):

		combinations = [None] * 1

		combinations[0] = 0.0134395 +0.0700213*inputs[0] -0.151784*inputs[1] +0.0173042*inputs[2] -0.0575696*inputs[3] -0.0287691*inputs[4] -0.0658386*inputs[5] +0.0632458*inputs[6] -0.2004*inputs[7] +0.0344881*inputs[8] -0.0871112*inputs[9] +0.0796907*inputs[10] +0.113821*inputs[11] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*3.449328661+124.8359451

		return outputs


	def bounding_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]

		return outputs


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_Perceptron_layer_1 = self.Perceptron_layer_1(output_scaling_layer)

		output_Perceptron_layer_2 = self.Perceptron_layer_2(output_Perceptron_layer_1)

		output_unscaling_layer = self.unscaling_layer(output_Perceptron_layer_2)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_Perceptron_layer_1 = self.Perceptron_layer_1(output_scaling_layer)

			output_Perceptron_layer_2 = self.Perceptron_layer_2(output_Perceptron_layer_1)

			output_unscaling_layer = self.unscaling_layer(output_Perceptron_layer_2)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
