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
	1 )clump_thickness
	2 )cell_size_uniformity
	3 )cell_shape_uniformity
	4 )marginal_adhesion
	5 )single_epithelial_cell_size
	6 )bare_nuclei
	7 )bland_chromatin
	8 )normal_nucleoli
	9 )mitoses

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
 
		self.parameters_number = 67
 
	def scaling_layer(self,inputs):

		outputs = [None] * 9

		outputs[0] = (inputs[0]-4.442166805)/2.820761204
		outputs[1] = (inputs[1]-3.150805235)/3.065144777
		outputs[2] = (inputs[2]-3.215226889)/2.988580704
		outputs[3] = (inputs[3]-2.830161095)/2.864562273
		outputs[4] = (inputs[4]-3.234260559)/2.223085403
		outputs[5] = (inputs[5]-3.544656038)/3.643857241
		outputs[6] = (inputs[6]-3.445095062)/2.449696541
		outputs[7] = (inputs[7]-2.869692564)/3.052666426
		outputs[8] = (inputs[8]-1.603221059)/1.732674122

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 6

		combinations[0] = 3.70429 +3.41644*inputs[0] +3.44443*inputs[1] +3.51931*inputs[2] +3.72499*inputs[3] +3.69408*inputs[4] +3.39786*inputs[5] +3.54709*inputs[6] +3.63395*inputs[7] +3.41914*inputs[8] 
		combinations[1] = 3.45426 +3.57896*inputs[0] +3.39489*inputs[1] +3.53734*inputs[2] +2.71027e-40*inputs[3] -6.7243e-40*inputs[4] +1.08593e-39*inputs[5] -1.0746e-39*inputs[6] -2.49049e-40*inputs[7] +9.97027e-40*inputs[8] 
		combinations[2] = 3.733 -1.18726e-39*inputs[0] -1.10941e-39*inputs[1] +7.75774e-40*inputs[2] +1.07932e-39*inputs[3] +2.55293e-40*inputs[4] -1.18815e-39*inputs[5] +7.17189e-40*inputs[6] +1.14458e-39*inputs[7] +7.11932e-40*inputs[8] 
		combinations[3] = 3.36151 -9.62406e-40*inputs[0] -6.33422e-40*inputs[1] +9.91183e-40*inputs[2] +1.23836e-39*inputs[3] +8.00182e-40*inputs[4] -9.79157e-40*inputs[5] -1.0085e-39*inputs[6] -1.07119e-39*inputs[7] -9.97782e-40*inputs[8] 
		combinations[4] = 3.73214 +1.20873e-39*inputs[0] -2.23339e-40*inputs[1] -1.27504e-39*inputs[2] -1.0906e-39*inputs[3] +2.95031e-40*inputs[4] -1.10799e-39*inputs[5] -1.11444e-39*inputs[6] -9.09521e-40*inputs[7] +1.03982e-39*inputs[8] 
		combinations[5] = 3.49928 -1.29599e-39*inputs[0] +5.87545e-40*inputs[1] +1.17464e-39*inputs[2] -0.128277*inputs[3] -0.16606*inputs[4] -0.144468*inputs[5] -0.0627427*inputs[6] +0.14851*inputs[7] +0.148684*inputs[8] 
		
		activations = [None] * 6

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 1

		combinations[0] = -0.352275 +2.32664*inputs[0] +2.09476*inputs[1] -0.590076*inputs[2] -0.349707*inputs[3] -0.25443*inputs[4] -0.407854*inputs[5] 
		
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
