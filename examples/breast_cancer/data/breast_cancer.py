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

		combinations[0] = 0.244577 +0.906687*inputs[0] +0.759496*inputs[1] +1.76256*inputs[2] +0.382582*inputs[3] +0.342114*inputs[4] +1.07983*inputs[5] -0.00602741*inputs[6] +1.44184*inputs[7] -0.0366609*inputs[8] 
		combinations[1] = -0.0409032 +1.47499*inputs[0] -0.862586*inputs[1] +0.666515*inputs[2] -0.614149*inputs[3] -0.918053*inputs[4] +1.07669*inputs[5] +0.0945446*inputs[6] +2.04449*inputs[7] +0.264392*inputs[8] 
		combinations[2] = -2.12664 +0.256234*inputs[0] +0.674226*inputs[1] -0.476106*inputs[2] +0.719289*inputs[3] +1.15516*inputs[4] -0.898875*inputs[5] +0.387625*inputs[6] +1.16791*inputs[7] +0.900421*inputs[8] 
		combinations[3] = -0.210046 +0.050026*inputs[0] -0.164905*inputs[1] -0.891915*inputs[2] -0.978673*inputs[3] +0.0203211*inputs[4] -1.99462*inputs[5] -0.47445*inputs[6] +1.51856*inputs[7] -0.166461*inputs[8] 
		combinations[4] = 1.00134 +2.50405*inputs[0] -0.495027*inputs[1] -0.109783*inputs[2] -1.39251*inputs[3] +2.63462*inputs[4] -1.16455*inputs[5] -0.978416*inputs[6] -0.43906*inputs[7] -0.0021004*inputs[8] 
		combinations[5] = -0.709447 -3.44641*inputs[0] +0.282477*inputs[1] +1.53466*inputs[2] +1.87098*inputs[3] -0.442781*inputs[4] +0.503129*inputs[5] +0.028696*inputs[6] +1.1542*inputs[7] -0.85607*inputs[8] 
		
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

		combinations[0] = 1.17273 +3.87907*inputs[0] +3.44563*inputs[1] +3.39344*inputs[2] -3.43661*inputs[3] -4.45167*inputs[4] -5.04438*inputs[5] 
		
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
