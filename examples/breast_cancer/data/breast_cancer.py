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

		combinations[0] = 1.96056 -1.74701*inputs[0] +0.238941*inputs[1] +1.2898*inputs[2] +1.46953*inputs[3] -2.53722*inputs[4] +0.118806*inputs[5] +0.637481*inputs[6] -0.184717*inputs[7] -0.365973*inputs[8] 
		combinations[1] = 0.340047 +0.0562003*inputs[0] +0.0910076*inputs[1] -0.0758806*inputs[2] -0.722346*inputs[3] -0.560783*inputs[4] +2.80715*inputs[5] +0.686939*inputs[6] +1.26853*inputs[7] -0.108185*inputs[8] 
		combinations[2] = 0.677734 +1.53907*inputs[0] -0.772097*inputs[1] -1.98917*inputs[2] -0.967579*inputs[3] +3.13899*inputs[4] +0.205123*inputs[5] -0.189297*inputs[6] -0.325449*inputs[7] -0.376298*inputs[8] 
		combinations[3] = 1.09076 -0.630748*inputs[0] -1.22475*inputs[1] -0.2717*inputs[2] +0.307185*inputs[3] -1.36184*inputs[4] -1.10769*inputs[5] -0.142647*inputs[6] +1.80629*inputs[7] -0.294909*inputs[8] 
		combinations[4] = 1.1587 +2.0793*inputs[0] -0.526391*inputs[1] -1.01813*inputs[2] +2.17645*inputs[3] +0.0669543*inputs[4] +0.822836*inputs[5] +0.198599*inputs[6] +0.626569*inputs[7] +0.83028*inputs[8] 
		combinations[5] = 0.0500248 +0.761403*inputs[0] +0.412157*inputs[1] +0.144025*inputs[2] -0.561245*inputs[3] +1.3211*inputs[4] -1.27346*inputs[5] -0.869639*inputs[6] +1.45727*inputs[7] +0.77791*inputs[8] 
		
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

		combinations[0] = 1.3739 -5.37083*inputs[0] +3.53896*inputs[1] -4.76714*inputs[2] -2.9762*inputs[3] +5.17525*inputs[4] -2.33778*inputs[5] 
		
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
