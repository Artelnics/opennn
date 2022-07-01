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

		combinations[0] = 0.454264 +1.77702*inputs[0] +0.200494*inputs[1] +1.78917*inputs[2] -0.4011*inputs[3] -1.27332*inputs[4] +0.768725*inputs[5] -0.281489*inputs[6] +0.227429*inputs[7] -0.11199*inputs[8] 
		combinations[1] = -0.196334 -0.438311*inputs[0] -0.690974*inputs[1] +0.007969*inputs[2] +1.51447*inputs[3] +1.43278*inputs[4] -2.52714*inputs[5] -0.124051*inputs[6] -1.27162*inputs[7] -0.359693*inputs[8] 
		combinations[2] = 0.326382 +0.223538*inputs[0] +0.832724*inputs[1] +1.74177*inputs[2] -0.216371*inputs[3] +0.884064*inputs[4] +0.308687*inputs[5] -0.0740288*inputs[6] +0.462171*inputs[7] +0.178247*inputs[8] 
		combinations[3] = -0.412684 +0.0153166*inputs[0] +0.519617*inputs[1] +0.174587*inputs[2] -1.22584*inputs[3] +0.103734*inputs[4] -1.88753*inputs[5] -2.26878*inputs[6] +0.627284*inputs[7] +0.349217*inputs[8] 
		combinations[4] = -1.28755 +0.352426*inputs[0] +0.903009*inputs[1] -0.0540493*inputs[2] +0.389099*inputs[3] +0.784332*inputs[4] -1.18888*inputs[5] -0.00419092*inputs[6] -0.786123*inputs[7] +1.74607*inputs[8] 
		combinations[5] = -0.645534 -0.954436*inputs[0] +0.828924*inputs[1] +2.45316*inputs[2] -0.549426*inputs[3] -1.17053*inputs[4] +0.412908*inputs[5] +1.16295*inputs[6] -1.95577*inputs[7] +0.0118666*inputs[8] 
		
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

		combinations[0] = -0.900619 +3.16153*inputs[0] -3.69624*inputs[1] +2.60757*inputs[2] -4.60476*inputs[3] +3.66728*inputs[4] -4.09019*inputs[5] 
		
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
