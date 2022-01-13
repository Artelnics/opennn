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

		combinations[0] = -0.617455 -1.10526*inputs[0] +1.23514*inputs[1] +1.2966*inputs[2] +0.96099*inputs[3] +0.719433*inputs[4] -2.57781*inputs[5] +0.0383907*inputs[6] -0.894202*inputs[7] -0.745367*inputs[8] 
		combinations[1] = 0.205737 -0.41805*inputs[0] -1.2195*inputs[1] -1.01949*inputs[2] +0.078814*inputs[3] +0.111573*inputs[4] -0.316568*inputs[5] +0.348827*inputs[6] -0.798851*inputs[7] +1.91406*inputs[8] 
		combinations[2] = 0.997588 +2.00212*inputs[0] -0.592221*inputs[1] +0.121691*inputs[2] -2.35505*inputs[3] +1.5066*inputs[4] +0.365527*inputs[5] -0.387985*inputs[6] -0.758878*inputs[7] +0.612353*inputs[8] 
		combinations[3] = 0.439836 +0.0451582*inputs[0] +0.628178*inputs[1] -0.979759*inputs[2] -1.08492*inputs[3] +1.59822*inputs[4] -0.856908*inputs[5] -0.340934*inputs[6] -0.249057*inputs[7] -2.1003*inputs[8] 
		combinations[4] = -0.730462 -0.324273*inputs[0] -1.60305*inputs[1] -1.41095*inputs[2] -0.399712*inputs[3] +0.805248*inputs[4] +0.325967*inputs[5] -1.48697*inputs[6] +1.53748*inputs[7] -0.148235*inputs[8] 
		combinations[5] = 2.15955 -2.57008*inputs[0] -0.0311834*inputs[1] +0.870531*inputs[2] +2.46846*inputs[3] -2.68832*inputs[4] -0.252128*inputs[5] -0.0912408*inputs[6] +0.10048*inputs[7] -1.19862*inputs[8] 
		
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

		combinations[0] = 1.6341 -4.15695*inputs[0] -3.66804*inputs[1] -3.86973*inputs[2] -2.79765*inputs[3] -4.32244*inputs[4] -6.08864*inputs[5] 
		
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
