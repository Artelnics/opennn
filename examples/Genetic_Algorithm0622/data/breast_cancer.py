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

		combinations[0] = 0.0379619 +0.0409196*inputs[0] -0.0158043*inputs[1] -0.140165*inputs[2] -0.0318442*inputs[3] -0.0318634*inputs[4] -0.141867*inputs[5] +0.0757628*inputs[6] +0.0811975*inputs[7] -0.101542*inputs[8] 
		combinations[1] = 0.175134 +0.0795227*inputs[0] -0.157645*inputs[1] +0.183973*inputs[2] -0.0860196*inputs[3] +0.160096*inputs[4] -0.0671317*inputs[5] +0.210777*inputs[6] -0.0342633*inputs[7] +0.0178031*inputs[8] 
		combinations[2] = -0.160985 -0.17469*inputs[0] +0.0346481*inputs[1] +0.104751*inputs[2] -0.101571*inputs[3] +0.197036*inputs[4] -0.115727*inputs[5] +0.106018*inputs[6] +0.0330969*inputs[7] -0.133989*inputs[8] 
		combinations[3] = -0.0466925 +0.123729*inputs[0] +0.141913*inputs[1] -0.184954*inputs[2] +0.0369268*inputs[3] -0.007148*inputs[4] +0.00344335*inputs[5] +0.10061*inputs[6] +0.165037*inputs[7] +0.1843*inputs[8] 
		combinations[4] = 0.160038 +0.194015*inputs[0] +0.0820236*inputs[1] +0.10035*inputs[2] -0.101486*inputs[3] -0.0636809*inputs[4] +0.111987*inputs[5] -0.142496*inputs[6] +0.0234071*inputs[7] +0.0402573*inputs[8] 
		combinations[5] = 0.0944048 -0.0365882*inputs[0] +0.0176206*inputs[1] +0.192869*inputs[2] +0.161544*inputs[3] +0.0183423*inputs[4] +0.00781967*inputs[5] +0.0482517*inputs[6] +0.095467*inputs[7] -0.0257112*inputs[8] 
		
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

		combinations[0] = -0.00239372 -0.0739934*inputs[0] +0.164357*inputs[1] +0.126973*inputs[2] -0.0157294*inputs[3] +0.126547*inputs[4] +0.199529*inputs[5] 
		
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
