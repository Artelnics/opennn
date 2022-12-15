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

		combinations[0] = 0.0634634 -0.26411*inputs[0] -0.15508*inputs[1] -0.192195*inputs[2] -0.0995892*inputs[3] -0.0974292*inputs[4] -0.248625*inputs[5] -0.141691*inputs[6] -0.169666*inputs[7] -0.120426*inputs[8] 
		combinations[1] = 0.0634895 -0.264043*inputs[0] -0.155027*inputs[1] -0.192136*inputs[2] -0.0995877*inputs[3] -0.0974019*inputs[4] -0.248535*inputs[5] -0.141686*inputs[6] -0.16963*inputs[7] -0.120412*inputs[8] 
		combinations[2] = 0.0634374 -0.264178*inputs[0] -0.155132*inputs[1] -0.192255*inputs[2] -0.0995908*inputs[3] -0.0974564*inputs[4] -0.248715*inputs[5] -0.141695*inputs[6] -0.169702*inputs[7] -0.12044*inputs[8] 
		combinations[3] = 0.0634341 -0.264186*inputs[0] -0.155139*inputs[1] -0.192262*inputs[2] -0.0995909*inputs[3] -0.0974598*inputs[4] -0.248726*inputs[5] -0.141696*inputs[6] -0.169706*inputs[7] -0.120442*inputs[8] 
		combinations[4] = 0.0637217 -0.263431*inputs[0] -0.154554*inputs[1] -0.1916*inputs[2] -0.0995716*inputs[3] -0.0971573*inputs[4] -0.247724*inputs[5] -0.141641*inputs[6] -0.169304*inputs[7] -0.12028*inputs[8] 
		combinations[5] = 0.063495 -0.264029*inputs[0] -0.155016*inputs[1] -0.192123*inputs[2] -0.0995873*inputs[3] -0.097396*inputs[4] -0.248516*inputs[5] -0.141685*inputs[6] -0.169622*inputs[7] -0.120409*inputs[8] 
		
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

		combinations[0] = -0.287027 -0.643353*inputs[0] -0.643162*inputs[1] -0.643544*inputs[2] -0.643568*inputs[3] -0.64143*inputs[4] -0.643121*inputs[5] 
		
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
