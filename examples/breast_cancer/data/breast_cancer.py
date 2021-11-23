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

		combinations[0] = -1.15157 -1.51288*inputs[0] +2.37549*inputs[1] -1.32676*inputs[2] -0.361859*inputs[3] -0.776358*inputs[4] +1.00973*inputs[5] +1.85039*inputs[6] +1.4841*inputs[7] -0.097832*inputs[8] 
		combinations[1] = -1.34675 +2.69362*inputs[0] -0.0508304*inputs[1] -0.878351*inputs[2] -0.668207*inputs[3] +1.00233*inputs[4] +0.0226824*inputs[5] +0.111524*inputs[6] +0.339845*inputs[7] +1.66189*inputs[8] 
		combinations[2] = 0.908588 +1.16732*inputs[0] +1.04912*inputs[1] +1.30598*inputs[2] +0.0245935*inputs[3] +0.572709*inputs[4] +0.851609*inputs[5] +0.648817*inputs[6] +1.12253*inputs[7] -0.24452*inputs[8] 
		combinations[3] = 0.605686 +0.270562*inputs[0] -0.858275*inputs[1] +0.910153*inputs[2] -0.100442*inputs[3] +2.65025*inputs[4] -1.13053*inputs[5] -1.75976*inputs[6] -0.775442*inputs[7] -0.75319*inputs[8] 
		combinations[4] = -1.92607 -1.96941*inputs[0] -0.885036*inputs[1] +2.01835*inputs[2] +0.982294*inputs[3] -0.346973*inputs[4] +1.15157*inputs[5] +0.630299*inputs[6] +0.479034*inputs[7] +1.11475*inputs[8] 
		combinations[5] = -0.176473 -0.210573*inputs[0] -0.992537*inputs[1] +0.127468*inputs[2] -1.04091*inputs[3] +0.403362*inputs[4] -1.28587*inputs[5] -0.00201292*inputs[6] +0.915769*inputs[7] -0.200227*inputs[8] 
		
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

		combinations[0] = 2.11184 -4.08212*inputs[0] +4.7254*inputs[1] +4.79482*inputs[2] -3.42322*inputs[3] +3.78602*inputs[4] -2.0318*inputs[5] 
		
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
