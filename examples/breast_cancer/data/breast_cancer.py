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

		combinations[0] = 0.572703 -2.16298*inputs[0] +1.56386*inputs[1] +0.195691*inputs[2] +0.340074*inputs[3] +0.194255*inputs[4] -1.66456*inputs[5] -0.925889*inputs[6] -0.8273*inputs[7] -0.736142*inputs[8] 
		combinations[1] = 0.737539 -0.0979139*inputs[0] -0.415254*inputs[1] +0.561917*inputs[2] +1.1173*inputs[3] -0.791066*inputs[4] -1.34281*inputs[5] +0.0464026*inputs[6] -0.387973*inputs[7] +0.911034*inputs[8] 
		combinations[2] = 0.260788 -0.971577*inputs[0] -0.98226*inputs[1] +0.457504*inputs[2] +1.21048*inputs[3] +0.222127*inputs[4] -0.219719*inputs[5] +0.391202*inputs[6] +1.0935*inputs[7] -0.0486817*inputs[8] 
		combinations[3] = 0.487868 +1.34642*inputs[0] +1.30093*inputs[1] +0.997797*inputs[2] +0.457585*inputs[3] -0.289861*inputs[4] +1.12372*inputs[5] +0.482236*inputs[6] -0.189171*inputs[7] +0.443838*inputs[8] 
		combinations[4] = 0.158362 -0.313299*inputs[0] +0.912137*inputs[1] -0.909995*inputs[2] -0.426303*inputs[3] -0.669366*inputs[4] +0.586743*inputs[5] -1.12265*inputs[6] -0.86263*inputs[7] -1.18633*inputs[8] 
		combinations[5] = -0.0361831 -1.35433*inputs[0] -0.0531736*inputs[1] -0.110873*inputs[2] -1.1073*inputs[3] -0.0277417*inputs[4] -0.710413*inputs[5] -1.19862*inputs[6] +0.419694*inputs[7] -0.766199*inputs[8] 
		
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

		combinations[0] = -0.126564 -4.77998*inputs[0] +2.18057*inputs[1] +2.04594*inputs[2] +3.13914*inputs[3] -2.49956*inputs[4] -2.84234*inputs[5] 
		
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
