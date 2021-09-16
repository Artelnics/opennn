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

		outputs[0] = inputs[0]*0.7090284824-3.149622679
		outputs[1] = inputs[1]*0.6524977088-2.055893183
		outputs[2] = inputs[2]*0.6692140102-2.151674747
		outputs[3] = inputs[3]*0.698186934-1.975981593
		outputs[4] = inputs[4]*0.8996505737-2.909704208
		outputs[5] = inputs[5]*0.5488688946-1.945551515
		outputs[6] = inputs[6]*0.8164276481-2.812670946
		outputs[7] = inputs[7]*0.655164957-1.880121946
		outputs[8] = inputs[8]*1.154285192-1.850574255

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 6

		combinations[0] = 2.20766 -1.47506*inputs[0] -0.48682*inputs[1] +0.463718*inputs[2] +0.0948794*inputs[3] -1.40492*inputs[4] +0.584118*inputs[5] -0.0105972*inputs[6] -0.543284*inputs[7] -0.915486*inputs[8] 
		combinations[1] = -0.969232 -0.589404*inputs[0] +0.235624*inputs[1] -0.196451*inputs[2] +3.08901*inputs[3] -0.92075*inputs[4] -1.07742*inputs[5] -0.0759106*inputs[6] +0.579723*inputs[7] -1.23425*inputs[8] 
		combinations[2] = -0.890357 -0.8772*inputs[0] -0.944523*inputs[1] -1.70565*inputs[2] -0.000985879*inputs[3] +0.0686955*inputs[4] -1.69228*inputs[5] -0.86085*inputs[6] -1.07605*inputs[7] -0.864665*inputs[8] 
		combinations[3] = 0.16715 +0.100619*inputs[0] -0.362451*inputs[1] -0.459303*inputs[2] -0.680734*inputs[3] -1.56051*inputs[4] +1.87062*inputs[5] +0.350353*inputs[6] +0.0766626*inputs[7] +1.14837*inputs[8] 
		combinations[4] = 0.727772 +1.43922*inputs[0] -0.57056*inputs[1] -0.617982*inputs[2] -0.11894*inputs[3] +2.54959*inputs[4] -0.401156*inputs[5] -1.3647*inputs[6] +0.108047*inputs[7] +0.641212*inputs[8] 
		combinations[5] = -2.07406 +1.63787*inputs[0] -0.389907*inputs[1] +0.581244*inputs[2] -3.03721*inputs[3] +1.93512*inputs[4] +0.589737*inputs[5] +0.660052*inputs[6] -0.334029*inputs[7] +1.72021*inputs[8] 
		
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

		combinations[0] = 2.96237 -3.06897*inputs[0] +4.1153*inputs[1] -5.16971*inputs[2] +2.2962*inputs[3] -3.24141*inputs[4] +5.0347*inputs[5] 
		
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
