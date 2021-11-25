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
	1 )frequency
	2 )angle_of_attack
	3 )chord_lenght
	4 )free-stream_velocity
	5 )suction_side_displacement_thickness

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
 
		self.parameters_number = 85
 
	def scaling_layer(self,inputs):

		outputs = [None] * 5

		outputs[0] = inputs[0]*0.0006344023859-1.831126809
		outputs[1] = inputs[1]*0.3379446864-2.292042971
		outputs[2] = inputs[2]*21.38105965-2.919546127
		outputs[3] = inputs[3]*0.1284291446-6.532002926
		outputs[4] = inputs[4]*152.0885315-1.694248438

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 12

		combinations[0] = -2.45171 -2.83248*inputs[0] -0.0154837*inputs[1] -0.285817*inputs[2] +0.0798651*inputs[3] -0.555983*inputs[4] 
		combinations[1] = 0.350715 -0.342367*inputs[0] -0.534176*inputs[1] +0.996336*inputs[2] +0.0685724*inputs[3] +0.1939*inputs[4] 
		combinations[2] = -1.60804 -2.99799*inputs[0] +0.215704*inputs[1] -0.117082*inputs[2] -0.00522491*inputs[3] -0.635365*inputs[4] 
		combinations[3] = -2.01765 -0.0712848*inputs[0] +1.63812*inputs[1] +0.193503*inputs[2] -0.277157*inputs[3] +0.0365911*inputs[4] 
		combinations[4] = 0.625705 -0.389163*inputs[0] +0.847483*inputs[1] -0.325186*inputs[2] -0.037907*inputs[3] -0.46383*inputs[4] 
		combinations[5] = 1.43889 +1.58736*inputs[0] +0.274995*inputs[1] +2.04122*inputs[2] -0.0924125*inputs[3] +0.121852*inputs[4] 
		combinations[6] = 0.366792 +0.643992*inputs[0] +1.03961*inputs[1] +0.177762*inputs[2] -0.27856*inputs[3] +0.875626*inputs[4] 
		combinations[7] = 0.92906 +0.101344*inputs[0] +0.550531*inputs[1] +0.840729*inputs[2] -0.0821249*inputs[3] -0.256404*inputs[4] 
		combinations[8] = -1.73428 -1.40604*inputs[0] -0.691719*inputs[1] -1.43395*inputs[2] +0.127188*inputs[3] +0.143665*inputs[4] 
		combinations[9] = -1.26807 +0.509053*inputs[0] +0.328064*inputs[1] -0.56155*inputs[2] -0.14906*inputs[3] +0.00631917*inputs[4] 
		combinations[10] = -1.29608 -1.19275*inputs[0] +1.72962*inputs[1] +0.184001*inputs[2] +0.0140756*inputs[3] -0.34267*inputs[4] 
		combinations[11] = -0.343156 -1.22936*inputs[0] +0.707738*inputs[1] -0.0582559*inputs[2] -0.0870025*inputs[3] -0.612563*inputs[4] 
		
		activations = [None] * 12

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])
		activations[6] = np.tanh(combinations[6])
		activations[7] = np.tanh(combinations[7])
		activations[8] = np.tanh(combinations[8])
		activations[9] = np.tanh(combinations[9])
		activations[10] = np.tanh(combinations[10])
		activations[11] = np.tanh(combinations[11])

		return activations;


	def perceptron_layer_2(self,inputs):

		combinations = [None] * 1

		combinations[0] = -1.13819 -2.77017*inputs[0] +1.20276*inputs[1] +1.73102*inputs[2] -1.67012*inputs[3] +0.998875*inputs[4] -1.51777*inputs[5] -1.15415*inputs[6] -1.45583*inputs[7] -1.9713*inputs[8] +1.74127*inputs[9] +1.70445*inputs[10] -1.04145*inputs[11] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*3.449333668+124.8359604

		return outputs


	def bounding_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]

		return outputs


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

		output_perceptron_layer_2 = self.perceptron_layer_2(output_perceptron_layer_1)

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_2)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_scaling_layer)

			output_perceptron_layer_2 = self.perceptron_layer_2(output_perceptron_layer_1)

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_2)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
