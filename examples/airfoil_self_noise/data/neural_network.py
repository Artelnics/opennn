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

		outputs[0] = inputs[0]*2.000005484+2.498405038e-08
		outputs[1] = inputs[1]*2.000005722-6.367999049e-06
		outputs[2] = inputs[2]*1.999923944-6.900472272e-06
		outputs[3] = inputs[3]*2.000022173+3.424160241e-05
		outputs[4] = inputs[4]*1.999999285+8.042856621e-07

		return outputs;


	def perceptron_layer_0(self,inputs):

		combinations = [None] * 12

		combinations[0] = 0.224754 +0.606994*inputs[0] +1.03101*inputs[1] +0.732843*inputs[2] -0.243108*inputs[3] +0.490888*inputs[4] 
		combinations[1] = 0.986529 +1.04135*inputs[0] +0.327358*inputs[1] +1.64483*inputs[2] -0.0931937*inputs[3] +0.149599*inputs[4] 
		combinations[2] = 0.969019 +0.360831*inputs[0] -0.947579*inputs[1] +0.674865*inputs[2] +0.072007*inputs[3] +0.749167*inputs[4] 
		combinations[3] = -3.34705 -3.54806*inputs[0] -0.0372676*inputs[1] -0.465287*inputs[2] +0.10605*inputs[3] -0.485448*inputs[4] 
		combinations[4] = 2.54435 +1.88319*inputs[0] +0.747704*inputs[1] +0.0716709*inputs[2] -0.12667*inputs[3] +1.56467*inputs[4] 
		combinations[5] = 1.47216 -0.29741*inputs[0] -0.378461*inputs[1] +0.708409*inputs[2] +0.115108*inputs[3] +0.361261*inputs[4] 
		combinations[6] = 0.90808 +0.948578*inputs[0] -0.93593*inputs[1] +0.17868*inputs[2] -0.244698*inputs[3] +0.229507*inputs[4] 
		combinations[7] = 0.948756 -0.533389*inputs[0] -0.160999*inputs[1] -0.146825*inputs[2] +0.233723*inputs[3] -0.521117*inputs[4] 
		combinations[8] = -0.0317754 +0.228526*inputs[0] +0.290227*inputs[1] -0.881549*inputs[2] +0.00847769*inputs[3] +0.243467*inputs[4] 
		combinations[9] = 1.82596 +2.71689*inputs[0] +0.717719*inputs[1] +0.00129796*inputs[2] -0.0703714*inputs[3] +0.134434*inputs[4] 
		combinations[10] = 0.498034 +0.107057*inputs[0] +0.0282174*inputs[1] -0.0594822*inputs[2] -0.0808321*inputs[3] +0.131948*inputs[4] 
		combinations[11] = -0.833019 +0.557124*inputs[0] +0.104484*inputs[1] -0.509617*inputs[2] -0.200699*inputs[3] +1.04966*inputs[4] 
		
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


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 1

		combinations[0] = -1.19695 -1.0291*inputs[0] -1.27829*inputs[1] +1.12676*inputs[2] -2.60768*inputs[3] +2.38958*inputs[4] -1.49483*inputs[5] -0.840579*inputs[6] +0.963969*inputs[7] -0.932136*inputs[8] -2.01229*inputs[9] -0.727141*inputs[10] +1.00174*inputs[11] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*3.449328661+124.8359451

		return outputs


	def bounding_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]

		return outputs


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_perceptron_layer_0 = self.perceptron_layer_0(output_scaling_layer)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_perceptron_layer_0)

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_1)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_perceptron_layer_0 = self.perceptron_layer_0(output_scaling_layer)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_perceptron_layer_0)

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_1)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
