'''
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'neural network' method.	
Example:

	sample = [input_1, input_2, input_3, input_4, ...] 	 	
	outputs = neural_network(sample)

	Inputs Names: 	
	                          frequency
                    angle_of_attack
                       chord_lenght
               free-stream_velocity
suction_side_displacement_thickness

Notice that only one sample is allowed as input. DataSetBatch of inputs are not yet implement,	
however you can loop through neural network function in order to get multiple outputs.	
'''

import numpy as np

def scaling_layer(inputs):

	outputs = [None] * 5

	outputs[0] = inputs[0]*1+0
	outputs[1] = inputs[1]*1+0
	outputs[2] = inputs[2]*1+0
	outputs[3] = inputs[3]*1+0
	outputs[4] = inputs[4]*1+0

	return outputs;


def perceptron_layer_0(inputs):

	combinations = [None] * 12

	combinations[0] = 0.634633 -0.481795*inputs[0] -1.6842*inputs[1] -0.149289*inputs[2] -0.32709*inputs[3] +0.551054*inputs[4] 
	combinations[1] = 0.0448996 -0.444546*inputs[0] +0.990089*inputs[1] -0.998241*inputs[2] +0.385156*inputs[3] +0.898606*inputs[4] 
	combinations[2] = -1.18718 +1.11978*inputs[0] +0.773625*inputs[1] -0.842091*inputs[2] -0.0533032*inputs[3] -0.725724*inputs[4] 
	combinations[3] = 0.372593 -0.0672449*inputs[0] +0.313956*inputs[1] -0.753665*inputs[2] +0.126055*inputs[3] -0.156953*inputs[4] 
	combinations[4] = 3.92162 +2.72534*inputs[0] +0.325506*inputs[1] +0.590943*inputs[2] -0.129484*inputs[3] +0.744502*inputs[4] 
	combinations[5] = -2.474 -1.69416*inputs[0] -1.18867*inputs[1] -0.196646*inputs[2] +0.0270279*inputs[3] -1.04437*inputs[4] 
	combinations[6] = -0.306698 +1.07073*inputs[0] -1.46254*inputs[1] +0.134703*inputs[2] -0.27411*inputs[3] -1.28313*inputs[4] 
	combinations[7] = 2.97146 +2.72748*inputs[0] +0.276044*inputs[1] +1.12543*inputs[2] -0.0705648*inputs[3] -0.0206729*inputs[4] 
	combinations[8] = 0.153684 -0.872525*inputs[0] +0.730216*inputs[1] -0.492265*inputs[2] -0.144793*inputs[3] -0.452021*inputs[4] 
	combinations[9] = -0.252472 +0.323011*inputs[0] +0.681261*inputs[1] -0.654571*inputs[2] -0.296809*inputs[3] -0.0611627*inputs[4] 
	combinations[10] = 1.38862 +2.25924*inputs[0] +0.960843*inputs[1] -0.0489971*inputs[2] -0.0789258*inputs[3] -0.104179*inputs[4] 
	combinations[11] = 0.136703 +0.0161593*inputs[0] +0.127963*inputs[1] -0.251298*inputs[2] -0.497724*inputs[3] -0.29338*inputs[4] 
	
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


def perceptron_layer_1(inputs):

	combinations = [None] * 1

	combinations[0] = -0.656562 +0.948865*inputs[0] -0.772016*inputs[1] +1.61825*inputs[2] -0.587631*inputs[3] +2.98619*inputs[4] +1.5254*inputs[5] -1.01491*inputs[6] -2.15285*inputs[7] +1.15962*inputs[8] -0.927031*inputs[9] +1.29812*inputs[10] -0.725414*inputs[11] 
	
	activations = [None] * 1

	activations[0] = combinations[0]

	return activations;


def unscaling_layer(inputs):

	outputs = [None] * 1

	outputs[0] = inputs[0]*18.80350113+122.1835022

	return outputs


def bounding_layer(inputs):

	outputs = [None] * 1

	outputs[0] = inputs[0]

	return outputs


def neural_network(inputs):

	outputs = [None] * len(inputs)

	outputs = scaling_layer(inputs)
	outputs = perceptron_layer_0(outputs)
	outputs = perceptron_layer_1(outputs)
	outputs = unscaling_layer(outputs)
	outputs = bounding_layer(outputs)

	return outputs;

