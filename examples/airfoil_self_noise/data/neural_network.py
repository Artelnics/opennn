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

	outputs[0] = inputs[0]*0.0001010101041-1.020202041
	outputs[1] = inputs[1]*0.09009008855-1
	outputs[2] = inputs[2]*7.158196449-1.181818247
	outputs[3] = inputs[3]*0.0505050458-2.601010084
	outputs[4] = inputs[4]*34.47644806-1.013814092

	return outputs;


def perceptron_layer_0(inputs):

	combinations = [None] * 10

	combinations[0] = 4.21985 +3.07692*inputs[0] +0.519039*inputs[1] +0.729914*inputs[2] -0.129161*inputs[3] +0.453557*inputs[4] 
	combinations[1] = 2.36083 +1.79857*inputs[0] +0.674082*inputs[1] -0.431339*inputs[2] -0.112814*inputs[3] +0.662418*inputs[4] 
	combinations[2] = -3.07361 -2.78209*inputs[0] -0.017739*inputs[1] -0.937102*inputs[2] +0.0814608*inputs[3] -0.2419*inputs[4] 
	combinations[3] = 0.87405 +1.81175*inputs[0] +0.433493*inputs[1] -0.134495*inputs[2] -0.121431*inputs[3] -0.361377*inputs[4] 
	combinations[4] = -0.346777 -0.41985*inputs[0] -0.972057*inputs[1] -1.53145*inputs[2] +0.0600634*inputs[3] +0.127766*inputs[4] 
	combinations[5] = 0.788 -0.949169*inputs[0] -0.179733*inputs[1] +2.26307*inputs[2] +0.160626*inputs[3] -0.178948*inputs[4] 
	combinations[6] = 0.688537 +0.199246*inputs[0] +1.648*inputs[1] +1.06513*inputs[2] -0.112339*inputs[3] -0.416742*inputs[4] 
	combinations[7] = -0.183634 +0.974582*inputs[0] -0.911554*inputs[1] +0.491647*inputs[2] -0.0255457*inputs[3] +0.447485*inputs[4] 
	combinations[8] = 0.678964 +1.28736*inputs[0] -2.01776*inputs[1] +0.0768674*inputs[2] -0.0477109*inputs[3] -0.172833*inputs[4] 
	combinations[9] = 1.38958 -0.332386*inputs[0] +0.698991*inputs[1] +1.42916*inputs[2] +0.0207123*inputs[3] -0.326195*inputs[4] 
	
	activations = [None] * 10

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

	return activations;


def perceptron_layer_1(inputs):

	combinations = [None] * 1

	combinations[0] = 0.384009 +3.12701*inputs[0] -1.74283*inputs[1] +1.99032*inputs[2] +1.76048*inputs[3] -1.19825*inputs[4] +1.06376*inputs[5] -1.44872*inputs[6] -1.18282*inputs[7] -0.948815*inputs[8] -1.65879*inputs[9] 
	
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

