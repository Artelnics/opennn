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

	combinations = [None] * 12

	combinations[0] = 0.588867 +0.0949097*inputs[0] +0.264771*inputs[1] -0.561584*inputs[2] +0.941162*inputs[3] +0.14209*inputs[4] 
	combinations[1] = -0.293152 +0.0706787*inputs[0] +0.713379*inputs[1] -0.567383*inputs[2] +0.534973*inputs[3] +0.84137*inputs[4] 
	combinations[2] = -0.390442 -0.0987549*inputs[0] -0.673645*inputs[1] +0.610596*inputs[2] +0.957092*inputs[3] -0.313721*inputs[4] 
	combinations[3] = -0.873169 +0.0940552*inputs[0] +0.380981*inputs[1] +0.101746*inputs[2] +0.857849*inputs[3] +0.844788*inputs[4] 
	combinations[4] = 0.913635 -0.914124*inputs[0] -0.146423*inputs[1] -0.212463*inputs[2] +0.144958*inputs[3] +0.984375*inputs[4] 
	combinations[5] = -0.506836 +0.561707*inputs[0] -0.950073*inputs[1] +0.404114*inputs[2] -0.553467*inputs[3] -0.680115*inputs[4] 
	combinations[6] = 0.0667114 +0.751648*inputs[0] +0.139771*inputs[1] -0.591064*inputs[2] +0.850586*inputs[3] -0.871399*inputs[4] 
	combinations[7] = 0.0141602 +0.433777*inputs[0] -0.744202*inputs[1] +0.111511*inputs[2] -0.0234985*inputs[3] +0.892639*inputs[4] 
	combinations[8] = 0.354004 +0.188232*inputs[0] +0.437744*inputs[1] -0.525208*inputs[2] -0.575989*inputs[3] +0.144592*inputs[4] 
	combinations[9] = 0.228394 +0.0309448*inputs[0] -0.0314941*inputs[1] -0.85791*inputs[2] +0.333679*inputs[3] +0.609253*inputs[4] 
	combinations[10] = 0.602417 -0.730713*inputs[0] +0.714783*inputs[1] -0.555725*inputs[2] -0.970581*inputs[3] -0.961792*inputs[4] 
	combinations[11] = 0.452393 -0.535461*inputs[0] -0.407654*inputs[1] -0.305603*inputs[2] +0.813477*inputs[3] -0.464783*inputs[4] 
	
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

	combinations[0] = -0.930481 +0.55835*inputs[0] -0.754822*inputs[1] +0.438721*inputs[2] -0.0453491*inputs[3] +0.542175*inputs[4] -0.900757*inputs[5] +0.292175*inputs[6] -0.348999*inputs[7] -0.828979*inputs[8] -0.40625*inputs[9] -0.94043*inputs[10] +0.41217*inputs[11] 
	
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

