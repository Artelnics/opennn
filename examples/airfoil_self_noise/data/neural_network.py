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

	combinations[0] = -0.805298 -0.396118*inputs[0] +0.45874*inputs[1] +0.941162*inputs[2] +0.437805*inputs[3] -0.13855*inputs[4] 
	combinations[1] = 0.369934 +0.614502*inputs[0] -0.0108032*inputs[1] -0.154968*inputs[2] +0.692383*inputs[3] +0.842102*inputs[4] 
	combinations[2] = 0.628906 -0.49176*inputs[0] -0.407166*inputs[1] -0.196045*inputs[2] +0.634338*inputs[3] +0.17688*inputs[4] 
	combinations[3] = 0.692261 -0.826355*inputs[0] -0.866516*inputs[1] -0.0663452*inputs[2] -0.0968018*inputs[3] +0.657043*inputs[4] 
	combinations[4] = 0.398438 +0.0903931*inputs[0] +0.657593*inputs[1] -0.502075*inputs[2] +0.920593*inputs[3] -0.886353*inputs[4] 
	combinations[5] = -0.659607 -0.724609*inputs[0] -0.80481*inputs[1] -0.662842*inputs[2] +0.858704*inputs[3] +0.497498*inputs[4] 
	combinations[6] = 0.0861816 -0.166809*inputs[0] -0.517334*inputs[1] -0.6745*inputs[2] -0.858826*inputs[3] -0.491821*inputs[4] 
	combinations[7] = 0.875122 +0.424988*inputs[0] +0.240906*inputs[1] +0.23407*inputs[2] -0.487122*inputs[3] +0.158142*inputs[4] 
	combinations[8] = -0.794556 -0.993225*inputs[0] +0.419128*inputs[1] +0.372864*inputs[2] +0.74292*inputs[3] -0.47229*inputs[4] 
	combinations[9] = -0.205933 +0.721985*inputs[0] -0.206726*inputs[1] -0.282593*inputs[2] +0.229492*inputs[3] -0.255066*inputs[4] 
	combinations[10] = -0.722961 -0.672241*inputs[0] -0.660706*inputs[1] +0.582214*inputs[2] -0.65155*inputs[3] -0.416809*inputs[4] 
	combinations[11] = 0.119446 +0.2229*inputs[0] +0.438965*inputs[1] -0.115784*inputs[2] +0.771179*inputs[3] +0.933289*inputs[4] 
	
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

	combinations[0] = -0.49585 -0.959167*inputs[0] +0.420654*inputs[1] +0.0996704*inputs[2] -0.757263*inputs[3] -0.634399*inputs[4] -0.937561*inputs[5] +0.610901*inputs[6] +0.543823*inputs[7] +0.524658*inputs[8] -0.300659*inputs[9] +0.455688*inputs[10] -0.549988*inputs[11] 
	
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

