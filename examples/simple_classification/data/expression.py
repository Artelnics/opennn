'''
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'neural network' method.	
Example:

	sample = [input_1, input_2, input_3, input_4, ...] 	 	
	outputs = neural_network(sample)

	Inputs Names: 	
	x1
x2

Notice that only one sample is allowed as input. DataSetBatch of inputs are not yet implement,	
however you can loop through neural network function in order to get multiple outputs.	
'''

import numpy as np

def scaling_layer(inputs):

	outputs = [None] * 2

	outputs[0] = inputs[0]*2.933202028-1.546182394
	outputs[1] = inputs[1]*2.756152153-1.411633015

	return outputs;


def perceptron_layer0(inputs):

	combinations = [None] * 10

	combinations[0] = 80.0221 +26.3867*inputs[0] -73.1528*inputs[1] 
	combinations[1] = 100.942 +27.955*inputs[0] -58.4409*inputs[1] 
	combinations[2] = -27.5201 -186.692*inputs[0] +152.727*inputs[1] 
	combinations[3] = 54.114 -242.031*inputs[0] +55.7108*inputs[1] 
	combinations[4] = -94.6595 -82.3016*inputs[0] +102.887*inputs[1] 
	combinations[5] = 35.7201 +56.8908*inputs[0] +13.3236*inputs[1] 
	combinations[6] = -58.77 -184.138*inputs[0] +120.111*inputs[1] 
	combinations[7] = 70.8452 +78.6646*inputs[0] -65.7619*inputs[1] 
	combinations[8] = 55.0612 -131.155*inputs[0] +162.058*inputs[1] 
	combinations[9] = -6.14748 +90.8408*inputs[0] -14.5926*inputs[1] 
	
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


def probabilistic_layer(inputs):

	combinations = [None] * 1

	combinations[0] = -90.0598 +70.512*inputs[0] +144.18*inputs[1] +284.931*inputs[2] +119.573*inputs[3] -72.2747*inputs[4] +19.1694*inputs[5] +221.424*inputs[6] +5.91975*inputs[7] +297.373*inputs[8] +42.3658*inputs[9] 
	
	activations = [None] * 1

	activations[0] = 1.0/(1.0 + np.exp(-combinations[0]));

	return activations;


def neural_network(inputs):

	outputs = [None] * len(inputs)

	outputs = scaling_layer(inputs)
	outputs = perceptron_layer0(outputs)
	outputs = probabilistic_layer(outputs)

	return outputs;

