'''
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'neural network' method.	
Example:

	sample = [input_1, input_2, input_3, input_4, ...] 	 	
	outputs = neural_network(sample)

	Inputs Names: 	
	   sepal_lenght
    sepal_width
   petal_lenght
    iris_setosa
iris_versicolor
 iris_virginica

Notice that only one sample is allowed as input. DataSetBatch of inputs are not yet implement,	
however you can loop through neural network function in order to get multiple outputs.	
'''

import numpy as np

def scaling_layer(inputs):

	outputs = [None] * 6

	outputs[0] = inputs[0]*0.555555582-3.388889074
	outputs[1] = inputs[1]*0.8333333135-2.666666508
	outputs[2] = inputs[2]*0.3389830589-1.338983059
	outputs[3] = inputs[3]*2-1
	outputs[4] = inputs[4]*2-1
	outputs[5] = inputs[5]*2-1

	return outputs;


def perceptron_layer0(inputs):

	combinations = [None] * 3

	combinations[0] = 0.643068 +0.05854*inputs[0] -0.277129*inputs[1] +0.887085*inputs[2] -0.705311*inputs[3] -0.0273775*inputs[4] +0.0709821*inputs[5] 
	combinations[1] = 0.83353 -0.0970188*inputs[0] +0.463912*inputs[1] +0.237977*inputs[2] -0.941656*inputs[3] +0.0320173*inputs[4] +0.0745258*inputs[5] 
	combinations[2] = -0.673662 -0.00867726*inputs[0] +0.105652*inputs[1] -0.967524*inputs[2] +0.628631*inputs[3] +0.0067033*inputs[4] +0.0566424*inputs[5] 
	
	activations = [None] * 3

	activations[0] = np.tanh(combinations[0])
	activations[1] = np.tanh(combinations[1])
	activations[2] = np.tanh(combinations[2])

	return activations;


def probabilistic_layer(inputs):

	combinations = [None] * 1

	combinations[0] = 2.82477 +2.32759*inputs[0] +2.79385*inputs[1] -2.36206*inputs[2] 
	
	activations = [None] * 1

	activations[0] = 1.0/(1.0 + np.exp(-combinations[0]));

	return activations;


def neural_network(inputs):

	outputs = [None] * len(inputs)

	outputs = scaling_layer(inputs)
	outputs = perceptron_layer0(outputs)
	outputs = probabilistic_layer(outputs)

	return outputs;

