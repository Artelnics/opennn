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
	1 )x1
	2 )x2

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
 
		self.parameters_number = 41
 
	def scaling_layer(self,inputs):

		outputs = [None] * 2

		outputs[0] = (inputs[0]-0.4971455932)/0.1903580129
		outputs[1] = (inputs[1]-0.4880183339)/0.1982036084

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 10

		combinations[0] = 0.155447 -0.245956*inputs[0] +0.703458*inputs[1] 
		combinations[1] = -0.160199 +0.246713*inputs[0] -0.705649*inputs[1] 
		combinations[2] = -0.179879 +0.246223*inputs[0] -0.715408*inputs[1] 
		combinations[3] = -0.35023 -3.38235*inputs[0] +1.12981*inputs[1] 
		combinations[4] = 0.152525 -0.247167*inputs[0] +0.703329*inputs[1] 
		combinations[5] = 0.157162 -0.246066*inputs[0] +0.704948*inputs[1] 
		combinations[6] = 0.202233 -0.245457*inputs[0] +0.726402*inputs[1] 
		combinations[7] = 0.155245 -0.245349*inputs[0] +0.7043*inputs[1] 
		combinations[8] = 2.0551 -0.478804*inputs[0] -2.51069*inputs[1] 
		combinations[9] = 0.554089 -3.17236*inputs[0] -3.66315*inputs[1] 
		
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


	def probabilistic_layer(self, inputs):

		combinations = [None] * 1

		combinations[0] = -0.0778721 +1.51648*inputs[0] -1.52377*inputs[1] -1.53565*inputs[2] -2.5192*inputs[3] +1.518*inputs[4] +1.51885*inputs[5] +1.55277*inputs[6] +1.51424*inputs[7] -3.26465*inputs[8] +5.12854*inputs[9] 
		
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
