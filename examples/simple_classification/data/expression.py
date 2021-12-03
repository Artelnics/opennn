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

		combinations[0] = -0.0660269 -0.287697*inputs[0] +0.429485*inputs[1] 
		combinations[1] = 0.066027 +0.287647*inputs[0] -0.429415*inputs[1] 
		combinations[2] = 0.0660276 +0.287664*inputs[0] -0.429437*inputs[1] 
		combinations[3] = -0.0660267 -0.287682*inputs[0] +0.429462*inputs[1] 
		combinations[4] = -0.0660276 -0.287693*inputs[0] +0.429478*inputs[1] 
		combinations[5] = -0.0660259 -0.287714*inputs[0] +0.429509*inputs[1] 
		combinations[6] = -0.0660255 -0.28772*inputs[0] +0.429517*inputs[1] 
		combinations[7] = -0.0660237 -0.287572*inputs[0] +0.429312*inputs[1] 
		combinations[8] = 0.066026 +0.287723*inputs[0] -0.42952*inputs[1] 
		combinations[9] = -0.0660262 -0.287643*inputs[0] +0.429408*inputs[1] 
		
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

		combinations[0] = -0.222093 +0.729153*inputs[0] -0.728814*inputs[1] -0.728935*inputs[2] +0.729058*inputs[3] +0.72913*inputs[4] +0.729254*inputs[5] +0.72929*inputs[6] +0.728278*inputs[7] -0.729315*inputs[8] +0.728783*inputs[9] 
		
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
