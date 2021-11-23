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
	1 )input_1
	2 )input_2
	3 )input_3
	4 )input_4
	5 )input_5
	6 )input_6
	7 )input_7
	8 )input_8
	9 )input_9

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
 
		self.parameters_number = 67
 
	def scaling_layer(self,inputs):

		outputs = [None] * 9

		outputs[0] = inputs[0]*2-0
		outputs[1] = inputs[1]*2-0
		outputs[2] = inputs[2]*2-0
		outputs[3] = inputs[3]*2-0
		outputs[4] = inputs[4]*2-0
		outputs[5] = inputs[5]*2-0
		outputs[6] = inputs[6]*2-0
		outputs[7] = inputs[7]*2-0
		outputs[8] = inputs[8]*2-0

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 6

		combinations[0] = 0.178894 -0.125624*inputs[0] +0.0511456*inputs[1] +0.0893532*inputs[2] -0.00831765*inputs[3] +0.0186658*inputs[4] +0.0306202*inputs[5] -0.136573*inputs[6] +0.124989*inputs[7] +0.0968349*inputs[8] 
		combinations[1] = -0.000925735 +0.0613518*inputs[0] -0.107184*inputs[1] +0.079784*inputs[2] -0.0239698*inputs[3] -0.074749*inputs[4] -0.14444*inputs[5] +0.0528972*inputs[6] -0.139529*inputs[7] +0.168973*inputs[8] 
		combinations[2] = -0.128784 -0.180153*inputs[0] -0.199181*inputs[1] +0.0569881*inputs[2] -0.0328005*inputs[3] +0.160032*inputs[4] -0.0892752*inputs[5] +0.0505851*inputs[6] +0.138926*inputs[7] +0.109799*inputs[8] 
		combinations[3] = -0.121273 +0.121801*inputs[0] -0.182347*inputs[1] +0.0155856*inputs[2] -0.00879075*inputs[3] -0.107971*inputs[4] -0.133269*inputs[5] -0.119438*inputs[6] +0.0837113*inputs[7] +0.085397*inputs[8] 
		combinations[4] = 0.105786 +0.111183*inputs[0] +0.147138*inputs[1] +0.0103855*inputs[2] +0.00801757*inputs[3] +0.00848976*inputs[4] +0.103201*inputs[5] -0.112198*inputs[6] +0.18452*inputs[7] -0.171548*inputs[8] 
		combinations[5] = 0.0694083 -0.0566387*inputs[0] +0.0374171*inputs[1] -0.111077*inputs[2] -0.0876656*inputs[3] +0.0572642*inputs[4] -0.110258*inputs[5] +0.169323*inputs[6] -0.175536*inputs[7] -0.150226*inputs[8] 
		
		activations = [None] * 6

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])
		activations[3] = np.tanh(combinations[3])
		activations[4] = np.tanh(combinations[4])
		activations[5] = np.tanh(combinations[5])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 1

		combinations[0] = -0.119953 +0.0750488*inputs[0] +0.1887*inputs[1] +0.189846*inputs[2] -0.00315019*inputs[3] -0.193647*inputs[4] +0.00543204*inputs[5] 
		
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
