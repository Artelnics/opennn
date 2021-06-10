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
	1 )sepal_lenght
	2 )sepal_width
	3 )petal_lenght
	4 )iris_setosa
	5 )iris_versicolor
	6 )iris_virginica

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
 
		self.parameters_number = 25
 
	def scaling_layer(self,inputs):

		outputs = [None] * 6

		outputs[0] = inputs[0]*2.415266275-14.11320591
		outputs[1] = inputs[1]*4.588562965-14.02876568
		outputs[2] = inputs[2]*1.132953048-4.257637501
		outputs[3] = inputs[3]*4.228474617-1.409491658
		outputs[4] = inputs[4]*4.228474617-1.409491658
		outputs[5] = inputs[5]*4.228474617-1.409491658

		return outputs;


	def Perceptron_layer_1(self,inputs):

		combinations = [None] * 3

		combinations[0] = 0.808051 -0.137501*inputs[0] +0.233128*inputs[1] +0.875576*inputs[2] -0.432395*inputs[3] +0.296181*inputs[4] +0.164928*inputs[5] 
		combinations[1] = -0.901862 +0.00684303*inputs[0] +0.0279776*inputs[1] -0.0950193*inputs[2] +0.47241*inputs[3] -0.196155*inputs[4] -0.268482*inputs[5] 
		combinations[2] = -0.738978 +0.0462208*inputs[0] -0.0988235*inputs[1] -0.477717*inputs[2] +0.532404*inputs[3] -0.293018*inputs[4] -0.250419*inputs[5] 
		
		activations = [None] * 3

		activations[0] = np.tanh(combinations[0])
		activations[1] = np.tanh(combinations[1])
		activations[2] = np.tanh(combinations[2])

		return activations;


	def probabilistic_layer(self, inputs):

		combinations = [None] * 1

		combinations[0] = 2.23115 +2.17447*inputs[0] -1.96864*inputs[1] -2.04471*inputs[2] 
		
		activations = [None] * 1

		activations[0] = 1.0/(1.0 + np.exp(-combinations[0]));

		return activations;


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_Perceptron_layer_1 = self.Perceptron_layer_1(output_scaling_layer)

		output_probabilistic_layer = self.probabilistic_layer(output_Perceptron_layer_1)

		return output_probabilistic_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_Perceptron_layer_1 = self.Perceptron_layer_1(output_scaling_layer)

			output_probabilistic_layer = self.probabilistic_layer(output_Perceptron_layer_1)

			output = np.append(output,output_probabilistic_layer, axis=0)

		return output
