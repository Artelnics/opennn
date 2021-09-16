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
	4 )petal_width

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
 
		self.parameters_number = 83
 
	def scaling_layer(self,inputs):

		outputs = [None] * 4

		outputs[0] = inputs[0]*2.415266275-14.11320591
		outputs[1] = inputs[1]*4.588562965-14.02876568
		outputs[2] = inputs[2]*1.132953048-4.257637501
		outputs[3] = inputs[3]*2.623853445-3.146874905

		return outputs;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 10

		combinations[0] = 0.395957 -0.125231*inputs[0] +0.326583*inputs[1] -0.388694*inputs[2] -0.389443*inputs[3] 
		combinations[1] = -0.374394 -0.420275*inputs[0] +0.36244*inputs[1] -0.533691*inputs[2] -0.274081*inputs[3] 
		combinations[2] = -0.52096 -0.0361089*inputs[0] +0.0703434*inputs[1] +0.549457*inputs[2] +0.420585*inputs[3] 
		combinations[3] = 0.522812 +0.146044*inputs[0] +0.062207*inputs[1] -0.411823*inputs[2] -0.750093*inputs[3] 
		combinations[4] = 0.364621 +0.504047*inputs[0] -0.437882*inputs[1] +0.291139*inputs[2] +0.340977*inputs[3] 
		combinations[5] = 0.384518 +0.541423*inputs[0] -0.367351*inputs[1] +0.498336*inputs[2] +0.172826*inputs[3] 
		combinations[6] = -0.600615 +0.0453112*inputs[0] -0.128768*inputs[1] +0.396013*inputs[2] +0.604464*inputs[3] 
		combinations[7] = 0.412966 +0.0497757*inputs[0] +0.19464*inputs[1] -0.500716*inputs[2] -0.42613*inputs[3] 
		combinations[8] = 0.549809 +0.54521*inputs[0] -0.34135*inputs[1] +0.36204*inputs[2] +0.325535*inputs[3] 
		combinations[9] = 0.401373 +0.0593159*inputs[0] +0.046146*inputs[1] -0.53164*inputs[2] -0.39144*inputs[3] 
		
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

		combinations = [None] * 3

		combinations[0] = -0.122392 +0.45217*inputs[0] +0.346158*inputs[1] -0.409535*inputs[2] +0.427543*inputs[3] -0.399237*inputs[4] -0.354133*inputs[5] -0.433496*inputs[6] +0.391537*inputs[7] -0.503915*inputs[8] +0.454471*inputs[9] 
		combinations[1] = 0.0803996 +0.308758*inputs[0] -0.64641*inputs[1] -0.816637*inputs[2] +0.532274*inputs[3] +0.488053*inputs[4] +0.684737*inputs[5] -0.628385*inputs[6] +0.246522*inputs[7] +0.673086*inputs[8] +0.456811*inputs[9] 
		combinations[2] = -0.0582235 -0.621868*inputs[0] -0.441797*inputs[1] +0.620912*inputs[2] -0.547483*inputs[3] +0.43949*inputs[4] +0.351803*inputs[5] +0.624807*inputs[6] -0.736667*inputs[7] +0.231193*inputs[8] -0.585228*inputs[9] 
		
		activations = [None] * 3

		sum_ = 0;

		sum_ = 	np.exp(combinations[0]) + 	np.exp(combinations[1]) + 	np.exp(combinations[2]);

		activations[0] = np.exp(combinations[0])/sum_;
		activations[1] = np.exp(combinations[1])/sum_;
		activations[2] = np.exp(combinations[2])/sum_;

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
