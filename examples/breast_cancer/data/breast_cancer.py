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
	1 )clump_thickness
	2 )cell_size_uniformity
	3 )cell_shape_uniformity
	4 )marginal_adhesion
	5 )single_epithelial_cell_size
	6 )bare_nuclei
	7 )bland_chromatin
	8 )normal_nucleoli
	9 )mitoses

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

		outputs[0] = inputs[0]*0.7090284824-3.149622679
		outputs[1] = inputs[1]*0.6524977088-2.055893183
		outputs[2] = inputs[2]*0.6692140102-2.151674747
		outputs[3] = inputs[3]*0.698186934-1.975981593
		outputs[4] = inputs[4]*0.8996505737-2.909704208
		outputs[5] = inputs[5]*0.5488688946-1.945551515
		outputs[6] = inputs[6]*0.8164276481-2.812670946
		outputs[7] = inputs[7]*0.655164957-1.880121946
		outputs[8] = inputs[8]*1.154285192-1.850574255

		return outputs;


	def Perceptron_layer_1(self,inputs):

		combinations = [None] * 6

		combinations[0] = 1.33407 -2.481*inputs[0] +0.184429*inputs[1] +1.86367*inputs[2] +1.47486*inputs[3] -1.62622*inputs[4] +0.794078*inputs[5] +0.881198*inputs[6] +0.385195*inputs[7] -1.41988*inputs[8] 
		combinations[1] = 0.467281 +1.77216*inputs[0] +1.36716*inputs[1] +0.109631*inputs[2] +2.65435*inputs[3] +1.17297*inputs[4] -1.68194*inputs[5] -0.922056*inputs[6] +0.486522*inputs[7] +0.664853*inputs[8] 
		combinations[2] = 1.19721 -0.560047*inputs[0] -0.260598*inputs[1] +0.295958*inputs[2] -3.2663*inputs[3] +2.01458*inputs[4] +0.208091*inputs[5] +0.447846*inputs[6] +0.767424*inputs[7] +0.764709*inputs[8] 
		combinations[3] = -0.618251 +2.28048*inputs[0] +0.927049*inputs[1] +2.40188*inputs[2] +1.11692*inputs[3] -1.64661*inputs[4] -1.33956*inputs[5] +1.12664*inputs[6] +0.768983*inputs[7] +0.221292*inputs[8] 
		combinations[4] = 1.16324 +1.8047*inputs[0] -0.0996242*inputs[1] -1.61572*inputs[2] +1.46085*inputs[3] +1.49361*inputs[4] +4.15914*inputs[5] +0.537808*inputs[6] +0.997991*inputs[7] +0.952639*inputs[8] 
		combinations[5] = -1.0991 -0.369718*inputs[0] +2.53937*inputs[1] +2.11851*inputs[2] -0.646452*inputs[3] +0.810625*inputs[4] +0.375829*inputs[5] -0.579907*inputs[6] -0.440093*inputs[7] -0.430474*inputs[8] 
		
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

		combinations[0] = 1.75674 -4.38345*inputs[0] -5.06139*inputs[1] -4.41969*inputs[2] +4.79607*inputs[3] +7.70276*inputs[4] +4.71981*inputs[5] 
		
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
