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
	0) sepal_lenght
	1) sepal_width
	2) petal_lenght
	3) petal_width


You can predict with a batch of samples using calculate_batch_output method	
IMPORTANT: input batch must be <class 'numpy.ndarray'> type	
Example_1:	
	model = NeuralNetwork()	
	input_batch = np.array([[1, 2], [4, 5]])	
	outputs = model.calculate_batch_output(input_batch)
Example_2:	
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})	
	outputs = model.calculate_batch_output(input_batch.values)
''' 


import numpy as np


class NeuralNetwork:
	def __init__(self):
		self.inputs_number = 4
		self.inputs_names = ['sepal_lenght', 'sepal_width', 'petal_lenght', 'petal_width']


	def calculate_outputs(self, inputs):
		sepal_lenght = inputs[0]
		sepal_width = inputs[1]
		petal_lenght = inputs[2]
		petal_width = inputs[3]

		scaled_sepal_lenght = (sepal_lenght-5.843333244)/0.8280661106
		scaled_sepal_width = (sepal_width-3.057333231)/0.4358662963
		scaled_petal_lenght = (petal_lenght-3.757999897)/1.765298247
		scaled_petal_width = (petal_width-1.19933331)/0.762237668
		
		perceptron_layer_1_output_0 = np.tanh( 2.44948 + (scaled_sepal_lenght*-1.42579) + (scaled_sepal_width*2.56502) + (scaled_petal_lenght*-2.67917) + (scaled_petal_width*-3.8776) )
		perceptron_layer_1_output_1 = np.tanh( -2.28658 + (scaled_sepal_lenght*-1.597) + (scaled_sepal_width*0.0771116) + (scaled_petal_lenght*3.89079) + (scaled_petal_width*1.66861) )
		perceptron_layer_1_output_2 = np.tanh( -1.27553 + (scaled_sepal_lenght*-2.07794) + (scaled_sepal_width*4.07168) + (scaled_petal_lenght*-4.45896) + (scaled_petal_width*-4.61889) )
		
		probabilistic_layer_combinations_0 = -0.913727 +1.90729*perceptron_layer_1_output_0 -5.04514*perceptron_layer_1_output_1 +3.83425*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_1 = 4.3078 +1.65102*perceptron_layer_1_output_0 -1.00319*perceptron_layer_1_output_1 -1.49737*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_2 = -2.97608 -3.59247*perceptron_layer_1_output_0 +5.90514*perceptron_layer_1_output_1 -2.33415*perceptron_layer_1_output_2 
			
		sum = np.exp(probabilistic_layer_combinations_0) + np.exp(probabilistic_layer_combinations_1) + np.exp(probabilistic_layer_combinations_2)
		
		iris_setosa = np.exp(probabilistic_layer_combinations_0)/sum
		iris_versicolor = np.exp(probabilistic_layer_combinations_1)/sum
		iris_virginica = np.exp(probabilistic_layer_combinations_2)/sum
		
		out = [None]*3

		out[0] = iris_setosa
		out[1] = iris_versicolor
		out[2] = iris_virginica

		return out


	def calculate_batch_output(self, input_batch):
		output_batch = [None]*input_batch.shape[0]

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output = self.calculate_outputs(inputs)

			output_batch[i] = output

		return output_batch
