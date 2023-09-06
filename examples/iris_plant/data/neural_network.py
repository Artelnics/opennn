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
		
		perceptron_layer_1_output_0 = np.tanh( 0.100419 + (scaled_sepal_lenght*0.0294449) + (scaled_sepal_width*-0.153129) + (scaled_petal_lenght*0.148748) + (scaled_petal_width*0.387044) )
		perceptron_layer_1_output_1 = np.tanh( 0.0908042 + (scaled_sepal_lenght*0.283474) + (scaled_sepal_width*-0.300182) + (scaled_petal_lenght*0.133461) + (scaled_petal_width*0.37122) )
		perceptron_layer_1_output_2 = np.tanh( -0.0502741 + (scaled_sepal_lenght*0.145957) + (scaled_sepal_width*-0.0762085) + (scaled_petal_lenght*0.0320756) + (scaled_petal_width*0.175429) )
		
		probabilistic_layer_combinations_0 = -0.0464519 +0.0948961*perceptron_layer_1_output_0 +0.0304029*perceptron_layer_1_output_1 -0.119937*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_1 = 0.240458 +0.175408*perceptron_layer_1_output_0 -0.197369*perceptron_layer_1_output_1 +0.181568*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_2 = -0.197357 +0.124013*perceptron_layer_1_output_0 +0.329514*perceptron_layer_1_output_1 +0.15323*perceptron_layer_1_output_2 
			
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
