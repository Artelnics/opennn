''' 
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the 'NeuralNetwork' class.	
Example:

	model = NeuralNetwork()	
	sample = [input_1, input_2, input_3, input_4, ...]	
	outputs = model.calculate_outputs(sample)


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
		
		perceptron_layer_1_output_0 = np.tanh( -5.60777 + (scaled_sepal_lenght*-0.685954) + (scaled_sepal_width*-0.598833) + (scaled_petal_lenght*6.30155) + (scaled_petal_width*2.63061) )
		perceptron_layer_1_output_1 = np.tanh( 0.976856 + (scaled_sepal_lenght*0.555019) + (scaled_sepal_width*-0.567254) + (scaled_petal_lenght*1.06559) + (scaled_petal_width*1.11434) )
		perceptron_layer_1_output_2 = np.tanh( -4.83908 + (scaled_sepal_lenght*-0.476799) + (scaled_sepal_width*-0.587606) + (scaled_petal_lenght*5.13233) + (scaled_petal_width*2.50134) )
		perceptron_layer_1_output_3 = np.tanh( 1.19863 + (scaled_sepal_lenght*1.02613) + (scaled_sepal_width*-0.362308) + (scaled_petal_lenght*0.995916) + (scaled_petal_width*1.17586) )
		perceptron_layer_1_output_4 = np.tanh( 1.22938 + (scaled_sepal_lenght*0.176014) + (scaled_sepal_width*-0.339203) + (scaled_petal_lenght*1.3905) + (scaled_petal_width*1.44477) )
		
		probabilistic_layer_combinations_0 = 3.28519 -1.49936*perceptron_layer_1_output_0 -0.708341*perceptron_layer_1_output_1 -0.00310124*perceptron_layer_1_output_2 -2.21698*perceptron_layer_1_output_3 -0.780115*perceptron_layer_1_output_4 
		probabilistic_layer_combinations_1 = -1.26361 -5.48409*perceptron_layer_1_output_0 +1.91626*perceptron_layer_1_output_1 -3.5194*perceptron_layer_1_output_2 +3.13697*perceptron_layer_1_output_3 +2.88783*perceptron_layer_1_output_4 
		probabilistic_layer_combinations_2 = 1.60921 +5.78282*perceptron_layer_1_output_0 +1.64901*perceptron_layer_1_output_1 +4.23274*perceptron_layer_1_output_2 +1.3884*perceptron_layer_1_output_3 +1.25327*perceptron_layer_1_output_4 
			
		sum = np.exp(probabilistic_layer_combinations_0) + np.exp(probabilistic_layer_combinations_1) + np.exp(probabilistic_layer_combinations_2)
		
		iri_s_setosa = np.exp(probabilistic_layer_combinations_0)/sum
		iri_s_versicolo_r = np.exp(probabilistic_layer_combinations_1)/sum
		iri_s_virgin_ica = np.exp(probabilistic_layer_combinations_2)/sum
		
		out = [None]*3

		out[0] = iri_s_setosa
		out[1] = iri_s_versicolo_r
		out[2] = iri_s_virgin_ica

		return out, sample_autoassociation_distance, sample_autoassociation_variables_distance


	def calculate_batch_output(self, input_batch):
		output_batch = [None]*input_batch.shape[0]

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output = self.calculate_outputs(inputs)

			output_batch[i] = output

		return output_batch

def main():

	inputs = []

	sepal_lenght = #- ENTER YOUR VALUE HERE -#
	inputs.append(sepal_lenght)

	sepal_width = #- ENTER YOUR VALUE HERE -#
	inputs.append(sepal_width)

	petal_lenght = #- ENTER YOUR VALUE HERE -#
	inputs.append(petal_lenght)

	petal_width = #- ENTER YOUR VALUE HERE -#
	inputs.append(petal_width)

	nn = NeuralNetwork()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

main()
