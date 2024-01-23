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
		
		perceptron_layer_1_output_0 = np.tanh( 0.705909 + (scaled_sepal_lenght*0.144892) + (scaled_sepal_width*0.289479) + (scaled_petal_lenght*0.45477) + (scaled_petal_width*0.916351) )
		perceptron_layer_1_output_1 = np.tanh( 0.478104 + (scaled_sepal_lenght*0.739366) + (scaled_sepal_width*0.0899907) + (scaled_petal_lenght*0.327944) + (scaled_petal_width*0.639982) )
		perceptron_layer_1_output_2 = np.tanh( 0.66575 + (scaled_sepal_lenght*0.299121) + (scaled_sepal_width*0.460243) + (scaled_petal_lenght*0.205209) + (scaled_petal_width*0.525338) )
		
		probabilistic_layer_combinations_0 = 0.766181 +0.488712*perceptron_layer_1_output_0 +0.869745*perceptron_layer_1_output_1 +0.0212694*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_1 = 0.955649 +0.254349*perceptron_layer_1_output_0 +0.544686*perceptron_layer_1_output_1 +0.785985*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_2 = 0.454777 +0.749801*perceptron_layer_1_output_0 +0.105201*perceptron_layer_1_output_1 +0.960574*perceptron_layer_1_output_2 
			
		sum = np.exp(probabilistic_layer_combinations_0) + np.exp(probabilistic_layer_combinations_1) + np.exp(probabilistic_layer_combinations_2)
		
		iri_s_setosa = np.exp(probabilistic_layer_combinations_0)/sum
		iri_s_versicolo_r = np.exp(probabilistic_layer_combinations_1)/sum
		iri_s_virgin_ica = np.exp(probabilistic_layer_combinations_2)/sum
		
		out = [None]*3

		out[0] = iri_s_setosa
		out[1] = iri_s_versicolo_r
		out[2] = iri_s_virgin_ica

		return out


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
