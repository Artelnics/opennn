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
		
		perceptron_layer_1_output_0 = np.tanh( 1.4765 + (scaled_sepal_lenght*0.0946495) + (scaled_sepal_width*0.261579) + (scaled_petal_lenght*-0.124456) + (scaled_petal_width*-0.926874) )
		perceptron_layer_1_output_1 = np.tanh( 0.6445 + (scaled_sepal_lenght*0.110657) + (scaled_sepal_width*-0.898096) + (scaled_petal_lenght*0.485323) + (scaled_petal_width*0.821016) )
		perceptron_layer_1_output_2 = np.tanh( 1.64926 + (scaled_sepal_lenght*-0.56336) + (scaled_sepal_width*-0.216393) + (scaled_petal_lenght*-0.241243) + (scaled_petal_width*-0.885878) )
		
		probabilistic_layer_combinations_0 = -0.504096 +3.34401*perceptron_layer_1_output_0 -4.21733*perceptron_layer_1_output_1 +2.15817*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_1 = 1.02967 +0.766028*perceptron_layer_1_output_0 +1.5728*perceptron_layer_1_output_1 +0.48875*perceptron_layer_1_output_2 
		probabilistic_layer_combinations_2 = -0.721184 -4.00392*perceptron_layer_1_output_0 +2.97112*perceptron_layer_1_output_1 -2.37397*perceptron_layer_1_output_2 
			
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
