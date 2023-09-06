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
	0) clump_thickness
	1) cell_size_uni_formity
	2) cell_shape_uni_formity
	3) marginal_adhesion
	4) single_epithelial_cell_size
	5) bare_nuclei
	6) bland_chromatin
	7) normal_nucleoli
	8) mitoses


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
		self.inputs_number = 9
		self.inputs_names = ['clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 'marginal_adhesion', 'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']


	def calculate_outputs(self, inputs):
		clump_thickness = inputs[0]
		cell_size_uni_formity = inputs[1]
		cell_shape_uni_formity = inputs[2]
		marginal_adhesion = inputs[3]
		single_epithelial_cell_size = inputs[4]
		bare_nuclei = inputs[5]
		bland_chromatin = inputs[6]
		normal_nucleoli = inputs[7]
		mitoses = inputs[8]

		scaled_clump_thickness = (clump_thickness-4.442166805)/2.820761204
		scaled_cell_size_uni_formity = (cell_size_uni_formity-3.150805235)/3.065144777
		scaled_cell_shape_uni_formity = (cell_shape_uni_formity-3.215226889)/2.988580704
		scaled_marginal_adhesion = (marginal_adhesion-2.830161095)/2.864562273
		scaled_single_epithelial_cell_size = (single_epithelial_cell_size-3.234260559)/2.223085403
		scaled_bare_nuclei = (bare_nuclei-3.544656038)/3.643857241
		scaled_bland_chromatin = (bland_chromatin-3.445095062)/2.449696541
		scaled_normal_nucleoli = (normal_nucleoli-2.869692564)/3.052666426
		scaled_mitoses = (mitoses-1.603221059)/1.732674122
		
		perceptron_layer_1_output_0 = np.tanh( -0.317386 + (scaled_clump_thickness*-0.410459) + (scaled_cell_size_uni_formity*-0.480636) + (scaled_cell_shape_uni_formity*-0.374227) + (scaled_marginal_adhesion*-0.12416) + (scaled_single_epithelial_cell_size*-0.48882) + (scaled_bare_nuclei*-1.01173) + (scaled_bland_chromatin*-0.6293) + (scaled_normal_nucleoli*-0.618955) + (scaled_mitoses*-0.476184) )
		perceptron_layer_1_output_1 = np.tanh( -0.83306 + (scaled_clump_thickness*-1.89173) + (scaled_cell_size_uni_formity*-0.0182843) + (scaled_cell_shape_uni_formity*0.421053) + (scaled_marginal_adhesion*-0.964893) + (scaled_single_epithelial_cell_size*0.755728) + (scaled_bare_nuclei*-0.279533) + (scaled_bland_chromatin*-1.481) + (scaled_normal_nucleoli*-0.552386) + (scaled_mitoses*-0.944644) )
		perceptron_layer_1_output_2 = np.tanh( -0.549009 + (scaled_clump_thickness*-0.725397) + (scaled_cell_size_uni_formity*-0.0445556) + (scaled_cell_shape_uni_formity*0.809674) + (scaled_marginal_adhesion*-0.57094) + (scaled_single_epithelial_cell_size*-1.14932) + (scaled_bare_nuclei*-1.36711) + (scaled_bland_chromatin*-0.5561) + (scaled_normal_nucleoli*-0.639428) + (scaled_mitoses*-1.0885) )
		perceptron_layer_1_output_3 = np.tanh( 0.441484 + (scaled_clump_thickness*1.31725) + (scaled_cell_size_uni_formity*0.110668) + (scaled_cell_shape_uni_formity*-0.854949) + (scaled_marginal_adhesion*0.85235) + (scaled_single_epithelial_cell_size*0.324598) + (scaled_bare_nuclei*0.548057) + (scaled_bland_chromatin*0.934058) + (scaled_normal_nucleoli*0.248769) + (scaled_mitoses*1.17359) )
		perceptron_layer_1_output_4 = np.tanh( -0.0689438 + (scaled_clump_thickness*0.360964) + (scaled_cell_size_uni_formity*0.0624462) + (scaled_cell_shape_uni_formity*-0.612776) + (scaled_marginal_adhesion*0.520807) + (scaled_single_epithelial_cell_size*0.409905) + (scaled_bare_nuclei*0.703815) + (scaled_bland_chromatin*0.438189) + (scaled_normal_nucleoli*-0.332442) + (scaled_mitoses*0.885094) )
		perceptron_layer_1_output_5 = np.tanh( -0.180121 + (scaled_clump_thickness*-0.224815) + (scaled_cell_size_uni_formity*-0.642703) + (scaled_cell_shape_uni_formity*-0.0383703) + (scaled_marginal_adhesion*-0.272209) + (scaled_single_epithelial_cell_size*-0.394833) + (scaled_bare_nuclei*-1.27944) + (scaled_bland_chromatin*-0.931646) + (scaled_normal_nucleoli*-0.827528) + (scaled_mitoses*-0.689493) )
		
		probabilistic_layer_combinations_0 = -2.25206 -0.526673*perceptron_layer_1_output_0 -2.2054*perceptron_layer_1_output_1 -1.8288*perceptron_layer_1_output_2 +1.36867*perceptron_layer_1_output_3 +1.02119*perceptron_layer_1_output_4 -0.754724*perceptron_layer_1_output_5 
			
		diagnose = 1.0/(1.0 + np.exp(-probabilistic_layer_combinations_0) )
		
		out = [None]*1

		out[0] = diagnose

		return out


	def calculate_batch_output(self, input_batch):
		output_batch = [None]*input_batch.shape[0]

		for i in range(input_batch.shape[0]):

			inputs = list(input_batch[i])

			output = self.calculate_outputs(inputs)

			output_batch[i] = output

		return output_batch
