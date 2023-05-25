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
		
		perceptron_layer_1_output_0 = np.tanh( 0.0648078 + (scaled_clump_thickness*-0.216702) + (scaled_cell_size_uni_formity*-0.188283) + (scaled_cell_shape_uni_formity*-0.151687) + (scaled_marginal_adhesion*-0.120968) + (scaled_single_epithelial_cell_size*-0.0258495) + (scaled_bare_nuclei*-0.236793) + (scaled_bland_chromatin*-0.197169) + (scaled_normal_nucleoli*-0.127434) + (scaled_mitoses*-0.170421) )
		perceptron_layer_1_output_1 = np.tanh( 0.0646582 + (scaled_clump_thickness*-0.217122) + (scaled_cell_size_uni_formity*-0.188688) + (scaled_cell_shape_uni_formity*-0.151809) + (scaled_marginal_adhesion*-0.121127) + (scaled_single_epithelial_cell_size*-0.0257215) + (scaled_bare_nuclei*-0.237452) + (scaled_bland_chromatin*-0.197313) + (scaled_normal_nucleoli*-0.127545) + (scaled_mitoses*-0.170937) )
		perceptron_layer_1_output_2 = np.tanh( 0.064643 + (scaled_clump_thickness*-0.217163) + (scaled_cell_size_uni_formity*-0.188728) + (scaled_cell_shape_uni_formity*-0.151821) + (scaled_marginal_adhesion*-0.121142) + (scaled_single_epithelial_cell_size*-0.0257094) + (scaled_bare_nuclei*-0.237518) + (scaled_bland_chromatin*-0.197327) + (scaled_normal_nucleoli*-0.127556) + (scaled_mitoses*-0.170987) )
		perceptron_layer_1_output_3 = np.tanh( 0.0646441 + (scaled_clump_thickness*-0.21716) + (scaled_cell_size_uni_formity*-0.188725) + (scaled_cell_shape_uni_formity*-0.15182) + (scaled_marginal_adhesion*-0.121141) + (scaled_single_epithelial_cell_size*-0.0257099) + (scaled_bare_nuclei*-0.237513) + (scaled_bland_chromatin*-0.197326) + (scaled_normal_nucleoli*-0.127556) + (scaled_mitoses*-0.170984) )
		perceptron_layer_1_output_4 = np.tanh( 0.0647373 + (scaled_clump_thickness*-0.216906) + (scaled_cell_size_uni_formity*-0.188478) + (scaled_cell_shape_uni_formity*-0.151747) + (scaled_marginal_adhesion*-0.121046) + (scaled_single_epithelial_cell_size*-0.0257855) + (scaled_bare_nuclei*-0.237109) + (scaled_bland_chromatin*-0.197242) + (scaled_normal_nucleoli*-0.127488) + (scaled_mitoses*-0.170671) )
		perceptron_layer_1_output_5 = np.tanh( -0.0647159 + (scaled_clump_thickness*0.216967) + (scaled_cell_size_uni_formity*0.188536) + (scaled_cell_shape_uni_formity*0.151764) + (scaled_marginal_adhesion*0.121069) + (scaled_single_epithelial_cell_size*0.0257672) + (scaled_bare_nuclei*0.237204) + (scaled_bland_chromatin*0.197263) + (scaled_normal_nucleoli*0.127504) + (scaled_mitoses*0.170746) )
		
		probabilistic_layer_combinations_0 = -0.244314 -0.650109*perceptron_layer_1_output_0 -0.651635*perceptron_layer_1_output_1 -0.651785*perceptron_layer_1_output_2 -0.651776*perceptron_layer_1_output_3 -0.650851*perceptron_layer_1_output_4 +0.65107*perceptron_layer_1_output_5 
			
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
