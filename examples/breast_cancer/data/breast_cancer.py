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
	0) clump_thickness
	1) cell_size_uni_fo_rmity
	2) cell_shape_uni_fo_rmity
	3) margin_al_adhesion
	4) sin_gle_epithelial_cell_size
	5) bare_nuclei
	6) blan_d_chromatin_
	7) no_rmal_nucleoli
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
		cell_size_uni_fo_rmity = inputs[1]
		cell_shape_uni_fo_rmity = inputs[2]
		margin_al_adhesion = inputs[3]
		sin_gle_epithelial_cell_size = inputs[4]
		bare_nuclei = inputs[5]
		blan_d_chromatin_ = inputs[6]
		no_rmal_nucleoli = inputs[7]
		mitoses = inputs[8]

		scaled_clump_thickness = (clump_thickness-4.442166805)/2.820761204
		scaled_cell_size_uni_fo_rmity = (cell_size_uni_fo_rmity-3.150805235)/3.065144777
		scaled_cell_shape_uni_fo_rmity = (cell_shape_uni_fo_rmity-3.215226889)/2.988580704
		scaled_margin_al_adhesion = (margin_al_adhesion-2.830161095)/2.864562273
		scaled_sin_gle_epithelial_cell_size = (sin_gle_epithelial_cell_size-3.234260559)/2.223085403
		scaled_bare_nuclei = (bare_nuclei-3.544656038)/3.643857241
		scaled_blan_d_chromatin_ = (blan_d_chromatin_-3.445095062)/2.449696541
		scaled_no_rmal_nucleoli = (no_rmal_nucleoli-2.869692564)/3.052666426
		scaled_mitoses = (mitoses-1.603221059)/1.732674122
		
		probabilistic_layer_combinations_0 = 0.787805 +0.108917*scaled_clump_thickness +0.0127159*scaled_cell_size_uni_fo_rmity +0.736585*scaled_cell_shape_uni_fo_rmity +0.565136*scaled_margin_al_adhesion +0.343192*scaled_sin_gle_epithelial_cell_size +0.9082*scaled_bare_nuclei +0.313351*scaled_blan_d_chromatin_ +0.703263*scaled_no_rmal_nucleoli +0.219006*scaled_mitoses 
			
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

def main():

	inputs = []

	clump_thickness = #- ENTER YOUR VALUE HERE -#
	inputs.append(clump_thickness)

	cell_size_uni_fo_rmity = #- ENTER YOUR VALUE HERE -#
	inputs.append(cell_size_uni_fo_rmity)

	cell_shape_uni_fo_rmity = #- ENTER YOUR VALUE HERE -#
	inputs.append(cell_shape_uni_fo_rmity)

	margin_al_adhesion = #- ENTER YOUR VALUE HERE -#
	inputs.append(margin_al_adhesion)

	sin_gle_epithelial_cell_size = #- ENTER YOUR VALUE HERE -#
	inputs.append(sin_gle_epithelial_cell_size)

	bare_nuclei = #- ENTER YOUR VALUE HERE -#
	inputs.append(bare_nuclei)

	blan_d_chromatin_ = #- ENTER YOUR VALUE HERE -#
	inputs.append(blan_d_chromatin_)

	no_rmal_nucleoli = #- ENTER YOUR VALUE HERE -#
	inputs.append(no_rmal_nucleoli)

	mitoses = #- ENTER YOUR VALUE HERE -#
	inputs.append(mitoses)

	nn = NeuralNetwork()
	outputs = nn.calculate_outputs(inputs)
	print(outputs)

main()
