''' 
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this python file.
You can manage it with the main method where you 	
can change the values of your inputs. For example:

if we want to add these 3 values (0.3, 2.5 and 1.8)
to our 3 inputs (Input_1, Input_2 and Input_1), the
main program has to look like this:

def main ():
	#default_val = 3.1416
	inputs = [None]*3
	
	Id_1 = 0.3
	Id_1 = 2.5
	Id_1 = 1.8
	
	inputs[0] = Input_1
	inputs[1] = Input_2
	inputs[2] = Input_3
	. . .


Inputs Names: 	
	0) clump_thickness
	1) cell_size_uniformity
	2) cell_shape_uniformity
	3) marginal_adhesion
	4) single_epithelial_cell_size
	5) bare_nuclei
	6) bland_chromatin
	7) normal_nucleoli
	8) mitoses


''' 


import math
import numpy as np


class NeuralNetwork:
	def __init__(self):
		self.inputs_number = 9


	def calculate_outputs(self, inputs):
		clump_thickness = inputs[0]
		cell_size_uniformity = inputs[1]
		cell_shape_uniformity = inputs[2]
		marginal_adhesion = inputs[3]
		single_epithelial_cell_size = inputs[4]
		bare_nuclei = inputs[5]
		bland_chromatin = inputs[6]
		normal_nucleoli = inputs[7]
		mitoses = inputs[8]

		scaled_clump_thickness = (clump_thickness-4.442166805)/2.820761204;
		scaled_cell_size_uniformity = (cell_size_uniformity-3.150805235)/3.065144777;
		scaled_cell_shape_uniformity = (cell_shape_uniformity-3.215226889)/2.988580704;
		scaled_marginal_adhesion = (marginal_adhesion-2.830161095)/2.864562273;
		scaled_single_epithelial_cell_size = (single_epithelial_cell_size-3.234260559)/2.223085403;
		scaled_bare_nuclei = (bare_nuclei-3.544656038)/3.643857241;
		scaled_bland_chromatin = (bland_chromatin-3.445095062)/2.449696541;
		scaled_normal_nucleoli = (normal_nucleoli-2.869692564)/3.052666426;
		scaled_mitoses = (mitoses-1.603221059)/1.732674122;
		
		perceptron_layer_1_output_0 = np.tanh( -0.0592967 + (scaled_clump_thickness*0.276099) + (scaled_cell_size_uniformity*0.0823649) + (scaled_cell_shape_uniformity*0.0231975) + (scaled_marginal_adhesion*0.255724) + (scaled_single_epithelial_cell_size*0.053679) + (scaled_bare_nuclei*0.28364) + (scaled_bland_chromatin*0.23913) + (scaled_normal_nucleoli*0.0659976) + (scaled_mitoses*0.130841) );
		perceptron_layer_1_output_1 = np.tanh( 0.0593057 + (scaled_clump_thickness*-0.275866) + (scaled_cell_size_uniformity*-0.0823609) + (scaled_cell_shape_uniformity*-0.0233227) + (scaled_marginal_adhesion*-0.255457) + (scaled_single_epithelial_cell_size*-0.0537224) + (scaled_bare_nuclei*-0.283424) + (scaled_bland_chromatin*-0.238934) + (scaled_normal_nucleoli*-0.066031) + (scaled_mitoses*-0.130737) );
		perceptron_layer_1_output_2 = np.tanh( -0.0592987 + (scaled_clump_thickness*0.275413) + (scaled_cell_size_uniformity*0.0824055) + (scaled_cell_shape_uniformity*0.0236064) + (scaled_marginal_adhesion*0.25494) + (scaled_single_epithelial_cell_size*0.0538086) + (scaled_bare_nuclei*0.283027) + (scaled_bland_chromatin*0.238538) + (scaled_normal_nucleoli*0.0660599) + (scaled_mitoses*0.130527) );
		perceptron_layer_1_output_3 = np.tanh( 0.0592715 + (scaled_clump_thickness*-0.275095) + (scaled_cell_size_uniformity*-0.0824944) + (scaled_cell_shape_uniformity*-0.0238276) + (scaled_marginal_adhesion*-0.25458) + (scaled_single_epithelial_cell_size*-0.0538729) + (scaled_bare_nuclei*-0.28277) + (scaled_bland_chromatin*-0.238246) + (scaled_normal_nucleoli*-0.0660464) + (scaled_mitoses*-0.130375) );
		perceptron_layer_1_output_4 = np.tanh( 0.0593064 + (scaled_clump_thickness*-0.275611) + (scaled_cell_size_uniformity*-0.0823717) + (scaled_cell_shape_uniformity*-0.0234803) + (scaled_marginal_adhesion*-0.255165) + (scaled_single_epithelial_cell_size*-0.0537694) + (scaled_bare_nuclei*-0.283195) + (scaled_bland_chromatin*-0.238714) + (scaled_normal_nucleoli*-0.066055) + (scaled_mitoses*-0.13062) );
		perceptron_layer_1_output_5 = np.tanh( 0.0593036 + (scaled_clump_thickness*-0.275709) + (scaled_cell_size_uniformity*-0.0823749) + (scaled_cell_shape_uniformity*-0.0234216) + (scaled_marginal_adhesion*-0.255278) + (scaled_single_epithelial_cell_size*-0.053752) + (scaled_bare_nuclei*-0.283286) + (scaled_bland_chromatin*-0.238797) + (scaled_normal_nucleoli*-0.0660418) + (scaled_mitoses*-0.130664) );
		
		probabilistic_layer_combinations_0 = -0.238591 +0.675394*perceptron_layer_1_output_0 -0.674767*perceptron_layer_1_output_1 +0.673574*perceptron_layer_1_output_2 -0.672763*perceptron_layer_1_output_3 -0.67409*perceptron_layer_1_output_4 -0.674353*perceptron_layer_1_output_5 ;
			
		diagnose = 1.0/(1.0 + np.exp(-probabilistic_layer_combinations_0) );
		
		out = [None]*1
		out[0] = diagnose

		return out;


	def main (self):
		default_val = 3.1416
		inputs = [None]*9

		clump_thickness = default_val	#Change this value
		inputs[0] = clump_thickness

		cell_size_uniformity = default_val	#Change this value
		inputs[1] = cell_size_uniformity

		cell_shape_uniformity = default_val	#Change this value
		inputs[2] = cell_shape_uniformity

		marginal_adhesion = default_val	#Change this value
		inputs[3] = marginal_adhesion

		single_epithelial_cell_size = default_val	#Change this value
		inputs[4] = single_epithelial_cell_size

		bare_nuclei = default_val	#Change this value
		inputs[5] = bare_nuclei

		bland_chromatin = default_val	#Change this value
		inputs[6] = bland_chromatin

		normal_nucleoli = default_val	#Change this value
		inputs[7] = normal_nucleoli

		mitoses = default_val	#Change this value
		inputs[8] = mitoses


		outputs = NeuralNetwork.calculate_outputs(self, inputs)

		print("\nThese are your outputs:\n")
		print( "\t diagnose:" + str(outputs[0]) + "\n" )

nn = NeuralNetwork()
nn.main()
