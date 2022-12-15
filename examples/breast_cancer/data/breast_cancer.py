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
		
		perceptron_layer_1_output_0 = np.tanh( 0.0580635 + (scaled_clump_thickness*-0.216212) + (scaled_cell_size_uniformity*-0.197006) + (scaled_cell_shape_uniformity*-0.227819) + (scaled_marginal_adhesion*-0.127374) + (scaled_single_epithelial_cell_size*0.0266439) + (scaled_bare_nuclei*-0.156168) + (scaled_bland_chromatin*-0.143736) + (scaled_normal_nucleoli*-0.275137) + (scaled_mitoses*-0.147034) );
		perceptron_layer_1_output_1 = np.tanh( -0.0580506 + (scaled_clump_thickness*0.216311) + (scaled_cell_size_uniformity*0.197089) + (scaled_cell_shape_uniformity*0.227931) + (scaled_marginal_adhesion*0.127414) + (scaled_single_epithelial_cell_size*-0.0267486) + (scaled_bare_nuclei*0.156179) + (scaled_bland_chromatin*0.143769) + (scaled_normal_nucleoli*0.275283) + (scaled_mitoses*0.147106) );
		perceptron_layer_1_output_2 = np.tanh( -0.0580149 + (scaled_clump_thickness*0.216586) + (scaled_cell_size_uniformity*0.197322) + (scaled_cell_shape_uniformity*0.22824) + (scaled_marginal_adhesion*0.127525) + (scaled_single_epithelial_cell_size*-0.0270404) + (scaled_bare_nuclei*0.156209) + (scaled_bland_chromatin*0.143861) + (scaled_normal_nucleoli*0.275687) + (scaled_mitoses*0.147304) );
		perceptron_layer_1_output_3 = np.tanh( -0.0580825 + (scaled_clump_thickness*0.216067) + (scaled_cell_size_uniformity*0.196882) + (scaled_cell_shape_uniformity*0.227656) + (scaled_marginal_adhesion*0.127315) + (scaled_single_epithelial_cell_size*-0.0264899) + (scaled_bare_nuclei*0.156151) + (scaled_bland_chromatin*0.143688) + (scaled_normal_nucleoli*0.274924) + (scaled_mitoses*0.146929) );
		perceptron_layer_1_output_4 = np.tanh( 0.0580168 + (scaled_clump_thickness*-0.216571) + (scaled_cell_size_uniformity*-0.19731) + (scaled_cell_shape_uniformity*-0.228223) + (scaled_marginal_adhesion*-0.127519) + (scaled_single_epithelial_cell_size*0.027025) + (scaled_bare_nuclei*-0.156207) + (scaled_bland_chromatin*-0.143857) + (scaled_normal_nucleoli*-0.275665) + (scaled_mitoses*-0.147293) );
		perceptron_layer_1_output_5 = np.tanh( 0.0579935 + (scaled_clump_thickness*-0.216752) + (scaled_cell_size_uniformity*-0.197463) + (scaled_cell_shape_uniformity*-0.228427) + (scaled_marginal_adhesion*-0.127592) + (scaled_single_epithelial_cell_size*0.0272171) + (scaled_bare_nuclei*-0.156226) + (scaled_bland_chromatin*-0.143917) + (scaled_normal_nucleoli*-0.275931) + (scaled_mitoses*-0.147423) );
		
		probabilistic_layer_combinations_0 = -0.248227 -0.666071*perceptron_layer_1_output_0 +0.666404*perceptron_layer_1_output_1 +0.66733*perceptron_layer_1_output_2 +0.665579*perceptron_layer_1_output_3 -0.667282*perceptron_layer_1_output_4 -0.66789*perceptron_layer_1_output_5 ;
			
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
