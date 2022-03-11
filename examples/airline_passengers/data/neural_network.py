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
	1 )Passengers_lag_1
	2 )Passengers_lag_0

You can predict with a batch of samples using calculate_batch_output method	
IMPORTANT: input batch must be <class 'numpy.ndarray'> type	
Example_1:	
	model = NeuralNetwork()	
	input_batch = np.array([[1, 2], [4, 5]], np.int32)	
	outputs = model.calculate_batch_output(input_batch)
Example_2:	
	input_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})	
	outputs = model.calculate_batch_output(input_batch.values)
'''

import numpy as np

class NeuralNetwork:
 
	def __init__(self):
 
		self.timestep = 3
 
		self.hidden_states = 10*[0]
 
		self.cell_states = 10*[0]
 
		self.parameters_number = 531
 
	def scaling_layer(self,inputs):

		outputs = [None] * 2

		outputs[0] = (inputs[0]-278.4577332)/119.7667694
		outputs[1] = (inputs[1]-280.4154968)/119.2977219

		return outputs;


	def long_short_term_memory_layer(self,inputs):

		forget_gate_combinations = [None] * 10

		forget_gate_combinations[0] = 0.00803859 + inputs[0] * (-0.0275584) + inputs[1] * (0.0261046) + self.hidden_states[0] * (0.00082569) + self.hidden_states[1] * (0.00987047) + self.hidden_states[2] * (0.000632635) + self.hidden_states[3] * (0.000707546) + self.hidden_states[4] * (0.000561909) + self.hidden_states[5] * (0.00120922) + self.hidden_states[6] * (0.00064023) + self.hidden_states[7] * (0.000574543) + self.hidden_states[8] * (0.000813194) + self.hidden_states[9] * (0.000656644) 
		forget_gate_combinations[1] = -0.167302 + inputs[0] * (0.603622) + inputs[1] * (-0.334284) + self.hidden_states[0] * (-0.0148915) + self.hidden_states[1] * (-0.23004) + self.hidden_states[2] * (-0.0173139) + self.hidden_states[3] * (-0.0141899) + self.hidden_states[4] * (-0.0141874) + self.hidden_states[5] * (-0.0164432) + self.hidden_states[6] * (-0.0143579) + self.hidden_states[7] * (-0.0157503) + self.hidden_states[8] * (-0.0146414) + self.hidden_states[9] * (-0.0146752) 
		forget_gate_combinations[2] = 0.00919492 + inputs[0] * (-0.0360598) + inputs[1] * (0.0287887) + self.hidden_states[0] * (0.000754224) + self.hidden_states[1] * (0.0128327) + self.hidden_states[2] * (0.00115855) + self.hidden_states[3] * (0.000767372) + self.hidden_states[4] * (0.00086785) + self.hidden_states[5] * (0.00059654) + self.hidden_states[6] * (0.000854936) + self.hidden_states[7] * (0.00101653) + self.hidden_states[8] * (0.000725777) + self.hidden_states[9] * (0.000839288) 
		forget_gate_combinations[3] = 0.00632986 + inputs[0] * (-0.0235459) + inputs[1] * (0.0224358) + self.hidden_states[0] * (0.000563075) + self.hidden_states[1] * (0.00831864) + self.hidden_states[2] * (0.000500267) + self.hidden_states[3] * (0.000469259) + self.hidden_states[4] * (0.000457713) + self.hidden_states[5] * (0.000646076) + self.hidden_states[6] * (0.000477876) + self.hidden_states[7] * (0.00047148) + self.hidden_states[8] * (0.000543232) + self.hidden_states[9] * (0.000488795) 
		forget_gate_combinations[4] = 0.00712667 + inputs[0] * (-0.0286043) + inputs[1] * (0.0255626) + self.hidden_states[0] * (0.000586877) + self.hidden_states[1] * (0.00998729) + self.hidden_states[2] * (0.000733297) + self.hidden_states[3] * (0.000549671) + self.hidden_states[4] * (0.000545615) + self.hidden_states[5] * (0.000659467) + self.hidden_states[6] * (0.00052277) + self.hidden_states[7] * (0.000650274) + self.hidden_states[8] * (0.000541603) + self.hidden_states[9] * (0.00057251) 
		forget_gate_combinations[5] = 0.00619311 + inputs[0] * (-0.029334) + inputs[1] * (0.0254096) + self.hidden_states[0] * (0.000263868) + self.hidden_states[1] * (0.00972663) + self.hidden_states[2] * (0.0011437) + self.hidden_states[3] * (0.000349555) + self.hidden_states[4] * (0.000627727) + self.hidden_states[5] * (-0.000397297) + self.hidden_states[6] * (0.000556555) + self.hidden_states[7] * (0.00100004) + self.hidden_states[8] * (0.000216395) + self.hidden_states[9] * (0.000551189) 
		forget_gate_combinations[6] = 0.00709542 + inputs[0] * (-0.0278736) + inputs[1] * (0.0251944) + self.hidden_states[0] * (0.000554225) + self.hidden_states[1] * (0.00976061) + self.hidden_states[2] * (0.000721031) + self.hidden_states[3] * (0.00052745) + self.hidden_states[4] * (0.000518485) + self.hidden_states[5] * (0.0006319) + self.hidden_states[6] * (0.000551741) + self.hidden_states[7] * (0.000637425) + self.hidden_states[8] * (0.000573778) + self.hidden_states[9] * (0.000573146) 
		forget_gate_combinations[7] = 0.00883707 + inputs[0] * (-0.035308) + inputs[1] * (0.0291107) + self.hidden_states[0] * (0.000702541) + self.hidden_states[1] * (0.0124451) + self.hidden_states[2] * (0.00111573) + self.hidden_states[3] * (0.000746236) + self.hidden_states[4] * (0.000783241) + self.hidden_states[5] * (0.000760537) + self.hidden_states[6] * (0.000782923) + self.hidden_states[7] * (0.000966905) + self.hidden_states[8] * (0.000673896) + self.hidden_states[9] * (0.00074553) 
		forget_gate_combinations[8] = 0.00919848 + inputs[0] * (-0.0330417) + inputs[1] * (0.0304121) + self.hidden_states[0] * (0.000948539) + self.hidden_states[1] * (0.0116923) + self.hidden_states[2] * (0.000792497) + self.hidden_states[3] * (0.000817106) + self.hidden_states[4] * (0.00066692) + self.hidden_states[5] * (0.00152273) + self.hidden_states[6] * (0.000724714) + self.hidden_states[7] * (0.000644555) + self.hidden_states[8] * (0.000928728) + self.hidden_states[9] * (0.000750055) 
		forget_gate_combinations[9] = 0.00843676 + inputs[0] * (-0.0342668) + inputs[1] * (0.02929) + self.hidden_states[0] * (0.000657434) + self.hidden_states[1] * (0.01187) + self.hidden_states[2] * (0.000989832) + self.hidden_states[3] * (0.000678705) + self.hidden_states[4] * (0.000677252) + self.hidden_states[5] * (0.000891992) + self.hidden_states[6] * (0.00066963) + self.hidden_states[7] * (0.000813492) + self.hidden_states[8] * (0.000620417) + self.hidden_states[9] * (0.000663564) 
		
		forget_gate_activations = [None] * 10

		forget_gate_activations[0] = 		forget_gate_activations[1] = 		forget_gate_activations[2] = 		forget_gate_activations[3] = 		forget_gate_activations[4] = 		forget_gate_activations[5] = 		forget_gate_activations[6] = 		forget_gate_activations[7] = 		forget_gate_activations[8] = 		forget_gate_activations[9] = 		
		input_gate_combinations = [None] * 10

		input_gate_combinations[0] = 0.0176276 + inputs[0] * (-0.0340125) + inputs[1] * (0.0432024) + self.hidden_states[0] * (0.00214194) + self.hidden_states[1] * (0.0190427) + self.hidden_states[2] * (0.00190765) + self.hidden_states[3] * (0.00196144) + self.hidden_states[4] * (0.00176933) + self.hidden_states[5] * (0.00286158) + self.hidden_states[6] * (0.00186664) + self.hidden_states[7] * (0.00172901) + self.hidden_states[8] * (0.00209271) + self.hidden_states[9] * (0.00187516) 
		input_gate_combinations[1] = 0.672754 + inputs[0] * (-0.0872063) + inputs[1] * (-1.00009) + self.hidden_states[0] * (0.059322) + self.hidden_states[1] * (-0.10306) + self.hidden_states[2] * (0.0572434) + self.hidden_states[3] * (0.0596224) + self.hidden_states[4] * (0.0594064) + self.hidden_states[5] * (0.0583082) + self.hidden_states[6] * (0.0593521) + self.hidden_states[7] * (0.0583443) + self.hidden_states[8] * (0.0598722) + self.hidden_states[9] * (0.0595045) 
		input_gate_combinations[2] = 0.0186398 + inputs[0] * (-0.0471833) + inputs[1] * (0.046681) + self.hidden_states[0] * (0.00190146) + self.hidden_states[1] * (0.0239691) + self.hidden_states[2] * (0.00244729) + self.hidden_states[3] * (0.00196171) + self.hidden_states[4] * (0.00200428) + self.hidden_states[5] * (0.00140317) + self.hidden_states[6] * (0.00199994) + self.hidden_states[7] * (0.00224082) + self.hidden_states[8] * (0.00183304) + self.hidden_states[9] * (0.00198593) 
		input_gate_combinations[3] = 0.0148139 + inputs[0] * (-0.027043) + inputs[1] * (0.0392159) + self.hidden_states[0] * (0.00169757) + self.hidden_states[1] * (0.0162539) + self.hidden_states[2] * (0.00166303) + self.hidden_states[3] * (0.00159273) + self.hidden_states[4] * (0.00153037) + self.hidden_states[5] * (0.0019149) + self.hidden_states[6] * (0.00158955) + self.hidden_states[7] * (0.00154728) + self.hidden_states[8] * (0.00169651) + self.hidden_states[9] * (0.00160324) 
		input_gate_combinations[4] = 0.016981 + inputs[0] * (-0.035944) + inputs[1] * (0.0426402) + self.hidden_states[0] * (0.00185864) + self.hidden_states[1] * (0.0195497) + self.hidden_states[2] * (0.00208582) + self.hidden_states[3] * (0.00183362) + self.hidden_states[4] * (0.00180965) + self.hidden_states[5] * (0.00197766) + self.hidden_states[6] * (0.00183283) + self.hidden_states[7] * (0.00191805) + self.hidden_states[8] * (0.00183379) + self.hidden_states[9] * (0.00180264) 
		input_gate_combinations[5] = 0.0230976 + inputs[0] * (-0.0371257) + inputs[1] * (0.044703) + self.hidden_states[0] * (0.00223153) + self.hidden_states[1] * (0.021201) + self.hidden_states[2] * (0.00314467) + self.hidden_states[3] * (0.00228663) + self.hidden_states[4] * (0.00255907) + self.hidden_states[5] * (0.00117858) + self.hidden_states[6] * (0.00247892) + self.hidden_states[7] * (0.00298236) + self.hidden_states[8] * (0.00225506) + self.hidden_states[9] * (0.00247786) 
		input_gate_combinations[6] = 0.0168101 + inputs[0] * (-0.034195) + inputs[1] * (0.0426737) + self.hidden_states[0] * (0.00184788) + self.hidden_states[1] * (0.019059) + self.hidden_states[2] * (0.00206886) + self.hidden_states[3] * (0.00179522) + self.hidden_states[4] * (0.00178979) + self.hidden_states[5] * (0.00195218) + self.hidden_states[6] * (0.00180587) + self.hidden_states[7] * (0.00191085) + self.hidden_states[8] * (0.00181269) + self.hidden_states[9] * (0.00178656) 
		input_gate_combinations[7] = 0.0194946 + inputs[0] * (-0.0466734) + inputs[1] * (0.0468312) + self.hidden_states[0] * (0.00202281) + self.hidden_states[1] * (0.0237444) + self.hidden_states[2] * (0.00258375) + self.hidden_states[3] * (0.00206601) + self.hidden_states[4] * (0.00211476) + self.hidden_states[5] * (0.00196551) + self.hidden_states[6] * (0.00207468) + self.hidden_states[7] * (0.00235314) + self.hidden_states[8] * (0.00197136) + self.hidden_states[9] * (0.00203735) 
		input_gate_combinations[8] = 0.0207704 + inputs[0] * (-0.0443447) + inputs[1] * (0.0473307) + self.hidden_states[0] * (0.002452) + self.hidden_states[1] * (0.0227785) + self.hidden_states[2] * (0.00223134) + self.hidden_states[3] * (0.00230091) + self.hidden_states[4] * (0.0020667) + self.hidden_states[5] * (0.00345606) + self.hidden_states[6] * (0.00213908) + self.hidden_states[7] * (0.00202932) + self.hidden_states[8] * (0.00238984) + self.hidden_states[9] * (0.00216796) 
		input_gate_combinations[9] = 0.0202158 + inputs[0] * (-0.0456931) + inputs[1] * (0.0467394) + self.hidden_states[0] * (0.00219185) + self.hidden_states[1] * (0.023269) + self.hidden_states[2] * (0.0026402) + self.hidden_states[3] * (0.00216116) + self.hidden_states[4] * (0.00213768) + self.hidden_states[5] * (0.00246841) + self.hidden_states[6] * (0.00213548) + self.hidden_states[7] * (0.00239803) + self.hidden_states[8] * (0.00208769) + self.hidden_states[9] * (0.00214266) 
		
		input_gate_activations = [None] * 10

		input_gate_activations[0] = 		input_gate_activations[1] = 		input_gate_activations[2] = 		input_gate_activations[3] = 		input_gate_activations[4] = 		input_gate_activations[5] = 		input_gate_activations[6] = 		input_gate_activations[7] = 		input_gate_activations[8] = 		input_gate_activations[9] = 		
		state_gate_combinations = [None] * 10

		state_gate_combinations[0] = 0.0175706 + inputs[0] * (-0.0337105) + inputs[1] * (0.0431378) + self.hidden_states[0] * (0.00392834) + self.hidden_states[1] * (0.0320447) + self.hidden_states[2] * (0.00336832) + self.hidden_states[3] * (0.00357003) + self.hidden_states[4] * (0.00321513) + self.hidden_states[5] * (0.0051634) + self.hidden_states[6] * (0.00332979) + self.hidden_states[7] * (0.00314822) + self.hidden_states[8] * (0.00386861) + self.hidden_states[9] * (0.00341174) 
		state_gate_combinations[1] = 0.661653 + inputs[0] * (-0.0835521) + inputs[1] * (-1.00013) + self.hidden_states[0] * (0.110543) + self.hidden_states[1] * (-0.270952) + self.hidden_states[2] * (0.103755) + self.hidden_states[3] * (0.111749) + self.hidden_states[4] * (0.110579) + self.hidden_states[5] * (0.10767) + self.hidden_states[6] * (0.110486) + self.hidden_states[7] * (0.106806) + self.hidden_states[8] * (0.11129) + self.hidden_states[9] * (0.10995) 
		state_gate_combinations[2] = 0.0184789 + inputs[0] * (-0.0468289) + inputs[1] * (0.0465518) + self.hidden_states[0] * (0.00381087) + self.hidden_states[1] * (0.0395473) + self.hidden_states[2] * (0.00492971) + self.hidden_states[3] * (0.0038468) + self.hidden_states[4] * (0.00402974) + self.hidden_states[5] * (0.00326195) + self.hidden_states[6] * (0.00399828) + self.hidden_states[7] * (0.0045242) + self.hidden_states[8] * (0.00368071) + self.hidden_states[9] * (0.0039655) 
		state_gate_combinations[3] = 0.0147609 + inputs[0] * (-0.0269642) + inputs[1] * (0.0390816) + self.hidden_states[0] * (0.00296083) + self.hidden_states[1] * (0.0272992) + self.hidden_states[2] * (0.00285801) + self.hidden_states[3] * (0.00278336) + self.hidden_states[4] * (0.00269086) + self.hidden_states[5] * (0.00336498) + self.hidden_states[6] * (0.00275278) + self.hidden_states[7] * (0.00272047) + self.hidden_states[8] * (0.00301081) + self.hidden_states[9] * (0.00280904) 
		state_gate_combinations[4] = 0.0168894 + inputs[0] * (-0.0356992) + inputs[1] * (0.0426167) + self.hidden_states[0] * (0.00336027) + self.hidden_states[1] * (0.0323937) + self.hidden_states[2] * (0.00376366) + self.hidden_states[3] * (0.00327449) + self.hidden_states[4] * (0.00324819) + self.hidden_states[5] * (0.00359522) + self.hidden_states[6] * (0.00327526) + self.hidden_states[7] * (0.00353157) + self.hidden_states[8] * (0.00330966) + self.hidden_states[9] * (0.00330025) 
		state_gate_combinations[5] = 0.0228113 + inputs[0] * (-0.0369745) + inputs[1] * (0.0444857) + self.hidden_states[0] * (0.00364875) + self.hidden_states[1] * (0.0349813) + self.hidden_states[2] * (0.00575005) + self.hidden_states[3] * (0.00385189) + self.hidden_states[4] * (0.00441785) + self.hidden_states[5] * (0.00198366) + self.hidden_states[6] * (0.00423081) + self.hidden_states[7] * (0.00533142) + self.hidden_states[8] * (0.00366572) + self.hidden_states[9] * (0.00423308) 
		state_gate_combinations[6] = 0.0166661 + inputs[0] * (-0.03406) + inputs[1] * (0.0426425) + self.hidden_states[0] * (0.00332321) + self.hidden_states[1] * (0.0317556) + self.hidden_states[2] * (0.00368227) + self.hidden_states[3] * (0.00322603) + self.hidden_states[4] * (0.00318735) + self.hidden_states[5] * (0.00351814) + self.hidden_states[6] * (0.00322785) + self.hidden_states[7] * (0.00342458) + self.hidden_states[8] * (0.00325438) + self.hidden_states[9] * (0.00325033) 
		state_gate_combinations[7] = 0.0192867 + inputs[0] * (-0.0461845) + inputs[1] * (0.0468558) + self.hidden_states[0] * (0.00391057) + self.hidden_states[1] * (0.0390684) + self.hidden_states[2] * (0.00494561) + self.hidden_states[3] * (0.00393503) + self.hidden_states[4] * (0.00402589) + self.hidden_states[5] * (0.00392697) + self.hidden_states[6] * (0.00398945) + self.hidden_states[7] * (0.0044738) + self.hidden_states[8] * (0.00373741) + self.hidden_states[9] * (0.00399429) 
		state_gate_combinations[8] = 0.0205146 + inputs[0] * (-0.0438142) + inputs[1] * (0.0470894) + self.hidden_states[0] * (0.00465488) + self.hidden_states[1] * (0.0378092) + self.hidden_states[2] * (0.00413086) + self.hidden_states[3] * (0.00427789) + self.hidden_states[4] * (0.00382565) + self.hidden_states[5] * (0.00644306) + self.hidden_states[6] * (0.00401473) + self.hidden_states[7] * (0.0037739) + self.hidden_states[8] * (0.00453462) + self.hidden_states[9] * (0.00407657) 
		state_gate_combinations[9] = 0.0200801 + inputs[0] * (-0.0454267) + inputs[1] * (0.0465411) + self.hidden_states[0] * (0.004045) + self.hidden_states[1] * (0.0383498) + self.hidden_states[2] * (0.00491298) + self.hidden_states[3] * (0.00401761) + self.hidden_states[4] * (0.00404207) + self.hidden_states[5] * (0.00459768) + self.hidden_states[6] * (0.00406493) + self.hidden_states[7] * (0.00445457) + self.hidden_states[8] * (0.00389549) + self.hidden_states[9] * (0.00403223) 
		
		state_gate_activations = [None] * 10

		state_gate_activations[0] = np.tanh(state_gate_combinations[0])
		state_gate_activations[1] = np.tanh(state_gate_combinations[1])
		state_gate_activations[2] = np.tanh(state_gate_combinations[2])
		state_gate_activations[3] = np.tanh(state_gate_combinations[3])
		state_gate_activations[4] = np.tanh(state_gate_combinations[4])
		state_gate_activations[5] = np.tanh(state_gate_combinations[5])
		state_gate_activations[6] = np.tanh(state_gate_combinations[6])
		state_gate_activations[7] = np.tanh(state_gate_combinations[7])
		state_gate_activations[8] = np.tanh(state_gate_combinations[8])
		state_gate_activations[9] = np.tanh(state_gate_combinations[9])
		
		output_gate_combinations = [None] * 10

		output_gate_combinations[0] = 0.017127 + inputs[0] * (-0.0111786) + inputs[1] * (0.10849) + self.hidden_states[0] * (0.00428628) + self.hidden_states[1] * (0.0172037) + self.hidden_states[2] * (0.00377335) + self.hidden_states[3] * (0.00394769) + self.hidden_states[4] * (0.00364315) + self.hidden_states[5] * (0.00526597) + self.hidden_states[6] * (0.00374628) + self.hidden_states[7] * (0.00360156) + self.hidden_states[8] * (0.00428816) + self.hidden_states[9] * (0.00387079) 
		output_gate_combinations[1] = 0.1455 + inputs[0] * (0.147166) + inputs[1] * (-1.99185) + self.hidden_states[0] * (0.0199207) + self.hidden_states[1] * (-0.210751) + self.hidden_states[2] * (0.0144708) + self.hidden_states[3] * (0.0214415) + self.hidden_states[4] * (0.0205377) + self.hidden_states[5] * (0.0171515) + self.hidden_states[6] * (0.0203739) + self.hidden_states[7] * (0.0170357) + self.hidden_states[8] * (0.0196968) + self.hidden_states[9] * (0.0192204) 
		output_gate_combinations[2] = 0.0184388 + inputs[0] * (-0.0201843) + inputs[1] * (0.126002) + self.hidden_states[0] * (0.00401114) + self.hidden_states[1] * (0.0252514) + self.hidden_states[2] * (0.00483946) + self.hidden_states[3] * (0.00403232) + self.hidden_states[4] * (0.00421094) + self.hidden_states[5] * (0.00342906) + self.hidden_states[6] * (0.00419413) + self.hidden_states[7] * (0.00460105) + self.hidden_states[8] * (0.00399288) + self.hidden_states[9] * (0.00427033) 
		output_gate_combinations[3] = 0.014811 + inputs[0] * (-0.00687494) + inputs[1] * (0.0948071) + self.hidden_states[0] * (0.00354529) + self.hidden_states[1] * (0.0133907) + self.hidden_states[2] * (0.00345139) + self.hidden_states[3] * (0.00336187) + self.hidden_states[4] * (0.00330354) + self.hidden_states[5] * (0.00378485) + self.hidden_states[6] * (0.00335091) + self.hidden_states[7] * (0.00337505) + self.hidden_states[8] * (0.00362378) + self.hidden_states[9] * (0.00344134) 
		output_gate_combinations[4] = 0.0166319 + inputs[0] * (-0.0124138) + inputs[1] * (0.10831) + self.hidden_states[0] * (0.00376134) + self.hidden_states[1] * (0.017803) + self.hidden_states[2] * (0.00413051) + self.hidden_states[3] * (0.00368202) + self.hidden_states[4] * (0.00368756) + self.hidden_states[5] * (0.00397376) + self.hidden_states[6] * (0.00371116) + self.hidden_states[7] * (0.00394032) + self.hidden_states[8] * (0.00380115) + self.hidden_states[9] * (0.00379224) 
		output_gate_combinations[5] = 0.0191007 + inputs[0] * (-0.0100324) + inputs[1] * (0.112009) + self.hidden_states[0] * (0.00343065) + self.hidden_states[1] * (0.0168231) + self.hidden_states[2] * (0.00521374) + self.hidden_states[3] * (0.00364137) + self.hidden_states[4] * (0.00423076) + self.hidden_states[5] * (0.00235407) + self.hidden_states[6] * (0.00411414) + self.hidden_states[7] * (0.00498509) + self.hidden_states[8] * (0.00353251) + self.hidden_states[9] * (0.00421751) 
		output_gate_combinations[6] = 0.0164688 + inputs[0] * (-0.0109788) + inputs[1] * (0.107115) + self.hidden_states[0] * (0.00378182) + self.hidden_states[1] * (0.0169231) + self.hidden_states[2] * (0.00409478) + self.hidden_states[3] * (0.00366725) + self.hidden_states[4] * (0.00370133) + self.hidden_states[5] * (0.00393889) + self.hidden_states[6] * (0.00371362) + self.hidden_states[7] * (0.00392078) + self.hidden_states[8] * (0.00380429) + self.hidden_states[9] * (0.00377092) 
		output_gate_combinations[7] = 0.0190377 + inputs[0] * (-0.0194365) + inputs[1] * (0.125354) + self.hidden_states[0] * (0.00413762) + self.hidden_states[1] * (0.0240054) + self.hidden_states[2] * (0.00494959) + self.hidden_states[3] * (0.00413782) + self.hidden_states[4] * (0.0042874) + self.hidden_states[5] * (0.00412047) + self.hidden_states[6] * (0.00424499) + self.hidden_states[7] * (0.00465384) + self.hidden_states[8] * (0.00411132) + self.hidden_states[9] * (0.00431758) 
		output_gate_combinations[8] = 0.0198186 + inputs[0] * (-0.0174855) + inputs[1] * (0.125069) + self.hidden_states[0] * (0.00486458) + self.hidden_states[1] * (0.0221853) + self.hidden_states[2] * (0.00439834) + self.hidden_states[3] * (0.00448623) + self.hidden_states[4] * (0.00412505) + self.hidden_states[5] * (0.00640946) + self.hidden_states[6] * (0.00429195) + self.hidden_states[7] * (0.00411958) + self.hidden_states[8] * (0.00481733) + self.hidden_states[9] * (0.00437415) 
		output_gate_combinations[9] = 0.0195348 + inputs[0] * (-0.0184717) + inputs[1] * (0.123993) + self.hidden_states[0] * (0.00427598) + self.hidden_states[1] * (0.0223815) + self.hidden_states[2] * (0.0050048) + self.hidden_states[3] * (0.0042173) + self.hidden_states[4] * (0.00429086) + self.hidden_states[5] * (0.00481208) + self.hidden_states[6] * (0.00427221) + self.hidden_states[7] * (0.00466249) + self.hidden_states[8] * (0.00420605) + self.hidden_states[9] * (0.00435793) 
		
		output_gate_activations = [None] * 10

		output_gate_activations[0] = np.tanh(output_gate_combinations[0])
		output_gate_activations[1] = np.tanh(output_gate_combinations[1])
		output_gate_activations[2] = np.tanh(output_gate_combinations[2])
		output_gate_activations[3] = np.tanh(output_gate_combinations[3])
		output_gate_activations[4] = np.tanh(output_gate_combinations[4])
		output_gate_activations[5] = np.tanh(output_gate_combinations[5])
		output_gate_activations[6] = np.tanh(output_gate_combinations[6])
		output_gate_activations[7] = np.tanh(output_gate_combinations[7])
		output_gate_activations[8] = np.tanh(output_gate_combinations[8])
		output_gate_activations[9] = np.tanh(output_gate_combinations[9])
		
		self.cell_states[0] = forget_gate_activations[0] * self.cell_states[0] + input_gate_activations[0] * state_gate_activations[0] 
		self.cell_states[1] = forget_gate_activations[1] * self.cell_states[1] + input_gate_activations[1] * state_gate_activations[1] 
		self.cell_states[2] = forget_gate_activations[2] * self.cell_states[2] + input_gate_activations[2] * state_gate_activations[2] 
		self.cell_states[3] = forget_gate_activations[3] * self.cell_states[3] + input_gate_activations[3] * state_gate_activations[3] 
		self.cell_states[4] = forget_gate_activations[4] * self.cell_states[4] + input_gate_activations[4] * state_gate_activations[4] 
		self.cell_states[5] = forget_gate_activations[5] * self.cell_states[5] + input_gate_activations[5] * state_gate_activations[5] 
		self.cell_states[6] = forget_gate_activations[6] * self.cell_states[6] + input_gate_activations[6] * state_gate_activations[6] 
		self.cell_states[7] = forget_gate_activations[7] * self.cell_states[7] + input_gate_activations[7] * state_gate_activations[7] 
		self.cell_states[8] = forget_gate_activations[8] * self.cell_states[8] + input_gate_activations[8] * state_gate_activations[8] 
		self.cell_states[9] = forget_gate_activations[9] * self.cell_states[9] + input_gate_activations[9] * state_gate_activations[9] 
 
		
		cell_state_activations = [None] * 10

		cell_state_activations[0] = np.tanh(self.cell_states[0])
		cell_state_activations[1] = np.tanh(self.cell_states[1])
		cell_state_activations[2] = np.tanh(self.cell_states[2])
		cell_state_activations[3] = np.tanh(self.cell_states[3])
		cell_state_activations[4] = np.tanh(self.cell_states[4])
		cell_state_activations[5] = np.tanh(self.cell_states[5])
		cell_state_activations[6] = np.tanh(self.cell_states[6])
		cell_state_activations[7] = np.tanh(self.cell_states[7])
		cell_state_activations[8] = np.tanh(self.cell_states[8])
		cell_state_activations[9] = np.tanh(self.cell_states[9])
		
		self.hidden_states[0] = output_gate_activations[0] * cell_state_activations[0]
		self.hidden_states[1] = output_gate_activations[1] * cell_state_activations[1]
		self.hidden_states[2] = output_gate_activations[2] * cell_state_activations[2]
		self.hidden_states[3] = output_gate_activations[3] * cell_state_activations[3]
		self.hidden_states[4] = output_gate_activations[4] * cell_state_activations[4]
		self.hidden_states[5] = output_gate_activations[5] * cell_state_activations[5]
		self.hidden_states[6] = output_gate_activations[6] * cell_state_activations[6]
		self.hidden_states[7] = output_gate_activations[7] * cell_state_activations[7]
		self.hidden_states[8] = output_gate_activations[8] * cell_state_activations[8]
		self.hidden_states[9] = output_gate_activations[9] * cell_state_activations[9]
 
		
		long_short_term_memory_output = [None] * 10

		long_short_term_memory_output[0] = self.hidden_states[0]
		long_short_term_memory_output[1] = self.hidden_states[1]
		long_short_term_memory_output[2] = self.hidden_states[2]
		long_short_term_memory_output[3] = self.hidden_states[3]
		long_short_term_memory_output[4] = self.hidden_states[4]
		long_short_term_memory_output[5] = self.hidden_states[5]
		long_short_term_memory_output[6] = self.hidden_states[6]
		long_short_term_memory_output[7] = self.hidden_states[7]
		long_short_term_memory_output[8] = self.hidden_states[8]
		long_short_term_memory_output[9] = self.hidden_states[9]

		return long_short_term_memory_output;


	def perceptron_layer_1(self,inputs):

		combinations = [None] * 1

		combinations[0] = 0.244336 +0.212536*inputs[0] -2.43113*inputs[1] +0.236059*inputs[2] +0.194586*inputs[3] +0.21471*inputs[4] +0.217435*inputs[5] +0.212919*inputs[6] +0.23988*inputs[7] +0.240089*inputs[8] +0.241067*inputs[9] 
		
		activations = [None] * 1

		activations[0] = combinations[0]

		return activations;


	def unscaling_layer(self,inputs):

		outputs = [None] * 1

		outputs[0] = inputs[0]*119.1759262+282.62677

		return outputs


	def bounding_layer(self,inputs):

		outputs = [None] * 1

		if inputs[0] < -3.40282e+38:

			outputs[0] = -3.40282e+38

		elif inputs[0] >3.40282e+38:

			outputs[0] = 3.40282e+38

		else:

			outputs[0] = inputs[0]


		return outputs


	def calculate_output(self, inputs):

		output_scaling_layer = self.scaling_layer(inputs)

		output_long_short_term_memory_layer = self.long_short_term_memory_layer(output_scaling_layer)

		output_perceptron_layer_1 = self.perceptron_layer_1(output_long_short_term_memory_layer)

		output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_1)

		output_bounding_layer = self.bounding_layer(output_unscaling_layer)

		return output_bounding_layer


	def calculate_batch_output(self, input_batch):

		output = []

		for i in range(input_batch.shape[0]):

			if(i%self.timestep==0):

				self.hidden_states = 10*[0]

				self.cell_states = 10*[0]

			inputs = list(input_batch[i])

			output_scaling_layer = self.scaling_layer(inputs)

			output_long_short_term_memory_layer = self.long_short_term_memory_layer(output_scaling_layer)

			output_perceptron_layer_1 = self.perceptron_layer_1(output_long_short_term_memory_layer)

			output_unscaling_layer = self.unscaling_layer(output_perceptron_layer_1)

			output_bounding_layer = self.bounding_layer(output_unscaling_layer)

			output = np.append(output,output_bounding_layer, axis=0)

		return output
