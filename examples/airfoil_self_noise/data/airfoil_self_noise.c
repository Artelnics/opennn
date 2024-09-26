// Artificial Intelligence Techniques SL	
// artelnics@artelnics.com	
// Your model has been exported to this c file.
// You can manage it with the main method, where you 	
// can change the values of your inputs. For example:
// if we want to add these 3 values (0.3, 2.5 and 1.8)
// to our 3 inputs (Input_1, Input_2 and Input_1), the
// main program has to look like this:
// 	
// int main(){ 
// 	vector<float> inputs(3);
// 	
// 	const float asdas  = 0.3;
// 	inputs[0] = asdas;
// 	const float input2 = 2.5;
// 	inputs[1] = input2;
// 	const float input3 = 1.8;
// 	inputs[2] = input3;
// 	. . .
// 

// Inputs Names:
// Artificial Intelligence Techniques SL	
// artelnics@artelnics.com	
// Your model has been exported to this c file.
// You can manage it with the main method, where you 	
// can change the values of your inputs. For example:
// if we want to add these 3 values (0.3, 2.5 and 1.8)
// to our 3 inputs (Input_1, Input_2 and Input_1), the
// main program has to look like this:
// 	
// int main(){ 
// 	vector<float> inputs(3);
// 	
// 	const float asdas  = 0.3;
// 	inputs[0] = asdas;
// 	const float input2 = 2.5;
// 	inputs[1] = input2;
// 	const float input3 = 1.8;
// 	inputs[2] = input3;
// 	. . .
// 

// Inputs Names:
	0) frequency
	1) angle_of_attack
	2) cho_rd_lenght
	3) free_res_stream_velocity
	4) suction_side_di_splacement_thickness


#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>


using namespace std;


vector<float> calculate_outputs(const vector<float>& inputs)
{
	const float frequency = inputs[0];
	const float angle_of_attack = inputs[1];
	const float cho_rd_lenght = inputs[2];
	const float free_res_stream_velocity = inputs[3];
	const float suction_side_di_splacement_thickness = inputs[4];

	double scaled_frequency = (frequency-2886.380615)/3152.573242;
	double scaled_angle_of_attack = (angle_of_attack-6.782301903)/5.918128014;
	double scaled_cho_rd_lenght = (cho_rd_lenght-0.136548236)/0.09354072809;
	double scaled_free_res_stream_velocity = (free_res_stream_velocity-50.86074448)/15.57278538;
	double scaled_suction_side_di_splacement_thickness = (suction_side_di_splacement_thickness-0.01113987993)/0.01315023471;

	double perceptron_layer_1_output_0 = tanh( 1.26713 + (scaled_frequency*-0.51216) + (scaled_angle_of_attack*-0.40555) + (scaled_cho_rd_lenght*-0.352785) + (scaled_free_res_stream_velocity*0.372466) + (scaled_suction_side_di_splacement_thickness*-0.399477));
	double perceptron_layer_1_output_1 = tanh( -1.51904 + (scaled_frequency*0.256619) + (scaled_angle_of_attack*-0.0432016) + (scaled_cho_rd_lenght*-0.993426) + (scaled_free_res_stream_velocity*0.112787) + (scaled_suction_side_di_splacement_thickness*0.232773));
	double perceptron_layer_1_output_2 = tanh( -0.329098 + (scaled_frequency*-0.810824) + (scaled_angle_of_attack*-1.29774) + (scaled_cho_rd_lenght*-0.927838) + (scaled_free_res_stream_velocity*0.333309) + (scaled_suction_side_di_splacement_thickness*-0.173634));
	double perceptron_layer_1_output_3 = tanh( 2.19617 + (scaled_frequency*2.67431) + (scaled_angle_of_attack*0.0244207) + (scaled_cho_rd_lenght*0.347087) + (scaled_free_res_stream_velocity*-0.064493) + (scaled_suction_side_di_splacement_thickness*0.389655));
	double perceptron_layer_1_output_4 = tanh( 2.24028 + (scaled_frequency*3.89387) + (scaled_angle_of_attack*-0.349069) + (scaled_cho_rd_lenght*0.196766) + (scaled_free_res_stream_velocity*-0.0201806) + (scaled_suction_side_di_splacement_thickness*0.639687));
	double perceptron_layer_1_output_5 = tanh( 1.34164 + (scaled_frequency*1.96521) + (scaled_angle_of_attack*1.03398) + (scaled_cho_rd_lenght*-0.0602287) + (scaled_free_res_stream_velocity*-0.112321) + (scaled_suction_side_di_splacement_thickness*0.0984556));
	double perceptron_layer_1_output_6 = tanh( 1.16102 + (scaled_frequency*-0.440533) + (scaled_angle_of_attack*-0.350233) + (scaled_cho_rd_lenght*0.481604) + (scaled_free_res_stream_velocity*0.345555) + (scaled_suction_side_di_splacement_thickness*0.138386));
	double perceptron_layer_1_output_7 = tanh( -0.903978 + (scaled_frequency*-1.05925) + (scaled_angle_of_attack*-0.301737) + (scaled_cho_rd_lenght*-1.69443) + (scaled_free_res_stream_velocity*0.108331) + (scaled_suction_side_di_splacement_thickness*0.153441));
	double perceptron_layer_1_output_8 = tanh( 2.16687 + (scaled_frequency*1.41202) + (scaled_angle_of_attack*0.7309) + (scaled_cho_rd_lenght*0.071258) + (scaled_free_res_stream_velocity*-0.075113) + (scaled_suction_side_di_splacement_thickness*1.61964));
	double perceptron_layer_1_output_9 = tanh( 0.588039 + (scaled_frequency*-0.3058) + (scaled_angle_of_attack*-0.473221) + (scaled_cho_rd_lenght*1.08822) + (scaled_free_res_stream_velocity*0.0728023) + (scaled_suction_side_di_splacement_thickness*0.189927));

	double perceptron_layer_2_output_0 = ( -1.21926 + (perceptron_layer_1_output_0*1.13342) + (perceptron_layer_1_output_1*1.49223) + (perceptron_layer_1_output_2*0.983743) + (perceptron_layer_1_output_3*3.61884) + (perceptron_layer_1_output_4*-1.58387) + (perceptron_layer_1_output_5*-1.5153) + (perceptron_layer_1_output_6*-1.26098) + (perceptron_layer_1_output_7*1.14688) + (perceptron_layer_1_output_8*1.99269) + (perceptron_layer_1_output_9*1.3938));

	double unscaling_layer_output_0=perceptron_layer_2_output_0*6.898656845+124.8359451;

	double scaled_sound_pressure_level = max(-3.402823466e+38, unscaling_layer_output_0);
	scaled_sound_pressure_level = min(3.402823466e+38, scaled_sound_pressure_level);

	vector<float> out(1);
	out[0] = scaled_sound_pressure_level;

	return out;
}


int main(){ 

	vector<float> inputs(5); 

	const float frequency = /*enter your value here*/; 
	inputs[0] = frequency;
	const float angle_of_attack = /*enter your value here*/; 
	inputs[1] = angle_of_attack;
	const float cho_rd_lenght = /*enter your value here*/; 
	inputs[2] = cho_rd_lenght;
	const float free_res_stream_velocity = /*enter your value here*/; 
	inputs[3] = free_res_stream_velocity;
	const float suction_side_di_splacement_thickness = /*enter your value here*/; 
	inputs[4] = suction_side_di_splacement_thickness;

	vector<float> outputs(1);

	outputs = calculate_outputs(inputs);

	printf("These are your outputs:\n");
	printf( "scaled_sound_pressure_level: %f \n", outputs[0]);

	return 0;
} 

