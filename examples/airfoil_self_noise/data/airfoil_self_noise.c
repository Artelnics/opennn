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

	double perceptron_layer_1_output_0 = tanh( 0.281743 + (scaled_frequency*-0.0498102) + (scaled_angle_of_attack*-0.160158) + (scaled_cho_rd_lenght*0.23739) + (scaled_free_res_stream_velocity*-0.0258554) + (scaled_suction_side_di_splacement_thickness*0.337127) );
	double perceptron_layer_1_output_1 = tanh( -1.19294 + (scaled_frequency*-1.01125) + (scaled_angle_of_attack*1.79716) + (scaled_cho_rd_lenght*0.290144) + (scaled_free_res_stream_velocity*0.0870537) + (scaled_suction_side_di_splacement_thickness*-0.176599) );
	double perceptron_layer_1_output_2 = tanh( 0.805965 + (scaled_frequency*1.42334) + (scaled_angle_of_attack*0.409964) + (scaled_cho_rd_lenght*0.0390813) + (scaled_free_res_stream_velocity*-0.117074) + (scaled_suction_side_di_splacement_thickness*1.42101) );
	double perceptron_layer_1_output_3 = tanh( 0.0701492 + (scaled_frequency*-0.491163) + (scaled_angle_of_attack*-0.962853) + (scaled_cho_rd_lenght*0.00854579) + (scaled_free_res_stream_velocity*0.187136) + (scaled_suction_side_di_splacement_thickness*0.446933) );
	double perceptron_layer_1_output_4 = tanh( -1.75599 + (scaled_frequency*-0.0260065) + (scaled_angle_of_attack*1.37487) + (scaled_cho_rd_lenght*0.0834852) + (scaled_free_res_stream_velocity*-0.185576) + (scaled_suction_side_di_splacement_thickness*0.074427) );
	double perceptron_layer_1_output_5 = tanh( -0.0820673 + (scaled_frequency*1.70111) + (scaled_angle_of_attack*0.880725) + (scaled_cho_rd_lenght*0.209083) + (scaled_free_res_stream_velocity*-0.175279) + (scaled_suction_side_di_splacement_thickness*-0.196982) );
	double perceptron_layer_1_output_6 = tanh( 2.77255 + (scaled_frequency*3.0308) + (scaled_angle_of_attack*-0.0920887) + (scaled_cho_rd_lenght*0.183362) + (scaled_free_res_stream_velocity*-0.0945159) + (scaled_suction_side_di_splacement_thickness*0.891258) );
	double perceptron_layer_1_output_7 = tanh( 1.31498 + (scaled_frequency*0.152372) + (scaled_angle_of_attack*0.295315) + (scaled_cho_rd_lenght*1.18654) + (scaled_free_res_stream_velocity*-0.0172373) + (scaled_suction_side_di_splacement_thickness*-0.0606558) );
	double perceptron_layer_1_output_8 = tanh( -1.48483 + (scaled_frequency*-2.5356) + (scaled_angle_of_attack*0.274121) + (scaled_cho_rd_lenght*-0.190981) + (scaled_free_res_stream_velocity*0.0466201) + (scaled_suction_side_di_splacement_thickness*-0.334505) );
	double perceptron_layer_1_output_9 = tanh( -2.89778 + (scaled_frequency*-1.55566) + (scaled_angle_of_attack*-0.409413) + (scaled_cho_rd_lenght*-2.03125) + (scaled_free_res_stream_velocity*0.16037) + (scaled_suction_side_di_splacement_thickness*-0.151207) );

	double perceptron_layer_2_output_0 = ( -1.51415 + (perceptron_layer_1_output_0*-1.35434) + (perceptron_layer_1_output_1*1.43578) + (perceptron_layer_1_output_2*-1.47971) + (perceptron_layer_1_output_3*1.12681) + (perceptron_layer_1_output_4*-1.42869) + (perceptron_layer_1_output_5*1.11637) + (perceptron_layer_1_output_6*2.85657) + (perceptron_layer_1_output_7*-1.85809) + (perceptron_layer_1_output_8*1.81979) + (perceptron_layer_1_output_9*-1.92395) );

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

