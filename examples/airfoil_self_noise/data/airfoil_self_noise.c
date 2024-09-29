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

	double perceptron_layer_output_0 = tanh( 0.783419 + (scaled_frequency*1.36737) + (scaled_angle_of_attack*0.64857) + (scaled_cho_rd_lenght*0.206079) + (scaled_free_res_stream_velocity*-0.10074) + (scaled_suction_side_di_splacement_thickness*1.29916));
	double perceptron_layer_output_1 = tanh( 1.46653 + (scaled_frequency*2.11101) + (scaled_angle_of_attack*-0.305926) + (scaled_cho_rd_lenght*0.362197) + (scaled_free_res_stream_velocity*-0.0443311) + (scaled_suction_side_di_splacement_thickness*0.175638));
	double perceptron_layer_output_2 = tanh( 0.0103453 + (scaled_frequency*-1.70146) + (scaled_angle_of_attack*-0.863519) + (scaled_cho_rd_lenght*-0.147676) + (scaled_free_res_stream_velocity*0.134617) + (scaled_suction_side_di_splacement_thickness*0.0252888));
	double perceptron_layer_output_3 = tanh( 1.05913 + (scaled_frequency*0.903567) + (scaled_angle_of_attack*-1.85509) + (scaled_cho_rd_lenght*-0.380891) + (scaled_free_res_stream_velocity*-0.0606268) + (scaled_suction_side_di_splacement_thickness*0.17986));
	double perceptron_layer_output_4 = tanh( -2.66614 + (scaled_frequency*-1.24948) + (scaled_angle_of_attack*0.232789) + (scaled_cho_rd_lenght*-1.92057) + (scaled_free_res_stream_velocity*-0.0641314) + (scaled_suction_side_di_splacement_thickness*-0.249244));
	double perceptron_layer_output_5 = tanh( 0.171305 + (scaled_frequency*-0.401354) + (scaled_angle_of_attack*-0.891312) + (scaled_cho_rd_lenght*-0.422879) + (scaled_free_res_stream_velocity*0.270921) + (scaled_suction_side_di_splacement_thickness*0.12908));
	double perceptron_layer_output_6 = tanh( -1.68418 + (scaled_frequency*0.041753) + (scaled_angle_of_attack*1.22784) + (scaled_cho_rd_lenght*0.271428) + (scaled_free_res_stream_velocity*-0.0805406) + (scaled_suction_side_di_splacement_thickness*0.105383));
	double perceptron_layer_output_7 = tanh( -3.13325 + (scaled_frequency*-3.2572) + (scaled_angle_of_attack*0.300492) + (scaled_cho_rd_lenght*-0.200202) + (scaled_free_res_stream_velocity*0.0872755) + (scaled_suction_side_di_splacement_thickness*-1.21936));
	double perceptron_layer_output_8 = tanh( -1.72937 + (scaled_frequency*0.0490424) + (scaled_angle_of_attack*0.149384) + (scaled_cho_rd_lenght*-1.18746) + (scaled_free_res_stream_velocity*-0.0869152) + (scaled_suction_side_di_splacement_thickness*-0.228476));
	double perceptron_layer_output_9 = tanh( 1.21573 + (scaled_frequency*0.325947) + (scaled_angle_of_attack*-0.983932) + (scaled_cho_rd_lenght*-0.602102) + (scaled_free_res_stream_velocity*0.186944) + (scaled_suction_side_di_splacement_thickness*0.703003));

	perceptron_layer_output_0 = ( -0.723118 + (perceptron_layer_output_0*-1.7474) + (perceptron_layer_output_1*-2.30086) + (perceptron_layer_output_2*-1.35327) + (perceptron_layer_output_3*-1.25119) + (perceptron_layer_output_4*-1.65987) + (perceptron_layer_output_5*1.26445) + (perceptron_layer_output_6*-1.59419) + (perceptron_layer_output_7*-3.1747) + (perceptron_layer_output_8*1.82137) + (perceptron_layer_output_9*-1.30046));

	double unscaling_layer_output_0=perceptron_layer_output_0*6.898656845+124.8359451;

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

