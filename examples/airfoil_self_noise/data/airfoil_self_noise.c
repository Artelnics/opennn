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

	double perceptron_layer_output_0 = tanh( -0.43066 + (scaled_frequency*-1.37817) + (scaled_angle_of_attack*-0.766317) + (scaled_cho_rd_lenght*-0.345206) + (scaled_free_res_stream_velocity*-0.036344) + (scaled_suction_side_di_splacement_thickness*-1.4823));
	double perceptron_layer_output_1 = tanh( 0.0199577 + (scaled_frequency*-0.45505) + (scaled_angle_of_attack*-1.29019) + (scaled_cho_rd_lenght*-0.777663) + (scaled_free_res_stream_velocity*0.40476) + (scaled_suction_side_di_splacement_thickness*0.257532));
	double perceptron_layer_output_2 = tanh( -0.953694 + (scaled_frequency*0.99192) + (scaled_angle_of_attack*0.201111) + (scaled_cho_rd_lenght*0.601202) + (scaled_free_res_stream_velocity*-0.239695) + (scaled_suction_side_di_splacement_thickness*-1.15738));
	double perceptron_layer_output_3 = tanh( -0.11731 + (scaled_frequency*-0.472005) + (scaled_angle_of_attack*1.35784) + (scaled_cho_rd_lenght*0.0348261) + (scaled_free_res_stream_velocity*0.147726) + (scaled_suction_side_di_splacement_thickness*1.10042));
	double perceptron_layer_output_4 = tanh( 0.226509 + (scaled_frequency*0.886237) + (scaled_angle_of_attack*0.728696) + (scaled_cho_rd_lenght*-0.776285) + (scaled_free_res_stream_velocity*-0.0840911) + (scaled_suction_side_di_splacement_thickness*-0.143528));
	double perceptron_layer_output_5 = tanh( 2.26284 + (scaled_frequency*2.48026) + (scaled_angle_of_attack*0.1249) + (scaled_cho_rd_lenght*0.624151) + (scaled_free_res_stream_velocity*-0.138179) + (scaled_suction_side_di_splacement_thickness*0.365686));
	double perceptron_layer_output_6 = tanh( 1.02217 + (scaled_frequency*0.215311) + (scaled_angle_of_attack*0.789923) + (scaled_cho_rd_lenght*-0.390527) + (scaled_free_res_stream_velocity*0.107331) + (scaled_suction_side_di_splacement_thickness*-0.358407));
	double perceptron_layer_output_7 = tanh( 0.915223 + (scaled_frequency*-0.0905813) + (scaled_angle_of_attack*0.183601) + (scaled_cho_rd_lenght*0.350058) + (scaled_free_res_stream_velocity*-0.0400608) + (scaled_suction_side_di_splacement_thickness*0.202897));
	double perceptron_layer_output_8 = tanh( -0.670966 + (scaled_frequency*-0.863787) + (scaled_angle_of_attack*-0.10811) + (scaled_cho_rd_lenght*-1.54244) + (scaled_free_res_stream_velocity*0.128318) + (scaled_suction_side_di_splacement_thickness*-0.127051));
	double perceptron_layer_output_9 = tanh( 1.57871 + (scaled_frequency*3.20015) + (scaled_angle_of_attack*-0.316131) + (scaled_cho_rd_lenght*0.0568833) + (scaled_free_res_stream_velocity*-0.00996267) + (scaled_suction_side_di_splacement_thickness*0.631031));

	perceptron_layer_output_0 = ( -0.700178 + (perceptron_layer_output_0*0.695348) + (perceptron_layer_output_1*0.797668) + (perceptron_layer_output_2*1.42346) + (perceptron_layer_output_3*1.26254) + (perceptron_layer_output_4*-1.1657) + (perceptron_layer_output_5*3.24271) + (perceptron_layer_output_6*1.30689) + (perceptron_layer_output_7*-1.35722) + (perceptron_layer_output_8*1.24732) + (perceptron_layer_output_9*-1.2756));

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

