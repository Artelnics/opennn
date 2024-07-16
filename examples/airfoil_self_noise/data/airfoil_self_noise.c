/*
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this c file.
You can manage it with the main method, where you 	
can change the values of your inputs. For example:

if we want to add these 3 values (0.3, 2.5 and 1.8)
to our 3 inputs (Input_1, Input_2 and Input_1), the
main program has to look like this:
	
int main(){ 
	vector<float> inputs(3);
	
	const float asdas  = 0.3;
	inputs[0] = asdas;
	const float input2 = 2.5;
	inputs[1] = input2;
	const float input3 = 1.8;
	inputs[2] = input3;
	. . .


Inputs Names:
/*
Artificial Intelligence Techniques SL	
artelnics@artelnics.com	

Your model has been exported to this c file.
You can manage it with the main method, where you 	
can change the values of your inputs. For example:

if we want to add these 3 values (0.3, 2.5 and 1.8)
to our 3 inputs (Input_1, Input_2 and Input_1), the
main program has to look like this:
	
int main(){ 
	vector<float> inputs(3);
	
	const float asdas  = 0.3;
	inputs[0] = asdas;
	const float input2 = 2.5;
	inputs[1] = input2;
	const float input3 = 1.8;
	inputs[2] = input3;
	. . .


Inputs Names:
	0) frequency
	1) angle_of_attack
	2) cho_rd_lenght
	3) free_res_stream_velocity
	4) suction_side_di_splacement_thickness
*/


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

	double perceptron_layer_1_output_0 = tanh( -1.22268 + (scaled_frequency*-1.95977) + (scaled_angle_of_attack*-0.796555) + (scaled_cho_rd_lenght*0.936787) + (scaled_free_res_stream_velocity*0.72767) + (scaled_suction_side_di_splacement_thickness*0.634167) );
	double perceptron_layer_1_output_1 = tanh( 0.231223 + (scaled_frequency*-2.35311) + (scaled_angle_of_attack*0.248932) + (scaled_cho_rd_lenght*-0.72506) + (scaled_free_res_stream_velocity*0.388384) + (scaled_suction_side_di_splacement_thickness*-0.939537) );
	double perceptron_layer_1_output_2 = tanh( 0.00504883 + (scaled_frequency*-1.19117) + (scaled_angle_of_attack*-0.207591) + (scaled_cho_rd_lenght*-1.9228) + (scaled_free_res_stream_velocity*0.236353) + (scaled_suction_side_di_splacement_thickness*-0.651306) );
	double perceptron_layer_1_output_3 = tanh( -0.9337 + (scaled_frequency*-0.165062) + (scaled_angle_of_attack*-1.18128) + (scaled_cho_rd_lenght*-1.51249) + (scaled_free_res_stream_velocity*0.317092) + (scaled_suction_side_di_splacement_thickness*-0.37751) );
	double perceptron_layer_1_output_4 = tanh( -1.26051 + (scaled_frequency*-0.357835) + (scaled_angle_of_attack*-1.11836) + (scaled_cho_rd_lenght*-1.65392) + (scaled_free_res_stream_velocity*0.23729) + (scaled_suction_side_di_splacement_thickness*-0.485936) );
	double perceptron_layer_1_output_5 = tanh( 1.3352 + (scaled_frequency*-2.26105) + (scaled_angle_of_attack*-0.00374809) + (scaled_cho_rd_lenght*-1.4946) + (scaled_free_res_stream_velocity*0.0905592) + (scaled_suction_side_di_splacement_thickness*-1.58351) );
	double perceptron_layer_1_output_6 = tanh( 0.0911167 + (scaled_frequency*-1.79852) + (scaled_angle_of_attack*0.0190505) + (scaled_cho_rd_lenght*-0.738619) + (scaled_free_res_stream_velocity*0.340801) + (scaled_suction_side_di_splacement_thickness*-0.889534) );
	double perceptron_layer_1_output_7 = tanh( 0.308262 + (scaled_frequency*-1.63535) + (scaled_angle_of_attack*-0.318982) + (scaled_cho_rd_lenght*-0.664122) + (scaled_free_res_stream_velocity*0.213817) + (scaled_suction_side_di_splacement_thickness*-0.842149) );
	double perceptron_layer_1_output_8 = tanh( 0.687692 + (scaled_frequency*-1.6759) + (scaled_angle_of_attack*-0.0048879) + (scaled_cho_rd_lenght*-1.14382) + (scaled_free_res_stream_velocity*0.383426) + (scaled_suction_side_di_splacement_thickness*-1.29968) );
	double perceptron_layer_1_output_9 = tanh( 0.581341 + (scaled_frequency*-1.84544) + (scaled_angle_of_attack*-2.40139) + (scaled_cho_rd_lenght*-1.25778) + (scaled_free_res_stream_velocity*0.483748) + (scaled_suction_side_di_splacement_thickness*0.0812451) );

	double perceptron_layer_2_output_0 = ( 0.035622 + (perceptron_layer_1_output_0*0.0977402) + (perceptron_layer_1_output_1*0.0472181) + (perceptron_layer_1_output_2*0.041901) + (perceptron_layer_1_output_3*0.0885276) + (perceptron_layer_1_output_4*0.132985) + (perceptron_layer_1_output_5*0.0799248) + (perceptron_layer_1_output_6*0.031935) + (perceptron_layer_1_output_7*0.0864048) + (perceptron_layer_1_output_8*0.0978213) + (perceptron_layer_1_output_9*0.101047) );

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

