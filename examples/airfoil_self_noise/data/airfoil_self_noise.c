/**
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
/**
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
	2) chord_lenght
	3) free_res_stream_velocity
	4) suction_side_displacement_thickness
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
	const float chord_lenght = inputs[2];
	const float free_res_stream_velocity = inputs[3];
	const float suction_side_displacement_thickness = inputs[4];

double double 	double scaled_frequency = (frequency-2886.380615)/3152.573242;
	double scaled_angle_of_attack = (angle_of_attack-6.782301903)/5.918128014;
	double scaled_chord_lenght = (chord_lenght-0.136548236)/0.09354072809;
	double scaled_free_res_stream_velocity = (free_res_stream_velocity-50.86074448)/15.57278538;
	double scaled_suction_side_displacement_thickness = (suction_side_displacement_thickness-0.01113987993)/0.01315023471;

	double perceptron_layer_1_output_0 = tanh( 2.07627 + (scaled_frequency*0.528796) + (scaled_angle_of_attack*0.391262) + (scaled_chord_lenght*0.599787) + (scaled_free_res_stream_velocity*-0.318633) + (scaled_suction_side_displacement_thickness*-0.436121) );
	double perceptron_layer_1_output_1 = tanh( 2.37527 + (scaled_frequency*-0.632887) + (scaled_angle_of_attack*-1.25413) + (scaled_chord_lenght*0.107052) + (scaled_free_res_stream_velocity*0.0393711) + (scaled_suction_side_displacement_thickness*-0.870149) );
	double perceptron_layer_1_output_2 = tanh( -0.569327 + (scaled_frequency*-0.948761) + (scaled_angle_of_attack*1.13019) + (scaled_chord_lenght*0.404369) + (scaled_free_res_stream_velocity*0.166001) + (scaled_suction_side_displacement_thickness*0.521353) );
	double perceptron_layer_1_output_3 = tanh( 1.9081 + (scaled_frequency*2.24286) + (scaled_angle_of_attack*0.0964783) + (scaled_chord_lenght*0.528599) + (scaled_free_res_stream_velocity*-0.153455) + (scaled_suction_side_displacement_thickness*0.59037) );
	double perceptron_layer_1_output_4 = tanh( -2.01938 + (scaled_frequency*-3.71532) + (scaled_angle_of_attack*0.146756) + (scaled_chord_lenght*-0.109178) + (scaled_free_res_stream_velocity*0.0550664) + (scaled_suction_side_displacement_thickness*-0.531925) );
	double perceptron_layer_1_output_5 = tanh( 2.5989 + (scaled_frequency*-0.742692) + (scaled_angle_of_attack*-0.485469) + (scaled_chord_lenght*1.63213) + (scaled_free_res_stream_velocity*0.165259) + (scaled_suction_side_displacement_thickness*0.12643) );
	double perceptron_layer_1_output_6 = tanh( 1.64794 + (scaled_frequency*2.37116) + (scaled_angle_of_attack*-0.108316) + (scaled_chord_lenght*-0.0504669) + (scaled_free_res_stream_velocity*-0.0844953) + (scaled_suction_side_displacement_thickness*0.294953) );
	double perceptron_layer_1_output_7 = tanh( 0.223635 + (scaled_frequency*1.18491) + (scaled_angle_of_attack*1.1516) + (scaled_chord_lenght*0.390523) + (scaled_free_res_stream_velocity*-0.197042) + (scaled_suction_side_displacement_thickness*1.15873) );
	double perceptron_layer_1_output_8 = tanh( 0.796372 + (scaled_frequency*0.90088) + (scaled_angle_of_attack*-0.0444644) + (scaled_chord_lenght*0.129147) + (scaled_free_res_stream_velocity*0.133678) + (scaled_suction_side_displacement_thickness*0.0527956) );
	double perceptron_layer_1_output_9 = tanh( 0.826256 + (scaled_frequency*2.29744) + (scaled_angle_of_attack*0.361018) + (scaled_chord_lenght*1.37248) + (scaled_free_res_stream_velocity*-0.155758) + (scaled_suction_side_displacement_thickness*-0.0666298) );

	double perceptron_layer_2_output_0 = ( -0.726768 + (perceptron_layer_1_output_0*-2.02798) + (perceptron_layer_1_output_1*0.909777) + (perceptron_layer_1_output_2*0.799011) + (perceptron_layer_1_output_3*2.96479) + (perceptron_layer_1_output_4*3.01927) + (perceptron_layer_1_output_5*-1.13932) + (perceptron_layer_1_output_6*2.71118) + (perceptron_layer_1_output_7*-1.32353) + (perceptron_layer_1_output_8*1.7588) + (perceptron_layer_1_output_9*-1.26556) );

	unscaling_layer_output_0=perceptron_layer_2_output_0*6.898656845+124.8359451;

	scaled_sound_pressure_level = max(-3.402823466e+38, unscaling_layer_output_0);
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
	const float chord_lenght = /*enter your value here*/; 
	inputs[2] = chord_lenght;
	const float free_res_stream_velocity = /*enter your value here*/; 
	inputs[3] = free_res_stream_velocity;
	const float suction_side_displacement_thickness = /*enter your value here*/; 
	inputs[4] = suction_side_displacement_thickness;

	vector<float> outputs(1);

	outputs = calculate_outputs(inputs);

	printf("These are your outputs:\n");
	printf( "scaled_sound_pressure_level: %f \n", outputs[0]);

	return 0;
} 

