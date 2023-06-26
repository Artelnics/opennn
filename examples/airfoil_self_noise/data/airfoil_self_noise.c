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

	double perceptron_layer_1_output_0 = tanh( 0.0913045 + (scaled_frequency*-0.282807) + (scaled_angle_of_attack*0.0928292) + (scaled_chord_lenght*-0.300315) + (scaled_free_res_stream_velocity*0.0290196) + (scaled_suction_side_displacement_thickness*-0.223754) );
	double perceptron_layer_1_output_1 = tanh( -0.193199 + (scaled_frequency*0.188383) + (scaled_angle_of_attack*-0.00210555) + (scaled_chord_lenght*-0.283065) + (scaled_free_res_stream_velocity*0.381337) + (scaled_suction_side_displacement_thickness*0.0848709) );
	double perceptron_layer_1_output_2 = tanh( 1.12202 + (scaled_frequency*-1.04179) + (scaled_angle_of_attack*-0.527808) + (scaled_chord_lenght*-0.711294) + (scaled_free_res_stream_velocity*0.10131) + (scaled_suction_side_displacement_thickness*-0.419465) );
	double perceptron_layer_1_output_3 = tanh( -0.192263 + (scaled_frequency*0.689051) + (scaled_angle_of_attack*-0.0744363) + (scaled_chord_lenght*0.166568) + (scaled_free_res_stream_velocity*-0.0152808) + (scaled_suction_side_displacement_thickness*0.203705) );
	double perceptron_layer_1_output_4 = tanh( 0.0181145 + (scaled_frequency*-0.0188858) + (scaled_angle_of_attack*-0.146249) + (scaled_chord_lenght*0.0274199) + (scaled_free_res_stream_velocity*-0.0503161) + (scaled_suction_side_displacement_thickness*-0.124638) );
	double perceptron_layer_1_output_5 = tanh( -0.0189489 + (scaled_frequency*0.988166) + (scaled_angle_of_attack*0.0653548) + (scaled_chord_lenght*-0.235379) + (scaled_free_res_stream_velocity*-0.0921775) + (scaled_suction_side_displacement_thickness*-0.142748) );
	double perceptron_layer_1_output_6 = tanh( 0.215003 + (scaled_frequency*-0.321256) + (scaled_angle_of_attack*0.164217) + (scaled_chord_lenght*0.125233) + (scaled_free_res_stream_velocity*-0.0448092) + (scaled_suction_side_displacement_thickness*-0.0287405) );
	double perceptron_layer_1_output_7 = tanh( 0.189334 + (scaled_frequency*0.0825739) + (scaled_angle_of_attack*-0.0950488) + (scaled_chord_lenght*-0.101343) + (scaled_free_res_stream_velocity*0.0890101) + (scaled_suction_side_displacement_thickness*-0.0845698) );
	double perceptron_layer_1_output_8 = tanh( 0.0368382 + (scaled_frequency*-0.117503) + (scaled_angle_of_attack*-0.294535) + (scaled_chord_lenght*0.0979208) + (scaled_free_res_stream_velocity*0.0955674) + (scaled_suction_side_displacement_thickness*-0.0649704) );
	double perceptron_layer_1_output_9 = tanh( 0.51505 + (scaled_frequency*-0.28184) + (scaled_angle_of_attack*0.203644) + (scaled_chord_lenght*0.071601) + (scaled_free_res_stream_velocity*0.146957) + (scaled_suction_side_displacement_thickness*0.233283) );

	double perceptron_layer_2_output_0 = ( -0.302371 + (perceptron_layer_1_output_0*0.104693) + (perceptron_layer_1_output_1*0.567519) + (perceptron_layer_1_output_2*1.1732) + (perceptron_layer_1_output_3*-0.299872) + (perceptron_layer_1_output_4*-0.00973633) + (perceptron_layer_1_output_5*-0.527841) + (perceptron_layer_1_output_6*-0.675085) + (perceptron_layer_1_output_7*-0.146942) + (perceptron_layer_1_output_8*0.0373969) + (perceptron_layer_1_output_9*-0.582093) );

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

