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

	double perceptron_layer_1_output_0 = tanh( 0.601958 + (scaled_frequency*-0.817688) + (scaled_angle_of_attack*-1.10421) + (scaled_chord_lenght*-0.258234) + (scaled_free_res_stream_velocity*0.225282) + (scaled_suction_side_displacement_thickness*-0.322878) );
	double perceptron_layer_1_output_1 = tanh( -0.848538 + (scaled_frequency*0.904396) + (scaled_angle_of_attack*0.212252) + (scaled_chord_lenght*0.765746) + (scaled_free_res_stream_velocity*-0.20075) + (scaled_suction_side_displacement_thickness*0.491352) );
	double perceptron_layer_1_output_2 = tanh( 0.003517 + (scaled_frequency*0.116347) + (scaled_angle_of_attack*-0.162766) + (scaled_chord_lenght*-0.381116) + (scaled_free_res_stream_velocity*-0.00460313) + (scaled_suction_side_displacement_thickness*-0.354185) );
	double perceptron_layer_1_output_3 = tanh( 0.0980684 + (scaled_frequency*0.548135) + (scaled_angle_of_attack*0.257357) + (scaled_chord_lenght*0.510571) + (scaled_free_res_stream_velocity*-0.146107) + (scaled_suction_side_displacement_thickness*0.530126) );
	double perceptron_layer_1_output_4 = tanh( -0.253968 + (scaled_frequency*0.752096) + (scaled_angle_of_attack*0.244459) + (scaled_chord_lenght*-0.0675238) + (scaled_free_res_stream_velocity*-0.171921) + (scaled_suction_side_displacement_thickness*0.773128) );
	double perceptron_layer_1_output_5 = tanh( -0.261273 + (scaled_frequency*1.11133) + (scaled_angle_of_attack*0.0969361) + (scaled_chord_lenght*0.236828) + (scaled_free_res_stream_velocity*-0.16206) + (scaled_suction_side_displacement_thickness*0.298382) );
	double perceptron_layer_1_output_6 = tanh( 0.0413499 + (scaled_frequency*0.224242) + (scaled_angle_of_attack*-0.47201) + (scaled_chord_lenght*-0.217019) + (scaled_free_res_stream_velocity*0.212974) + (scaled_suction_side_displacement_thickness*-0.289527) );
	double perceptron_layer_1_output_7 = tanh( -0.0357212 + (scaled_frequency*0.304276) + (scaled_angle_of_attack*0.568027) + (scaled_chord_lenght*0.119922) + (scaled_free_res_stream_velocity*-0.21729) + (scaled_suction_side_displacement_thickness*0.326297) );
	double perceptron_layer_1_output_8 = tanh( 0.875971 + (scaled_frequency*1.04171) + (scaled_angle_of_attack*-0.404864) + (scaled_chord_lenght*-0.8364) + (scaled_free_res_stream_velocity*-0.129116) + (scaled_suction_side_displacement_thickness*-0.0941613) );
	double perceptron_layer_1_output_9 = tanh( 0.684929 + (scaled_frequency*-0.696939) + (scaled_angle_of_attack*-0.0687035) + (scaled_chord_lenght*-1.26846) + (scaled_free_res_stream_velocity*0.25051) + (scaled_suction_side_displacement_thickness*-0.53731) );

	double perceptron_layer_2_output_0 = ( -0.39744 + (perceptron_layer_1_output_0*0.611077) + (perceptron_layer_1_output_1*-0.38078) + (perceptron_layer_1_output_2*0.301687) + (perceptron_layer_1_output_3*0.43876) + (perceptron_layer_1_output_4*0.330836) + (perceptron_layer_1_output_5*-0.232196) + (perceptron_layer_1_output_6*0.417043) + (perceptron_layer_1_output_7*-0.029908) + (perceptron_layer_1_output_8*-0.68835) + (perceptron_layer_1_output_9*0.854178) );

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

