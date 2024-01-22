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

	double perceptron_layer_1_output_0 = tanh( -0.133821 + (scaled_frequency*-0.237494) + (scaled_angle_of_attack*0.475424) + (scaled_cho_rd_lenght*0.273246) + (scaled_free_res_stream_velocity*0.331822) + (scaled_suction_side_di_splacement_thickness*0.207652) );
	double perceptron_layer_1_output_1 = tanh( 0.765565 + (scaled_frequency*0.28683) + (scaled_angle_of_attack*0.81502) + (scaled_cho_rd_lenght*0.662762) + (scaled_free_res_stream_velocity*0.103104) + (scaled_suction_side_di_splacement_thickness*0.551086) );
	double perceptron_layer_1_output_2 = tanh( 0.230807 + (scaled_frequency*0.402935) + (scaled_angle_of_attack*0.701968) + (scaled_cho_rd_lenght*0.183547) + (scaled_free_res_stream_velocity*0.161019) + (scaled_suction_side_di_splacement_thickness*0.38084) );
	double perceptron_layer_1_output_3 = tanh( 0.180955 + (scaled_frequency*0.256053) + (scaled_angle_of_attack*-0.134764) + (scaled_cho_rd_lenght*0.0431097) + (scaled_free_res_stream_velocity*0.465406) + (scaled_suction_side_di_splacement_thickness*-0.310993) );
	double perceptron_layer_1_output_4 = tanh( 0.850346 + (scaled_frequency*1.03022) + (scaled_angle_of_attack*0.299739) + (scaled_cho_rd_lenght*0.460703) + (scaled_free_res_stream_velocity*0.217246) + (scaled_suction_side_di_splacement_thickness*0.961099) );
	double perceptron_layer_1_output_5 = tanh( 0.455221 + (scaled_frequency*-0.210067) + (scaled_angle_of_attack*0.889043) + (scaled_cho_rd_lenght*0.641203) + (scaled_free_res_stream_velocity*-0.0718783) + (scaled_suction_side_di_splacement_thickness*1.00719) );
	double perceptron_layer_1_output_6 = tanh( 0.99593 + (scaled_frequency*0.190253) + (scaled_angle_of_attack*0.0497835) + (scaled_cho_rd_lenght*0.688123) + (scaled_free_res_stream_velocity*0.772832) + (scaled_suction_side_di_splacement_thickness*0.290587) );
	double perceptron_layer_1_output_7 = tanh( 0.69638 + (scaled_frequency*-0.0493369) + (scaled_angle_of_attack*0.180389) + (scaled_cho_rd_lenght*0.27387) + (scaled_free_res_stream_velocity*0.973379) + (scaled_suction_side_di_splacement_thickness*0.937931) );
	double perceptron_layer_1_output_8 = tanh( 0.377391 + (scaled_frequency*0.861515) + (scaled_angle_of_attack*0.901562) + (scaled_cho_rd_lenght*0.682572) + (scaled_free_res_stream_velocity*0.332775) + (scaled_suction_side_di_splacement_thickness*0.0598009) );
	double perceptron_layer_1_output_9 = tanh( 0.0327805 + (scaled_frequency*0.609163) + (scaled_angle_of_attack*-0.0172566) + (scaled_cho_rd_lenght*0.868256) + (scaled_free_res_stream_velocity*0.921814) + (scaled_suction_side_di_splacement_thickness*0.0689697) );

	double perceptron_layer_2_output_0 = ( 0.303874 + (perceptron_layer_1_output_0*0.346765) + (perceptron_layer_1_output_1*-0.574396) + (perceptron_layer_1_output_2*-0.105433) + (perceptron_layer_1_output_3*0.513394) + (perceptron_layer_1_output_4*-0.514077) + (perceptron_layer_1_output_5*0.308418) + (perceptron_layer_1_output_6*0.00786745) + (perceptron_layer_1_output_7*0.110302) + (perceptron_layer_1_output_8*-0.121047) + (perceptron_layer_1_output_9*-0.124655) );

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

