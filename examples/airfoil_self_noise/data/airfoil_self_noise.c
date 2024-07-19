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

	double perceptron_layer_1_output_0 = tanh( 0.017487 + (scaled_frequency*0.00104164) + (scaled_angle_of_attack*0.0347911) + (scaled_cho_rd_lenght*0.0414467) + (scaled_free_res_stream_velocity*-0.016968) + (scaled_suction_side_di_splacement_thickness*0.0410217) );
	double perceptron_layer_1_output_1 = tanh( 0.00647834 + (scaled_frequency*0.0288753) + (scaled_angle_of_attack*0.129842) + (scaled_cho_rd_lenght*0.139232) + (scaled_free_res_stream_velocity*-0.0593722) + (scaled_suction_side_di_splacement_thickness*0.101256) );
	double perceptron_layer_1_output_2 = tanh( 0.0140508 + (scaled_frequency*0.0326293) + (scaled_angle_of_attack*0.137952) + (scaled_cho_rd_lenght*0.174569) + (scaled_free_res_stream_velocity*-0.0691102) + (scaled_suction_side_di_splacement_thickness*0.111836) );
	double perceptron_layer_1_output_3 = tanh( 0.190203 + (scaled_frequency*-0.04049) + (scaled_angle_of_attack*0.128901) + (scaled_cho_rd_lenght*0.298001) + (scaled_free_res_stream_velocity*-0.0391023) + (scaled_suction_side_di_splacement_thickness*0.120144) );
	double perceptron_layer_1_output_4 = tanh( -0.00710907 + (scaled_frequency*-0.00206686) + (scaled_angle_of_attack*-0.0640928) + (scaled_cho_rd_lenght*-0.0951827) + (scaled_free_res_stream_velocity*0.0531945) + (scaled_suction_side_di_splacement_thickness*-0.083536) );
	double perceptron_layer_1_output_5 = tanh( -0.000242638 + (scaled_frequency*0.0373775) + (scaled_angle_of_attack*0.138991) + (scaled_cho_rd_lenght*0.158404) + (scaled_free_res_stream_velocity*-0.0662042) + (scaled_suction_side_di_splacement_thickness*0.111262) );
	double perceptron_layer_1_output_6 = tanh( -0.365925 + (scaled_frequency*-0.501179) + (scaled_angle_of_attack*0.0948785) + (scaled_cho_rd_lenght*0.317873) + (scaled_free_res_stream_velocity*0.0573041) + (scaled_suction_side_di_splacement_thickness*0.0547072) );
	double perceptron_layer_1_output_7 = tanh( 0.0278028 + (scaled_frequency*0.00595215) + (scaled_angle_of_attack*0.0950467) + (scaled_cho_rd_lenght*0.129549) + (scaled_free_res_stream_velocity*-0.0410503) + (scaled_suction_side_di_splacement_thickness*0.0963656) );
	double perceptron_layer_1_output_8 = tanh( 0.0219545 + (scaled_frequency*0.013146) + (scaled_angle_of_attack*0.0268585) + (scaled_cho_rd_lenght*0.032059) + (scaled_free_res_stream_velocity*-0.00563042) + (scaled_suction_side_di_splacement_thickness*0.0320868) );
	double perceptron_layer_1_output_9 = tanh( -0.436289 + (scaled_frequency*0.744262) + (scaled_angle_of_attack*0.241364) + (scaled_cho_rd_lenght*0.488199) + (scaled_free_res_stream_velocity*-0.132239) + (scaled_suction_side_di_splacement_thickness*0.274402) );

	double perceptron_layer_2_output_0 = ( -0.0993684 + (perceptron_layer_1_output_0*-0.0716737) + (perceptron_layer_1_output_1*-0.237193) + (perceptron_layer_1_output_2*-0.28242) + (perceptron_layer_1_output_3*-0.403847) + (perceptron_layer_1_output_4*0.157983) + (perceptron_layer_1_output_5*-0.26693) + (perceptron_layer_1_output_6*0.522348) + (perceptron_layer_1_output_7*-0.205437) + (perceptron_layer_1_output_8*-0.0521771) + (perceptron_layer_1_output_9*-0.973342) );

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

