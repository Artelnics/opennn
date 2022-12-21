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
	0) sepal_lenght
	1) sepal_width
	2) petal_lenght
	3) petal_width
*/


#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>


using namespace std;


vector<float> calculate_outputs(const vector<float>& inputs)
{
	const float sepal_lenght = inputs[0];
	const float sepal_width = inputs[1];
	const float petal_lenght = inputs[2];
	const float petal_width = inputs[3];

	double scaled_sepal_lenght = (sepal_lenght-5.843333244)/0.8280661106;
	double scaled_sepal_width = (sepal_width-3.057333231)/0.4358662963;
	double scaled_petal_lenght = (petal_lenght-3.757999897)/1.765298247;
	double scaled_petal_width = (petal_width-1.19933331)/0.762237668;

<<<<<<< HEAD
	double perceptron_layer_1_output_0 = tanh( -0.494739 + (scaled_sepal_lenght*-0.296986) + (scaled_sepal_width*0.396066) + (scaled_petal_lenght*-0.620677) + (scaled_petal_width*-0.613801) );
	double perceptron_layer_1_output_1 = tanh( 1.98493 + (scaled_sepal_lenght*0.69041) + (scaled_sepal_width*0.221568) + (scaled_petal_lenght*-1.89975) + (scaled_petal_width*-1.7502) );
	double perceptron_layer_1_output_2 = tanh( -0.500013 + (scaled_sepal_lenght*-0.292184) + (scaled_sepal_width*0.393804) + (scaled_petal_lenght*-0.627022) + (scaled_petal_width*-0.618154) );

	double probabilistic_layer_combinations_0 = 0.254467 +1.35186*perceptron_layer_1_output_0 +0.593689*perceptron_layer_1_output_1 +1.35014*perceptron_layer_1_output_2 ;
	double probabilistic_layer_combinations_1 = -0.42925 -0.923573*perceptron_layer_1_output_0 +2.14923*perceptron_layer_1_output_1 -0.923791*perceptron_layer_1_output_2 ;
	double probabilistic_layer_combinations_2 = 0.174789 -0.428276*perceptron_layer_1_output_0 -2.74292*perceptron_layer_1_output_1 -0.426361*perceptron_layer_1_output_2 ;
=======
	double perceptron_layer_1_output_0 = tanh( 8.33051e-05 + (scaled_sepal_lenght*0.00141706) + (scaled_sepal_width*-0.00185295) + (scaled_petal_lenght*0.00262099) + (scaled_petal_width*0.00226507) );
	double perceptron_layer_1_output_1 = tanh( 0.000256285 + (scaled_sepal_lenght*0.000395207) + (scaled_sepal_width*-0.00191348) + (scaled_petal_lenght*0.000522551) + (scaled_petal_width*-1.50365e-05) );
	double perceptron_layer_1_output_2 = tanh( 3.0283e-05 + (scaled_sepal_lenght*0.000540985) + (scaled_sepal_width*-0.00119134) + (scaled_petal_lenght*0.000349308) + (scaled_petal_width*0.000541984) );

	double probabilistic_layer_combinations_0 = 0.000899555 +0.00107503*perceptron_layer_1_output_0 +0.00150672*perceptron_layer_1_output_1 +0.000919525*perceptron_layer_1_output_2 ;
	double probabilistic_layer_combinations_1 = 0.000577413 +0.00106253*perceptron_layer_1_output_0 +0.000619704*perceptron_layer_1_output_1 -0.000209563*perceptron_layer_1_output_2 ;
	double probabilistic_layer_combinations_2 = -0.0007532 -0.00109418*perceptron_layer_1_output_0 -0.00135421*perceptron_layer_1_output_1 -0.00149063*perceptron_layer_1_output_2 ;
>>>>>>> dev

	double sum = exp(probabilistic_layer_combinations_0) + exp(probabilistic_layer_combinations_1) + exp(probabilistic_layer_combinations_2);

	double iris_setosa = exp(probabilistic_layer_combinations_0)/sum;
	double iris_versicolor = exp(probabilistic_layer_combinations_1)/sum;
	double iris_virginica = exp(probabilistic_layer_combinations_2)/sum;

	vector<float> out(3);
	out[0] = iris_setosa;
	out[1] = iris_versicolor;
	out[2] = iris_virginica;

	return out;
}


int main(){ 

	vector<float> inputs(4); 

	const float sepal_lenght = /*enter your value here*/; 
	inputs[0] = sepal_lenght;
	const float sepal_width = /*enter your value here*/; 
	inputs[1] = sepal_width;
	const float petal_lenght = /*enter your value here*/; 
	inputs[2] = petal_lenght;
	const float petal_width = /*enter your value here*/; 
	inputs[3] = petal_width;

	vector<float> outputs(3);

	outputs = calculate_outputs(inputs);

	printf("These are your outputs:\n");
	printf( "iris_setosa: %f \n", outputs[0]);
	printf( "iris_versicolor: %f \n", outputs[1]);
	printf( "iris_virginica: %f \n", outputs[2]);

	return 0;
} 

