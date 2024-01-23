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

	double perceptron_layer_1_output_0 = tanh( 0.705909 + (scaled_sepal_lenght*0.144892) + (scaled_sepal_width*0.289479) + (scaled_petal_lenght*0.45477) + (scaled_petal_width*0.916351) );
	double perceptron_layer_1_output_1 = tanh( 0.478104 + (scaled_sepal_lenght*0.739366) + (scaled_sepal_width*0.0899907) + (scaled_petal_lenght*0.327944) + (scaled_petal_width*0.639982) );
	double perceptron_layer_1_output_2 = tanh( 0.66575 + (scaled_sepal_lenght*0.299121) + (scaled_sepal_width*0.460243) + (scaled_petal_lenght*0.205209) + (scaled_petal_width*0.525338) );

	double probabilistic_layer_combinations_0 = 0.766181 +0.488712*perceptron_layer_1_output_0 +0.869745*perceptron_layer_1_output_1 +0.0212694*perceptron_layer_1_output_2 ;
	double probabilistic_layer_combinations_1 = 0.955649 +0.254349*perceptron_layer_1_output_0 +0.544686*perceptron_layer_1_output_1 +0.785985*perceptron_layer_1_output_2 ;
	double probabilistic_layer_combinations_2 = 0.454777 +0.749801*perceptron_layer_1_output_0 +0.105201*perceptron_layer_1_output_1 +0.960574*perceptron_layer_1_output_2 ;

	double sum = exp(probabilistic_layer_combinations_0) + exp(probabilistic_layer_combinations_1) + exp(probabilistic_layer_combinations_2);

	double iri_s_setosa = exp(probabilistic_layer_combinations_0)/sum;
	double iri_s_versicolo_r = exp(probabilistic_layer_combinations_1)/sum;
	double iri_s_virgin_ica = exp(probabilistic_layer_combinations_2)/sum;

	vector<float> out(3);
	out[0] = iri_s_setosa;
	out[1] = iri_s_versicolo_r;
	out[2] = iri_s_virgin_ica;

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

