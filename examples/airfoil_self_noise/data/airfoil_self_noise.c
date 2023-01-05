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
	3) free_res__stream_velocity
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
	const float free_res__stream_velocity = inputs[3];
	const float suction_side_displacement_thickness = inputs[4];

	double scaled_frequency = (frequency-2886.380615)/3152.573242;
	double scaled_angle_of_attack = (angle_of_attack-6.782301903)/5.918128014;
	double scaled_chord_lenght = (chord_lenght-0.136548236)/0.09354072809;
	double scaled_free_res__stream_velocity = (free_res__stream_velocity-50.86074448)/15.57278538;
	double scaled_suction_side_displacement_thickness = (suction_side_displacement_thickness-0.01113987993)/0.01315023471;

	double perceptron_layer_1_output_0 = tanh( -0.108772 + (scaled_frequency*-0.822778) + (scaled_angle_of_attack*-0.0342082) + (scaled_chord_lenght*-0.536083) + (scaled_free_res__stream_velocity*0.203159) + (scaled_suction_side_displacement_thickness*-0.401746) );
	double perceptron_layer_1_output_1 = tanh( -0.282602 + (scaled_frequency*-0.271566) + (scaled_angle_of_attack*0.184933) + (scaled_chord_lenght*0.264628) + (scaled_free_res__stream_velocity*0.0297797) + (scaled_suction_side_displacement_thickness*-0.0180576) );
	double perceptron_layer_1_output_2 = tanh( 0.117133 + (scaled_frequency*-0.148523) + (scaled_angle_of_attack*0.0545441) + (scaled_chord_lenght*0.227875) + (scaled_free_res__stream_velocity*-0.182429) + (scaled_suction_side_displacement_thickness*0.0363498) );
	double perceptron_layer_1_output_3 = tanh( 0.00879293 + (scaled_frequency*0.111951) + (scaled_angle_of_attack*0.108756) + (scaled_chord_lenght*-0.133781) + (scaled_free_res__stream_velocity*0.0167702) + (scaled_suction_side_displacement_thickness*-0.21269) );
	double perceptron_layer_1_output_4 = tanh( 0.0647118 + (scaled_frequency*0.1176) + (scaled_angle_of_attack*0.138485) + (scaled_chord_lenght*0.00617065) + (scaled_free_res__stream_velocity*-0.144209) + (scaled_suction_side_displacement_thickness*0.0263794) );
	double perceptron_layer_1_output_5 = tanh( 1.34097 + (scaled_frequency*1.20477) + (scaled_angle_of_attack*-0.932095) + (scaled_chord_lenght*-0.0242305) + (scaled_free_res__stream_velocity*-0.0184953) + (scaled_suction_side_displacement_thickness*-0.897085) );
	double perceptron_layer_1_output_6 = tanh( -0.156899 + (scaled_frequency*-0.0328856) + (scaled_angle_of_attack*0.830346) + (scaled_chord_lenght*0.359399) + (scaled_free_res__stream_velocity*-0.483714) + (scaled_suction_side_displacement_thickness*-0.338764) );
	double perceptron_layer_1_output_7 = tanh( -0.223748 + (scaled_frequency*-0.192506) + (scaled_angle_of_attack*0.450773) + (scaled_chord_lenght*0.0386252) + (scaled_free_res__stream_velocity*-0.352974) + (scaled_suction_side_displacement_thickness*-0.247088) );
	double perceptron_layer_1_output_8 = tanh( 0.235322 + (scaled_frequency*0.0472326) + (scaled_angle_of_attack*-0.222592) + (scaled_chord_lenght*0.0270406) + (scaled_free_res__stream_velocity*0.0548989) + (scaled_suction_side_displacement_thickness*-0.0575989) );
	double perceptron_layer_1_output_9 = tanh( -0.436781 + (scaled_frequency*-0.0483961) + (scaled_angle_of_attack*-0.280166) + (scaled_chord_lenght*0.245498) + (scaled_free_res__stream_velocity*-0.172182) + (scaled_suction_side_displacement_thickness*-0.111325) );
	double perceptron_layer_1_output_10 = tanh( 0.316818 + (scaled_frequency*-0.146738) + (scaled_angle_of_attack*0.135967) + (scaled_chord_lenght*0.259964) + (scaled_free_res__stream_velocity*-0.149921) + (scaled_suction_side_displacement_thickness*-0.053858) );
	double perceptron_layer_1_output_11 = tanh( -0.395874 + (scaled_frequency*-0.69323) + (scaled_angle_of_attack*-0.605224) + (scaled_chord_lenght*-1.29743) + (scaled_free_res__stream_velocity*0.609498) + (scaled_suction_side_displacement_thickness*0.0439384) );
	double perceptron_layer_1_output_12 = tanh( -0.591594 + (scaled_frequency*0.268556) + (scaled_angle_of_attack*0.00968784) + (scaled_chord_lenght*0.364712) + (scaled_free_res__stream_velocity*-0.1912) + (scaled_suction_side_displacement_thickness*0.29259) );
	double perceptron_layer_1_output_13 = tanh( 2.00671 + (scaled_frequency*2.20708) + (scaled_angle_of_attack*-0.175433) + (scaled_chord_lenght*-0.131762) + (scaled_free_res__stream_velocity*-0.0105271) + (scaled_suction_side_displacement_thickness*0.226899) );
	double perceptron_layer_1_output_14 = tanh( -1.17796 + (scaled_frequency*-2.63942) + (scaled_angle_of_attack*-1.25855) + (scaled_chord_lenght*-0.953693) + (scaled_free_res__stream_velocity*0.453881) + (scaled_suction_side_displacement_thickness*-1.10844) );
	double perceptron_layer_1_output_15 = tanh( -0.378298 + (scaled_frequency*0.24234) + (scaled_angle_of_attack*0.290745) + (scaled_chord_lenght*-0.108149) + (scaled_free_res__stream_velocity*-0.0321275) + (scaled_suction_side_displacement_thickness*-0.430198) );
	double perceptron_layer_1_output_16 = tanh( 0.0587879 + (scaled_frequency*-0.060184) + (scaled_angle_of_attack*0.0438497) + (scaled_chord_lenght*0.205359) + (scaled_free_res__stream_velocity*-0.0261677) + (scaled_suction_side_displacement_thickness*0.0517619) );
	double perceptron_layer_1_output_17 = tanh( 0.0267782 + (scaled_frequency*0.102294) + (scaled_angle_of_attack*0.0285733) + (scaled_chord_lenght*-0.0448557) + (scaled_free_res__stream_velocity*0.0846667) + (scaled_suction_side_displacement_thickness*-0.0564547) );
	double perceptron_layer_1_output_18 = tanh( -0.0991707 + (scaled_frequency*-0.647122) + (scaled_angle_of_attack*-0.214824) + (scaled_chord_lenght*0.25472) + (scaled_free_res__stream_velocity*0.174161) + (scaled_suction_side_displacement_thickness*-0.0934703) );
	double perceptron_layer_1_output_19 = tanh( 0.297482 + (scaled_frequency*0.30985) + (scaled_angle_of_attack*0.0507664) + (scaled_chord_lenght*-0.76919) + (scaled_free_res__stream_velocity*0.138474) + (scaled_suction_side_displacement_thickness*-0.0164342) );
	double perceptron_layer_1_output_20 = tanh( -0.0597964 + (scaled_frequency*-0.103335) + (scaled_angle_of_attack*0.0200781) + (scaled_chord_lenght*-0.047772) + (scaled_free_res__stream_velocity*0.0357378) + (scaled_suction_side_displacement_thickness*0.0727065) );
	double perceptron_layer_1_output_21 = tanh( -2.02418 + (scaled_frequency*0.0482028) + (scaled_angle_of_attack*-0.537896) + (scaled_chord_lenght*0.218723) + (scaled_free_res__stream_velocity*0.197463) + (scaled_suction_side_displacement_thickness*1.20684) );
	double perceptron_layer_1_output_22 = tanh( -0.0555094 + (scaled_frequency*0.127148) + (scaled_angle_of_attack*0.106781) + (scaled_chord_lenght*-0.137736) + (scaled_free_res__stream_velocity*-0.232485) + (scaled_suction_side_displacement_thickness*0.288765) );
	double perceptron_layer_1_output_23 = tanh( -1.03504 + (scaled_frequency*0.342214) + (scaled_angle_of_attack*1.10877) + (scaled_chord_lenght*0.154571) + (scaled_free_res__stream_velocity*0.563288) + (scaled_suction_side_displacement_thickness*-0.844854) );
	double perceptron_layer_1_output_24 = tanh( 0.360769 + (scaled_frequency*0.132421) + (scaled_angle_of_attack*0.253538) + (scaled_chord_lenght*0.342833) + (scaled_free_res__stream_velocity*0.119897) + (scaled_suction_side_displacement_thickness*-0.116425) );
	double perceptron_layer_1_output_25 = tanh( 0.240677 + (scaled_frequency*0.562757) + (scaled_angle_of_attack*-0.210239) + (scaled_chord_lenght*-0.221423) + (scaled_free_res__stream_velocity*0.0798368) + (scaled_suction_side_displacement_thickness*0.203986) );
	double perceptron_layer_1_output_26 = tanh( 0.387623 + (scaled_frequency*0.964749) + (scaled_angle_of_attack*0.512878) + (scaled_chord_lenght*0.474222) + (scaled_free_res__stream_velocity*-0.0447422) + (scaled_suction_side_displacement_thickness*0.136833) );
	double perceptron_layer_1_output_27 = tanh( -0.0473973 + (scaled_frequency*0.355457) + (scaled_angle_of_attack*0.603166) + (scaled_chord_lenght*0.516417) + (scaled_free_res__stream_velocity*-0.145432) + (scaled_suction_side_displacement_thickness*-0.248246) );
	double perceptron_layer_1_output_28 = tanh( -0.35146 + (scaled_frequency*-0.841176) + (scaled_angle_of_attack*-0.474518) + (scaled_chord_lenght*-0.390698) + (scaled_free_res__stream_velocity*0.0798696) + (scaled_suction_side_displacement_thickness*-0.104136) );
	double perceptron_layer_1_output_29 = tanh( -0.00202072 + (scaled_frequency*-0.184196) + (scaled_angle_of_attack*-0.0997546) + (scaled_chord_lenght*-0.0126696) + (scaled_free_res__stream_velocity*0.141039) + (scaled_suction_side_displacement_thickness*0.0589357) );
	double perceptron_layer_1_output_30 = tanh( -0.311452 + (scaled_frequency*0.237382) + (scaled_angle_of_attack*0.0833069) + (scaled_chord_lenght*0.330918) + (scaled_free_res__stream_velocity*-0.0777295) + (scaled_suction_side_displacement_thickness*0.12972) );
	double perceptron_layer_1_output_31 = tanh( -1.10226 + (scaled_frequency*-0.0662032) + (scaled_angle_of_attack*0.556514) + (scaled_chord_lenght*0.193538) + (scaled_free_res__stream_velocity*-0.078465) + (scaled_suction_side_displacement_thickness*-0.283297) );
	double perceptron_layer_1_output_32 = tanh( 0.402597 + (scaled_frequency*1.6593) + (scaled_angle_of_attack*0.61175) + (scaled_chord_lenght*1.27947) + (scaled_free_res__stream_velocity*0.00426785) + (scaled_suction_side_displacement_thickness*-0.0400469) );
	double perceptron_layer_1_output_33 = tanh( -1.00214 + (scaled_frequency*-0.800772) + (scaled_angle_of_attack*1.26597) + (scaled_chord_lenght*-1.67943) + (scaled_free_res__stream_velocity*0.333359) + (scaled_suction_side_displacement_thickness*0.600901) );
	double perceptron_layer_1_output_34 = tanh( 0.150987 + (scaled_frequency*-0.0148976) + (scaled_angle_of_attack*0.0446946) + (scaled_chord_lenght*-0.0205196) + (scaled_free_res__stream_velocity*0.00856836) + (scaled_suction_side_displacement_thickness*-0.0153098) );
	double perceptron_layer_1_output_35 = tanh( -0.562105 + (scaled_frequency*0.285454) + (scaled_angle_of_attack*0.408653) + (scaled_chord_lenght*-0.0687724) + (scaled_free_res__stream_velocity*-0.110904) + (scaled_suction_side_displacement_thickness*-0.461117) );
	double perceptron_layer_1_output_36 = tanh( -1.10705 + (scaled_frequency*1.052) + (scaled_angle_of_attack*-0.116523) + (scaled_chord_lenght*0.623001) + (scaled_free_res__stream_velocity*0.285593) + (scaled_suction_side_displacement_thickness*0.123149) );
	double perceptron_layer_1_output_37 = tanh( -0.20143 + (scaled_frequency*0.148398) + (scaled_angle_of_attack*0.0564329) + (scaled_chord_lenght*-0.21991) + (scaled_free_res__stream_velocity*0.0643118) + (scaled_suction_side_displacement_thickness*-0.176699) );
	double perceptron_layer_1_output_38 = tanh( -0.0098203 + (scaled_frequency*-0.0677982) + (scaled_angle_of_attack*0.00639374) + (scaled_chord_lenght*0.0910272) + (scaled_free_res__stream_velocity*0.0564417) + (scaled_suction_side_displacement_thickness*0.0950478) );
	double perceptron_layer_1_output_39 = tanh( -2.77956 + (scaled_frequency*-4.55481) + (scaled_angle_of_attack*0.204687) + (scaled_chord_lenght*-0.161399) + (scaled_free_res__stream_velocity*-0.048356) + (scaled_suction_side_displacement_thickness*-0.591488) );
	double perceptron_layer_1_output_40 = tanh( 0.1633 + (scaled_frequency*-0.049521) + (scaled_angle_of_attack*-0.0708652) + (scaled_chord_lenght*0.246918) + (scaled_free_res__stream_velocity*0.112946) + (scaled_suction_side_displacement_thickness*-0.145935) );
	double perceptron_layer_1_output_41 = tanh( -0.609034 + (scaled_frequency*-0.0856183) + (scaled_angle_of_attack*-0.481705) + (scaled_chord_lenght*0.805992) + (scaled_free_res__stream_velocity*-0.105976) + (scaled_suction_side_displacement_thickness*-0.183001) );
	double perceptron_layer_1_output_42 = tanh( 0.168581 + (scaled_frequency*-0.781595) + (scaled_angle_of_attack*-0.212565) + (scaled_chord_lenght*-1.12662) + (scaled_free_res__stream_velocity*-0.479802) + (scaled_suction_side_displacement_thickness*0.23057) );
	double perceptron_layer_1_output_43 = tanh( -0.0289578 + (scaled_frequency*0.123435) + (scaled_angle_of_attack*-0.101915) + (scaled_chord_lenght*-0.212316) + (scaled_free_res__stream_velocity*0.20995) + (scaled_suction_side_displacement_thickness*-0.00355514) );
	double perceptron_layer_1_output_44 = tanh( -0.389387 + (scaled_frequency*-0.0410919) + (scaled_angle_of_attack*-0.00777579) + (scaled_chord_lenght*-0.264857) + (scaled_free_res__stream_velocity*-0.0772722) + (scaled_suction_side_displacement_thickness*-0.103332) );
	double perceptron_layer_1_output_45 = tanh( 0.055035 + (scaled_frequency*0.207257) + (scaled_angle_of_attack*-0.071513) + (scaled_chord_lenght*-0.0392079) + (scaled_free_res__stream_velocity*0.298868) + (scaled_suction_side_displacement_thickness*0.0543697) );
	double perceptron_layer_1_output_46 = tanh( -0.194367 + (scaled_frequency*-0.561548) + (scaled_angle_of_attack*0.0538431) + (scaled_chord_lenght*-0.198053) + (scaled_free_res__stream_velocity*-0.158253) + (scaled_suction_side_displacement_thickness*0.0798506) );
	double perceptron_layer_1_output_47 = tanh( 0.257503 + (scaled_frequency*-0.0708679) + (scaled_angle_of_attack*0.0716484) + (scaled_chord_lenght*0.218253) + (scaled_free_res__stream_velocity*0.00294924) + (scaled_suction_side_displacement_thickness*0.059767) );
	double perceptron_layer_1_output_48 = tanh( -0.218539 + (scaled_frequency*-0.245321) + (scaled_angle_of_attack*0.283993) + (scaled_chord_lenght*-0.133209) + (scaled_free_res__stream_velocity*-0.214869) + (scaled_suction_side_displacement_thickness*-0.11495) );
	double perceptron_layer_1_output_49 = tanh( 0.0823506 + (scaled_frequency*-0.000510856) + (scaled_angle_of_attack*0.0520757) + (scaled_chord_lenght*0.136743) + (scaled_free_res__stream_velocity*-0.100433) + (scaled_suction_side_displacement_thickness*0.0935511) );
	double perceptron_layer_1_output_50 = tanh( -0.0385173 + (scaled_frequency*-0.0608569) + (scaled_angle_of_attack*0.0069527) + (scaled_chord_lenght*0.230969) + (scaled_free_res__stream_velocity*-0.104355) + (scaled_suction_side_displacement_thickness*0.114998) );
	double perceptron_layer_1_output_51 = tanh( 0.0333773 + (scaled_frequency*0.591091) + (scaled_angle_of_attack*-0.0839183) + (scaled_chord_lenght*-0.0953389) + (scaled_free_res__stream_velocity*-0.0594778) + (scaled_suction_side_displacement_thickness*0.210138) );
	double perceptron_layer_1_output_52 = tanh( -0.294497 + (scaled_frequency*-0.296189) + (scaled_angle_of_attack*0.0950457) + (scaled_chord_lenght*0.893583) + (scaled_free_res__stream_velocity*-0.198917) + (scaled_suction_side_displacement_thickness*0.0382229) );
	double perceptron_layer_1_output_53 = tanh( 0.0334455 + (scaled_frequency*-0.0359849) + (scaled_angle_of_attack*0.0169095) + (scaled_chord_lenght*-0.224146) + (scaled_free_res__stream_velocity*-0.0152111) + (scaled_suction_side_displacement_thickness*-0.0166938) );
	double perceptron_layer_1_output_54 = tanh( -0.26875 + (scaled_frequency*0.262928) + (scaled_angle_of_attack*0.0913809) + (scaled_chord_lenght*0.114645) + (scaled_free_res__stream_velocity*-0.0887331) + (scaled_suction_side_displacement_thickness*-0.251381) );
	double perceptron_layer_1_output_55 = tanh( -0.363309 + (scaled_frequency*-0.837275) + (scaled_angle_of_attack*1.68834) + (scaled_chord_lenght*-0.15349) + (scaled_free_res__stream_velocity*0.283479) + (scaled_suction_side_displacement_thickness*0.901851) );
	double perceptron_layer_1_output_56 = tanh( -0.0105731 + (scaled_frequency*0.0406351) + (scaled_angle_of_attack*-0.0661694) + (scaled_chord_lenght*0.060802) + (scaled_free_res__stream_velocity*-0.0551411) + (scaled_suction_side_displacement_thickness*-6.41738e-05) );
	double perceptron_layer_1_output_57 = tanh( 0.141962 + (scaled_frequency*-0.00992439) + (scaled_angle_of_attack*0.149016) + (scaled_chord_lenght*-0.124535) + (scaled_free_res__stream_velocity*0.0329353) + (scaled_suction_side_displacement_thickness*-0.0618866) );
	double perceptron_layer_1_output_58 = tanh( 0.0434411 + (scaled_frequency*-0.0750332) + (scaled_angle_of_attack*-0.0549483) + (scaled_chord_lenght*-0.242587) + (scaled_free_res__stream_velocity*-0.0112152) + (scaled_suction_side_displacement_thickness*-0.0980702) );
	double perceptron_layer_1_output_59 = tanh( 0.152452 + (scaled_frequency*0.470726) + (scaled_angle_of_attack*0.0601312) + (scaled_chord_lenght*0.22655) + (scaled_free_res__stream_velocity*0.0736075) + (scaled_suction_side_displacement_thickness*-0.0294103) );
	double perceptron_layer_1_output_60 = tanh( -0.43653 + (scaled_frequency*1.24497) + (scaled_angle_of_attack*0.24342) + (scaled_chord_lenght*0.662718) + (scaled_free_res__stream_velocity*-0.0448683) + (scaled_suction_side_displacement_thickness*-0.729295) );
	double perceptron_layer_1_output_61 = tanh( -0.347679 + (scaled_frequency*-0.746806) + (scaled_angle_of_attack*0.134313) + (scaled_chord_lenght*-0.143117) + (scaled_free_res__stream_velocity*-0.268461) + (scaled_suction_side_displacement_thickness*-0.399243) );
	double perceptron_layer_1_output_62 = tanh( -0.660335 + (scaled_frequency*0.459287) + (scaled_angle_of_attack*0.216843) + (scaled_chord_lenght*0.052976) + (scaled_free_res__stream_velocity*-0.673182) + (scaled_suction_side_displacement_thickness*-0.663951) );
	double perceptron_layer_1_output_63 = tanh( -0.0590572 + (scaled_frequency*-0.211852) + (scaled_angle_of_attack*0.259615) + (scaled_chord_lenght*0.0448309) + (scaled_free_res__stream_velocity*-0.0846795) + (scaled_suction_side_displacement_thickness*0.263276) );
	double perceptron_layer_1_output_64 = tanh( -0.564339 + (scaled_frequency*-0.946591) + (scaled_angle_of_attack*-0.576808) + (scaled_chord_lenght*-0.0222181) + (scaled_free_res__stream_velocity*-0.636572) + (scaled_suction_side_displacement_thickness*0.452414) );
	double perceptron_layer_1_output_65 = tanh( -0.882963 + (scaled_frequency*-0.0496901) + (scaled_angle_of_attack*0.124155) + (scaled_chord_lenght*0.534224) + (scaled_free_res__stream_velocity*0.63019) + (scaled_suction_side_displacement_thickness*-0.517674) );
	double perceptron_layer_1_output_66 = tanh( 2.08237 + (scaled_frequency*-0.0755422) + (scaled_angle_of_attack*-1.80186) + (scaled_chord_lenght*0.0837535) + (scaled_free_res__stream_velocity*0.239037) + (scaled_suction_side_displacement_thickness*0.376337) );
	double perceptron_layer_1_output_67 = tanh( -0.028556 + (scaled_frequency*0.0545799) + (scaled_angle_of_attack*-0.90532) + (scaled_chord_lenght*0.0165755) + (scaled_free_res__stream_velocity*-0.60151) + (scaled_suction_side_displacement_thickness*0.312992) );
	double perceptron_layer_1_output_68 = tanh( 0.154744 + (scaled_frequency*0.548668) + (scaled_angle_of_attack*0.0578522) + (scaled_chord_lenght*-0.0251007) + (scaled_free_res__stream_velocity*0.00746654) + (scaled_suction_side_displacement_thickness*-0.46534) );
	double perceptron_layer_1_output_69 = tanh( 0.223712 + (scaled_frequency*0.450009) + (scaled_angle_of_attack*-0.263777) + (scaled_chord_lenght*0.132907) + (scaled_free_res__stream_velocity*0.376462) + (scaled_suction_side_displacement_thickness*0.250422) );
	double perceptron_layer_1_output_70 = tanh( -0.016583 + (scaled_frequency*0.11724) + (scaled_angle_of_attack*-0.406262) + (scaled_chord_lenght*-0.195537) + (scaled_free_res__stream_velocity*0.10169) + (scaled_suction_side_displacement_thickness*-0.454062) );
	double perceptron_layer_1_output_71 = tanh( 0.684511 + (scaled_frequency*0.0368781) + (scaled_angle_of_attack*-0.358847) + (scaled_chord_lenght*-0.233262) + (scaled_free_res__stream_velocity*-0.101551) + (scaled_suction_side_displacement_thickness*0.0647162) );
	double perceptron_layer_1_output_72 = tanh( -0.196919 + (scaled_frequency*0.241755) + (scaled_angle_of_attack*0.139208) + (scaled_chord_lenght*0.371913) + (scaled_free_res__stream_velocity*0.0170902) + (scaled_suction_side_displacement_thickness*0.047942) );
	double perceptron_layer_1_output_73 = tanh( -0.177591 + (scaled_frequency*-0.31024) + (scaled_angle_of_attack*0.0246392) + (scaled_chord_lenght*0.184588) + (scaled_free_res__stream_velocity*-0.0197341) + (scaled_suction_side_displacement_thickness*0.0710399) );
	double perceptron_layer_1_output_74 = tanh( 0.0887988 + (scaled_frequency*0.550548) + (scaled_angle_of_attack*-0.0207454) + (scaled_chord_lenght*0.119081) + (scaled_free_res__stream_velocity*0.0885068) + (scaled_suction_side_displacement_thickness*-0.292853) );
	double perceptron_layer_1_output_75 = tanh( 0.749118 + (scaled_frequency*-0.598181) + (scaled_angle_of_attack*-0.695072) + (scaled_chord_lenght*1.89349) + (scaled_free_res__stream_velocity*0.17172) + (scaled_suction_side_displacement_thickness*-0.301191) );
	double perceptron_layer_1_output_76 = tanh( -0.0285802 + (scaled_frequency*6.17835e-05) + (scaled_angle_of_attack*-0.0194886) + (scaled_chord_lenght*-0.204387) + (scaled_free_res__stream_velocity*-0.039104) + (scaled_suction_side_displacement_thickness*-0.0649409) );
	double perceptron_layer_1_output_77 = tanh( -1.42357 + (scaled_frequency*-1.97392) + (scaled_angle_of_attack*-0.37113) + (scaled_chord_lenght*0.206557) + (scaled_free_res__stream_velocity*0.0748161) + (scaled_suction_side_displacement_thickness*-0.802463) );
	double perceptron_layer_1_output_78 = tanh( -2.91023 + (scaled_frequency*-3.60513) + (scaled_angle_of_attack*-0.135537) + (scaled_chord_lenght*-0.535309) + (scaled_free_res__stream_velocity*0.0818079) + (scaled_suction_side_displacement_thickness*-0.560801) );
	double perceptron_layer_1_output_79 = tanh( 0.137745 + (scaled_frequency*0.692542) + (scaled_angle_of_attack*0.400574) + (scaled_chord_lenght*0.261542) + (scaled_free_res__stream_velocity*-0.204707) + (scaled_suction_side_displacement_thickness*-0.52348) );
	double perceptron_layer_1_output_80 = tanh( -0.279128 + (scaled_frequency*-1.18132) + (scaled_angle_of_attack*-1.05839) + (scaled_chord_lenght*-0.381787) + (scaled_free_res__stream_velocity*0.0557709) + (scaled_suction_side_displacement_thickness*0.702687) );
	double perceptron_layer_1_output_81 = tanh( 0.0278725 + (scaled_frequency*0.169133) + (scaled_angle_of_attack*0.138722) + (scaled_chord_lenght*-0.0623697) + (scaled_free_res__stream_velocity*-0.176174) + (scaled_suction_side_displacement_thickness*0.240369) );
	double perceptron_layer_1_output_82 = tanh( 0.339689 + (scaled_frequency*0.645242) + (scaled_angle_of_attack*0.420421) + (scaled_chord_lenght*-0.465699) + (scaled_free_res__stream_velocity*-0.143304) + (scaled_suction_side_displacement_thickness*0.420847) );
	double perceptron_layer_1_output_83 = tanh( -0.219684 + (scaled_frequency*-0.0914689) + (scaled_angle_of_attack*0.497582) + (scaled_chord_lenght*0.178658) + (scaled_free_res__stream_velocity*-0.121184) + (scaled_suction_side_displacement_thickness*0.346788) );
	double perceptron_layer_1_output_84 = tanh( -0.144488 + (scaled_frequency*-0.133668) + (scaled_angle_of_attack*0.324437) + (scaled_chord_lenght*0.0604022) + (scaled_free_res__stream_velocity*-0.274211) + (scaled_suction_side_displacement_thickness*-0.104636) );
	double perceptron_layer_1_output_85 = tanh( -0.0643686 + (scaled_frequency*0.00680344) + (scaled_angle_of_attack*0.105691) + (scaled_chord_lenght*-0.136657) + (scaled_free_res__stream_velocity*-0.0340154) + (scaled_suction_side_displacement_thickness*-0.133823) );
	double perceptron_layer_1_output_86 = tanh( -1.32732 + (scaled_frequency*-1.23191) + (scaled_angle_of_attack*0.110397) + (scaled_chord_lenght*-1.86292) + (scaled_free_res__stream_velocity*0.00360114) + (scaled_suction_side_displacement_thickness*-0.120912) );
	double perceptron_layer_1_output_87 = tanh( 0.868459 + (scaled_frequency*1.27644) + (scaled_angle_of_attack*0.375177) + (scaled_chord_lenght*0.387655) + (scaled_free_res__stream_velocity*-0.347368) + (scaled_suction_side_displacement_thickness*0.129844) );
	double perceptron_layer_1_output_88 = tanh( -0.0347282 + (scaled_frequency*0.141928) + (scaled_angle_of_attack*0.00221115) + (scaled_chord_lenght*-0.123071) + (scaled_free_res__stream_velocity*0.173766) + (scaled_suction_side_displacement_thickness*0.0117281) );
	double perceptron_layer_1_output_89 = tanh( -0.467659 + (scaled_frequency*-0.141174) + (scaled_angle_of_attack*0.204305) + (scaled_chord_lenght*-0.344618) + (scaled_free_res__stream_velocity*-0.68398) + (scaled_suction_side_displacement_thickness*-0.633228) );
	double perceptron_layer_1_output_90 = tanh( 0.0188997 + (scaled_frequency*0.116959) + (scaled_angle_of_attack*-0.288609) + (scaled_chord_lenght*-0.164829) + (scaled_free_res__stream_velocity*0.264228) + (scaled_suction_side_displacement_thickness*0.110442) );
	double perceptron_layer_1_output_91 = tanh( -0.0743774 + (scaled_frequency*-0.592739) + (scaled_angle_of_attack*-0.335009) + (scaled_chord_lenght*0.391341) + (scaled_free_res__stream_velocity*0.0570525) + (scaled_suction_side_displacement_thickness*-0.160958) );
	double perceptron_layer_1_output_92 = tanh( 0.246822 + (scaled_frequency*0.845123) + (scaled_angle_of_attack*0.641985) + (scaled_chord_lenght*0.320619) + (scaled_free_res__stream_velocity*-0.184016) + (scaled_suction_side_displacement_thickness*-0.184469) );
	double perceptron_layer_1_output_93 = tanh( 0.152481 + (scaled_frequency*-0.761339) + (scaled_angle_of_attack*-0.869285) + (scaled_chord_lenght*-0.262123) + (scaled_free_res__stream_velocity*0.101526) + (scaled_suction_side_displacement_thickness*-0.257758) );
	double perceptron_layer_1_output_94 = tanh( -0.239636 + (scaled_frequency*0.0227852) + (scaled_angle_of_attack*-0.0089042) + (scaled_chord_lenght*-0.25876) + (scaled_free_res__stream_velocity*-0.215434) + (scaled_suction_side_displacement_thickness*0.210991) );
	double perceptron_layer_1_output_95 = tanh( -0.000606495 + (scaled_frequency*-0.0663617) + (scaled_angle_of_attack*-0.0496431) + (scaled_chord_lenght*0.147581) + (scaled_free_res__stream_velocity*0.284227) + (scaled_suction_side_displacement_thickness*-0.346141) );
	double perceptron_layer_1_output_96 = tanh( 0.0843298 + (scaled_frequency*-0.166519) + (scaled_angle_of_attack*-0.184119) + (scaled_chord_lenght*-0.285765) + (scaled_free_res__stream_velocity*-0.00788713) + (scaled_suction_side_displacement_thickness*0.00780662) );
	double perceptron_layer_1_output_97 = tanh( 0.22863 + (scaled_frequency*0.0294097) + (scaled_angle_of_attack*-0.0514155) + (scaled_chord_lenght*-0.228788) + (scaled_free_res__stream_velocity*-0.0349159) + (scaled_suction_side_displacement_thickness*-0.0779597) );
	double perceptron_layer_1_output_98 = tanh( -0.0442395 + (scaled_frequency*0.151098) + (scaled_angle_of_attack*-0.0442536) + (scaled_chord_lenght*-0.120739) + (scaled_free_res__stream_velocity*0.19357) + (scaled_suction_side_displacement_thickness*-0.0335633) );
	double perceptron_layer_1_output_99 = tanh( -0.340413 + (scaled_frequency*-0.535146) + (scaled_angle_of_attack*0.11274) + (scaled_chord_lenght*-0.268284) + (scaled_free_res__stream_velocity*-0.387009) + (scaled_suction_side_displacement_thickness*-0.396058) );

	double perceptron_layer_2_output_0 = ( -0.520761 + (perceptron_layer_1_output_0*-0.312049) + (perceptron_layer_1_output_1*0.118254) + (perceptron_layer_1_output_2*-0.395956) + (perceptron_layer_1_output_3*0.270397) + (perceptron_layer_1_output_4*0.0514557) + (perceptron_layer_1_output_5*-1.08646) + (perceptron_layer_1_output_6*-0.382812) + (perceptron_layer_1_output_7*-0.385736) + (perceptron_layer_1_output_8*-0.147531) + (perceptron_layer_1_output_9*-0.31537) + (perceptron_layer_1_output_10*-0.394219) + (perceptron_layer_1_output_11*0.570561) + (perceptron_layer_1_output_12*-0.54234) + (perceptron_layer_1_output_13*1.7886) + (perceptron_layer_1_output_14*0.750085) + (perceptron_layer_1_output_15*0.660219) + (perceptron_layer_1_output_16*-0.238191) + (perceptron_layer_1_output_17*0.0750291) + (perceptron_layer_1_output_18*0.421445) + (perceptron_layer_1_output_19*-0.44169) + (perceptron_layer_1_output_20*-0.0282381) + (perceptron_layer_1_output_21*0.664286) + (perceptron_layer_1_output_22*0.240903) + (perceptron_layer_1_output_23*-1.00215) + (perceptron_layer_1_output_24*-0.530865) + (perceptron_layer_1_output_25*-0.214262) + (perceptron_layer_1_output_26*0.625651) + (perceptron_layer_1_output_27*-0.438638) + (perceptron_layer_1_output_28*-0.430771) + (perceptron_layer_1_output_29*-0.129384) + (perceptron_layer_1_output_30*-0.414847) + (perceptron_layer_1_output_31*0.687854) + (perceptron_layer_1_output_32*-0.834737) + (perceptron_layer_1_output_33*-0.583959) + (perceptron_layer_1_output_34*0.0312181) + (perceptron_layer_1_output_35*0.720205) + (perceptron_layer_1_output_36*-0.811571) + (perceptron_layer_1_output_37*0.410803) + (perceptron_layer_1_output_38*-0.151722) + (perceptron_layer_1_output_39*2.26411) + (perceptron_layer_1_output_40*-0.415528) + (perceptron_layer_1_output_41*0.205661) + (perceptron_layer_1_output_42*0.55349) + (perceptron_layer_1_output_43*0.324089) + (perceptron_layer_1_output_44*0.450903) + (perceptron_layer_1_output_45*0.249691) + (perceptron_layer_1_output_46*-0.0309251) + (perceptron_layer_1_output_47*-0.343517) + (perceptron_layer_1_output_48*-0.253504) + (perceptron_layer_1_output_49*-0.126148) + (perceptron_layer_1_output_50*-0.318923) + (perceptron_layer_1_output_51*-0.381645) + (perceptron_layer_1_output_52*0.640265) + (perceptron_layer_1_output_53*0.353657) + (perceptron_layer_1_output_54*0.164429) + (perceptron_layer_1_output_55*1.10891) + (perceptron_layer_1_output_56*-0.0572899) + (perceptron_layer_1_output_57*0.14957) + (perceptron_layer_1_output_58*0.254036) + (perceptron_layer_1_output_59*0.152482) + (perceptron_layer_1_output_60*1.28192) + (perceptron_layer_1_output_61*-0.00817594) + (perceptron_layer_1_output_62*0.878786) + (perceptron_layer_1_output_63*0.124561) + (perceptron_layer_1_output_64*0.689058) + (perceptron_layer_1_output_65*0.517406) + (perceptron_layer_1_output_66*1.10616) + (perceptron_layer_1_output_67*-0.544478) + (perceptron_layer_1_output_68*0.291574) + (perceptron_layer_1_output_69*0.30661) + (perceptron_layer_1_output_70*-0.237437) + (perceptron_layer_1_output_71*-0.425612) + (perceptron_layer_1_output_72*-0.347682) + (perceptron_layer_1_output_73*0.0270882) + (perceptron_layer_1_output_74*0.145005) + (perceptron_layer_1_output_75*0.695668) + (perceptron_layer_1_output_76*0.256685) + (perceptron_layer_1_output_77*1.36045) + (perceptron_layer_1_output_78*-3.46976) + (perceptron_layer_1_output_79*0.422019) + (perceptron_layer_1_output_80*-0.859801) + (perceptron_layer_1_output_81*0.183009) + (perceptron_layer_1_output_82*-0.641955) + (perceptron_layer_1_output_83*0.24591) + (perceptron_layer_1_output_84*-0.183177) + (perceptron_layer_1_output_85*0.188481) + (perceptron_layer_1_output_86*1.34742) + (perceptron_layer_1_output_87*1.03136) + (perceptron_layer_1_output_88*0.190237) + (perceptron_layer_1_output_89*-0.558566) + (perceptron_layer_1_output_90*0.234342) + (perceptron_layer_1_output_91*0.464698) + (perceptron_layer_1_output_92*0.452613) + (perceptron_layer_1_output_93*0.319881) + (perceptron_layer_1_output_94*0.328435) + (perceptron_layer_1_output_95*-0.291532) + (perceptron_layer_1_output_96*0.30554) + (perceptron_layer_1_output_97*0.161348) + (perceptron_layer_1_output_98*0.255078) + (perceptron_layer_1_output_99*-0.453942) );

	double unscaling_layer_output_0=perceptron_layer_2_output_0*6.898656845+124.8359451;

	double scaled_sound_pressure_level = max(-3.402823466e+38, unscaling_layer_output_0);
	scaled_sound_pressure_level = min(3.402823466e+38, unscaling_layer_output_0);

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
	const float free_res__stream_velocity = /*enter your value here*/; 
	inputs[3] = free_res__stream_velocity;
	const float suction_side_displacement_thickness = /*enter your value here*/; 
	inputs[4] = suction_side_displacement_thickness;

	vector<float> outputs(1);

	outputs = calculate_outputs(inputs);

	printf("These are your outputs:\n");
	printf( "scaled_sound_pressure_level: %f \n", outputs[0]);

	return 0;
} 

