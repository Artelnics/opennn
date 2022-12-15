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

	double perceptron_layer_1_output_0 = tanh( 0.0581426 + (scaled_frequency*0.0533972) + (scaled_angle_of_attack*0.0530112) + (scaled_chord_lenght*0.132047) + (scaled_free_res__stream_velocity*-0.10355) + (scaled_suction_side_displacement_thickness*0.118668) );
	double perceptron_layer_1_output_1 = tanh( 0.0551679 + (scaled_frequency*0.0725646) + (scaled_angle_of_attack*-0.026407) + (scaled_chord_lenght*0.0792294) + (scaled_free_res__stream_velocity*-0.0893853) + (scaled_suction_side_displacement_thickness*-0.0410026) );
	double perceptron_layer_1_output_2 = tanh( -0.128566 + (scaled_frequency*0.111093) + (scaled_angle_of_attack*-0.117648) + (scaled_chord_lenght*0.0224293) + (scaled_free_res__stream_velocity*0.0744251) + (scaled_suction_side_displacement_thickness*0.0288616) );
	double perceptron_layer_1_output_3 = tanh( -0.110146 + (scaled_frequency*0.0824609) + (scaled_angle_of_attack*0.0344866) + (scaled_chord_lenght*-0.0498873) + (scaled_free_res__stream_velocity*-0.00204463) + (scaled_suction_side_displacement_thickness*0.0579702) );
	double perceptron_layer_1_output_4 = tanh( 0.0719086 + (scaled_frequency*0.0619238) + (scaled_angle_of_attack*-0.104123) + (scaled_chord_lenght*-0.109598) + (scaled_free_res__stream_velocity*0.0887826) + (scaled_suction_side_displacement_thickness*-0.0759988) );
	double perceptron_layer_1_output_5 = tanh( 0.0269654 + (scaled_frequency*0.0967146) + (scaled_angle_of_attack*-0.144197) + (scaled_chord_lenght*0.0309866) + (scaled_free_res__stream_velocity*-0.0330541) + (scaled_suction_side_displacement_thickness*-0.0935791) );
	double perceptron_layer_1_output_6 = tanh( -0.0247906 + (scaled_frequency*0.0534032) + (scaled_angle_of_attack*-0.0849166) + (scaled_chord_lenght*0.0845842) + (scaled_free_res__stream_velocity*0.0202357) + (scaled_suction_side_displacement_thickness*-0.0764997) );
	double perceptron_layer_1_output_7 = tanh( 0.0872863 + (scaled_frequency*-0.0295515) + (scaled_angle_of_attack*-0.101032) + (scaled_chord_lenght*0.088661) + (scaled_free_res__stream_velocity*-0.0696067) + (scaled_suction_side_displacement_thickness*0.0108188) );
	double perceptron_layer_1_output_8 = tanh( -0.0751995 + (scaled_frequency*-0.0462659) + (scaled_angle_of_attack*-0.00858151) + (scaled_chord_lenght*0.124136) + (scaled_free_res__stream_velocity*-0.0486291) + (scaled_suction_side_displacement_thickness*-0.127502) );
	double perceptron_layer_1_output_9 = tanh( -0.118257 + (scaled_frequency*0.0660875) + (scaled_angle_of_attack*0.0390527) + (scaled_chord_lenght*-0.0613898) + (scaled_free_res__stream_velocity*-0.0921771) + (scaled_suction_side_displacement_thickness*0.0780167) );
	double perceptron_layer_1_output_10 = tanh( -0.0609405 + (scaled_frequency*-0.00359068) + (scaled_angle_of_attack*0.105046) + (scaled_chord_lenght*-0.168663) + (scaled_free_res__stream_velocity*0.104011) + (scaled_suction_side_displacement_thickness*0.009473) );
	double perceptron_layer_1_output_11 = tanh( -0.106427 + (scaled_frequency*0.0474479) + (scaled_angle_of_attack*0.106427) + (scaled_chord_lenght*0.125783) + (scaled_free_res__stream_velocity*-0.030318) + (scaled_suction_side_displacement_thickness*0.0750092) );
	double perceptron_layer_1_output_12 = tanh( 0.0958727 + (scaled_frequency*-0.0652598) + (scaled_angle_of_attack*0.0149096) + (scaled_chord_lenght*-0.0834685) + (scaled_free_res__stream_velocity*0.0812834) + (scaled_suction_side_displacement_thickness*0.101319) );
	double perceptron_layer_1_output_13 = tanh( 0.121893 + (scaled_frequency*-0.0931519) + (scaled_angle_of_attack*0.04527) + (scaled_chord_lenght*-0.0695546) + (scaled_free_res__stream_velocity*0.0410322) + (scaled_suction_side_displacement_thickness*-0.0639872) );
	double perceptron_layer_1_output_14 = tanh( 0.0589603 + (scaled_frequency*0.123343) + (scaled_angle_of_attack*0.0246062) + (scaled_chord_lenght*0.0417927) + (scaled_free_res__stream_velocity*-0.12795) + (scaled_suction_side_displacement_thickness*-0.00853406) );
	double perceptron_layer_1_output_15 = tanh( 0.0432151 + (scaled_frequency*-0.084648) + (scaled_angle_of_attack*0.121707) + (scaled_chord_lenght*-0.0988161) + (scaled_free_res__stream_velocity*0.0787827) + (scaled_suction_side_displacement_thickness*0.00126925) );
	double perceptron_layer_1_output_16 = tanh( 0.0184079 + (scaled_frequency*0.122238) + (scaled_angle_of_attack*0.0555079) + (scaled_chord_lenght*-0.025473) + (scaled_free_res__stream_velocity*0.0768394) + (scaled_suction_side_displacement_thickness*-0.0847967) );
	double perceptron_layer_1_output_17 = tanh( -0.110335 + (scaled_frequency*0.066933) + (scaled_angle_of_attack*0.0185492) + (scaled_chord_lenght*0.154287) + (scaled_free_res__stream_velocity*-0.0174952) + (scaled_suction_side_displacement_thickness*0.0901958) );
	double perceptron_layer_1_output_18 = tanh( -0.0504929 + (scaled_frequency*-0.0634997) + (scaled_angle_of_attack*0.0330547) + (scaled_chord_lenght*-0.128682) + (scaled_free_res__stream_velocity*0.0291374) + (scaled_suction_side_displacement_thickness*0.0988648) );
	double perceptron_layer_1_output_19 = tanh( 0.0318443 + (scaled_frequency*-0.00894894) + (scaled_angle_of_attack*0.0657802) + (scaled_chord_lenght*0.0900946) + (scaled_free_res__stream_velocity*-0.049366) + (scaled_suction_side_displacement_thickness*0.0129364) );
	double perceptron_layer_1_output_20 = tanh( -0.0705791 + (scaled_frequency*-0.127209) + (scaled_angle_of_attack*-0.0491753) + (scaled_chord_lenght*0.0270683) + (scaled_free_res__stream_velocity*0.0386473) + (scaled_suction_side_displacement_thickness*0.0919311) );
	double perceptron_layer_1_output_21 = tanh( 0.11711 + (scaled_frequency*-0.088647) + (scaled_angle_of_attack*-0.0808502) + (scaled_chord_lenght*0.0029541) + (scaled_free_res__stream_velocity*-0.110338) + (scaled_suction_side_displacement_thickness*0.0334109) );
	double perceptron_layer_1_output_22 = tanh( 0.00219364 + (scaled_frequency*-0.0138536) + (scaled_angle_of_attack*0.0847813) + (scaled_chord_lenght*-0.0971464) + (scaled_free_res__stream_velocity*-0.128445) + (scaled_suction_side_displacement_thickness*0.0397957) );
	double perceptron_layer_1_output_23 = tanh( -0.117416 + (scaled_frequency*-0.0969534) + (scaled_angle_of_attack*-0.0800721) + (scaled_chord_lenght*0.101838) + (scaled_free_res__stream_velocity*0.0984409) + (scaled_suction_side_displacement_thickness*-0.0558717) );
	double perceptron_layer_1_output_24 = tanh( -0.0546421 + (scaled_frequency*0.116886) + (scaled_angle_of_attack*0.125312) + (scaled_chord_lenght*0.00265598) + (scaled_free_res__stream_velocity*0.0396312) + (scaled_suction_side_displacement_thickness*0.0413139) );
	double perceptron_layer_1_output_25 = tanh( -0.028635 + (scaled_frequency*0.130535) + (scaled_angle_of_attack*-0.113783) + (scaled_chord_lenght*0.000590454) + (scaled_free_res__stream_velocity*-0.134909) + (scaled_suction_side_displacement_thickness*0.101565) );
	double perceptron_layer_1_output_26 = tanh( 0.123665 + (scaled_frequency*0.0721176) + (scaled_angle_of_attack*0.103075) + (scaled_chord_lenght*-0.0567342) + (scaled_free_res__stream_velocity*-0.127084) + (scaled_suction_side_displacement_thickness*-0.0280223) );
	double perceptron_layer_1_output_27 = tanh( 0.0900386 + (scaled_frequency*-0.116962) + (scaled_angle_of_attack*0.036701) + (scaled_chord_lenght*0.032073) + (scaled_free_res__stream_velocity*0.0377112) + (scaled_suction_side_displacement_thickness*0.0161882) );
	double perceptron_layer_1_output_28 = tanh( -0.0454836 + (scaled_frequency*-0.0990519) + (scaled_angle_of_attack*-0.0341115) + (scaled_chord_lenght*0.0181862) + (scaled_free_res__stream_velocity*-0.00604114) + (scaled_suction_side_displacement_thickness*-0.116547) );
	double perceptron_layer_1_output_29 = tanh( 0.0563301 + (scaled_frequency*-0.0906383) + (scaled_angle_of_attack*-0.0644623) + (scaled_chord_lenght*0.0279593) + (scaled_free_res__stream_velocity*-0.107085) + (scaled_suction_side_displacement_thickness*0.0813224) );
	double perceptron_layer_1_output_30 = tanh( -0.134004 + (scaled_frequency*0.0401372) + (scaled_angle_of_attack*-0.0463364) + (scaled_chord_lenght*0.0522827) + (scaled_free_res__stream_velocity*0.0340408) + (scaled_suction_side_displacement_thickness*0.0692768) );
	double perceptron_layer_1_output_31 = tanh( -0.0920937 + (scaled_frequency*0.00310392) + (scaled_angle_of_attack*-0.0931082) + (scaled_chord_lenght*-0.061183) + (scaled_free_res__stream_velocity*0.0320807) + (scaled_suction_side_displacement_thickness*0.0588708) );
	double perceptron_layer_1_output_32 = tanh( -0.0654029 + (scaled_frequency*-0.0614295) + (scaled_angle_of_attack*-0.0578733) + (scaled_chord_lenght*0.0381489) + (scaled_free_res__stream_velocity*0.0563949) + (scaled_suction_side_displacement_thickness*0.125701) );
	double perceptron_layer_1_output_33 = tanh( -0.0266646 + (scaled_frequency*0.14118) + (scaled_angle_of_attack*-0.0406761) + (scaled_chord_lenght*-0.0635983) + (scaled_free_res__stream_velocity*-0.00881787) + (scaled_suction_side_displacement_thickness*0.0766405) );
	double perceptron_layer_1_output_34 = tanh( 0.0706566 + (scaled_frequency*0.0312746) + (scaled_angle_of_attack*-0.0536101) + (scaled_chord_lenght*0.105998) + (scaled_free_res__stream_velocity*0.139574) + (scaled_suction_side_displacement_thickness*-0.06376) );
	double perceptron_layer_1_output_35 = tanh( -0.0274471 + (scaled_frequency*-0.0293565) + (scaled_angle_of_attack*-0.138586) + (scaled_chord_lenght*-0.0610592) + (scaled_free_res__stream_velocity*0.0347274) + (scaled_suction_side_displacement_thickness*-0.080743) );
	double perceptron_layer_1_output_36 = tanh( -0.00461088 + (scaled_frequency*-0.0534124) + (scaled_angle_of_attack*-0.0837289) + (scaled_chord_lenght*-0.0813274) + (scaled_free_res__stream_velocity*0.00982704) + (scaled_suction_side_displacement_thickness*0.0689482) );
	double perceptron_layer_1_output_37 = tanh( -0.0714704 + (scaled_frequency*-0.0223219) + (scaled_angle_of_attack*0.0126421) + (scaled_chord_lenght*0.166151) + (scaled_free_res__stream_velocity*0.100512) + (scaled_suction_side_displacement_thickness*0.0204431) );
	double perceptron_layer_1_output_38 = tanh( -0.112885 + (scaled_frequency*-0.120866) + (scaled_angle_of_attack*0.107371) + (scaled_chord_lenght*-0.0186577) + (scaled_free_res__stream_velocity*0.105048) + (scaled_suction_side_displacement_thickness*-0.00303376) );
	double perceptron_layer_1_output_39 = tanh( 0.0977828 + (scaled_frequency*0.113436) + (scaled_angle_of_attack*-0.0813458) + (scaled_chord_lenght*0.0959442) + (scaled_free_res__stream_velocity*-0.0957348) + (scaled_suction_side_displacement_thickness*0.0037763) );
	double perceptron_layer_1_output_40 = tanh( 0.0372981 + (scaled_frequency*-0.0845791) + (scaled_angle_of_attack*-0.0707715) + (scaled_chord_lenght*-0.140975) + (scaled_free_res__stream_velocity*-0.0616966) + (scaled_suction_side_displacement_thickness*-0.130979) );
	double perceptron_layer_1_output_41 = tanh( 0.0985063 + (scaled_frequency*-0.0669421) + (scaled_angle_of_attack*-0.0911845) + (scaled_chord_lenght*-0.109068) + (scaled_free_res__stream_velocity*0.047551) + (scaled_suction_side_displacement_thickness*-0.135931) );
	double perceptron_layer_1_output_42 = tanh( 0.0720539 + (scaled_frequency*-0.0476747) + (scaled_angle_of_attack*0.078578) + (scaled_chord_lenght*0.0531397) + (scaled_free_res__stream_velocity*-0.0969795) + (scaled_suction_side_displacement_thickness*0.0289647) );
	double perceptron_layer_1_output_43 = tanh( 0.00345982 + (scaled_frequency*0.0204922) + (scaled_angle_of_attack*-0.0179387) + (scaled_chord_lenght*0.0925215) + (scaled_free_res__stream_velocity*-0.0119202) + (scaled_suction_side_displacement_thickness*0.0770268) );
	double perceptron_layer_1_output_44 = tanh( -0.0751068 + (scaled_frequency*0.0695404) + (scaled_angle_of_attack*0.098235) + (scaled_chord_lenght*-0.0808393) + (scaled_free_res__stream_velocity*0.000300232) + (scaled_suction_side_displacement_thickness*-0.025195) );
	double perceptron_layer_1_output_45 = tanh( -0.0661021 + (scaled_frequency*0.151831) + (scaled_angle_of_attack*0.00992463) + (scaled_chord_lenght*0.0619535) + (scaled_free_res__stream_velocity*-0.123545) + (scaled_suction_side_displacement_thickness*-0.066803) );
	double perceptron_layer_1_output_46 = tanh( 0.112811 + (scaled_frequency*0.0395383) + (scaled_angle_of_attack*-0.111304) + (scaled_chord_lenght*-0.0653706) + (scaled_free_res__stream_velocity*-0.0487734) + (scaled_suction_side_displacement_thickness*0.0695179) );
	double perceptron_layer_1_output_47 = tanh( 0.0587568 + (scaled_frequency*-0.0562562) + (scaled_angle_of_attack*-0.113846) + (scaled_chord_lenght*-0.0786099) + (scaled_free_res__stream_velocity*-0.040669) + (scaled_suction_side_displacement_thickness*-0.104245) );
	double perceptron_layer_1_output_48 = tanh( -0.107655 + (scaled_frequency*0.0523518) + (scaled_angle_of_attack*0.107201) + (scaled_chord_lenght*0.0560455) + (scaled_free_res__stream_velocity*0.0348776) + (scaled_suction_side_displacement_thickness*0.122398) );
	double perceptron_layer_1_output_49 = tanh( -0.0380147 + (scaled_frequency*-0.129403) + (scaled_angle_of_attack*0.0546375) + (scaled_chord_lenght*0.088408) + (scaled_free_res__stream_velocity*-0.0921991) + (scaled_suction_side_displacement_thickness*0.00465523) );
	double perceptron_layer_1_output_50 = tanh( -0.112884 + (scaled_frequency*0.121363) + (scaled_angle_of_attack*-0.0206632) + (scaled_chord_lenght*0.146893) + (scaled_free_res__stream_velocity*0.00594613) + (scaled_suction_side_displacement_thickness*0.152906) );
	double perceptron_layer_1_output_51 = tanh( -0.123188 + (scaled_frequency*0.0982331) + (scaled_angle_of_attack*0.0179543) + (scaled_chord_lenght*0.0240328) + (scaled_free_res__stream_velocity*-0.0977674) + (scaled_suction_side_displacement_thickness*-0.0535253) );
	double perceptron_layer_1_output_52 = tanh( 0.0445101 + (scaled_frequency*0.123928) + (scaled_angle_of_attack*-0.0935524) + (scaled_chord_lenght*-0.105794) + (scaled_free_res__stream_velocity*0.0684594) + (scaled_suction_side_displacement_thickness*0.0162033) );
	double perceptron_layer_1_output_53 = tanh( 0.000535579 + (scaled_frequency*0.0886133) + (scaled_angle_of_attack*-0.10262) + (scaled_chord_lenght*-0.0283389) + (scaled_free_res__stream_velocity*0.0796175) + (scaled_suction_side_displacement_thickness*-0.129546) );
	double perceptron_layer_1_output_54 = tanh( 0.070074 + (scaled_frequency*0.110097) + (scaled_angle_of_attack*0.0462503) + (scaled_chord_lenght*-0.0329062) + (scaled_free_res__stream_velocity*0.0752957) + (scaled_suction_side_displacement_thickness*0.0334697) );
	double perceptron_layer_1_output_55 = tanh( 0.114383 + (scaled_frequency*0.0199124) + (scaled_angle_of_attack*-0.0534014) + (scaled_chord_lenght*0.0449813) + (scaled_free_res__stream_velocity*0.112086) + (scaled_suction_side_displacement_thickness*0.0923808) );
	double perceptron_layer_1_output_56 = tanh( -0.10862 + (scaled_frequency*-0.0761028) + (scaled_angle_of_attack*0.120015) + (scaled_chord_lenght*0.0107956) + (scaled_free_res__stream_velocity*-0.049721) + (scaled_suction_side_displacement_thickness*-0.0549094) );
	double perceptron_layer_1_output_57 = tanh( -0.0176566 + (scaled_frequency*-0.0877722) + (scaled_angle_of_attack*-0.0764352) + (scaled_chord_lenght*-0.0271802) + (scaled_free_res__stream_velocity*0.0473273) + (scaled_suction_side_displacement_thickness*-0.105174) );
	double perceptron_layer_1_output_58 = tanh( 0.0857674 + (scaled_frequency*0.0674313) + (scaled_angle_of_attack*0.115703) + (scaled_chord_lenght*-0.0247645) + (scaled_free_res__stream_velocity*-0.132002) + (scaled_suction_side_displacement_thickness*0.0542636) );
	double perceptron_layer_1_output_59 = tanh( 0.0971698 + (scaled_frequency*0.122822) + (scaled_angle_of_attack*-0.0492844) + (scaled_chord_lenght*0.0526231) + (scaled_free_res__stream_velocity*-0.0128071) + (scaled_suction_side_displacement_thickness*-0.0914944) );
	double perceptron_layer_1_output_60 = tanh( -0.143479 + (scaled_frequency*-0.0312215) + (scaled_angle_of_attack*-0.0264164) + (scaled_chord_lenght*-0.0540512) + (scaled_free_res__stream_velocity*-0.0963891) + (scaled_suction_side_displacement_thickness*-0.111976) );
	double perceptron_layer_1_output_61 = tanh( -0.072836 + (scaled_frequency*0.0396249) + (scaled_angle_of_attack*0.0065123) + (scaled_chord_lenght*0.0575213) + (scaled_free_res__stream_velocity*-0.0655753) + (scaled_suction_side_displacement_thickness*0.0349177) );
	double perceptron_layer_1_output_62 = tanh( 0.0507907 + (scaled_frequency*-0.0413352) + (scaled_angle_of_attack*0.0356989) + (scaled_chord_lenght*-0.0842469) + (scaled_free_res__stream_velocity*-0.128041) + (scaled_suction_side_displacement_thickness*0.0884699) );
	double perceptron_layer_1_output_63 = tanh( 0.0175737 + (scaled_frequency*-0.154008) + (scaled_angle_of_attack*0.00877618) + (scaled_chord_lenght*-0.00998815) + (scaled_free_res__stream_velocity*-0.0437902) + (scaled_suction_side_displacement_thickness*-0.142687) );
	double perceptron_layer_1_output_64 = tanh( -0.12469 + (scaled_frequency*0.0627774) + (scaled_angle_of_attack*-0.115842) + (scaled_chord_lenght*0.00181675) + (scaled_free_res__stream_velocity*0.0752705) + (scaled_suction_side_displacement_thickness*-0.0978371) );
	double perceptron_layer_1_output_65 = tanh( 0.0898275 + (scaled_frequency*0.0221311) + (scaled_angle_of_attack*0.0617564) + (scaled_chord_lenght*-0.0168021) + (scaled_free_res__stream_velocity*-0.0439183) + (scaled_suction_side_displacement_thickness*-0.126298) );
	double perceptron_layer_1_output_66 = tanh( -0.0231005 + (scaled_frequency*-0.0636396) + (scaled_angle_of_attack*0.111688) + (scaled_chord_lenght*-0.0167821) + (scaled_free_res__stream_velocity*-0.0226996) + (scaled_suction_side_displacement_thickness*-0.103557) );
	double perceptron_layer_1_output_67 = tanh( -0.0309494 + (scaled_frequency*-0.0399814) + (scaled_angle_of_attack*0.0324285) + (scaled_chord_lenght*-0.0772604) + (scaled_free_res__stream_velocity*-0.103566) + (scaled_suction_side_displacement_thickness*-0.0294345) );
	double perceptron_layer_1_output_68 = tanh( -0.0764993 + (scaled_frequency*0.0639572) + (scaled_angle_of_attack*0.0668878) + (scaled_chord_lenght*-0.00868136) + (scaled_free_res__stream_velocity*-0.141531) + (scaled_suction_side_displacement_thickness*0.0456831) );
	double perceptron_layer_1_output_69 = tanh( 0.0846789 + (scaled_frequency*0.100196) + (scaled_angle_of_attack*0.105173) + (scaled_chord_lenght*0.087653) + (scaled_free_res__stream_velocity*0.0910024) + (scaled_suction_side_displacement_thickness*-0.06481) );
	double perceptron_layer_1_output_70 = tanh( -0.0105309 + (scaled_frequency*0.0805287) + (scaled_angle_of_attack*-0.0251379) + (scaled_chord_lenght*-0.0482217) + (scaled_free_res__stream_velocity*-0.0792344) + (scaled_suction_side_displacement_thickness*0.104536) );
	double perceptron_layer_1_output_71 = tanh( 0.0122112 + (scaled_frequency*0.0844785) + (scaled_angle_of_attack*-0.0259709) + (scaled_chord_lenght*0.0100859) + (scaled_free_res__stream_velocity*-0.0464108) + (scaled_suction_side_displacement_thickness*0.0394237) );
	double perceptron_layer_1_output_72 = tanh( -0.0955341 + (scaled_frequency*0.162409) + (scaled_angle_of_attack*0.032781) + (scaled_chord_lenght*-0.0798832) + (scaled_free_res__stream_velocity*0.0371821) + (scaled_suction_side_displacement_thickness*0.111923) );
	double perceptron_layer_1_output_73 = tanh( -0.0303568 + (scaled_frequency*0.146224) + (scaled_angle_of_attack*0.0203964) + (scaled_chord_lenght*-0.0890804) + (scaled_free_res__stream_velocity*0.119634) + (scaled_suction_side_displacement_thickness*0.0325778) );
	double perceptron_layer_1_output_74 = tanh( 0.0315803 + (scaled_frequency*0.044838) + (scaled_angle_of_attack*-0.106154) + (scaled_chord_lenght*-0.0329659) + (scaled_free_res__stream_velocity*-0.0372314) + (scaled_suction_side_displacement_thickness*-0.138468) );
	double perceptron_layer_1_output_75 = tanh( -0.0328803 + (scaled_frequency*-0.0178716) + (scaled_angle_of_attack*-0.0435206) + (scaled_chord_lenght*-0.0595412) + (scaled_free_res__stream_velocity*0.0604221) + (scaled_suction_side_displacement_thickness*-0.110084) );
	double perceptron_layer_1_output_76 = tanh( -0.0365843 + (scaled_frequency*0.116251) + (scaled_angle_of_attack*0.0806087) + (scaled_chord_lenght*0.108806) + (scaled_free_res__stream_velocity*-0.111514) + (scaled_suction_side_displacement_thickness*0.0192122) );
	double perceptron_layer_1_output_77 = tanh( -0.00679309 + (scaled_frequency*0.00172622) + (scaled_angle_of_attack*-0.0190425) + (scaled_chord_lenght*-0.0763444) + (scaled_free_res__stream_velocity*0.100656) + (scaled_suction_side_displacement_thickness*0.101824) );
	double perceptron_layer_1_output_78 = tanh( 0.0830328 + (scaled_frequency*-0.177536) + (scaled_angle_of_attack*-0.100813) + (scaled_chord_lenght*0.0165027) + (scaled_free_res__stream_velocity*0.0039146) + (scaled_suction_side_displacement_thickness*-0.123063) );
	double perceptron_layer_1_output_79 = tanh( -0.0120049 + (scaled_frequency*-0.0235597) + (scaled_angle_of_attack*-0.120036) + (scaled_chord_lenght*-0.103836) + (scaled_free_res__stream_velocity*0.110679) + (scaled_suction_side_displacement_thickness*0.0680932) );
	double perceptron_layer_1_output_80 = tanh( 0.0526194 + (scaled_frequency*-0.0389749) + (scaled_angle_of_attack*0.0182947) + (scaled_chord_lenght*0.0895276) + (scaled_free_res__stream_velocity*-0.136115) + (scaled_suction_side_displacement_thickness*-0.0932547) );
	double perceptron_layer_1_output_81 = tanh( -0.0706219 + (scaled_frequency*-0.0585385) + (scaled_angle_of_attack*0.0946062) + (scaled_chord_lenght*0.0959387) + (scaled_free_res__stream_velocity*0.0356843) + (scaled_suction_side_displacement_thickness*-0.0481104) );
	double perceptron_layer_1_output_82 = tanh( 0.0801556 + (scaled_frequency*0.104721) + (scaled_angle_of_attack*0.00103433) + (scaled_chord_lenght*-0.0732989) + (scaled_free_res__stream_velocity*-0.118948) + (scaled_suction_side_displacement_thickness*0.127439) );
	double perceptron_layer_1_output_83 = tanh( -0.0222097 + (scaled_frequency*0.0148052) + (scaled_angle_of_attack*0.119966) + (scaled_chord_lenght*0.109546) + (scaled_free_res__stream_velocity*-0.0232659) + (scaled_suction_side_displacement_thickness*0.00562245) );
	double perceptron_layer_1_output_84 = tanh( 0.000408199 + (scaled_frequency*0.17767) + (scaled_angle_of_attack*-0.0128953) + (scaled_chord_lenght*0.0936188) + (scaled_free_res__stream_velocity*0.0837722) + (scaled_suction_side_displacement_thickness*0.0479215) );
	double perceptron_layer_1_output_85 = tanh( -0.0180922 + (scaled_frequency*0.0283503) + (scaled_angle_of_attack*0.0978914) + (scaled_chord_lenght*0.0950495) + (scaled_free_res__stream_velocity*0.109424) + (scaled_suction_side_displacement_thickness*0.033983) );
	double perceptron_layer_1_output_86 = tanh( -0.0646387 + (scaled_frequency*0.0350954) + (scaled_angle_of_attack*-0.0210307) + (scaled_chord_lenght*-0.12351) + (scaled_free_res__stream_velocity*0.0999085) + (scaled_suction_side_displacement_thickness*-0.0677003) );
	double perceptron_layer_1_output_87 = tanh( 0.0295489 + (scaled_frequency*0.0808064) + (scaled_angle_of_attack*-0.0758908) + (scaled_chord_lenght*0.0197268) + (scaled_free_res__stream_velocity*-0.102053) + (scaled_suction_side_displacement_thickness*-0.115014) );
	double perceptron_layer_1_output_88 = tanh( -0.015517 + (scaled_frequency*0.150818) + (scaled_angle_of_attack*-0.0518415) + (scaled_chord_lenght*-0.0913201) + (scaled_free_res__stream_velocity*-0.0280027) + (scaled_suction_side_displacement_thickness*0.00412745) );
	double perceptron_layer_1_output_89 = tanh( -0.0471204 + (scaled_frequency*0.00166146) + (scaled_angle_of_attack*0.0847619) + (scaled_chord_lenght*0.154029) + (scaled_free_res__stream_velocity*-0.1199) + (scaled_suction_side_displacement_thickness*-0.0462148) );
	double perceptron_layer_1_output_90 = tanh( -0.0614267 + (scaled_frequency*0.0957468) + (scaled_angle_of_attack*0.0248489) + (scaled_chord_lenght*0.165277) + (scaled_free_res__stream_velocity*-0.00602362) + (scaled_suction_side_displacement_thickness*-0.0535799) );
	double perceptron_layer_1_output_91 = tanh( -0.037158 + (scaled_frequency*0.0135765) + (scaled_angle_of_attack*0.029822) + (scaled_chord_lenght*0.0595196) + (scaled_free_res__stream_velocity*0.119176) + (scaled_suction_side_displacement_thickness*0.074082) );
	double perceptron_layer_1_output_92 = tanh( -0.0266959 + (scaled_frequency*0.0642074) + (scaled_angle_of_attack*0.0191109) + (scaled_chord_lenght*0.0326548) + (scaled_free_res__stream_velocity*-0.138634) + (scaled_suction_side_displacement_thickness*-0.00529874) );
	double perceptron_layer_1_output_93 = tanh( -0.129081 + (scaled_frequency*-0.104011) + (scaled_angle_of_attack*-0.00915393) + (scaled_chord_lenght*0.0390531) + (scaled_free_res__stream_velocity*0.0252345) + (scaled_suction_side_displacement_thickness*-0.0980907) );
	double perceptron_layer_1_output_94 = tanh( -0.124824 + (scaled_frequency*-0.105372) + (scaled_angle_of_attack*0.118716) + (scaled_chord_lenght*-0.084554) + (scaled_free_res__stream_velocity*0.0616376) + (scaled_suction_side_displacement_thickness*-0.106241) );
	double perceptron_layer_1_output_95 = tanh( 0.088007 + (scaled_frequency*0.0551612) + (scaled_angle_of_attack*0.110307) + (scaled_chord_lenght*-0.0779217) + (scaled_free_res__stream_velocity*0.0237148) + (scaled_suction_side_displacement_thickness*0.0300523) );
	double perceptron_layer_1_output_96 = tanh( 0.0310232 + (scaled_frequency*-0.103354) + (scaled_angle_of_attack*-0.0722028) + (scaled_chord_lenght*-0.0170319) + (scaled_free_res__stream_velocity*0.11386) + (scaled_suction_side_displacement_thickness*0.111294) );
	double perceptron_layer_1_output_97 = tanh( -0.111935 + (scaled_frequency*0.0313903) + (scaled_angle_of_attack*0.0047315) + (scaled_chord_lenght*0.0393849) + (scaled_free_res__stream_velocity*-0.0955027) + (scaled_suction_side_displacement_thickness*0.0779279) );
	double perceptron_layer_1_output_98 = tanh( -0.1212 + (scaled_frequency*0.119484) + (scaled_angle_of_attack*0.0398203) + (scaled_chord_lenght*0.0107637) + (scaled_free_res__stream_velocity*0.0858583) + (scaled_suction_side_displacement_thickness*0.110643) );
	double perceptron_layer_1_output_99 = tanh( -0.0330017 + (scaled_frequency*-0.074953) + (scaled_angle_of_attack*-0.0497197) + (scaled_chord_lenght*0.0902219) + (scaled_free_res__stream_velocity*0.10366) + (scaled_suction_side_displacement_thickness*0.0220054) );

	double perceptron_layer_2_output_0 = ( -0.0669833 + (perceptron_layer_1_output_0*-0.177415) + (perceptron_layer_1_output_1*-0.0967244) + (perceptron_layer_1_output_2*-0.00667994) + (perceptron_layer_1_output_3*-0.0587502) + (perceptron_layer_1_output_4*0.123191) + (perceptron_layer_1_output_5*0.0762691) + (perceptron_layer_1_output_6*-0.0983914) + (perceptron_layer_1_output_7*-0.0407303) + (perceptron_layer_1_output_8*0.0214296) + (perceptron_layer_1_output_9*-0.0996766) + (perceptron_layer_1_output_10*0.127096) + (perceptron_layer_1_output_11*-0.129202) + (perceptron_layer_1_output_12*-0.0122903) + (perceptron_layer_1_output_13*0.0619982) + (perceptron_layer_1_output_14*-0.100582) + (perceptron_layer_1_output_15*0.0745469) + (perceptron_layer_1_output_16*-0.103283) + (perceptron_layer_1_output_17*-0.12367) + (perceptron_layer_1_output_18*0.0616818) + (perceptron_layer_1_output_19*-0.119555) + (perceptron_layer_1_output_20*0.111813) + (perceptron_layer_1_output_21*-0.0465798) + (perceptron_layer_1_output_22*-0.0638131) + (perceptron_layer_1_output_23*-0.0371717) + (perceptron_layer_1_output_24*-0.124987) + (perceptron_layer_1_output_25*-0.132737) + (perceptron_layer_1_output_26*-0.0839663) + (perceptron_layer_1_output_27*0.107712) + (perceptron_layer_1_output_28*0.000377047) + (perceptron_layer_1_output_29*0.0236525) + (perceptron_layer_1_output_30*0.0606202) + (perceptron_layer_1_output_31*0.00141359) + (perceptron_layer_1_output_32*-0.0688363) + (perceptron_layer_1_output_33*-0.126595) + (perceptron_layer_1_output_34*0.0668242) + (perceptron_layer_1_output_35*0.116544) + (perceptron_layer_1_output_36*0.11997) + (perceptron_layer_1_output_37*-0.11148) + (perceptron_layer_1_output_38*0.00604416) + (perceptron_layer_1_output_39*-0.0148926) + (perceptron_layer_1_output_40*0.148982) + (perceptron_layer_1_output_41*0.158511) + (perceptron_layer_1_output_42*-0.030402) + (perceptron_layer_1_output_43*-0.142264) + (perceptron_layer_1_output_44*0.0799467) + (perceptron_layer_1_output_45*-0.125832) + (perceptron_layer_1_output_46*-0.0623396) + (perceptron_layer_1_output_47*0.122067) + (perceptron_layer_1_output_48*-0.0477831) + (perceptron_layer_1_output_49*0.051529) + (perceptron_layer_1_output_50*-0.186334) + (perceptron_layer_1_output_51*-0.0679602) + (perceptron_layer_1_output_52*0.0111681) + (perceptron_layer_1_output_53*0.00992664) + (perceptron_layer_1_output_54*-0.00222778) + (perceptron_layer_1_output_55*-0.0630092) + (perceptron_layer_1_output_56*0.0512482) + (perceptron_layer_1_output_57*-0.00995993) + (perceptron_layer_1_output_58*-0.130293) + (perceptron_layer_1_output_59*-0.0658029) + (perceptron_layer_1_output_60*0.140125) + (perceptron_layer_1_output_61*-0.0963461) + (perceptron_layer_1_output_62*0.028583) + (perceptron_layer_1_output_63*0.167178) + (perceptron_layer_1_output_64*-0.011842) + (perceptron_layer_1_output_65*0.0616728) + (perceptron_layer_1_output_66*0.0882013) + (perceptron_layer_1_output_67*0.116477) + (perceptron_layer_1_output_68*-0.111743) + (perceptron_layer_1_output_69*-0.010728) + (perceptron_layer_1_output_70*-0.0999489) + (perceptron_layer_1_output_71*-0.016957) + (perceptron_layer_1_output_72*-0.14135) + (perceptron_layer_1_output_73*-0.0659211) + (perceptron_layer_1_output_74*0.127221) + (perceptron_layer_1_output_75*0.00390438) + (perceptron_layer_1_output_76*-0.182615) + (perceptron_layer_1_output_77*-0.00119895) + (perceptron_layer_1_output_78*0.171147) + (perceptron_layer_1_output_79*0.075672) + (perceptron_layer_1_output_80*-0.0569168) + (perceptron_layer_1_output_81*0.0412569) + (perceptron_layer_1_output_82*-0.0831998) + (perceptron_layer_1_output_83*-0.141378) + (perceptron_layer_1_output_84*-0.176842) + (perceptron_layer_1_output_85*-0.0729887) + (perceptron_layer_1_output_86*0.110265) + (perceptron_layer_1_output_87*0.00893191) + (perceptron_layer_1_output_88*-0.0631658) + (perceptron_layer_1_output_89*-0.137337) + (perceptron_layer_1_output_90*-0.139307) + (perceptron_layer_1_output_91*-0.0705433) + (perceptron_layer_1_output_92*-0.0798256) + (perceptron_layer_1_output_93*0.12111) + (perceptron_layer_1_output_94*0.0506196) + (perceptron_layer_1_output_95*-0.0742325) + (perceptron_layer_1_output_96*0.0555989) + (perceptron_layer_1_output_97*-0.10362) + (perceptron_layer_1_output_98*-0.124158) + (perceptron_layer_1_output_99*-0.0677084) );

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

