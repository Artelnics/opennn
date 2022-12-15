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
	0) input_0
	1) input_1
	2) input_2
	3) input_3
	4) input_4
	0) ouput_0
*/


#include <iostream>
#include <vector>
#include <math.h>
#include <stdio.h>


using namespace std;


vector<float> calculate_outputs(const vector<float>& inputs)
{
	const float input_0 = inputs[0];
	const float input_1 = inputs[1];
	const float input_2 = inputs[2];
	const float input_3 = inputs[3];
	const float input_4 = inputs[4];

	double scaled_input_0 = (input_0-2886.380615)/3152.573242;
	double scaled_input_1 = (input_1-6.782301903)/5.918128014;
	double scaled_input_2 = (input_2-0.136548236)/0.09354072809;
	double scaled_input_3 = (input_3-50.86074448)/15.57278538;
	double scaled_input_4 = (input_4-0.01113987993)/0.01315023471;

	double perceptron_layer_1_output_0 = tanh( 0.643919 + (scaled_*-0.594028) + (scaled_*-0.102154) + (scaled_*-0.868622) + (scaled_*0.221574) + (scaled_*-0.145109) );
	double perceptron_layer_1_output_1 = tanh( 0.185916 + (scaled_*0.144227) + (scaled_*0.0590693) + (scaled_*-0.0809972) + (scaled_*0.191018) + (scaled_*0.184544) );
	double perceptron_layer_1_output_2 = tanh( 0.108583 + (scaled_*1.21012) + (scaled_*0.975115) + (scaled_*-1.16161) + (scaled_*-0.154521) + (scaled_*-0.464992) );
	double perceptron_layer_1_output_3 = tanh( -0.146318 + (scaled_*-0.0764351) + (scaled_*0.193485) + (scaled_*0.14585) + (scaled_*0.18401) + (scaled_*-0.120219) );
	double perceptron_layer_1_double output_4 = tanh( 0.00816371 + (scaled_*0.0324083) + (scaled_*0.212127) + (scaled_*-0.219034) + (scaled_*-0.114936) + (scaled_*0.133852) );
	double perceptron_layer_1_output_5 = tanh( 1.62018 + (scaled_*1.68499) + (scaled_*-0.300188) + (scaled_*0.455115) + (scaled_*0.46788) + (scaled_*0.380378) );
	double perceptron_layer_1_output_6 = tanh( -0.0485595 + (scaled_*0.0634535) + (scaled_*-0.465133) + (scaled_*0.0542446) + (scaled_*-0.388361) + (scaled_*0.159494) );
	double perceptron_layer_1_output_7 = tanh( 0.0357532 + (scaled_*-0.0729044) + (scaled_*0.10693) + (scaled_*0.159654) + (scaled_*-0.111869) + (scaled_*-0.113801) );
	double perceptron_layer_1_output_8 = tanh( -1.37787 + (scaled_*-1.58569) + (scaled_*1.30629) + (scaled_*0.00396758) + (scaled_*0.0157961) + (scaled_*0.248692) );
	double perceptron_layer_1_output_9 = tanh( 0.412771 + (scaled_*0.216747) + (scaled_*-1.20893) + (scaled_*0.213271) + (scaled_*0.909054) + (scaled_*-0.361389) );
	double perceptron_layer_1_output_10 = tanh( 0.814776 + (scaled_*0.129118) + (scaled_*0.290146) + (scaled_*1.00732) + (scaled_*0.173542) + (scaled_*-0.77338) );
	double perceptron_layer_1_output_11 = tanh( 0.162515 + (scaled_*-0.0278774) + (scaled_*-0.0855513) + (scaled_*0.271165) + (scaled_*-0.0123834) + (scaled_*0.167958) );
	double perceptron_layer_1_output_12 = tanh( -0.128562 + (scaled_*0.10715) + (scaled_*0.0816353) + (scaled_*-0.280996) + (scaled_*0.137199) + (scaled_*0.14978) );
	double perceptron_layer_1_output_13 = tanh( 0.998629 + (scaled_*0.703659) + (scaled_*-0.119222) + (scaled_*0.576745) + (scaled_*-0.346351) + (scaled_*1.21933) );
	double perceptron_layer_1_output_14 = tanh( -0.115955 + (scaled_*0.0976702) + (scaled_*0.138614) + (scaled_*0.187865) + (scaled_*0.178554) + (scaled_*-0.054008) );
	double perceptron_layer_1_output_15 = tanh( -0.742596 + (scaled_*0.529844) + (scaled_*-0.654644) + (scaled_*0.5297) + (scaled_*-0.300071) + (scaled_*0.0998728) );
	double perceptron_layer_1_output_16 = tanh( -0.409654 + (scaled_*-0.0731382) + (scaled_*-0.523467) + (scaled_*0.943288) + (scaled_*-0.525875) + (scaled_*0.36504) );
	double perceptron_layer_1_output_17 = tanh( -0.0767377 + (scaled_*-0.0417581) + (scaled_*0.122802) + (scaled_*0.0813451) + (scaled_*0.144534) + (scaled_*-0.128455) );
	double perceptron_layer_1_output_18 = tanh( 0.153437 + (scaled_*0.193388) + (scaled_*-0.0734411) + (scaled_*-0.175957) + (scaled_*-0.236308) + (scaled_*0.165722) );
	double perceptron_layer_1_output_19 = tanh( -0.775774 + (scaled_*-0.395496) + (scaled_*-0.256842) + (scaled_*-0.283968) + (scaled_*0.326949) + (scaled_*-0.60096) );
	double perceptron_layer_1_output_20 = tanh( 2.08069 + (scaled_*-0.528782) + (scaled_*-1.57966) + (scaled_*-0.341087) + (scaled_*-0.295081) + (scaled_*0.494887) );
	double perceptron_layer_1_output_21 = tanh( 0.938147 + (scaled_*0.713931) + (scaled_*0.529195) + (scaled_*0.927952) + (scaled_*0.10587) + (scaled_*0.127334) );
	double perceptron_layer_1_output_22 = tanh( -0.934895 + (scaled_*1.07448) + (scaled_*0.0466623) + (scaled_*0.325726) + (scaled_*0.0695758) + (scaled_*0.477545) );
	double perceptron_layer_1_output_23 = tanh( 0.0383572 + (scaled_*0.24387) + (scaled_*0.27828) + (scaled_*-0.010099) + (scaled_*0.244825) + (scaled_*0.168929) );
	double perceptron_layer_1_output_24 = tanh( -0.181073 + (scaled_*-0.0427041) + (scaled_*-0.0804859) + (scaled_*-0.226652) + (scaled_*-0.118263) + (scaled_*-0.068492) );
	double perceptron_layer_1_output_25 = tanh( 0.0707144 + (scaled_*-0.140345) + (scaled_*-0.0477795) + (scaled_*-0.123571) + (scaled_*-0.168664) + (scaled_*-0.140804) );
	double perceptron_layer_1_output_26 = tanh( 0.172781 + (scaled_*0.133336) + (scaled_*-0.312112) + (scaled_*0.0187486) + (scaled_*0.0967072) + (scaled_*0.00949857) );
	double perceptron_layer_1_output_27 = tanh( -0.110677 + (scaled_*0.133144) + (scaled_*0.0946114) + (scaled_*0.230999) + (scaled_*0.0643557) + (scaled_*0.0501479) );
	double perceptron_layer_1_output_28 = tanh( -0.72813 + (scaled_*-0.180047) + (scaled_*-0.412537) + (scaled_*-0.661762) + (scaled_*-0.0901612) + (scaled_*0.36367) );
	double perceptron_layer_1_output_29 = tanh( 0.0411201 + (scaled_*0.182678) + (scaled_*-0.154616) + (scaled_*-0.0720856) + (scaled_*0.00526289) + (scaled_*-0.126274) );
	double perceptron_layer_1_output_30 = tanh( -2.28283 + (scaled_*-2.66344) + (scaled_*-0.476387) + (scaled_*0.579319) + (scaled_*0.262415) + (scaled_*1.32031) );
	double perceptron_layer_1_output_31 = tanh( 0.175262 + (scaled_*-0.134958) + (scaled_*0.180923) + (scaled_*0.169164) + (scaled_*-0.103352) + (scaled_*0.0229379) );
	double perceptron_layer_1_output_32 = tanh( 0.669908 + (scaled_*-0.00599568) + (scaled_*-0.0545351) + (scaled_*-0.379955) + (scaled_*-0.286859) + (scaled_*-0.973591) );
	double perceptron_layer_1_output_33 = tanh( -0.334142 + (scaled_*-0.538545) + (scaled_*-0.339999) + (scaled_*0.15772) + (scaled_*0.0242066) + (scaled_*0.118649) );
	double perceptron_layer_1_output_34 = tanh( -0.00326882 + (scaled_*-0.138194) + (scaled_*-0.0645933) + (scaled_*-0.184386) + (scaled_*0.144806) + (scaled_*-0.123196) );
	double perceptron_layer_1_output_35 = tanh( -0.000719587 + (scaled_*0.394824) + (scaled_*-0.120924) + (scaled_*-0.107688) + (scaled_*-0.209714) + (scaled_*-0.231054) );
	double perceptron_layer_1_output_36 = tanh( 0.110647 + (scaled_*-1.3779) + (scaled_*0.827955) + (scaled_*0.599462) + (scaled_*-0.151988) + (scaled_*0.566579) );
	double perceptron_layer_1_output_37 = tanh( 0.165553 + (scaled_*0.0818642) + (scaled_*-0.0113753) + (scaled_*-0.0396362) + (scaled_*0.12037) + (scaled_*-0.112669) );
	double perceptron_layer_1_output_38 = tanh( 0.282099 + (scaled_*0.259311) + (scaled_*0.0776433) + (scaled_*0.14121) + (scaled_*-0.0585107) + (scaled_*0.208758) );
	double perceptron_layer_1_output_39 = tanh( 0.505877 + (scaled_*0.318362) + (scaled_*0.28832) + (scaled_*0.31649) + (scaled_*-0.120393) + (scaled_*0.332025) );
	double perceptron_layer_1_output_40 = tanh( -0.017447 + (scaled_*-0.374449) + (scaled_*-0.133581) + (scaled_*-0.234294) + (scaled_*-0.336565) + (scaled_*-0.445613) );
	double perceptron_layer_1_output_41 = tanh( 0.214997 + (scaled_*0.00292207) + (scaled_*-0.487267) + (scaled_*-0.120279) + (scaled_*0.310835) + (scaled_*-0.0107261) );
	double perceptron_layer_1_output_42 = tanh( 0.205284 + (scaled_*0.141314) + (scaled_*-0.10132) + (scaled_*-0.123961) + (scaled_*-0.0387978) + (scaled_*0.0836628) );
	double perceptron_layer_1_output_43 = tanh( -0.17549 + (scaled_*0.0893281) + (scaled_*-0.00266937) + (scaled_*0.222599) + (scaled_*-0.0301058) + (scaled_*0.133876) );
	double perceptron_layer_1_output_44 = tanh( -0.0973073 + (scaled_*0.0896338) + (scaled_*0.104991) + (scaled_*0.0493422) + (scaled_*0.0105825) + (scaled_*-0.114916) );
	double perceptron_layer_1_output_45 = tanh( -0.220223 + (scaled_*-0.129615) + (scaled_*0.153929) + (scaled_*0.18009) + (scaled_*-0.161148) + (scaled_*-0.0749689) );
	double perceptron_layer_1_output_46 = tanh( -0.00903378 + (scaled_*0.00926416) + (scaled_*-0.0578438) + (scaled_*0.210238) + (scaled_*-0.0439274) + (scaled_*0.161373) );
	double perceptron_layer_1_output_47 = tanh( -0.0905298 + (scaled_*0.323174) + (scaled_*-0.147056) + (scaled_*-0.32768) + (scaled_*0.476011) + (scaled_*-0.764583) );
	double perceptron_layer_1_output_48 = tanh( -0.47133 + (scaled_*0.857388) + (scaled_*1.48482) + (scaled_*0.34879) + (scaled_*-0.125704) + (scaled_*-0.0113532) );
	double perceptron_layer_1_output_49 = tanh( 0.123305 + (scaled_*0.0101806) + (scaled_*0.107922) + (scaled_*0.0832219) + (scaled_*0.106908) + (scaled_*0.0764241) );
	double perceptron_layer_1_output_50 = tanh( -1.81783 + (scaled_*-0.212909) + (scaled_*1.44329) + (scaled_*0.93563) + (scaled_*0.00545602) + (scaled_*-0.666803) );
	double perceptron_layer_1_output_51 = tanh( -0.341217 + (scaled_*0.217233) + (scaled_*-0.842882) + (scaled_*0.111186) + (scaled_*-0.437928) + (scaled_*-0.884543) );
	double perceptron_layer_1_output_52 = tanh( 0.812335 + (scaled_*-0.409608) + (scaled_*-0.0085211) + (scaled_*0.216908) + (scaled_*-0.317421) + (scaled_*-0.298833) );
	double perceptron_layer_1_output_53 = tanh( 1.17152 + (scaled_*1.55818) + (scaled_*0.28487) + (scaled_*1.41118) + (scaled_*-0.116902) + (scaled_*0.628183) );
	double perceptron_layer_1_output_54 = tanh( -0.473027 + (scaled_*0.302644) + (scaled_*0.334822) + (scaled_*0.0198361) + (scaled_*-0.046429) + (scaled_*-0.328254) );
	double perceptron_layer_1_output_55 = tanh( -0.101118 + (scaled_*0.0630049) + (scaled_*-0.0111419) + (scaled_*0.202784) + (scaled_*0.039796) + (scaled_*0.126447) );
	double perceptron_layer_1_output_56 = tanh( 0.635951 + (scaled_*1.40725) + (scaled_*1.2436) + (scaled_*1.08542) + (scaled_*-0.313132) + (scaled_*0.36502) );
	double perceptron_layer_1_output_57 = tanh( -2.4195 + (scaled_*-3.17465) + (scaled_*0.14364) + (scaled_*0.0429908) + (scaled_*0.147083) + (scaled_*-1.53182) );
	double perceptron_layer_1_output_58 = tanh( -3.0937 + (scaled_*-4.80404) + (scaled_*0.375327) + (scaled_*-0.177181) + (scaled_*0.0640797) + (scaled_*-0.662182) );
	double perceptron_layer_1_output_59 = tanh( -0.103612 + (scaled_*0.136356) + (scaled_*-0.376235) + (scaled_*-0.141768) + (scaled_*0.1186) + (scaled_*0.0595966) );
	double perceptron_layer_1_output_60 = tanh( 0.00743489 + (scaled_*0.012985) + (scaled_*0.0820185) + (scaled_*-0.207136) + (scaled_*0.0331804) + (scaled_*-0.18702) );
	double perceptron_layer_1_output_61 = tanh( -0.103655 + (scaled_*-0.202219) + (scaled_*0.00147657) + (scaled_*-0.0217945) + (scaled_*0.119499) + (scaled_*0.213322) );
	double perceptron_layer_1_output_62 = tanh( 0.211613 + (scaled_*-0.00590132) + (scaled_*0.295501) + (scaled_*-0.507404) + (scaled_*0.0051908) + (scaled_*0.366352) );
	double perceptron_layer_1_output_63 = tanh( -1.08545 + (scaled_*0.570067) + (scaled_*0.980077) + (scaled_*-1.35245) + (scaled_*-0.18916) + (scaled_*-0.499815) );
	double perceptron_layer_1_output_64 = tanh( -0.16562 + (scaled_*-0.0345172) + (scaled_*0.00632182) + (scaled_*0.143613) + (scaled_*-0.121836) + (scaled_*0.0928976) );
	double perceptron_layer_1_output_65 = tanh( 0.378089 + (scaled_*-0.0572145) + (scaled_*0.18709) + (scaled_*-0.268652) + (scaled_*0.255188) + (scaled_*-0.356244) );
	double perceptron_layer_1_output_66 = tanh( 0.041463 + (scaled_*-0.124718) + (scaled_*-0.245615) + (scaled_*-0.160542) + (scaled_*-0.0593831) + (scaled_*0.0352114) );
	double perceptron_layer_1_output_67 = tanh( 0.287055 + (scaled_*0.473477) + (scaled_*-0.160279) + (scaled_*0.939358) + (scaled_*-0.816994) + (scaled_*0.445784) );
	double perceptron_layer_1_output_68 = tanh( 0.235187 + (scaled_*0.315787) + (scaled_*0.113288) + (scaled_*-0.270437) + (scaled_*-0.10072) + (scaled_*0.0191489) );
	double perceptron_layer_1_output_69 = tanh( 1.1886 + (scaled_*2.4322) + (scaled_*-1.07758) + (scaled_*-0.19262) + (scaled_*0.078145) + (scaled_*-0.608338) );
	double perceptron_layer_1_output_70 = tanh( 0.0549624 + (scaled_*0.122149) + (scaled_*0.160383) + (scaled_*0.198718) + (scaled_*0.101306) + (scaled_*-0.0139897) );
	double perceptron_layer_1_output_71 = tanh( 0.000419917 + (scaled_*-0.224174) + (scaled_*0.0998027) + (scaled_*0.0351046) + (scaled_*0.0674186) + (scaled_*0.122164) );
	double perceptron_layer_1_output_72 = tanh( 0.421641 + (scaled_*-0.0435334) + (scaled_*-0.769298) + (scaled_*-0.0878589) + (scaled_*0.47388) + (scaled_*0.161513) );
	double perceptron_layer_1_output_73 = tanh( 0.281336 + (scaled_*-0.475803) + (scaled_*0.230367) + (scaled_*0.169396) + (scaled_*1.00138) + (scaled_*-0.353899) );
	double perceptron_layer_1_output_74 = tanh( -0.029858 + (scaled_*0.311453) + (scaled_*1.40914) + (scaled_*-0.164176) + (scaled_*-0.553544) + (scaled_*-0.807027) );
	double perceptron_layer_1_output_75 = tanh( 0.0250845 + (scaled_*-0.118911) + (scaled_*-0.0486356) + (scaled_*-0.275471) + (scaled_*-0.122976) + (scaled_*-0.0421399) );
	double perceptron_layer_1_output_76 = tanh( 0.167285 + (scaled_*0.166836) + (scaled_*0.0927212) + (scaled_*-0.296374) + (scaled_*-0.917746) + (scaled_*0.65998) );
	double perceptron_layer_1_output_77 = tanh( 0.300948 + (scaled_*0.0394368) + (scaled_*0.37562) + (scaled_*-0.341614) + (scaled_*0.0890927) + (scaled_*0.150389) );
	double perceptron_layer_1_output_78 = tanh( -0.36313 + (scaled_*0.000651902) + (scaled_*0.0604584) + (scaled_*-0.326537) + (scaled_*-0.103978) + (scaled_*-0.137392) );
	double perceptron_layer_1_output_79 = tanh( 0.410524 + (scaled_*-0.29622) + (scaled_*-0.186537) + (scaled_*-0.437902) + (scaled_*-0.133775) + (scaled_*-0.0417957) );
	double perceptron_layer_1_output_80 = tanh( -0.389911 + (scaled_*-0.291664) + (scaled_*0.0260256) + (scaled_*0.733337) + (scaled_*-0.147238) + (scaled_*-0.176655) );
	double perceptron_layer_1_output_81 = tanh( -0.158092 + (scaled_*0.204963) + (scaled_*0.0474772) + (scaled_*-0.0551232) + (scaled_*-0.0606602) + (scaled_*-0.201447) );
	double perceptron_layer_1_output_82 = tanh( 0.038766 + (scaled_*0.177881) + (scaled_*-0.205133) + (scaled_*0.0319683) + (scaled_*0.0845458) + (scaled_*0.0306687) );
	double perceptron_layer_1_output_83 = tanh( -0.215886 + (scaled_*0.0933463) + (scaled_*0.476076) + (scaled_*-0.580093) + (scaled_*0.888135) + (scaled_*-0.359949) );
	double perceptron_layer_1_output_84 = tanh( -0.196901 + (scaled_*-0.26014) + (scaled_*0.04582) + (scaled_*-0.0627642) + (scaled_*0.0992004) + (scaled_*0.0664914) );
	double perceptron_layer_1_output_85 = tanh( 0.0245486 + (scaled_*0.00689128) + (scaled_*0.108552) + (scaled_*0.179695) + (scaled_*0.142708) + (scaled_*0.00476263) );
	double perceptron_layer_1_output_86 = tanh( 2.28959 + (scaled_*3.85765) + (scaled_*0.178444) + (scaled_*0.251911) + (scaled_*-0.331968) + (scaled_*0.383546) );
	double perceptron_layer_1_output_87 = tanh( -1.45119 + (scaled_*-0.932511) + (scaled_*-0.0339315) + (scaled_*1.16958) + (scaled_*-0.0288431) + (scaled_*0.692823) );
	double perceptron_layer_1_output_88 = tanh( 0.930987 + (scaled_*-1.24331) + (scaled_*-0.854004) + (scaled_*0.0413623) + (scaled_*0.163686) + (scaled_*-0.388623) );
	double perceptron_layer_1_output_89 = tanh( 0.323338 + (scaled_*-0.0589426) + (scaled_*-0.291958) + (scaled_*-0.144147) + (scaled_*0.243515) + (scaled_*-0.147315) );
	double perceptron_layer_1_output_90 = tanh( -0.624921 + (scaled_*0.184053) + (scaled_*0.0289524) + (scaled_*1.50374) + (scaled_*-0.240252) + (scaled_*0.0179131) );
	double perceptron_layer_1_output_91 = tanh( -1.24767 + (scaled_*1.42497) + (scaled_*-0.436601) + (scaled_*-0.184763) + (scaled_*0.296443) + (scaled_*0.423824) );
	double perceptron_layer_1_output_92 = tanh( -0.0635145 + (scaled_*0.0810235) + (scaled_*0.119007) + (scaled_*-0.0500792) + (scaled_*0.106396) + (scaled_*-0.135983) );
	double perceptron_layer_1_output_93 = tanh( -0.183043 + (scaled_*-0.120801) + (scaled_*-0.390855) + (scaled_*0.515875) + (scaled_*-0.0114771) + (scaled_*-0.451761) );
	double perceptron_layer_1_output_94 = tanh( 0.71188 + (scaled_*-0.412473) + (scaled_*-0.41684) + (scaled_*-0.0959718) + (scaled_*0.34276) + (scaled_*0.904636) );
	double perceptron_layer_1_output_95 = tanh( -0.437739 + (scaled_*-1.1204) + (scaled_*-1.41716) + (scaled_*-0.859526) + (scaled_*0.221352) + (scaled_*0.609454) );
	double perceptron_layer_1_output_96 = tanh( -0.0989964 + (scaled_*0.17219) + (scaled_*-0.0816459) + (scaled_*-0.0387207) + (scaled_*0.120232) + (scaled_*0.0570321) );
	double perceptron_layer_1_output_97 = tanh( -0.153168 + (scaled_*-0.149843) + (scaled_*-0.101888) + (scaled_*-0.386934) + (scaled_*-0.26059) + (scaled_*0.0685152) );
	double perceptron_layer_1_output_98 = tanh( 0.0139667 + (scaled_*-0.0605234) + (scaled_*0.0884043) + (scaled_*0.194978) + (scaled_*-0.120837) + (scaled_*0.0835552) );
	double perceptron_layer_1_output_99 = tanh( -3.19098 + (scaled_*-3.96537) + (scaled_*0.164638) + (scaled_*-0.326796) + (scaled_*0.0932564) + (scaled_*-0.8567) );

	double perceptron_layer_2_output_0 = ( -0.190094 + (perceptron_layer_1_output_0*0.962713) + (perceptron_layer_1_output_1*0.232163) + (perceptron_layer_1_output_2*-1.09266) + (perceptron_layer_1_output_3*0.11656) + (perceptron_layer_1_output_4*0.199491) + (perceptron_layer_1_output_5*1.1126) + (perceptron_layer_1_output_6*-0.371509) + (perceptron_layer_1_output_7*-0.15473) + (perceptron_layer_1_output_8*2.80812) + (perceptron_layer_1_output_9*-0.605129) + (perceptron_layer_1_output_10*-1.03815) + (perceptron_layer_1_output_11*-0.380928) + (perceptron_layer_1_output_12*0.294216) + (perceptron_layer_1_output_13*1.34421) + (perceptron_layer_1_output_14*-0.134167) + (perceptron_layer_1_output_15*1.08158) + (perceptron_layer_1_output_16*-1.01342) + (perceptron_layer_1_output_17*-0.000229261) + (perceptron_layer_1_output_18*-0.274797) + (perceptron_layer_1_output_19*-0.86289) + (perceptron_layer_1_output_20*1.58781) + (perceptron_layer_1_output_21*1.19568) + (perceptron_layer_1_output_22*-2.10067) + (perceptron_layer_1_output_23*0.299368) + (perceptron_layer_1_output_24*0.217899) + (perceptron_layer_1_output_25*0.0270415) + (perceptron_layer_1_output_26*0.306888) + (perceptron_layer_1_output_27*-0.212362) + (perceptron_layer_1_output_28*0.913117) + (perceptron_layer_1_output_29*0.16992) + (perceptron_layer_1_output_30*-1.36209) + (perceptron_layer_1_output_31*-0.458995) + (perceptron_layer_1_output_32*-0.719512) + (perceptron_layer_1_output_33*0.437344) + (perceptron_layer_1_output_34*0.149129) + (perceptron_layer_1_output_35*0.585303) + (perceptron_layer_1_output_36*-1.31392) + (perceptron_layer_1_output_37*0.157007) + (perceptron_layer_1_output_38*0.208938) + (perceptron_layer_1_output_39*0.57222) + (perceptron_layer_1_output_40*-0.607026) + (perceptron_layer_1_output_41*0.540999) + (perceptron_layer_1_output_42*0.00607258) + (perceptron_layer_1_output_43*-0.292672) + (perceptron_layer_1_output_44*0.0275012) + (perceptron_layer_1_output_45*-0.0696947) + (perceptron_layer_1_output_46*-0.297209) + (perceptron_layer_1_output_47*-0.906578) + (perceptron_layer_1_output_48*-1.4827) + (perceptron_layer_1_output_49*-0.0706596) + (perceptron_layer_1_output_50*1.77643) + (perceptron_layer_1_output_51*-0.795111) + (perceptron_layer_1_output_52*-0.728717) + (perceptron_layer_1_output_53*-1.92957) + (perceptron_layer_1_output_54*0.666141) + (perceptron_layer_1_output_55*-0.259004) + (perceptron_layer_1_output_56*-1.73512) + (perceptron_layer_1_output_57*3.12673) + (perceptron_layer_1_output_58*3.22317) + (perceptron_layer_1_output_59*0.387149) + (perceptron_layer_1_output_60*0.327717) + (perceptron_layer_1_output_61*-0.192066) + (perceptron_layer_1_output_62*0.555014) + (perceptron_layer_1_output_63*-1.87264) + (perceptron_layer_1_output_64*-0.235892) + (perceptron_layer_1_output_65*0.641436) + (perceptron_layer_1_output_66*0.163051) + (perceptron_layer_1_output_67*-0.762146) + (perceptron_layer_1_output_68*-0.234961) + (perceptron_layer_1_output_69*1.35547) + (perceptron_layer_1_output_70*-0.128727) + (perceptron_layer_1_output_71*-0.2733) + (perceptron_layer_1_output_72*0.769795) + (perceptron_layer_1_output_73*-0.400411) + (perceptron_layer_1_output_74*1.38017) + (perceptron_layer_1_output_75*0.284505) + (perceptron_layer_1_output_76*-0.857361) + (perceptron_layer_1_output_77*0.385967) + (perceptron_layer_1_output_78*0.513918) + (perceptron_layer_1_output_79*0.465696) + (perceptron_layer_1_output_80*0.885218) + (perceptron_layer_1_output_81*0.266996) + (perceptron_layer_1_output_82*0.28458) + (perceptron_layer_1_output_83*-0.788805) + (perceptron_layer_1_output_84*-0.189311) + (perceptron_layer_1_output_85*-0.116117) + (perceptron_layer_1_output_86*2.02627) + (perceptron_layer_1_output_87*1.50552) + (perceptron_layer_1_output_88*-0.988967) + (perceptron_layer_1_output_89*0.406076) + (perceptron_layer_1_output_90*1.27442) + (perceptron_layer_1_output_91*1.08058) + (perceptron_layer_1_output_92*0.106909) + (perceptron_layer_1_output_93*-0.58715) + (perceptron_layer_1_output_94*-0.871181) + (perceptron_layer_1_output_95*-1.39286) + (perceptron_layer_1_output_96*0.107477) + (perceptron_layer_1_output_97*0.45318) + (perceptron_layer_1_output_98*-0.413362) + (perceptron_layer_1_output_99*-4.69429) );

	double unscaling_layer_output_0=perceptron_layer_2_output_0*6.898656845+124.8359451;

	output_4 = max(-3.402823466e+38, unscaling_layer_output_0);
	output_4 = min(3.402823466e+38, unscaling_layer_output_0);

	vector<float> out(1);
	out[0] = output0;

	return out;
}


int main(){ 

	vector<float> inputs(5); 

	const float input_0 = /*enter your value here*/; 
	inputs[0] = input_0;
	const float input_1 = /*enter your value here*/; 
	inputs[1] = input_1;
	const float input_2 = /*enter your value here*/; 
	inputs[2] = input_2;
	const float input_3 = /*enter your value here*/; 
	inputs[3] = input_3;
	const float input_4 = /*enter your value here*/; 
	inputs[4] = input_4;

	vector<float> outputs(1);

	outputs = calculate_outputs(inputs);

	printf("These are your outputs:\n");
	printf( "output0: %f \n", outputs[0]);

	return 0;
} 

