// Artificial Intelligence Techniques SL	
// artelnics@artelnics.com	
// 
// Your model has been exported to this file.
// You can manage it with the 'neural network' method.	
// Example:
// 
// 	vector<float> sample(n);	
// 	sample[0] = 1;	
// 	sample[1] = 2;	
// 	sample[n] = 10;	
// 	vector<float> outputs = neural_network(sample);
// 
// Notice that only one sample is allowed as input. DataSetBatch of inputs are not yet implement,	
// however you can loop through neural network function to get multiple outputs.	

#include <vector>

using namespace std;

vector<float> scaling_layer(const vector<float>& inputs)
{
	vector<float> outputs(5);

	outputs[0] = (inputs[0]-2886.380615)/3152.573242;
	outputs[1] = (inputs[1]-6.782301903)/5.918128014;
	outputs[2] = (inputs[2]-0.136548236)/0.09354072809;
	outputs[3] = (inputs[3]-50.86074448)/15.57278538;
	outputs[4] = (inputs[4]-0.01113987993)/0.01315023471;

	return outputs;
}

vector<float> perceptron_layer_1(const vector<float>& inputs)
{
	vector<float> combinations(100);

	combinations[0] = -0.0698364 +0.00586576*inputs[0] -0.0355319*inputs[1] -0.0943614*inputs[2] +0.0568472*inputs[3] -0.0510303*inputs[4];
	combinations[1] = 0.0699732 -0.00585386*inputs[0] +0.0355901*inputs[1] +0.094504*inputs[2] -0.0569039*inputs[3] +0.0511303*inputs[4];
	combinations[2] = 0.0701439 -0.00583899*inputs[0] +0.0356625*inputs[1] +0.0946814*inputs[2] -0.0569744*inputs[3] +0.0512547*inputs[4];
	combinations[3] = 0.292772 +0.0153026*inputs[0] +0.336029*inputs[1] +0.490543*inputs[2] +0.399071*inputs[3] -0.410977*inputs[4];
	combinations[4] = -0.07001 +0.00585065*inputs[0] -0.0356057*inputs[1] -0.0945423*inputs[2] +0.0569192*inputs[3] -0.0511571*inputs[4];
	combinations[5] = -0.0699933 +0.00585209*inputs[0] -0.0355986*inputs[1] -0.0945249*inputs[2] +0.0569123*inputs[3] -0.0511449*inputs[4];
	combinations[6] = 0.0700872 -0.00584397*inputs[0] +0.0356384*inputs[1] +0.0946224*inputs[2] -0.056951*inputs[3] +0.0512133*inputs[4];
	combinations[7] = 0.0703364 -0.00582227*inputs[0] +0.0357438*inputs[1] +0.0948816*inputs[2] -0.0570534*inputs[3] +0.0513953*inputs[4];
	combinations[8] = -0.300838 -1.03197*inputs[0] -0.694751*inputs[1] -0.402167*inputs[2] +0.170191*inputs[3] +0.508642*inputs[4];
	combinations[9] = 2.40272 +2.58235*inputs[0] +0.183794*inputs[1] +0.628919*inputs[2] -0.0931334*inputs[3] +0.611445*inputs[4];
	combinations[10] = 0.0703256 -0.00582323*inputs[0] +0.0357392*inputs[1] +0.0948704*inputs[2] -0.057049*inputs[3] +0.0513874*inputs[4];
	combinations[11] = -0.0702112 +0.00583317*inputs[0] -0.0356909*inputs[1] -0.0947516*inputs[2] +0.0570021*inputs[3] -0.0513039*inputs[4];
	combinations[12] = 0.0701531 -0.00583821*inputs[0] +0.0356663*inputs[1] +0.0946911*inputs[2] -0.0569782*inputs[3] +0.0512615*inputs[4];
	combinations[13] = 0.534189 -0.147723*inputs[0] +0.734402*inputs[1] -0.434176*inputs[2] +0.0647858*inputs[3] -0.142865*inputs[4];
	combinations[14] = 0.430706 +0.541506*inputs[0] +0.168497*inputs[1] -0.724096*inputs[2] -0.14356*inputs[3] +0.0508948*inputs[4];
	combinations[15] = -0.0700475 +0.00584742*inputs[0] -0.0356216*inputs[1] -0.0945812*inputs[2] +0.0569346*inputs[3] -0.0511843*inputs[4];
	combinations[16] = 0.635895 -0.96083*inputs[0] -0.0320288*inputs[1] -0.571953*inputs[2] +0.107277*inputs[3] +0.972872*inputs[4];
	combinations[17] = -0.0700463 +0.00584751*inputs[0] -0.0356211*inputs[1] -0.09458*inputs[2] +0.0569341*inputs[3] -0.0511835*inputs[4];
	combinations[18] = 0.0700636 -0.00584601*inputs[0] +0.0356284*inputs[1] +0.094598*inputs[2] -0.0569413*inputs[3] +0.0511962*inputs[4];
	combinations[19] = -0.070094 +0.00584337*inputs[0] -0.0356413*inputs[1] -0.0946295*inputs[2] +0.0569538*inputs[3] -0.0512183*inputs[4];
	combinations[20] = -0.0701073 +0.00584219*inputs[0] -0.0356469*inputs[1] -0.0946435*inputs[2] +0.0569593*inputs[3] -0.051228*inputs[4];
	combinations[21] = 0.0701614 -0.00583751*inputs[0] +0.0356699*inputs[1] +0.0946997*inputs[2] -0.0569816*inputs[3] +0.0512676*inputs[4];
	combinations[22] = 0.0699873 -0.00585263*inputs[0] +0.0355961*inputs[1] +0.0945187*inputs[2] -0.0569098*inputs[3] +0.0511405*inputs[4];
	combinations[23] = -0.0713143 +0.00573711*inputs[0] -0.0361537*inputs[1] -0.0958933*inputs[2] +0.0574486*inputs[3] -0.0521084*inputs[4];
	combinations[24] = 0.0702161 -0.00583273*inputs[0] +0.035693*inputs[1] +0.0947566*inputs[2] -0.0570041*inputs[3] +0.0513075*inputs[4];
	combinations[25] = -0.0700707 +0.00584538*inputs[0] -0.0356315*inputs[1] -0.0946055*inputs[2] +0.0569442*inputs[3] -0.0512014*inputs[4];
	combinations[26] = 0.0702049 -0.00583373*inputs[0] +0.0356882*inputs[1] +0.0947449*inputs[2] -0.0569995*inputs[3] +0.0512993*inputs[4];
	combinations[27] = -0.0702945 +0.00582591*inputs[0] -0.0357261*inputs[1] -0.094838*inputs[2] +0.0570363*inputs[3] -0.0513647*inputs[4];
	combinations[28] = 0.0713024 -0.00573824*inputs[0] +0.0361483*inputs[1] +0.0958812*inputs[2] -0.0574436*inputs[3] +0.0521005*inputs[4];
	combinations[29] = -0.0700925 +0.00584348*inputs[0] -0.0356407*inputs[1] -0.0946281*inputs[2] +0.0569532*inputs[3] -0.0512173*inputs[4];
	combinations[30] = -0.0728758 +0.00560182*inputs[0] -0.0367926*inputs[1] -0.0974956*inputs[2] +0.058056*inputs[3] -0.0532506*inputs[4];
	combinations[31] = -0.301689 -0.445224*inputs[0] -0.733411*inputs[1] -0.823282*inputs[2] +0.215949*inputs[3] +0.102626*inputs[4];
	combinations[32] = 0.0413508 +0.296616*inputs[0] -0.033985*inputs[1] +0.0725159*inputs[2] +0.562451*inputs[3] +0.42986*inputs[4];
	combinations[33] = 0.17403 -0.115243*inputs[0] -0.176865*inputs[1] +0.256444*inputs[2] -0.822029*inputs[3] +0.453595*inputs[4];
	combinations[34] = -0.0700633 +0.00584603*inputs[0] -0.0356283*inputs[1] -0.0945978*inputs[2] +0.0569412*inputs[3] -0.051196*inputs[4];
	combinations[35] = -0.0699021 +0.00586006*inputs[0] -0.0355599*inputs[1] -0.09443*inputs[2] +0.0568745*inputs[3] -0.0510783*inputs[4];
	combinations[36] = 0.602222 -0.127697*inputs[0] -0.416866*inputs[1] +0.310509*inputs[2] -0.567189*inputs[3] -0.787158*inputs[4];
	combinations[37] = -0.648684 +0.471303*inputs[0] +0.58066*inputs[1] -1.31845*inputs[2] -0.203147*inputs[3] -0.184904*inputs[4];
	combinations[38] = 0.0702372 -0.00583093*inputs[0] +0.0357019*inputs[1] +0.0947784*inputs[2] -0.0570127*inputs[3] +0.0513228*inputs[4];
	combinations[39] = -0.0700418 +0.00584788*inputs[0] -0.0356193*inputs[1] -0.0945755*inputs[2] +0.0569324*inputs[3] -0.0511804*inputs[4];
	combinations[40] = 1.04243 +0.172531*inputs[0] -1.23632*inputs[1] +0.140292*inputs[2] +0.247966*inputs[3] +0.632746*inputs[4];
	combinations[41] = -0.936958 -1.09982*inputs[0] +0.674647*inputs[1] -0.167692*inputs[2] +0.13818*inputs[3] +0.0656001*inputs[4];
	combinations[42] = 0.137355 -0.159152*inputs[0] +0.606902*inputs[1] +0.0414891*inputs[2] -0.0436777*inputs[3] +0.244032*inputs[4];
	combinations[43] = -0.070111 +0.00584189*inputs[0] -0.0356485*inputs[1] -0.0946474*inputs[2] +0.0569609*inputs[3] -0.0512308*inputs[4];
	combinations[44] = -0.0699104 +0.0058593*inputs[0] -0.0355635*inputs[1] -0.0944387*inputs[2] +0.0568779*inputs[3] -0.0510845*inputs[4];
	combinations[45] = 0.0701955 -0.0058345*inputs[0] +0.0356843*inputs[1] +0.0947352*inputs[2] -0.0569956*inputs[3] +0.0512925*inputs[4];
	combinations[46] = -0.0701136 +0.00584168*inputs[0] -0.0356496*inputs[1] -0.09465*inputs[2] +0.0569619*inputs[3] -0.0512326*inputs[4];
	combinations[47] = 0.0698618 -0.00586352*inputs[0] +0.0355428*inputs[1] +0.094388*inputs[2] -0.0568578*inputs[3] +0.051049*inputs[4];
	combinations[48] = 0.0725758 -0.00562769*inputs[0] +0.0366712*inputs[1] +0.0971892*inputs[2] -0.0579413*inputs[3] +0.0530312*inputs[4];
	combinations[49] = -0.542015 +0.521274*inputs[0] +0.132218*inputs[1] +0.0304543*inputs[2] -0.502821*inputs[3] -0.236892*inputs[4];
	combinations[50] = -0.580343 -0.825687*inputs[0] -0.399925*inputs[1] +0.434355*inputs[2] -0.0251213*inputs[3] +0.0877762*inputs[4];
	combinations[51] = -0.0698535 +0.00586425*inputs[0] -0.0355392*inputs[1] -0.0943793*inputs[2] +0.0568543*inputs[3] -0.0510429*inputs[4];
	combinations[52] = -0.0701318 +0.00584009*inputs[0] -0.0356573*inputs[1] -0.0946688*inputs[2] +0.0569694*inputs[3] -0.0512459*inputs[4];
	combinations[53] = 0.0699989 -0.00585162*inputs[0] +0.035601*inputs[1] +0.0945307*inputs[2] -0.0569145*inputs[3] +0.051149*inputs[4];
	combinations[54] = 0.00934391 +1.71223*inputs[0] +0.76199*inputs[1] +0.0602311*inputs[2] -0.168522*inputs[3] +0.048208*inputs[4];
	combinations[55] = 0.0701552 -0.00583808*inputs[0] +0.0356672*inputs[1] +0.0946933*inputs[2] -0.056979*inputs[3] +0.051263*inputs[4];
	combinations[56] = -0.0701156 +0.00584149*inputs[0] -0.0356505*inputs[1] -0.094652*inputs[2] +0.0569627*inputs[3] -0.0512341*inputs[4];
	combinations[57] = -0.0701277 +0.00584043*inputs[0] -0.0356556*inputs[1] -0.0946646*inputs[2] +0.0569677*inputs[3] -0.0512429*inputs[4];
	combinations[58] = -0.0701492 +0.00583854*inputs[0] -0.0356647*inputs[1] -0.094687*inputs[2] +0.0569766*inputs[3] -0.0512586*inputs[4];
	combinations[59] = -0.0699693 +0.00585418*inputs[0] -0.0355885*inputs[1] -0.0944999*inputs[2] +0.0569023*inputs[3] -0.0511274*inputs[4];
	combinations[60] = 0.810077 +1.49109*inputs[0] +0.30672*inputs[1] -0.00561604*inputs[2] -0.122331*inputs[3] +1.46692*inputs[4];
	combinations[61] = -0.0699547 +0.00585545*inputs[0] -0.0355823*inputs[1] -0.0944847*inputs[2] +0.0568963*inputs[3] -0.0511167*inputs[4];
	combinations[62] = -0.0700732 +0.00584518*inputs[0] -0.0356325*inputs[1] -0.0946079*inputs[2] +0.0569452*inputs[3] -0.0512031*inputs[4];
	combinations[63] = 1.62579 +3.21992*inputs[0] -0.135122*inputs[1] +0.0989514*inputs[2] -0.0353498*inputs[3] +0.465153*inputs[4];
	combinations[64] = 0.0698018 -0.00586876*inputs[0] +0.0355171*inputs[1] +0.0943253*inputs[2] -0.0568328*inputs[3] +0.0510051*inputs[4];
	combinations[65] = 0.070021 -0.0058497*inputs[0] +0.0356104*inputs[1] +0.0945536*inputs[2] -0.0569237*inputs[3] +0.0511651*inputs[4];
	combinations[66] = 0.0699728 -0.00585392*inputs[0] +0.0355899*inputs[1] +0.0945036*inputs[2] -0.0569038*inputs[3] +0.0511299*inputs[4];
	combinations[67] = 0.0701665 -0.00583707*inputs[0] +0.035672*inputs[1] +0.094705*inputs[2] -0.0569836*inputs[3] +0.0512712*inputs[4];
	combinations[68] = -0.0699604 +0.00585496*inputs[0] -0.0355847*inputs[1] -0.0944907*inputs[2] +0.0568986*inputs[3] -0.0511209*inputs[4];
	combinations[69] = 0.0703097 -0.00582461*inputs[0] +0.0357325*inputs[1] +0.0948538*inputs[2] -0.0570425*inputs[3] +0.0513757*inputs[4];
	combinations[70] = 0.0700722 -0.00584526*inputs[0] +0.0356321*inputs[1] +0.094607*inputs[2] -0.0569449*inputs[3] +0.0512024*inputs[4];
	combinations[71] = 0.0729813 -0.00559271*inputs[0] +0.0368351*inputs[1] +0.0976034*inputs[2] -0.058096*inputs[3] +0.0533278*inputs[4];
	combinations[72] = -0.070076 +0.00584493*inputs[0] -0.0356337*inputs[1] -0.094611*inputs[2] +0.0569463*inputs[3] -0.0512053*inputs[4];
	combinations[73] = 0.112056 -0.00643377*inputs[0] +0.0427328*inputs[1] +0.13178*inputs[2] -0.0623581*inputs[3] +0.0817912*inputs[4];
	combinations[74] = -0.0700254 +0.00584934*inputs[0] -0.0356122*inputs[1] -0.0945582*inputs[2] +0.0569254*inputs[3] -0.0511682*inputs[4];
	combinations[75] = -0.0699443 +0.00585635*inputs[0] -0.0355778*inputs[1] -0.0944739*inputs[2] +0.056892*inputs[3] -0.0511092*inputs[4];
	combinations[76] = 0.0702473 -0.00583005*inputs[0] +0.0357062*inputs[1] +0.0947889*inputs[2] -0.0570169*inputs[3] +0.0513302*inputs[4];
	combinations[77] = 0.0705093 -0.00580724*inputs[0] +0.0358167*inputs[1] +0.095061*inputs[2] -0.0571241*inputs[3] +0.0515215*inputs[4];
	combinations[78] = -0.0700753 +0.00584501*inputs[0] -0.0356334*inputs[1] -0.0946102*inputs[2] +0.0569461*inputs[3] -0.0512047*inputs[4];
	combinations[79] = 0.0701118 -0.00584183*inputs[0] +0.0356488*inputs[1] +0.0946481*inputs[2] -0.0569611*inputs[3] +0.0512313*inputs[4];
	combinations[80] = -0.0700717 +0.0058453*inputs[0] -0.0356319*inputs[1] -0.0946064*inputs[2] +0.0569446*inputs[3] -0.051202*inputs[4];
	combinations[81] = 0.0704624 -0.00581132*inputs[0] +0.0357969*inputs[1] +0.0950123*inputs[2] -0.0571049*inputs[3] +0.0514872*inputs[4];
	combinations[82] = 0.0701397 -0.0058394*inputs[0] +0.0356607*inputs[1] +0.0946771*inputs[2] -0.0569726*inputs[3] +0.0512517*inputs[4];
	combinations[83] = 0.0701803 -0.00583584*inputs[0] +0.0356778*inputs[1] +0.0947194*inputs[2] -0.0569894*inputs[3] +0.0512813*inputs[4];
	combinations[84] = -0.0700556 +0.00584671*inputs[0] -0.035625*inputs[1] -0.0945897*inputs[2] +0.056938*inputs[3] -0.0511903*inputs[4];
	combinations[85] = 0.070093 -0.00584344*inputs[0] +0.0356409*inputs[1] +0.0946286*inputs[2] -0.0569534*inputs[3] +0.0512176*inputs[4];
	combinations[86] = 0.0701658 -0.00583711*inputs[0] +0.0356717*inputs[1] +0.0947043*inputs[2] -0.0569834*inputs[3] +0.0512707*inputs[4];
	combinations[87] = 0.0700552 -0.00584672*inputs[0] +0.0356249*inputs[1] +0.0945893*inputs[2] -0.0569378*inputs[3] +0.0511901*inputs[4];
	combinations[88] = 0.0701946 -0.00583462*inputs[0] +0.0356839*inputs[1] +0.0947342*inputs[2] -0.0569953*inputs[3] +0.0512918*inputs[4];
	combinations[89] = 0.578202 +0.461313*inputs[0] -1.63509*inputs[1] -0.0978218*inputs[2] -0.108864*inputs[3] -0.451392*inputs[4];
	combinations[90] = 0.0702083 -0.00583342*inputs[0] +0.0356897*inputs[1] +0.0947485*inputs[2] -0.0570009*inputs[3] +0.0513018*inputs[4];
	combinations[91] = 0.0698146 -0.00586763*inputs[0] +0.0355227*inputs[1] +0.0943389*inputs[2] -0.0568382*inputs[3] +0.0510146*inputs[4];
	combinations[92] = -0.0701588 +0.00583773*inputs[0] -0.0356688*inputs[1] -0.094697*inputs[2] +0.0569805*inputs[3] -0.0512656*inputs[4];
	combinations[93] = 0.07009 -0.00584371*inputs[0] +0.0356396*inputs[1] +0.0946255*inputs[2] -0.0569522*inputs[3] +0.0512154*inputs[4];
	combinations[94] = -0.0701745 +0.00583637*inputs[0] -0.0356754*inputs[1] -0.0947132*inputs[2] +0.056987*inputs[3] -0.0512771*inputs[4];
	combinations[95] = -0.0700915 +0.00584357*inputs[0] -0.0356403*inputs[1] -0.0946271*inputs[2] +0.0569528*inputs[3] -0.0512166*inputs[4];
	combinations[96] = -0.0706354 +0.00579625*inputs[0] -0.0358697*inputs[1] -0.0951916*inputs[2] +0.0571754*inputs[3] -0.0516135*inputs[4];
	combinations[97] = 0.0704735 -0.00581036*inputs[0] +0.0358016*inputs[1] +0.0950237*inputs[2] -0.0571094*inputs[3] +0.0514953*inputs[4];
	combinations[98] = 1.43994 +1.71*inputs[0] -0.101341*inputs[1] -0.0237892*inputs[2] -0.189118*inputs[3] +0.332381*inputs[4];
	combinations[99] = 0.915222 +0.885383*inputs[0] -0.069161*inputs[1] +1.72475*inputs[2] +0.0380077*inputs[3] +0.278743*inputs[4];

	vector<float> activations(100);

	activations[0] = tanh(combinations[0]);
	activations[1] = tanh(combinations[1]);
	activations[2] = tanh(combinations[2]);
	activations[3] = tanh(combinations[3]);
	activations[4] = tanh(combinations[4]);
	activations[5] = tanh(combinations[5]);
	activations[6] = tanh(combinations[6]);
	activations[7] = tanh(combinations[7]);
	activations[8] = tanh(combinations[8]);
	activations[9] = tanh(combinations[9]);
	activations[10] = tanh(combinations[10]);
	activations[11] = tanh(combinations[11]);
	activations[12] = tanh(combinations[12]);
	activations[13] = tanh(combinations[13]);
	activations[14] = tanh(combinations[14]);
	activations[15] = tanh(combinations[15]);
	activations[16] = tanh(combinations[16]);
	activations[17] = tanh(combinations[17]);
	activations[18] = tanh(combinations[18]);
	activations[19] = tanh(combinations[19]);
	activations[20] = tanh(combinations[20]);
	activations[21] = tanh(combinations[21]);
	activations[22] = tanh(combinations[22]);
	activations[23] = tanh(combinations[23]);
	activations[24] = tanh(combinations[24]);
	activations[25] = tanh(combinations[25]);
	activations[26] = tanh(combinations[26]);
	activations[27] = tanh(combinations[27]);
	activations[28] = tanh(combinations[28]);
	activations[29] = tanh(combinations[29]);
	activations[30] = tanh(combinations[30]);
	activations[31] = tanh(combinations[31]);
	activations[32] = tanh(combinations[32]);
	activations[33] = tanh(combinations[33]);
	activations[34] = tanh(combinations[34]);
	activations[35] = tanh(combinations[35]);
	activations[36] = tanh(combinations[36]);
	activations[37] = tanh(combinations[37]);
	activations[38] = tanh(combinations[38]);
	activations[39] = tanh(combinations[39]);
	activations[40] = tanh(combinations[40]);
	activations[41] = tanh(combinations[41]);
	activations[42] = tanh(combinations[42]);
	activations[43] = tanh(combinations[43]);
	activations[44] = tanh(combinations[44]);
	activations[45] = tanh(combinations[45]);
	activations[46] = tanh(combinations[46]);
	activations[47] = tanh(combinations[47]);
	activations[48] = tanh(combinations[48]);
	activations[49] = tanh(combinations[49]);
	activations[50] = tanh(combinations[50]);
	activations[51] = tanh(combinations[51]);
	activations[52] = tanh(combinations[52]);
	activations[53] = tanh(combinations[53]);
	activations[54] = tanh(combinations[54]);
	activations[55] = tanh(combinations[55]);
	activations[56] = tanh(combinations[56]);
	activations[57] = tanh(combinations[57]);
	activations[58] = tanh(combinations[58]);
	activations[59] = tanh(combinations[59]);
	activations[60] = tanh(combinations[60]);
	activations[61] = tanh(combinations[61]);
	activations[62] = tanh(combinations[62]);
	activations[63] = tanh(combinations[63]);
	activations[64] = tanh(combinations[64]);
	activations[65] = tanh(combinations[65]);
	activations[66] = tanh(combinations[66]);
	activations[67] = tanh(combinations[67]);
	activations[68] = tanh(combinations[68]);
	activations[69] = tanh(combinations[69]);
	activations[70] = tanh(combinations[70]);
	activations[71] = tanh(combinations[71]);
	activations[72] = tanh(combinations[72]);
	activations[73] = tanh(combinations[73]);
	activations[74] = tanh(combinations[74]);
	activations[75] = tanh(combinations[75]);
	activations[76] = tanh(combinations[76]);
	activations[77] = tanh(combinations[77]);
	activations[78] = tanh(combinations[78]);
	activations[79] = tanh(combinations[79]);
	activations[80] = tanh(combinations[80]);
	activations[81] = tanh(combinations[81]);
	activations[82] = tanh(combinations[82]);
	activations[83] = tanh(combinations[83]);
	activations[84] = tanh(combinations[84]);
	activations[85] = tanh(combinations[85]);
	activations[86] = tanh(combinations[86]);
	activations[87] = tanh(combinations[87]);
	activations[88] = tanh(combinations[88]);
	activations[89] = tanh(combinations[89]);
	activations[90] = tanh(combinations[90]);
	activations[91] = tanh(combinations[91]);
	activations[92] = tanh(combinations[92]);
	activations[93] = tanh(combinations[93]);
	activations[94] = tanh(combinations[94]);
	activations[95] = tanh(combinations[95]);
	activations[96] = tanh(combinations[96]);
	activations[97] = tanh(combinations[97]);
	activations[98] = tanh(combinations[98]);
	activations[99] = tanh(combinations[99]);

	return activations;
}

vector<float> perceptron_layer_2(const vector<float>& inputs)
{
	vector<float> combinations(1);

	combinations[0] = -0.464471 +0.15179*inputs[0] -0.152049*inputs[1] -0.152371*inputs[2] -0.709238*inputs[3] +0.152118*inputs[4] +0.152087*inputs[5] -0.152264*inputs[6] -0.152734*inputs[7] -1.01086*inputs[8] +3.00891*inputs[9] -0.152714*inputs[10] +0.152498*inputs[11] -0.152389*inputs[12] +0.820702*inputs[13] -0.989275*inputs[14] +0.152189*inputs[15] -0.976993*inputs[16] +0.152187*inputs[17] -0.15222*inputs[18] +0.152277*inputs[19] +0.152302*inputs[20] -0.152404*inputs[21] -0.152076*inputs[22] +0.154574*inputs[23] -0.152508*inputs[24] +0.152233*inputs[25] -0.152486*inputs[26] +0.152655*inputs[27] -0.154552*inputs[28] +0.152274*inputs[29] +0.157495*inputs[30] +0.893211*inputs[31] +0.629855*inputs[32] +0.595588*inputs[33] +0.152219*inputs[34] +0.151915*inputs[35] +0.810117*inputs[36] -1.00446*inputs[37] -0.152547*inputs[38] +0.152179*inputs[39] +1.27685*inputs[40] +1.26841*inputs[41] +0.522878*inputs[42] +0.152309*inputs[43] +0.15193*inputs[44] -0.152469*inputs[45] +0.152314*inputs[46] -0.151839*inputs[47] -0.156935*inputs[48] +0.709495*inputs[49] +0.963772*inputs[50] +0.151823*inputs[51] +0.152348*inputs[52] -0.152098*inputs[53] +0.938756*inputs[54] -0.152393*inputs[55] +0.152318*inputs[56] +0.152341*inputs[57] +0.152381*inputs[58] +0.152042*inputs[59] -1.61002*inputs[60] +0.152014*inputs[61] +0.152238*inputs[62] -1.66798*inputs[63] -0.151725*inputs[64] -0.152139*inputs[65] -0.152048*inputs[66] -0.152414*inputs[67] +0.152025*inputs[68] -0.152684*inputs[69] -0.152236*inputs[70] -0.157691*inputs[71] +0.152243*inputs[72] -0.221168*inputs[73] +0.152147*inputs[74] +0.151994*inputs[75] -0.152566*inputs[76] -0.15306*inputs[77] +0.152242*inputs[78] -0.152311*inputs[79] +0.152235*inputs[80] -0.152972*inputs[81] -0.152363*inputs[82] -0.15244*inputs[83] +0.152205*inputs[84] -0.152275*inputs[85] -0.152412*inputs[86] -0.152204*inputs[87] -0.152467*inputs[88] -1.5504*inputs[89] -0.152493*inputs[90] -0.151749*inputs[91] +0.152399*inputs[92] -0.152269*inputs[93] +0.152429*inputs[94] +0.152272*inputs[95] +0.153298*inputs[96] -0.152993*inputs[97] +1.53986*inputs[98] -1.28226*inputs[99];

	vector<float> activations(1);

	activations[0] = combinations[0];

	return activations;
}

vector<float> unscaling_layer(const vector<float>& inputs)
{
	vector<float> outputs(1);

	outputs[0] = inputs[0]*6.898656845+124.8359451;

	return outputs;
}

vector<float> bounding_layer(const vector<float>& inputs)
{
	vector<float> outputs(1);

	if(inputs[0] < -3.40282e+38)
	{
	    outputs[0] = -3.40282e+38
	}
	else if(inputs[0] > 3.40282e+38)
	{
	    outputs[0] = 3.40282e+38
	}
	else
	{
	    outputs[0] = inputs[0];
	}

	return outputs;
}

vector<float> neural_network(const vector<float>& inputs)
{
	vector<float> outputs;

	outputs = scaling_layer(inputs);
	outputs = perceptron_layer_1(outputs);
	outputs = perceptron_layer_2(outputs);
	outputs = unscaling_layer(outputs);
	outputs = bounding_layer(outputs);

	return outputs;
}
int main(){return 0;}
