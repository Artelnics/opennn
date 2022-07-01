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

	combinations[0] = -0.0439474 +0.039634*inputs[0] -0.0205256*inputs[1] -0.0565197*inputs[2] +0.0167137*inputs[3] -0.0255055*inputs[4];
	combinations[1] = 0.0433267 -0.0413183*inputs[0] +0.0201267*inputs[1] +0.0571869*inputs[2] -0.0160281*inputs[3] +0.0250784*inputs[4];
	combinations[2] = -0.151964 +0.069059*inputs[0] -0.0560195*inputs[1] -0.14156*inputs[2] +0.0150746*inputs[3] -0.0587773*inputs[4];
	combinations[3] = 0.0720548 -0.053604*inputs[0] +0.0301037*inputs[1] +0.0849651*inputs[2] -0.0182436*inputs[3] +0.0378256*inputs[4];
	combinations[4] = -0.0540578 +0.0458834*inputs[0] -0.0241482*inputs[1] -0.067815*inputs[2] +0.0177258*inputs[3] -0.0301709*inputs[4];
	combinations[5] = 0.0699986 -0.0520449*inputs[0] +0.0296964*inputs[1] +0.0828392*inputs[2] -0.0186462*inputs[3] +0.0371019*inputs[4];
	combinations[6] = 0.0888998 -0.0581826*inputs[0] +0.0356963*inputs[1] +0.0986882*inputs[2] -0.0179779*inputs[3] +0.0442951*inputs[4];
	combinations[7] = -0.720691 +0.169961*inputs[0] -1.05309*inputs[1] +0.294955*inputs[2] +0.0147051*inputs[3] +0.468969*inputs[4];
	combinations[8] = 0.0881479 -0.0573215*inputs[0] +0.0359332*inputs[1] +0.0981118*inputs[2] -0.0183571*inputs[3] +0.0441208*inputs[4];
	combinations[9] = -0.0591758 +0.0488411*inputs[0] -0.025861*inputs[1] -0.0732718*inputs[2] +0.0178005*inputs[3] -0.0324132*inputs[4];
	combinations[10] = 0.0806381 -0.0555514*inputs[0] +0.0333485*inputs[1] +0.0921268*inputs[2] -0.0183496*inputs[3] +0.0413395*inputs[4];
	combinations[11] = -0.966965 -0.98377*inputs[0] +1.35151*inputs[1] -0.0196748*inputs[2] +0.0603371*inputs[3] -0.047068*inputs[4];
	combinations[12] = -0.0882652 +0.0579932*inputs[0] -0.035535*inputs[1] -0.0981928*inputs[2] +0.0180055*inputs[3] -0.0440724*inputs[4];
	combinations[13] = -0.0906844 +0.0584108*inputs[0] -0.036385*inputs[1] -0.100071*inputs[2] +0.0180297*inputs[3] -0.0449467*inputs[4];
	combinations[14] = -0.785606 +0.403483*inputs[0] +0.615947*inputs[1] -1.27857*inputs[2] -0.113365*inputs[3] -0.338069*inputs[4];
	combinations[15] = 0.0822672 -0.0566119*inputs[0] +0.0334146*inputs[1] +0.0934855*inputs[2] -0.018129*inputs[3] +0.0418466*inputs[4];
	combinations[16] = 0.0786406 -0.0553691*inputs[0] +0.0325015*inputs[1] +0.0904909*inputs[2] -0.0182158*inputs[3] +0.0405163*inputs[4];
	combinations[17] = -0.110777 +0.0620904*inputs[0] -0.0432944*inputs[1] -0.114975*inputs[2] +0.0174233*inputs[3] -0.0512761*inputs[4];
	combinations[18] = 0.0341645 -0.0356598*inputs[0] +0.0164397*inputs[1] +0.0466646*inputs[2] -0.0142153*inputs[3] +0.0203987*inputs[4];
	combinations[19] = 0.0740265 +0.7144*inputs[0] +0.546123*inputs[1] +0.398773*inputs[2] -0.412945*inputs[3] -0.430235*inputs[4];
	combinations[20] = 0.16184 +0.308692*inputs[0] -0.0883706*inputs[1] +0.059954*inputs[2] +0.472565*inputs[3] +0.308568*inputs[4];
	combinations[21] = -0.0951328 +1.2088*inputs[0] -0.645221*inputs[1] +0.193693*inputs[2] -0.134095*inputs[3] -0.064002*inputs[4];
	combinations[22] = 0.011479 -0.01533*inputs[0] +0.0062436*inputs[1] +0.0175009*inputs[2] -0.00595155*inputs[3] +0.00753721*inputs[4];
	combinations[23] = -0.100773 +0.060079*inputs[0] -0.0400735*inputs[1] -0.107746*inputs[2] +0.01796*inputs[3] -0.0483125*inputs[4];
	combinations[24] = 0.0700173 -0.0523923*inputs[0] +0.029681*inputs[1] +0.0830559*inputs[2] -0.0184306*inputs[3] +0.0370892*inputs[4];
	combinations[25] = 0.0335275 -0.0363482*inputs[0] +0.0160973*inputs[1] +0.0465255*inputs[2] -0.0138829*inputs[3] +0.0200002*inputs[4];
	combinations[26] = -0.111488 +0.0628045*inputs[0] -0.0430322*inputs[1] -0.115229*inputs[2] +0.0170911*inputs[3] -0.0513049*inputs[4];
	combinations[27] = -0.0823093 -0.557379*inputs[0] -0.12427*inputs[1] -0.427911*inputs[2] -0.0415786*inputs[3] -0.351307*inputs[4];
	combinations[28] = 0.244052 -0.0812524*inputs[0] +0.0843853*inputs[1] +0.187669*inputs[2] -0.0127396*inputs[3] +0.0527647*inputs[4];
	combinations[29] = -0.71355 +0.394125*inputs[0] +1.05214*inputs[1] +0.213825*inputs[2] -0.173321*inputs[3] -0.256863*inputs[4];
	combinations[30] = 0.222775 -0.079177*inputs[0] +0.0761263*inputs[1] +0.177922*inputs[2] -0.012392*inputs[3] +0.0577146*inputs[4];
	combinations[31] = -2.2355 -2.52206*inputs[0] -0.0376177*inputs[1] -0.258977*inputs[2] +0.0545359*inputs[3] -0.420516*inputs[4];
	combinations[32] = -0.0209375 +0.023453*inputs[0] -0.0107779*inputs[1] -0.0291965*inputs[2] +0.0101943*inputs[3] -0.0132105*inputs[4];
	combinations[33] = -0.0867641 +0.056857*inputs[0] -0.0355726*inputs[1] -0.0970344*inputs[2] +0.0183889*inputs[3] -0.0436389*inputs[4];
	combinations[34] = 0.155919 -0.0702547*inputs[0] +0.0564606*inputs[1] +0.143403*inputs[2] -0.0145846*inputs[3] +0.059271*inputs[4];
	combinations[35] = 0.140427 -0.0675154*inputs[0] +0.0523436*inputs[1] +0.134425*inputs[2] -0.0156536*inputs[3] +0.057261*inputs[4];
	combinations[36] = -0.108632 +0.0615747*inputs[0] -0.0427492*inputs[1] -0.113491*inputs[2] +0.0176204*inputs[3] -0.0505753*inputs[4];
	combinations[37] = -0.0792677 +0.0550088*inputs[0] -0.0328259*inputs[1] -0.0909043*inputs[2] +0.0185429*inputs[3] -0.0408177*inputs[4];
	combinations[38] = -0.0114011 +0.012726*inputs[0] -0.0060138*inputs[1] -0.0156381*inputs[2] +0.00588473*inputs[3] -0.00736356*inputs[4];
	combinations[39] = 0.316764 +0.302325*inputs[0] -0.0237019*inputs[1] -0.628239*inputs[2] +0.195537*inputs[3] -0.0390227*inputs[4];
	combinations[40] = 0.0371035 -0.0384857*inputs[0] +0.0176353*inputs[1] +0.0506916*inputs[2] -0.0146614*inputs[3] +0.0219259*inputs[4];
	combinations[41] = -0.0819819 +0.0563183*inputs[0] -0.0336237*inputs[1] -0.093193*inputs[2] +0.0181217*inputs[3] -0.0417905*inputs[4];
	combinations[42] = -0.0798198 +0.0549642*inputs[0] -0.0331975*inputs[1] -0.0914065*inputs[2] +0.0185628*inputs[3] -0.0410711*inputs[4];
	combinations[43] = 0.0951414 -0.124987*inputs[0] +0.0398098*inputs[1] +0.0427145*inputs[2] -0.322379*inputs[3] -0.246385*inputs[4];
	combinations[44] = 0.0582786 -0.0489501*inputs[0] +0.0253999*inputs[1] +0.0725784*inputs[2] -0.0176137*inputs[3] +0.0319329*inputs[4];
	combinations[45] = 0.116028 -0.0634033*inputs[0] +0.0447065*inputs[1] +0.118477*inputs[2] -0.0169983*inputs[3] +0.0524903*inputs[4];
	combinations[46] = 0.408125 +0.217487*inputs[0] -0.15594*inputs[1] +0.4394*inputs[2] -0.825092*inputs[3] -0.865219*inputs[4];
	combinations[47] = 0.0808823 -0.0557395*inputs[0] +0.0332664*inputs[1] +0.0923109*inputs[2] -0.0183766*inputs[3] +0.0414012*inputs[4];
	combinations[48] = -0.229716 +0.0809676*inputs[0] -0.0750223*inputs[1] -0.180315*inputs[2] +0.0116366*inputs[3] -0.0598762*inputs[4];
	combinations[49] = -0.161259 +0.0711204*inputs[0] -0.0581012*inputs[1] -0.146368*inputs[2] +0.0142914*inputs[3] -0.0594957*inputs[4];
	combinations[50] = 0.0193367 +1.89902*inputs[0] +0.866401*inputs[1] +0.25895*inputs[2] -0.14275*inputs[3] +0.0955886*inputs[4];
	combinations[51] = -0.06958 +0.0530324*inputs[0] -0.0293207*inputs[1] -0.0828784*inputs[2] +0.0180008*inputs[3] -0.0368237*inputs[4];
	combinations[52] = -0.074403 +0.0533731*inputs[0] -0.0313956*inputs[1] -0.0868357*inputs[2] +0.0185726*inputs[3] -0.0389599*inputs[4];
	combinations[53] = -0.0919584 +0.0586144*inputs[0] -0.0369031*inputs[1] -0.101046*inputs[2] +0.01802*inputs[3] -0.0453896*inputs[4];
	combinations[54] = -1.56961 -2.96523*inputs[0] +0.208347*inputs[1] -0.148926*inputs[2] +0.0284314*inputs[3] -0.534937*inputs[4];
	combinations[55] = -0.818889 -1.1327*inputs[0] -0.627322*inputs[1] +0.405325*inputs[2] +0.0325762*inputs[3] +0.223841*inputs[4];
	combinations[56] = 0.0599117 -0.049698*inputs[0] +0.02605*inputs[1] +0.074182*inputs[2] -0.0175517*inputs[3] +0.0327076*inputs[4];
	combinations[57] = 0.171976 +0.528314*inputs[0] +0.999767*inputs[1] +0.825206*inputs[2] -0.326855*inputs[3] -0.168605*inputs[4];
	combinations[58] = 0.0267775 -0.027901*inputs[0] +0.0134517*inputs[1] +0.0366507*inputs[2] -0.0123689*inputs[3] +0.0165005*inputs[4];
	combinations[59] = 0.0483771 -0.0441472*inputs[0] +0.0220385*inputs[1] +0.0626763*inputs[2] -0.0166699*inputs[3] +0.0275169*inputs[4];
	combinations[60] = 0.0594713 -0.0497974*inputs[0] +0.0256765*inputs[1] +0.0738596*inputs[2] -0.0176011*inputs[3] +0.0323762*inputs[4];
	combinations[61] = -0.0761423 +0.0548354*inputs[0] -0.0314877*inputs[1] -0.0884446*inputs[2] +0.0182308*inputs[3] -0.0394862*inputs[4];
	combinations[62] = 2.01285 +2.27614*inputs[0] +0.253539*inputs[1] +0.599397*inputs[2] -0.19387*inputs[3] +0.56693*inputs[4];
	combinations[63] = -0.0917728 +0.051249*inputs[0] -0.0372074*inputs[1] -0.115367*inputs[2] +0.01905*inputs[3] -0.0704253*inputs[4];
	combinations[64] = 0.0843549 -0.0565238*inputs[0] +0.0346043*inputs[1] +0.0951075*inputs[2] -0.0182893*inputs[3] +0.042731*inputs[4];
	combinations[65] = 0.0790961 -0.0550351*inputs[0] +0.0329347*inputs[1] +0.0908227*inputs[2] -0.0183972*inputs[3] +0.0407788*inputs[4];
	combinations[66] = 0.0365885 -0.0357969*inputs[0] +0.0175173*inputs[1] +0.0485764*inputs[2] -0.0151018*inputs[3] +0.0216936*inputs[4];
	combinations[67] = 0.0144233 -0.0168399*inputs[0] +0.00763615*inputs[1] +0.0204712*inputs[2] -0.00758687*inputs[3] +0.00938743*inputs[4];
	combinations[68] = 0.0914956 -0.197398*inputs[0] +0.0876061*inputs[1] -0.225485*inputs[2] -0.129703*inputs[3] -0.064501*inputs[4];
	combinations[69] = -0.0751392 +0.0543366*inputs[0] -0.0313853*inputs[1] -0.0875689*inputs[2] +0.0182343*inputs[3] -0.0391545*inputs[4];
	combinations[70] = -0.140582 +0.0671944*inputs[0] -0.0525122*inputs[1] -0.134755*inputs[2] +0.0157756*inputs[3] -0.0575242*inputs[4];
	combinations[71] = -0.00916339 -0.0903834*inputs[0] +0.159151*inputs[1] -0.120766*inputs[2] +0.703163*inputs[3] -0.412621*inputs[4];
	combinations[72] = 0.036256 -0.0383371*inputs[0] +0.0171537*inputs[1] +0.0497894*inputs[2] -0.0144998*inputs[3] +0.0213779*inputs[4];
	combinations[73] = -0.214225 +0.0827107*inputs[0] -0.055638*inputs[1] -0.171094*inputs[2] +0.00827269*inputs[3] -0.0772239*inputs[4];
	combinations[74] = -0.300451 -0.35133*inputs[0] -0.0679754*inputs[1] +0.181768*inputs[2] +0.76839*inputs[3] +0.463768*inputs[4];
	combinations[75] = -0.0205635 +0.0242954*inputs[0] -0.0106443*inputs[1] -0.0296847*inputs[2] +0.00994263*inputs[3] -0.0129764*inputs[4];
	combinations[76] = 0.0124618 -0.0153097*inputs[0] +0.00667727*inputs[1] +0.0180972*inputs[2] -0.00645787*inputs[3] +0.00802111*inputs[4];
	combinations[77] = 0.118135 -0.0637799*inputs[0] +0.0453798*inputs[1] +0.119903*inputs[2] -0.016864*inputs[3] +0.0529684*inputs[4];
	combinations[78] = 0.0636956 -0.0490523*inputs[0] +0.0279057*inputs[1] +0.0769872*inputs[2] -0.0186854*inputs[3] +0.034578*inputs[4];
	combinations[79] = -0.203551 -0.84476*inputs[0] +0.404197*inputs[1] -0.279509*inputs[2] +0.653327*inputs[3] +0.960379*inputs[4];
	combinations[80] = 0.253448 +0.592301*inputs[0] +0.380528*inputs[1] -0.651276*inputs[2] -0.0918479*inputs[3] +0.123539*inputs[4];
	combinations[81] = 0.129584 -0.0655193*inputs[0] +0.0490998*inputs[1] +0.127695*inputs[2] -0.0163555*inputs[3] +0.0555772*inputs[4];
	combinations[82] = -0.0568633 +0.0481225*inputs[0] -0.0251257*inputs[1] -0.0711858*inputs[2] +0.0174186*inputs[3] -0.0314373*inputs[4];
	combinations[83] = 0.101644 -0.060718*inputs[0] +0.0400173*inputs[1] +0.108283*inputs[2] -0.0176464*inputs[3] +0.0485282*inputs[4];
	combinations[84] = -0.115844 +0.0636469*inputs[0] -0.0443873*inputs[1] -0.118211*inputs[2] +0.0168432*inputs[3] -0.0524081*inputs[4];
	combinations[85] = -0.14298 +0.0678637*inputs[0] -0.0531807*inputs[1] -0.136026*inputs[2] +0.0155298*inputs[3] -0.0576277*inputs[4];
	combinations[86] = 0.0724838 -0.0523621*inputs[0] +0.030765*inputs[1] +0.0850139*inputs[2] -0.018829*inputs[3] +0.0382199*inputs[4];
	combinations[87] = 0.26825 -0.0173009*inputs[0] +0.797991*inputs[1] +0.163873*inputs[2] -0.145344*inputs[3] +0.371633*inputs[4];
	combinations[88] = -0.0498892 +0.0427068*inputs[0] -0.022625*inputs[1] -0.062384*inputs[2] +0.0177818*inputs[3] -0.0282464*inputs[4];
	combinations[89] = 0.115073 -0.0626553*inputs[0] +0.0445447*inputs[1] +0.118039*inputs[2] -0.0173502*inputs[3] +0.0526189*inputs[4];
	combinations[90] = 0.101123 -0.0594405*inputs[0] +0.0408291*inputs[1] +0.108197*inputs[2] -0.0182669*inputs[3] +0.0483793*inputs[4];
	combinations[91] = -0.105262 +0.0604761*inputs[0] -0.0419499*inputs[1] -0.111183*inputs[2] +0.0179985*inputs[3] -0.0496815*inputs[4];
	combinations[92] = 0.104594 -0.061364*inputs[0] +0.0409291*inputs[1] +0.110401*inputs[2] -0.0174973*inputs[3] +0.0494004*inputs[4];
	combinations[93] = 0.643161 +1.56053*inputs[0] +0.607548*inputs[1] +0.209986*inputs[2] -0.139229*inputs[3] +1.21863*inputs[4];
	combinations[94] = 0.858145 +0.472639*inputs[0] +0.0308398*inputs[1] +1.54343*inputs[2] -0.0267775*inputs[3] +0.204138*inputs[4];
	combinations[95] = -0.0155001 +0.0194365*inputs[0] -0.00821119*inputs[1] -0.0229514*inputs[2] +0.00784145*inputs[3] -0.00997843*inputs[4];
	combinations[96] = -0.723711 -1.12341*inputs[0] +0.0407733*inputs[1] -0.926443*inputs[2] +0.217597*inputs[3] -0.0595846*inputs[4];
	combinations[97] = 0.0258767 -0.0288961*inputs[0] +0.0128583*inputs[1] +0.0360461*inputs[2] -0.0118912*inputs[3] +0.0158516*inputs[4];
	combinations[98] = -0.0688291 +0.0525515*inputs[0] -0.0292331*inputs[1] -0.0821538*inputs[2] +0.0179897*inputs[3] -0.036585*inputs[4];
	combinations[99] = 0.0335856 -0.0343868*inputs[0] +0.0161134*inputs[1] +0.0449181*inputs[2] -0.0142974*inputs[3] +0.0200314*inputs[4];

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

	combinations[0] = -0.470303 +0.0926513*inputs[0] -0.0904245*inputs[1] +0.248534*inputs[2] -0.140502*inputs[3] +0.110695*inputs[4] -0.137503*inputs[5] -0.16519*inputs[6] -1.02582*inputs[7] -0.163706*inputs[8] +0.118843*inputs[9] -0.152623*inputs[10] +1.20175*inputs[11] +0.164164*inputs[12] +0.167827*inputs[13] -1.11087*inputs[14] -0.155885*inputs[15] -0.149586*inputs[16] +0.195954*inputs[17] -0.0738205*inputs[18] +0.963572*inputs[19] +0.422167*inputs[20] +0.935156*inputs[21] -0.0253064*inputs[22] +0.182094*inputs[23] -0.136987*inputs[24] -0.0726183*inputs[25] +0.196578*inputs[26] -0.426567*inputs[27] -0.356374*inputs[28] -1.02589*inputs[29] -0.331841*inputs[30] -2.33227*inputs[31] +0.0468544*inputs[32] +0.161463*inputs[33] -0.25361*inputs[34] -0.234799*inputs[35] +0.193113*inputs[36] +0.151396*inputs[37] +0.0257625*inputs[38] -0.567907*inputs[39] -0.0784945*inputs[40] +0.154212*inputs[41] +0.151676*inputs[42] -0.348395*inputs[43] -0.117624*inputs[44] -0.202899*inputs[45] +0.925395*inputs[46] -0.153706*inputs[47] +0.340916*inputs[48] +0.260056*inputs[49] +1.30081*inputs[50] +0.13543*inputs[51] +0.143138*inputs[52] +0.16947*inputs[53] +1.88829*inputs[54] +1.26818*inputs[55] -0.119088*inputs[56] -0.939674*inputs[57] -0.0586133*inputs[58] -0.099165*inputs[59] -0.120082*inputs[60] +0.146544*inputs[61] +2.33198*inputs[62] +0.184821*inputs[63] -0.157905*inputs[64] -0.149671*inputs[65] -0.0788176*inputs[66] -0.0330818*inputs[67] +0.278568*inputs[68] +0.144055*inputs[69] +0.23472*inputs[70] -0.555604*inputs[71] -0.0781482*inputs[72] +0.324583*inputs[73] +0.551803*inputs[74] +0.0453161*inputs[75] -0.027898*inputs[76] -0.205426*inputs[77] -0.126881*inputs[78] +0.944366*inputs[79] -0.819287*inputs[80] -0.220756*inputs[81] +0.113347*inputs[82] -0.183246*inputs[83] +0.202389*inputs[84] +0.237857*inputs[85] -0.141042*inputs[86] +0.686625*inputs[87] +0.104606*inputs[88] -0.202173*inputs[89] -0.182661*inputs[90] +0.188394*inputs[91] -0.187309*inputs[92] -1.37963*inputs[93] -1.24295*inputs[94] +0.0346546*inputs[95] +0.992768*inputs[96] -0.0580053*inputs[97] +0.133324*inputs[98] -0.0742971*inputs[99];

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
