//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "model_expression.h"
#include "scaling_layer_2d.h"
#include "strings_utilities.h"
#include "dataset.h"
#include "neural_network.h"

namespace opennn {

ModelExpression::ModelExpression(){}

string ModelExpression::write_comments_c()
{
    return
        "// Artificial Intelligence Techniques SL\n"
        "// artelnics@artelnics.com\n"
        "// Your model has been exported to this c file.\n"
        "// You can manage it with the main method, where you \n"
        "// can change the values of your inputs. For example:\n\n"
        "// if we want to add these 3 values (0.3, 2.5 and 1.8)\n"
        "// to our 3 inputs (Input_1, Input_2 and Input_1), the\n"
        "// main program has to look like this:\n"
        "// \n"
        "// int main(){ \n"
        "// \tvector<float> inputs(3);"
        "// \n"
        "// \tconst float asdas  = 0.3;\n"
        "// \tinputs[0] = asdas;\n"
        "// \tconst float input2 = 2.5;\n"
        "// \tinputs[1] = input2;\n"
        "// \tconst float input3 = 1.8;\n"
        "// \tinputs[2] = input3;\n"
        "// \t. . .\n"
        "// \n"
        "// Input names:";
}


string ModelExpression::write_logistic_c()
{
    return
        "float Logistic (float x) {\n"
        "float z = 1/(1+exp(-x));\n"
        "return z;\n"
        "}\n"
        "\n";
}


string ModelExpression::write_relu_c()
{
    return
        "float ReLU(float x) {\n"
        "float z = max(0, x);\n"
        "return z;\n"
        "}\n"
        "\n";
}


string ModelExpression::write_exponential_linear_c()
{
    return
        "float ExponentialLinear(float x) {\n"
        "float z;\n"
        "float alpha = 1.67326;\n"
        "if(x>0){\n"
        "z = x;\n"
        "}else{\n"
        "z = alpha*(exp(x)-1);\n"
        "}\n"
        "return z;\n"
        "}\n"
        "\n";
}


string ModelExpression::write_selu_c()
{
    return
        "float SELU(float x) {\n"
        "float z;\n"
        "float alpha  = 1.67326;\n"
        "float lambda = 1.05070;\n"
        "if(x > 0){\n"
        "z = lambda*x;\n"
        "}else{\n"
        "z = lambda*alpha*(exp(x)-1);\n"
        "}\n"
        "return z;\n"
        "}\n"
        "\n";
}


void ModelExpression::auto_association_c(const NeuralNetwork& neural_network)
{
    /*
    const NeuralNetwork::ModelType model_type = neural_network.get_model_type();

    string expression;

    size_t index = 0;

    const size_t index = expression.find("sample_autoassociation_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);

    const size_t index = expression.find("sample_autoassociation_variables_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);
*/
}


string ModelExpression::get_expression_c(const NeuralNetwork& neural_network)
{
//    const NeuralNetwork::ModelType model_type = neural_network.get_model_type();

    string aux;
    ostringstream buffer;
    ostringstream outputs_buffer;

    vector<string> input_names =  neural_network.get_input_names();
    vector<string> output_names = neural_network.get_output_names();
    /*
    fix_input_names(input_names);
    fix_output_names(output_names);
*/
    const Index inputs_number = neural_network.get_inputs_number();
    const Index outputs_number = neural_network.get_outputs_number();

    // int cell_states_counter = 0;
    // int hidden_state_counter = 0;

    bool logistic = false;
    bool ReLU = false;
    bool ExpLinear = false;
    bool SExpLinear = false;

    buffer << write_comments_c();

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\n// \t " << i << ")  " << input_names[i];

    buffer << "\n \n \n#include <iostream>\n"
              "#include <vector>\n"
              "#include <math.h>\n"
              "#include <stdio.h>\n \n"
              "using namespace std; \n \n";

    string line;
    const string expression = neural_network.get_expression();

    stringstream string_stream(expression);

    vector<string> lines;

    while(getline(string_stream, line, '\n'))
    {
        if (line.size() > 1)
        {
            if (line.back() == '{')
                break;

            if (line.back() != ';')
                line.append(";");
        }

        lines.push_back(line);
    }

    const vector<pair<string, bool*>> activation_targets =
        {{"Logistic", &logistic},
         {"ReLU", &ReLU},
         {"ExponentialLinear", &ExpLinear},
         {"SELU", &SExpLinear}};

    const Index lines_number = lines.size();

    vector<string> variable_names;

    for (size_t i = 0; i < lines_number; i++)
    {
        const string first_word = get_first_word(lines[i]);

        if (first_word.size() > 1 && !contains(variable_names, first_word))
            variable_names.push_back(first_word);

        for (const auto& [target, flag] : activation_targets)
            if (line.find(target) != string::npos) *flag = true;
    }

    if(logistic) buffer << write_logistic_c();
    if(ReLU) buffer << write_relu_c();
    if(ExpLinear) buffer << write_exponential_linear_c();
    if (SExpLinear) buffer << write_exponential_linear_c();

    buffer << "vector<float> calculate_outputs(const vector<float>& inputs)" << endl
           << "{" << endl;

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\t" << "const float " << input_names[i] << " = " << "inputs[" << to_string(i) << "];" << endl;

    buffer << endl;

    for(size_t i = 0; i < lines_number; i++)
        lines[i].size() <= 1
            ? outputs_buffer << endl
            : outputs_buffer << "\t" << lines[i] << endl;

    const string keyword = "double";

    string outputs_expression = outputs_buffer.str();

    replace_substring_in_string(variable_names, outputs_expression, keyword);

    buffer << outputs_expression;

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, output_names, ProgrammingLanguage::C);

    if(fixed_outputs.size() > 0)
        for(size_t i = 0; i < outputs_number; i++)
            buffer << "\t" << fixed_outputs[i] << endl;

    buffer << "\t" << "vector<float> out(" << outputs_number << ");" << endl;


    for(size_t i = 0; i < outputs_number; i++)
        buffer << "\t" << "out[" << to_string(i) << "] = " << output_names[i] << ";" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}\n"  << endl
           << "int main(){ \n" << endl
           << "\tvector<float> inputs(" << to_string(inputs_number) << "); \n" << endl;

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\t" << "const float " << input_names[i] << " =" << " //enter your value here; " << endl
               << "\t" << "inputs[" << to_string(i) << "] = " << input_names[i] << ";" << endl;

    buffer << endl
           << "\tvector<float> outputs(" << outputs_number <<");" << endl
           << "\n\toutputs = calculate_outputs(inputs);" << endl
           << endl
           << "\t" << "printf(\"These are your outputs:\\n\");" << endl;


    for(size_t i = 0; i < outputs_number; i++)
        buffer << "\t" << "printf( \""<< output_names[i] << ":" << " %f \\n\", "<< "outputs[" << to_string(i) << "]" << ");" << endl;

    buffer << "\n\t" << "return 0;" << endl
           << "} \n" << endl;

    string out = buffer.str();
    replace_all_appearances(out, "double double double", "double");
    replace_all_appearances(out, "double double", "double");

    return out;
}

string ModelExpression::write_header_api()
{
    return
        "<!DOCTYPE html> \n"
        "<!--\n"
        "Artificial Intelligence Techniques SL\n"
        "artelnics@artelnics.com\n\n"
        "Your model has been exported to this php file.\n"
        "You can manage it writting your parameters in the url of your browser.\n"
        "Example:\n\n"
        "\turl = http://localhost/API_example/\n"
        "\tparameters in the url = http://localhost/API_example/?num=5&num=2&...\n"
        "\tTo see the ouput refresh the page\n\n"
        "\tInputs Names: ";

}
string ModelExpression::write_subheader_api(){
    return
        "\n-->\n \n\n"
        "<html lang = \"en\">\n\n"
        "<head>\n\n"
        "<title>Rest API Client Side Demo</title>\n\n"
        "<meta charset = \"utf-8\">\n"
        "<meta name = \"viewport\" content = \"width=device-width, initial-scale=1\">\n"
        "<link rel = \"stylesheet\" href = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css\">\n"
        "<script source = \"https://ajax.googleapis.com/ajax/libs/jquery/3.2.0/jquery.min.js\"></script>\n"
        "<script source = \"https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js\"></script>\n"
        "</head>\n"
        "<style>\n"
        ".btn{\n"
        "\tbackground-color: #7393B3\n"
        "\tborder: none;\n"
        "\tcolor: white;\n"
        "\tpadding: 15px 32px;\n"
        "\ttext-align: center;\n"
        "\tfont-size: 16px;\n"
        "}\n"
        "</style>\n"
        "<body>\n"
        "<div class = \"container\">\n"
        "<br></br>"
        "<div class = \"form-group\">\n"
        "<p>"
        "follow the steps defined in the \"index.php\" file"
        "</p>\n"
        "<p>"
        "Refresh the page to see the prediction"
        "</p>\n"
        "</div>\n"
        "<h4>\n"
       "<?php\n\n";
}


void ModelExpression::autoassociation_api(const NeuralNetwork& neural_network)
{
    string expression;

    size_t index = 0;

    index = expression.find("sample_autoassociation_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);

    index = expression.find("sample_autoassociation_variables_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);
}


string ModelExpression::logistic_api()
{
    return
        "<?php"
        "function Logistic(int $x) {"
        "$z = 1/(1+exp(-$x));"
        "return $z;"
        "}"
        "?>"
        "\n";
}


string ModelExpression::relu_api()
{
    return
        "<?php"
        "function ReLU(int $x) {"
        "$z = max(0, $x);"
        "return $z;"
        "}"
        "?>"
        "\n";
}

string ModelExpression::exponential_linear_api()
{
    return
        "<?php"
        "function ExponentialLinear(int $x) {"
        "$alpha = 1.6732632423543772848170429916717;"
        "if($x>0){"
        "$z=$x;"
        "}else{"
        "$z=$alpha*(exp($x)-1);"
        "}"
        "return $z;"
        "}"
        "?>"
        "\n";
}


string ModelExpression::scaled_exponential_linear_api()
{
    return
        "<?php"
        "function SELU(int $x) {"
        "$alpha  = 1.67326;"
        "$lambda = 1.05070;"
        "if($x>0){"
        "$z=$lambda*$x;"
        "}else{"
        "$z=$lambda*$alpha*(exp($x)-1);"
        "}"
        "return $z;"
        "}"
        "?>"
        "\n";
}


string ModelExpression::get_expression_api(const NeuralNetwork& neural_network)
{

    ostringstream buffer;
    vector<string> found_tokens;

    vector<string> input_names =  neural_network.get_input_names();
    vector<string> output_names = neural_network.get_output_names();

    const Index inputs_number = neural_network.get_inputs_number();
    const Index outputs_number = neural_network.get_outputs_number();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;

    buffer << write_header_api();

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\n\t\t" << i << ")  " << input_names[i];

    buffer << write_subheader_api();

    string line;
    string expression = neural_network.get_expression();

    stringstream string_stream(expression);
    vector<string> lines;

    while(getline(string_stream, line, '\n'))
    {
        if (line.size() <= 1)
            continue;

        if (line.back() == '{')
            break;

        if (line.back() != ';')
            line.append(";");

        lines.push_back(line);
    }

    const Index lines_number = lines.size();

    string word;

    for(size_t i = 0; i < lines_number; i++)
    {
        string line = lines[i];
        word = get_first_word(line);

        if(word.size() > 1)
            found_tokens.push_back(word);
    }

    buffer << "session_start();" << endl
           << "if(isset($_SESSION['lastpage']) && $_SESSION['lastpage'] == __FILE__) { " << endl
           << "if(isset($_SERVER['HTTPS']) && $_SERVER['HTTPS'] === 'on') " << endl
           << "\t$url = \"https://\"; " << endl
           << "else" << endl
           << "\t$url = \"http://\";\n" << endl
           << "$url.= $_SERVER['HTTP_HOST'];" << endl
           << "$url.= $_SERVER['REQUEST_URI'];" << endl
           << "$url_components = parse_url($url);" << endl
           << "parse_str($url_components['query'], $params);\n" << endl;

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "$num" + to_string(i) << " = " << "$params['num" + to_string(i) << "'];" << endl
               << "$" << input_names[i]      << " = intval(" << "$num"  + to_string(i) << ");"  << endl;

    buffer << "if(" << endl;

    for(size_t i = 0; i < inputs_number; i++)
        i != inputs_number - 1
            ? buffer << "is_numeric(" << "$" << "num" + to_string(i) << ") &&" << endl
            : buffer << "is_numeric(" << "$" << "num" + to_string(i) << "))" << endl;

    buffer << "{" << endl
           << "$status=200;" << endl
           << "$status_msg = 'valid parameters';" << endl
           << "}" << endl
           << "else" << endl
           << "{" << endl
           << "$status =400;" << endl
           << "$status_msg = 'invalid parameters';" << endl
           << "}\n"   << endl;

    string target_string0("Logistic");
    string target_string1("ReLU");
    string target_string4("ExponentialLinear");
    string target_string5("SELU");

    size_t substring_length0;
    size_t substring_length1;
    size_t substring_length4;
    size_t substring_length5;

    string new_word;

    vector<string> found_tokens_and_input_names = concatenate_string_vectors(input_names, found_tokens);
    found_tokens_and_input_names = sort_string_vector(found_tokens_and_input_names);

    for(size_t i = 0; i < lines.size(); i++)
    {
        string t = lines[i];

        substring_length0 = t.find(target_string0);
        substring_length1 = t.find(target_string1);
        substring_length4 = t.find(target_string4);
        substring_length5 = t.find(target_string5);

        if(substring_length0 < t.size() && substring_length0!=0){ logistic     = true; }
        if(substring_length1 < t.size() && substring_length1!=0){ ReLU         = true; }
        if(substring_length4 < t.size() && substring_length4!=0){ ExpLinear    = true; }
        if(substring_length5 < t.size() && substring_length5!=0){ SExpLinear   = true; }

        for(size_t i = 0; i < found_tokens_and_input_names.size(); i++)
        {
            new_word = "$" + found_tokens_and_input_names[i];

            replace_all_word_appearances(t, found_tokens_and_input_names[i], new_word);
        }

        buffer << t << endl;
    }

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, output_names, ProgrammingLanguage::PHP);

    for(size_t i = 0; i < fixed_outputs.size(); i++)
        buffer << fixed_outputs[i] << endl;

    buffer << "if($status === 200){" << endl
           << "$response = ['status' => $status,  'status_message' => $status_msg" << endl;

    for(size_t i = 0; i < outputs_number; i++)
        buffer << ", '" << output_names[i] << "' => " << "$" << output_names[i] << endl;

    buffer << "];" << endl
           << "}" << endl
           << "else" << endl
           << "{" << endl
           << "$response = ['status' => $status,  'status_message' => $status_msg" << "];" << endl
           << "}" << endl;

    buffer << "\n$json_response_pretty = json_encode($response, JSON_PRETTY_PRINT);" << endl
           << "echo nl2br(\"\\n\" . $json_response_pretty . \"\\n\");" << endl
           << "}else{" << endl
           << "echo \"New page\";" << endl
           << "}" << endl
           << "$_SESSION['lastpage'] = __FILE__;" << endl
           << "?>\n" << endl;

    if (logistic)
        buffer << logistic_api();

    if(ReLU)
        buffer << "<?php" << endl
               << "function ReLU(int $x) {" << endl
               << "$z = max(0, $x);" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>\n" << endl;

    if(ExpLinear)
        buffer << "<?php" << endl
               << "function ExponentialLinear(int $x) {" << endl
               << "$alpha = 1.6732632423543772848170429916717;" << endl
               << "if($x>0){" << endl
               << "$z=$x;" << endl
               << "}else{" << endl
               << "$z=$alpha*(exp($x)-1);" << endl
               << "}" << endl
               << "return $z;" << endl
               << "}\n" << endl;

    if(SExpLinear)
        buffer << "<?php" << endl
               << "function SELU(int $x) {" << endl
               << "$alpha  = 1.67326;" << endl
               << "$lambda = 1.05070;" << endl
               << "if($x>0){" << endl
               << "$z=$lambda*$x;" << endl
               << "}else{" << endl
               << "$z=$lambda*$alpha*(exp($x)-1);" << endl
               << "}" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>\n" << endl;

    buffer << "</h4>" << endl
           << "</div>" << endl
           << "</body>" << endl
           << "</html>" << endl;

    string out = buffer.str();

    replace_all_appearances(out, "$$", "$");
    replace_all_appearances(out, "_$", "_");

    return out;

    return string();
}


string ModelExpression::autoassociaton_javascript(const NeuralNetwork& neural_network)
{
    string expression;

    size_t index = 0;

    index = expression.find("sample_autoassociation_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);

    index = expression.find("sample_autoassociation_variables_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);

    return expression;
}


string ModelExpression::logistic_javascript()
{
    return
        "function Logistic(x) {\n"
        "\tvar z = 1/(1+Math.exp(x));\n"
        "\treturn z;\n"
        "}\n";
}

string ModelExpression::relu_javascript()
{
    return
        "function ReLU(x) {\n"
        "\tvar z = Math.max(0, x);\n"
        "\treturn z;\n"
        "}\n";
}


string ModelExpression::exponential_linear_javascript()
{
    return
        "function ExponentialLinear(x) {\n"
        "\tvar alpha = 1.67326;\n"
        "\tif(x>0){\n"
        "\t\tvar z = x;\n"
        "\t}else{\n"
        "\t\tvar z = alpha*(Math.exp(x)-1);\n"
        "\t}\n"
        "\treturn z;\n"
        "}\n";
}

string ModelExpression::selu_javascript()
{
    return
        "function SELU(x) {\n"
        "\tvar alpha  = 1.67326;\n"
        "\tvar lambda = 1.05070;\n"
        "\tif(x>0){\n"
        "\t\tvar z = lambda*x;\n"
        "\t}else{\n"
        "\t\tvar z = lambda*alpha*(Math.exp(x)-1);\n"
        "\t}\n"
        "return z;\n"
        "}\n";
}


string ModelExpression::header_javascript()
{
    return
        "<!--\n"
        "Artificial Intelligence Techniques SL\n"
        "artelnics@artelnics.com\n\n"
        "Your model has been exported to this JavaScript file.\n"
        "You can manage it with the main method, where you \n"
        "can change the values of your inputs. For example:\n"
        "if we want to add these 3 values (0.3, 2.5 and 1.8)\n"
        "to our 3 inputs (Input_1, Input_2 and Input_1), the\n"
        "main program has to look like this:\n\n"
        "int neuralNetwork(){\n "
        "\t vector<float> inputs(3);\n"
        "\t const float asdas  = 0.3;\n"
        "\t inputs[0] = asdas;\n"
        "\t const float input2 = 2.5;\n"
        "\t inputs[1] = input2;\n"
        "\t const float input3 = 1.8;\n"
        "\t inputs[2] = input3;\n"
        "\t . . .\n\n"
        "Inputs Names:";
}

string ModelExpression::subheader_javascript()
{
    return
        "\n-->\n\n\n"
        "<!DOCTYPE HTML>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "\t<link href=\"https://www.neuraldesigner.com/assets/css/neuraldesigner.css\" rel=\"stylesheet\" />\n"
        "\t<link href=\"https://www.neuraldesigner.com/images/fav.ico\" rel=\"shortcut icon\" type=\"image/x-icon\" />\n"
        "</head>\n\n"
        "<style>\n\n"
        "body {\n"
        "\tdisplay: flex;\n"
        "\tjustify-content: center;\n"
        "\talign-items: center;\n"
        "\tmin-height: 100vh;\n"
        "\tmargin: 0;\n"
        "\tbackground-color: #f0f0f0;\n"
        "\tfont-family: Arial, sans-serif;\n"
        "}\n\n"
        ".form {\n"
        "\tborder-collapse: collapse;\n"
        "\twidth: 80%; \n"
        "\tmax-width: 600px; \n"
        "\tmargin: 0 auto; \n"
        "\tbackground-color: #fff; \n"
        "\tbox-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); \n"
        "\tborder: 1px solid #777; \n"
        "\tborder-radius: 5px; \n"
        "}\n\n"
        "input[type=\"number\"] {\n"
        "\twidth: 60px; \n"
        "\ttext-align: center; \n"
        "}\n\n"
        ".form th,\n"
        ".form td {\n"
        "\tpadding: 10px;\n"
        "\ttext-align: center\n;"
        "\tfont-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif; \n"
        "}\n\n"
        ".btn {\n"
        "\tbackground-color: #5da9e9;\n"
        "\tborder: none;\n"
        "\tcolor: white;\n"
        "\ttext-align: center;\n"
        "\tfont-size: 16px;\n"
        "\tmargin: 4px;\n"
        "\tcursor: pointer;\n"
        "\tpadding: 10px 20px;\n"
        "\tborder-radius: 5px;\n"
        "\ttransition: background-color 0.3s ease;\n"
        "\tfont-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;\n"
        "}\n\n"
        ".btn:hover {\n"
        "\tbackground-color: #4b92d3; \n"
        "}\n\n"
        "input[type=\"range\"]::-webkit-slider-runnable-track {\n"
        "\tbackground: #5da9e9;\n"
        "\theight: 0.5rem;\n"
        "}\n\n"
        "input[type=\"range\"]::-moz-range-track {\n"
        "\tbackground: #5da9e9;\n"
        "\theight: 0.5rem;\n"
        "}\n\n"
        ".tabla {\n"
        "\twidth: 100%;\n"
        "\tpadding: 5px;\n"
        "\tmargin: 0; \n"
        "}\n\n"
        ".form th {\n"
        "\tbackground-color: #f2f2f2;\n"
        "\tfont-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;\n"
        "}\n"
        "</style>\n\n"
        "<body>\n\n"
        "<section>\n"
        "<br/>\n\n"
        "<div align=\"center\" style=\"display:block;text-align: center;\">\n"
        "<!-- MENU OPTIONS HERE  -->\n"
        "<form style=\"display: inline-block;margin-left: auto; margin-right: auto;\">\n\n"
        "<table border=\"1px\" class=\"form\">\n\n"
        "<h4>INPUTS</h4>\n";
}


string ModelExpression::get_expression_javascript(const NeuralNetwork& neural_network, const vector<Dataset::RawVariable>& raw_variables)
{
    vector<string> lines;
    vector<string> found_tokens;
    vector<string> found_mathematical_expressions;

    vector<string> input_names;
    vector<string> output_names;

    for (const auto& raw_variable : raw_variables)
        if(raw_variable.use == Dataset::VariableUse::Input)
            input_names.push_back(raw_variable.name);
        else if(raw_variable.use == Dataset::VariableUse::Target)
            output_names.push_back(raw_variable.name);


    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    vector<string> fixes_input_names = fix_input_names(input_names);
    vector<string> fixes_output_names = fix_output_names(output_names);

    string token;
    string expression = neural_network.get_expression();

    const int maximum_output_variable_numbers = 5;

    stringstream ss(expression);

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;

    ostringstream buffer;

    buffer << header_javascript();

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\n\t " << i + 1 << ")  " << input_names[i];

    buffer << subheader_javascript();

    if(neural_network.has(Layer::Type::Scaling2d) || neural_network.has(Layer::Type::Scaling4d))
    {
        const vector<Descriptives> inputs_descriptives = static_cast<Scaling2d*>(neural_network.get_first(Layer::Type::Scaling2d))->get_descriptives();

        float min_value;
        float max_value;

        Index i = 0; // Input vector pointers index
        Index j = 0; // Number of categorical & binary variables found

        for (int k = 0; k < inputs_number; ++k) {

            const vector<string> raw_variable_categories = raw_variables[k].categories;

            if (raw_variables[k].type == Dataset::RawVariableType::Numeric) // INPUT & NUMERIC
            {
                min_value = inputs_descriptives[i].minimum;
                max_value = inputs_descriptives[i].maximum;

                buffer << "<!-- "<< to_string(i) <<"scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;
                buffer << "<td> " << input_names[k] << " </td>" << endl;
                buffer << "<td style=\"text-align:center\">" << endl;

                if (min_value==0 && min_value==0)
                {
                    buffer << "<input type=\"range\" id=\"" << fixes_input_names[k] << "\" value=\"" << min_value << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"" << (max_value - min_value)/100 << "\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[k] << "_text')\" />" << endl;
                    buffer << "<input class=\"tabla\" type=\"number\" id=\"" << fixes_input_names[k] << "_text\" value=\"" << min_value << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"" << (max_value - min_value)/100 << "\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[k] << "')\">" << endl;
                }
                else
                {
                    buffer << "<input type=\"range\" id=\"" << fixes_input_names[k] << "\" value=\"" << (min_value + max_value)/2 << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"" << (max_value - min_value)/100 << "\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[k] << "_text')\" />" << endl;
                    buffer << "<input class=\"tabla\" type=\"number\" id=\"" << fixes_input_names[k] << "_text\" value=\"" << (min_value + max_value)/2 << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"" << (max_value - min_value)/100 << "\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[k] << "')\">" << endl;
                }

                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                i += 1;
            }
            else if (raw_variables[k].type == Dataset::RawVariableType::Binary && raw_variable_categories.size() == 2 &&
                     ((raw_variable_categories[0]=="1" && raw_variable_categories[1]=="0") || (raw_variable_categories[1]=="1" && raw_variable_categories[0]=="0")))// INPUT & BINARY (1,0)
            {
                buffer << "<!-- ComboBox Ultima pasada-->" << endl;
                buffer << "<!-- 5scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;

                buffer << "<td> " << input_names[k] << " </td>" << endl;

                buffer << "<td style=\"text-align:center\">" << endl;
                buffer << "<select id=\"Select" << j << "\">" << endl;

                buffer << "<option value=\"" << 0 << "\">" << 0 << "</option>" << endl;
                buffer << "<option value=\"" << 1 << "\">" << 1 << "</option>" << endl;

                buffer << "</select>" << endl;
                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                j += 1;
                i += 1;
            }
            else if (raw_variables[k].type == Dataset::RawVariableType::Binary && raw_variable_categories.size() == 2) // INPUT & BINARY (A,B)
            {
                buffer << "<!-- ComboBox Ultima pasada-->" << endl;
                buffer << "<!-- 5scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;

                buffer << "<td> " << input_names[k] << " </td>" << endl;

                buffer << "<td style=\"text-align:center\">" << endl;
                buffer << "<select id=\"Select" << j << "\">" << endl;

                buffer << "<option value=\"" << 0 << "\">" << raw_variable_categories[0] << "</option>" << endl;
                buffer << "<option value=\"" << 1 << "\">" << raw_variable_categories[1] << "</option>" << endl;

                buffer << "</select>" << endl;
                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                j += 1;
                i += 1;
            }
            else if (raw_variables[k].type == Dataset::RawVariableType::Categorical) // INPUT & CATEGORICAL
            {
                buffer << "<!-- ComboBox Ultima pasada-->" << endl;
                buffer << "<!-- 5scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;

                buffer << "<td> " << input_names[k] << " </td>" << endl;

                buffer << "<td style=\"text-align:center\">" << endl;
                buffer << "<select id=\"Select" << j << "\">" << endl;

                for (int l = 0; l < raw_variable_categories.size(); ++l)
                {
                    buffer << "<option value=\"" << l << "\">" << raw_variable_categories[l] << "</option>" << endl;
                }

                buffer << "</select>" << endl;
                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                j += 1;
                i += raw_variable_categories.size();
            }
        }
    }
    else
        for(size_t i = 0; i < inputs_number; i++)
            buffer << "<!-- "<< to_string(i) <<"no scaling layer -->" << endl
                   << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << input_names[i] << " </td>" << endl
                   << "<td style=\"text-align:center\">" << endl
                   << "<input type=\"range\" id=\"" << fixes_input_names[i] << "\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[i] << "_text')\" />" << endl
                   << "<input class=\"tabla\" type=\"number\" id=\"" << fixes_input_names[i] << "_text\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[i] << "')\">" << endl
                   << "</td>" << endl
                   << "</tr>\n" << endl;


    buffer << "</table>" << endl
           << "</form>\n" << endl;

    if(outputs_number > maximum_output_variable_numbers)
    {
        buffer << "<!-- HIDDEN INPUTS -->" << endl;

        for(size_t i = 0; i < outputs_number; i++)
            buffer << "<input type=\"hidden\" id=\"" << fixes_output_names[i] << "\" value=\"\">" << endl;

        buffer << "\n" << endl;
    }

    buffer << "<div align=\"center\">" << endl
           << "<!-- BUTTON HERE -->" << endl
           << "<button class=\"btn\" onclick=\"neuralNetwork()\">calculate outputs</button>" << endl
           << "</div>\n" << endl
           << "<br/>\n" << endl
           << "<table border=\"1px\" class=\"form\">" << endl
           << "<h4> OUTPUTS </h4>" << endl;

    if(outputs_number > maximum_output_variable_numbers)
    {
        buffer << "<tr style=\"height:3.5em\">" << endl
               << "<td> Target </td>" << endl
               << "<td>" << endl
               << "<select id=\"category_select\" onchange=\"updateSelectedCategory()\">" << endl;

        for(size_t i = 0; i < outputs_number; i++)
            buffer << "<option value=\"" << output_names[i] << "\">" << output_names[i] << "</option>" << endl;

        buffer << "</select>" << endl
               << "</td>" << endl
               << "</tr>\n" << endl
               << "<tr style=\"height:3.5em\">" << endl
               << "<td> Value </td>" << endl
               << "<td>" << endl
               << "<input style=\"text-align:right; padding-right:20px;\" id=\"selected_value\" value=\"\" type=\"text\"  disabled/>" << endl
               << "</td>" << endl
               << "</tr>\n" << endl;
    }
    else
        for(size_t i = 0; i < outputs_number; i++)
            buffer << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << output_names[i] << " </td>" << endl
                   << "<td>" << endl
                   << "<input style=\"text-align:right; padding-right:20px;\" id=\"" << fixes_output_names[i] << "\" value=\"\" type=\"text\"  disabled/>" << endl
                   << "</td>" << endl
                   << "</tr>\n" << endl;


    buffer << "</table>\n" << endl
           << "</form>" << endl
           << "</div>\n" << endl
           << "</section>\n" << endl
           << "<script>" << endl;

    if(outputs_number > maximum_output_variable_numbers)
    {
        buffer << "function updateSelectedCategory() {" << endl
               << "\tvar selectedCategory = document.getElementById(\"category_select\").value;" << endl
               << "\tvar selectedValueElement = document.getElementById(\"selected_value\");" << endl;

        for(size_t i = 0; i < outputs_number; i++)
            buffer << "\tif(selectedCategory === \"" << fixes_output_names[i] << "\") {" << endl
                   << "\t\tselectedValueElement.value = document.getElementById(\"" << fixes_output_names[i] << "\").value;" << endl
                   << "\t}" << endl;

        buffer << "}\n" << endl;
    }

    buffer << "function neuralNetwork()" << endl
           << "{" << endl
           << "\t" << "var inputs = [];" << endl;

    Index j = 0;

    vector<string> variables_input_fixed;
    vector<string> variables_input;

    for (int k = 0; k < inputs_number; ++k) {

        const vector<string> raw_variable_categories = raw_variables[k].categories;

        if (raw_variables[k].type == Dataset::RawVariableType::Numeric) // INPUT & NUMERIC
        {
            buffer << "\t" << "var " << fixes_input_names[k] << " =" << " document.getElementById(\"" << fixes_input_names[k] << "\").value; " << endl;
            buffer << "\t" << "inputs.push(" << fixes_input_names[k] << ");" << endl;

            variables_input_fixed.push_back(fixes_input_names[k]);
            variables_input.push_back(input_names[k]);
        }
        else if (raw_variables[k].type == Dataset::RawVariableType::Binary)// INPUT BINARY
        {
            string aux_buffer = "";

            buffer << "\t" << "var selectElement" << j << "= document.getElementById('Select" << j << "');" << endl;
            buffer << "\t" << "var selectedValue" << j << "= +selectElement" << j << ".value;" << endl;

            buffer << "\t" << "var " << fixes_input_names[k] << "= 0;" << endl;
            aux_buffer = aux_buffer + "inputs.push(" + fixes_input_names[k] + ");" + "\n";

            buffer << "switch (selectedValue" << j << "){" << endl;

            buffer << "\t" << "case " << 0 << ":";
            buffer << "\n" << "\t\t" << fixes_input_names[k] << " = " << "0;" << endl;
            buffer << "\tbreak;" << endl;

            buffer << "case " << 1 << ":";
            buffer << "\n" << "\t\t" << fixes_input_names[k] << " = " << "1;" << endl;
            buffer << "\tbreak;" << endl;

            buffer << "\t" << "default:" << endl;
            buffer << "\t" << "\tbreak;" << endl;
            buffer << "\t" << "}\n" << endl;

            buffer << aux_buffer << endl;

            j += 1;
            variables_input_fixed.push_back(fixes_input_names[k]);
            variables_input.push_back(input_names[k]);
        }
        else if (raw_variables[k].type == Dataset::RawVariableType::Categorical) // INPUT & CATEGORICAL
        {
            string aux_buffer = "";

            buffer << "\t" << "var selectElement" << j << "= document.getElementById('Select" << j << "');" << endl;
            buffer << "\t" << "var selectedValue" << j << "= +selectElement" << j << ".value;" << endl;

            for (int l = 0; l < raw_variable_categories.size(); ++l)
            {
                string category = raw_variable_categories[l];
                string fixed_category = replace_reserved_keywords(category);

                buffer << "\t" << "var " << fixed_category << "= 0;" << endl;
                aux_buffer = aux_buffer + "inputs.push(" + fixed_category + ");" + "\n";
                variables_input_fixed.push_back(fixed_category);
                variables_input.push_back(category);
            }

            buffer << "switch (selectedValue" << j << "){" << endl;

            for (int l = 0; l < raw_variable_categories.size(); ++l)
            {
                string category = raw_variable_categories[l];
                string fixed_category = replace_reserved_keywords(category);

                buffer << "\t" << "case " << l << ":";
                buffer << "\n" << "\t\t" << fixed_category << " = " << "1;" << endl;
                buffer << "\t" << "\tbreak;" << endl;
            }

            buffer << "\t" << "default:" << endl;
            buffer << "\t" << "\tbreak;" << endl;
            buffer << "\t" << "}" << endl;

            buffer << aux_buffer << endl;

            j += 1;
        }
    }

    buffer << "\n" << "\t" << "var outputs = calculate_outputs(inputs); " << endl;

    if(outputs_number > maximum_output_variable_numbers)
        buffer << "\t" << "updateSelectedCategory();" << endl;
    else
        for(size_t i = 0; i < outputs_number; i++)
            buffer << "\t" << "var " << fixes_output_names[i] << " = document.getElementById(\"" << fixes_output_names[i] << "\");" << endl
                   << "\t" << fixes_output_names[i] << ".value = outputs[" << to_string(i) << "].toFixed(4);" << endl;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{')
            break;

        if(token.size() > 1 && token.back() != ';')
            token += ';';

        lines.push_back(token);
    }

    vector<string> variable_scaled(variables_input.size());
    for (size_t i = 0; i < variables_input.size(); ++i)
        variable_scaled[i] = "scaled_" + variables_input[i];

    buffer << "}" << endl
           << "function calculate_outputs(inputs)" << endl
           << "{" << endl;

    for(size_t i = 0; i < variables_input_fixed.size(); i++)
        buffer << "\t" << "var " << variables_input_fixed[i] << " = " << "+inputs[" << to_string(i) << "];" << endl;

    buffer << endl;

    for(size_t i = 0; i < lines.size(); i++)
    {
        const string word = get_first_word(lines[i]);

        if(word.size() > 1)
            found_tokens.push_back(word);
    }

    string target_string_0("Logistic");
    string target_string_1("ReLU");
    string target_string_4("ExponentialLinear");
    string target_string_5("SELU");

    string sufix = "Math.";

    found_mathematical_expressions.push_back("exp");
    found_mathematical_expressions.push_back("tanh");
    found_mathematical_expressions.push_back("max");
    found_mathematical_expressions.push_back("min");

    for(size_t i = 0; i < lines.size(); i++)
    {
        string line = lines[i];

        const size_t substring_length_0 = line.find(target_string_0);
        const size_t substring_length_1 = line.find(target_string_1);
        const size_t substring_length_4 = line.find(target_string_4);
        const size_t substring_length_5 = line.find(target_string_5);

        if(substring_length_1 < line.size() && substring_length_1!=0) ReLU = true;
        if(substring_length_0 < line.size() && substring_length_0!=0) logistic = true;
        if(substring_length_4 < line.size() && substring_length_4!=0) ExpLinear = true;
        if(substring_length_5 < line.size() && substring_length_5!=0) SExpLinear = true;

        for (size_t i = 0; i < variables_input.size(); ++i)
            if(line.find(variables_input[i]) != string::npos)
            {
                string original_input_name = variables_input[i];
                string fix_input_name = replace_reserved_keywords(original_input_name);
                int position = 0;

                while((position = line.find(original_input_name, position)) != string::npos){
                    line.replace(position, original_input_name.length(), fix_input_name);
                    position += fix_input_name.length();
                }
            }

        for (size_t i = 0; i < variable_scaled.size(); ++i)
            if(line.find(variable_scaled[i]) != string::npos)
            {
                string original_input_name = variable_scaled[i];
                string fix_input_name = replace_reserved_keywords(original_input_name);
                int position = 0;

                while((position = line.find(original_input_name, position)) != string::npos){
                    line.replace(position, original_input_name.length(), fix_input_name);
                    position += fix_input_name.length();
                }
            }

        for (size_t i = 0; i < output_names.size(); ++i)
            if(line.find(output_names[i]) != string::npos)
            {
                string original_output_name = output_names[i];
                string fix_output_name = replace_reserved_keywords(original_output_name);
                int position = 0;

                while((position = line.find(original_output_name, position)) != string::npos){
                    line.replace(position, original_output_name.length(), fix_output_name);
                    position += fix_output_name.length();
                }
            }

        for(size_t i = 0; i < found_mathematical_expressions.size(); i++)
        {
            string key_word = found_mathematical_expressions[i];
            string new_word = sufix + key_word;
            replace_all_appearances(line, key_word, new_word);
        }

        line.size() <= 1
            ? buffer << endl
            : buffer << "\t" << "var " << line << endl;
    }

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, output_names, ProgrammingLanguage::JavaScript);

    for(size_t i = 0; i < fixed_outputs.size(); i++)
        buffer << fixed_outputs[i] << endl;

    buffer << "\t" << "var out = [];" << endl;

    for(size_t i = 0; i < outputs_number; i++)
        buffer << "\t" << "out.push(" << fixes_output_names[i] << ");" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}\n" << endl;

    if (logistic) buffer << logistic_javascript();
    if (ReLU) buffer << relu_javascript();
    if (ExpLinear) buffer << exponential_linear_javascript();
    if (SExpLinear) buffer << "scaled_exponential_linear()";

    buffer << "function updateTextInput1(val, id)" << endl
           << "{" << endl
           << "\t"<< "document.getElementById(id).value = val;" << endl
           << "}\n" << endl
           << "window.onresize = showDiv;\n" << endl
           << "</script>\n" << endl
           << "<!--script source=\"https://www.neuraldesigner.com/app/htmlparts/footer.js\"></script-->\n" << endl
           << "</body>\n" << endl
           << "</html>" << endl;

    return buffer.str();
}

string ModelExpression::write_header_python()
{
    return "\'\'\' \n"
           "Artificial Intelligence Techniques SL\n"
           "artelnics@artelnics.com\n\n"
           "Your model has been exported to this python file.\n"
           "You can manage it with the 'NeuralNetwork' class.\n"
           "Example:\n \n"
           "\tmodel = NeuralNetwork()\n"
           "\tsample = [input_1, input_2, input_3, input_4, ...]\n"
           "\toutputs = model.calculate_outputs(sample)\n \n \n"
           "Inputs Names: \n";
}

string ModelExpression::write_subheader_python()
{
    return "\nYou can predict with a batch of samples using calculate_batch_output method\t \n"
           "IMPORTANT: input batch must be <class 'numpy.ndarray'> type\n"
           "Example_1:\n"
           "\tmodel = NeuralNetwork()\n"
           "\tinput_batch = np.array([[1, 2], [4, 5]])\n"
           "\toutputs = model.calculate_batch_output(input_batch)\n"
           "Example_2:\n"
           "\tinput_batch = pd.DataFrame( {'col1': [1, 2], 'col2': [3, 4]})\n"
           "\toutputs = model.calculate_batch_output(input_batch.values)\n"
           "\'\'\' \n" ;

}
string ModelExpression::get_expression_python(const NeuralNetwork& neural_network)
{
    ostringstream buffer;

    vector<string> found_tokens;
    vector<string> found_mathematical_expressions;

    vector<string> inputs = neural_network.get_input_names();
    vector<string> original_inputs = neural_network.get_input_names();
    vector<string> outputs = neural_network.get_output_names();

    const Index inputs_number = inputs.size();
    const Index outputs_number = outputs.size();

    const NeuralNetwork::ModelType model_type = neural_network.get_model_type();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;

    buffer << write_header_python();

    for(Index i = 0; i < inputs_number; i++)
        buffer << "\t" << i << ") " << inputs[i] << endl;

    buffer << write_subheader_python();

    vector<string> lines;

    string expression = neural_network.get_expression();
    string line;

    stringstream string_stream(expression);

    while(getline(string_stream, line, '\n'))
    {
        if(line.size() > 1 && line.back() == '{')
            break;

        if(line.size() > 1 && line.back() != ';')
            line += ';';

        lines.push_back(line);
    }

    const string target_string0("Logistic");
    const string target_string1("ReLU");
    const string target_string4("ExponentialLinear");
    const string target_string5("SELU");

    for(size_t i = 0; i < lines.size(); i++)
    {
        string word;
        string line = lines[i];

        const size_t substring_length0 = line.find(target_string0);
        const size_t substring_length1 = line.find(target_string1);
        const size_t substring_length4 = line.find(target_string4);
        const size_t substring_length5 = line.find(target_string5);

        if(substring_length0 < line.size() && substring_length0 != 0)
            logistic = true;
        if(substring_length1 < line.size() && substring_length1 != 0)
            ReLU = true;
        if(substring_length4 < line.size() && substring_length4 != 0)
            ExpLinear = true;
        if(substring_length5 < line.size() && substring_length5 != 0)
            SExpLinear = true;

        word = get_first_word(line);

        if(word.size() > 1)
            found_tokens.push_back(word);
    }

    buffer << "import numpy as np\n" << endl;

    if(model_type == NeuralNetwork::ModelType::AutoAssociation)
        buffer << "def calculate_distances(input, output):" << endl
               << "\t" << "return (np.linalg.norm(np.array(input)-np.array(output)))/len(input)\n" << endl
               << "def calculate_variables_distances(input, output):" << endl
               << "\t" << "length_vector = len(input)" << endl
               << "\t" << "variables_distances = [None] * length_vector" << endl
               << "\t" << "for i in range(length_vector):" << endl
               << "\t\t" << "variables_distances[i] = (np.linalg.norm(np.array(input[i])-np.array(output[i])))" << endl
               << "\t" << "return variables_distances\n" << endl;

    buffer << "class NeuralNetwork:" << endl;

    if(model_type == NeuralNetwork::ModelType::AutoAssociation)
    {
        /*
        buffer << "\t" << "minimum = " << to_string(distance_descriptives.minimum) << endl;
        buffer << "\t" << "first_quartile = " << to_string(auto_associative_distances_box_plot.first_quartile) << endl;
        buffer << "\t" << "median = " << to_string(auto_associative_distances_box_plot.median) << endl;
        buffer << "\t" << "mean = " << to_string(distance_descriptives.mean) << endl;
        buffer << "\t" << "third_quartile = "  << to_string(auto_associative_distances_box_plot.third_quartile) << endl;
        buffer << "\t" << "maximum = " << to_string(distance_descriptives.maximum) << endl;
        buffer << "\t" << "standard_deviation = " << to_string(distance_descriptives.standard_deviation) << endl;
        buffer << "\n" << endl;
*/
    }

    string inputs_list;

    for(size_t i = 0; i < original_inputs.size();i++)
    {
        inputs_list += "'" + original_inputs[i] + "'";

        if(i < original_inputs.size() - 1)
            inputs_list += ", ";
    }

    buffer << "\t" << "def __init__(self):" << endl
            << "\t\t" << "self.inputs_number = " << to_string(inputs_number) << endl
            << "\t\t" << "self.input_names = [" << inputs_list << "]" << endl;

    buffer << "\n" << endl;

    if(logistic)
        buffer << "\tdef Logistic (x):" << endl
               << "\t\t" << "z = 1/(1+np.exp(-x))" << endl
               << "\t\t" << "return z\n" << endl;

    if(ReLU)
        buffer << "\tdef ReLU (x):" << endl
               << "\t\t" << "z = max(0, x)" << endl
               << "\t\t" << "return z\n" << endl;

    if(ExpLinear)
        buffer << "\tdef ExponentialLinear (x):" << endl
               << "\t\t"   << "float alpha = 1.67326" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "z = x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "z = alpha*(np.exp(x)-1)" << endl
               << "\t\t"   << "return z\n" << endl;

    if(SExpLinear)
        buffer << "\tdef SELU (x):" << endl
               << "\t\t"   << "float alpha = 1.67326" << endl
               << "\t\t"   << "float lambda = 1.05070" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "z = lambda*x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "z = lambda*alpha*(np.exp(x)-1)" << endl
               << "\t\t"   << "return z\n" << endl;

    buffer << "\t" << "def calculate_outputs(self, inputs):" << endl;

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\t\t" << inputs[i] << " = " << "inputs[" << to_string(i) << "]" << endl;

    buffer << endl;

    found_tokens.resize(0);
    found_tokens.push_back("log");
    found_tokens.push_back("exp");
    found_tokens.push_back("tanh");

    found_mathematical_expressions.push_back("Logistic");
    found_mathematical_expressions.push_back("ReLU");
    found_mathematical_expressions.push_back("ExponentialLinear");
    found_mathematical_expressions.push_back("SELU");

    string sufix;
    string new_word;
    string key_word ;

    const Index lines_number = lines.size();

    for(size_t i = 0; i < lines_number; i++)
    {
        string line = lines[i];

        sufix = "np.";

        for(int j = 0; j < found_tokens.size(); j++)
        {
            key_word = found_tokens[j];
            new_word = sufix + key_word;
            replace_all_appearances(line, key_word, new_word);
        }

        sufix = "NeuralNetwork.";

        for(int j = 0; j < found_mathematical_expressions.size(); j++)
        {
            key_word = found_mathematical_expressions[j];
            new_word = sufix + key_word;
            replace_all_appearances(line, key_word, new_word);
        }

        buffer << "\t\t" << line << endl;
    }

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, outputs, ProgrammingLanguage::Python);

    if(model_type != NeuralNetwork::ModelType::AutoAssociation)
        for(size_t i = 0; i < fixed_outputs.size(); i++)
            buffer << "\t\t" << fixed_outputs[i] << endl;

    buffer << "\t\t" << "out = " << "[None]*" << outputs_number << "\n" << endl;

    for(size_t i = 0; i < outputs_number; i++)
        buffer << "\t\t" << "out[" << to_string(i) << "] = " << outputs[i] << endl;

    model_type != NeuralNetwork::ModelType::AutoAssociation
        ? buffer << "\n\t\t" << "return out;" << endl
        : buffer << "\n\t\t" << "return out, sample_autoassociation_distance, sample_autoassociation_variables_distance;" << endl;

    buffer << "\t" << "def calculate_batch_output(self, input_batch):" << endl
           << "\t\toutput_batch = [None]*input_batch.shape[0]\n" << endl
           << "\t\tfor i in range(input_batch.shape[0]):\n" << endl;
    /*
    if(has_recurrent_layer())
        buffer << "\t\t\tif(i%self.current_combination_derivatives == 0):\n" << endl
               << "\t\t\t\tself.hidden_states = "+ to_string(get_recurrent_layer()->get_neurons_number())+"*[0]\n" << endl;
*/
    buffer << "\t\t\tinputs = list(input_batch[i])\n" << endl
           << "\t\t\toutput = self.calculate_outputs(inputs)\n" << endl
           << "\t\t\toutput_batch[i] = output\n"<< endl
           << "\t\treturn output_batch\n"<<endl
           << "def main():" << endl
           << "\n\tinputs = []\n" << endl;

    for(Index i = 0; i < inputs_number; i++)
        buffer << "\t" << inputs[i] << " = " << "#- ENTER YOUR VALUE HERE -#" << endl
               << "\t" << "inputs.append(" << inputs[i] << ")\n" << endl;

    buffer << "\t" << "nn = NeuralNetwork()" << endl
           << "\t" << "outputs = nn.calculate_outputs(inputs)" << endl
           << "\t" << "print(outputs)" << endl
           << "\n" << "main()" << endl;

    string out = buffer.str();

    replace(out, ";", "");

    return out;
}


string ModelExpression::replace_reserved_keywords(string& s)
{
    string out = "";

    if(s[0] == '$')
        out=s;

    for (char c : s)
    {
        if (c == ' ') out += "_";
        else if (c == '.') out += "_dot_";
        else if (c == '/') out += "_div_";
        else if (c == '*') out += "_mul_";
        else if (c == '+') out += "_sum_";
        else if (c == '-') out += "_res_";
        else if (c == '=') out += "_equ_";
        else if (c == '!') out += "_not_";
        else if (c == ',') out += "_colon_";
        else if (c == ';') out += "_semic_";
        else if (c == '\\') out += "_slash_";
        else if (c == '&') out += "_amprsn_";
        else if (c == '?') out += "_ntrgtn_";
        else if (c == '<') out += "_lower_";
        else if (c == '>') out += "_higher_";
        else if (isalnum(c) || c == '_') out += c;
    }

    if(!out.empty() && isdigit(out[0]))
        out = '_' + out;

    unordered_map<string, string> sprcialWords = {
        {"min", "mi_n"},
        {"max", "ma_x"},
        {"exp", "ex_p"},
        {"tanh", "ta_nh"}
    };

    for (const auto& pair : sprcialWords)
    {
        int position = 0;

        while ((position = out.find(pair.first, position)) != string::npos)
        {
            out.replace(position, pair.first.length(), pair.second);
            position += pair.second.length();
        }
    }

    return out;

}


vector<string> ModelExpression::fix_get_expression_outputs(const string& str,
                                                           const vector<string>& outputs,
                                                           const ProgrammingLanguage& programming_Language)
{
    vector<string> out;
    vector<string> tokens;
    vector<string> found_tokens;

    string token;
    string out_string;
    string new_variable;
    string old_variable;
    string expression = str;

    stringstream ss(expression);

    const Index outputs_number = outputs.size();

    const size_t dimension = outputs_number;

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{')
            break;

        if(token.size() > 1 && token.back() != ';')
            token += ';';

        tokens.push_back(token);
    }

    for(size_t i = 0; i < tokens.size(); i++)
    {
        string s = tokens[i];
        string word;

        for(char& c : s)
            if(c!=' ' && c!='=')
                word += c;
            else
                break;

        if(word.size() > 1)
            found_tokens.push_back(word);
    }

    new_variable = found_tokens[found_tokens.size()-1];
    old_variable = outputs[dimension-1];

    if(new_variable != old_variable)
    {
        Index j = found_tokens.size();

        for(Index i = dimension; i --> 0;)
        {
            j -= 1;

            new_variable = found_tokens[j];
            old_variable = outputs[i];

            switch(programming_Language)
            {
            //JavaScript
            case  ProgrammingLanguage::JavaScript:
                out_string = "\tvar "
                             + old_variable
                             + " = "
                             + new_variable
                             + ";";
                break;

                //Php
            case  ProgrammingLanguage::PHP:
                out_string = "$"
                             + old_variable
                             + " = "
                             + "$"
                             + new_variable
                             + ";";
                break;

                //Python
            case  ProgrammingLanguage::Python:
                out_string = old_variable
                             + " = "
                             + new_variable;
                break;

                //C
            case  ProgrammingLanguage::C:
                out_string = "double "
                             + old_variable
                             + " = "
                             + new_variable
                             + ";";
                break;

            default:
                break;
            }

            out.push_back(out_string);
        }
    }
    return out;
}


vector<string> ModelExpression::fix_input_names(vector<string>& input_names)
{
    const Index inputs_number = input_names.size();
    vector<string> fixes_input_names(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
        if(input_names[i].empty())
            fixes_input_names[i] = "input_" + to_string(i);
        else
            fixes_input_names[i] = replace_reserved_keywords(input_names[i]);

    return fixes_input_names;

}


vector<string> ModelExpression::fix_output_names(vector<string>& output_names)
{

    const Index outputs_number = output_names.size();

    vector<string> fixes_output_names(outputs_number);

    for (size_t i = 0; i < outputs_number; i++)
        if (output_names[i].empty())
            fixes_output_names[i] = "output_" + to_string(i);
        else
            fixes_output_names[i] = replace_reserved_keywords(output_names[i]);

    return fixes_output_names;
}


void ModelExpression::save_expression(const string& file_name,
                                      const ProgrammingLanguage& programming_language,
                                      const NeuralNetwork* neural_network,
                                      const vector<Dataset::RawVariable>& raw_variables)
{
    ofstream file(file_name);

    if(!file.is_open())
        throw runtime_error("Cannot open expression text file.\n");

    switch (programming_language) {
        case ProgrammingLanguage::Python:
            file << get_expression_python(*neural_network);
            break;
        case ProgrammingLanguage::C:
            file << get_expression_c(*neural_network);
            break;
        case ProgrammingLanguage::JavaScript:
            file << get_expression_javascript(*neural_network, raw_variables);
            break;
        case ProgrammingLanguage::PHP:
            file << get_expression_api(*neural_network);
            break;
        }

    file.close();
}

}
// Namespace

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
