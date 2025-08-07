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

ModelExpression::ModelExpression(const NeuralNetwork* neural_network) : neural_network(neural_network) {}

string ModelExpression::write_comments_c() const
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


string ModelExpression::write_logistic_c() const
{
    return
        "float Logistic(float x) {\n"
        "\tfloat z = 1.0f / (1.0f + expf(-x));\n"
        "\treturn z;\n"
        "}\n\n";
}


string ModelExpression::write_relu_c() const
{
    return
        "float RectifiedLinear(float x) {\n"
        "\tfloat z = fmaxf(0.0f, x);\n"
        "\treturn z;\n"
        "}\n\n";
}


string ModelExpression::write_exponential_linear_c() const
{
    return
        "float ExponentialLinear(float x) {\n"
        "\tfloat z;\n"
        "\tconst float alpha = 1.67326f;\n"
        "\tif (x > 0.0f) {\n"
        "\t\tz = x;\n"
        "\t} else {\n"
        "\t\tz = alpha * (expf(x) - 1.0f);\n"
        "\t}\n"
        "\treturn z;\n"
        "}\n\n";
}


string ModelExpression::write_selu_c() const
{
    return
        "float SELU(float x) {\n"
        "\tfloat z;\n"
        "\tconst float alpha = 1.67326f;\n"
        "\tconst float lambda = 1.05070f;\n"
        "\tif (x > 0.0f) {\n"
        "\t\tz = lambda * x;\n"
        "\t} else {\n"
        "\t\tz = lambda * alpha * (expf(x) - 1.0f);\n"
        "\t}\n"
        "\treturn z;\n"
        "}\n\n";
}

/*
void ModelExpression::auto_association_c() const

{
    const NeuralNetwork::ModelType model_type = neural_network.get_model_type();

    string expression;

    size_t index = 0;

    const size_t index = expression.find("sample_autoassociation_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);

    const size_t index = expression.find("sample_autoassociation_variables_distance =");

    if (index != string::npos)
        expression.erase(index, string::npos);
}
*/


string ModelExpression::get_expression_c() const
{
    string aux;
    ostringstream buffer;
    ostringstream outputs_buffer;

    vector<string> input_names =  neural_network->get_input_names();
    vector<string> output_names = neural_network->get_output_names();

    vector<string> fixes_input_names = fix_input_names(input_names);
    vector<string> fixes_output_names = fix_output_names(output_names);

    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    // int cell_states_counter = 0;
    // int hidden_state_counter = 0;

    bool logistic = false;
    bool ReLU = false;
    bool ExpLinear = false;
    bool SExpLinear = false;

    buffer << write_comments_c();

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\n// \t " << i << ")  " << input_names[i];

    buffer << "\n \n \n#include <stdio.h>\n"
              "#include <stdlib.h>\n"
              "#include <math.h>\n\n";

    string line;
    const string expression = neural_network->get_expression();

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
         {"RectifiedLinear", &ReLU},
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

    buffer << "\nfloat Linear (float x) {\n"
           << "\treturn x;\n"
           << "}\n\n";

    if (expression.find("Logistic") != string::npos) buffer << write_logistic_c();
    if (expression.find("RectifiedLinear") != string::npos) buffer << write_relu_c();
    if (expression.find("ExponentialLinear") != string::npos) buffer << write_exponential_linear_c();
    if (expression.find("SELU") != string::npos) buffer << write_selu_c();
    buffer << "\n";

    buffer << "float* calculate_outputs(const float* inputs)" << endl
           << "{" << endl;

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\t" << "const float " << fixes_input_names[i] << " = " << "inputs[" << to_string(i) << "];" << endl;

    buffer << endl;

    for(size_t i = 0; i < lines_number; i++)
        lines[i].size() <= 1
            ? outputs_buffer << endl
            : outputs_buffer << "\t" << lines[i] << endl;

    const string keyword = "double";

    string outputs_expression = outputs_buffer.str();

    replace_substring_in_string(variable_names, outputs_expression, keyword);

    for(size_t i = 0; i < inputs_number; i++)
    {
        replace_all_word_appearances(outputs_expression, "scaled_" + input_names[i], "scaled_" + fixes_input_names[i]);
        replace_all_word_appearances(outputs_expression, input_names[i], fixes_input_names[i]);
    }

    buffer << outputs_expression;

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, output_names, ProgrammingLanguage::C);

    for(size_t i = 0; i < outputs_number; i++)
        buffer << "\t" << fixed_outputs[i] << endl;

    buffer << endl;
    buffer << "\t" << "float* out = (float*)malloc(" << outputs_number << " * sizeof(float));" << endl;
    buffer << "\t" << "if (out == NULL) {" << endl
           << "\t\t" << "printf(\"Error: Memory allocation failed in calculate_outputs.\\n\");" << endl
           << "\t\t" << "return NULL;" << endl
           << "\t" << "}" << endl << endl;

    for(size_t i = 0; i < outputs_number; i++)
        buffer << "\t" << "out[" << to_string(i) << "] = " << fixes_output_names[i] << ";" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}\n"  << endl
           << "\n" << "int main() { \n" << endl;

    buffer << "\t" << "float* inputs = (float*)malloc(" << to_string(inputs_number) << " * sizeof(float)); \n"
           << "\t" << "if (inputs == NULL) {\n"
           << "\t\t" << "printf(\"Error: Memory allocation failed for inputs.\\n\");\n"
           << "\t\t" << "return 1;\n"
           << "\t" << "}\n" << endl;

    buffer << "\t" << "// Please enter your values here:" << endl;
    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\t" << "inputs[" << to_string(i) << "] = 0.0f; // " << fixes_input_names[i] << endl;
    buffer << endl;

    buffer << "\t" << "float* outputs;" << endl
           << "\n\toutputs = calculate_outputs(inputs);" << endl << endl
           << "\t" << "if (outputs != NULL) {" << endl
           << "\t\t" << "printf(\"These are your outputs:\\n\");" << endl;

    for(size_t i = 0; i < outputs_number; i++)
        buffer << "\t\t" << "printf(\""<< fixes_output_names[i] << ": %f \\n\", outputs[" << to_string(i) << "]);" << endl;
    buffer << "\t" << "}" << endl << endl;

    buffer << "\t" << "// Free the allocated memory" << endl
           << "\t" << "free(inputs);" << endl
           << "\t" << "free(outputs);" << endl << endl;

    buffer << "\t" << "return 0;" << endl
           << "} \n" << endl;

    string out = buffer.str();
    replace_all_appearances(out, "double double double", "double");
    replace_all_appearances(out, "double double", "double");

    return out;
}

string ModelExpression::write_header_api() const
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
string ModelExpression::write_subheader_api() const{
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
        "\tbackground-color: #7393B3;\n"
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


void ModelExpression::autoassociation_api() const
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


string ModelExpression::logistic_api() const
{
    return
        "<?php"
        "function Logistic(float $x) {"
        "$z = 1/(1+exp(-$x));"
        "return $z;"
        "}"
        "?>"
        "\n";
}


string ModelExpression::relu_api() const
{
    return
        "<?php"
        "function RectifiedLinear(int $x) {"
        "$z = max(0, $x);"
        "return $z;"
        "}"
        "?>"
        "\n";
}

string ModelExpression::exponential_linear_api() const
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


string ModelExpression::scaled_exponential_linear_api() const
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


string ModelExpression::get_expression_api() const
{

    ostringstream buffer;
    vector<string> found_tokens;

    const vector<string> input_names =  neural_network->get_input_names();
    const vector<string> output_names = neural_network->get_output_names();
    const vector<string> fixes_input_names =  fix_input_names(input_names);
    const vector<string> fixes_output_names = fix_output_names(output_names);

    const Index inputs_number = neural_network->get_inputs_number();
    const Index outputs_number = neural_network->get_outputs_number();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;

    buffer << write_header_api();

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\n\t\t" << i << ")  " << input_names[i];

    buffer << write_subheader_api();

    string line;
    string expression = neural_network->get_expression();

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
               << "$" << fixes_input_names[i]      << " = floatval(" << "$num"  + to_string(i) << ");"  << endl;

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
    string target_string1("RectifiedLinear");
    string target_string4("ExponentialLinear");
    string target_string5("SELU");

    size_t substring_length0;
    size_t substring_length1;
    size_t substring_length4;
    size_t substring_length5;

    string new_word;

    vector<string> found_tokens_and_input_names = concatenate_string_vectors(input_names, found_tokens);
    sort_string_vector(found_tokens_and_input_names);
    vector<string> fixes_tokens = fix_input_names(found_tokens_and_input_names);

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

        for(size_t j = 0; j < found_tokens_and_input_names.size(); j++)
        {
            new_word = "$" + fixes_tokens[j];
            replace_all_word_appearances(t, found_tokens_and_input_names[j], new_word);
        }

        buffer << t << endl;
    }

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, output_names, ProgrammingLanguage::PHP);

    for(size_t i = 0; i < fixed_outputs.size(); i++)
        buffer << fixed_outputs[i] << endl;

    buffer << "if($status === 200){" << endl
           << "$response = ['status' => $status,  'status_message' => $status_msg" << endl;

    for(size_t i = 0; i < outputs_number; i++)
        buffer << ", '" << fixes_output_names[i] << "' => " << "$" << fixes_output_names[i] << endl;

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

    buffer << "<?php" << endl
           << "function Linear(float $x) {" << endl
           << "return $x;" << endl
           << "}" << endl
           << "?>\n" << endl;

    if (logistic)
        buffer << logistic_api();

    if(ReLU)
        buffer << "<?php" << endl
               << "function RectifiedLinear(float $x) {" << endl
               << "$z = max(0, $x);" << endl
               << "return $z;" << endl
               << "}" << endl
               << "?>\n" << endl;

    if(ExpLinear)
        buffer << "<?php" << endl
               << "function ExponentialLinear(float $x) {" << endl
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
               << "function SELU(float $x) {" << endl
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


// string ModelExpression::autoassociaton_javascript() const
// {
//     string expression;

//     size_t index = 0;

//     index = expression.find("sample_autoassociation_distance =");

//     if (index != string::npos)
//         expression.erase(index, string::npos);

//     index = expression.find("sample_autoassociation_variables_distance =");

//     if (index != string::npos)
//         expression.erase(index, string::npos);

//     return expression;
// }


string ModelExpression::logistic_javascript() const
{
    return
        "function Logistic(x) {\n"
        "\tvar z = 1/(1+Math.exp(x));\n"
        "\treturn z;\n"
        "}\n";
}

string ModelExpression::relu_javascript() const
{
    return
        "function RectifiedLinear(x) {\n"
        "\tvar z = Math.max(0, x);\n"
        "\treturn z;\n"
        "}\n";
}


string ModelExpression::exponential_linear_javascript() const
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

string ModelExpression::selu_javascript() const
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


string ModelExpression::header_javascript() const
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

string ModelExpression::subheader_javascript() const
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


string ModelExpression::get_expression_javascript(const vector<Dataset::RawVariable>& raw_variables) const
{
    vector<string> lines;
    vector<string> found_tokens;
    vector<string> found_mathematical_expressions;

    vector<string> input_names;
    vector<string> output_names;

    for (const auto& raw_variable : raw_variables)
        if(raw_variable.use == "Input")
            input_names.push_back(raw_variable.name);
        else if(raw_variable.use == "Target")
            output_names.push_back(raw_variable.name);


    const Index inputs_number = input_names.size();
    const Index outputs_number = output_names.size();

    vector<string> fixes_input_names = fix_input_names(input_names);
    vector<string> fixes_output_names = fix_output_names(output_names);

    string token;
    string expression = neural_network->get_expression();

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

    if(neural_network->has("Scaling2d") || neural_network->has("Scaling4d"))
    {
        const vector<Descriptives> inputs_descriptives = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"))->get_descriptives();

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

    buffer << "function Linear(x) { return x; }\n";

    if (expression.find("Logistic") != string::npos) buffer << logistic_javascript();
    if (expression.find("RectifiedLinear") != string::npos) buffer << relu_javascript();
    if (expression.find("ExponentialLinear") != string::npos) buffer << exponential_linear_javascript();
    if (expression.find("SELU") != string::npos) buffer << selu_javascript();
    buffer << "\n";

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
    string target_string_1("RectifiedLinear");
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
           << "</script>\n" << endl
           << "<!--script source=\"https://www.neuraldesigner.com/app/htmlparts/footer.js\"></script-->\n" << endl
           << "</body>\n" << endl
           << "</html>" << endl;

    return buffer.str();
}

string ModelExpression::write_header_python() const
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

string ModelExpression::write_subheader_python() const
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


string ModelExpression::get_expression_python() const
{
    ostringstream buffer;

    vector<string> found_tokens;
    vector<string> found_mathematical_expressions;

    vector<string> original_inputs = neural_network->get_input_names();
    vector<string> inputs = fix_input_names(original_inputs);
    vector<string> original_outputs = neural_network->get_output_names();
    vector<string> outputs = fix_input_names(original_outputs);

    const Index inputs_number = inputs.size();
    const Index outputs_number = outputs.size();

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;

    buffer << write_header_python();

    for(Index i = 0; i < inputs_number; i++)
        buffer << "\t" << i << ") " << inputs[i] << endl;

    buffer << write_subheader_python();

    vector<string> lines;

    string expression = neural_network->get_expression();
    string line;

    stringstream string_stream(expression);

    while(getline(string_stream, line, '\n'))
    {
        if(line.size() > 1 && line.back() == '{')
            break;

        if(line.size() > 1 && line.back() != ';')
            line += ';';

        for(size_t i = 0; i < inputs_number; i++)
        {
            replace_all_appearances(line, "scaled_" + original_inputs[i], "scaled_" + inputs[i]);
            replace_all_appearances(line, original_inputs[i], inputs[i]);
        }

        lines.push_back(line);
    }

    const string target_string0("Logistic");
    const string target_string1("RectifiedLinear");
    const string target_string4("ExponentialLinear");
    const string target_string5("SELU");

    const Index lines_number = lines.size();

    for(size_t i = 0; i < lines_number; i++)
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

    buffer << "import numpy as np" << endl
           << "import pandas as pd\n" << endl
           << "class NeuralNetwork:\n" << endl;

    string inputs_list;

    for(size_t i = 0; i < inputs_number; i++)
    {
        inputs_list += "'" + inputs[i] + "'";

        if(i < inputs_number - 1)
            inputs_list += ", ";
    }

    buffer << "\tdef __init__(self):" << endl
           << "\t\t" << "self.inputs_number = " << to_string(inputs_number) << endl
           << "\t\t" << "self.input_names = [" << inputs_list << "]\n" << endl;

    buffer << "\t@staticmethod" << endl
           << "\tdef Linear(x):" << endl
           << "\t\t" << "return x\n" << endl;

    if(logistic)
        buffer << "\t@staticmethod" << endl
               << "\tdef Logistic (x):" << endl
               << "\t\t" << "z = 1/(1+np.exp(-x))" << endl
               << "\t\t" << "return z\n" << endl;

    if(ReLU)
        buffer << "\t@staticmethod" << endl
               << "\tdef RectifiedLinear (x):" << endl
               << "\t\t" << "z = max(0, x)" << endl
               << "\t\t" << "return z\n" << endl;

    if(ExpLinear)
        buffer << "\t@staticmethod" << endl
               << "\tdef ExponentialLinear (x):" << endl
               << "\t\t"   << "alpha = 1.67326" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "return x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "return alpha*(np.exp(x)-1)\n" << endl;

    if(SExpLinear)
        buffer << "\t@staticmethod" << endl
               << "\tdef SELU (x):" << endl
               << "\t\t"   << "alpha = 1.67326" << endl
               << "\t\t"   << "lambda_val = 1.05070" << endl
               << "\t\t"   << "if(x>0):" << endl
               << "\t\t\t" << "return lambda_val*x" << endl
               << "\t\t"   << "else:" << endl
               << "\t\t\t" << "return lambda_val*alpha*(np.exp(x)-1)\n" << endl;

    buffer << "\t" << "def calculate_outputs(self, inputs):" << endl;

    for(size_t i = 0; i < inputs_number; i++)
        buffer << "\t\t" << inputs[i] << " = " << "inputs[" << to_string(i) << "]" << endl;

    buffer << endl;

    found_tokens.resize(0);
    found_tokens.push_back("log");
    found_tokens.push_back("exp");
    found_tokens.push_back("tanh");

    found_mathematical_expressions.push_back("Logistic");
    found_mathematical_expressions.push_back("RectifiedLinear");
    found_mathematical_expressions.push_back("ExponentialLinear");
    found_mathematical_expressions.push_back("SELU");
    found_mathematical_expressions.push_back(" Linear");

    string sufix;
    string new_word;
    string key_word ;

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

        sufix = "self.";

        for(int j = 0; j < found_mathematical_expressions.size(); j++)
        {
            key_word = found_mathematical_expressions[j];
            if(key_word == " Linear")
                new_word = " " + sufix + "Linear";
            else
                new_word = sufix + key_word;
            replace_all_appearances(line, key_word, new_word);
        }

        buffer << "\t\t" << line << endl;
    }

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, outputs, ProgrammingLanguage::Python);

    for (const string& assignment : fixed_outputs)
        buffer << "\t\t" << assignment << endl;

    string return_list = "[";
    for(size_t i = 0; i < outputs.size(); ++i)
    {
        return_list += outputs[i];
        if (i < outputs.size() - 1)
        {
            return_list += ", ";
        }
    }
    return_list += "]";
    buffer << "\t\treturn " << return_list << "\n" << endl;

    buffer << "\tdef calculate_batch_output(self, input_batch):" << endl
           << "\t\t" << "output_batch = np.zeros((len(input_batch), " << to_string(outputs_number) << "))" << endl
           << "\t\t" << "for i in range(len(input_batch)):" << endl
           << "\t\t\t" << "inputs = list(input_batch[i])" << endl
           << "\t\t\t" << "output = self.calculate_outputs(inputs)" << endl
           << "\t\t\t" << "output_batch[i] = output" << endl
           << "\t\t" << "return output_batch\n" << endl;

    buffer << "def main():" << endl
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


string ModelExpression::replace_reserved_keywords(const string& s) const
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
                                                           const ProgrammingLanguage& programming_Language) const
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

            vector<string> fixes_outputs = fix_output_names(outputs);
            for(size_t i = 0; i < outputs.size(); i++)
                replace_all_appearances(out_string, outputs[i], fixes_outputs[i]);
            out.push_back(out_string);
        }
    }
    return out;
}


vector<string> ModelExpression::fix_input_names(const vector<string>& input_names) const
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


vector<string> ModelExpression::fix_output_names(const vector<string>& output_names) const
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


void ModelExpression::save_python(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    file << get_expression_python();
}


void ModelExpression::save_c(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    file << get_expression_c();
}


void ModelExpression::save_javascript(const filesystem::path& file_name, const vector<Dataset::RawVariable>& raw_variables) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    file << get_expression_javascript(raw_variables);
}


void ModelExpression::save_api(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;


    file << get_expression_api();
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
