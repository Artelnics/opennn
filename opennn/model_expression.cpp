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

string ModelExpression::get_expression_c(const vector<Dataset::RawVariable>& raw_variables) const
{
    ostringstream buffer;

    vector<string> input_names;
    vector<string> output_names;

    for(const Dataset::RawVariable& raw_variable : raw_variables)
        if(raw_variable.use == "Input")
            input_names.push_back(raw_variable.name);
        else if(raw_variable.use == "Target")
            output_names.push_back(raw_variable.name);

    vector<string> fixed_output_names = fix_output_names(output_names);
    const Index outputs_number = output_names.size();

    bool logistic = false;
    bool relu = false;
    bool exp_linear = false;
    bool selu = false;
    bool tanh = false;

    buffer << write_comments_c();

    for(size_t i = 0; i < input_names.size(); i++)
        buffer << "\n// \t " << i << ")  " << input_names[i];

    buffer << "\n \n \n#include <stdio.h>\n"
              "#include <stdlib.h>\n"
              "#include <math.h>\n\n";

    string expression = neural_network->get_expression();

    for (const Dataset::RawVariable& raw_var : raw_variables)
    {
        if (raw_var.use != "Input")
            continue;

        if (raw_var.type == Dataset::RawVariableType::Binary || raw_var.type == Dataset::RawVariableType::Categorical)
        {
            vector<string> names_to_process;

            if (raw_var.type == Dataset::RawVariableType::Binary)
                names_to_process.push_back(raw_var.name);
            else
                names_to_process = raw_var.categories;

            sort(names_to_process.begin(), names_to_process.end(), [](const string& a, const string& b) { return a.length() > b.length(); });

            for (const string& original_name : names_to_process)
            {
                string scaled_name = "scaled_" + original_name;
                string search_pattern = scaled_name + " = ";
                size_t start_pos = expression.find(search_pattern);

                if (start_pos != string::npos)
                {
                    size_t end_pos = expression.find('\n', start_pos);

                    if (end_pos != string::npos)
                        expression.erase(start_pos, end_pos - start_pos + 1);
                    else
                        expression.erase(start_pos);
                }

                replace_all_appearances(expression, scaled_name, original_name);
            }
        }
    }

    stringstream string_stream(expression);
    string ss_line;

    while(getline(string_stream, ss_line, '\n'))
    {
        if (ss_line.find("=") != string::npos)
        {
            string var_name = get_first_word(ss_line);

            if (!var_name.empty())
                replace_all_word_appearances(expression, var_name, replace_reserved_keywords(var_name));
        }
    }

    for (const Dataset::RawVariable& raw_var : raw_variables)
    {
        if (raw_var.use == "Input")
        {
            if (raw_var.type == Dataset::RawVariableType::Categorical)
                for (const string& cat : raw_var.categories)
                    replace_all_word_appearances(expression, cat, replace_reserved_keywords(cat));
            else
                replace_all_word_appearances(expression, raw_var.name, replace_reserved_keywords(raw_var.name));
        }
    }

    for (const string& name : output_names)
        replace_all_word_appearances(expression, name, replace_reserved_keywords(name));

    stringstream ss(expression);
    string line;
    vector<string> lines;

    while (getline(ss, line, '\n'))
    {
        if (line.empty() || all_of(line.begin(), line.end(), [](char c){ return isspace(c); }))
            continue;

        if (line.find("{") != string::npos)
            break;

        if (line.back() != ';')
            line += ';';

        lines.push_back(line);
    }

    if(expression.find("Logistic") != string::npos) logistic = true;
    if(expression.find("RectifiedLinear") != string::npos) relu = true;
    if(expression.find("ExponentialLinear") != string::npos) exp_linear = true;
    if(expression.find("SELU") != string::npos) selu = true;
    if(expression.find("HyperbolicTangent") != string::npos) tanh = true;

    buffer << "float Linear (float x) {\n\treturn x;\n}\n\n";
    if(logistic) buffer << write_logistic_c();
    if(relu) buffer << write_relu_c();
    if(exp_linear) buffer << write_exponential_linear_c();
    if(selu) buffer << write_selu_c();
    if(tanh) buffer << "float HyperbolicTangent(float x) {\n\treturn tanhf(x);\n}\n\n";

    buffer << "float* calculate_outputs(const float* inputs)\n{\n";

    Index input_index = 0;

    for(const Dataset::RawVariable& raw_var : raw_variables)
    {
        if (raw_var.use != "Input")
            continue;

        if (raw_var.type == Dataset::RawVariableType::Categorical)
            for (const string& category : raw_var.categories)
                buffer << "\tconst float " << replace_reserved_keywords(category) << " = inputs[" << input_index++ << "];\n";
        else
            buffer << "\tconst float " << replace_reserved_keywords(raw_var.name) << " = inputs[" << input_index++ << "];\n";
    }

    buffer << "\n";

    for(const string& l : lines)
        buffer << "\tdouble " << l << "\n";

    const vector<string> fixed_outputs = fix_get_expression_outputs(expression, output_names, ProgrammingLanguage::C);

    if (!fixed_outputs.empty())
    {
        buffer << "\n";

        for(const string& l : fixed_outputs)
            buffer << "\t" << l << "\n";
    }

    buffer << "\n";
    buffer << "\tfloat* out = (float*)malloc(" << outputs_number << " * sizeof(float));\n";
    buffer << "\tif (out == NULL) {\n";
    buffer << "\t\tprintf(\"Error: Memory allocation failed in calculate_outputs.\\n\");\n";
    buffer << "\t\treturn NULL;\n";
    buffer << "\t}\n\n";

    for(Index i = 0; i < outputs_number; i++)
        buffer << "\tout[" << i << "] = " << fixed_output_names[i] << ";\n";

    buffer << "\n\treturn out;\n}\n\n";

    buffer << "int main() { \n\n";
    buffer << "\tfloat* inputs = (float*)malloc(" << input_index << " * sizeof(float)); \n";
    buffer << "\tif (inputs == NULL) {\n";
    buffer << "\t\tprintf(\"Error: Memory allocation failed for inputs.\\n\");\n";
    buffer << "\t\treturn 1;\n";
    buffer << "\t}\n\n";

    buffer << "\t// Please enter your values here:\n";

    input_index = 0;
    for(const Dataset::RawVariable& raw_var : raw_variables)
    {
        if (raw_var.use != "Input")
            continue;

        if (raw_var.type == Dataset::RawVariableType::Categorical)
        {
            buffer << "\t// One-hot encoding for " << raw_var.name << ": set one to 1.0f and the rest to 0.0f\n";

            for (const string& category : raw_var.categories)
                buffer << "\tinputs[" << input_index++ << "] = 0.0f; // " << replace_reserved_keywords(category) << "\n";
        }
        else
            buffer << "\tinputs[" << input_index++ << "] = 0.0f; // " << replace_reserved_keywords(raw_var.name) << "\n";
    }

    buffer << "\n\tfloat* outputs;\n";
    buffer << "\n\toutputs = calculate_outputs(inputs);\n\n";
    buffer << "\tif (outputs != NULL) {\n";
    buffer << "\t\tprintf(\"These are your outputs:\\n\");\n";

    for(Index i = 0; i < outputs_number; i++)
        buffer << "\t\tprintf(\""<< fixed_output_names[i] << ": %f \\n\", outputs[" << i << "]);\n";

    buffer << "\t}\n\n";
    buffer << "\t// Free the allocated memory\n";
    buffer << "\tfree(inputs);\n";
    buffer << "\tfree(outputs);\n\n";
    buffer << "\treturn 0;\n} \n";

    return buffer.str();
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


string ModelExpression::get_expression_api(const vector<Dataset::RawVariable>& raw_variables) const
{
    ostringstream buffer;

    vector<string> original_inputs;
    vector<string> output_names = neural_network->get_output_names();

    for(const Dataset::RawVariable& var : raw_variables)
        if(var.use == "Input")
            original_inputs.push_back(var.name);

    const vector<string> fixed_input_names = fix_input_names(original_inputs);
    const vector<string> fixed_output_names = fix_output_names(output_names);
    const Index inputs_number = original_inputs.size();
    const Index outputs_number = output_names.size();

    buffer << write_header_api();
    for(Index i = 0; i < inputs_number; i++)
        buffer << "\n\t\t" << i << ")  " << original_inputs[i];
    buffer << write_subheader_api();

    string expression = neural_network->get_expression();

    if(expression.find("Linear") != string::npos)
        buffer << "function Linear($x) { return $x; }\n";
    if(expression.find("Logistic") != string::npos)
        buffer << "function Logistic($x) { return 1 / (1 + exp(-$x)); }\n";
    if(expression.find("RectifiedLinear") != string::npos)
        buffer << "function RectifiedLinear($x) { return max(0, $x); }\n";
    if(expression.find("HyperbolicTangent") != string::npos)
        buffer << "function HyperbolicTangent($x) { return tanh($x); }\n";
    if(expression.find("ExponentialLinear") != string::npos)
        buffer << "function ExponentialLinear($x) { $alpha = 1.67326; return ($x > 0) ? $x : $alpha * (exp($x) - 1); }\n";
    if(expression.find("SELU") != string::npos)
        buffer << "function SELU($x) { $alpha = 1.67326; $lambda = 1.05070; return $lambda * (($x > 0) ? $x : $alpha * (exp($x) - 1)); }\n";

    buffer << "\nsession_start();\n";
    buffer << "if(isset($_GET['num0'])) { \n";
    buffer << "$params = $_GET;\n\n";

    for(Index i = 0; i < inputs_number; i++)
        buffer << "$" << fixed_input_names[i] << " = isset($params['num" << i << "']) ? floatval($params['num" << i << "']) : 0;\n";

    buffer << "\n// One-hot encoding conversion (do not modify)\n";

    map<string, Dataset::RawVariable> raw_vars_map;

    for(const Dataset::RawVariable& var : raw_variables)
        raw_vars_map[var.name] = var;

    for(Index i = 0; i < inputs_number; ++i)
    {
        const Dataset::RawVariable& var = raw_vars_map[original_inputs[i]];

        if(var.type == Dataset::RawVariableType::Categorical)
            for(size_t j = 0; j < var.categories.size(); ++j)
                buffer << "$" << replace_reserved_keywords(var.categories[j]) << " = ($" << fixed_input_names[i] << " == " << j << ") ? 1 : 0;\n";
    }

    buffer << "\n";

    for(const Dataset::RawVariable& raw_var : raw_variables)
    {
        if(raw_var.use != "Input")
            continue;

        if(raw_var.type == Dataset::RawVariableType::Binary || raw_var.type == Dataset::RawVariableType::Categorical)
        {
            vector<string> names_to_process = (raw_var.type == Dataset::RawVariableType::Binary) ? vector<string>{raw_var.name} : raw_var.categories;
            sort(names_to_process.begin(), names_to_process.end(), [](const string& a, const string& b) { return a.length() > b.length(); });

            for(const string& original_name : names_to_process)
            {
                string scaled_name = "scaled_" + original_name;
                string search_pattern = scaled_name + " = ";
                size_t start_pos = expression.find(search_pattern);

                if(start_pos != string::npos)
                {
                    size_t end_pos = expression.find('\n', start_pos);
                    expression.erase(start_pos, (end_pos != string::npos) ? (end_pos - start_pos + 1) : string::npos);
                }

                replace_all_appearances(expression, scaled_name, original_name);
            }
        }
    }

    vector<string> all_possible_vars;
    stringstream temp_ss(expression);
    string temp_line;

    while(getline(temp_ss, temp_line, '\n'))
    {
        if(temp_line.find("=") != string::npos)
        {
            string var_name = get_first_word(temp_line);

            if(!var_name.empty())
                all_possible_vars.push_back(var_name);
        }
    }

    for(const Dataset::RawVariable& var : raw_variables)
    {
        if(var.use == "Input")
        {
            if(var.type == Dataset::RawVariableType::Categorical)
                for (const string& cat : var.categories) all_possible_vars.push_back(cat);
            else
                all_possible_vars.push_back(var.name);
        }
    }

    for(const string& name : output_names) all_possible_vars.push_back(name);

    sort(all_possible_vars.begin(), all_possible_vars.end());
    all_possible_vars.erase(unique(all_possible_vars.begin(), all_possible_vars.end()), all_possible_vars.end());
    sort(all_possible_vars.begin(), all_possible_vars.end(), [](const string& a, const string& b){ return a.length() > b.length(); });

    for(const string& var_name : all_possible_vars)
        replace_all_word_appearances(expression, var_name, "$" + replace_reserved_keywords(var_name));

    stringstream ss(expression);
    string line;
    while(getline(ss, line, '\n'))
    {
        if(line.empty() || all_of(line.begin(), line.end(), [](char c){ return isspace(c); }))
            continue;

        if(line.find("{") != string::npos)
            break;

        if(line.back() != ';')
            line += ';';

        buffer << line << "\n";
    }

    buffer << "\nif(true){ // Simplified status check\n";
    buffer << "$response = ['status' => 200,  'status_message' => 'ok'";

    for(Index i = 0; i < outputs_number; i++)
        buffer << ", '" << output_names[i] << "' => $" << fixed_output_names[i];

    buffer << "];\n} else {\n";
    buffer << "$response = ['status' => 400,  'status_message' => 'invalid parameters'];\n}\n\n";

    buffer << "$json_response_pretty = json_encode($response, JSON_PRETTY_PRINT);\n";
    buffer << "echo nl2br(\"\\n\" . $json_response_pretty . \"\\n\");\n";
    buffer << "} else {\n echo \"Please provide input values in the URL (e.g., ?num0=value0&num1=value1...)\"; \n}\n";
    buffer << "$_SESSION['lastpage'] = __FILE__;\n";
    buffer << "?>\n</h4>\n</div>\n</body>\n</html>";

    return buffer.str();
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
        "\tvar z = 1/(1+Math.exp(-x));\n"
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


string ModelExpression::hyperbolic_tangent_javascript() const
{
    return
        "function HyperbolicTangent(x) {\n"
        "\treturn Math.tanh(x);\n"
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
        "-->\n\n"
        "<!DOCTYPE HTML>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "<link href=\"https://www.neuraldesigner.com/assets/css/neuraldesigner.css\" rel=\"stylesheet\" />\n"
        "<link href=\"https://www.neuraldesigner.com/images/fav.ico\" rel=\"shortcut icon\" type=\"image/x-icon\" />\n"
        "</head>\n\n"
        "<style>\n"
        "body {\n"
        "display: flex;\n"
        "justify-content: center;\n"
        "align-items: center;\n"
        "min-height: 100vh;\n"
        "margin: 0;\n"
        "padding: 2em 0;\n"
        "background-color: #f0f0f0;\n"
        "font-family: Arial, sans-serif;\n"
        "box-sizing: border-box;\n"
        "}\n\n"
        ".content-wrapper {\n"
        "width: 100%;\n"
        "max-width: 800px;\n"
        "margin: 0 auto;\n"
        "padding: 20px;\n"
        "background-color: #fff;\n"
        "box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);\n"
        "border: 1px solid #777;\n"
        "border-radius: 5px;\n"
        "text-align: center;\n"
        "}\n\n"
        ".form-table {\n"
        "border-collapse: collapse;\n"
        "width: 100%;\n"
        "margin-bottom: 20px;\n"
        "border: 1px solid #808080;\n"
        "}\n\n"
        ".form-table th,\n"
        ".form-table td {\n"
        "padding: 10px;\n"
        "text-align: left;\n"
        "vertical-align: middle;\n"
        "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;\n"
        "}\n\n"
        ".form-table tr:not(:last-child) {\n"
        "border-bottom: 1px solid #808080;\n"
        "}\n\n"
        ".neural-cell {\n"
        "text-align: right;\n"
        "width: 50%;\n"
        "}\n\n"
        ".neural-cell input[type=\"range\"] {\n"
        "display: block;\n"
        "margin-left: auto;\n"
        "margin-right: 0;\n"
        "box-sizing: border-box;\n"
        "max-width: 200px;\n"
        "width: 90%;\n"
        "}\n\n"
        ".neural-cell input[type=\"number\"],\n"
        ".neural-cell input[type=\"text\"],\n"
        ".neural-cell select {\n"
        "display: block;\n"
        "margin-left: auto;\n"
        "margin-right: 0;\n"
        "box-sizing: border-box;\n"
        "max-width: 200px;\n"
        "width: 90%;\n"
        "padding: 5px;\n"
        "text-align: right;\n"
        "text-align-last: right;\n"
        "}\n\n"
        ".neural-cell input[type=\"number\"] {\n"
        "margin-top: 8px;\n"
        "}\n\n"
        ".btn {\n"
        "background-color: #5da9e9;\n"
        "border: none;\n"
        "color: white;\n"
        "text-align: center;\n"
        "font-size: 16px;\n"
        "margin-top: 10px;\n"
        "margin-bottom: 20px;\n"
        "cursor: pointer;\n"
        "padding: 10px 20px;\n"
        "border-radius: 5px;\n"
        "transition: background-color 0.3s ease;\n"
        "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;\n"
        "}\n\n"
        ".btn:hover {\n"
        "background-color: #4b92d3;\n"
        "}\n\n"
        "input[type=\"range\"]::-webkit-slider-runnable-track {\n"
        "background: #8fc4f0;\n"
        "height: 0.5rem;\n"
        "}\n\n"
        "input[type=\"range\"]::-moz-range-track {\n"
        "background: #8fc4f0;\n"
        "height: 0.5rem;\n"
        "}\n\n"
        "input[type=\"range\"]::-webkit-slider-thumb {\n"
        "-webkit-appearance: none;\n"
        "appearance: none;\n"
        "margin-top: -5px;\n"
        "background-color: #5da9e9;\n"
        "border-radius: 50%;\n"
        "height: 20px;\n"
        "width: 20px;\n"
        "border: 2px solid #000000;\n"
        "box-shadow: 0 0 5px rgba(0,0,0,0.25);\n"
        "cursor: pointer;\n"
        "}\n\n"
        "input[type=\"range\"]::-moz-range-thumb {\n"
        "background-color: #5da9e9;\n"
        "border-radius: 50%;\n"
        "margin-top: -5px;\n"
        "height: 20px;\n"
        "width: 20px;\n"
        "border: 2px solid #000000;\n"
        "box-shadow: 0 0 5px rgba(0,0,0,0.25);\n"
        "cursor: pointer;\n"
        "}\n\n"
        ".form-table th {\n"
        "background-color: #f2f2f2;\n"
        "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;\n"
        "}\n\n"
        "h4 {\n"
        "font-family: \"Helvetica Neue\", Helvetica, Arial, sans-serif;\n"
        "color: #333;\n"
        "}\n"
        "</style>\n\n"
        "<body>\n\n"
        "<section>\n"
        "<div class=\"content-wrapper\">\n"
        "<form onsubmit=\"neuralNetwork(); return false;\">\n"
        "<h4>INPUTS</h4>\n"
        "<table class=\"form-table\">\n";
}


string ModelExpression::get_expression_javascript(const vector<Dataset::RawVariable>& raw_variables) const
{
    vector<string> lines;
    vector<string> found_tokens;
    vector<string> found_mathematical_expressions;

    vector<string> input_names;
    vector<string> output_names;

    for(const Dataset::RawVariable& raw_variable : raw_variables)
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

    for (const Dataset::RawVariable& raw_var : raw_variables)
    {
        if(raw_var.use != "Input")
            continue;

        if(raw_var.type == Dataset::RawVariableType::Binary || raw_var.type == Dataset::RawVariableType::Categorical)
        {
            vector<string> names_to_process;

            if (raw_var.type == Dataset::RawVariableType::Binary)
                names_to_process.push_back(raw_var.name);
            else
                names_to_process = raw_var.categories;

            sort(names_to_process.begin(), names_to_process.end(),
                [](const string& a, const string& b) {
                  return a.length() > b.length();
                });

            for (const string& original_name : names_to_process)
            {
                string scaled_name = "scaled_" + original_name;

                string search_pattern = scaled_name + " = ";
                size_t start_pos = expression.find(search_pattern);

                if (start_pos != string::npos)
                {
                    size_t end_pos = expression.find('\n', start_pos);

                    if (end_pos != string::npos)
                        expression.erase(start_pos, end_pos - start_pos + 1);
                    else
                        expression.erase(start_pos);
                }

                replace_all_appearances(expression, scaled_name, original_name);
            }
        }
    }

    for(int i = 0 ; i < outputs_number; i++)
        replace_all_word_appearances(expression, output_names[i], fixes_output_names[i]);

    const int maximum_output_variable_numbers = 5;

    stringstream ss(expression);

    bool logistic     = false;
    bool ReLU         = false;
    bool ExpLinear    = false;
    bool SExpLinear   = false;

    ostringstream buffer;

    buffer << header_javascript();

    for(Index i = 0; i < inputs_number; i++)
        buffer << "\n\t " << i + 1 << ")  " << input_names[i];

    buffer << subheader_javascript();

    if(neural_network->has("Scaling2d") || neural_network->has("Scaling4d"))
    {
        const vector<Descriptives> inputs_descriptives = static_cast<Scaling2d*>(neural_network->get_first("Scaling2d"))->get_descriptives();

        float min_value;
        float max_value;

        Index input = 0;
        Index categories = 0;
        Index inputs_processed = 0;

        for(size_t k = 0; k < raw_variables.size() && inputs_processed < inputs_number; ++k)
        {
            if(raw_variables[k].use != "Input")
                continue;

            const vector<string> raw_variable_categories = raw_variables[k].categories;

            if(raw_variables[k].type == Dataset::RawVariableType::Numeric) // INPUT & NUMERIC
            {
                min_value = inputs_descriptives[input].minimum;
                max_value = inputs_descriptives[input].maximum;

                buffer << "<!-- "<< to_string(input) <<"scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;
                buffer << "<td> " << input_names[inputs_processed] << " </td>" << endl;
                buffer << "<td class=\"neural-cell\">" << endl;

                if(min_value==0 && max_value==0)
                {
                    buffer << "<input type=\"range\" id=\"" << fixes_input_names[inputs_processed] << "\" value=\"" << min_value << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"" << (max_value - min_value)/100 << "\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[inputs_processed] << "_text')\" />" << endl;
                    buffer << "<input class=\"tabla\" type=\"number\" id=\"" << fixes_input_names[inputs_processed] << "_text\" value=\"" << min_value << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"any\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[inputs_processed] << "')\">" << endl;
                }
                else
                {
                    buffer << "<input type=\"range\" id=\"" << fixes_input_names[inputs_processed] << "\" value=\"" << (min_value + max_value)/2 << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"" << (max_value - min_value)/100 << "\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[inputs_processed] << "_text')\" />" << endl;
                    buffer << "<input class=\"tabla\" type=\"number\" id=\"" << fixes_input_names[inputs_processed] << "_text\" value=\"" << (min_value + max_value)/2 << "\" min=\"" << min_value << "\" max=\"" << max_value << "\" step=\"any\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[inputs_processed] << "')\">" << endl;
                }

                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                input += 1;
            }
            else if(raw_variables[k].type == Dataset::RawVariableType::Binary && raw_variable_categories.size() == 2 &&
                    ((raw_variable_categories[0]=="1" && raw_variable_categories[1]=="0") || (raw_variable_categories[1]=="1" && raw_variable_categories[0]=="0"))) // INPUT & BINARY (1,0)
            {
                buffer << "<!-- ComboBox Ultima pasada-->" << endl;
                buffer << "<!-- 5scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;

                buffer << "<td> " << input_names[inputs_processed] << " </td>" << endl;

                buffer << "<td class=\"neural-cell\">" << endl;
                buffer << "<select id=\"Select" << categories << "\">" << endl;

                buffer << "<option value=\"" << 0 << "\">" << 0 << "</option>" << endl;
                buffer << "<option value=\"" << 1 << "\">" << 1 << "</option>" << endl;

                buffer << "</select>" << endl;
                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                categories += 1;
                input += 1;
            }
            else if(raw_variables[k].type == Dataset::RawVariableType::Binary && raw_variable_categories.size() == 2) // INPUT & BINARY (A,B)
            {
                buffer << "<!-- ComboBox Ultima pasada-->" << endl;
                buffer << "<!-- 5scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;

                buffer << "<td> " << input_names[inputs_processed] << " </td>" << endl;

                buffer << "<td class=\"neural-cell\">" << endl;
                buffer << "<select id=\"Select" << categories << "\">" << endl;

                buffer << "<option value=\"" << 0 << "\">" << raw_variable_categories[0] << "</option>" << endl;
                buffer << "<option value=\"" << 1 << "\">" << raw_variable_categories[1] << "</option>" << endl;

                buffer << "</select>" << endl;
                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                categories += 1;
                input += 1;
            }
            else if(raw_variables[k].type == Dataset::RawVariableType::Categorical) // INPUT & CATEGORICAL
            {
                buffer << "<!-- ComboBox Ultima pasada-->" << endl;
                buffer << "<!-- 5scaling layer -->" << endl;
                buffer << "<tr style=\"height:3.5em\">" << endl;

                buffer << "<td> " << input_names[inputs_processed] << " </td>" << endl;

                buffer << "<td class=\"neural-cell\">" << endl;
                buffer << "<select id=\"Select" << categories << "\">" << endl;

                for (size_t l = 0; l < raw_variable_categories.size(); ++l)
                {
                    buffer << "<option value=\"" << l << "\">" << raw_variable_categories[l] << "</option>" << endl;
                }

                buffer << "</select>" << endl;
                buffer << "</td>" << endl;
                buffer << "</tr>" << endl;
                buffer << "\n" << endl;

                categories += 1;
                input += raw_variable_categories.size();
            }

            inputs_processed++;
        }
    }
    else
        for(Index i = 0; i < inputs_number; i++)
            buffer << "<!-- "<< to_string(i) <<"no scaling layer -->" << endl
                   << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << input_names[i] << " </td>" << endl
                   << "<td class=\"neural-cell\">" << endl
                   << "<input type=\"range\" id=\"" << fixes_input_names[i] << "\" value=\"0\" min=\"-1\" max=\"1\" step=\"0.01\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[i] << "_text')\" />" << endl
                   << "<input type=\"number\" id=\"" << fixes_input_names[i] << "_text\" value=\"0\" min=\"-1\" max=\"1\" step=\"any\" onchange=\"updateTextInput1(this.value, '" << fixes_input_names[i] << "')\">" << endl
                   << "</td>" << endl
                   << "</tr>\n" << endl;

    buffer << "</table>" << endl;

    if(outputs_number > maximum_output_variable_numbers)
    {
        buffer << "<!-- HIDDEN INPUTS -->" << endl;

        for(Index i = 0; i < outputs_number; i++)
            buffer << "<input type=\"hidden\" id=\"" << fixes_output_names[i] << "\" value=\"\">" << endl;

        buffer << "\n" << endl;
    }

    buffer << "<!-- BUTTON HERE -->" << endl
           << "<button class=\"btn\" onclick=\"neuralNetwork()\">calculate outputs</button>" << endl
           << "<br/>\n" << endl
           << "<table border=\"1px\" class=\"form-table\">" << endl
           << "<h4> OUTPUTS </h4>" << endl;

    if(outputs_number > maximum_output_variable_numbers)
    {
        buffer << "<tr style=\"height:3.5em\">" << endl
               << "<td> Target </td>" << endl
               << "<td class=\"neural-cell\">" << endl
               << "<select id=\"category_select\" onchange=\"updateSelectedCategory()\">" << endl;

        for(Index i = 0; i < outputs_number; i++)
            buffer << "<option value=\"" << output_names[i] << "\">" << output_names[i] << "</option>" << endl;

        buffer << "</select>" << endl
               << "</td>" << endl
               << "</tr>\n" << endl
               << "<tr style=\"height:3.5em\">" << endl
               << "<td> Value </td>" << endl
               << "<td class=\"neural-cell\">" << endl
               << "<input style=\"text-align:right; padding-right:20px;\" id=\"selected_value\" value=\"\" type=\"text\"  disabled/>" << endl
               << "</td>" << endl
               << "</tr>\n" << endl;
    }
    else
        for(Index i = 0; i < outputs_number; i++)
            buffer << "<tr style=\"height:3.5em\">" << endl
                   << "<td> " << output_names[i] << " </td>" << endl
                   << "<td class=\"neural-cell\">" << endl
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

        for(Index i = 0; i < outputs_number; i++)
            buffer << "\tif(selectedCategory === \"" << fixes_output_names[i] << "\") {" << endl
                   << "\t\tselectedValueElement.value = document.getElementById(\"" << fixes_output_names[i] << "\").value;" << endl
                   << "\t}" << endl;

        buffer << "}\n" << endl;
    }

    buffer << "\nfunction Linear(x) {\n"
              "\treturn x;\n"
              "}\n";

    if(expression.find("Logistic") != string::npos) buffer << logistic_javascript();
    if(expression.find("RectifiedLinear") != string::npos) buffer << relu_javascript();
    if(expression.find("ExponentialLinear") != string::npos) buffer << exponential_linear_javascript();
    if(expression.find("SELU") != string::npos) buffer << selu_javascript();
    if(expression.find("HyperbolicTangent") != string::npos) buffer << hyperbolic_tangent_javascript();
    buffer << "\n";

    buffer << "function neuralNetwork()" << endl
           << "{" << endl
           << "\t" << "var inputs = [];" << endl;

    Index j = 0;

    vector<string> variables_input_fixed;
    vector<string> variables_input;

    Index inputs_processed = 0;

    for(size_t k = 0; k < raw_variables.size() && inputs_processed < inputs_number; ++k)
    {
        if(raw_variables[k].use != "Input")
            continue;

        const vector<string> raw_variable_categories = raw_variables[k].categories;

        if(raw_variables[k].type == Dataset::RawVariableType::Numeric) // INPUT & NUMERIC
        {
            buffer << "\t" << "var " << fixes_input_names[inputs_processed] << " = (parseFloat(document.getElementById(\"" << fixes_input_names[inputs_processed] << "_text\").value) || 0); " << endl;
            buffer << "\t" << "inputs.push(" << fixes_input_names[inputs_processed] << ");" << endl;

            variables_input_fixed.push_back(fixes_input_names[inputs_processed]);
            variables_input.push_back(input_names[inputs_processed]);
        }
        else if(raw_variables[k].type == Dataset::RawVariableType::Binary)// INPUT & BINARY
        {
            string aux_buffer = "";

            buffer << "\t" << "var selectElement" << j << "= document.getElementById('Select" << j << "');" << endl;
            buffer << "\t" << "var selectedValue" << j << "= +selectElement" << j << ".value;" << endl;

            buffer << "\t" << "var " << fixes_input_names[inputs_processed] << "= 0;" << endl;
            aux_buffer = aux_buffer + "inputs.push(" + fixes_input_names[inputs_processed] + ");" + "\n";

            buffer << "switch (selectedValue" << j << "){" << endl;

            buffer << "\t" << "case " << 0 << ":";
            buffer << "\n" << "\t\t" << fixes_input_names[inputs_processed] << " = " << "0;" << endl;
            buffer << "\tbreak;" << endl;

            buffer << "case " << 1 << ":";
            buffer << "\n" << "\t\t" << fixes_input_names[inputs_processed] << " = " << "1;" << endl;
            buffer << "\tbreak;" << endl;

            buffer << "\t" << "default:" << endl;
            buffer << "\t" << "\tbreak;" << endl;
            buffer << "\t" << "}\n" << endl;

            buffer << aux_buffer << endl;

            j += 1;
            variables_input_fixed.push_back(fixes_input_names[inputs_processed]);
            variables_input.push_back(input_names[inputs_processed]);
        }
        else if(raw_variables[k].type == Dataset::RawVariableType::Categorical) // INPUT & CATEGORICAL
        {
            string aux_buffer = "";

            buffer << "\t" << "var selectElement" << j << "= document.getElementById('Select" << j << "');" << endl;
            buffer << "\t" << "var selectedValue" << j << "= +selectElement" << j << ".value;" << endl;

            for(size_t l = 0; l < raw_variable_categories.size(); ++l)
            {
                string category = raw_variable_categories[l];
                string fixed_category = replace_reserved_keywords(category);

                buffer << "\t" << "var " << fixed_category << "= 0;" << endl;
                aux_buffer = aux_buffer + "inputs.push(" + fixed_category + ");" + "\n";
                variables_input_fixed.push_back(fixed_category);
                variables_input.push_back(category);
            }

            buffer << "switch (selectedValue" << j << "){" << endl;

            for(size_t l = 0; l < raw_variable_categories.size(); ++l)
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

        inputs_processed++;
    }

    buffer << "\n" << "\t" << "var outputs = calculate_outputs(inputs); " << endl;

    if(outputs_number > maximum_output_variable_numbers)
        buffer << "\t" << "updateSelectedCategory();" << endl;
    else
        for(Index i = 0; i < outputs_number; i++)
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
    for(size_t i = 0; i < variables_input.size(); ++i)
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

        for(size_t j = 0; j < variables_input.size(); ++j)
        {
            if(line.find(variables_input[j]) != string::npos)
                replace_all_word_appearances(line, variables_input[j], variables_input_fixed[j]);

            if(line.find(variable_scaled[j]) != string::npos)
                replace_all_appearances(line, variable_scaled[j], replace_reserved_keywords(variable_scaled[j]));
        }

        for(size_t j = 0; j < found_mathematical_expressions.size(); j++)
        {
            string key_word = found_mathematical_expressions[j];
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

    for(Index i = 0; i < outputs_number; i++)
        buffer << "\t" << "out.push(" << fixes_output_names[i] << ");" << endl;

    buffer << "\n\t" << "return out;" << endl
           << "}\n" << endl;

    if(logistic) buffer << logistic_javascript();
    if(ReLU) buffer << relu_javascript();
    if(ExpLinear) buffer << exponential_linear_javascript();
    if(SExpLinear) buffer << "scaled_exponential_linear()";

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


string ModelExpression::get_expression_python(const vector<Dataset::RawVariable>& raw_variables) const
{
    ostringstream buffer;

    vector<string> original_inputs;

    for(const Dataset::RawVariable& var : raw_variables)
        if (var.use == "Input")
            original_inputs.push_back(var.name);

    vector<string> original_outputs = neural_network->get_output_names();
    vector<string> outputs = fix_output_names(original_outputs);

    const Index inputs_number = original_inputs.size();
    const Index outputs_number = outputs.size();

    bool logistic     = false;
    bool relu         = false;
    bool exp_linear    = false;
    bool selu   = false;
    bool hyper_tan     = false;

    buffer << write_header_python();

    for(Index i = 0; i < inputs_number; i++)
        buffer << "\t" << i << ") " << original_inputs[i] << endl;

    buffer << write_subheader_python();

    string expression = neural_network->get_expression();

    for(const Dataset::RawVariable& raw_var : raw_variables)
    {
        if(raw_var.use != "Input")
            continue;

        if(raw_var.type == Dataset::RawVariableType::Binary || raw_var.type == Dataset::RawVariableType::Categorical)
        {
            vector<string> names_to_process;

            if (raw_var.type == Dataset::RawVariableType::Binary)
                names_to_process.push_back(raw_var.name);
            else
                names_to_process = raw_var.categories;

            sort(names_to_process.begin(), names_to_process.end(),
                    [](const string& a, const string& b) { return a.length() > b.length(); });

            for(const string& original_name : names_to_process)
            {
                string scaled_name = "scaled_" + original_name;
                string search_pattern = scaled_name + " = ";
                size_t start_pos = expression.find(search_pattern);

                if(start_pos != string::npos)
                {
                    size_t end_pos = expression.find('\n', start_pos);

                    if (end_pos != string::npos)
                        expression.erase(start_pos, end_pos - start_pos + 1);
                    else
                        expression.erase(start_pos);
                }

                replace_all_appearances(expression, scaled_name, original_name);
            }
        }
    }

    stringstream string_stream(expression);
    string ss_line;
    while(getline(string_stream, ss_line, '\n'))
    {
        if(ss_line.find("=") != string::npos)
        {
            string var_name = get_first_word(ss_line);

            if(!var_name.empty())
                replace_all_word_appearances(expression, var_name, replace_reserved_keywords(var_name));
        }
    }

    for(const Dataset::RawVariable& raw_var : raw_variables)
    {
        if(raw_var.use == "Input")
        {
            if(raw_var.type == Dataset::RawVariableType::Categorical)
                for(const string& cat : raw_var.categories)
                    replace_all_word_appearances(expression, cat, replace_reserved_keywords(cat));
            else
                replace_all_word_appearances(expression, raw_var.name, replace_reserved_keywords(raw_var.name));
        }
    }

    for(const string& name : original_outputs)
        replace_all_word_appearances(expression, name, replace_reserved_keywords(name));

    stringstream ss(expression);
    string line;
    vector<string> lines;
    while(getline(ss, line, '\n'))
    {
        if (line.empty() || all_of(line.begin(), line.end(), [](char c){ return isspace(c); })) continue;
        if (line.find("{") != string::npos) break;
        if (line.back() != ';') line += ';';
        lines.push_back(line);
    }

    if(expression.find("Logistic") != string::npos) logistic = true;
    if(expression.find("RectifiedLinear") != string::npos) relu = true;
    if(expression.find("ExponentialLinear") != string::npos) exp_linear = true;
    if(expression.find("SELU") != string::npos) selu = true;
    if(expression.find("HyperbolicTangent") != string::npos) hyper_tan = true;

    buffer << "import numpy as np\n"
           << "import pandas as pd\n\n"
           << "class NeuralNetwork:\n\n";

    string inputs_list_str;
    Index total_input_vars = 0;
    for(const Dataset::RawVariable& var : raw_variables)
    {
        if(var.use == "Input")
        {
            if(var.type == Dataset::RawVariableType::Categorical)
            {
                for(const string& cat : var.categories)
                {
                    inputs_list_str += "'" + replace_reserved_keywords(cat) + "', ";
                    total_input_vars++;
                }
            }
            else
            {
                inputs_list_str += "'" + replace_reserved_keywords(var.name) + "', ";
                total_input_vars++;
            }
        }
    }
    if(!inputs_list_str.empty())
    {
        inputs_list_str.pop_back();
        inputs_list_str.pop_back();
    }


    buffer << "\tdef __init__(self):\n"
           << "\t\t" << "self.inputs_number = " << to_string(total_input_vars) << "\n"
           << "\t\t" << "self.input_names = [" << inputs_list_str << "]\n\n";

    buffer << "\t@staticmethod\n"
           << "\tdef Linear(x):\n"
           << "\t\t" << "return x\n\n";

    if(logistic)
        buffer << "\t@staticmethod\n"
               << "\tdef Logistic (x):\n"
               << "\t\t" << "z = 1/(1+np.exp(-x))\n"
               << "\t\t" << "return z\n\n";

    if(relu)
        buffer << "\t@staticmethod\n"
               << "\tdef RectifiedLinear (x):\n"
               << "\t\t" << "z = np.maximum(0, x)\n"
               << "\t\t" << "return z\n\n";

    if(exp_linear)
        buffer << "\t@staticmethod\n"
               << "\tdef ExponentialLinear (x):\n"
               << "\t\t"   << "alpha = 1.67326\n"
               << "\t\t"   << "return np.where(x > 0, x, alpha * (np.exp(x) - 1))\n\n";

    if(selu)
        buffer << "\t@staticmethod\n"
               << "\tdef SELU (x):\n"
               << "\t\t"   << "alpha = 1.67326\n"
               << "\t\t"   << "lambda_val = 1.05070\n"
               << "\t\t"   << "return lambda_val * np.where(x > 0, x, alpha * (np.exp(x) - 1))\n\n";

    if(hyper_tan)
        buffer << "\t@staticmethod\n"
               << "\tdef HyperbolicTangent(x):\n"
               << "\t\t" << "return np.tanh(x)\n\n";

    buffer << "\t" << "def calculate_outputs(self, inputs):\n";

    Index input_idx = 0;
    for(const Dataset::RawVariable& var : raw_variables)
    {
        if(var.use == "Input")
        {
            if(var.type == Dataset::RawVariableType::Categorical)
                for(const string& cat : var.categories)
                    buffer << "\t\t" << replace_reserved_keywords(cat) << " = inputs[" << input_idx++ << "]\n";
            else
                buffer << "\t\t" << replace_reserved_keywords(var.name) << " = inputs[" << input_idx++ << "]\n";
        }
    }

    buffer << "\n";

    for(const string& l : lines)
    {
        string processed_line = l;

        replace_all_word_appearances(processed_line, "Linear", "self.Linear");
        replace_all_word_appearances(processed_line, "Logistic", "self.Logistic");
        replace_all_word_appearances(processed_line, "RectifiedLinear", "self.RectifiedLinear");
        replace_all_word_appearances(processed_line, "ExponentialLinear", "self.ExponentialLinear");
        replace_all_word_appearances(processed_line, "SELU", "self.SELU");
        replace_all_word_appearances(processed_line, "HyperbolicTangent", "self.HyperbolicTangent");

        replace(processed_line, ";", "");
        buffer << "\t\t" << processed_line << "\n";
    }

    string return_list = "[";
    for(size_t i = 0; i < outputs.size(); ++i)
    {
        return_list += outputs[i];

        if (i < outputs.size() - 1)
            return_list += ", ";
    }
    return_list += "]";
    buffer << "\t\treturn " << return_list << "\n\n";

    buffer << "\tdef calculate_batch_output(self, input_batch):\n"
           << "\t\t" << "output_batch = np.zeros((len(input_batch), " << to_string(outputs_number) << "))\n"
           << "\t\t" << "for i in range(len(input_batch)):\n"
           << "\t\t\t" << "inputs = list(input_batch[i])\n"
           << "\t\t\t" << "output = self.calculate_outputs(inputs)\n"
           << "\t\t\t" << "output_batch[i] = output\n"
           << "\t\t" << "return output_batch\n\n";

    buffer << "def main():\n"
           << "\n\t# Introduce your input values here\n";

    vector<string> fixed_raw_names = fix_input_names(original_inputs);

    map<string, Dataset::RawVariable> raw_vars_map;
    for(const Dataset::RawVariable& var : raw_variables)
        raw_vars_map[var.name] = var;

    for(Index i = 0; i < inputs_number; ++i)
    {
        const Dataset::RawVariable& var = raw_vars_map[original_inputs[i]];

        if(var.type == Dataset::RawVariableType::Numeric)
            buffer << "\t" << fixed_raw_names[i] << " = 0  # Example: 32.5\n";
        else
        {
            buffer << "\t" << fixed_raw_names[i] << " = 0  # Options: ";

            for(size_t j = 0; j < var.categories.size(); ++j)
                buffer << j << " ('" << var.categories[j] << "')" << (j < var.categories.size() - 1 ? ", " : "");

            buffer << "\n";
        }
    }

    buffer << "\n\t# --- Data conversion (DO NOT modify) ---\n";
    buffer << "\tinputs = []\n\n";

    for(Index i = 0; i < inputs_number; ++i)
    {
        const Dataset::RawVariable& var = raw_vars_map[original_inputs[i]];

        if(var.type == Dataset::RawVariableType::Numeric)
            buffer << "\tinputs.append(" << fixed_raw_names[i] << ")\n";
        else if(var.type == Dataset::RawVariableType::Binary)
            buffer << "\tinputs.append(" << fixed_raw_names[i] << ")\n";
        else if(var.type == Dataset::RawVariableType::Categorical)
            for(size_t j = 0; j < var.categories.size(); ++j)
                buffer << "\tinputs.append(1 if " << fixed_raw_names[i] << " == " << j << " else 0)\n";
    }

    buffer << "\n\t" << "nn = NeuralNetwork()\n"
           << "\t" << "outputs = nn.calculate_outputs(inputs)\n"
           << "\t" << "print(outputs)\n\n"
           << "if __name__ == \"__main__\":\n"
           << "\tmain()\n";

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
        size_t position = 0;

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

    string token;
    stringstream ss(str);

    const size_t num_outputs = outputs.size();

    while(getline(ss, token, '\n'))
    {
        if (token.empty() || all_of(token.begin(), token.end(), [](char c){ return isspace(c); }))
            continue;

        if (token.find("{") != string::npos)
            break;

        if (token.find("=") != string::npos)
            tokens.push_back(token);
    }

    if (tokens.size() < num_outputs)
        return {};

    for (size_t i = 0; i < num_outputs; ++i)
    {
        string intermediate_var_line = tokens[tokens.size() - num_outputs + i];
        string intermediate_var_name = get_first_word(intermediate_var_line);
        string final_output_name = fix_output_names(outputs)[i];

        if(final_output_name != intermediate_var_name)
        {
            string declaration_line;

            switch(programming_Language)
            {
            case ProgrammingLanguage::C:
                declaration_line = "double " + final_output_name + " = " + intermediate_var_name + ";";
                break;
            case ProgrammingLanguage::JavaScript:
                declaration_line = "\tvar " + final_output_name + " = " + intermediate_var_name + ";";
                break;
            case ProgrammingLanguage::Python:
                declaration_line = final_output_name + " = " + intermediate_var_name;
                break;
            case ProgrammingLanguage::PHP:
                declaration_line = "$" + final_output_name + " = $" + intermediate_var_name + ";";
                break;
            }

            out.push_back(declaration_line);
        }
    }

    return out;
}


vector<string> ModelExpression::fix_input_names(const vector<string>& input_names) const
{
    const Index inputs_number = input_names.size();
    vector<string> fixes_input_names(inputs_number);

    for(Index i = 0; i < inputs_number; i++)
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

    for (Index i = 0; i < outputs_number; i++)
        if (output_names[i].empty())
            fixes_output_names[i] = "output_" + to_string(i);
        else
            fixes_output_names[i] = replace_reserved_keywords(output_names[i]);

    return fixes_output_names;
}


void ModelExpression::save_python(const filesystem::path& file_name, const vector<Dataset::RawVariable>& raw_variables) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    file << get_expression_python(raw_variables);
}


void ModelExpression::save_c(const filesystem::path& file_name, const vector<Dataset::RawVariable>& raw_variables) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    file << get_expression_c(raw_variables);
}


void ModelExpression::save_javascript(const filesystem::path& file_name, const vector<Dataset::RawVariable>& raw_variables) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    file << get_expression_javascript(raw_variables);
}


void ModelExpression::save_api(const filesystem::path& file_name, const vector<Dataset::RawVariable>& raw_variables) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;


    file << get_expression_api(raw_variables);
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
