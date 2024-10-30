//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <set>
#include <map>
#include <numeric>
#include <regex>
#include <codecvt>

#include "language_data_set.h"
#include "strings_utilities.h"

namespace opennn
{

LanguageDataSet::LanguageDataSet() : DataSet()
{
    context_dimensions.resize(1);
    context_dimensions.setZero();
}


// string LanguageDataSet::get_text_separator_string() const
// {
//     switch(text_separator)
//     {
//     case Separator::Tab:
//         return "Tab";

//     case Separator::Semicolon:
//         return "Semicolon";

//     default:
//         return string();
//     }
// }


Tensor<string, 1> LanguageDataSet::get_context_vocabulary() const
{
    return context_vocabulary;
}


Tensor<string, 1> LanguageDataSet::get_completion_vocabulary() const
{
    return completion_vocabulary;
}

Index LanguageDataSet::get_context_vocabulary_size() const
{
    return context_vocabulary.size();
}

Index LanguageDataSet::get_completion_vocabulary_size() const
{
    return completion_vocabulary.size();
}

Index LanguageDataSet::get_context_length() const
{
    return max_context_length + 2;
}


Index LanguageDataSet::get_completion_length() const
{
    return max_completion_length + 1;
}


const Tensor<Index, 1>& LanguageDataSet::get_context_variables_dimensions() const
{
    return context_dimensions;
}


const Tensor<Tensor<string, 1>, 1> LanguageDataSet::get_documents() const
{
    return documents;
}

const Tensor<Tensor<string, 1>, 1> LanguageDataSet::get_targets() const
{
    return targets;
}


void LanguageDataSet::set_default_raw_variables_uses()
{
    DataSet::set_default_raw_variables_uses();

    if(raw_variables.size() > 1)
        context_dimensions.resize(1);
}


void LanguageDataSet::set_raw_variables_uses(const Tensor<string, 1>& new_raw_variables_uses)
{
    DataSet::set_raw_variables_uses(new_raw_variables_uses);

    context_dimensions.resize(1);
    context_dimensions.setConstant(get_variables_number(DataSet::VariableUse::Context));
}


void LanguageDataSet::set_raw_variables_uses(const Tensor<VariableUse, 1>& new_raw_variables_uses)
{
    DataSet::set_raw_variables_uses(new_raw_variables_uses);

    context_dimensions.resize(1);
    context_dimensions.setConstant(get_variables_number(DataSet::VariableUse::Context));
}


void LanguageDataSet::set_context_variables_dimensions(const Tensor<Index, 1>& new_context_dimensions)
{
    context_dimensions = new_context_dimensions;
}


void LanguageDataSet::set_context_vocabulary_path(const string& new_context_vocabulary_path)
{
    context_vocabulary_path = new_context_vocabulary_path;
}


void LanguageDataSet::set_completion_vocabulary_path(const string& new_completion_vocabulary_path)
{
    completion_vocabulary_path = new_completion_vocabulary_path;
}


void LanguageDataSet::set_data_random_language_model(const Index& batch_samples_number,
                                                     const Index& completion_length,
                                                     const Index& context_length,
                                                     const Index& completion_dimension,
                                                     const Index& context_dimension)
{
    data_path.clear();

    set(batch_samples_number, context_length + 2 * completion_length);

    for(Index i = 0; i < batch_samples_number; i++)
    {
        for(Index j = 0; j < context_length; j++)
            data(i, j) = type(rand() % context_dimension);

        for(Index j = 0; j < 2 * completion_length; j++)
            data(i, j + context_length) = type(rand() % completion_dimension);
    }

    for(Index i = 0; i < context_length; i++)
        set_raw_variable_use(i, DataSet::VariableUse::Context);

    for(Index i = 0; i < completion_length; i++)
        set_raw_variable_use(i + context_length, DataSet::VariableUse::Input);

    for(Index i = 0; i < completion_length; i++)
        set_raw_variable_use(i + context_length + completion_length, DataSet::VariableUse::Target);
}


void LanguageDataSet::set_default()
{
    DataSet::set_default();

    context_dimensions.resize(1);

    context_dimensions.setConstant(get_variables_number(DataSet::VariableUse::Context));
}


Tensor<string, 2> LanguageDataSet::get_text_data_file_preview() const
{
    return text_data_file_preview;
}


void LanguageDataSet::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataSource");

    if(model_type != ModelType::ImageClassification)
    {
        // File type
        {
            file_stream.OpenElement("FileType");

            file_stream.PushText("csv");

            file_stream.CloseElement();
        }
    }
    else
    {
        // File type
        {
            file_stream.OpenElement("FileType");

            file_stream.PushText("bmp");

            file_stream.CloseElement();
        }
    }

    // Data file name
    {
        file_stream.OpenElement("Path");

        file_stream.PushText(data_path.c_str());

        file_stream.CloseElement();
    }

    // Separator
    {
        file_stream.OpenElement("Separator");
        file_stream.PushText(get_separator_string().c_str());
        file_stream.CloseElement();
    }

    // Raw variables names
    {
        file_stream.OpenElement("HasHeader");
        file_stream.PushText(to_string(has_header).c_str());
        file_stream.CloseElement();
    }

    // Samples id
    {
        file_stream.OpenElement("HasSamplesId");

        buffer.str("");

        buffer << has_sample_ids;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Missing values label
    {
        file_stream.OpenElement("MissingValuesLabel");

        file_stream.PushText(missing_values_label.c_str());

        file_stream.CloseElement();
    }

    // Codification

    file_stream.OpenElement("Codification");

    buffer.str("");
    buffer << get_codification_string();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Close DataFile

    file_stream.CloseElement();

    // Raw variables

    file_stream.OpenElement("RawVariables");

    // Raw variables number
    {
        file_stream.OpenElement("RawVariablesNumber");

        buffer.str("");
        buffer << get_raw_variables_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Raw variables items

    const Index raw_variables_number = get_raw_variables_number();

    {
        for(Index i = 0; i < raw_variables_number; i++)
        {
            file_stream.OpenElement("RawVariable");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            raw_variables(i).to_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close raw_variables

    file_stream.CloseElement();

    // Samples id

    if(has_sample_ids)
    {
        const Index rows_labels_number = samples_id.size();

        file_stream.OpenElement("HasSamplesId");

        buffer.str("");

        for(Index i = 0; i < rows_labels_number; i++)
        {
            buffer << samples_id(i);

            if(i != rows_labels_number-1) buffer << ",";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Samples

    file_stream.OpenElement("Samples");

    // Samples number
    {
        file_stream.OpenElement("SamplesNumber");

        buffer.str("");
        buffer << get_samples_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Samples uses

    {
        file_stream.OpenElement("SamplesUses");

        buffer.str("");

        const Index samples_number = get_samples_number();

        for(Index i = 0; i < samples_number; i++)
        {
            SampleUse sample_use = sample_uses(i);

            buffer << Index(sample_use);

            if(i < (samples_number-1)) buffer << " ";
        }

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Close samples

    file_stream.CloseElement();

    // Missing values

    file_stream.OpenElement("MissingValues");

    // Missing values method

    {
        file_stream.OpenElement("MissingValuesMethod");

        if(missing_values_method == MissingValuesMethod::Mean)
        {
            file_stream.PushText("Mean");
        }
        else if(missing_values_method == MissingValuesMethod::Median)
        {
            file_stream.PushText("Median");
        }
        else if(missing_values_method == MissingValuesMethod::Unuse)
        {
            file_stream.PushText("Unuse");
        }
        else if(missing_values_method == MissingValuesMethod::Interpolation)
        {
            file_stream.PushText("Interpolation");
        }

        file_stream.CloseElement();
    }

    // Missing values number

    {
        file_stream.OpenElement("MissingValuesNumber");

        buffer.str("");
        buffer << missing_values_number;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    if(missing_values_number > 0)
    {
        // Raw variables missing values number
        {
            file_stream.OpenElement("RawVariablesMissingValuesNumber");

            const Index raw_variables_number = raw_variables_missing_values_number.size();

            buffer.str("");

            for(Index i = 0; i < raw_variables_number; i++)
            {
                buffer << raw_variables_missing_values_number(i);

                if(i != (raw_variables_number-1)) buffer << " ";
            }

            file_stream.PushText(buffer.str().c_str());

            file_stream.CloseElement();
        }

        // Rows missing values number
        {
            file_stream.OpenElement("RowsMissingValuesNumber");

            buffer.str("");
            buffer << rows_missing_values_number;

            file_stream.PushText(buffer.str().c_str());

            file_stream.CloseElement();
        }
    }

    // Missing values

    file_stream.CloseElement();

    // Preview data

    file_stream.OpenElement("PreviewData");

    file_stream.OpenElement("PreviewSize");

    buffer.str("");

    if(model_type != ModelType::TextClassification)
    {
        buffer << data_file_preview.size();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        for(Index i = 0; i < data_file_preview.size(); i++)
        {
            file_stream.OpenElement("Row");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            for(Index j = 0; j < data_file_preview(i).size(); j++)
            {
                file_stream.PushText(data_file_preview(i)(j).c_str());

                if(j != data_file_preview(i).size()-1)
                {
                    file_stream.PushText(",");
                }
            }

            file_stream.CloseElement();
        }
    }
    else
    {
        buffer << text_data_file_preview.dimension(0);

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        for(Index i = 0; i < text_data_file_preview.dimension(0); i++)
        {
            file_stream.OpenElement("Row");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            file_stream.PushText(text_data_file_preview(i,0).c_str());

            file_stream.CloseElement();
        }

        for(Index i = 0; i < text_data_file_preview.dimension(0); i++)
        {
            file_stream.OpenElement("Target");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            file_stream.PushText(text_data_file_preview(i,1).c_str());

            file_stream.CloseElement();
        }
    }

    // Close preview data

    file_stream.CloseElement();

    // Close data set

    file_stream.CloseElement();

    time(&finish);
}


void LanguageDataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{

    ostringstream buffer;

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    const tinyxml2::XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data file element is nullptr.\n");

    set_data_source_path(read_xml_string(data_source_element, "Path"));
    set_separator_name(read_xml_string(data_source_element, "Separator"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_xml_string(data_source_element, "Codification"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    // RawVariables

    const tinyxml2::XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    // Raw variables number

    const tinyxml2::XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
        throw runtime_error("RawVariablesNumber element is nullptr.\n");

    Index new_raw_variables_number = 0;

    if(raw_variables_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(raw_variables_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // Raw variables

    const tinyxml2::XMLElement* start_element = raw_variables_number_element;

    for(Index i = 0; i < new_raw_variables_number; i++)
    {
        const tinyxml2::XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Raw_variable item number (" + to_string(i + 1) + ") exception.\n");

        RawVariable& raw_variable = raw_variables(i);
        raw_variable.name = read_xml_string(raw_variable_element, "Name");
        raw_variable.set_scaler(read_xml_string(raw_variable_element, "Scaler"));
        raw_variable.set_use(read_xml_string(raw_variable_element, "Use"));
        raw_variable.set_type(read_xml_string(raw_variable_element, "Type"));


        if(raw_variables(i).type == RawVariableType::Categorical || raw_variables(i).type == RawVariableType::Binary)
            raw_variable.categories = get_tokens(read_xml_string(raw_variable_element, "Categories"), ";");

//        raw_variable_element = raw_variable_element->NextSiblingElement("RawVariable");
    }

    // Rows label

    if(has_sample_ids)
        samples_id = get_tokens(read_xml_string(data_set_element, "SamplesId"), ",");

    // Samples

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
        throw runtime_error("Samples number element is nullptr.\n");

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = Index(atoi(samples_number_element->GetText()));

        sample_uses.resize(new_samples_number);

        set(DataSet::SampleUse::Training);
    }

    // Samples uses

    const tinyxml2::XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
        throw runtime_error("Samples uses element is nullptr.\n");

    if(samples_uses_element->GetText())
        set_sample_uses(get_tokens(samples_uses_element->GetText(), " "));

    // Missing values

    const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    // Missing values method

    const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
        throw runtime_error("Missing values method element is nullptr.\n");

    if(missing_values_method_element->GetText())
        set_missing_values_method(missing_values_method_element->GetText());

    // Missing values number

    const tinyxml2::XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
        throw runtime_error("Missing values number element is nullptr.\n");

    if(missing_values_number_element->GetText())
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));

    if(missing_values_number > 0)
    {
        // Raw variables Missing values number

        const tinyxml2::XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("");

        if(!raw_variables_missing_values_number_element)
            throw runtime_error("RawVariablesMissingValuesNumber element is nullptr.\n");

        if(raw_variables_missing_values_number_element->GetText())
        {
            const Tensor<string, 1> new_raw_variables_missing_values_number
                = get_tokens(raw_variables_missing_values_number_element->GetText(), " ");

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number(i).c_str());
        }

        // Rows missing values number

        const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
            throw runtime_error("Rows missing values number element is nullptr.\n");

        if(rows_missing_values_number_element->GetText())
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
    }

    // Preview data

    const tinyxml2::XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n");

    // Preview size

    const tinyxml2::XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
        throw runtime_error("Preview size element is nullptr.\n");

    Index new_preview_size = 0;

    if(preview_size_element->GetText())
    {
        new_preview_size = Index(atoi(preview_size_element->GetText()));

        if(new_preview_size > 0) data_file_preview.resize(new_preview_size);
        if(new_preview_size > 0) text_data_file_preview.resize(new_preview_size, 2);
    }

    // Preview data

    start_element = preview_size_element;

    if(model_type != ModelType::TextClassification)
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                data_file_preview(i) = get_tokens(row_element->GetText(), ",");
        }
    }
    else
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                text_data_file_preview(i,0) = row_element->GetText();
        }

        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Target");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Target item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                text_data_file_preview(i,1) = row_element->GetText();
        }
    }

    // Display

    set_display(read_xml_bool(data_set_element, "Display"));
}


void LanguageDataSet::import_vocabulary(const string& path, Tensor<string, 1>& vocabulary)
{
    ifstream file(path.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open data file: " + path + "\n");

    Index vocabulary_size = 0;

    string line;

    while(getline(file, line))
    {
        if(line.empty()) continue;

        vocabulary_size++;

        if(file.peek() == EOF) break;
    }

    file.clear();
    file.seekg(0, ios::beg);

    vocabulary.resize(vocabulary_size);

    Index count = 0;

    while(getline(file, line))
    {
        if(line.empty()) continue;

        vocabulary(count++) = line;

        if(file.peek() == EOF) break;
    }
}


struct WordpieceAlgorithmParameters
{
    Index upper_threshold;
    Index lower_threshold;
    Index interations_number;
    Index max_input_tokens;
    Index max_token_length;
    Index max_unique_characters;
    Index vocabulary_size;
    float slack_ratio;
    bool include_joiner_token;
    string joiner;
    vector<string> reserved_tokens;
};


set<char> extract_character_tokens(const vector<pair<string, int>>& word_counts)
{
    set<char> seen_chars;

    for(const auto& [word, _] : word_counts)
    {
        for(char c : word)
        {
            seen_chars.insert(c);
        }
    }

    return seen_chars;
}


map<string, int> ensure_all_tokens_exist(const set<string>& input_tokens,
    map<string, int> output_tokens,
    bool include_joiner_token,
    const string& joiner)
{
    for(const string& token : input_tokens)
    {
        if(output_tokens.find(token) == output_tokens.end())
        {
            output_tokens[token] = 1;
        }

        if(include_joiner_token)
        {
            string joined_token = joiner + token;

            if(output_tokens.find(joined_token) == output_tokens.end())
            {
                output_tokens[joined_token] = 1;
            }
        }
    }

    return output_tokens;
}


vector<int> get_split_indices(const string& word, 
                              const map<string, int>& current_tokens, 
                              bool include_joiner_token, 
                              const string& joiner)
{
    vector<int> indices;
    int start = 0;

    while(start < word.size())
    {
        size_t end = word.size();

        while(end > start)
        {
            string subtoken = word.substr(start, end - start);

            if(include_joiner_token && start > 0)    
                subtoken = joiner + subtoken;

            if(current_tokens.find(subtoken) != current_tokens.end())
            {
                indices.push_back(end);
                break;
            }

            end--;
        }

        if(end == start)
        {
            return {};
        }

        start = end;
    }
    return indices;
}


tuple<int, int> calculate_thresholds(const vector<pair<string, int>>& word_counts, int upper_threshold, int lower_threshold)
{
    vector<int> counts;
    for(const auto& [_, count] : word_counts)    counts.push_back(count);

    int max_count = *max_element(counts.begin(), counts.end());
    int min_count = *min_element(counts.begin(), counts.end());

    int upper_search = upper_threshold == -1 ? max_count : min(upper_threshold, max_count);
    int lower_search = lower_threshold == -1 ? min_count : max(lower_threshold, min_count);

    return { upper_search, lower_search };
}


vector<pair<string, int>> trim_inputs(const vector<pair<string, int>>& word_counts,
    const vector<string>& reserved_tokens,
    int max_token_length)
{
    vector<pair<string, int>> trimmed_counts;

    for(const auto& [word, count] : word_counts)
    {
        if(word.size() > max_token_length || find(reserved_tokens.begin(), reserved_tokens.end(), word) != reserved_tokens.end())
        {
            continue;
        }

        trimmed_counts.push_back({ word, count });
    }

    return trimmed_counts;
}


set<char> get_allowed_characters(const vector<pair<string, int>>& trimmed_counts, int max_unique_characters)
{
    map<char, int> character_counts;

    for(const auto& [word, count] : trimmed_counts)
    {
        for(char c : word)
        {
            character_counts[c] += count;
        }
    }

    vector<pair<char, int>> sorted_counts(character_counts.begin(), character_counts.end());

    sort(sorted_counts.begin(), sorted_counts.end(), [](const pair<char, int>& a, const pair<char, int>& b)
        {
            if(a.second != b.second)
                return a.second > b.second;
            return a.first < b.first;
        }
);

    set<char> allowed_characters;
    for(int i = 0; i < min((int)sorted_counts.size(), max_unique_characters); i++)    allowed_characters.insert(sorted_counts[i].first);

    return allowed_characters;
}

vector<pair<string, int>> filter_inputs(const vector<pair<string, int>>& trimmed_counts, const set<char>& allowed_characters, int max_input_tokens)
{
    vector<pair<string, int>> sorted_counts = trimmed_counts;

    sort(sorted_counts.begin(), sorted_counts.end(), [](const pair<string, int>& a, const pair<string, int>& b)
        {
            return a.second > b.second;
        }
);

    vector<pair<string, int>> filtered_counts;

    for(const auto& [word, count] : sorted_counts)
    {
        if(max_input_tokens != -1 && filtered_counts.size() >= max_input_tokens)    break;

        bool has_unallowed_characters = false;
        for(char c : word)
        {
            if(allowed_characters.find(c) == allowed_characters.end())
            {
                has_unallowed_characters = true;
                break;
            }
        }

        if(has_unallowed_characters)
        {
            continue;
        }

        filtered_counts.push_back({ word, count });
    }

    return filtered_counts;
}

vector<string> generate_final_vocabulary(const vector<string>& reserved_tokens,
    const set<char>& character_tokens,
    const map<string, int>& current_tokens)
{
    vector<string> vocabulary;
    vocabulary.insert(vocabulary.end(), reserved_tokens.begin(), reserved_tokens.end());

    vector<string> sorted_character_tokens;
    for(const char ch : character_tokens)    sorted_character_tokens.push_back(string(1, ch));

    sort(sorted_character_tokens.begin(), sorted_character_tokens.end());
    vocabulary.insert(vocabulary.end(), sorted_character_tokens.begin(), sorted_character_tokens.end());

    vector<pair<string, int>> sorted_tokens(current_tokens.begin(), current_tokens.end());
    sort(sorted_tokens.begin(), sorted_tokens.end(), [](const pair<string, int>& a, const pair<string, int>& b)
        {
            if(a.second != b.second)    return a.second > b.second;
            return a.first < b.first;
        }
);

    for(const auto& [token, _] : sorted_tokens)
    {
        vocabulary.push_back(token);
    }

    set<string> seen_tokens;
    vector<string> final_vocabulary;

    for(const string& word : vocabulary)
    {
        if(seen_tokens.find(word) == seen_tokens.end())
        {
            seen_tokens.insert(word);
            final_vocabulary.push_back(word);
        }
    }

    return final_vocabulary;
}


vector<string> calculate_vocabulary_with_threshold(const vector<pair<string, int>>& word_counts,
                                                   int threshold,
                                                   const WordpieceAlgorithmParameters& parameters)
{
    set<char> character_tokens = extract_character_tokens(word_counts);
    set<string> string_tokens;
    for(const char ch : character_tokens)    string_tokens.insert(string(1, ch));

    map<string, int> current_tokens = ensure_all_tokens_exist(string_tokens, map<string, int>(), parameters.include_joiner_token, parameters.joiner);

    for(int iteration = 0; iteration < parameters.interations_number; ++iteration)
    {
        vector<map<string, int>> subtokens(parameters.max_token_length + 1);

        for(const auto& [word, count] : word_counts)
        {
            vector<int> split_indices;

            if(iteration == 0)
            {
                split_indices = vector<int>(word.size());
                iota(split_indices.begin(), split_indices.end(), 1);
            }
            else
            {
                split_indices = get_split_indices(word, current_tokens, parameters.include_joiner_token, parameters.joiner);
                if(split_indices.empty()) continue;
            }

            size_t start = 0;
            for(int split_index : split_indices)
            {
                for(int end = start + 1; end <= word.size(); ++end)
                {
                    string subtoken = word.substr(start, end - start);
                    int length = subtoken.size();

                    if(parameters.include_joiner_token && start > 0)    
                        subtoken = parameters.joiner + subtoken;

                    subtokens[length][subtoken] += count;
                }
                start = split_index;
            }
        }

        map<string, int> next_tokens;

        for(size_t length = parameters.max_token_length; length > 0; --length)
        {
            for(const auto& [token, count] : subtokens[length])
            {
                if(count >= threshold)    next_tokens[token] = count;

                if(token.size() > length)
                {
                    const size_t joiner_length = parameters.joiner.size();

                    for(size_t i = 1 + joiner_length; i <= length + joiner_length; i++)
                    {
                        string prefix = token.substr(0, i);

                        if(subtokens[i - joiner_length].find(prefix) != subtokens[i - joiner_length].end())
                            subtokens[i - joiner_length][prefix] -= count;
                    }
                }
                else
                {
                    for(int i = 1; i < length; i++)
                    {
                        const string prefix = token.substr(0, i);

                        if(subtokens[i].find(prefix) != subtokens[i].end())   
                            subtokens[i][prefix] -= count;
                    }
                }
            }
        }

        current_tokens = ensure_all_tokens_exist(string_tokens, next_tokens, parameters.include_joiner_token, parameters.joiner);
    }

    return generate_final_vocabulary(parameters.reserved_tokens, character_tokens, current_tokens);
}


vector<string> calculate_vocabulary_binary_search(const vector<pair<string, int>>& word_counts,
    int lower_bound,
    int upper_bound,
    const WordpieceAlgorithmParameters& parameters)
{
    const int threshold = (upper_bound + lower_bound) / 2;

    const vector<string> current_vocabulary = calculate_vocabulary_with_threshold(word_counts, threshold, parameters);

    const int current_vocabulary_size = current_vocabulary.size();

    int slack = parameters.slack_ratio * parameters.vocabulary_size;
    if(slack < 0)    slack = 0;

    const bool is_within_slack = (current_vocabulary_size <= parameters.vocabulary_size) && (parameters.vocabulary_size - current_vocabulary_size <= slack);

    if(is_within_slack || lower_bound >= upper_bound || threshold <= 1)    
        return current_vocabulary;

    if(current_vocabulary_size > parameters.vocabulary_size)
        return calculate_vocabulary_binary_search(word_counts, threshold + 1, upper_bound, parameters);
    else
        return calculate_vocabulary_binary_search(word_counts, lower_bound, threshold - 1, parameters);
}


const Tensor<string, 1> LanguageDataSet::calculate_vocabulary(const Tensor<Tensor<string, 1>, 1>& tokens,
                                                              const Index& vocabulary_size,
                                                              const vector<string>& reserved_tokens,
                                                              const Index& upper_threshold,
                                                              const Index& lower_threshold,
                                                              const Index& interations_number,
                                                              const Index& max_input_tokens,
                                                              const Index& max_token_length,
                                                              const Index& max_unique_characters,
                                                              const float& slack_ratio,
                                                              const bool& include_joiner_token,
                                                              const string& joiner)
{
    const Tensor<string, 1> total_tokens = tokens_list(tokens);

    const vector<pair<string, int>> word_counts = count_words(total_tokens);

    const WordpieceAlgorithmParameters parameters = {upper_threshold,
                                                     lower_threshold,
                                                     interations_number,
                                                     max_input_tokens,
                                                     max_token_length,
                                                     max_unique_characters,
                                                     vocabulary_size,
                                                     slack_ratio,
                                                     include_joiner_token,
                                                     joiner,
                                                     reserved_tokens};

    const auto [upper_search, lower_search] = calculate_thresholds(word_counts, parameters.upper_threshold, parameters.lower_threshold);

    const vector<pair<string, int>> trimmed_counts = trim_inputs(word_counts, parameters.reserved_tokens, parameters.max_token_length);

    const std::set<char> allowed_characters = get_allowed_characters(trimmed_counts, parameters.max_unique_characters);

    const vector<pair<string, int>> filtered_counts = filter_inputs(trimmed_counts, allowed_characters, parameters.max_input_tokens);

    const vector<string> vocabulary = calculate_vocabulary_binary_search(filtered_counts, lower_search, upper_search, parameters);

    Tensor<string, 1> vocabulary_tensor(vocabulary.size());

    for(Index i = 0; i < Index(vocabulary.size()); i++)
        vocabulary_tensor(i) = vocabulary[i];

    return vocabulary_tensor;
}


void LanguageDataSet::load_documents(const string& path)
{
    const Index original_size = documents.size();

    if(path.empty())
        throw runtime_error("Data file name is empty.\n");

    ifstream file(path.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open data file: " + path + "\n");

    Tensor<Tensor<string,1>, 1> documents_copy(documents);

    documents.resize(original_size + 1);

    Tensor<Tensor<string,1>, 1> targets_copy(targets);

    targets.resize(original_size + 1);

    for(Index i = 0; i < original_size; i++)
    {
        documents(i) = documents_copy(i);
        targets(i) = targets_copy(i);
    }

    Index lines_count = 0;
    Index lines_number = 0;

    string line;

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

        lines_number++;

        if(file.peek() == EOF) break;
    }

    //file.close();
    Tensor<string, 1> document(lines_number);
    Tensor<string, 1> document_target(lines_number);

    file.seekg (0, ios::beg);

    Index tokens_number = 0;

    string delimiter;
    const string separator = get_separator_string();

    while(getline(file, line))
    {
        if(line.empty()) continue;

        if(line[0] == '"')
        {
            replace(line,"\"\"", "\"");
            line = "\"" + line;
            delimiter = "\"\"";
        }

        if(line.find("\"" + separator) != string::npos)
            replace(line,"\"" + separator, "\"\"" + separator);

        //tokens_number = count_tokens(line,delimiter + separator);
        const Tensor<string,1> tokens = get_tokens(line, delimiter + separator);
        tokens_number = tokens.size();

        if(tokens_number == 1)
        {
            if(tokens(0).find(delimiter,0) == 0)

                document(lines_count) += tokens(0).substr(delimiter.length(), tokens(0).size());

            else

                document(lines_count) += " " + tokens(0);


            lines_count++;
        }
        else
        {
            if(tokens_number > 2)
                throw runtime_error("Found more than one separator in line: " + line + "\n");

            if(tokens(0).empty() && tokens(1).empty())  
                continue;

            document(lines_count) += " " + tokens(0);
            document_target(lines_count) += tokens(1);
            delimiter.clear();
            lines_count++;
        }

        if(file.peek() == EOF) 
            break;
    }

    Tensor<string,1> document_copy(lines_count);
    Tensor<string,1> document_target_copy(lines_count);

    copy(document.data(),
        document.data() + lines_count,
        document_copy.data());

    copy(document_target.data(),
        document_target.data() + lines_count,
        document_target_copy.data());

    documents(original_size) = document_copy;
    targets(original_size) = document_target_copy;

    file.close();
}


// void LanguageDataSet::extractDataPreview()
// {
//     ifstream file(data_path.c_str());

//     const bool is_float = is_same<type, float>::value;

//     const string separator_string = get_separator_string();

//     string line;

//     //skip_header(file);

//     // Read data

//     const Index raw_variables_number = has_sample_ids ? get_raw_variables_number() + 1 : get_raw_variables_number();

//     Tensor<string, 1> tokens(raw_variables_number);

//     const Index samples_number = data.dimension(0);

//     if(has_sample_ids) samples_id.resize(samples_number);

//     if(display) cout << "Reading data..." << endl;

//     Index sample_index = 0;
//     Index raw_variable_index = 0;

//     while(getline(file, line))
//     {
//         prepare_line(line);

//         if(line.empty()) continue;

//         fill_tokens(line, separator_string, tokens);

//         for(Index j = 0; j < raw_variables_number; j++)
//         {
//             trim(tokens(j));

//             if(has_sample_ids && j == 0){
//                 samples_id(sample_index) = tokens(j); continue;
//             }
//             if(tokens(j) == missing_values_label || tokens(j).empty())
//                 data(sample_index, raw_variable_index++) = type(NAN);
//             else if(is_float)
//                 data(sample_index, raw_variable_index++) = type(strtof(tokens(j).data(), nullptr));
//             else
//                 data(sample_index, raw_variable_index++) = type(stof(tokens(j)));
//         }
//         raw_variable_index = 0;
//         sample_index++;
//     }

//     data_file_preview(has_header ? 3 : 2) = tokens;

//     file.close();

//     if(display) cout << "Data read successfully..." << endl;
// }


// void LanguageDataSet::readDataFilePreview()
// {
//     if (display) cout << "Path: " << data_path << endl;
//     if (data_path.empty()) throw runtime_error("Data file name is empty.");

//     std::ifstream file;

// #ifdef _WIN32

//     if(std::regex_search(data_path, accent_regex))
//     {
//         std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
//         std::wstring file_name_wide = conv.from_bytes(data_path);
//         file.open(file_name_wide);
//     }
//     else
//     {
//         file.open(data_path.c_str());
//     }
// #else
//     file.open(data_path.c_str());
// #endif

//     if (!file.is_open()) throw runtime_error("Cannot open data file: " + data_path);


//     const string separator_char = get_separator_string();


//     if(display) cout << "Setting data file preview..." << endl;

//     Index lines_number = has_binary_raw_variables()? 4 : 3;

//     data_file_preview.resize(lines_number);

//     string line;

//     Index lines_count = 0;
//     while (file.good() && lines_count < data_file_preview.size())
//     {
//         getline(file, line);
//         decode(line);
//         trim(line);
//         erase(line, '"');
//         if (line.empty()) continue;
//         check_separators(line);
//         data_file_preview(lines_count++) = get_tokens(line, separator_char);
//     }

//     file.close();

//     if(data_file_preview(0).size() == 0)
//     {
//         ostringstream buffer;

//         buffer << "OpenNN Exception: DataSet class.\n"
//                << "void read_csv_1() method.\n"
//                << "File " << data_path << " is empty.\n";

//         throw runtime_error(buffer.str());
//     }

//     // Set rows labels and raw_variables names

//     if(display) cout << "Setting rows labels..." << endl;

//     string first_name = data_file_preview(0)(0);
//     transform(first_name.begin(), first_name.end(), first_name.begin(), ::tolower);

//     const Index raw_variables_number = get_has_rows_labels() ? data_file_preview(0).size()-1 : data_file_preview(0).size();

//     raw_variables.resize(raw_variables_number);

//     // Check if header has numeric value

//     if(has_binary_raw_variables() && has_numbers(data_file_preview(0))){
//         ostringstream buffer;

//         buffer << "OpenNN Exception: DataSet class.\n"
//                << "void read_csv_1() method.\n"
//                << "Some raw_variables names are numeric.\n";

//         throw runtime_error(buffer.str());
//     }

//     // raw_variables names

//     if(display) cout << "Setting raw_variables names..." << endl;

//     if(has_binary_raw_variables()){
//         get_has_rows_labels() ? set_raw_variable_names(data_file_preview(0).slice(Eigen::array<Eigen::Index, 1>({1}),
//                                                                                   Eigen::array<Eigen::Index, 1>({data_file_preview(0).size()-1})))
//                               : set_raw_variable_names(data_file_preview(0));
//     }
//     else{
//         set_raw_variable_names(get_default_raw_variables_names(raw_variables_number));
//     }

//     // Check raw_variables with all missing values

//     bool has_nans_raw_variables = false;

//     do
//     {
//         has_nans_raw_variables = false;

//         if(lines_number > 10)
//             break;

//         for(Index i = 0; i < data_file_preview(0).dimension(0); i++)
//         {
//             if(get_has_rows_labels() && i == 0) continue;

//             // Check if all are missing values

//             if( data_file_preview(1)(i) == missing_values_label
//                 && data_file_preview(2)(i) == missing_values_label
//                 && data_file_preview(lines_number-2)(i) == missing_values_label
//                 && data_file_preview(lines_number-1)(i) == missing_values_label)
//             {
//                 has_nans_raw_variables = true;
//             }
//             else
//             {
//                 has_nans_raw_variables = false;
//             }

//             if(has_nans_raw_variables)
//             {
//                 lines_number++;
//                 data_file_preview.resize(lines_number);

//                 string line;
//                 Index lines_count = 0;

//                 file.open(data_path.c_str());

//                 if(!file.is_open())
//                 {
//                     ostringstream buffer;

//                     buffer << "OpenNN Exception: DataSet class.\n"
//                            << "void read_csv() method.\n"
//                            << "Cannot open data file: " << data_path << "\n";

//                     throw runtime_error(buffer.str());
//                 }

//                 while(file.good())
//                 {
//                     getline(file, line);
//                     decode(line);
//                     trim(line);
//                     erase(line, '"');
//                     if(line.empty()) continue;
//                     check_separators(line);
//                     data_file_preview(lines_count) = get_tokens(line, separator_char);
//                     lines_count++;
//                     if(lines_count == lines_number) break;
//                 }
//                 file.close();
//             }
//         }
//     }while(has_nans_raw_variables);

//     // raw_variables types

//     if(display) cout << "Setting raw_variables types..." << endl;

//     Index raw_variable_index = 0;

//     for(Index i = 0; i < data_file_preview(0).dimension(0); i++)
//     {
//         if(get_has_rows_labels() && i == 0) continue;

//         string data_file_preview_1 = data_file_preview(1)(i);
//         string data_file_preview_2 = data_file_preview(2)(i);
//         string data_file_preview_3 = data_file_preview(lines_number-2)(i);
//         string data_file_preview_4 = data_file_preview(lines_number-1)(i);

//         string preview1 = data_file_preview(1)(i);
//         string preview2 = data_file_preview(2)(i);
//         string preview_last1 = data_file_preview(data_file_preview.size() - 2)(i);
//         string preview_last2 = data_file_preview(data_file_preview.size() - 1)(i);

//         if (is_date_time_string(preview1) || is_date_time_string(preview2) || is_date_time_string(preview_last1) || is_date_time_string(preview_last2))
//             raw_variables(i).type = RawVariableType::DateTime;
//         else if (is_numeric_string(preview1) || is_numeric_string(preview2) || is_numeric_string(preview_last1) || is_numeric_string(preview_last2))
//             raw_variables(i).type = RawVariableType::Numeric;
//         else
//             raw_variables(i).type = RawVariableType::Categorical;


//     }

//     // Resize data file preview to original

//     if(data_file_preview.size() > 4)
//     {
//         lines_number = has_binary_raw_variables() ? 4 : 3;

//         Tensor<Tensor<string, 1>, 1> data_file_preview_copy(data_file_preview);

//         data_file_preview.resize(lines_number);

//         data_file_preview(0) = data_file_preview_copy(1);
//         data_file_preview(1) = data_file_preview_copy(1);
//         data_file_preview(2) = data_file_preview_copy(2);
//         data_file_preview(lines_number - 2) = data_file_preview_copy(data_file_preview_copy.size()-2);
//         data_file_preview(lines_number - 1) = data_file_preview_copy(data_file_preview_copy.size()-1);
//     }
// }

// void LanguageDataSet::validateDataIntegrity()
// {
//     regex accent_regex("[\\xC0-\\xFF]");
//     std::ifstream file;

// #ifdef _WIN32

//     if(regex_search(data_path, accent_regex))
//     {
//         wstring_convert<codecvt_utf8<wchar_t>> conv;
//         wstring file_name_wide = conv.from_bytes(data_path);
//         file.open(file_name_wide);
//     }else
//     {
//         file.open(data_path.c_str());
//     }

// #else
//     file.open(data_path.c_str());
// #endif

//     if (!file.is_open()) throw runtime_error("Cannot open data file: " + data_path);

//     string line;
//     Index line_number = 0;

//     Index samples_count = 0;
//     const string separator_char = get_separator_string();
//     const Index raw_variables_number = get_has_rows_labels() ? get_raw_variables_number() + 1 : get_raw_variables_number();

//     while (getline(file, line))
//     {
//         line_number++;
//         trim(line);
//         erase(line, '"');
//         if (line.empty()) continue;

//         if (count_tokens(line, separator_char) != raw_variables_number)
//             throw runtime_error("Line " + to_string(line_number) + ": Incorrect number of tokens.");

//         samples_count++;
//     }


//     file.close();

//     data.resize(samples_count, get_raw_variables_number());

//     set_default_raw_variables_uses();

//     //samples_uses.resize(samples_count);
//     //samples_uses.setConstant(SampleUse::Training);

//     split_samples_random();
// }



// void LanguageDataSet::read_csv_language_model()
// {
//     read_csv_1();

//     read_csv_2_simple();

//     read_csv_3_language_model();
// }

void LanguageDataSet::read_csv_language_model()
{
    if (display) cout << "Path: " << data_path << endl;
    if (data_path.empty()) throw runtime_error("Data file name is empty.");

    std::ifstream file;

#ifdef _WIN32

    if(std::regex_search(data_path, accent_regex))
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        std::wstring file_name_wide = conv.from_bytes(data_path);
        file.open(file_name_wide);
    }
    else
    {
        file.open(data_path.c_str());
    }
#else
    file.open(data_path.c_str());
#endif

    if (!file.is_open()) throw runtime_error("Cannot open data file: " + data_path);


    string separator_char = get_separator_string();


    if(display) cout << "Setting data file preview..." << endl;

    Index lines_number = has_binary_raw_variables()? 4 : 3;

    data_file_preview.resize(lines_number);

    string line;

    Index lines_count = 0;
    while (file.good() && lines_count < data_file_preview.size())
    {
        getline(file, line);
        decode(line);
        trim(line);
        erase(line, '"');
        if (line.empty()) continue;
        check_separators(line);
        data_file_preview(lines_count++) = get_tokens(line, separator_char);
    }

    file.close();

    if(data_file_preview(0).size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "File " << data_path << " is empty.\n";

        throw runtime_error(buffer.str());
    }

    // Set rows labels and raw_variables names

    if(display) cout << "Setting rows labels..." << endl;

    string first_name = data_file_preview(0)(0);
    transform(first_name.begin(), first_name.end(), first_name.begin(), ::tolower);

    Index raw_variables_number = get_has_rows_labels() ? data_file_preview(0).size()-1 : data_file_preview(0).size();

    raw_variables.resize(raw_variables_number);

    // Check if header has numeric value

    if(has_binary_raw_variables() && has_numbers(data_file_preview(0))){
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "Some raw_variables names are numeric.\n";

        throw runtime_error(buffer.str());
    }

    // raw_variables names

    if(display) cout << "Setting raw_variables names..." << endl;

    if(has_binary_raw_variables()){
        get_has_rows_labels() ? set_raw_variable_names(data_file_preview(0).slice(Eigen::array<Eigen::Index, 1>({1}),
                                                                                  Eigen::array<Eigen::Index, 1>({data_file_preview(0).size()-1})))
                              : set_raw_variable_names(data_file_preview(0));
    }
    else{
        set_raw_variable_names(get_default_raw_variables_names(raw_variables_number));
    }

    // Check raw_variables with all missing values

    bool has_nans_raw_variables = false;

    do
    {
        has_nans_raw_variables = false;

        if(lines_number > 10)
            break;

        for(Index i = 0; i < data_file_preview(0).dimension(0); i++)
        {
            if(get_has_rows_labels() && i == 0) continue;

            // Check if all are missing values

            if( data_file_preview(1)(i) == missing_values_label
                && data_file_preview(2)(i) == missing_values_label
                && data_file_preview(lines_number-2)(i) == missing_values_label
                && data_file_preview(lines_number-1)(i) == missing_values_label)
            {
                has_nans_raw_variables = true;
            }
            else
            {
                has_nans_raw_variables = false;
            }

            if(has_nans_raw_variables)
            {
                lines_number++;
                data_file_preview.resize(lines_number);

                string line;
                Index lines_count = 0;

                file.open(data_path.c_str());

                if(!file.is_open())
                {
                    ostringstream buffer;

                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void read_csv() method.\n"
                           << "Cannot open data file: " << data_path << "\n";

                    throw runtime_error(buffer.str());
                }

                while(file.good())
                {
                    getline(file, line);
                    decode(line);
                    trim(line);
                    erase(line, '"');
                    if(line.empty()) continue;
                    check_separators(line);
                    data_file_preview(lines_count) = get_tokens(line, separator_char);
                    lines_count++;
                    if(lines_count == lines_number) break;
                }
                file.close();
            }
        }
    }while(has_nans_raw_variables);

    // raw_variables types

    if(display) cout << "Setting raw_variables types..." << endl;

    Index raw_variable_index = 0;

    for(Index i = 0; i < data_file_preview(0).dimension(0); i++)
    {
        if(get_has_rows_labels() && i == 0) continue;

        string data_file_preview_1 = data_file_preview(1)(i);
        string data_file_preview_2 = data_file_preview(2)(i);
        string data_file_preview_3 = data_file_preview(lines_number-2)(i);
        string data_file_preview_4 = data_file_preview(lines_number-1)(i);

        string preview1 = data_file_preview(1)(i);
        string preview2 = data_file_preview(2)(i);
        string preview_last1 = data_file_preview(data_file_preview.size() - 2)(i);
        string preview_last2 = data_file_preview(data_file_preview.size() - 1)(i);

        if (is_date_time_string(preview1) || is_date_time_string(preview2) || is_date_time_string(preview_last1) || is_date_time_string(preview_last2))
            raw_variables(i).type = RawVariableType::DateTime;
        else if (is_numeric_string(preview1) || is_numeric_string(preview2) || is_numeric_string(preview_last1) || is_numeric_string(preview_last2))
            raw_variables(i).type = RawVariableType::Numeric;
        else
            raw_variables(i).type = RawVariableType::Categorical;


    }

    // Resize data file preview to original

    if(data_file_preview.size() > 4)
    {
        lines_number = has_binary_raw_variables() ? 4 : 3;

        Tensor<Tensor<string, 1>, 1> data_file_preview_copy(data_file_preview);

        data_file_preview.resize(lines_number);

        data_file_preview(0) = data_file_preview_copy(1);
        data_file_preview(1) = data_file_preview_copy(1);
        data_file_preview(2) = data_file_preview_copy(2);
        data_file_preview(lines_number - 2) = data_file_preview_copy(data_file_preview_copy.size()-2);
        data_file_preview(lines_number - 1) = data_file_preview_copy(data_file_preview_copy.size()-1);
    }


    regex accent_regex("[\\xC0-\\xFF]");
    //std::ifstream file;

#ifdef _WIN32

    if(regex_search(data_path, accent_regex))
    {
        wstring_convert<codecvt_utf8<wchar_t>> conv;
        wstring file_name_wide = conv.from_bytes(data_path);
        file.open(file_name_wide);
    }else
    {
        file.open(data_path.c_str());
    }

#else
    file.open(data_path.c_str());
#endif

    if (!file.is_open()) throw runtime_error("Cannot open data file: " + data_path);

    //string line;
    Index line_number = 0;

    Index samples_count = 0;
    separator_char = get_separator_string();
    raw_variables_number = get_has_rows_labels() ? get_raw_variables_number() + 1 : get_raw_variables_number();

    while (getline(file, line))
    {
        line_number++;
        trim(line);
        erase(line, '"');
        if (line.empty()) continue;

        if (count_tokens(line, separator_char) != raw_variables_number)
            throw runtime_error("Line " + to_string(line_number) + ": Incorrect number of tokens.");

        samples_count++;
    }


    file.close();

    data.resize(samples_count, get_raw_variables_number());

    set_default_raw_variables_uses();

    //samples_uses.resize(samples_count);
    //samples_uses.setConstant(SampleUse::Training);

    split_samples_random();


    //ifstream file(data_path.c_str());

    const bool is_float = is_same<type, float>::value;

    const string separator_string = get_separator_string();

    //string line;

    //skip_header(file);

    // Read data

    raw_variables_number = has_sample_ids ? get_raw_variables_number() + 1 : get_raw_variables_number();

    Tensor<string, 1> tokens(raw_variables_number);

    const Index samples_number = data.dimension(0);

    if(has_sample_ids) samples_id.resize(samples_number);

    if(display) cout << "Reading data..." << endl;

    Index sample_index = 0;
    raw_variable_index = 0;

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

        fill_tokens(line, separator_string, tokens);

        for(Index j = 0; j < raw_variables_number; j++)
        {
            trim(tokens(j));

            if(has_sample_ids && j == 0){
                samples_id(sample_index) = tokens(j); continue;
            }
            if(tokens(j) == missing_values_label || tokens(j).empty())
                data(sample_index, raw_variable_index++) = type(NAN);
            else if(is_float)
                data(sample_index, raw_variable_index++) = type(strtof(tokens(j).data(), nullptr));
            else
                data(sample_index, raw_variable_index++) = type(stof(tokens(j)));
        }
        raw_variable_index = 0;
        sample_index++;
    }

    data_file_preview(has_header ? 3 : 2) = tokens;

    file.close();

    if(display) cout << "Data read successfully..." << endl;
}

// void DataSet::read_csv()
// {
//     read_csv_1();

//     if(!has_time_raw_variables() && !has_categorical_raw_variables())
//     {
//         read_csv_2_simple();

//         read_csv_3_simple();
//     }
//     else
//     {
//         read_csv_2_complete();

//         read_csv_3_complete();
//     }
// }


// void LanguageDataSet::read_txt_language_model()
// {
//     cout << "Reading .txt file..." << endl;

//     load_documents(data_path);

//     Index entry_number = documents(0).size();


//     for(Index i = 1; i < documents.size(); i++)
//         entry_number += documents(i).size();

//     Index completion_entry_number = targets(0).size();

//     for(Index i = 1; i < targets.size(); i++)
//         completion_entry_number += targets(i).size();

//     if(entry_number != completion_entry_number)
//         throw runtime_error("Context number of entries (" + to_string(entry_number) + ") not equal to completion number of entries (" + to_string(completion_entry_number) + ").\n");

//     Tensor<string, 1> context(entry_number);

//     Index entry_index = 0;


//     for(Index i = 0; i < documents.size(); i++)
//         for(Index j = 0; j < documents(i).size(); j++)
//             context(entry_index++) = documents(i)(j);


//     Tensor<string, 1> completion(entry_number);

//     entry_index = 0;

//     for(Index i = 0; i < targets.size(); i++)
//         for(Index j = 0; j < targets(i).size(); j++)
//             completion(entry_index++) = targets(i)(j);

//     cout << "Processing documents..." << endl;

//     const Tensor<Tensor<string, 1>, 1> context_tokens = preprocess_language_documents(context);
//     const Tensor<Tensor<string, 1>, 1> completion_tokens = preprocess_language_documents(completion);

//     bool imported_vocabulary = false;

//     if(context_vocabulary_path.empty() || completion_vocabulary_path.empty())
//     {
//         cout << "Calculating vocabularies..." << endl;

//         const Index target_vocabulary_size = 8000;

//         vector<string> reserved_tokens = { "[PAD]", "[UNK]", "[START]", "[END]" };

//         context_vocabulary= calculate_vocabulary(context_tokens, target_vocabulary_size, reserved_tokens);
//         completion_vocabulary= calculate_vocabulary(completion_tokens, target_vocabulary_size, reserved_tokens);
//     }
//     else
//     {
//         cout << "Importing vocabularies..." << endl;

//         //imported_vocabulary = true;
//         import_vocabulary(context_vocabulary_path, context_vocabulary);
//         import_vocabulary(completion_vocabulary_path, completion_vocabulary);
//     }

//     const Index LIMIT = 126;

//     Index max_context_tokens = context_tokens(0).size();

//     for(Index i = 0; i < entry_number; i++)
//         if(context_tokens(i).size() > max_context_tokens)
//             max_context_tokens = context_tokens(i).size();

//     max_context_length = max_context_tokens > LIMIT ? LIMIT : max_context_tokens;

//     Index max_completion_tokens = completion_tokens(0).size();

//     for(Index i = 0; i < entry_number; i++)
//         if(completion_tokens(i).size() > max_completion_tokens)
//             max_completion_tokens = completion_tokens(i).size();

//     max_completion_length = max_completion_tokens > LIMIT + 1 ? LIMIT + 1 : max_completion_tokens;

//     // Output

//     cout << "Writting data file..." << endl;
    
//     string transformed_data_path = data_path;
//     replace(transformed_data_path,".txt","_data.txt");
//     replace(transformed_data_path,".csv","_data.csv");

//     ofstream file;
//     file.open(transformed_data_path);

//     // @todo maybe context does NOT need start and end tokens

//     for(Index i  = type(0); i < max_context_length + 2; i++) // there is start and end indicators
//         file << "context_token_position_" << i << ";";

//     for(Index i  = type(0); i < max_completion_length + 1; i++)
//         file << "input_token_position_" << i << ";";

//     for(Index i  = type(0); i < max_completion_length; i++)
//         file << "target_token_position_" << i << ";";

//     file << "target_token_position_" << max_completion_length << "\n";

//     // Data file preview

//     Index preview_size = 4;

//     text_data_file_preview.resize(preview_size, 2);

//     for(Index i = 0; i < preview_size - 1; i++)
//     {
//         text_data_file_preview(i,0) = context(i);
//         text_data_file_preview(i,1) = completion(i);
//     }

//     text_data_file_preview(preview_size - 1, 0) = context(context.size()-1);
//     text_data_file_preview(preview_size - 1, 1) = completion(completion.size()-1);
    
//     //if(!imported_vocabulary)    write_data_file_whitespace(file, context_tokens, completion_tokens);
//     //else
//     write_data_file_wordpiece(file, context_tokens, completion_tokens);

//     file.close();

//     data_path = transformed_data_path;
//     separator = Separator::Semicolon;
//     has_header = true;


//     read_csv_language_model();

//     set_raw_variable_types(RawVariableType::Numeric);
//     cout<<"Works properly"<<endl;
//     for(Index i = 0; i < max_context_length + 2; i++)
//         set_raw_variable_use(i, VariableUse::Context);

//     for(Index i = 0; i < max_completion_length + 1; i++)
//         set_raw_variable_use(i + max_context_length + 2, VariableUse::Input);

//     for(Index i = 0; i < max_completion_length + 1; i++)
//         set_raw_variable_use(i + max_context_length + max_completion_length + 3, VariableUse::Target);
//     cout<<"Works properly"<<endl;
// }

void LanguageDataSet::read_txt_language_model()
{
    cout << "Reading .txt file..." << endl;

    load_documents(data_path);

    Index entry_number = documents(0).size();

    for(Index i = 1; i < documents.size(); i++)
        entry_number += documents(i).size();

    Index completion_entry_number = targets(0).size();

    for(Index i = 1; i < targets.size(); i++)
        completion_entry_number += targets(i).size();

    if(entry_number != completion_entry_number)
        throw runtime_error("Context number of entries (" + to_string(entry_number) + ") not equal to completion number of entries (" + to_string(completion_entry_number) + ").\n");

    Tensor<string, 1> context(entry_number);

    Index entry_index = 0;

    for(Index i = 0; i < documents.size(); i++)
        for(Index j = 0; j < documents(i).size(); j++)
            context(entry_index++) = documents(i)(j);

    Tensor<string, 1> completion(entry_number);

    entry_index = 0;

    for (Index i = 0; i < targets.size(); i++)
    {
        for (Index j = 0; j < targets(i).size(); j++)
        {
            completion(entry_index) = targets(i)(j);
            entry_index++;
        }
    }

    cout << "Processing documents..." << endl;

    const Tensor<Tensor<string, 1>, 1> context_tokens = preprocess_language_documents(context);
    const Tensor<Tensor<string, 1>, 1> completion_tokens = preprocess_language_documents(completion);


    bool imported_vocabulary = false;

    if (context_vocabulary_path.empty() || completion_vocabulary_path.empty())
    {
        cout << "Calculating vocabularies..." << endl;

        const Index target_vocabulary_size = 8000;
        vector<string> reserved_tokens = { "[PAD]", "[UNK]", "[START]", "[END]" };

        context_vocabulary= calculate_vocabulary(context_tokens, target_vocabulary_size, reserved_tokens);
        completion_vocabulary= calculate_vocabulary(completion_tokens, target_vocabulary_size, reserved_tokens);
    }
    else
    {
        cout << "Importing vocabularies..." << endl;

        imported_vocabulary = true;
        import_vocabulary(context_vocabulary_path, context_vocabulary);
        import_vocabulary(completion_vocabulary_path, completion_vocabulary);
    }

    const Index LIMIT = 126;

    Index max_context_tokens = context_tokens(0).size();

    for(Index i = 0; i < entry_number; i++){
        if(context_tokens(i).size() > max_context_tokens)
            max_context_tokens = context_tokens(i).size();
    }

    max_context_length = max_context_tokens > LIMIT ? LIMIT : max_context_tokens;

    Index max_completion_tokens = completion_tokens(0).size();

    for(Index i = 0; i < entry_number; i++){
        if(completion_tokens(i).size() > max_completion_tokens)
            max_completion_tokens = completion_tokens(i).size();
    }

    max_completion_length = max_completion_tokens > LIMIT + 1 ? LIMIT + 1 : max_completion_tokens;

    // Output

    cout << "Writting data file..." << endl;

    string transformed_data_path = data_path;
    replace(transformed_data_path,".txt","_data.txt");
    replace(transformed_data_path,".csv","_data.csv");

    std::ofstream file;
    file.open(transformed_data_path);

    // @todo maybe context does NOT need start and end tokens

    for(Index i  = type(0); i < max_context_length + 2; i++) /// there is start and end indicators
        file << "context_token_position_" << i << ";";

    for(Index i  = type(0); i < max_completion_length + 1; i++)
        file << "input_token_position_" << i << ";";

    for(Index i  = type(0); i < max_completion_length; i++)
        file << "target_token_position_" << i << ";";
    file << "target_token_position_" << max_completion_length << "\n";

    // Data file preview

    Index preview_size = 4;

    text_data_file_preview.resize(preview_size, 2);

    for(Index i = 0; i < preview_size - 1; i++)
    {
        text_data_file_preview(i,0) = context(i);
        text_data_file_preview(i,1) = completion(i);
    }

    text_data_file_preview(preview_size - 1, 0) = context(context.size()-1);
    text_data_file_preview(preview_size - 1, 1) = completion(completion.size()-1);

    //if (!imported_vocabulary)    write_data_file_whitespace(file, context_tokens, completion_tokens);
    //else
    write_data_file_wordpiece(file, context_tokens, completion_tokens);

    file.close();

    data_path = transformed_data_path;
    separator = Separator::Semicolon;
    bool has_raw_variables_names = true;

    read_csv_language_model();

    set_raw_variable_types(RawVariableType::Numeric);

    for(Index i = 0; i < max_context_length + 2; i++)
        set_raw_variable_use(i, VariableUse::Context);

    for (Index i = 0; i < max_completion_length + 1; i++)
        set_raw_variable_use(i + max_context_length + 2, VariableUse::Input);

    for (Index i = 0; i < max_completion_length + 1; i++)
        set_raw_variable_use(i + max_context_length + max_completion_length + 3, VariableUse::Target);

}


// void LanguageDataSet::write_data_file_whitespace(ofstream& file,
//                                                  const Tensor<Tensor<string, 1>, 1>& context_tokens,
//                                                  const Tensor<Tensor<string, 1>, 1>& completion_tokens)
// {
//     const Index entry_number = context_tokens.dimension(0);

//     const Index context_vocabulary_size = context_vocabulary.size();
//     const Index completion_vocabulary_size = completion_vocabulary.size();

//     Tensor<type, 1> context_row(max_context_length + 2);
//     Tensor<type, 1> completion_row(max_completion_length + 2);

//     Tensor<string, 1> line_tokens;
//     bool line_ended;

//     for(Index i = 0; i < entry_number; i++)
//     {
//         // Context

//         context_row.setZero();
//         context_row(0) = 1; // start indicator

//         line_ended = false;

//         line_tokens = context_tokens(i);

//         for(Index j = 0; j < max_context_length + 1; j++)
//         {
//             if(j < line_tokens.size())
//             {
//                 auto it = find(context_vocabulary.data(), context_vocabulary.data() + context_vocabulary_size, line_tokens(j));

//                 const Index word_index = it - context_vocabulary.data();

//                 context_row(j + 1) = type(word_index);
//             }
//             else
//             {
//                 if(j == line_tokens.size() || (j == max_context_length && !line_ended))
//                 {
//                     context_row(j + 1) = 2; // end indicator
//                     line_ended = true;
//                 }
//                 else
//                 {
//                     context_row(j + 1) = 0; // pad indicator
//                 }
//             }
//         }

//         for(Index j = 0; j < max_context_length + 2; j++)
//             file << context_row(j) << ";";

//         // Completion

//         completion_row.setZero();
//         completion_row(0) = 1;

//         line_ended = false;

//         line_tokens = completion_tokens(i);

//         for(Index j = 0; j < max_completion_length + 1; j++)
//         {
//             if(j < line_tokens.size())
//             {
//                 auto it = find(completion_vocabulary.data(), completion_vocabulary.data() + completion_vocabulary_size, line_tokens(j));

//                 const Index word_index = it - completion_vocabulary.data();

//                 completion_row(j + 1) = type(word_index);
//             }
//             else
//             {
//                 if(j == line_tokens.size() || (j == max_completion_length && !line_ended))
//                 {
//                     completion_row(j + 1) = 2;
//                     line_ended = true;
//                 }
//                 else
//                 {
//                     completion_row(j + 1) = 0;
//                 }
//             }
//         }

//         for(Index j = 0; j < max_completion_length + 1; j++)
//             file << completion_row(j) << ";";

//         for(Index j = 1; j < max_completion_length + 1; j++) // Target is input shifted 1 position to the left
//             file << completion_row(j) << ";";
//         file << completion_row(max_completion_length + 1) << "\n";
//     }
// }


void LanguageDataSet::write_data_file_wordpiece(ofstream& file,
                                                const Tensor<Tensor<string, 1>, 1>& context_tokens,
                                                const Tensor<Tensor<string, 1>, 1>& completion_tokens)
{
    const Index entry_number = context_tokens.dimension(0);

    unordered_map<std::string, type> context_vocabulary_map;
    for(Index i = 0; i < context_vocabulary.size(); i++)    context_vocabulary_map[context_vocabulary(i)] = type(i);

    unordered_map<std::string, type> completion_vocabulary_map;
    for(Index i = 0; i < completion_vocabulary.size(); i++)    completion_vocabulary_map[completion_vocabulary(i)] = type(i);

//    const Index context_vocabulary_size = context_vocabulary.size();
//    const Index completion_vocabulary_size = completion_vocabulary.size();

    Tensor<type, 1> context_row(max_context_length + 2);
    Tensor<type, 1> completion_row(max_completion_length + 2);

    Tensor<string, 1> line_tokens;
    Index token_counter;
    bool line_ended;

    string word;
    string wordpiece;
    string rest;

    auto wordpiece_entry = context_vocabulary_map.find("");

    bool tokenized;

    for(Index i = 0; i < entry_number; i++)
    {        
        // Context

        context_row.setZero();
        context_row(0) = 2; // start indicator

        token_counter = 1;

        line_ended = false;

        line_tokens = context_tokens(i);
        
        for(Index j = 0; j < max_context_length + 1; j++)
        {
            if(j < line_tokens.size() && token_counter < max_context_length + 1)
            {
                word = line_tokens(j);

                wordpiece_entry = context_vocabulary_map.find(word);
                
                if(wordpiece_entry != context_vocabulary_map.end())
                {
                    context_row(token_counter) = wordpiece_entry->second;
                    token_counter++;
                    continue;
                }
                
                tokenized = false;

                for(Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
                {
                    if(token_counter == max_context_length + 1)
                    {
                        tokenized = true;
                        break;
                    }

                    wordpiece = word.substr(0, wordpiece_length);
                    wordpiece_entry = context_vocabulary_map.find(wordpiece);

                    if(wordpiece_entry != context_vocabulary_map.end())
                    {
                        context_row(token_counter) = wordpiece_entry->second;
                        token_counter++;
                        
                        rest = word.substr(wordpiece_length);

                        if(rest.empty())
                        {
                            tokenized = true;
                            break;
                        }

                        word = "##" + rest;
                        wordpiece_length = word.length() + 1;
                    }
                }

                if(!tokenized)
                {
                    context_row(token_counter) = 1; // unknown indicator
                    token_counter++;
                }
            }
            else
            {
                if(token_counter > max_context_length + 1)    break;
                if(j == line_tokens.size() || (token_counter == max_context_length + 1 && !line_ended))
                {
                    context_row(token_counter) = 3; // end indicator
                    token_counter++;
                    line_ended = true;
                }
                else
                {
                    context_row(token_counter) = type(0); // padding
                    token_counter++;
                }
            }
        }
        
        for(Index j = 0; j < max_context_length + 2; j++)
            file << context_row(j) << ";";
        
        
        // Completion

        completion_row.setZero();
        completion_row(0) = 2; // start indicator

        token_counter = 1;

        line_ended = false;

        line_tokens = completion_tokens(i);

        for(Index j = 0; j < max_completion_length + 1; j++)
        {
            if(j < line_tokens.size() && token_counter < max_completion_length + 1)
            {
                word = line_tokens(j);
                
                wordpiece_entry = completion_vocabulary_map.find(word);

                if(wordpiece_entry != completion_vocabulary_map.end())
                {
                    completion_row(token_counter) = wordpiece_entry->second;
                    token_counter++;
                    continue;
                }

                tokenized = false;

                for(Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
                {
                    if(token_counter == max_completion_length + 1)
                    {
                        tokenized = true;
                        break;
                    }

                    wordpiece = word.substr(0, wordpiece_length);
                    wordpiece_entry = completion_vocabulary_map.find(wordpiece);

                    if(wordpiece_entry != completion_vocabulary_map.end())
                    {
                        completion_row(token_counter) = wordpiece_entry->second;
                        token_counter++;

                        rest = word.substr(wordpiece_length);

                        if(rest.empty())
                        {
                            tokenized = true;
                            break;
                        }

                        word = "##" + rest;
                        wordpiece_length = word.length() + 1;
                    }
                }

                if(!tokenized)
                {
                    completion_row(token_counter) = 1; // unknown indicator
                    token_counter++;
                }
            }
            else
            {
                if(token_counter > max_completion_length + 1)    break;

                if(j == line_tokens.size() || (token_counter == max_completion_length + 1 && !line_ended))
                {
                    completion_row(token_counter) = 3; // end indicator
                    token_counter++;
                    line_ended = true;
                }
                else
                {
                    completion_row(token_counter) = 0; // padding
                    token_counter++;
                }
            }
        }

        for(Index j = 0; j < max_completion_length + 1; j++)
            file << completion_row(j) << ";";

        for(Index j = 1; j < max_completion_length + 1; j++) // Target is input shifted 1 position to the left
            file << completion_row(j) << ";";

        file << completion_row(max_completion_length + 1) << "\n";        
    }
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
