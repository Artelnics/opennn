//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "language_data_set.h"
#include "strings_utilities.h"

namespace opennn
{

LanguageDataSet::LanguageDataSet(const dimensions& context_dimensionms, 
                                 const dimensions& completion_dimensionms)
{
    set_context_dimensions(context_dimensionms);
    set_completion_dimensions(completion_dimensionms);
}


LanguageDataSet::LanguageDataSet(const filesystem::path& data_path) : DataSet()
{
    set_data_path(data_path);
    set_separator(DataSet::Separator::Tab);
    read_txt();
    set_raw_variable_scalers(Scaler::None);

    completion_vocabulary = get_completion_vocabulary();
    context_vocabulary = get_context_vocabulary();

    completion_dimensions = {get_completion_length(), get_completion_vocabulary_size()};
    context_dimensions = {get_context_length(), get_context_vocabulary_size()};
}


const vector<string>& LanguageDataSet::get_context_vocabulary() const
{
    return context_vocabulary;
}


const vector<string>& LanguageDataSet::get_completion_vocabulary() const
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
    return maximum_context_length + 2;
}


Index LanguageDataSet::get_completion_length() const
{
    // return maximum_completion_length + 2;
    return maximum_completion_length + 1;
}


const dimensions& LanguageDataSet::get_completion_dimensions() const
{
    return completion_dimensions;
}


const dimensions& LanguageDataSet::get_context_dimensions() const
{
    return context_dimensions;
}


const vector<vector<string>>& LanguageDataSet::get_documents() const
{
    return documents;
}


const vector<vector<string>>& LanguageDataSet::get_targets() const
{
    return targets;
}


void LanguageDataSet::set_default_raw_variables_uses()
{
    DataSet::set_default_raw_variables_uses();

    if(raw_variables.size() > 1)
        context_dimensions.resize(1);
}


void LanguageDataSet::set_raw_variable_uses(const vector<string>& new_raw_variables_uses)
{
    DataSet::set_raw_variable_uses(new_raw_variables_uses);

    context_dimensions = { get_variables_number(DataSet::VariableUse::Context) };
}


void LanguageDataSet::set_raw_variable_uses(const vector<VariableUse>& new_raw_variables_uses)
{
    DataSet::set_raw_variable_uses(new_raw_variables_uses);

    context_dimensions = { get_variables_number(DataSet::VariableUse::Context) };
}


void LanguageDataSet::set_context_dimensions(const dimensions& new_context_dimensions)
{
    context_dimensions = new_context_dimensions;
}


void LanguageDataSet::set_completion_dimensions(const dimensions& new_completion_dimensions)
{
    completion_dimensions = new_completion_dimensions;
}


void LanguageDataSet::set_context_vocabulary_path(const string& new_context_vocabulary_path)
{
    context_vocabulary_path = new_context_vocabulary_path;
}


void LanguageDataSet::set_completion_vocabulary_path(const string& new_completion_vocabulary_path)
{
    completion_vocabulary_path = new_completion_vocabulary_path;
}


void LanguageDataSet::set_context_vocabulary(const vector<string> & new_context_vocabulary)
{
    context_vocabulary = new_context_vocabulary;
}


void LanguageDataSet::set_completion_vocabulary(const vector<string> & new_completion_vocabulary)
{
    completion_vocabulary = new_completion_vocabulary;
}


void LanguageDataSet::set_data_random()
{
/*
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
*/
}


void LanguageDataSet::set_default()
{
    DataSet::set_default();

    context_dimensions = { get_variables_number(DataSet::VariableUse::Context) };
}


void LanguageDataSet::to_XML(XMLPrinter& printer) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    printer.OpenElement("DataSet");

    // Data file

    printer.OpenElement("DataSource");

    add_xml_element(printer, "FileType", "csv");
    add_xml_element(printer, "DataPath", data_path.string());
    add_xml_element(printer, "Separator", get_separator_string());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "Codification", get_codification_string());
    printer.CloseElement();

    const Index raw_variables_number = get_raw_variables_number();

    printer.OpenElement("RawVariables");
    add_xml_element(printer, "RawVariablesNumber", to_string(raw_variables_number));

    for(Index i = 0; i < raw_variables_number; i++)
    {
        printer.OpenElement("RawVariable");
        printer.PushAttribute("Item", to_string(i+1).c_str());
        raw_variables[i].to_XML(printer);
        printer.CloseElement();
    }

    printer.CloseElement();

    // Samples id

    if(has_sample_ids)
    {
        const Index rows_labels_number = sample_ids.size();

        printer.OpenElement("HasSamplesId");

        buffer.str("");

        for(Index i = 0; i < rows_labels_number; i++)
        {
            buffer << sample_ids[i];

            if(i != rows_labels_number-1) buffer << ",";
        }

        printer.PushText(buffer.str().c_str());

        printer.CloseElement();
    }

    // Samples

    printer.OpenElement("Samples");

    add_xml_element(printer, "SamplesNumber", to_string(get_samples_number()));


    // Samples uses

    {
        printer.OpenElement("SamplesUses");

        buffer.str("");

        const Index samples_number = get_samples_number();

        for(Index i = 0; i < samples_number; i++)
        {
            SampleUse sample_use = sample_uses[i];

            buffer << Index(sample_use);

            if(i < (samples_number-1)) buffer << " ";
        }

        printer.PushText(buffer.str().c_str());

        printer.CloseElement();
    }

    // Close samples

    printer.CloseElement();

    // Missing values

    printer.OpenElement("MissingValues");

    // Missing values method

    {
        printer.OpenElement("MissingValuesMethod");

        if(missing_values_method == MissingValuesMethod::Mean)
            printer.PushText("Mean");
        else if(missing_values_method == MissingValuesMethod::Median)
            printer.PushText("Median");
        else if(missing_values_method == MissingValuesMethod::Unuse)
            printer.PushText("Unuse");
        else if(missing_values_method == MissingValuesMethod::Interpolation)
            printer.PushText("Interpolation");

        printer.CloseElement();
    }

    // Missing values number

    {
        printer.OpenElement("MissingValuesNumber");

        buffer.str("");
        buffer << missing_values_number;

        printer.PushText(buffer.str().c_str());

        printer.CloseElement();
    }

    if(missing_values_number > 0)
    {
        // Raw variables missing values number
        {
            printer.OpenElement("RawVariablesMissingValuesNumber");

            buffer.str("");

            for(Index i = 0; i < raw_variables_number; i++)
            {
                buffer << raw_variables_missing_values_number(i);

                if(i != raw_variables_number - 1) buffer << " ";
            }

            printer.PushText(buffer.str().c_str());

            printer.CloseElement();
        }

        // Rows missing values number
        {
            printer.OpenElement("RowsMissingValuesNumber");

            buffer.str("");
            buffer << rows_missing_values_number;

            printer.PushText(buffer.str().c_str());

            printer.CloseElement();
        }
    }

    // Missing values

    printer.CloseElement();

    // Preview data

    printer.OpenElement("PreviewData");

    printer.OpenElement("PreviewSize");

    buffer.str("");

    if(model_type != ModelType::TextClassification)
    {
        buffer << data_file_preview.size();

        printer.PushText(buffer.str().c_str());

        printer.CloseElement();

        for(size_t i = 0; i < data_file_preview.size(); i++)
        {
            printer.OpenElement("Row");

            printer.PushAttribute("Item", to_string(i+1).c_str());

            for(size_t j = 0; j < data_file_preview[i].size(); j++)
            {
                printer.PushText(data_file_preview[i][j].c_str());

                if(j != data_file_preview[i].size()-1)
                    printer.PushText(",");
            }

            printer.CloseElement();
        }
    }
    else
    {
        buffer << data_file_preview.size();

        printer.PushText(buffer.str().c_str());

        printer.CloseElement();

        for(size_t i = 0; i < data_file_preview.size(); i++)
        {
            printer.OpenElement("Row");
            printer.PushAttribute("Item", to_string(i+1).c_str());
            printer.PushText(data_file_preview[i][0].c_str());
            printer.CloseElement();
        }

        for(size_t i = 0; i < data_file_preview.size(); i++)
        {
            printer.OpenElement("Target");
            printer.PushAttribute("Item", to_string(i+1).c_str());
            printer.PushText(data_file_preview[i][1].c_str());
            printer.CloseElement();
        }
    }

    // Close preview data

    printer.CloseElement();

    // Completion Vocabulary

    printer.OpenElement("CompletionVocabulary");
    for (const auto& word : completion_vocabulary) {
        printer.OpenElement("Word");
        printer.PushText(word.c_str());
        printer.CloseElement();
    }

    printer.CloseElement();

    // Context Vocabulary
    printer.OpenElement("ContextVocabulary");

    for (const auto& word : context_vocabulary) 
    {
        printer.OpenElement("Word");
        printer.PushText(word.c_str());
        printer.CloseElement();
    }

    printer.CloseElement();

    // Completion Dimensions
    printer.OpenElement("CompletionDimensions");
    printer.PushText(to_string(maximum_completion_length).c_str());
    printer.CloseElement();

    // Context Dimensions
    printer.OpenElement("ContextDimensions");
    printer.PushText(to_string(maximum_context_length).c_str());
    printer.CloseElement();

    // Close data set

    printer.CloseElement();

    time(&finish);
}


void LanguageDataSet::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    const XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data file element is nullptr.\n");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_separator_string(read_xml_string(data_source_element, "Separator"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_xml_string(data_source_element, "Codification"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));

    // RawVariables

    const XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    // Raw variables number

    const XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
        throw runtime_error("RawVariablesNumber element is nullptr.\n");

    Index new_raw_variables_number = 0;

    if(raw_variables_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(raw_variables_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // Raw variables

    const XMLElement* start_element = raw_variables_number_element;

    for(Index i = 0; i < new_raw_variables_number; i++)
    {
        const XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Raw_variable item number (" + to_string(i + 1) + ") exception.\n");

        RawVariable& raw_variable = raw_variables[i];
        raw_variable.name = read_xml_string(raw_variable_element, "Name");
        raw_variable.set_scaler(read_xml_string(raw_variable_element, "Scaler"));
        raw_variable.set_use(read_xml_string(raw_variable_element, "Use"));
        raw_variable.set_type(read_xml_string(raw_variable_element, "Type"));


        if(raw_variables[i].type == RawVariableType::Categorical || raw_variables[i].type == RawVariableType::Binary)
            raw_variable.categories = get_tokens(read_xml_string(raw_variable_element, "Categories"), ";");

        //        raw_variable_element = raw_variable_element->NextSiblingElement("RawVariable");
    }

    // Rows label

    if(has_sample_ids)
        sample_ids = get_tokens(read_xml_string(data_set_element, "SamplesId"), ",");

    // Samples

    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    // Samples number

    const XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
        throw runtime_error("Samples number element is nullptr.\n");

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = Index(atoi(samples_number_element->GetText()));

        sample_uses.resize(new_samples_number);

        set(DataSet::SampleUse::Training);
    }

    // Samples uses

    const XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
        throw runtime_error("Samples uses element is nullptr.\n");

    if(samples_uses_element->GetText())
        set_sample_uses(get_tokens(samples_uses_element->GetText(), " "));

    // Missing values

    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    // Missing values method

    const XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
        throw runtime_error("Missing values method element is nullptr.\n");

    if(missing_values_method_element->GetText())
        set_missing_values_method(missing_values_method_element->GetText());

    // Missing values number

    const XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
        throw runtime_error("Missing values number element is nullptr.\n");

    if(missing_values_number_element->GetText())
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));

    if(missing_values_number > 0)
    {
        // Raw variables Missing values number

        const XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("");

        if(!raw_variables_missing_values_number_element)
            throw runtime_error("RawVariablesMissingValuesNumber element is nullptr.\n");

        if(raw_variables_missing_values_number_element->GetText())
        {
            const vector<string> new_raw_variables_missing_values_number
                = get_tokens(raw_variables_missing_values_number_element->GetText(), " ");

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(size_t i = 0; i < new_raw_variables_missing_values_number.size(); i++)
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number[i].c_str());
        }

        // Rows missing values number

        const XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
            throw runtime_error("Rows missing values number element is nullptr.\n");

        if(rows_missing_values_number_element->GetText())
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
    }

    // Preview data

    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n");

    // Preview size

    const XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
        throw runtime_error("Preview size element is nullptr.\n");

    Index new_preview_size = 0;

    if(preview_size_element->GetText())
    {
        new_preview_size = Index(atoi(preview_size_element->GetText()));

        if(new_preview_size > 0)
            data_file_preview.resize(new_preview_size);
    }

    // Preview data

    start_element = preview_size_element;

    if(model_type != ModelType::TextClassification)
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                data_file_preview[i] = get_tokens(row_element->GetText(), ",");
        }
    }
    else
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                data_file_preview[i][0] = row_element->GetText();
        }

        for(Index i = 0; i < new_preview_size; i++)
        {
            const XMLElement* row_element = start_element->NextSiblingElement("Target");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
                throw runtime_error("Target item number (" + to_string(i + 1) + ") exception.\n");

            if(row_element->GetText())
                data_file_preview[i][1] = row_element->GetText();
        }
    }

    // Completion Vocabulary

    const XMLElement* completion_vocabulary_element = data_set_element->FirstChildElement("CompletionVocabulary");
    if(completion_vocabulary_element)
    {
        completion_vocabulary.clear();
        for (const XMLElement* word_element = completion_vocabulary_element->FirstChildElement(); word_element; word_element = word_element->NextSiblingElement())
        {
            if (word_element->GetText())
                completion_vocabulary.push_back(word_element->GetText());
        }
    }

    // Context Vocabulary

    const XMLElement* context_vocabulary_element = data_set_element->FirstChildElement("ContextVocabulary");
    if(context_vocabulary_element)
    {
        context_vocabulary.clear();

        for (const XMLElement* word_element = context_vocabulary_element->FirstChildElement(); word_element; word_element = word_element->NextSiblingElement())
        {
            if (word_element->GetText())
                context_vocabulary.push_back(word_element->GetText());
        }
    }

    // Context Dimensions

    const XMLElement* completion_dimensions_element = data_set_element->FirstChildElement("CompletionDimensions");
    if (completion_dimensions_element && completion_dimensions_element->GetText())
        maximum_completion_length = atoi(completion_dimensions_element->GetText());

    // Context Dimensions

    const XMLElement* context_dimensions_element = data_set_element->FirstChildElement("ContextDimensions");
    if (context_dimensions_element && context_dimensions_element->GetText()) {
        maximum_context_length = atoi(context_dimensions_element->GetText());
    }

    // Display

    // set_display(read_xml_bool(data_set_element, "Display"));
}


/*
void LanguageDataSet::save_vocabulary(const filesystem::path& path, const vector<string>& vocabulary)
{
    ofstream file(path.c_str());

    if (!file.is_open())
        throw runtime_error("Cannot open file to save vocabulary: " + path.string() + "\n");

    for (const auto& word : vocabulary)
        file << word << '\n';

    file.close();
}
*/



void LanguageDataSet::import_vocabulary(const filesystem::path& path, 
                                        vector<string>& vocabulary)
{
    ifstream file(path.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open vocabulary file: " + path.string() + "\n");

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

        vocabulary[count++] = line;

        if(file.peek() == EOF) break;
    }
}


/*
void LanguageDataSet::save_lengths(const filesystem::path& path, const Index& input_length, const Index& context_length)
{
    ofstream file(path.c_str());

    if (!file.is_open())
        throw runtime_error("Cannot open file to export lengths: " + path.string() + "\n");

    file << input_length << '\n';
    file << context_length << '\n';

    file.close();
}
*/


void LanguageDataSet::import_lengths(const filesystem::path& path, Index& input_length, Index& context_length)
{
    ifstream file(path.c_str());

    if (!file.is_open())
        throw runtime_error("Cannot open file to import lengths: " + path.string() + "\n");

    file >> input_length
         >> context_length;

    file.close();
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


set<char> extract_character_tokens(const vector<pair<string, Index>>& word_counts)
{
    set<char> seen_chars;

    for(const auto& [word, _] : word_counts)
        for(char c : word)
            seen_chars.insert(c);

    return seen_chars;
}


map<string, int> ensure_all_tokens_exist(const set<string>& input_tokens,
                                         map<string, int> output_tokens,
                                         const bool& include_joiner_token,
                                         const string& joiner)
{
    for(const string& token : input_tokens)
    {
        if(output_tokens.find(token) == output_tokens.end())
            output_tokens[token] = 1;

        if(include_joiner_token)
        {
            const string joined_token = joiner + token;

            if(output_tokens.find(joined_token) == output_tokens.end())
                output_tokens[joined_token] = 1;
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
    size_t start = 0;

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
            return {};

        start = end;
    }

    return indices;
}


tuple<Index, Index> calculate_thresholds(const vector<pair<string, Index>>& word_counts, 
                                         const Index& upper_threshold, 
                                         const Index& lower_threshold)
{
    vector<int> counts;

    for(const auto& [_, count] : word_counts)
        counts.push_back(count);

    const Index max_count = *max_element(counts.begin(), counts.end());
    const Index min_count = *min_element(counts.begin(), counts.end());

    const Index upper_search = upper_threshold == -1 
        ? max_count 
        : min(upper_threshold, max_count);

    const Index lower_search = lower_threshold == -1 
        ? min_count 
        : max(lower_threshold, min_count);

    return { upper_search, lower_search };
}


// @todo move to strings utilities?

vector<pair<string, Index>> trim_inputs(const vector<pair<string, Index>>& word_counts,
                                        const vector<string>& reserved_tokens,
                                        const Index& max_token_length)
{
    vector<pair<string, Index>> trimmed_counts;

    for(const auto& [word, count] : word_counts)
    {
        if(word.size() > max_token_length
        || find(reserved_tokens.begin(), reserved_tokens.end(), word) != reserved_tokens.end())
            continue;

        trimmed_counts.push_back({ word, count });
    }

    return trimmed_counts;
}


set<char> get_allowed_characters(const vector<pair<string, Index>>& trimmed_counts, 
                                 const Index& max_unique_characters)
{
    map<char, Index> character_counts;

    for(const auto& [word, count] : trimmed_counts)
        for(char c : word)
            character_counts[c] += count;

    vector<pair<char, Index>> sorted_counts(character_counts.begin(), character_counts.end());

    sort(sorted_counts.begin(), sorted_counts.end(), [](const pair<char, Index>& a, const pair<char, Index>& b)
    {
        if(a.second != b.second)
            return a.second > b.second;

        return a.first < b.first;
    }
);

    set<char> allowed_characters;

    for(int i = 0; i < min((Index)sorted_counts.size(), max_unique_characters); i++)
        allowed_characters.insert(sorted_counts[i].first);

    return allowed_characters;
}


vector<pair<string, Index>> filter_inputs(const vector<pair<string, Index>>& trimmed_counts, 
                                          const set<char>& allowed_characters, 
                                          const Index& max_input_tokens)
{
    vector<pair<string, Index>> sorted_counts = trimmed_counts;

    sort(sorted_counts.begin(), sorted_counts.end(), [](const pair<string, Index>& a, const pair<string, Index>& b)
    {
        return a.second > b.second;
    }
);

    vector<pair<string, Index>> filtered_counts;

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
            continue;

        filtered_counts.push_back({ word, count });
    }

    return filtered_counts;
}


vector<string> generate_override_vocabulary(const vector<string>& reserved_tokens,
                                         const set<char>& character_tokens,
                                         const map<string, int>& current_tokens)
{
    vector<string> vocabulary;
    vocabulary.insert(vocabulary.end(), reserved_tokens.begin(), reserved_tokens.end());

    vector<string> sorted_character_tokens;
    for(const char ch : character_tokens)
        sorted_character_tokens.push_back(string(1, ch));

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
        vocabulary.push_back(token);

    set<string> seen_tokens;
    vector<string> override_vocabulary;

    for(const string& word : vocabulary)
    {
        if(seen_tokens.find(word) == seen_tokens.end())
        {
            seen_tokens.insert(word);
            override_vocabulary.push_back(word);
        }
    }

    return override_vocabulary;
}


vector<string> calculate_vocabulary_with_threshold(const vector<pair<string, Index>>& word_counts,
                                                   const Index& threshold,
                                                   const WordpieceAlgorithmParameters& parameters)
{
    const set<char> character_tokens = extract_character_tokens(word_counts);

    set<string> string_tokens;

    for(const char ch : character_tokens)
        string_tokens.insert(string(1, ch));

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
                
                if(split_indices.empty())
                    continue;
            }

            size_t start = 0;

            for(int split_index : split_indices)
            {
                for(size_t end = start + 1; end <= word.size(); ++end)
                {
                    string subtoken = word.substr(start, end - start);
                    const Index length = subtoken.size();

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
                if(count >= threshold)
                    next_tokens[token] = count;

                if(token.size() > length)
                {
                    const size_t joiner_length = parameters.joiner.size();

                    for(size_t i = 1 + joiner_length; i <= length + joiner_length; i++)
                    {
                        const string prefix = token.substr(0, i);

                        if(subtokens[i - joiner_length].find(prefix) != subtokens[i - joiner_length].end())
                            subtokens[i - joiner_length][prefix] -= count;
                    }
                }
                else
                {
                    for(size_t i = 1; i < length; i++)
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

    return generate_override_vocabulary(parameters.reserved_tokens, character_tokens, current_tokens);
}


vector<string> calculate_vocabulary_binary_search(const vector<pair<string, Index>>& word_counts,
                                                  const Index& lower_bound,
                                                  const Index& upper_bound,
                                                  const WordpieceAlgorithmParameters& parameters)
{
    const int threshold = (upper_bound + lower_bound) / 2;

    const vector<string> current_vocabulary = calculate_vocabulary_with_threshold(word_counts, 
                                                                                  threshold, 
                                                                                  parameters);

    const Index current_vocabulary_size = current_vocabulary.size();

    const Index slack = max(0, int(parameters.slack_ratio * parameters.vocabulary_size));

    const bool is_within_slack = current_vocabulary_size <= parameters.vocabulary_size 
                              && parameters.vocabulary_size - current_vocabulary_size <= slack;

    if(is_within_slack || lower_bound >= upper_bound || threshold <= 1)
        return current_vocabulary;

    if(current_vocabulary_size > parameters.vocabulary_size)
        return calculate_vocabulary_binary_search(word_counts, threshold + 1, upper_bound, parameters);
    else
        return calculate_vocabulary_binary_search(word_counts, lower_bound, threshold - 1, parameters);
}


vector<string> LanguageDataSet::calculate_vocabulary(const vector<vector<string>>& tokens,
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

    const vector<string> total_tokens = tokens_list(tokens);

    const vector<pair<string, Index>> word_counts = count_words(total_tokens);

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

    const vector<pair<string, Index>> trimmed_counts = trim_inputs(word_counts, parameters.reserved_tokens, parameters.max_token_length);

    const std::set<char> allowed_characters = get_allowed_characters(trimmed_counts, parameters.max_unique_characters);

    const vector<pair<string, Index>> filtered_counts = filter_inputs(trimmed_counts, allowed_characters, parameters.max_input_tokens);

    const vector<string> vocabulary = calculate_vocabulary_binary_search(filtered_counts, lower_search, upper_search, parameters);

//    vector<string> vocabulary_tensor(vocabulary.size());

//    std::copy(vocabulary.begin(), vocabulary.end(), vocabulary_tensor.begin());

//    return vocabulary_tensor;

    return vocabulary;
}


void LanguageDataSet::load_documents(const filesystem::path& path)
{
    ifstream file(path);

    if(!file.is_open())
        throw runtime_error("Cannot open document file: " + path.string() + "\n");

    const Index original_size = documents.size();

    documents.resize(original_size + 1);
    targets.resize(original_size + 1);

    Index lines_count = 0;
    Index lines_number = 0;

    string line;

    while(getline(file, line))
    {
        prepare_line(line);

        if(!line.empty())
            lines_number++;
    }

    vector<string> document(lines_number);
    vector<string> document_target(lines_number);

    file.clear();
    file.seekg(0, ios::beg);

    const string separator_string = get_separator_string();

    Index tokens_number = 0;

    string delimiter;

    while(getline(file, line))
    {
        if(line.empty()) continue;

        if(line[0] == '"')
        {
            replace(line,"\"\"", "\"");
            line = "\"" + line;
            delimiter = "\"\"";
        }

        if(line.find("\"" + separator_string) != string::npos)
            replace(line,"\"" + separator_string, "\"\"" + separator_string);

        const vector<string> tokens = get_tokens(line, delimiter + separator_string);

        tokens_number = tokens.size();

        if(tokens_number == 1)
        {
            document[lines_count++] += tokens[0].find(delimiter) == 0
                ? tokens[0].substr(delimiter.length())
                : " " + tokens[0];
        }
        else if(tokens_number == 2)
        {
            if(tokens[0].empty() && tokens[1].empty())
                continue;

            document[lines_count] += " " + tokens[0];
            document_target[lines_count] += tokens[1];
            delimiter.clear();
            lines_count++;
        }
        else if (tokens_number > 2)
        {
            throw runtime_error("Found more than one separator in line: " + line + "\n");
        }

        if(file.peek() == EOF)
            break;
    }

    document.resize(lines_count);
    document_target.resize(lines_count);

    documents[original_size] = move(document);
    targets[original_size] = move(document_target);

    file.close();
}


void LanguageDataSet::read_csv_3_language_model()
{
    ifstream file(data_path);

    const bool is_float = is_same<type, float>::value;

    const string separator_string = get_separator_string();

    string line;

    //skip_header(file);

    // Read data

    const Index raw_variables_number = has_sample_ids ? get_raw_variables_number() + 1 : get_raw_variables_number();

    vector<string> tokens(raw_variables_number);

    const Index samples_number = data.dimension(0);

    if(has_sample_ids) sample_ids.resize(samples_number);

    if(display) cout << "Reading data..." << endl;

    Index sample_index = 0;
    Index raw_variable_index = 0;

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

        fill_tokens(line, separator_string, tokens);

        for(Index j = 0; j < raw_variables_number; j++)
        {
            trim(tokens[j]);

            if(has_sample_ids && j == 0)
            {
                sample_ids[sample_index] = tokens[j];

                continue;
            }

            if(tokens[j] == missing_values_label || tokens[j].empty())
                data(sample_index, raw_variable_index) = type(NAN);
            else if(is_float)
                data(sample_index, raw_variable_index) = type(strtof(tokens[j].data(), nullptr));
            else
                data(sample_index, raw_variable_index) = type(stof(tokens[j]));

            raw_variable_index++;
        }

        raw_variable_index = 0;
        sample_index++;
    }

    const Index data_file_preview_index = has_header ? 3 : 2;

    data_file_preview[data_file_preview_index] = tokens;

    file.close();

    if(display) cout << "Data read successfully..." << endl;
}


void LanguageDataSet::read_csv_1()
{
    if(display) cout << "Path: " << data_path << endl;

    if(data_path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv() method.\n"
               << "Data file name is empty.\n";

        throw runtime_error(buffer.str());
    }

    regex accent_regex("[\\xC0-\\xFF]");
    ifstream file(data_path);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv() method.\n"
               << "Cannot open data file: " << data_path << "\n";

        throw runtime_error(buffer.str());
    }

    const string separator_char = get_separator_string();

    if(display) cout << "Setting data file preview..." << endl;

    Index lines_number = has_binary_raw_variables()? 4 : 3;

    data_file_preview.resize(lines_number);

    string line;

    Index lines_count = 0;

    while(file.good())
    {
        getline(file, line);

        decode(line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        check_separators(line);

        data_file_preview[lines_count] = get_tokens(line, separator_char);

        lines_count++;

        if(lines_count == lines_number) break;
    }

    file.close();

    // Check empty file

    if(data_file_preview[0].size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "File " << data_path << " is empty.\n";

        throw runtime_error(buffer.str());
    }

    // Set rows labels and raw_variables names

    if(display) cout << "Setting rows labels..." << endl;

    string first_name = data_file_preview[0][0];
    transform(first_name.begin(), first_name.end(), first_name.begin(), ::tolower);

    const Index raw_variables_number = get_has_rows_labels() ? data_file_preview[0].size()-1 : data_file_preview[0].size();

    raw_variables.resize(raw_variables_number);

    // Check if header has numeric value

    if(has_binary_raw_variables() && has_numbers(data_file_preview[0]))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "Some raw_variables names are numeric.\n";

        throw runtime_error(buffer.str());
    }

    // raw_variables names

    if(display) cout << "Setting raw_variables names..." << endl;

    if(has_binary_raw_variables())
    {
/*
        get_has_rows_labels() ? set_raw_variable_names(data_file_preview[0].slice(Eigen::array<Eigen::Index, 1>({1}),
                                                                                  Eigen::array<Eigen::Index, 1>({data_file_preview[0].size()-1})))
                              : set_raw_variable_names(data_file_preview[0]);
*/
    }
    else
    {
        set_raw_variable_names(get_default_raw_variables_names(raw_variables_number));
    }

    // Check raw_variables with all missing values

    bool has_nans_raw_variables = false;

    do
    {
        has_nans_raw_variables = false;

        if(lines_number > 10)
            break;

        for(size_t i = 0; i < data_file_preview[0].size(); i++)
        {
            if(get_has_rows_labels() && i == 0) continue;

            // Check if all are missing values

            if(data_file_preview[1][i] == missing_values_label
            && data_file_preview[2][i] == missing_values_label
            && data_file_preview[lines_number-2][i] == missing_values_label
            && data_file_preview[lines_number-1][i] == missing_values_label)
                has_nans_raw_variables = true;
            else
                has_nans_raw_variables = false;

            if(has_nans_raw_variables)
            {
                lines_number++;
                data_file_preview.resize(lines_number);

                lines_count = 0;

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
                    data_file_preview[lines_count] = get_tokens(line, separator_char);
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

    for(size_t i = 0; i < data_file_preview[0].size(); i++)
    {
        if(get_has_rows_labels() && i == 0) continue;

        string data_file_preview_1 = data_file_preview[1][i];
        string data_file_preview_2 = data_file_preview[2][i];
        string data_file_preview_3 = data_file_preview[lines_number-2][i];
        string data_file_preview_4 = data_file_preview[lines_number-1][i];

        /*        if(nans_columns(column_index))
        {
            columns(column_index).type = ColumnType::Constant;
            column_index++;
        }
        else*/ if((is_date_time_string(data_file_preview_1) && data_file_preview_1 != missing_values_label)
            || (is_date_time_string(data_file_preview_2) && data_file_preview_2 != missing_values_label)
            || (is_date_time_string(data_file_preview_3) && data_file_preview_3 != missing_values_label)
            || (is_date_time_string(data_file_preview_4) && data_file_preview_4 != missing_values_label))
        {
            raw_variables[raw_variable_index].type = RawVariableType::DateTime;
            //            time_column = raw_variables[raw_variable_index].name;
            raw_variable_index++;
        }
        else if(((is_numeric_string(data_file_preview_1) && data_file_preview_1 != missing_values_label) || data_file_preview_1.empty())
                 || ((is_numeric_string(data_file_preview_2) && data_file_preview_2 != missing_values_label) || data_file_preview_2.empty())
                 || ((is_numeric_string(data_file_preview_3) && data_file_preview_3 != missing_values_label) || data_file_preview_3.empty())
                 || ((is_numeric_string(data_file_preview_4) && data_file_preview_4 != missing_values_label) || data_file_preview_4.empty()))
        {
            raw_variables[raw_variable_index].type = RawVariableType::Numeric;
            raw_variable_index++;
        }
        else
        {
            raw_variables[raw_variable_index].type = RawVariableType::Categorical;
            raw_variable_index++;
        }
    }

    // Resize data file preview to original

    if(data_file_preview.size() > 4)
    {
        lines_number = has_binary_raw_variables() ? 4 : 3;

        vector<vector<string>> data_file_preview_copy(data_file_preview);

        data_file_preview.resize(lines_number);

        data_file_preview[0] = data_file_preview_copy[1];
        data_file_preview[1] = data_file_preview_copy[1];
        data_file_preview[2] = data_file_preview_copy[2];
        data_file_preview[lines_number - 2] = data_file_preview_copy[data_file_preview_copy.size()-2];
        data_file_preview[lines_number - 1] = data_file_preview_copy[data_file_preview_copy.size()-1];
    }
}


void LanguageDataSet::read_csv_2_simple()
{
    regex accent_regex("[\\xC0-\\xFF]");
    ifstream file(data_path);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_2_simple() method.\n"
               << "Cannot open data file: " << data_path << "\n";

        throw runtime_error(buffer.str());
    }

    string line;
    Index line_number = 0;

    if(has_binary_raw_variables())
    {
        while(file.good())
        {
            line_number++;

            getline(file, line);

            trim(line);

            erase(line, '"');

            if(line.empty()) continue;

            break;
        }
    }

    Index samples_count = 0;

    Index tokens_count;

    if(display) cout << "Setting data dimensions..." << endl;

    const string separator_string = get_separator_string();

    const Index raw_variables_number = get_raw_variables_number();
    const Index raw_raw_variables_number = get_has_rows_labels() ? raw_variables_number + 1 : raw_variables_number;

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens_count = count_tokens(line, separator_string);

        if(tokens_count != raw_raw_variables_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void read_csv_2_simple() method.\n"
                   << "Line " << line_number << ": Size of tokens("
                   << tokens_count << ") is not equal to number of raw_variables("
                   << raw_raw_variables_number << ").\n";

            throw runtime_error(buffer.str());
        }

        samples_count++;
    }

    file.close();

    data.resize(samples_count, raw_variables_number);

    set_default_raw_variables_uses();

    sample_uses.resize(samples_count);
    // sample_uses.set(SampleUse::Training);

    split_samples_random();
}


void LanguageDataSet::read_csv()
{
    read_csv_1();

    read_csv_2_simple();

    read_csv_3_language_model();
}

// void LanguageDataSet::read_txt()
// {
//     cout << "Reading .txt file..." << endl;

//     load_documents(data_path);

//     Index entry_number = documents(0).size();


//     for(Index i = 1; i < documents.size(); i++)
//         entry_number += documents[i].size();

//     Index completion_entry_number = targets(0).size();

//     for(Index i = 1; i < targets.size(); i++)
//         completion_entry_number += targets(i).size();

//     if(entry_number != completion_entry_number)
//         throw runtime_error("Context number of entries (" + to_string(entry_number) + ") not equal to completion number of entries (" + to_string(completion_entry_number) + ").\n");

//     vector<string> context(entry_number);

//     Index entry_index = 0;

//     for(Index i = 0; i < documents.size(); i++)
//         for(Index j = 0; j < documents[i].size(); j++)
//             context(entry_index++) = documents[i][j];


//     vector<string> completion(entry_number);

//     entry_index = 0;

//     for(Index i = 0; i < targets.size(); i++)
//         for(Index j = 0; j < targets(i).size(); j++)
//             completion(entry_index++) = targets(i)(j);

//     cout << "Processing documents..." << endl;

//     const vector<vector<string>> context_tokens = preprocess_language_documents(context);
//     const vector<vector<string>> completion_tokens = preprocess_language_documents(completion);

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

//     Index maximum_context_tokens = context_tokens[0].size();

//     for(Index i = 0; i < entry_number; i++)
//         if(context_tokens[i].size() > maximum_context_tokens)
//             maximum_context_tokens = context_tokens[i].size();

//     maximum_context_length = maximum_context_tokens > LIMIT ? LIMIT : maximum_context_tokens;

//     Index maximum_completion_tokens = completion_tokens[0].size();

//     for(Index i = 0; i < entry_number; i++)
//         if(completion_tokens[i].size() > maximum_completion_tokens)
//             maximum_completion_tokens = completion_tokens[i].size();

//     maximum_completion_length = maximum_completion_tokens > LIMIT + 1 ? LIMIT + 1 : maximum_completion_tokens;

//     // Output

//     cout << "Writting data file..." << endl;

//     string transformed_data_path = data_path;
//     replace(transformed_data_path,".txt","_data.txt");
//     replace(transformed_data_path,".csv","_data.csv");

//     ofstream file(transformed_data_path);

     // @todo maybe context does NOT need start and end tokens

//     for(Index i  = type(0); i < maximum_context_length + 2; i++) // there is start and end indicators
//         file << "context_token_position_" << i << ";";

//     for(Index i  = type(0); i < maximum_completion_length + 1; i++)
//         file << "input_token_position_" << i << ";";

//     for(Index i  = type(0); i < maximum_completion_length; i++)
//         file << "target_token_position_" << i << ";";

//     file << "target_token_position_" << maximum_completion_length << "\n";

//     // Data file preview

//     Index preview_size = 4;

//     data_file_preview.resize(preview_size, 2);

//     for(Index i = 0; i < preview_size - 1; i++)
//     {
//         data_file_preview[i][0] = context[i];
//         data_file_preview[i][1] = completion[i];
//     }

//     data_file_preview(preview_size - 1, 0) = context(context.size()-1);
//     data_file_preview(preview_size - 1, 1) = completion(completion.size()-1);

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
//     for(Index i = 0; i < maximum_context_length + 2; i++)
//         set_raw_variable_use(i, VariableUse::Context);

//     for(Index i = 0; i < maximum_completion_length + 1; i++)
//         set_raw_variable_use(i + maximum_context_length + 2, VariableUse::Input);

//     for(Index i = 0; i < maximum_completion_length + 1; i++)
//         set_raw_variable_use(i + maximum_context_length + maximum_completion_length + 3, VariableUse::Target);
//     cout<<"Works properly"<<endl;
// }


void LanguageDataSet::read_txt()
{
    cout << "Reading .txt file..." << endl;

    load_documents(data_path.string());

    const size_t documents_number = documents.size();

    const size_t entry_number = accumulate(documents.begin(), documents.end(), 0,
                                [](size_t sum, const vector<string>& doc) { return sum + doc.size(); });

    const size_t completion_entry_number = accumulate(targets.begin(), targets.end(), 0,
                                           [](size_t sum, const vector<string>& target) { return sum + target.size(); });

    if(entry_number != completion_entry_number)
        throw runtime_error("Entry number (" + to_string(entry_number) + ") not equal to completion entry number (" + to_string(completion_entry_number) + ").\n");

    vector<string> context(entry_number);

    Index context_index = 0;

    for(size_t i = 0; i < documents_number; i++)
        for(size_t j = 0; j < documents[i].size(); j++)
            context[context_index++] = documents[i][j];

    vector<string> completion(entry_number);

    Index completion_index = 0;

    for (size_t i = 0; i < targets.size(); i++)
        for (size_t j = 0; j < targets[i].size(); j++)
            completion[completion_index++] = targets[i][j];

    cout << "Processing documents..." << endl;

    const vector<vector<string>> context_tokens = preprocess_language_documents(context);
    const vector<vector<string>> completion_tokens = preprocess_language_documents(completion);

    if (context_vocabulary_path.empty() || completion_vocabulary_path.empty())
    {
        cout << "Calculating vocabularies..." << endl;

        const Index target_vocabulary_size = 8000;
        const vector<string> reserved_tokens = { "[PAD]", "[UNK]", "[START]", "[END]" };

        context_vocabulary = calculate_vocabulary(context_tokens, target_vocabulary_size, reserved_tokens);
        completion_vocabulary = calculate_vocabulary(completion_tokens, target_vocabulary_size, reserved_tokens);
        // completion_vocabulary = {"[PAD]", "[UNK]", "[START]", "[END]", "Good", "Bad"};
        // completion_vocabulary = {"[START]", "[END]", "Good", "Bad"};
    }
    else
    {
        cout << "Importing vocabularies..." << endl;

        import_vocabulary(context_vocabulary_path, context_vocabulary);
        import_vocabulary(completion_vocabulary_path, completion_vocabulary);
    }

    const size_t LIMIT = 126;

    size_t maximum_context_tokens = context_tokens[0].size();

    for(size_t i = 0; i < entry_number; i++)
        if(context_tokens[i].size() > maximum_context_tokens)
            maximum_context_tokens = context_tokens[i].size();

    maximum_context_length = std::min(maximum_context_tokens, LIMIT);

    size_t maximum_completion_tokens = completion_tokens[0].size();

    for(size_t i = 0; i < entry_number; i++)
        if(completion_tokens[i].size() > maximum_completion_tokens)
            maximum_completion_tokens = completion_tokens[i].size();

    maximum_completion_length = std::min(maximum_completion_tokens, LIMIT + 1);

    // Output

    cout << "Writting data file..." << endl;

    string transformed_data_path = data_path.string();

    replace(transformed_data_path,".txt","_data.txt");
    replace(transformed_data_path,".csv","_data.csv");

    ofstream file(transformed_data_path);

    // @todo maybe context does NOT need start and end tokens

    // there is start and end indicators

    for(Index i  = 0; i < maximum_context_length + 2; i++) 
        file << "context_token_position_" << i << ";";

    for(Index i = 0; i < maximum_completion_length + 1; i++)
        file << "input_token_position_" << i << ";";

    for(Index i = 0; i < maximum_completion_length; i++)
        file << "target_token_position_" << i << ";";

    file << "target_token_position_" << maximum_completion_length << "\n";

    // Data file preview

    Index preview_size = 4;

    // data_file_preview.resize(preview_size, 2);

    data_file_preview.resize(preview_size);

    for (Index i = 0; i < preview_size; ++i) 
        data_file_preview[i].resize(2);
    
    for(Index i = 0; i < preview_size - 1; i++)
    {
        data_file_preview[i][0] = context[i];
        data_file_preview[i][1] = completion[i];
    }

    data_file_preview[preview_size - 1][0] = context[context.size()-1];
    data_file_preview[preview_size - 1][1] = completion[completion.size()-1];

    //if (!imported_vocabulary)    write_data_file_whitespace(file, context_tokens, completion_tokens);
    //else
    
    write_data_file_wordpiece(file, context_tokens, completion_tokens);

    file.close();

    data_path = transformed_data_path;
    separator = Separator::Semicolon;

    read_csv();

    set_raw_variable_types(RawVariableType::Numeric);

    for(Index i = 0; i < maximum_context_length + 2; i++)
        set_raw_variable_use(i, VariableUse::Context);

    for (Index i = 0; i < maximum_completion_length + 1; i++)
        set_raw_variable_use(i + maximum_context_length + 2, VariableUse::Input);

    for (Index i = 0; i < maximum_completion_length + 1; i++)
        set_raw_variable_use(i + maximum_context_length + maximum_completion_length + 3, VariableUse::Target);
}


void LanguageDataSet::write_data_file_wordpiece(ofstream& file,
                                                const vector<vector<string>>& context_tokens,
                                                const vector<vector<string>>& completion_tokens)
{
    const Index entry_number = context_tokens.size();

    //const unordered_map<string, type> context_vocabulary_map(context_vocabulary.begin(), context_vocabulary.end());
    //const unordered_map<string, type> completion_vocabulary_map(completion_vocabulary.begin(), completion_vocabulary.end());

    unordered_map<string, type> context_vocabulary_map;
    for(size_t i = 0; i < context_vocabulary.size(); i++)
         context_vocabulary_map[context_vocabulary[i]] = type(i);

    unordered_map<string, type> completion_vocabulary_map;
    for(size_t i = 0; i < completion_vocabulary.size(); i++)
         completion_vocabulary_map[completion_vocabulary[i]] = type(i);

    //    const Index context_vocabulary_size = context_vocabulary.size();
    //    const Index completion_vocabulary_size = completion_vocabulary.size();

    Tensor<type, 1> context_row(maximum_context_length + 2);
    Tensor<type, 1> completion_row(maximum_completion_length + 2);

    vector<string> line_tokens;
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

        line_tokens = context_tokens[i];

        for(Index j = 0; j < maximum_context_length + 1; j++)
        {
            if(j < Index(line_tokens.size()) && token_counter < maximum_context_length + 1)
            {
                word = line_tokens[j];

                wordpiece_entry = context_vocabulary_map.find(word);

                if(wordpiece_entry != context_vocabulary_map.end())
                {
                    context_row(token_counter++) = wordpiece_entry->second;
                    continue;
                }

                tokenized = false;

                for(Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
                {
                    if(token_counter == maximum_context_length + 1)
                    {
                        tokenized = true;
                        break;
                    }

                    wordpiece = word.substr(0, wordpiece_length);
                    wordpiece_entry = context_vocabulary_map.find(wordpiece);

                    if(wordpiece_entry != context_vocabulary_map.end())
                    {
                        context_row(token_counter++) = wordpiece_entry->second;

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
                    context_row(token_counter++) = 1; // unknown indicator
            }
            else
            {
                // @todo max_context_length is not defined
/*
                if(token_counter > max_context_length + 1)    
                    break;

                if(j == line_tokens.size() || (token_counter == max_context_length + 1 && !line_ended))
                {
                    context_row(token_counter++) = 3; // end indicator
                    line_ended = true;
                }
                else
                {
                    context_row(token_counter++) = type(0); // padding
                }
*/
            }
        }

        for(Index j = 0; j < maximum_context_length + 2; j++)
            file << context_row(j) << ";";

        // Completion

        completion_row.setZero();
        completion_row(0) = 2; // start indicator

        token_counter = 1;

        line_ended = false;

        line_tokens = completion_tokens[i];

        for(Index j = 0; j < maximum_completion_length + 1; j++)
        {
            if(j < Index(line_tokens.size()) && token_counter < maximum_completion_length + 1)
            {
                word = line_tokens[j];

                wordpiece_entry = completion_vocabulary_map.find(word);

                if(wordpiece_entry != completion_vocabulary_map.end())
                {
                    completion_row(token_counter++) = wordpiece_entry->second;
                    continue;
                }

                tokenized = false;

                for(Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
                {
                    if(token_counter == maximum_completion_length + 1)
                    {
                        tokenized = true;
                        break;
                    }

                    wordpiece = word.substr(0, wordpiece_length);
                    wordpiece_entry = completion_vocabulary_map.find(wordpiece);

                    if(wordpiece_entry != completion_vocabulary_map.end())
                    {
                        completion_row(token_counter++) = wordpiece_entry->second;

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
                    completion_row(token_counter++) = 1; // unknown indicator
            }
            else
            {
                if(token_counter > maximum_completion_length + 1)
                    break;

                if(j == Index(line_tokens.size())
                || (token_counter == maximum_completion_length + 1 && !line_ended))
                {
                    completion_row(token_counter++) = 3; // end indicator
                    line_ended = true;
                }
                else
                {
                    completion_row(token_counter++) = 0; // padding
                }
            }
        }

        for(Index j = 0; j < maximum_completion_length + 1; j++)
            file << completion_row(j) << ";";

        for(Index j = 1; j < maximum_completion_length + 1; j++) // Target is input shifted 1 position to the left
            file << completion_row(j) << ";";

        file << completion_row(maximum_completion_length + 1) << "\n";
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
