//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A N G U A G E  D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "language_data_set.h"

namespace opennn
{

LanguageDataSet::LanguageDataSet() : DataSet()
{
    context_variables_dimensions.resize(1);
    context_variables_dimensions.setZero();
}


/// Returns the string which will be used as separator in the data file for Text Classification.

string LanguageDataSet::get_text_separator_string() const
{
    switch(text_separator)
    {
    case Separator::Tab:
        return "Tab";

    case Separator::Semicolon:
        return "Semicolon";

    default:
        return string();
    }
}


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


Index LanguageDataSet::get_context_variables_number() const
{

    Index context_number = 0;

    for (Index i = 0; i < raw_variables.size(); i++)
    {
        if (raw_variables(i).type == RawVariableType::Categorical)
        {
            for (Index j = 0; j < raw_variables(i).categories_uses.size(); j++)
            {
                if (raw_variables(i).categories_uses(j) == VariableUse::Context)
                {
                    context_number++;
                }
            }
        }
        else if (raw_variables(i).raw_variable_use == VariableUse::Context)
        {
            context_number++;
        }
    }

    return context_number;
}


const Tensor<Index, 1>& LanguageDataSet::get_context_variables_dimensions() const
{
    return context_variables_dimensions;
}


/// Returns the indices of the input variables.

Tensor<Index, 1> LanguageDataSet::get_context_variables_indices() const
{
    const Index context_number = get_context_variables_number();

    const Tensor<Index, 1> context_raw_variables_indices = get_context_raw_variables_indices();

    Tensor<Index, 1> context_variables_indices(context_number);

    Index context_index = 0;
    Index context_variable_index = 0;

    for (Index i = 0; i < raw_variables.size(); i++)
    {
        if (raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index current_categories_number = raw_variables(i).get_categories_number();

            for (Index j = 0; j < current_categories_number; j++)
            {
                if (raw_variables(i).categories_uses(j) == VariableUse::Context)
                {
                    context_variables_indices(context_index) = context_variable_index;
                    context_index++;
                }

                context_variable_index++;
            }
        }
        else if (raw_variables(i).raw_variable_use == VariableUse::Context) // Binary, numeric
        {
            context_variables_indices(context_index) = context_variable_index;
            context_index++;
            context_variable_index++;
        }
        else
        {
            context_variable_index++;
        }
    }

    return context_variables_indices;
}


/// Returns the number of raw_variables whose uses are Input.

Index LanguageDataSet::get_context_raw_variables_number() const
{
    Index context_raw_variables_number = 0;

    for (Index i = 0; i < raw_variables.size(); i++)
    {
        if (raw_variables(i).raw_variable_use == VariableUse::Context)
        {
            context_raw_variables_number++;
        }
    }

    return context_raw_variables_number;
}


/// Returns a indices vector with the positions of the inputs.

Tensor<Index, 1> LanguageDataSet::get_context_raw_variables_indices() const
{
    const Index context_raw_variables_number = get_context_raw_variables_number();

    Tensor<Index, 1> context_raw_variables_indices(context_raw_variables_number);

    Index index = 0;

    for (Index i = 0; i < raw_variables.size(); i++)
    {
        if (raw_variables(i).raw_variable_use == VariableUse::Context)
        {
            context_raw_variables_indices(index) = i;
            index++;
        }
    }

    return context_raw_variables_indices;
}


const Tensor<Tensor<string, 1>, 1> LanguageDataSet::get_documents() const
{
    return documents;
}

const Tensor<Tensor<string, 1>, 1> LanguageDataSet::get_targets() const
{
    return targets;
}


Tensor<type, 2> LanguageDataSet::get_context_data() const
{
    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;
    initialize_sequential(indices, 0, 1, samples_number - 1);

    const Tensor<Index, 1> context_variables_indices = get_context_variables_indices();

    return get_subtensor_data(indices, context_variables_indices);
}

void LanguageDataSet::set_default_raw_variables_uses()
{
    DataSet::set_default_raw_variables_uses();

    if (raw_variables.size() > 1)
        context_variables_dimensions.resize(1);
}


void LanguageDataSet::set_raw_variables_uses(const Tensor<string, 1>& new_raw_variables_uses)
{
    DataSet::set_raw_variables_uses(new_raw_variables_uses);

    context_variables_dimensions.resize(1);
    context_variables_dimensions.setConstant(get_context_variables_number());
}


void LanguageDataSet::set_raw_variables_uses(const Tensor<VariableUse, 1>& new_raw_variables_uses)
{
    DataSet::set_raw_variables_uses(new_raw_variables_uses);

    context_variables_dimensions.resize(1);
    context_variables_dimensions.setConstant(get_context_variables_number());
}


void LanguageDataSet::set_context_variables_dimensions(const Tensor<Index, 1>& new_context_dimensions)
{
    context_variables_dimensions = new_context_dimensions;
}


/// Sets a new separator.
/// @param new_separator Separator value.

void LanguageDataSet::set_text_separator(const Separator& new_separator)
{
    separator = new_separator;
}


/// Sets a new separator from a string.
/// @param new_separator Char with the separator value.

void LanguageDataSet::set_text_separator(const string& new_separator_string)
{
    if(new_separator_string == "Tab")
    {
        text_separator = Separator::Tab;
    }
    else if(new_separator_string == "Semicolon")
    {
        text_separator = Separator::Semicolon;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_text_separator(const string&) method.\n"
               << "Unknown separator: " << new_separator_string << ".\n";

        throw runtime_error(buffer.str());
    }
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
    data_source_path = "";

    set(batch_samples_number, context_length + 2 * completion_length);

    for (Index i = 0; i < batch_samples_number; i++)
    {
        for (Index j = 0; j < context_length; j++)
            data(i, j) = type(rand() % context_dimension);

        for (Index j = 0; j < 2 * completion_length; j++)
            data(i, j + context_length) = type(rand() % completion_dimension);
    }

    for (Index i = 0; i < context_length; i++)
        set_raw_variable_use(i, DataSet::VariableUse::Context);

    for (Index i = 0; i < completion_length; i++)
        set_raw_variable_use(i + context_length, DataSet::VariableUse::Input);

    for (Index i = 0; i < completion_length; i++)
        set_raw_variable_use(i + context_length + completion_length, DataSet::VariableUse::Target);
}


void LanguageDataSet::set_default()
{
    DataSet::set_default();

    context_variables_dimensions.resize(1);

    context_variables_dimensions.setConstant(get_context_variables_number());
}

Tensor<string, 2> LanguageDataSet::get_text_data_file_preview() const
{
    return text_data_file_preview;
}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void LanguageDataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    time_t start, finish;
    time(&start);

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataFile");

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
        file_stream.OpenElement("DataSourcePath");

        file_stream.PushText(data_source_path.c_str());

        file_stream.CloseElement();
    }

    // Separator
    {
        file_stream.OpenElement("Separator");

        file_stream.PushText(get_separator_string().c_str());

        file_stream.CloseElement();
    }

    // raw_variables names
    {
        file_stream.OpenElement("RawVariablesNames");

        buffer.str("");
        buffer << has_raw_variables_names;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Rows labels
    {
        file_stream.OpenElement("RowsLabels");

        buffer.str("");

        buffer << has_rows_labels;

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

    // raw_variables

    file_stream.OpenElement("RawVariables");

    // raw_variables number
    {
        file_stream.OpenElement("RawVariablesNumber");

        buffer.str("");
        buffer << get_raw_variables_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // raw_variables items

    const Index raw_variables_number = get_raw_variables_number();

    {
        for(Index i = 0; i < raw_variables_number; i++)
        {
            file_stream.OpenElement("RawVariable");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            raw_variables(i).write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close raw_variables

    file_stream.CloseElement();

    // Rows labels

    if(has_rows_labels)
    {
        const Index rows_labels_number = rows_labels.size();

        file_stream.OpenElement("RowsLabels");

        buffer.str("");

        for(Index i = 0; i < rows_labels_number; i++)
        {
            buffer << rows_labels(i);

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
            SampleUse sample_use = samples_uses(i);

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
        // raw_variables missing values number
        {
            file_stream.OpenElement("raw_variablesMissingValuesNumber");

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
/*
    ostringstream buffer;

    // Data set element

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data set element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data file element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Data file name

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataSourcePath");

    if(!data_file_name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataSourcePath element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(data_file_name_element->GetText())
    {
        const string new_data_file_name = data_file_name_element->GetText();

        set_data_source_path(new_data_file_name);
    }

    // Separator

    const tinyxml2::XMLElement* separator_element = data_file_element->FirstChildElement("Separator");

    if(separator_element)
    {
        if(separator_element->GetText())
        {
            const string new_separator = separator_element->GetText();

            set_separator(new_separator);
        }
        else
        {
            set_separator("Comma");
        }
    }
    else
    {
        set_separator("Comma");
    }

    // Text separator

    const tinyxml2::XMLElement* text_separator_element = data_file_element->FirstChildElement("TextSeparator");

    if(text_separator_element)
    {
        if(text_separator_element->GetText())
        {
            const string new_separator = text_separator_element->GetText();

            try
            {
                set_text_separator(new_separator);
            }
            catch(const exception& e)
            {
                cerr << e.what() << endl;
            }
        }
    }

    // Has raw_variables names

    const tinyxml2::XMLElement* raw_variables_names_element = data_file_element->FirstChildElement("RawVariablesNames");

    if(raw_variables_names_element)
    {
        const string new_raw_variables_names_string = raw_variables_names_element->GetText();

        try
        {
            set_has_raw_variables_names(new_raw_variables_names_string == "1");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Rows labels

    const tinyxml2::XMLElement* rows_label_element = data_file_element->FirstChildElement("RowsLabels");

    if(rows_label_element)
    {
        const string new_rows_label_string = rows_label_element->GetText();

        try
        {
            set_has_rows_label(new_rows_label_string == "1");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Missing values label

    const tinyxml2::XMLElement* missing_values_label_element = data_file_element->FirstChildElement("MissingValuesLabel");

    if(missing_values_label_element)
    {
        if(missing_values_label_element->GetText())
        {
            const string new_missing_values_label = missing_values_label_element->GetText();

            set_missing_values_label(new_missing_values_label);
        }
        else
        {
            set_missing_values_label("NA");
        }
    }
    else
    {
        set_missing_values_label("NA");
    }

    // Codification

    const tinyxml2::XMLElement* codification_element = data_file_element->FirstChildElement("Codification");

    if(codification_element)
    {
        if(codification_element->GetText())
        {
            const string new_codification = codification_element->GetText();

            set_codification(new_codification);
        }
    }

    // raw_variables

    const tinyxml2::XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variables element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // raw_variables number

    const tinyxml2::XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variables number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    Index new_raw_variables_number = 0;

    if(raw_variables_number_element->GetText())
    {
        new_raw_variables_number = Index(atoi(raw_variables_number_element->GetText()));

        set_raw_variables_number(new_raw_variables_number);
    }

    // raw_variables

    const tinyxml2::XMLElement* start_element = raw_variables_number_element;

    if(new_raw_variables_number > 0)
    {
        for(Index i = 0; i < new_raw_variables_number; i++)
        {
            const tinyxml2::XMLElement* column_element = start_element->NextSiblingElement("RawVariable");
            start_element = column_element;

            if(column_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "raw_variable item number (" << i+1 << ") does not match (" << column_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* name_element = column_element->FirstChildElement("Name");

            if(!name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Name element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(name_element->GetText())
            {
                const string new_name = name_element->GetText();

                raw_variables(i).name = new_name;
            }

            // Scaler

            const tinyxml2::XMLElement* scaler_element = column_element->FirstChildElement("Scaler");

            if(!scaler_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Scaler element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(scaler_element->GetText())
            {
                const string new_scaler = scaler_element->GetText();

                raw_variables(i).set_scaler(new_scaler);
            }

            // raw_variable use

            const tinyxml2::XMLElement* raw_variable_use_element = column_element->FirstChildElement("RawVariableUse");

            if(!raw_variable_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "raw_variable use element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(raw_variable_use_element->GetText())
            {
                const string new_raw_variable_use = raw_variable_use_element->GetText();

                raw_variables(i).set_use(new_raw_variable_use);
            }

            // Type

            const tinyxml2::XMLElement* type_element = column_element->FirstChildElement("Type");

            if(!type_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Type element is nullptr.\n";

                throw runtime_error(buffer.str());
            }

            if(type_element->GetText())
            {
                const string new_type = type_element->GetText();
                raw_variables(i).set_type(new_type);
            }

            if(raw_variables(i).type == RawVariableType::Categorical || raw_variables(i).type == RawVariableType::Binary)
            {
                // Categories

                const tinyxml2::XMLElement* categories_element = column_element->FirstChildElement("Categories");

                if(!categories_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_element->GetText())
                {
                    const string new_categories = categories_element->GetText();

                    raw_variables(i).categories = get_tokens(new_categories, ';');
                }

                // Categories uses

                const tinyxml2::XMLElement* categories_uses_element = column_element->FirstChildElement("CategoriesUses");

                if(!categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories uses element is nullptr.\n";

                    throw runtime_error(buffer.str());
                }

                if(categories_uses_element->GetText())
                {
                    const string new_categories_uses = categories_uses_element->GetText();

                    raw_variables(i).set_categories_uses(get_tokens(new_categories_uses, ';'));
                }
            }
        }
    }

    // Rows label

    if(has_rows_labels)
    {
        // Rows labels begin tag

        const tinyxml2::XMLElement* rows_labels_element = data_set_element->FirstChildElement("RowsLabels");

        if(!rows_labels_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows labels element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        // Rows labels

        if(rows_labels_element->GetText())
        {
            const string new_rows_labels = rows_labels_element->GetText();

            char separator = ',';

            if(new_rows_labels.find(",") == string::npos
                    && new_rows_labels.find(";") != string::npos) {
                separator = ';';
            }

            rows_labels = get_tokens(new_rows_labels, separator);
        }
    }

    // Samples

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = Index(atoi(samples_number_element->GetText()));

        samples_uses.resize(new_samples_number);

        set_training();
    }

    // Samples uses

    const tinyxml2::XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples uses element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(samples_uses_element->GetText())
    {
        set_samples_uses(get_tokens(samples_uses_element->GetText(), ' '));
    }

    // Missing values

    const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Missing values method

    const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values method element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(missing_values_method_element->GetText())
    {
        set_missing_values_method(missing_values_method_element->GetText());
    }

    // Missing values number

    const tinyxml2::XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values number element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(missing_values_number_element->GetText())
    {
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));
    }

    if(missing_values_number > 0)
    {
        // raw_variables Missing values number

        const tinyxml2::XMLElement* raw_variables_missing_values_number_element = missing_values_element->FirstChildElement("raw_variablesMissingValuesNumber");

        if(!raw_variables_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "raw_variables missing values number element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        if(raw_variables_missing_values_number_element->GetText())
        {
            Tensor<string, 1> new_raw_variables_missing_values_number = get_tokens(raw_variables_missing_values_number_element->GetText(), ' ');

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
            {
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number(i).c_str());
            }
        }

        // Rows missing values number

        const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows missing values number element is nullptr.\n";

            throw runtime_error(buffer.str());
        }

        if(rows_missing_values_number_element->GetText())
        {
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
        }
    }

    // Preview data

    const tinyxml2::XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview data element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    // Preview size

    const tinyxml2::XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview size element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

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
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Row item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            if(row_element->GetText())
            {
                data_file_preview(i) = get_tokens(row_element->GetText(), ',');
            }
        }
    }
    else
    {
        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Row item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            if(row_element->GetText())
            {
                text_data_file_preview(i,0) = row_element->GetText();
            }
        }

        for(Index i = 0; i < new_preview_size; i++)
        {
            const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Target");
            start_element = row_element;

            if(row_element->Attribute("Item") != to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Target item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

                throw runtime_error(buffer.str());
            }

            if(row_element->GetText())
            {
                text_data_file_preview(i,1) = row_element->GetText();
            }
        }
    }

    // Display

    const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

    if(display_element)
    {
        const string new_display_string = display_element->GetText();

        try
        {
            set_display(new_display_string != "0");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }
*/
}


void LanguageDataSet::import_vocabulary(const string& path, Tensor<string, 1>& vocabulary)
{
    ifstream file(path.c_str());

    if (!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
            << "void import_vocabulary() method.\n"
            << "Cannot open data file: " << path << "\n";

        throw runtime_error(buffer.str());
    }

    Index vocabulary_size = 0;

    string line;

    while (file.good())
    {
        getline(file, line);

        if (line.empty()) continue;

        vocabulary_size++;

        if (file.peek() == EOF) break;
    }

    file.clear();
    file.seekg(0, ios::beg);

    vocabulary.resize(vocabulary_size);

    Index counter = 0;

    while (file.good())
    {
        getline(file, line);

        if (line.empty()) continue;

        vocabulary(counter) = line;
        counter++;

        if (file.peek() == EOF) break;
    }
}


void LanguageDataSet::load_documents(const string& path)
{
    const Index original_size = documents.size();

    if(path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
                << "void load_documents() method.\n"
                << "Data file name is empty.\n";

        throw runtime_error(buffer.str());
    }

    ifstream file(path.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
                << "void load_documents() method.\n"
                << "Cannot open data file: " << path << "\n";

        throw runtime_error(buffer.str());
    }

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

    while(file.good())
    {
        getline(file, line);
        trim(line);
        erase(line, '"');

        if(line.empty()) continue;

        lines_number++;

        if(file.peek() == EOF) break;
    }

    file.close();

    Tensor<string, 1> document(lines_number);
    Tensor<string, 1> document_target(lines_number);

    ifstream file2(path.c_str());

    Index tokens_number = 0;

    string delimiter = "";
    char separator = get_separator_char();

    while(file2.good())
    {
        getline(file2, line);

        if(line.empty()) continue;

        if(line[0]=='"')
        {
            replace(line,"\"\"", "\"");
            line = "\""+line;
            delimiter = "\"\"";
        }

        if( line.find("\"" + separator) != string::npos) replace(line,"\"" + separator, "\"\"" + separator);

        //tokens_number = count_tokens(line,delimiter + separator);
        Tensor<string,1> tokens = get_tokens(line, delimiter + separator);
        tokens_number = tokens.size();

        if(tokens_number == 1)
        {
            if(tokens(0).find(delimiter,0) == 0) document(lines_count) += tokens(0).substr(delimiter.length(), tokens(0).size());
            else document(lines_count) += " " + tokens(0);
        }
        else
        {
            if(tokens_number > 2)
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: TextAnalytics class.\n"
                        << "void load_documents() method.\n"
                        << "Found more than one separator in line: " << line << "\n";

                throw runtime_error(buffer.str());
            }
            if(tokens(0).empty() && tokens(1).empty())  continue;

            document(lines_count) += " " + tokens(0);
            document_target(lines_count) += tokens(1);
            delimiter = "";
            lines_count++;

        }

        if(file2.peek() == EOF) break;
    }

    Tensor<string,1> document_copy(lines_count);
    Tensor<string,1> document_target_copy(lines_count);

    copy(/*execution::par,*/
        document.data(),
        document.data() + lines_count,
        document_copy.data());

    copy(/*execution::par,*/
        document_target.data(),
        document_target.data() + lines_count,
        document_target_copy.data());

    documents(original_size) = document_copy;
    targets(original_size) = document_target_copy;

    file2.close();
}


void LanguageDataSet::read_csv_3_language_model()
{
    std::regex accent_regex("[\\xC0-\\xFF]");
    std::ifstream file;

#ifdef _WIN32

    if (std::regex_search(data_source_path, accent_regex))
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
        std::wstring file_name_wide = conv.from_bytes(data_source_path);
        file.open(file_name_wide);
    }
    else
    {
        file.open(data_source_path.c_str());
    }

#else
    file.open(data_source_path.c_str());
#endif

    if (!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
            << "void read_csv_3_simple() method.\n"
            << "Cannot open data file: " << data_source_path << "\n";

        throw runtime_error(buffer.str());
    }

    const bool is_float = is_same<type, float>::value;

    const char separator_char = get_separator_char();

    string line;

    // Read header

    if (has_raw_variables_names)
    {
        while (file.good())
        {
            getline(file, line);

            line = decode(line);

            if (line.empty()) continue;

            break;
        }
    }

    // Read data

    const Index raw_raw_variables_number = has_rows_labels ? get_raw_variables_number() + 1 : get_raw_variables_number();

    Tensor<string, 1> tokens(raw_raw_variables_number);

    const Index samples_number = data.dimension(0);

    if (has_rows_labels) rows_labels.resize(samples_number);

    if (display) cout << "Reading data..." << endl;

    Index sample_index = 0;
    Index raw_variable_index = 0;

    while (file.good())
    {
        getline(file, line);

        line = decode(line);

        trim(line);

        erase(line, '"');

        if (line.empty()) continue;

        fill_tokens(line, separator_char, tokens);

        for (Index j = 0; j < raw_raw_variables_number; j++)
        {
            trim(tokens(j));

            if (has_rows_labels && j == 0)
            {
                rows_labels(sample_index) = tokens(j);
            }
            else if (tokens(j) == missing_values_label || tokens(j).empty())
            {
                data(sample_index, raw_variable_index) = type(NAN);
                raw_variable_index++;
            }
            else if (is_float)
            {
                data(sample_index, raw_variable_index) = type(strtof(tokens(j).data(), nullptr));
                raw_variable_index++;
            }
            else
            {
                data(sample_index, raw_variable_index) = type(stof(tokens(j)));
                raw_variable_index++;
            }
        }

        raw_variable_index = 0;
        sample_index++;
    }

    const Index data_file_preview_index = has_raw_variables_names ? 3 : 2;

    data_file_preview(data_file_preview_index) = tokens;

    file.close();

    if (display) cout << "Data read succesfully..." << endl;
}


void LanguageDataSet::read_csv_language_model()
{
    read_csv_1();

    read_csv_2_simple();

    read_csv_3_language_model();
}


void LanguageDataSet::read_txt_language_model()
{
    cout << "Reading .txt file..." << endl;

    load_documents(data_source_path);

    Index entry_number = documents(0).size();

    for(Index i = 1; i < documents.size(); i++)
        entry_number += documents(i).size();

    Index completion_entry_number = targets(0).size();
    
    for(Index i = 1; i < targets.size(); i++)
        completion_entry_number += targets(i).size();

    if (entry_number != completion_entry_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
            << "void read_txt_language_model() method.\n"
            << "Context number of entries (" << entry_number << ") not equal to completion number of entries (" << completion_entry_number << ").\n";

        throw runtime_error(buffer.str());
    }

    Tensor<string, 1> context(entry_number);

    Index entry_index = 0;

    for (Index i = 0; i < documents.size(); i++)
    {
        for (Index j = 0; j < documents(i).size(); j++)
        {
            context(entry_index) = documents(i)(j);
            entry_index++;
        }
    }

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

    cout << "Calculating vocabularies..." << endl;

    bool imported_vocabulary = false;

    if (context_vocabulary_path.empty() || completion_vocabulary_path.empty())
    {
        context_vocabulary = calculate_vocabulary(context_tokens);
        completion_vocabulary = calculate_vocabulary(completion_tokens);
    }
    else
    {
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
    
    string transformed_data_path = data_source_path;
    replace(transformed_data_path,".txt","_data.txt");
    replace(transformed_data_path,".csv","_data.csv");

    std::ofstream file;
    file.open(transformed_data_path);

    // @todo maybe context does NOT need start and end tokens

    for(Index i  = type(0); i < max_context_length + 2; i++) /// there is start (=1) and end (=2) indicators
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

    if (!imported_vocabulary)    write_data_file_whitespace(file, context_tokens, completion_tokens);
    else    write_data_file_wordpiece(file, context_tokens, completion_tokens);

    file.close();

    data_source_path = transformed_data_path;
    separator = Separator::Semicolon;
    has_raw_variables_names = true;

    read_csv_language_model();

    set_all_raw_variables_type(RawVariableType::Numeric);

    for(Index i = 0; i < max_context_length + 2; i++)
        set_raw_variable_use(i, VariableUse::Context);

    for (Index i = 0; i < max_completion_length + 1; i++)
        set_raw_variable_use(i + max_context_length + 2, VariableUse::Input);

    for (Index i = 0; i < max_completion_length + 1; i++)
        set_raw_variable_use(i + max_context_length + max_completion_length + 3, VariableUse::Target);
    
}


void LanguageDataSet::write_data_file_whitespace(ofstream& file,
                                                 const Tensor<Tensor<string, 1>, 1>& context_tokens,
                                                 const Tensor<Tensor<string, 1>, 1>& completion_tokens)
{
    const Index entry_number = context_tokens.dimension(0);

    const Index context_vocabulary_size = context_vocabulary.size();
    const Index completion_vocabulary_size = completion_vocabulary.size();

    Tensor<type, 1> context_row(max_context_length + 2);
    Tensor<type, 1> completion_row(max_completion_length + 2);

    Tensor<string, 1> line_tokens;
    bool line_ended;

    for (Index i = 0; i < entry_number; i++)
    {
        // Context

        context_row.setZero();
        context_row(0) = 1; // start indicator

        line_ended = false;

        line_tokens = context_tokens(i);

        for (Index j = 0; j < max_context_length + 1; j++)
        {
            if (j < line_tokens.size())
            {
                auto it = find(context_vocabulary.data(), context_vocabulary.data() + context_vocabulary_size, line_tokens(j));

                const Index word_index = it - context_vocabulary.data();

                context_row(j + 1) = type(word_index);
            }
            else
            {
                if (j == line_tokens.size() || (j == max_context_length && !line_ended))
                {
                    context_row(j + 1) = 2; // end indicator
                    line_ended = true;
                }
                else
                {
                    context_row(j + 1) = 0; // pad indicator
                }
            }
        }

        for (Index j = 0; j < max_context_length + 2; j++)
            file << context_row(j) << ";";

        // Completion

        completion_row.setZero();
        completion_row(0) = 1;

        line_ended = false;

        line_tokens = completion_tokens(i);

        for (Index j = 0; j < max_completion_length + 1; j++)
        {
            if (j < line_tokens.size())
            {
                auto it = find(completion_vocabulary.data(), completion_vocabulary.data() + completion_vocabulary_size, line_tokens(j));

                const Index word_index = it - completion_vocabulary.data();

                completion_row(j + 1) = type(word_index);
            }
            else
            {
                if (j == line_tokens.size() || (j == max_completion_length && !line_ended))
                {
                    completion_row(j + 1) = 2;
                    line_ended = true;
                }
                else
                {
                    completion_row(j + 1) = 0;
                }
            }
        }

        for (Index j = 0; j < max_completion_length + 1; j++)
            file << completion_row(j) << ";";

        for (Index j = 1; j < max_completion_length + 1; j++) // Target is input shifted 1 position to the left
            file << completion_row(j) << ";";
        file << completion_row(max_completion_length + 1) << "\n";

    }
}


void LanguageDataSet::write_data_file_wordpiece(ofstream& file,
                                                const Tensor<Tensor<string, 1>, 1>& context_tokens,
                                                const Tensor<Tensor<string, 1>, 1>& completion_tokens)
{
    const Index entry_number = context_tokens.dimension(0);

    unordered_map<std::string, type> context_vocabulary_map;
    for (Index i = 0; i < context_vocabulary.size(); i++)    context_vocabulary_map[context_vocabulary(i)] = type(i);

    unordered_map<std::string, type> completion_vocabulary_map;
    for (Index i = 0; i < completion_vocabulary.size(); i++)    completion_vocabulary_map[completion_vocabulary(i)] = type(i);

    const Index context_vocabulary_size = context_vocabulary.size();
    const Index completion_vocabulary_size = completion_vocabulary.size();

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
    
    for (Index i = 0; i < entry_number; i++)
    {
        
        // Context

        context_row.setZero();
        context_row(0) = 2; // start indicator

        token_counter = 1;

        line_ended = false;

        line_tokens = context_tokens(i);
        
        for (Index j = 0; j < max_context_length + 1; j++)
        {
            if (j < line_tokens.size() && token_counter < max_context_length + 1)
            {
                word = line_tokens(j);

                wordpiece_entry = context_vocabulary_map.find(word);
                
                if (wordpiece_entry != context_vocabulary_map.end())
                {
                    context_row(token_counter) = wordpiece_entry->second;
                    token_counter++;
                    continue;
                }
                
                tokenized = false;

                for (Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
                {
                    if (token_counter == max_context_length + 1)
                    {
                        tokenized = true;
                        break;
                    }
                    
                    wordpiece = word.substr(0, wordpiece_length);
                    wordpiece_entry = context_vocabulary_map.find(wordpiece);

                    if (wordpiece_entry != context_vocabulary_map.end())
                    {
                        context_row(token_counter) = wordpiece_entry->second;
                        token_counter++;
                        
                        rest = word.substr(wordpiece_length);

                        if (rest.empty())
                        {
                            tokenized = true;
                            break;
                        }

                        word = "##" + rest;
                        wordpiece_length = word.length() + 1;
                    }
                }

                if (!tokenized)
                {
                    context_row(token_counter) = 1; // unknown indicator
                    token_counter++;
                }
            }
            else
            {
                if (j == line_tokens.size() || (token_counter == max_context_length + 1 && !line_ended))
                {
                    context_row(token_counter) = 3; // end indicator
                    token_counter++;
                    line_ended = true;
                }
                else
                {
                    if (token_counter > max_context_length + 1)    break;
                    context_row(token_counter) = type(0); // padding
                    token_counter++;
                }
            }
        }
        
        for (Index j = 0; j < max_context_length + 2; j++)
            file << context_row(j) << ";";
        
        
        // Completion

        completion_row.setZero();
        completion_row(0) = 2; // start indicator

        token_counter = 1;

        line_ended = false;

        line_tokens = completion_tokens(i);

        for (Index j = 0; j < max_completion_length + 1; j++)
        {
            if (j < line_tokens.size() && token_counter < max_completion_length + 1)
            {
                word = line_tokens(j);
                
                wordpiece_entry = completion_vocabulary_map.find(word);

                if (wordpiece_entry != completion_vocabulary_map.end())
                {
                    completion_row(token_counter) = wordpiece_entry->second;
                    token_counter++;
                    continue;
                }

                tokenized = false;

                for (Index wordpiece_length = word.length(); wordpiece_length > 0; wordpiece_length--)
                {
                    if (token_counter == max_completion_length + 1)
                    {
                        tokenized = true;
                        break;
                    }

                    wordpiece = word.substr(0, wordpiece_length);
                    wordpiece_entry = completion_vocabulary_map.find(wordpiece);

                    if (wordpiece_entry != completion_vocabulary_map.end())
                    {
                        completion_row(token_counter) = wordpiece_entry->second;
                        token_counter++;

                        rest = word.substr(wordpiece_length);

                        if (rest.empty())
                        {
                            tokenized = true;
                            break;
                        }

                        word = "##" + rest;
                        wordpiece_length = word.length() + 1;
                    }
                }

                if (!tokenized)
                {
                    completion_row(token_counter) = 1; // unknown indicator
                    token_counter++;
                }
            }
            else
            {
                if (j == line_tokens.size() || (token_counter == max_completion_length + 1 && !line_ended))
                {
                    completion_row(token_counter) = 3; // end indicator
                    token_counter++;
                    line_ended = true;
                }
                else
                {
                    if (token_counter > max_context_length + 1)    break;
                    completion_row(token_counter) = 0; // padding
                    token_counter++;
                }
            }
        }

        for (Index j = 0; j < max_completion_length + 1; j++)
            file << completion_row(j) << ";";

        for (Index j = 1; j < max_completion_length + 1; j++) // Target is input shifted 1 position to the left
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
