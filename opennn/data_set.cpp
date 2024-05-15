//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "data_set.h"
#include "statistics.h"
#include "correlations.h"
#include "tensors.h"
#include "codification.h"

using namespace opennn;
//using namespace std;


namespace opennn
{

/// Default constructor.
/// It creates a data set object with zero samples and zero inputs and target variables.
/// It also initializes the rest of class members to their default values.

DataSet::DataSet()
{
    set();

    set_default();
}


/// Default constructor. It creates a data set object from data Eigen Matrix.
/// It also initializes the rest of class members to their default values.
/// @param data Data Tensor<type, 2>.

DataSet::DataSet(const Tensor<type, 2>& data)
{
    set(data);

    set_default();
}

/// @brief Indra DataSet constructor.
/// @param data 
/// @param indra 
DataSet::DataSet(const Tensor<type, 2>& data, const Index& samples_number, const Tensor<string, 1>& columns_names, bool& indra)
{
    set();

    set_default();

    set(data);

    samples_uses.resize(samples_number);

    samples_uses.setConstant(SampleUse::Training);

    split_samples_sequential();

//    set_default_columns_scalers();
}


/// Samples and variables number constructor.
/// It creates a data set object with given samples and variables numbers.
/// All the variables are set as inputs.
/// It also initializes the rest of class members to their default values.
/// @param new_samples_number Number of samples in the data set.
/// @param new_variables_number Number of variables.

DataSet::DataSet(const Index& new_samples_number, const Index& new_variables_number)
{
    set(new_samples_number, new_variables_number);

    set_default();
}


/// Samples number, input variables number and target variables number constructor.
/// It creates a data set object with given samples and inputs and target variables numbers.
/// It also initializes the rest of class members to their default values.
/// @param new_samples_number Number of samples in the data set.
/// @param new_inputs_number Number of input variables.
/// @param new_targets_number Number of target variables.

DataSet::DataSet(const Index& new_samples_number, const Index& new_inputs_number, const Index& new_targets_number)
{
    set(new_samples_number, new_inputs_number, new_targets_number);

    set_default();
}


DataSet::DataSet(const Tensor<type, 1>& inputs_variables_dimensions, const Index& channels_number)
{
    set(inputs_variables_dimensions, channels_number);

    set_default();
}


/// File and separator constructor. It creates a data set object by loading the object members from a data file.
/// It also sets a separator.
/// Please mind about the file format. This is specified in the User's Guide.
/// @param data_source_path Data file name.
/// @param separator Data file separator between raw_variables.
/// @param has_raw_variables_names True if data file contains a row with raw_variables names, False otherwise.
/// @param data_codification String codification of the input file

DataSet::DataSet(const string& data_source_path, const char& separator, const bool& has_raw_variables_names, const Codification& data_codification)
{
    cout << "This construictor" << endl;
    set(data_source_path, separator, has_raw_variables_names, data_codification);
}


/// Destructor.

DataSet::~DataSet()
{
    delete thread_pool;
    delete thread_pool_device;
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& DataSet::get_display() const
{
    return display;
}


/// raw_variable default constructor

DataSet::RawVariable::RawVariable()
{
    name = "";
    raw_variable_use = VariableUse::Input;
    type = RawVariableType::Numeric;
    categories.resize(0);
    categories_uses.resize(0);

    scaler = Scaler::MeanStandardDeviation;
}


/// raw_variable default constructor

DataSet::RawVariable::RawVariable(const string& new_name,
                        const VariableUse& new_raw_variable_use,
                        const RawVariableType& new_type,
                        const Scaler& new_scaler,
                        const Tensor<string, 1>& new_categories,
                        const Tensor<VariableUse, 1>& new_categories_uses)
{
    name = new_name;
    scaler = new_scaler;
    raw_variable_use = new_raw_variable_use;
    type = new_type;
    categories = new_categories;
    categories_uses = new_categories_uses;
}


void DataSet::RawVariable::set_scaler(const Scaler& new_scaler)
{
    scaler = new_scaler;
}


void DataSet::RawVariable::set_scaler(const string& new_scaler)
{
    if(new_scaler == "NoScaling")
    {
        set_scaler(Scaler::NoScaling);
    }
    else if(new_scaler == "MinimumMaximum")
    {
        set_scaler(Scaler::MinimumMaximum);
    }
    else if(new_scaler == "MeanStandardDeviation")
    {
        set_scaler(Scaler::MeanStandardDeviation);
    }
    else if(new_scaler == "StandardDeviation")
    {
        set_scaler(Scaler::StandardDeviation);
    }
    else if(new_scaler == "Logarithm")
    {
        set_scaler(Scaler::Logarithm);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_scaler(const string&) method.\n"
               << "Unknown scaler: " << new_scaler << "\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets the use of the raw_variable and of the categories.
/// @param new_raw_variable_use New use of the raw_variable.

void DataSet::RawVariable::set_use(const VariableUse& new_raw_variable_use)
{
    raw_variable_use = new_raw_variable_use;

    for(Index i = 0; i < categories_uses.size(); i++)
    {
        categories_uses(i) = new_raw_variable_use;
    }
}


/// Sets the use of the raw_variable and of the categories.
/// @param new_raw_variable_use New use of the raw_variable in string format.

void DataSet::RawVariable::set_use(const string& new_raw_variable_use)
{
    if(new_raw_variable_use == "Input")
    {
        set_use(VariableUse::Input);
    }
    else if(new_raw_variable_use == "Target")
    {
        set_use(VariableUse::Target);
    }
    else if(new_raw_variable_use == "Time")
    {
        set_use(VariableUse::Time);
    }
    else if(new_raw_variable_use == "Unused")
    {
        set_use(VariableUse::Unused);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_use(const string&) method.\n"
               << "Unknown raw_variable use: " << new_raw_variable_use << "\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets the raw_variable type.
/// @param new_raw_variable_type raw_variable type in string format.

void DataSet::RawVariable::set_type(const string& new_raw_variable_type)
{
    if(new_raw_variable_type == "Numeric")
    {
        type = RawVariableType::Numeric;
    }
    else if(new_raw_variable_type == "Binary")
    {
        type = RawVariableType::Binary;
    }
    else if(new_raw_variable_type == "Categorical")
    {
        type = RawVariableType::Categorical;
    }
    else if(new_raw_variable_type == "DateTime")
    {
        type = RawVariableType::DateTime;
    }
    else if(new_raw_variable_type == "Constant")
    {
        type = RawVariableType::Constant;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void raw_variable::set_type(const string&) method.\n"
               << "raw_variable type not valid (" << new_raw_variable_type << ").\n";

        throw runtime_error(buffer.str());

    }
}


/// Adds a category to the categories vector of this raw_variable.
/// It also adds a default use for the category
/// @param new_category String that contains the name of the new category

void DataSet::RawVariable::add_category(const string & new_category)
{
    const Index old_categories_number = categories.size();

    Tensor<string, 1> old_categories = categories;
    Tensor<VariableUse, 1> old_categories_uses = categories_uses;

    categories.resize(old_categories_number+1);
    categories_uses.resize(old_categories_number+1);

    for(Index category_index = 0; category_index < old_categories_number; category_index++)
    {
        categories(category_index) = old_categories(category_index);
        categories_uses(category_index) = raw_variable_use;
    }

    categories(old_categories_number) = new_category;

    categories_uses(old_categories_number) = raw_variable_use;
}


void DataSet::RawVariable::set_categories(const Tensor<string, 1>& new_categories)
{
    categories.resize(new_categories.size());

    categories = new_categories;
}


/// Sets the categories uses in the data set.
/// @param new_categories_uses String vector that contains the new categories of the data set.

void DataSet::RawVariable::set_categories_uses(const Tensor<string, 1>& new_categories_uses)
{
    const Index new_categories_uses_number = new_categories_uses.size();

    categories_uses.resize(new_categories_uses_number);

    for(Index i = 0; i < new_categories_uses.size(); i++)
    {
        if(new_categories_uses(i) == "Input")
        {
            categories_uses(i) = VariableUse::Input;
        }
        else if(new_categories_uses(i) == "Target")
        {
            categories_uses(i) = VariableUse::Target;
        }
        else if(new_categories_uses(i) == "Time")
        {
            categories_uses(i) = VariableUse::Time;
        }
        else if(new_categories_uses(i) == "Unused")
        {
            categories_uses(i) = VariableUse::Unused;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void raw_variable::set_categories_uses(const Tensor<string, 1>&) method.\n"
                   << "Category use not valid (" << new_categories_uses(i) << ").\n";

            throw runtime_error(buffer.str());
        }
    }
}


/// Sets the categories uses in the data set.
/// @param new_categories_use New categories use

void DataSet::RawVariable::set_categories_uses(const VariableUse& new_categories_use)
{
    categories_uses.setConstant(new_categories_use);
}


void DataSet::RawVariable::from_XML(const tinyxml2::XMLDocument& column_document)
{
    ostringstream buffer;

    // Name

    const tinyxml2::XMLElement* name_element = column_document.FirstChildElement("Name");

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

        name = new_name;
    }

    // Scaler

    const tinyxml2::XMLElement* scaler_element = column_document.FirstChildElement("Scaler");

    if(!scaler_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Scaler element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(scaler_element->GetText())
    {
        const string new_scaler = scaler_element->GetText();

        set_scaler(new_scaler);
    }

    // raw_variable use

    const tinyxml2::XMLElement* raw_variable_use_element = column_document.FirstChildElement("RawVariableUse");

    if(!raw_variable_use_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variable use element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    if(raw_variable_use_element->GetText())
    {
        const string new_raw_variable_use = raw_variable_use_element->GetText();

        set_use(new_raw_variable_use);
    }

    // Type

    const tinyxml2::XMLElement* type_element = column_document.FirstChildElement("Type");

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
        set_type(new_type);
    }

    if(type == RawVariableType::Categorical)
    {
        // Categories

        const tinyxml2::XMLElement* categories_element = column_document.FirstChildElement("Categories");

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

            categories = get_tokens(new_categories, ';');
        }

        // Categories uses

        const tinyxml2::XMLElement* categories_uses_element = column_document.FirstChildElement("CategoriesUses");

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

            set_categories_uses(get_tokens(new_categories_uses, ';'));
        }
    }
}


void DataSet::RawVariable::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Name

    file_stream.OpenElement("Name");

    file_stream.PushText(name.c_str());

    file_stream.CloseElement();

    // Scaler

    file_stream.OpenElement("Scaler");

    switch(scaler)
    {
    case Scaler::NoScaling: file_stream.PushText("NoScaling"); break;

    case Scaler::MinimumMaximum: file_stream.PushText("MinimumMaximum"); break;

    case Scaler::MeanStandardDeviation: file_stream.PushText("MeanStandardDeviation"); break;

    case Scaler::StandardDeviation: file_stream.PushText("StandardDeviation"); break;

    case Scaler::Logarithm: file_stream.PushText("Logarithm"); break;

    default: break;
    }

    file_stream.CloseElement();

    // raw_variable use

    file_stream.OpenElement("RawVariableUse");

    switch(raw_variable_use)
    {
    case VariableUse::Input: file_stream.PushText("Input"); break;

    case VariableUse::Target: file_stream.PushText("Target"); break;

    case VariableUse::Unused: file_stream.PushText("Unused"); break;

    case VariableUse::Time: file_stream.PushText("Time"); break;

    case VariableUse::Id: file_stream.PushText("Id"); break;

    default: break;
    }

    file_stream.CloseElement();

    // Type

    file_stream.OpenElement("Type");

    switch(type)
    {
    case RawVariableType::Numeric: file_stream.PushText("Numeric"); break;

    case RawVariableType::Binary: file_stream.PushText("Binary"); break;

    case RawVariableType::Categorical: file_stream.PushText("Categorical"); break;

    case RawVariableType::Constant: file_stream.PushText("Constant"); break;

    case RawVariableType::DateTime: file_stream.PushText("DateTime"); break;

    default: break;
    }

    file_stream.CloseElement();

    if(type == RawVariableType::Categorical || type == RawVariableType::Binary)
    {
        if(categories.size() == 0) return;

        // Categories

        file_stream.OpenElement("Categories");

        for(Index i = 0; i < categories.size(); i++)
        {
            file_stream.PushText(categories(i).c_str());

            if(i != categories.size()-1)
            {
                file_stream.PushText(";");
            }
        }

        file_stream.CloseElement();

        // Categories uses

        file_stream.OpenElement("CategoriesUses");

        for(Index i = 0; i < categories_uses.size(); i++)
        {
            switch(categories_uses(i))
            {
            case VariableUse::Input: file_stream.PushText("Input"); break;

            case VariableUse::Target: file_stream.PushText("Target"); break;

            case VariableUse::Time: file_stream.PushText("Time"); break;

            case VariableUse::Unused: file_stream.PushText("Unused"); break;

            case VariableUse::Id: file_stream.PushText("Id"); break;

            default: break;
            }

            if(i != categories_uses.size()-1)
            {
                file_stream.PushText(";");
            }
        }

        file_stream.CloseElement();
    }
}


void DataSet::RawVariable::print() const
{
    cout << "Name: " << name << endl;

    cout << "raw_variable use: ";

    switch(raw_variable_use)
    {
    case VariableUse::Input:
        cout << "Input" << endl;
        break;

    case VariableUse::Target:
        cout << "Target" << endl;
        break;

    case VariableUse::Unused:
        cout << "Unused" << endl;
        break;

    case VariableUse::Time:
        cout << "Time" << endl;
        break;

    case VariableUse::Id:
        cout << "Id" << endl;
        break;

    default:
        break;
    }

    cout << "raw_variable type: ";

    switch(type)
    {
    case RawVariableType::Numeric:
        cout << "Numeric" << endl;
        break;

    case RawVariableType::Binary:
        cout << "Binary" << endl;
        cout << "Categories: " << categories << endl;
        break;

    case RawVariableType::Categorical:
        cout << "Categorical" << endl;
        cout << "Categories: " << categories << endl;
        break;

    case RawVariableType::DateTime:
        cout << "DateTime" << endl;
        break;

    case RawVariableType::Constant:
        cout << "Constant" << endl;
        break;

    default:
        break;
    }

    cout << "Scaler: ";

    switch(scaler)
    {
    case Scaler::NoScaling:
        cout << "NoScaling" << endl;
        break;

    case Scaler::MinimumMaximum:
        cout << "MinimumMaximum" << endl;
        break;

    case Scaler::MeanStandardDeviation:
        cout << "MeanStandardDeviation" << endl;
        break;

    case Scaler::StandardDeviation:
        cout << "StandardDeviation" << endl;
        break;

    case Scaler::Logarithm:
        cout << "Logarithm" << endl;
        break;

    default:
        break;
    }
}


DataSet::ModelType DataSet::get_model_type() const
{
    return model_type;
}


string DataSet::get_model_type_string(const DataSet::ModelType& new_model_type) const
{
    if(new_model_type == ModelType::Approximation)
    {
        return "Approximation";
    }
    else if(new_model_type == ModelType::Classification)
    {
        return "Classification";
    }
    else if(new_model_type == ModelType::Forecasting)
    {
        return "Forecasting";
    }
    else if(new_model_type == ModelType::AutoAssociation)
    {
        return "AutoAssociation";
    }
    else if(new_model_type == ModelType::TextClassification)
    {
        return "TextClassification";
    }
    else if(new_model_type == ModelType::ImageClassification)
    {
        return "ImageClassification";
    }
    else
    {
        return "NA";
    }
}


Index DataSet::RawVariable::get_variables_number() const
{
    if(type == RawVariableType::Categorical)
    {
        return categories.size();
    }
    else
    {
        return 1;
    }
}


/// Returns the number of categories.

Index DataSet::RawVariable::get_categories_number() const
{
    return categories.size();
}


/// Returns the number of used categories.

Index DataSet::RawVariable::get_used_categories_number() const
{
    Index used_categories_number = 0;

    for(Index i = 0; i < categories.size(); i++)
    {
        if(categories_uses(i) != VariableUse::Unused) used_categories_number++;
    }

    return used_categories_number;
}


/// Returns a string vector that contains the names of the used variables in the data set.

Tensor<string, 1> DataSet::RawVariable::get_used_variables_names() const
{
    Tensor<string, 1> used_variables_names;

    if(type != RawVariableType::Categorical && raw_variable_use != VariableUse::Unused)
    {
        used_variables_names.resize(1);
        used_variables_names.setConstant(name);
    }
    else if(type == RawVariableType::Categorical)
    {
        used_variables_names.resize(get_used_categories_number());

        Index category_index = 0;

        for(Index i = 0; i < categories.size(); i++)
        {
            if(categories_uses(i) != VariableUse::Unused)
            {
                used_variables_names(category_index) = categories(i);

                category_index++;
            }
        }
    }

    return used_variables_names;
}


/// Returns true if a given sample is to be used for training, selection or testing,
/// and false if it is to be unused.
/// @param index Sample index.

bool DataSet::is_sample_used(const Index& index) const
{
    if(samples_uses(index) == SampleUse::Unused)
    {
        return false;
    }
    else
    {
        return true;
    }
}


/// Returns true if a given sample is to be unused and false in other case.
/// @param index Sample index.

bool DataSet::is_sample_unused(const Index& index) const
{
    if(samples_uses(index) == SampleUse::Unused)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns a vector with the number of training, selection, testing
/// and unused samples.
/// The size of that vector is therefore four.

Tensor<Index, 1> DataSet::get_samples_uses_numbers() const
{
    Tensor<Index, 1> count(4);

    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Training)
        {
            count(0)++;
        }
        else if(samples_uses(i) == SampleUse::Selection)
        {
            count(1)++;
        }
        else if(samples_uses(i) == SampleUse::Testing)
        {
            count(2)++;
        }
        else
        {
            count(3)++;
        }
    }

    return count;
}


/// Returns a vector with the uses of the samples in percentages of the data set.
/// Uses: training, selection, testing and unused samples.
/// Note that the vector size is four.

Tensor<type, 1> DataSet::get_samples_uses_percentages() const
{
    const Index samples_number = get_samples_number();
    const Index training_samples_number = get_training_samples_number();
    const Index selection_samples_number = get_selection_samples_number();
    const Index testing_samples_number = get_testing_samples_number();
    const Index unused_samples_number = get_unused_samples_number();

    const type training_samples_percentage = type(training_samples_number*100)/type(samples_number);
    const type selection_samples_percentage = type(selection_samples_number*100)/type(samples_number);
    const type testing_samples_percentage = type(testing_samples_number*100)/type(samples_number);
    const type unused_samples_percentage = type(unused_samples_number*100)/type(samples_number);

    Tensor<type, 1> samples_uses_percentage(4);

    samples_uses_percentage.setValues({training_samples_percentage,
                                       selection_samples_percentage,
                                       testing_samples_percentage,
                                       unused_samples_percentage});

    return samples_uses_percentage;
}


/// Returns a string with the values of the sample corresponding to the given index.
/// The values will be separated by the given separator char.
/// @param sample_index Index of the sample.
/// @param separator Separator.

string DataSet::get_sample_string(const Index& sample_index, const string& separator) const
{
    const Tensor<type, 1> sample = data.chip(sample_index, 0);

    string sample_string = "";

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        switch(raw_variables(i).type)
        {
        case RawVariableType::Numeric:
            if(isnan(data(sample_index, variable_index))) sample_string += missing_values_label;
            else sample_string += to_string(double(data(sample_index, variable_index)));
            variable_index++;
            break;

        case RawVariableType::Binary:
            if(isnan(data(sample_index, variable_index))) sample_string += missing_values_label;
            else sample_string += raw_variables(i).categories(Index(data(sample_index, variable_index)));
            variable_index++;
            break;

        case RawVariableType::DateTime:
            if(isnan(data(sample_index, variable_index))) 
                sample_string += missing_values_label;
            else 
                sample_string += to_string(double(data(sample_index, variable_index)));
            
            variable_index++;
            break;

        case RawVariableType::Categorical:
            if(isnan(data(sample_index, variable_index)))
            {
                sample_string += missing_values_label;
            }
            else
            {
                const Index categories_number = raw_variables(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                {
                    if(abs(data(sample_index, variable_index+j) - type(1)) < type(NUMERIC_LIMITS_MIN))
                    {
                        sample_string += raw_variables(i).categories(j);
                        break;
                    }
                }
                variable_index += categories_number;
            }
            break;

        case RawVariableType::Constant:
            if(isnan(data(sample_index, variable_index))) sample_string += missing_values_label;
            else sample_string += to_string(double(data(sample_index, variable_index)));
            variable_index++;
            break;

        default:
            break;
        }

        if(i != raw_variables_number-1) sample_string += separator + " ";
    }

    return sample_string;
}


/// Returns the indices of the samples which will be used for training.

Tensor<Index, 1> DataSet::get_training_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index training_samples_number = get_training_samples_number();

    Tensor<Index, 1> training_indices(training_samples_number);

    Index count = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Training)
        {
            training_indices(count) = i;
            count++;
        }
    }

    return training_indices;
}


/// Returns the indices of the samples which will be used for selection.

Tensor<Index, 1> DataSet::get_selection_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index selection_samples_number = get_selection_samples_number();

    Tensor<Index, 1> selection_indices(selection_samples_number);

    Index count = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Selection)
        {
            selection_indices(count) = i;
            count++;
        }
    }

    return selection_indices;
}


/// Returns the indices of the samples which will be used for testing.

Tensor<Index, 1> DataSet::get_testing_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index testing_samples_number = get_testing_samples_number();

    Tensor<Index, 1> testing_indices(testing_samples_number);

    Index count = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Testing)
        {
            testing_indices(count) = i;
            count++;
        }
    }

    return testing_indices;
}


/// Returns the indices of the used samples(those which are not set unused).

Tensor<Index, 1> DataSet::get_used_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index used_samples_number = samples_number - get_unused_samples_number();

    Tensor<Index, 1> used_indices(used_samples_number);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
    {

        if(samples_uses(i) != SampleUse::Unused)
        {
            used_indices(index) = i;
            index++;
        }
    }

    return used_indices;
}


/// Returns the indices of the samples set unused.

Tensor<Index, 1> DataSet::get_unused_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index unused_samples_number = get_unused_samples_number();

    Tensor<Index, 1> unused_indices(unused_samples_number);

    Index count = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Unused)
        {
            unused_indices(count) = i;
            count++;
        }
    }

    return unused_indices;
}


/// Returns the use of a single sample.
/// @param index Sample index.

DataSet::SampleUse DataSet::get_sample_use(const Index& index) const
{
    return samples_uses(index);
}


/// Returns the use of every sample (training, selection, testing or unused) in a vector.

const Tensor<DataSet::SampleUse,1 >& DataSet::get_samples_uses() const
{
    return samples_uses;
}


/// Returns a vector, where each element is a vector that contains the indices of the different batches of the training samples.
/// @param shuffle Is a boleean.
/// If shuffle is true, then the indices are shuffled into batches, and false otherwise

Tensor<Index, 2> DataSet::get_batches(const Tensor<Index,1>& samples_indices,
                                      const Index& batch_samples_number,
                                      const bool& shuffle,
                                      const Index& new_buffer_size) const
{
    if(!shuffle) return split_samples(samples_indices, batch_samples_number);

    random_device rng;
    mt19937 urng(rng());

    const Index samples_number = samples_indices.size();

    Index buffer_size = new_buffer_size;
    Index batches_number;
    Index batch_size = batch_samples_number;

    // When samples_number is less than 100 (small sample)

    if(buffer_size > samples_number)
    {
        buffer_size = samples_number;
    }

    // Check batch size and samples number

    if(samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
        buffer_size = batch_size;

        Tensor<Index,1> samples_copy(samples_indices);

        Tensor<Index, 2> batches(batches_number, batch_size);

        // Shuffle

        std::shuffle(samples_copy.data(), samples_copy.data() + samples_copy.size(), urng);

        for(Index i = 0; i < batch_size; i++)
        {
            batches(0, i) = samples_copy(i);
        }

        return batches;
    }
    else
    {
        batches_number = samples_number / batch_size;
    }

    Tensor<Index, 2> batches(batches_number, batch_size);

    Tensor<Index, 1> buffer(buffer_size);

    for(Index i = 0; i < buffer_size; i++) buffer(i) = i;

    Index next_index = buffer_size;
    Index random_index = 0;

    // Heuristic cases for batch shuffling

    if(batch_size < buffer_size)
    {
        Index diff = buffer_size/ batch_size;

        // Main Loop

        for(Index i = 0; i < batches_number; i++)
        {
            // Last batch

            if(i == batches_number-diff)
            {
                Index buffer_index = 0;

                for(Index k = batches_number-diff; k < batches_number; k++)
                {
                    for(Index j = 0; j < batch_size; j++)
                    {
                        batches(k,j) = buffer(buffer_index);

                        buffer_index++;
                    }
                }

                break;
            }

            // Shuffle batches

            for(Index j = 0; j < batch_size; j++)
            {
                random_index = Index(rand()%buffer_size);

                batches(i, j) = buffer(random_index);

                buffer(random_index) = samples_indices(next_index);

                next_index++;
            }
        }

        return batches;
    }
    else // buffer_size <= batch_size
    {
        // Main Loop

        for(Index i = 0; i < batches_number; i++)
        {
            // Last batch

            if(i == batches_number-1)
            {
                std::shuffle(buffer.data(), buffer.data() +  buffer.size(), urng);

                if(batch_size <= buffer_size)
                {
                    for(Index j = 0; j < batch_size;j++)
                    {
                        batches(i,j) = buffer(j);
                    }
                }
                else //buffer_size < batch_size
                {
                    for(Index j = 0; j < buffer_size; j++)
                    {
                        batches(i,j) = buffer(j);
                    }

                    for(Index j = buffer_size; j < batch_size; j++)
                    {
                        batches(i,j) = samples_indices(next_index);

                        next_index++;
                    }
                }

                break;
            }

            // Shuffle batches

            for(Index j = 0; j < batch_size; j++)
            {
                random_index = Index(rand()%buffer_size);

                batches(i, j) = buffer(random_index);

                buffer(random_index) = samples_indices(next_index);

                next_index++;
            }
        }

        return batches;
    }
}


/// Returns the number of samples in the data set which will be used for training.

Index DataSet::get_training_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index training_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Training)
        {
            training_samples_number++;
        }
    }

    return training_samples_number;
}


/// Returns the number of samples in the data set which will be used for selection.

Index DataSet::get_selection_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index selection_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Selection)
        {
            selection_samples_number++;
        }
    }

    return selection_samples_number;
}


/// Returns the number of samples in the data set which will be used for testing.

Index DataSet::get_testing_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index testing_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Testing)
        {
            testing_samples_number++;
        }
    }

    return testing_samples_number;
}


/// Returns the total number of training, selection and testing samples,
/// i.e. those which are not "Unused".

Index DataSet::get_used_samples_number() const
{
    const Index samples_number = get_samples_number();
    const Index unused_samples_number = get_unused_samples_number();

    return (samples_number - unused_samples_number);
}


/// Returns the number of samples in the data set which will neither be used
/// for training, selection or testing.

Index DataSet::get_unused_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index unused_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Unused)
        {
            unused_samples_number++;
        }
    }

    return unused_samples_number;
}


/// Sets all the samples in the data set for training.

void DataSet::set_training()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = SampleUse::Training;
    }
}


/// Sets all the samples in the data set for selection.

void DataSet::set_selection()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = SampleUse::Selection;
    }
}


/// Sets all the samples in the data set for testing.

void DataSet::set_testing()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = SampleUse::Testing;
    }
}


/// Sets samples with given indices in the data set for training.
/// @param indices Indices vector with the index of samples in the data set for training.

void DataSet::set_training(const Tensor<Index, 1>& indices)
{
    Index index = 0;

    for(Index i = 0; i < indices.size(); i++)
    {
        index = indices(i);

        samples_uses(index) = SampleUse::Training;
    }
}


/// Sets samples with given indices in the data set for selection.
/// @param indices Indices vector with the index of samples in the data set for selection.

void DataSet::set_selection(const Tensor<Index, 1>& indices)
{
    Index index = 0;

    for(Index i = 0; i < indices.size(); i++)
    {
        index = indices(i);

        samples_uses(index) = SampleUse::Selection;
    }
}


/// Sets samples with given indices in the data set for testing.
/// @param indices Indices vector with the index of samples in the data set for testing.

void DataSet::set_testing(const Tensor<Index, 1>& indices)
{
    Index index = 0;

    for(Index i = 0; i < indices.size(); i++)
    {
        index = indices(i);

        samples_uses(index) = SampleUse::Testing;
    }
}


/// Sets all the samples in the data set for unused.

void DataSet::set_samples_unused()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = SampleUse::Unused;
    }
}


/// Sets samples with given indices in the data set for unused.
/// @param indices Indices vector with the index of samples in the data set for unused.

void DataSet::set_samples_unused(const Tensor<Index, 1>& indices)
{
    for(Index i = 0; i < Index(indices.size()); i++)
    {
        const Index index = indices(i);

        samples_uses(index) = SampleUse::Unused;
    }
}


/// Sets the use of a single sample.
/// @param index Index of sample.
/// @param new_use Use for that sample.

void DataSet::set_sample_use(const Index& index, const SampleUse& new_use)
{
    const Index samples_number = get_samples_number();

    if(index >= samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_sample_use(const Index&, const SampleUse&) method.\n"
               << "Index must be less than samples number.\n";

        throw runtime_error(buffer.str());
    }

    samples_uses(index) = new_use;
}


/// Sets the use of a single sample from a string.
/// @param index Index of sample.
/// @param new_use String with the use name("Training", "Selection", "Testing" or "Unused")

void DataSet::set_sample_use(const Index& index, const string& new_use)
{
    if(new_use == "Training")
    {
        samples_uses(index) = SampleUse::Training;
    }
    else if(new_use == "Selection")
    {
        samples_uses(index) = SampleUse::Selection;
    }
    else if(new_use == "Testing")
    {
        samples_uses(index) = SampleUse::Testing;
    }
    else if(new_use == "Unused")
    {
        samples_uses(index) = SampleUse::Unused;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_sample_use(const string&) method.\n"
               << "Unknown sample use: " << new_use << "\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets new uses to all the samples from a single vector.
/// @param new_uses vector of use structures.
/// The size of given vector must be equal to the number of samples.

void DataSet::set_samples_uses(const Tensor<SampleUse, 1>& new_uses)
{
    const Index samples_number = get_samples_number();

#ifdef OPENNN_DEBUG

    const Index new_uses_size = new_uses.size();

    if(new_uses_size != samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_samples_uses(const Tensor<SampleUse, 1>&) method.\n"
               << "Size of uses(" << new_uses_size << ") must be equal to number of samples(" << samples_number << ").\n";

        throw runtime_error(buffer.str());
    }

#endif

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = new_uses(i);
    }
}


/// Sets new uses to all the samples from a single vector of strings.
/// @param new_uses vector of use strings.
/// Possible values for the elements are "Training", "Selection", "Testing" and "Unused".
/// The size of given vector must be equal to the number of samples.

void DataSet::set_samples_uses(const Tensor<string, 1>& new_uses)
{
    const Index samples_number = get_samples_number();

    ostringstream buffer;

#ifdef OPENNN_DEBUG

    const Index new_uses_size = new_uses.size();

    if(new_uses_size != samples_number)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_samples_uses(const Tensor<string, 1>&) method.\n"
               << "Size of uses(" << new_uses_size << ") must be equal to number of samples(" << samples_number << ").\n";

        throw runtime_error(buffer.str());
    }

#endif

    for(Index i = 0; i < samples_number; i++)
    {
        if(new_uses(i).compare("Training") == 0 || new_uses(i).compare("0") == 0)
        {
            samples_uses(i) = SampleUse::Training;
        }
        else if(new_uses(i).compare("Selection") == 0 || new_uses(i).compare("1") == 0)
        {
            samples_uses(i) = SampleUse::Selection;
        }
        else if(new_uses(i).compare("Testing") == 0 || new_uses(i).compare("2") == 0)
        {
            samples_uses(i) = SampleUse::Testing;
        }
        else if(new_uses(i).compare("Unused") == 0 || new_uses(i).compare("3") == 0)
        {
            samples_uses(i) = SampleUse::Unused;
        }
        else
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void set_samples_uses(const Tensor<string, 1>&) method.\n"
                   << "Unknown sample use: " << new_uses(i) << ".\n";

            throw runtime_error(buffer.str());
        }
    }
}


void DataSet::set_samples_uses(const Tensor<Index, 1>& indices, const SampleUse sample_use)
{
    for(Index i = 0; i < indices.size(); i++)
    {
        set_sample_use(indices(i), sample_use);
    }
}


/// Creates new training, selection and testing indices at random.
/// @param training_samples_ratio Ratio of training samples in the data set.
/// @param selection_samples_ratio Ratio of selection samples in the data set.
/// @param testing_samples_ratio Ratio of testing samples in the data set.

void DataSet::split_samples_random(const type& training_samples_ratio,
                                   const type& selection_samples_ratio,
                                   const type& testing_samples_ratio)
{
    random_device rng;
    mt19937 urng(rng());

    const Index used_samples_number = get_used_samples_number();

    if(used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    // Get number of samples for training, selection and testing

    const Index selection_samples_number = Index((selection_samples_ratio * used_samples_number)/total_ratio);
    const Index testing_samples_number = Index((testing_samples_ratio * used_samples_number)/ total_ratio);

    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Warning: DataSet class.\n"
               << "void split_samples_random(const type&, const type&, const type&) method.\n"
               << "Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n";

        throw runtime_error(buffer.str());
    }

    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;

    initialize_sequential(indices, 0, 1, samples_number-1);

    std::shuffle(indices.data(), indices.data() + indices.size(), urng);

    Index i = 0;
    Index index;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != SampleUse::Unused)
        {
            samples_uses(index)= SampleUse::Training;
            count_training++;
        }

        i++;
    }

    // Selection

    Index count_selection = 0;

    while(count_selection != selection_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != SampleUse::Unused)
        {
            samples_uses(index) = SampleUse::Selection;
            count_selection++;
        }

        i++;
    }

    // Testing

    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != SampleUse::Unused)
        {
            samples_uses(index) = SampleUse::Testing;
            count_testing++;
        }

        i++;
    }
}


/// Creates new training, selection and testing indices with sequential indices.
/// @param training_samples_ratio Ratio of training samples in the data set.
/// @param selection_samples_ratio Ratio of selection samples in the data set.
/// @param testing_samples_ratio Ratio of testing samples in the data set.

void DataSet::split_samples_sequential(const type& training_samples_ratio,
                                       const type& selection_samples_ratio,
                                       const type& testing_samples_ratio)
{
    const Index used_samples_number = get_used_samples_number();

    if(used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    // Get number of samples for training, selection and testing

    const Index selection_samples_number = Index(selection_samples_ratio* type(used_samples_number)/ type(total_ratio));
    const Index testing_samples_number = Index(testing_samples_ratio* type(used_samples_number)/ type(total_ratio));
    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Warning: Samples class.\n"
               << "void split_samples_sequential(const type&, const type&, const type&) method.\n"
               << "Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n";

        throw runtime_error(buffer.str());
    }

    Index i = 0;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        if(samples_uses(i) != SampleUse::Unused)
        {
            samples_uses(i) = SampleUse::Training;
            count_training++;
        }

        i++;
    }

    // Selection

    Index count_selection = 0;

    while(count_selection != selection_samples_number)
    {
        if(samples_uses(i) != SampleUse::Unused)
        {
            samples_uses(i) = SampleUse::Selection;
            count_selection++;
        }

        i++;
    }

    // Testing

    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        if(samples_uses(i) != SampleUse::Unused)
        {
            samples_uses(i) = SampleUse::Testing;
            count_testing++;
        }

        i++;
    }
}


void DataSet::set_raw_variables(const Tensor<RawVariable, 1>& new_raw_variables)
{
    raw_variables = new_raw_variables;
}


/// This method sets the n raw_variables of the data_set by default,
/// i.e. until raw_variable n-1 are Input and raw_variable n is Target.

void DataSet::set_default_raw_variables_uses()
{
    const Index raw_variables_number = raw_variables.size();

    bool target = false;

    if(raw_variables_number == 0)
    {
        return;
    }

    else if(raw_variables_number == 1)
    {
        raw_variables(0).set_use(VariableUse::Unused);
    }

    else
    {
        set_input();

        for(Index i = raw_variables.size()-1; i >= 0; i--)
        {
            if(raw_variables(i).type == RawVariableType::Constant || raw_variables(i).type == RawVariableType::DateTime)
            {
                raw_variables(i).set_use(VariableUse::Unused);
                continue;
            }

            if(!target)
            {
                raw_variables(i).set_use(VariableUse::Target);

                target = true;

                continue;
            }
        }

        input_variables_dimensions.resize(1);
        target_variables_dimensions.resize(1);
    }
}


/// This method puts the names of the raw_variables in the data_set.
/// This is used when the data_set does not have a header,
/// the default names are: column_0, column_1, ..., column_n.

void DataSet::set_default_raw_variables_names()
{
    const Index raw_variables_number = raw_variables.size();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables(i).name = "column_" + to_string(1+i);
    }
}


/// Sets the name of a single raw_variable.
/// @param index Index of raw_variable.
/// @param new_use Use for that raw_variable.

void DataSet::set_raw_variable_name(const Index& raw_variable_index, const string& new_name)
{
    raw_variables(raw_variable_index).name = new_name;
}


/// Returns the use of a single variable.
/// @param index Index of variable.

DataSet::VariableUse DataSet::get_numeric_variable_use(const Index& index) const
{
    return get_variables_uses()(index);
}


/// Returns a vector containing the use of the raw_variable, without taking into account the categories.

DataSet::VariableUse DataSet::get_raw_variable_use(const Index&  index) const
{
    return raw_variables(index).raw_variable_use;
}


/// Returns the uses of each raw_variables of the data set.

Tensor<DataSet::VariableUse, 1> DataSet::get_raw_variables_uses() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Tensor<DataSet::VariableUse, 1> raw_variables_uses(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables_uses(i) = raw_variables(i).raw_variable_use;
    }

    return raw_variables_uses;
}


/// Returns a vector containing the use of each raw_variable, including the categories.
/// The size of the vector is equal to the number of variables.

Tensor<DataSet::VariableUse, 1> DataSet::get_variables_uses() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index variables_number = get_variables_number();

    Tensor<VariableUse, 1> variables_uses(variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index i = 0; i < (raw_variables(i).categories_uses).size(); i++)
            {
                variables_uses(i + index) = (raw_variables(i).categories_uses)(i);
            }
            index += raw_variables(i).categories.size();
        }
        else
        {
            variables_uses(index) = raw_variables(i).raw_variable_use;
            index++;
        }
    }

    return variables_uses;
}


/// Returns the name of a single variable in the data set.
/// @param index Index of variable.

string DataSet::get_numeric_variable_name(const Index& variable_index) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).get_categories_number(); j++)
            {
                if(index == variable_index)
                {
                    return raw_variables(i).categories(j);
                }
                else
                {
                    index++;
                }
            }
        }
        else
        {
            if(index == variable_index)
            {
                return raw_variables(i).name;
            }
            else
            {
                index++;
            }
        }
    }

    return string();
}


/// Returns a string vector with the names of all the variables in the data set.
/// The size of the vector is the number of variables.

Tensor<string, 1> DataSet::get_variables_names() const
{
    const Index variables_number = get_variables_number();

    Tensor<string, 1> variables_names(variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).categories.size(); j++)
            {
                variables_names(index) = raw_variables(i).categories(j);

                index++;
            }
        }
        else
        {
            variables_names(index) = raw_variables(i).name;

            index++;
        }
    }

    return variables_names;
}


/// Returns the names of the input variables in the data set.
/// The size of the vector is the number of input variables.

Tensor<string, 1> DataSet::get_input_variables_names() const
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();

    Tensor<string, 1> input_variables_names(input_variables_number);

    Index index = 0;

    for(Index i = 0; i < input_raw_variables_indices.size(); i++)
    {
        Index input_index = input_raw_variables_indices(i);

        const Tensor<string, 1> current_used_variables_names = raw_variables(input_index).get_used_variables_names();

        for(Index j = 0; j < current_used_variables_names.size(); j++)
        {
            input_variables_names(index + j) = current_used_variables_names(j);
        }

        index += current_used_variables_names.size();
    }

    return input_variables_names;
}


/// Returns the names of the target variables in the data set.
/// The size of the vector is the number of target variables.

Tensor<string, 1> DataSet::get_target_variables_names() const
{
    const Index target_variables_number = get_target_variables_number();

    const Tensor<Index, 1> target_raw_variables_indices = get_target_raw_variables_indices();

    Tensor<string, 1> target_variables_names(target_variables_number);

    Index index = 0;

    for(Index i = 0; i < target_raw_variables_indices.size(); i++)
    {
        const Index target_index = target_raw_variables_indices(i);

        const Tensor<string, 1> current_used_variables_names = raw_variables(target_index).get_used_variables_names();

        for(Index j = 0; j < current_used_variables_names.size(); j++)
        {
            target_variables_names(index + j) = current_used_variables_names(j);
        }

        index += current_used_variables_names.size();
    }

    return target_variables_names;
}


/// Returns the dimensions of the input variables.

const Tensor<Index, 1>& DataSet::get_input_variables_dimensions() const
{
    return input_variables_dimensions;
}


Index DataSet::get_input_variables_rank() const
{
    return input_variables_dimensions.rank();
}


const Tensor<Index, 1>& DataSet::get_target_variables_dimensions() const
{
    return target_variables_dimensions;
}


/// Returns the number of variables which are either input nor target.

Index DataSet::get_used_variables_number() const
{
    const Index variables_number = get_variables_number();

    const Index unused_variables_number = get_unused_variables_number();

    return (variables_number - unused_variables_number);
}


/// Returns a indices vector with the positions of the inputs.

Tensor<Index, 1> DataSet::get_input_raw_variables_indices() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();

    Tensor<Index, 1> input_raw_variables_indices(input_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Input)
        {
            input_raw_variables_indices(index) = i;
            index++;
        }
    }

    return input_raw_variables_indices;
}


/// Returns a indices vector with the positions of the targets.

Tensor<Index, 1> DataSet::get_target_raw_variables_indices() const
{
    const Index target_raw_variables_number = get_target_raw_variables_number();

    Tensor<Index, 1> target_raw_variables_indices(target_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Target)
        {
            target_raw_variables_indices(index) = i;
            index++;
        }
    }

    return target_raw_variables_indices;
}



/// Returns a indices vector with the positions of the unused raw_variables.

Tensor<Index, 1> DataSet::get_unused_raw_variables_indices() const
{
    const Index unused_raw_variables_number = get_unused_raw_variables_number();

    Tensor<Index, 1> unused_raw_variables_indices(unused_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Unused)
        {
            unused_raw_variables_indices(index) = i;
            index++;
        }
    }

    return unused_raw_variables_indices;
}


/// Returns a indices vector with the positions of the used raw_variables.

Tensor<Index, 1> DataSet::get_used_raw_variables_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index used_raw_variables_number = get_used_raw_variables_number();

    Tensor<Index, 1> used_indices(used_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).raw_variable_use  == VariableUse::Input
                || raw_variables(i).raw_variable_use  == VariableUse::Target
                || raw_variables(i).raw_variable_use  == VariableUse::Time)
        {
            used_indices(index) = i;
            index++;
        }
    }

    return used_indices;
}


Tensor<Scaler, 1> DataSet::get_raw_variables_scalers() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Tensor<Scaler, 1> raw_variables_scalers(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables_scalers(i) = raw_variables(i).scaler;
    }

    return raw_variables_scalers;
}


Tensor<Scaler, 1> DataSet::get_input_variables_scalers() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index input_variables_number = get_input_variables_number();

    const Tensor<RawVariable, 1> input_raw_variables = get_input_raw_variables();

    Tensor<Scaler, 1> input_variables_scalers(input_variables_number);

    Index index = 0;

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        for(Index j = 0;  j < input_raw_variables(i).get_variables_number(); j++)
        {
            input_variables_scalers(index) = input_raw_variables(i).scaler;
            index++;
        }
    }

    return input_variables_scalers;
}


Tensor<Scaler, 1> DataSet::get_target_variables_scalers() const
{
    const Index target_raw_variables_number = get_target_raw_variables_number();
    const Index target_variables_number = get_target_variables_number();

    const Tensor<RawVariable, 1> target_raw_variables = get_target_raw_variables();

    Tensor<Scaler, 1> target_variables_scalers(target_variables_number);

    Index index = 0;

    for(Index i = 0; i < target_raw_variables_number; i++)
    {
        for(Index j = 0;  j < target_raw_variables(i).get_variables_number(); j++)
        {
            target_variables_scalers(index) = target_raw_variables(i).scaler;
            index++;
        }
    }

    return target_variables_scalers;
}


/// Returns a string vector that contains the names of the raw_variables.

Tensor<string, 1> DataSet::get_raw_variables_names() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Tensor<string, 1> raw_variables_names(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables_names(i) = raw_variables(i).name;
    }

    return raw_variables_names;
}



/// Returns a string vector that contains the names of the raw_variables whose uses are Input.

Tensor<string, 1> DataSet::get_input_raw_variables_names() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();

    Tensor<string, 1> input_raw_variables_names(input_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Input)
        {
            input_raw_variables_names(index) = raw_variables(i).name;
            index++;
        }
    }

    return input_raw_variables_names;
}


/// Returns a string vector which contains the names of the raw_variables whose uses are Target.

Tensor<string, 1> DataSet::get_target_raw_variables_names() const
{
    const Index target_raw_variables_number = get_target_raw_variables_number();

    Tensor<string, 1> target_raw_variables_names(target_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Target)
        {
            target_raw_variables_names(index) = raw_variables(i).name;
            index++;
        }
    }

    return target_raw_variables_names;
}


/// Returns a string vector which contains the names of the raw_variables used whether Input, Target or Time.

Tensor<string, 1> DataSet::get_used_raw_variables_names() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();

    Tensor<string, 1> names(used_raw_variables_number);

    Index index = 0 ;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).raw_variable_use != VariableUse::Unused)
        {
            names(index) = raw_variables(i).name;
            index++;
        }
    }

    return names;
}


/// Returns the number of raw_variables whose uses are Input.

Index DataSet::get_input_raw_variables_number() const
{
    Index input_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Input)
        {
            input_raw_variables_number++;
        }
    }

    return input_raw_variables_number;
}



/// Returns the number of raw_variables whose uses are Target.

Index DataSet::get_target_raw_variables_number() const
{
    Index target_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Target)
        {
            target_raw_variables_number++;
        }
    }

    return target_raw_variables_number;
}



/// Returns the number of raw_variables whose uses are Time

Index DataSet::get_time_raw_variables_number() const
{
    Index time_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Time)
        {
            time_raw_variables_number++;
        }
    }

    return time_raw_variables_number;
}


/// Returns the number of raw_variables that are not used.

Index DataSet::get_unused_raw_variables_number() const
{
    Index unused_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Unused)
        {
            unused_raw_variables_number++;
        }
    }

    return unused_raw_variables_number;
}


/// Returns the number of raw_variables that are used.

Index DataSet::get_used_raw_variables_number() const
{
    Index used_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use != VariableUse::Unused)
        {
            used_raw_variables_number++;
        }
    }

    return used_raw_variables_number;
}


/// @todo change name of method

Index DataSet::get_variables_less_target() const
{
    Index raw_variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            if(raw_variables(i).raw_variable_use == VariableUse::Input)
            {
                raw_variables_number += raw_variables(i).categories_uses.size();
            }
            else if(raw_variables(i).raw_variable_use == VariableUse::Unused)
            {
                raw_variables_number += raw_variables(i).categories_uses.size();
            }
        }
        else
        {
            if(raw_variables(i).raw_variable_use == VariableUse::Input)
            {
                raw_variables_number++;
            }
            else if(raw_variables(i).raw_variable_use == VariableUse::Unused)
            {
                raw_variables_number ++;
            }
        }
    }

    return raw_variables_number;
}


// @todo to hello_world format.

Tensor<type, 1> DataSet::box_plot_from_histogram(Histogram& histogram, const Index& bins_number) const
{
    const Index samples_number = get_training_samples_number();

    const Tensor<type, 1>relative_frequencies = histogram.frequencies.cast<type>() *
           histogram.frequencies.constant(100.0).cast<type>() /
           histogram.frequencies.constant(samples_number).cast<type>();

    // Assuming you have the bin centers and relative frequencies in the following arrays:

    const Tensor<type, 1> bin_centers = histogram.centers;
    const Tensor<type, 1> binFrequencies = relative_frequencies;

    // Calculate the cumulative frequency distribution

    type cumulativeFrequencies[1000];

    cumulativeFrequencies[0] = binFrequencies[0];

    for(int i = 1; i < 1000; i++)
    {
        cumulativeFrequencies[i] = cumulativeFrequencies[i-1] + binFrequencies[i];
    }

    // Calculate total frequency
    type totalFrequency = cumulativeFrequencies[999];

    // Calculate quartile positions
    type Q1Position = type(0.25) * totalFrequency;
    type Q2Position = type(0.5) * totalFrequency;
    type Q3Position = type(0.75) * totalFrequency;

    // Find quartile bin numbers
    int Q1Bin = 0, Q2Bin = 0, Q3Bin = 0;

    for(int i = 0; i < 1000; i++) 
    {
        if(cumulativeFrequencies[i] >= Q1Position) 
        {
            Q1Bin = i;
            break;
        }
    }

    for(int i = 0; i < 1000; i++) 
    {
        if(cumulativeFrequencies[i] >= Q2Position) 
        {
            Q2Bin = i;
            break;
        }
    }

    for(int i = 0; i < 1000; i++) 
    {
        if(cumulativeFrequencies[i] >= Q3Position) 
        {
            Q3Bin = i;
            break;
        }
    }

    // Calculate quartile values

    const type bin_width = bin_centers[1] - bin_centers[0];
    const type q1 = bin_centers[Q1Bin] + ((Q1Position - cumulativeFrequencies[Q1Bin-1]) / binFrequencies[Q1Bin]) * bin_width;
    const type q2 = bin_centers[Q2Bin] + ((Q2Position - cumulativeFrequencies[Q2Bin-1]) / binFrequencies[Q2Bin]) * bin_width;
    const type q3 = bin_centers[Q3Bin] + ((Q3Position - cumulativeFrequencies[Q3Bin-1]) / binFrequencies[Q3Bin]) * bin_width;

    // Calculate the maximum and minimum values
    const type minimum = bin_centers[0] - bin_width / type(2);
    const type maximum = bin_centers[999] + bin_width / type(2);

    // Create a Tensor object with the necessary values for the box plot
    Tensor<type, 1> iqr_values(5);
    iqr_values(0) = minimum;
    iqr_values(1) = q1;
    iqr_values(2) = q2;
    iqr_values(3) = q3;
    iqr_values(4) = maximum;

    return iqr_values;
}


/// Returns the raw_variables of the data set.

Tensor<DataSet::RawVariable, 1> DataSet::get_raw_variables() const
{
    return raw_variables;
}


/// Returns the input raw_variables of the data set.

Tensor<DataSet::RawVariable, 1> DataSet::get_input_raw_variables() const
{
    const Index inputs_number = get_input_raw_variables_number();

    Tensor<RawVariable, 1> input_raw_variables(inputs_number);
    Index input_index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Input)
        {
            input_raw_variables(input_index) = raw_variables(i);
            input_index++;
        }
    }

    return input_raw_variables;
}


/// Returns the input raw_variables of the data set.

Tensor<bool, 1> DataSet::get_input_raw_variables_binary() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Tensor<bool, 1> input_raw_variables_binary(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Input)
            input_raw_variables_binary(i) = true;
        else
            input_raw_variables_binary(i) = false;
    }

    return input_raw_variables_binary;
}


/// Returns the target raw_variables of the data set.

Tensor<DataSet::RawVariable, 1> DataSet::get_target_raw_variables() const
{
    const Index targets_number = get_target_raw_variables_number();

    Tensor<RawVariable, 1> target_raw_variables(targets_number);
    Index target_index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Target)
        {
            target_raw_variables(target_index) = raw_variables(i);
            target_index++;
        }
    }

    return target_raw_variables;
}


/// Returns the used raw_variables of the data set.

Tensor<DataSet::RawVariable, 1> DataSet::get_used_raw_variables() const
{
    const Index used_raw_variables_number = get_used_raw_variables_number();

    const Tensor<Index, 1> used_raw_variables_indices = get_used_raw_variables_indices();

    Tensor<DataSet::RawVariable, 1> used_raw_variables(used_raw_variables_number);

    for(Index i = 0; i < used_raw_variables_number; i++)
    {
        used_raw_variables(i) = raw_variables(used_raw_variables_indices(i));
    }

    return used_raw_variables;
}


/// Returns the number of raw_variables in the data set.

Index DataSet::get_raw_variables_number() const
{
    return raw_variables.size();
}


/// Returns the number of constant raw_variables in the data set.

Index DataSet::get_constant_raw_variables_number() const
{
    Index constant_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Constant)
            constant_number++;
    }

    return constant_number;
}


/// Returns the number of variables in the data set.

Index DataSet::get_variables_number() const
{
    Index variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variables_number += raw_variables(i).categories.size();
        }
        else
        {
            variables_number++;
        }
    }

    return variables_number;
}



/// Returns the number of input variables of the data set.
/// Note that the number of variables does not have to equal the number of raw_variables in the data set,
/// because OpenNN recognizes the categorical raw_variables, separating these categories into variables of the data set.

Index DataSet::get_input_variables_number() const
{
    Index inputs_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).categories_uses.size(); j++)
            {
                if(raw_variables(i).categories_uses(j) == VariableUse::Input)
                {
                    inputs_number++;
                }
            }
        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Input)
        {
            inputs_number++;
        }
    }

    return inputs_number;
}


/// Returns the number of target variables of the data set.

Index DataSet::get_target_variables_number() const
{
    Index targets_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).categories_uses.size(); j++)
            {
                if(raw_variables(i).categories_uses(j) == VariableUse::Target)
                {
                    targets_number++;
                }
            }
        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Target)
        {
            targets_number++;
        }
    }

    return targets_number;
}


/// Returns the number of variables which will neither be used as input nor as target.

Index DataSet::get_unused_variables_number() const
{
    Index unused_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).categories_uses.size(); j++)
            {
                if(raw_variables(i).categories_uses(j) == VariableUse::Unused) unused_number++;
            }

        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Unused)
        {
            unused_number++;
        }
    }

    return unused_number;
}


/// Returns a variable index in the data set with given name.
/// @param name Name of variable.

Index DataSet::get_variable_index(const string& name) const
{
    const Index variables_number = get_variables_number();

    const Tensor<string, 1> variables_names = get_variables_names();

    for(Index i = 0; i < variables_number; i++)
    {
        if(variables_names(i) == name) return i;
    }

    return 0;
}


/// Returns the indices of the unused variables.

Tensor<Index, 1> DataSet::get_unused_variables_indices() const
{
    const Index unused_number = get_unused_variables_number();

    const Tensor<Index, 1> unused_raw_variables_indices = get_unused_raw_variables_indices();

    Tensor<Index, 1> unused_indices(unused_number);

    Index unused_index = 0;
    Index unused_variable_index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index current_categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(raw_variables(i).categories_uses(j) == VariableUse::Unused)
                {
                    unused_indices(unused_index) = unused_variable_index;
                    unused_index++;
                }

                unused_variable_index++;
            }
        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Unused)
        {
            unused_indices(unused_index) = i;
            unused_index++;
            unused_variable_index++;
        }
        else
        {
            unused_variable_index++;
        }
    }

    return unused_indices;
}


/// Returns the indices of the used variables.

Tensor<Index, 1> DataSet::get_used_variables_indices() const
{
    const Index used_number = get_used_variables_number();

    Tensor<Index, 1> used_indices(used_number);

    Index used_index = 0;
    Index used_variable_index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index current_categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(raw_variables(i).categories_uses(j) != VariableUse::Unused)
                {
                    used_indices(used_index) = used_variable_index;
                    used_index++;
                }

                used_variable_index++;
            }
        }
        else if(raw_variables(i).raw_variable_use != VariableUse::Unused)
        {
            used_indices(used_index) = used_variable_index;
            used_index++;
            used_variable_index++;
        }
        else
        {
            used_variable_index++;
        }
    }

    return used_indices;
}


/// Returns the indices of the input variables.

Tensor<Index, 1> DataSet::get_input_variables_indices() const
{
    const Index inputs_number = get_input_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();

    Tensor<Index, 1> input_variables_indices(inputs_number);

    Index input_index = 0;
    Index input_variable_index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index current_categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(raw_variables(i).categories_uses(j) == VariableUse::Input)
                {
                    input_variables_indices(input_index) = input_variable_index;
                    input_index++;
                }

                input_variable_index++;
            }
        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Input) // Binary, numeric
        {
            input_variables_indices(input_index) = input_variable_index;
            input_index++;
            input_variable_index++;
        }
        else
        {
            input_variable_index++;
        }
    }

    return input_variables_indices;
}


/// Returns the number of numeric inputs raw_variables

Index DataSet::get_numerical_input_raw_variables_number() const
{
    Index numeric_input_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if((raw_variables(i).type == RawVariableType::Numeric) && (raw_variables(i).raw_variable_use == VariableUse::Input))
        {
            numeric_input_raw_variables_number++;
        }
    }

    return numeric_input_raw_variables_number;
}


/// Returns the numeric inputs raw_variables indices

Tensor<Index, 1> DataSet::get_numeric_input_raw_variables() const
{
    Index numeric_input_raw_variables_number = get_numerical_input_raw_variables_number();

    Tensor<Index, 1> numeric_raw_variables_indices(numeric_input_raw_variables_number);

    Index numeric_raw_variables_index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if((raw_variables(i).type == RawVariableType::Numeric) && (raw_variables(i).raw_variable_use == VariableUse::Input))
        {
            numeric_raw_variables_indices(numeric_raw_variables_index) = i;
            numeric_raw_variables_index++;
        }
    }

    return numeric_raw_variables_indices;
}


/// Returns the indices of the numeric input variables.

Tensor<Index, 1> DataSet::get_numeric_input_variables_indices() const
{
    Index numeric_input_raw_variables_number = get_numerical_input_raw_variables_number();

    Index numeric_input_index = 0;
    Index input_variable_index = 0;

    Tensor<Index, 1> numeric_input_variables_indices(numeric_input_raw_variables_number);

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index current_categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                input_variable_index++;
            }
        }
        else if((raw_variables(i).type == RawVariableType::Binary) && (raw_variables(i).raw_variable_use == VariableUse::Input))
        {
            input_variable_index++;
        }
        else if((raw_variables(i).type == RawVariableType::Numeric) && (raw_variables(i).raw_variable_use == VariableUse::Input))
        {
            numeric_input_variables_indices(numeric_input_index) = input_variable_index;

            numeric_input_index++;
            input_variable_index++;
        }
        else
        {
            input_variable_index++;
        }
    }

    return numeric_input_variables_indices;
}


/// Returns the indices of the target variables.

Tensor<Index, 1> DataSet::get_target_variables_indices() const
{
    const Index targets_number = get_target_variables_number();

    const Tensor<Index, 1> target_raw_variables_indices = get_target_raw_variables_indices();

    Tensor<Index, 1> target_variables_indices(targets_number);

    Index target_index = 0;
    Index target_variable_index = 0;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index current_categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(raw_variables(i).categories_uses(j) == VariableUse::Target)
                {
                    target_variables_indices(target_index) = target_variable_index;
                    target_index++;
                }

                target_variable_index++;
            }
        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Target) // Binary, numeric
        {
            target_variables_indices(target_index) = target_variable_index;
            target_index++;
            target_variable_index++;
        }
        else
        {
            target_variable_index++;
        }
    }

    return target_variables_indices;
}


/// Sets the uses of the data set raw_variables.
/// @param new_raw_variables_uses String vector that contains the new uses to be set,
/// note that this vector needs to be the size of the number of raw_variables in the data set.

void DataSet::set_raw_variables_uses(const Tensor<string, 1>& new_raw_variables_uses)
{
    const Index new_raw_variables_uses_size = new_raw_variables_uses.size();

    if(new_raw_variables_uses_size != raw_variables.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_raw_variables_uses(const Tensor<string, 1>&) method.\n"
               << "Size of raw_variables uses ("
               << new_raw_variables_uses_size << ") must be equal to raw_variables size ("
               << raw_variables.size() << "). \n";

        throw runtime_error(buffer.str());
    }

    for(Index i = 0; i < new_raw_variables_uses.size(); i++)
    {
        raw_variables(i).set_use(new_raw_variables_uses(i));
    }

    input_variables_dimensions.resize(1);
    input_variables_dimensions.setConstant(get_input_variables_number());

    target_variables_dimensions.resize(1);
    target_variables_dimensions.setConstant(get_target_variables_number());
}


/// Sets the uses of the data set raw_variables.
/// @param new_raw_variables_uses DataSet::VariableUse vector that contains the new uses to be set,
/// note that this vector needs to be the size of the number of raw_variables in the data set.

void DataSet::set_raw_variables_uses(const Tensor<VariableUse, 1>& new_raw_variables_uses)
{
    const Index new_raw_variables_uses_size = new_raw_variables_uses.size();

    if(new_raw_variables_uses_size != raw_variables.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_raw_variables_uses(const Tensor<string, 1>&) method.\n"
               << "Size of raw_variables uses (" << new_raw_variables_uses_size << ") must be equal to raw_variables size (" << raw_variables.size() << "). \n";

        throw runtime_error(buffer.str());
    }

    for(Index i = 0; i < new_raw_variables_uses.size(); i++)
    {
        raw_variables(i).set_use(new_raw_variables_uses(i));
    }

    input_variables_dimensions.resize(1);
    input_variables_dimensions.setConstant(get_input_variables_number());

    target_variables_dimensions.resize(1);
    target_variables_dimensions.setConstant(get_target_variables_number());
}


/// Sets all raw_variables in the data_set as unused raw_variables.

void DataSet::set_raw_variables_unused()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        set_raw_variable_use(i, VariableUse::Unused);
    }
}



void DataSet::set_raw_variables_types(const Tensor<string, 1>& new_raw_variable_types)
{
    const Index new_raw_variable_types_size = new_raw_variable_types.size();

    if(new_raw_variable_types_size != raw_variables.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_all_raw_variable_types(const Tensor<ColumnType, 1>&) method.\n"
               << "Size of raw_variable types (" << new_raw_variable_types_size << ") must be equal to raw_variables size (" << raw_variables.size() << "). \n";

        throw runtime_error(buffer.str());
    }

    for(Index i = 0; i < new_raw_variable_types.size(); i++)
    {
        raw_variables(i).set_type(new_raw_variable_types(i));
    }

}

Tensor<string, 1> DataSet::get_raw_variables_types() const
{
    const Index raw_variables_number = raw_variables.size();

    Tensor<string, 1> column_types(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        column_types(i) = get_raw_variable_type_string(raw_variables(i).type);
    }

    return column_types;
}



void DataSet::set_input_target_raw_variables(const Tensor<Index, 1>& input_raw_variables, const Tensor<Index, 1>& target_raw_variables)
{
    set_raw_variables_unused();

    for(Index i = 0; i < input_raw_variables.size(); i++)
    {
        set_raw_variable_use(input_raw_variables(i), VariableUse::Input);
    }

    for(Index i = 0; i < target_raw_variables.size(); i++)
    {
        set_raw_variable_use(target_raw_variables(i), VariableUse::Target);
    }
}

void DataSet::set_input_target_raw_variables(const Tensor<string, 1>& input_raw_variables, const Tensor<string, 1>& target_raw_variables)
{
    set_raw_variables_unused();

    for(Index i = 0; i < input_raw_variables.size(); i++)
    {
        set_raw_variable_use(input_raw_variables(i), VariableUse::Input);
    }

    for(Index i = 0; i < target_raw_variables.size(); i++)
    {
        set_raw_variable_use(target_raw_variables(i), VariableUse::Target);
    }
}


/// Sets all input raw_variables in the data_set as unused raw_variables.

void DataSet::set_input_raw_variables_unused()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).raw_variable_use == DataSet::VariableUse::Input) set_raw_variable_use(i, VariableUse::Unused);
    }
}



void DataSet::set_input_raw_variables(const Tensor<Index, 1>& input_raw_variables_indices, const Tensor<bool, 1>& input_raw_variables_use)
{
    for(Index i = 0; i < input_raw_variables_indices.size(); i++)
    {
        if(input_raw_variables_use(i)) set_raw_variable_use(input_raw_variables_indices(i), VariableUse::Input);
        else set_raw_variable_use(input_raw_variables_indices(i), VariableUse::Unused);
    }
}


/// Sets the use of a single raw_variable.
/// @param index Index of raw_variable.
/// @param new_use Use for that raw_variable.

void DataSet::set_raw_variable_use(const Index& index, const VariableUse& new_use)
{
    raw_variables(index).raw_variable_use = new_use;

    if(raw_variables(index).type == RawVariableType::Categorical)
    {
        raw_variables(index).set_categories_uses(new_use);
    }
}

void DataSet::set_raw_variables_unused(const Tensor<Index, 1>& unused_raw_variables_index)
{
    for(Index i = 0; i < unused_raw_variables_index.size(); i++)
    {
        set_raw_variable_use(unused_raw_variables_index(i), VariableUse::Unused);
    }
}

/// Sets the use of a single raw_variable.
/// @param name Name of raw_variable.
/// @param new_use Use for that raw_variable.

void DataSet::set_raw_variable_use(const string& name, const VariableUse& new_use)
{
    const Index index = get_raw_variable_index(name);

    set_raw_variable_use(index, new_use);
}

void DataSet::set_raw_variable_type(const Index& index, const RawVariableType& new_type)
{
    raw_variables[index].type = new_type;
}


void DataSet::set_raw_variable_type(const string& name, const RawVariableType& new_type)
{
    const Index index = get_raw_variable_index(name);

    set_raw_variable_type(index, new_type);
}


void DataSet::set_all_raw_variables_type(const RawVariableType& new_type)
{
    for(Index i = 0; i < raw_variables.size(); i ++)
        raw_variables[i].type = new_type;
}


/// This method set the name of a single variable.
/// @param index Index of variable.
/// @param new_name Name of variable.

void DataSet::set_variable_name(const Index& variable_index, const string& new_variable_name)
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).get_categories_number(); j++)
            {
                if(index == variable_index)
                {
                    raw_variables(i).categories(j) = new_variable_name;
                    return;
                }
                else
                {
                    index++;
                }
            }
        }
        else
        {
            if(index == variable_index)
            {
                raw_variables(i).name = new_variable_name;
                return;
            }
            else
            {
                index++;
            }
        }
    }
}


/// Sets new names for the variables in the data set from a vector of strings.
/// The size of that vector must be equal to the total number of variables.
/// @param new_names Name of variables.

void DataSet::set_variables_names(const Tensor<string, 1>& new_variables_names)
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).get_categories_number(); j++)
            {
                raw_variables(i).categories(j) = new_variables_names(index);
                index++;
            }
        }
        else
        {
            raw_variables(i).name = new_variables_names(index);
            index++;
        }
    }
}


void DataSet::set_variables_names_from_raw_variables(const Tensor<string, 1>& new_variables_names,
                                               const Tensor<DataSet::RawVariable, 1>& new_raw_variables)
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            raw_variables(i).categories.resize(new_raw_variables(i).get_categories_number());

            for(Index j = 0; j < new_raw_variables(i).get_categories_number(); j++)
            {
                raw_variables(i).categories(j) = new_variables_names(index);
                index++;
            }
        }
        else
        {
            raw_variables(i).name = new_variables_names(index);
            index++;
        }
    }
}


/// Sets new names for the raw_variables in the data set from a vector of strings.
/// The size of that vector must be equal to the total number of variables.
/// @param new_names Name of variables.

void DataSet::set_raw_variables_names(const Tensor<string, 1>& new_names)
{
    const Index new_names_size = new_names.size();
    const Index raw_variables_number = get_raw_variables_number();

    if(new_names_size != raw_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_raw_variables_names(const Tensor<string, 1>&).\n"
               << "Size of names (" << new_names.size() << ") is not equal to raw_variables number (" << raw_variables_number << ").\n";

        throw runtime_error(buffer.str());
    }

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables(i).name = get_trimmed(new_names(i));
    }
}


/// Sets all the variables in the data set as input variables.

void DataSet::set_input()
{
    for(Index i = 0; i < raw_variables.size(); i++)
    {
        if(raw_variables(i).type == RawVariableType::Constant) continue;

        raw_variables(i).set_use(VariableUse::Input);
    }
}


/// Sets all the variables in the data set as target variables.

void DataSet::set_target()
{
    for(Index i = 0; i < raw_variables.size(); i++)
    {
        raw_variables(i).set_use(VariableUse::Target);
    }
}


/// Sets all the variables in the data set as unused variables.

void DataSet::set_variables_unused()
{
    for(Index i = 0; i < raw_variables.size(); i++)
    {
        raw_variables(i).set_use(VariableUse::Unused);
    }
}


/// Sets a new number of variables in the variables object.
/// All variables are set as inputs but the last one, which is set as targets.
/// @param new_raw_variables_number Number of variables.

void DataSet::set_raw_variables_number(const Index& new_raw_variables_number)
{
    raw_variables.resize(new_raw_variables_number);

    set_default_raw_variables_uses();
}


void DataSet::set_raw_variables_scalers(const Scaler& scalers)
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables(i).scaler = scalers;
    }
}


void DataSet::set_raw_variables_scalers(const Tensor<Scaler, 1>& new_scalers)
{
    const Index raw_variables_number = get_raw_variables_number();

    if(new_scalers.size() != raw_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_raw_variables_scalers(const Tensor<Scaler, 1>& new_scalers) method.\n"
               << "Size of raw_variable scalers(" << new_scalers.size() << ") has to be the same as raw_variables numbers(" << raw_variables_number << ").\n";

        throw runtime_error(buffer.str());
    }

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables(i).scaler = new_scalers[i];
    }

}


void DataSet::set_binary_simple_raw_variables()
{
    bool is_binary = true;

    Index variable_index = 0;

    Index different_values = 0;

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables.size(); raw_variable_index++)
    {
        if(raw_variables(raw_variable_index).type == RawVariableType::Numeric)
        {
            Tensor<type, 1> values(3);
            values.setRandom();
            different_values = 0;
            is_binary = true;

            for(Index row_index = 0; row_index < data.dimension(0); row_index++)
            {
                if(!isnan(data(row_index, variable_index))
                        && data(row_index, variable_index) != values(0)
                        && data(row_index, variable_index) != values(1))
                {
                    values(different_values) = data(row_index, variable_index);
                    different_values++;
                }

                if(row_index == (data.dimension(0)-1))
                {
                    if(different_values == 1)
                    {
                        is_binary = false;
                        break;
                    }
                }

                if(different_values > 2)
                {
                    is_binary = false;
                    break;
                }
            }

            if(is_binary)
            {
                raw_variables(raw_variable_index).type = RawVariableType::Binary;
                scale_minimum_maximum_binary(data, values(0), values(1), variable_index);
                raw_variables(raw_variable_index).categories.resize(2);

                if((abs(values(0)-type(0))<NUMERIC_LIMITS_MIN) && (abs(values(1)-type(1))<NUMERIC_LIMITS_MIN))
                {
                    if(abs(values(0) - int(values(0))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(1) = to_string(int(values(0)));
                    else
                        raw_variables(raw_variable_index).categories(1) = to_string(values(0));
                    if(abs(values(1) - int(values(1))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(0) = to_string(int(values(1)));
                    else
                        raw_variables(raw_variable_index).categories(0) = to_string(values(1));

                }
                else if(abs(values(0) - type(1))<NUMERIC_LIMITS_MIN && abs(values(1) - type(0))<NUMERIC_LIMITS_MIN)
                {
                    if(abs(values(0) - int(values(0))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(0) = to_string(int(values(0)));
                    else
                        raw_variables(raw_variable_index).categories(0) = to_string(values(0));
                    if(abs(values(1) - int(values(1))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(1) = to_string(int(values(1)));
                    else
                        raw_variables(raw_variable_index).categories(1) = to_string(values(1));
                }
                else if(values(0) > values(1))
                {
                    if(abs(values(0) - int(values(0))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(0) = to_string(int(values(0)));
                    else
                        raw_variables(raw_variable_index).categories(0) = to_string(values(0));
                    if(abs(values(1) - int(values(1))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(1) = to_string(int(values(1)));
                    else
                        raw_variables(raw_variable_index).categories(1) = to_string(values(1));
                }
                else if(values(0) < values(1))
                {
                    if(abs(values(0) - int(values(0))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(1) = to_string(int(values(0)));
                    else
                        raw_variables(raw_variable_index).categories(1) = to_string(values(0));
                    if(abs(values(1) - int(values(1))) < NUMERIC_LIMITS_MIN)
                        raw_variables(raw_variable_index).categories(0) = to_string(int(values(1)));
                    else
                        raw_variables(raw_variable_index).categories(0) = to_string(values(1));
                }

                const VariableUse raw_variable_use = raw_variables(raw_variable_index).raw_variable_use;
                raw_variables(raw_variable_index).categories_uses.resize(2);
                raw_variables(raw_variable_index).categories_uses(0) = raw_variable_use;
                raw_variables(raw_variable_index).categories_uses(1) = raw_variable_use;
            }

            variable_index++;
        }
        else if(raw_variables(raw_variable_index).type == RawVariableType::Binary)
        {
            Tensor<string,1> positive_words(4);
            Tensor<string,1> negative_words(4);

            positive_words.setValues({"yes","positive","+","true"});
            negative_words.setValues({"no","negative","-","false"});

            string first_category = raw_variables(raw_variable_index).categories(0);
            string original_first_category = raw_variables(raw_variable_index).categories(0);
            trim(first_category);

            string second_category = raw_variables(raw_variable_index).categories(1);
            string original_second_category = raw_variables(raw_variable_index).categories(1);
            trim(second_category);

            transform(first_category.begin(), first_category.end(), first_category.begin(), ::tolower);
            transform(second_category.begin(), second_category.end(), second_category.begin(), ::tolower);

            if( contains(positive_words, first_category) && contains(negative_words, second_category) )
            {
                raw_variables(raw_variable_index).categories(0) = original_first_category;
                raw_variables(raw_variable_index).categories(1) = original_second_category;
            }
            else if( contains(positive_words, second_category) && contains(negative_words, first_category) )
            {
                raw_variables(raw_variable_index).categories(0) = original_second_category;
                raw_variables(raw_variable_index).categories(1) = original_first_category;
            }

            variable_index++;
        }
        else if(raw_variables(raw_variable_index).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(raw_variable_index).get_categories_number();
        }
        else
        {
            variable_index++;
        }
    }

    if(display) cout << "Binary raw_variables checked " << endl;
}


void DataSet::check_constant_raw_variables()
{
    if(display) cout << "Checking constant raw_variables..." << endl;

    Index variable_index = 0;

    for(Index raw_variable = 0; raw_variable < get_raw_variables_number(); raw_variable++)
    {
        if(raw_variables(raw_variable).type == RawVariableType::Numeric)
        {
            const Tensor<type, 1> numeric_column = data.chip(variable_index, 1);

            if(is_constant(numeric_column))
            {
                raw_variables(raw_variable).type = RawVariableType::Constant;
                raw_variables(raw_variable).raw_variable_use = VariableUse::Unused;
            }
            variable_index++;
        }
        else if(raw_variables(raw_variable).type == RawVariableType::DateTime)
        {
            raw_variables(raw_variable).raw_variable_use = VariableUse::Unused;
            variable_index++;
        }
        else if(raw_variables(raw_variable).type == RawVariableType::Constant)
        {
            variable_index++;
        }
        else if(raw_variables(raw_variable).type == RawVariableType::Binary)
        {
            if(raw_variables(raw_variable).get_categories_number() == 1)
            {
                raw_variables(raw_variable).type = RawVariableType::Constant;
                raw_variables(raw_variable).raw_variable_use = VariableUse::Unused;
            }

            variable_index++;
        }
        else if(raw_variables(raw_variable).type == RawVariableType::Categorical)
        {
            if(raw_variables(raw_variable).get_categories_number() == 1)
            {
                raw_variables(raw_variable).type = RawVariableType::Constant;
                raw_variables(raw_variable).raw_variable_use = VariableUse::Unused;
            }

            variable_index += raw_variables(raw_variable).get_categories_number();
        }
    }
}


Tensor<type, 2> DataSet::transform_binary_column(const Tensor<type, 1>& raw_variable) const
{
    const Index rows_number = raw_variable.dimension(0);

    Tensor<type, 2> new_column(rows_number , 2);
    new_column.setZero();

    for(Index i = 0; i < rows_number; i++)
    {
        if(abs(raw_variable(i) - type(1)) < type(NUMERIC_LIMITS_MIN))
        {
            new_column(i,1) = type(1);
        }
        else if(abs(raw_variable(i)) < type(NUMERIC_LIMITS_MIN))
        {
            new_column(i,0) = type(1);
        }
        else
        {
            new_column(i,0) = type(NAN);
            new_column(i,1) = type(NAN);
        }
    }

    return new_column;
}


/// Sets new input dimensions in the data set.

void DataSet::set_input_variables_dimensions(const Tensor<Index, 1>& new_inputs_dimensions)
{
    input_variables_dimensions = new_inputs_dimensions;
}


void DataSet::set_target_variables_dimensions(const Tensor<Index, 1>& new_targets_dimensions)
{
    target_variables_dimensions = new_targets_dimensions;
}


/// Returns true if the data matrix is empty, and false otherwise.

bool DataSet::is_empty() const
{
    if(data.dimension(0) == 0 || data.dimension(1) == 0)
    {
        return true;
    }

    return false;
}


/// Returns a reference to the data matrix in the data set.
/// The number of rows is equal to the number of samples.
/// The number of raw_variables is equal to the number of variables.

const Tensor<type, 2>& DataSet::get_data() const
{
    return data;
}


Tensor<type, 2>* DataSet::get_data_p()
{
    return &data;
}


/// Returns a string with the method used.

DataSet::MissingValuesMethod DataSet::get_missing_values_method() const
{
    return missing_values_method;
}


/// Returns the name of the data file.

const string& DataSet::get_data_source_path() const
{
    return data_source_path;
}


/// Returns true if the first line of the data file has a header with the names of the variables, and false otherwise.

const bool& DataSet::get_header_line() const
{
    return has_raw_variables_names;
}


/// Returns true if the data file has rows label, and false otherwise.

const bool& DataSet::get_rows_label() const
{
    return has_rows_labels;
}


Tensor<string, 1> DataSet::get_rows_label_tensor() const
{
    return rows_labels;
}


Tensor<string, 1> DataSet::get_testing_rows_label_tensor()
{
    const Index testing_samples_number = get_testing_samples_number();
    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();
    Tensor<string, 1> testing_rows_label(testing_samples_number);

    for(Index i = 0; i < testing_samples_number; i++)
    {
        testing_rows_label(i) = rows_labels(testing_indices(i));
    }

    return testing_rows_label;
}


Tensor<string, 1> DataSet::get_selection_rows_label_tensor()
{
    const Index selection_samples_number = get_selection_samples_number();
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();
    Tensor<string, 1> selection_rows_label(selection_samples_number);

    for(Index i = 0; i < selection_samples_number; i++)
    {
        selection_rows_label(i) = rows_labels(selection_indices(i));
    }

    return selection_rows_label;
}


/// Returns the separator to be used in the data file.

const DataSet::Separator& DataSet::get_separator() const
{
    return separator;
}


/// Returns the string which will be used as separator in the data file.

char DataSet::get_separator_char() const
{
    switch(separator)
    {
    case Separator::Space:
        return ' ';

    case Separator::Tab:
        return '\t';

    case Separator::Comma:
        return ',';

    case Separator::Semicolon:
        return ';';

    default:
        return char();
    }

}


/// Returns the string which will be used as separator in the data file.

string DataSet::get_separator_string() const
{
    switch(separator)
    {
    case Separator::Space:
        return "Space";

    case Separator::Tab:
        return "Tab";

    case Separator::Comma:
        return "Comma";

    case Separator::Semicolon:
        return "Semicolon";

    default:
        return string();
    }
}


/// Returns the string codification used in the data file.

const DataSet::Codification DataSet::get_codification() const
{
    return codification;
}


/// Returns a string that contains the string codification used in the data file.

const string DataSet::get_codification_string() const
{
    switch(codification)
    {
    case Codification::UTF8:
        return "UTF-8";

    case Codification::SHIFT_JIS:
        return "SHIFT_JIS";

    default:
        return "UTF-8";
    }
}


/// Returns the string which will be used as label for the missing values in the data file.

const string& DataSet::get_missing_values_label() const
{
    return missing_values_label;
}


/// Returns a value of the scaling-unscaling method enumeration from a string containing the name of that method.
/// @param scaling_unscaling_method String with the name of the scaling and unscaling method.

Scaler DataSet::get_scaling_unscaling_method(const string& scaling_unscaling_method)
{
    if (scaling_unscaling_method == "NoScaling")
    {
        return Scaler::NoScaling;
    }
    else if (scaling_unscaling_method == "MinimumMaximum")
    {
        return Scaler::MinimumMaximum;
    }
    else if (scaling_unscaling_method == "Logarithmic")
    {
        return Scaler::Logarithm;
    }
    else if (scaling_unscaling_method == "MeanStandardDeviation")
    {
        return Scaler::MeanStandardDeviation;
    }
    else if (scaling_unscaling_method == "StandardDeviation")
    {
        return Scaler::StandardDeviation;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
            << "static Scaler get_scaling_unscaling_method(const string).\n"
            << "Unknown scaling-unscaling method: " << scaling_unscaling_method << ".\n";

        throw runtime_error(buffer.str());
    }
}


/// Returns a matrix with the training samples in the data set.
/// The number of rows is the number of training
/// The number of raw_variables is the number of variables.

Tensor<type, 2> DataSet::get_training_data() const
{
    const Tensor<Index, 1> variables_indices = get_used_variables_indices();

    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    return get_subtensor_data(training_indices, variables_indices);
}


/// Returns a matrix with the selection samples in the data set.
/// The number of rows is the number of selection
/// The number of raw_variables is the number of variables.

Tensor<type, 2> DataSet::get_selection_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Index variables_number = get_variables_number();

    Tensor<Index, 1> variables_indices;
    initialize_sequential(variables_indices, 0, 1, variables_number-1);

    return get_subtensor_data(selection_indices, variables_indices);
}


/// Returns a matrix with the testing samples in the data set.
/// The number of rows is the number of testing
/// The number of raw_variables is the number of variables.

Tensor<type, 2> DataSet::get_testing_data() const
{
    const Index variables_number = get_variables_number();

    Tensor<Index, 1> variables_indices;
    initialize_sequential(variables_indices, 0, 1, variables_number-1);

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    return get_subtensor_data(testing_indices, variables_indices);
}


/// Returns a matrix with the input variables in the data set.
/// The number of rows is the number of
/// The number of raw_variables is the number of input variables.

Tensor<type, 2> DataSet::get_input_data() const
{
    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;
    initialize_sequential(indices, 0, 1, samples_number-1);

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(indices, input_variables_indices);
}


/// Returns a matrix with the target variables in the data set.
/// The number of rows is the number of
/// The number of raw_variables is the number of target variables.

Tensor<type, 2> DataSet::get_target_data() const
{
    const Tensor<Index, 1> indices = get_used_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(indices, target_variables_indices);
}


/// Returns a tensor with the input variables in the data set.
/// The number of rows is the number of
/// The number of raw_variables is the number of input variables.

Tensor<type, 2> DataSet::get_input_data(const Tensor<Index, 1>& samples_indices) const
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(samples_indices, input_variables_indices);
}


/// Returns a tensor with the target variables in the data set.
/// The number of rows is the number of
/// The number of raw_variables is the number of input variables.

Tensor<type, 2> DataSet::get_target_data(const Tensor<Index, 1>& samples_indices) const
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(samples_indices, target_variables_indices);
}


/// Returns a matrix with training samples and input variables.
/// The number of rows is the number of training
/// The number of raw_variables is the number of input variables.

Tensor<type, 2> DataSet::get_training_input_data() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(training_indices, input_variables_indices);
}


/// Returns a tensor with training samples and target variables.
/// The number of rows is the number of training
/// The number of raw_variables is the number of target variables.

Tensor<type, 2> DataSet::get_training_target_data() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1>& target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(training_indices, target_variables_indices);
}


/// Returns a tensor with selection samples and input variables.
/// The number of rows is the number of selection
/// The number of raw_variables is the number of input variables.

Tensor<type, 2> DataSet::get_selection_input_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(selection_indices, input_variables_indices);
}


/// Returns a tensor with selection samples and target variables.
/// The number of rows is the number of selection
/// The number of raw_variables is the number of target variables.

Tensor<type, 2> DataSet::get_selection_target_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(selection_indices, target_variables_indices);
}


/// Returns a tensor with testing samples and input variables.
/// The number of rows is the number of testing
/// The number of raw_variables is the number of input variables.

Tensor<type, 2> DataSet::get_testing_input_data() const
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    return get_subtensor_data(testing_indices, input_variables_indices);
}


/// Returns a tensor with testing samples and target variables.
/// The number of rows is the number of testing
/// The number of raw_variables is the number of target variables.

Tensor<type, 2> DataSet::get_testing_target_data() const
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    return get_subtensor_data(testing_indices, target_variables_indices);
}


/// Returns the inputs and target values of a single sample in the data set.
/// @param index Index of the sample.

Tensor<type, 1> DataSet::get_sample_data(const Index& index) const
{

#ifdef OPENNN_DEBUG

    const Index samples_number = get_samples_number();

    if(index >= samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_sample(const Index&) const method.\n"
               << "Index of sample (" << index << ") must be less than number of samples (" << samples_number << ").\n";

        throw runtime_error(buffer.str());
    }

#endif

    // Get sample

    return data.chip(index,0);
}


/// Returns the inputs and target values of a single sample in the data set.
/// @param sample_index Index of the sample.
/// @param variables_indices Indices of the variables.

Tensor<type, 1> DataSet::get_sample_data(const Index& sample_index, const Tensor<Index, 1>& variables_indices) const
{
#ifdef OPENNN_DEBUG

    const Index samples_number = get_samples_number();

    if(sample_index >= samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_sample(const Index&, const Tensor<Index, 1>&) const method.\n"
               << "Index of sample must be less than number of \n";

        throw runtime_error(buffer.str());
    }

#endif

    const Index variables_number = variables_indices.size();

    Tensor<type, 1 > row(variables_number);

    for(Index i = 0; i < variables_number; i++)
    {
        Index variable_index = variables_indices(i);

        row(i) = data(sample_index, variable_index);
    }

    return row;
}


/// Returns the inputs values of a single sample in the data set.
/// @param sample_index Index of the sample.

Tensor<type, 2> DataSet::get_sample_input_data(const Index&  sample_index) const
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> inputs(1, input_variables_number);

    for(Index i = 0; i < input_variables_number; i++)
    {
        inputs(0, i) = data(sample_index, input_variables_indices(i));
    }

    return inputs;
}


/// Returns the target values of a single sample in the data set.
/// @param sample_index Index of the sample.

Tensor<type, 2> DataSet::get_sample_target_data(const Index&  sample_index) const
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(Tensor<Index, 1>(sample_index), target_variables_indices);
}


/// Returns the index from the raw_variable with a given name,
/// @param raw_variables_names Names of the raw_variables we want to know the index.

Tensor<Index, 1> DataSet::get_raw_variables_index(const Tensor<string, 1>& raw_variables_names) const
{
    Tensor<Index, 1> raw_variables_index(raw_variables_names.size());

    for(Index i = 0; i < raw_variables_names.size(); i++)
    {
        raw_variables_index(i) = get_raw_variable_index(raw_variables_names(i));
    }

    return raw_variables_index;
}

/// Returns the index of the raw_variable with the given name.
/// @param column_name Name of the raw_variable to be found.

Index DataSet::get_raw_variable_index(const string& column_name) const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).name == column_name) return i;
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: DataSet class.\n"
           << "Index get_raw_variable_index(const string&) const method.\n"
           << "Cannot find " << column_name << "\n";

    throw runtime_error(buffer.str());
}


/// Returns the index of the raw_variable to which a variable index belongs.
/// @param variable_index Index of the variable to be found.

Index DataSet::get_raw_variable_index(const Index& variable_index) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index total_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            total_variables_number += raw_variables(i).get_categories_number();
        }
        else
        {
            total_variables_number++;
        }

        if((variable_index+1) <= total_variables_number) return i;
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: DataSet class.\n"
           << "Index get_raw_variable_index(const type&) const method.\n"
           << "Cannot find variable index: " << variable_index << ".\n";

    throw runtime_error(buffer.str());
}

/// Returns the indices of a variable in the data set.
/// Note that the number of variables does not have to equal the number of raw_variables in the data set,
/// because OpenNN recognizes the categorical raw_variables, separating these categories into variables of the data set.

Tensor<Index, 1> DataSet::get_numeric_variable_indices(const Index& raw_variable_index) const
{
    Index index = 0;

    for(Index i = 0; i < raw_variable_index; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            index += raw_variables(i).categories.size();
        }
        else
        {
            index++;
        }
    }

    if(raw_variables(raw_variable_index).type == RawVariableType::Categorical)
    {
        Tensor<Index, 1> variable_indices(raw_variables(raw_variable_index).categories.size());

        for(Index j = 0; j<raw_variables(raw_variable_index).categories.size(); j++)
        {
            variable_indices(j) = index+j;
        }

        return variable_indices;
    }
    else
    {
        Tensor<Index, 1> indices(1);
        indices.setConstant(index);

        return indices;
    }
}


Tensor<Index, 1> DataSet::get_categorical_to_indices(const Index& raw_variable_index) const
{
    Tensor<type, 2> one_hot_data = get_raw_variable_data(raw_variable_index);

    Index rows_number = one_hot_data.dimension(0);
    Index categories_number = one_hot_data.dimension(1);

    Tensor<Index, 1> indices(rows_number);

    for(Index i = 0; i < rows_number; ++i) {
        for(Index j = 0; j < categories_number; ++j) {
            if(one_hot_data(i, j) == 1)
            {
                indices(i) = j + 1;
                break;
            }
        }
    }

    return indices;
}


/// Returns the data from the data set raw_variable with a given index,
/// these data can be stored in a matrix or a vector depending on whether the raw_variable is categorical or not(respectively).
/// @param raw_variable_index Index of the raw_variable.

Tensor<type, 2> DataSet::get_raw_variable_data(const Index& raw_variable_index) const
{
    Index raw_variables_number = 1;
    const Index rows_number = data.dimension(0);

    if(raw_variables(raw_variable_index).type == RawVariableType::Categorical)
    {
        raw_variables_number = raw_variables(raw_variable_index).get_categories_number();
    }

    const Eigen::array<Index, 2> extents = {rows_number, raw_variables_number};
    const Eigen::array<Index, 2> offsets = {0, get_numeric_variable_indices(raw_variable_index)(0)};

    return data.slice(offsets, extents);
}


Tensor<type, 1> DataSet::get_sample(const Index& sample_index) const
{
    if(sample_index >= data.dimension(0))
    {
        throw runtime_error("Sample index out of bounds.");
    }
    return data.chip(sample_index, 0);
}


void DataSet::add_sample(const Tensor<type, 1>& sample)
{
    Index current_samples = data.dimension(0);

    if(current_samples == 0)
    {
        Tensor<type, 2> new_data(1, sample.dimension(0));
        new_data.chip(0, 0) = sample;
        data = new_data;
        return;
    }

    if(sample.dimension(0) != data.dimension(1))
    {
        throw runtime_error("Sample size doesn't match data raw_variable size.");
    }

    Tensor<type, 2> new_data(current_samples + 1, data.dimension(1));

    for(Index i = 0; i < current_samples; ++i)
    {
        new_data.chip(i, 0) = data.chip(i, 0);
    }

    new_data.chip(current_samples, 0) = sample;

    data = new_data;
}


string DataSet::get_sample_category(const Index& sample_index, const Index& column_index_start) const
{

    if(raw_variables[column_index_start].type != RawVariableType::Categorical)
    {
        throw runtime_error("The specified raw_variable is not of categorical type.");
    }


    for(Index raw_variable_index = column_index_start; raw_variable_index < raw_variables.size(); ++raw_variable_index)
    {
        if(data(sample_index, raw_variable_index) == 1)
        {
            return raw_variables[column_index_start].categories(raw_variable_index - column_index_start);
        }
    }

    throw runtime_error("Sample does not have a valid one-hot encoded category.");
}


/// Returns the data from the data set raw_variable with a given index,
/// these data can be stored in a matrix or a vector depending on whether the raw_variable is categorical or not, respectively.
/// @param raw_variable_index Index of the raw_variable.

Tensor<type, 2> DataSet::get_raw_variables_data(const Tensor<Index, 1>& selected_raw_variable_indices) const
{
    const Index raw_variables_number = selected_raw_variable_indices.size();
    const Index rows_number = data.dimension(0);

    Tensor<type, 2> data_slice(rows_number, raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        Eigen::array<Index, 1> rows_number_to_reshape{{rows_number}};

        Tensor<type, 2> single_raw_variable_data = get_raw_variable_data(selected_raw_variable_indices(i));

        Tensor<type, 1> column_data = single_raw_variable_data.reshape(rows_number_to_reshape);

        data_slice.chip(i,1) = column_data;
    }

    return data_slice;
}


/// Returns the data from the data set raw_variable with a given index,
/// these data can be stored in a matrix or a vector depending on whether the raw_variable is categorical or not(respectively).
/// @param raw_variable_index Index of the raw_variable.
/// @param rows_indices Rows of the indices.

Tensor<type, 2> DataSet::get_raw_variable_data(const Index& raw_variable_index, const Tensor<Index, 1>& rows_indices) const
{
    return get_subtensor_data(rows_indices, get_numeric_variable_indices(raw_variable_index));
}


/// Returns the data from the data set raw_variable with a given name,
/// these data can be stored in a matrix or a vector depending on whether the raw_variable is categorical or not(respectively).
/// @param column_name Name of the raw_variable.

Tensor<type, 2> DataSet::get_raw_variable_data(const string& column_name) const
{
    const Index raw_variable_index = get_raw_variable_index(column_name);

    return get_raw_variable_data(raw_variable_index);
}


/// Returns all the samples of a single variable in the data set.
/// @param index Index of the variable.

Tensor<type, 1> DataSet::get_variable_data(const Index& index) const
{
    return data.chip(index, 1);
}


/// Returns all the samples of a single variable in the data set.
/// @param variable_name Name of the variable.

Tensor<type, 1> DataSet::get_variable_data(const string& variable_name) const
{
    const Tensor<string, 1> variable_names = get_variables_names();

    Index size = 0;

    for(Index i = 0; i < variable_names.size(); i++)
    {
        if(variable_names(i) ==  variable_name) size++;
    }

    Tensor<Index, 1> variable_index(size);

    Index index = 0;

    for(Index i = 0; i < variable_names.size(); i++)
    {
        if(variable_names(i) ==  variable_name)
        {
            variable_index(index) = i;

            index++;
        }
    }

#ifdef OPENNN_DEBUG

    const Index variables_size = variable_index.size();

    if(variables_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&) const method.\n"
               << "Variable: " << variable_name << " does not exist.\n";

        throw runtime_error(buffer.str());
    }

    if(variables_size > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&) const method.\n"
               << "Variable: " << variable_name << " appears more than once in the data set.\n";

        throw runtime_error(buffer.str());
    }

#endif

    return data.chip(variable_index(0), 1);
}


/// Returns a given set of samples of a single variable in the data set.
/// @param variable_index Index of the variable.
/// @param samples_indices Indices of the

Tensor<type, 1> DataSet::get_variable_data(const Index& variable_index, const Tensor<Index, 1>& samples_indices) const
{
    const Index samples_indices_size = samples_indices.size();

    Tensor<type, 1 > raw_variable(samples_indices_size);

    for(Index i = 0; i < samples_indices_size; i++)
    {
        Index sample_index = samples_indices(i);

        raw_variable(i) = data(sample_index, variable_index);
    }

    return raw_variable;
}


/// Returns a given set of samples of a single variable in the data set.
/// @param variable_name Name of the variable.
/// @param samples_indices Indices of the

Tensor<type, 1> DataSet::get_variable_data(const string& variable_name, const Tensor<Index, 1>& samples_indices) const
{
    const Tensor<string, 1> variable_names = get_variables_names();

    Index size = 0;

    for(Index i = 0; i < variable_names.size(); i++)
    {
        if(variable_names(i) ==  variable_name) size++;
    }

    Tensor<Index, 1> variable_index(size);

    Index index = 0;

    for(Index i = 0; i < variable_names.size(); i++)
    {
        if(variable_names(i) ==  variable_name)
        {
            variable_index(index) = i;

            index++;
        }
    }

#ifdef OPENNN_DEBUG

    const Index variables_size = variable_index.size();

    if(variables_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&) const method.\n"
               << "Variable: " << variable_name << " does not exist.\n";

        throw runtime_error(buffer.str());
    }

    if(variables_size > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&, const Tensor<Index, 1>&) const method.\n"
               << "Variable: " << variable_name << " appears more than once in the data set.\n";

        throw runtime_error(buffer.str());
    }

#endif

    const Index samples_indices_size = samples_indices.size();

    Tensor<type, 1 > raw_variable(samples_indices_size);

    for(Index i = 0; i < samples_indices_size; i++)
    {
        Index sample_index = samples_indices(i);

        raw_variable(i) = data(sample_index, variable_index(0));
    }

    return raw_variable;
}


Tensor<Tensor<string, 1>, 1> DataSet::get_data_file_preview() const
{
    return data_file_preview;
}


Tensor<type, 2> DataSet::get_subtensor_data(const Tensor<Index, 1> & rows_indices, const Tensor<Index, 1> & variables_indices) const
{
    const Index rows_number = rows_indices.size();
    const Index variables_number = variables_indices.size();

    Tensor<type, 2> subtensor(rows_number, variables_number);

    Index row_index;
    Index variable_index;

    const Tensor<type, 2>& data = get_data();

    for(Index i = 0; i < rows_number; i++)
    {
        row_index = rows_indices(i);

        for(Index j = 0; j < variables_number; j++)
        {
            variable_index = variables_indices(j);

            subtensor(i, j) = data(row_index, variable_index);
        }
    }

    return subtensor;
}


/// Sets zero samples and zero variables in the data set.

void DataSet::set()
{
    thread_pool = nullptr;
    thread_pool_device = nullptr;

    data.resize(0,0);

    samples_uses.resize(0);

    raw_variables.resize(0);

    //time_series_raw_variables.resize(0);

    raw_variables_missing_values_number.resize(0);
}


void DataSet::set(const Tensor<type, 1>& inputs_variables_dimensions, const Index& channels_number)
{
    // Set data

    const Index variables_number = inputs_variables_dimensions.dimension(0) + channels_number;
    const Index samples_number = 1;
    data.resize(samples_number, variables_number);

    // Set raw_variables

    for(Index i = 0; i < inputs_variables_dimensions.dimension(0);++i)
    {
        for(Index j = 0; j < inputs_variables_dimensions(i);++j)
        {
            raw_variables(i+j).name = "column_" + to_string(i+j+1);
            raw_variables(i+j).raw_variable_use = VariableUse::Input;
            raw_variables(i+j).type = RawVariableType::Numeric;
        }
    }

    for(Index i = 0; i < channels_number;++i)
    {
        raw_variables(inputs_variables_dimensions.dimension(0) + i).name = "column_" + to_string(inputs_variables_dimensions.dimension(0) + i + 1);
        raw_variables(inputs_variables_dimensions.dimension(0) + i).raw_variable_use = VariableUse::Target;
        raw_variables(inputs_variables_dimensions.dimension(0) + i).type = RawVariableType::Numeric;
    }
}


void DataSet::set(const string& data_source_path, const char& separator, const bool& new_has_raw_variables_names)
{
    set();

    set_default();

    set_data_source_path(data_source_path);

    set_separator(separator);

    set_has_raw_variables_names(new_has_raw_variables_names);

    read_csv();

    set_default_raw_variables_scalers();

    set_default_raw_variables_uses();
}


void DataSet::set(const string& data_source_path, const char& separator, const bool& new_has_raw_variables_names, const DataSet::Codification& new_codification)
{
    set();

    set_default();

    set_data_source_path(data_source_path);

    set_separator(separator);

    set_has_raw_variables_names(new_has_raw_variables_names);

    set_codification(new_codification);

    read_csv();

    set_default_raw_variables_scalers();

    set_default_raw_variables_uses();
}


/// Sets all variables from a data matrix.
/// @param new_data Data matrix.

void DataSet::set(const Tensor<type, 2>& new_data)
{
    data_source_path = "";
    
    const Index variables_number = new_data.dimension(1);
    const Index samples_number = new_data.dimension(0);

    set(samples_number, variables_number);

    data = new_data;

    set_default_raw_variables_uses();

}


/// Sets new numbers of samples and variables in the inputs targets data set.
/// All the variables are set as inputs.
/// @param new_samples_number Number of
/// @param new_variables_number Number of variables.

void DataSet::set(const Index& new_samples_number, const Index& new_variables_number)
{
    data.resize(new_samples_number, new_variables_number);

    raw_variables.resize(new_variables_number);

    for(Index index = 0; index < new_variables_number-1; index++)
    {
        raw_variables(index).name = "column_" + to_string(index+1);
        raw_variables(index).raw_variable_use = VariableUse::Input;
        raw_variables(index).type = RawVariableType::Numeric;
    }

    raw_variables(new_variables_number-1).name = "column_" + to_string(new_variables_number);
    raw_variables(new_variables_number-1).raw_variable_use = VariableUse::Target;
    raw_variables(new_variables_number-1).type = RawVariableType::Numeric;

    samples_uses.resize(new_samples_number);

    split_samples_random();
}


/// Sets new numbers of samples and inputs and target variables in the data set.
/// The variables in the data set are the number of inputs plus the number of targets.
/// @param new_samples_number Number of
/// @param new_inputs_number Number of input variables.
/// @param new_targets_number Number of target variables.

void DataSet::set(const Index& new_samples_number,
                  const Index& new_inputs_number,
                  const Index& new_targets_number)
{

    data_source_path = "";

    const Index new_variables_number = new_inputs_number + new_targets_number;

    data.resize(new_samples_number, new_variables_number);

    raw_variables.resize(new_variables_number);

    for(Index i = 0; i < new_variables_number; i++)
    {
        if(i < new_inputs_number)
        {
            raw_variables(i).name = "column_" + to_string(i+1);
            raw_variables(i).raw_variable_use = VariableUse::Input;
            raw_variables(i).type = RawVariableType::Numeric;
        }
        else
        {
            raw_variables(i).name = "column_" + to_string(i+1);
            raw_variables(i).raw_variable_use = VariableUse::Target;
            raw_variables(i).type = RawVariableType::Numeric;
        }
    }

    input_variables_dimensions.resize(1);
    target_variables_dimensions.resize(1);

    samples_uses.resize(new_samples_number);
    split_samples_random();
}


/// Sets the members of this data set object with those from another data set object.
/// @param other_data_set Data set object to be copied.

void DataSet::set(const DataSet& other_data_set)
{
    data_source_path = other_data_set.data_source_path;

    has_raw_variables_names = other_data_set.has_raw_variables_names;

    separator = other_data_set.separator;

    missing_values_label = other_data_set.missing_values_label;

    data = other_data_set.data;

    raw_variables = other_data_set.raw_variables;

    display = other_data_set.display;
}


/// Sets the data set members from a XML document.
/// @param data_set_document TinyXML document containing the member data.

void DataSet::set(const tinyxml2::XMLDocument& data_set_document)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    set_default();

    from_XML(data_set_document);
}


/// Sets the data set members by loading them from a XML file.
/// @param file_name Data set XML file_name.

void DataSet::set(const string& file_name)
{
    load(file_name);
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void DataSet::set_display(const bool& new_display)
{
    display = new_display;
}



/// Sets the default member values:
/// <ul>
/// <li> Display: True.
/// </ul>

void DataSet::set_default()
{
    const int n = omp_get_max_threads();
    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    has_raw_variables_names = false;

    separator = Separator::Comma;

    missing_values_label = "NA";

    set_default_raw_variables_uses();

    set_default_raw_variables_names();

    input_variables_dimensions.resize(1);

    input_variables_dimensions.setConstant(get_input_variables_number());

    target_variables_dimensions.resize(1);

    target_variables_dimensions.setConstant(get_target_variables_number());

}


void DataSet::set_model_type_string(const string& new_model_type)
{
    if(new_model_type == "Approximation")
    {
        set_model_type(ModelType::Approximation);
    }
    else if(new_model_type == "Classification")
    {
        set_model_type(ModelType::Classification);
    }
    else if(new_model_type == "Forecasting")
    {
        set_model_type(ModelType::Forecasting);
    }
    else if(new_model_type == "ImageClassification")
    {
        set_model_type(ModelType::ImageClassification);
    }
    else if(new_model_type == "TextClassification")
    {
        set_model_type(ModelType::TextClassification);
    }
    else if(new_model_type == "AutoAssociation")
    {
        set_model_type(ModelType::AutoAssociation);
    }
    else
    {
        const string message =
            "Data Set class exception:\n"
            "void set_model_type_string(const string&)\n"
            "Unknown project type: " + new_model_type + "\n";

        throw runtime_error(message);
    }
}


void DataSet::set_model_type(const DataSet::ModelType& new_model_type)
{
    model_type = new_model_type;
}


/// Sets a new data matrix.
/// The number of rows must be equal to the number of
/// The number of raw_variables must be equal to the number of variables.
/// Indices of all training, selection and testing samples and inputs and target variables do not change.
/// @param new_data Data matrix.

void DataSet::set_data(const Tensor<type, 2>& new_data)
{
    const Index samples_number = new_data.dimension(0);
    const Index variables_number = new_data.dimension(1);

    set(samples_number, variables_number);

    data = new_data;
}

void DataSet::set_data(const Tensor<type, 2>& new_data, const bool& new_samples)
{
    data = new_data;
}


/// Sets the name of the data file.
/// It also loads the data from that file.
/// Moreover, it sets the variables and samples objects.
/// @param new_data_file_name Name of the file containing the data.

void DataSet::set_data_source_path(const string& new_data_file_name)
{
    data_source_path = new_data_file_name;
}


/// Sets if the data file contains a header with the names of the raw_variables.

void DataSet::set_has_raw_variables_names(const bool& new_has_raw_variables_names)
{
    has_raw_variables_names = new_has_raw_variables_names;
}


/// Sets if the data file contains rows label.

void DataSet::set_has_rows_label(const bool& new_has_rows_label)
{
    has_rows_labels = new_has_rows_label;
}


/// Sets a new separator.
/// @param new_separator Separator value.

void DataSet::set_separator(const Separator& new_separator)
{
    separator = new_separator;
}


/// Sets a new separator from a char.
/// @param new_separator Char with the separator value.

void DataSet::set_separator(const char& new_separator)
{
    if(new_separator == ' ')
    {
        separator = Separator::Space;
    }
    else if(new_separator == '\t')
    {
        separator = Separator::Tab;
    }
    else if(new_separator == ',')
    {
        separator = Separator::Comma;
    }
    else if(new_separator == ';')
    {
        separator = Separator::Semicolon;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_separator(const char&) method.\n"
               << "Unknown separator: " << new_separator << ".\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets a new separator from a string.
/// @param new_separator Char with the separator value.

void DataSet::set_separator(const string& new_separator_string)
{
    if(new_separator_string == "Space")
    {
        separator = Separator::Space;
    }
    else if(new_separator_string == "Tab")
    {
        separator = Separator::Tab;
    }
    else if(new_separator_string == "Comma")
    {
        separator = Separator::Comma;
    }
    else if(new_separator_string == "Semicolon")
    {
        separator = Separator::Semicolon;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_separator(const string&) method.\n"
               << "Unknown separator: " << new_separator_string << ".\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets a new string codification for the dataset.
/// @param new_codification String codification for the dataset.

void DataSet::set_codification(const DataSet::Codification& new_codification)
{
    codification = new_codification;
}


/// Sets a new string codification for the dataset.
/// @param new_codification String codification for the dataset.

void DataSet::set_codification(const string& new_codification_string)
{
    if(new_codification_string == "UTF-8")
    {
        codification = Codification::UTF8;
    }
    else if(new_codification_string == "SHIFT_JIS")
    {
        codification = Codification::SHIFT_JIS;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_codification(const string&) method.\n"
               << "Unknown codification: " << new_codification_string << ".\n"
               << "Available codifications: UTF-8, SHIFT_JIS.\n";

        throw runtime_error(buffer.str());
    }
}


/// Sets a new label for the missing values.
/// @param new_missing_values_label Label for the missing values.

void DataSet::set_missing_values_label(const string& new_missing_values_label)
{
#ifdef OPENNN_DEBUG

    if(get_trimmed(new_missing_values_label).empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_missing_values_label(const string&) method.\n"
               << "Missing values label cannot be empty.\n";

        throw runtime_error(buffer.str());
    }

#endif

    missing_values_label = new_missing_values_label;
}


/// Sets a new method for the missing values.
/// @param new_missing_values_method Method for the missing values.

void DataSet::set_missing_values_method(const DataSet::MissingValuesMethod& new_missing_values_method)
{
    missing_values_method = new_missing_values_method;
}


void DataSet::set_missing_values_method(const string & new_missing_values_method)
{
    if(new_missing_values_method == "Unuse")
    {
        missing_values_method = MissingValuesMethod::Unuse;
    }
    else if(new_missing_values_method == "Mean")
    {
        missing_values_method = MissingValuesMethod::Mean;
    }
    else if(new_missing_values_method == "Median")
    {
        missing_values_method = MissingValuesMethod::Median;
    }
    else if(new_missing_values_method == "Interpolation")
    {
        missing_values_method = MissingValuesMethod::Interpolation;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_missing_values_method(const string & method.\n"
               << "Not known method type.\n";

        throw runtime_error(buffer.str());
    }
}


void DataSet::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


/// Sets a new number of samples in the data set.
/// All samples are also set for training.
/// The indices of the inputs and target variables do not change.
/// @param new_samples_number Number of samples.

void DataSet::set_samples_number(const Index& new_samples_number)
{
    const Index variables_number = get_variables_number();

    set(new_samples_number,variables_number);
}


/// Removes the input of target indices of that variables with zero standard deviation.
/// It might change the size of the vectors containing the inputs and targets indices.

Tensor<string, 1> DataSet::unuse_constant_raw_variables()
{
    const Index raw_variables_number = get_raw_variables_number();

#ifdef OPENNN_DEBUG

    if(raw_variables_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<string, 1> unuse_constant_raw_variables() method.\n"
               << "Number of raw_variables is zero.\n";

        throw runtime_error(buffer.str());
    }

#endif

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<string, 1> constant_raw_variables;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Constant)
        {
            raw_variables(i).set_use(VariableUse::Unused);

            push_back_string(constant_raw_variables, raw_variables(i).name);
        }
    }

    return constant_raw_variables;
}


/// Removes the training, selection and testing indices of that samples which are repeated in the data matrix.
/// It might change the size of the vectors containing the training, selection and testing indices.

Tensor<Index, 1> DataSet::unuse_repeated_samples()
{
    const Index samples_number = get_samples_number();

#ifdef OPENNN_DEBUG

    if(samples_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> unuse_repeated_samples() method.\n"
               << "Number of samples is zero.\n";

        throw runtime_error(buffer.str());
    }

#endif

    Tensor<Index, 1> repeated_samples(0);

    Tensor<type, 1> sample_i;
    Tensor<type, 1> sample_j;

    for(Index i = 0; i < samples_number; i++)
    {
        sample_i = get_sample_data(i);

        for(Index j = Index(i+1); j < samples_number; j++)
        {
            sample_j = get_sample_data(j);

            if(get_sample_use(j) != SampleUse::Unused
                && equal(sample_i.data(), sample_i.data()+sample_i.size(), sample_j.data()))
            {
                set_sample_use(j, SampleUse::Unused);

                push_back_index(repeated_samples, j);
            }
        }
    }

    return repeated_samples;
}


/// Return unused variables without correlation.
/// @param minimum_correlation Minimum correlation between variables.

Tensor<string, 1> DataSet::unuse_uncorrelated_raw_variables(const type& minimum_correlation)
{
    Tensor<string, 1> unused_raw_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variables_correlations();

    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index target_raw_variables_number = get_target_raw_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_raw_variable_index = input_raw_variables_indices(i);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            if(!isnan(correlations(i,j).r)
                && abs(correlations(i,j).r) < minimum_correlation
                && raw_variables(input_raw_variable_index).raw_variable_use != VariableUse::Unused)
            {
                raw_variables(input_raw_variable_index).set_use(VariableUse::Unused);

                push_back_string(unused_raw_variables, raw_variables(input_raw_variable_index).name);
            }
        }
    }

    return unused_raw_variables;
}


Tensor<string, 1> DataSet::unuse_multicollinear_raw_variables(Tensor<Index, 1>& original_variable_indices, Tensor<Index, 1>& final_variable_indices)
{
    // Original_raw_variables_indices and final_raw_variables_indices refers to the indices of the variables

    Tensor<string, 1> unused_raw_variables;

    for(Index i = 0; i < original_variable_indices.size(); i++)
    {
        const Index original_raw_variable_index = original_variable_indices(i);

        bool found = false;

        for(Index j = 0; j < final_variable_indices.size(); j++)
        {
            if(original_raw_variable_index == final_variable_indices(j))
            {
                found = true;
                break;
            }
        }

        const Index raw_variable_index = get_raw_variable_index(original_raw_variable_index);

        if(!found && raw_variables(raw_variable_index).raw_variable_use != VariableUse::Unused)
        {
            raw_variables(raw_variable_index).set_use(VariableUse::Unused);

            push_back_string(unused_raw_variables, raw_variables(raw_variable_index).name);
        }
    }

    return unused_raw_variables;
}


/// Returns the distribution of each of the raw_variables. In the case of numeric raw_variables, it returns a
/// histogram, for the case of categorical raw_variables, it returns the frequencies of each category and for the
/// binary raw_variables it returns the frequencies of the positives and negatives.
/// The default number of bins is 10.
/// @param bins_number Number of bins.

Tensor<Histogram, 1> DataSet::calculate_raw_variables_distribution(const Index& bins_number) const
{
    const Index raw_variables_number = raw_variables.size();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Index used_samples_number = used_samples_indices.size();

    Tensor<Histogram, 1> histograms(used_raw_variables_number);

    Index variable_index = 0;
    Index used_raw_variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Numeric)
        {
            if(raw_variables(i).raw_variable_use == VariableUse::Unused)
            {
                variable_index++;
            }
            else
            {
                Tensor<type, 1> raw_variable(used_samples_number);

                for(Index j = 0; j < used_samples_number; j++)
                {
                    raw_variable(j) = data(used_samples_indices(j), variable_index);
                }

                histograms(used_raw_variable_index) = histogram(raw_variable, bins_number);

                variable_index++;
                used_raw_variable_index++;
            }
        }
        else if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variables(i).get_categories_number();

            if(raw_variables(i).raw_variable_use == VariableUse::Unused)
            {
                variable_index += categories_number;
            }
            else
            {
                Tensor<Index, 1> categories_frequencies(categories_number);
                categories_frequencies.setZero();
                Tensor<type, 1> centers(categories_number);

                for(Index j = 0; j < categories_number; j++)
                {
                    for(Index k = 0; k < used_samples_number; k++)
                    {
                        if(abs(data(used_samples_indices(k), variable_index) - type(1)) < type(NUMERIC_LIMITS_MIN))
                        {
                            categories_frequencies(j)++;
                        }
                    }

                    centers(j) = type(j);

                    variable_index++;
                }

                histograms(used_raw_variable_index).frequencies = categories_frequencies;
                histograms(used_raw_variable_index).centers = centers;

                used_raw_variable_index++;
            }
        }
        else if(raw_variables(i).type == RawVariableType::Binary)
        {
            if(raw_variables(i).raw_variable_use == VariableUse::Unused)
            {
                variable_index++;
            }
            else
            {
                Tensor<Index, 1> binary_frequencies(2);
                binary_frequencies.setZero();

                for(Index j = 0; j < used_samples_number; j++)
                {
                    if(abs(data(used_samples_indices(j), variable_index) - type(1)) < type(NUMERIC_LIMITS_MIN))
                    {
                        binary_frequencies(0)++;
                    }
                    else
                    {
                        binary_frequencies(1)++;
                    }
                }

                histograms(used_raw_variable_index).frequencies = binary_frequencies;
                variable_index++;
                used_raw_variable_index++;
            }
        }
        else if(raw_variables(i).type == RawVariableType::DateTime)
        {
            // @todo

            if(raw_variables(i).raw_variable_use == VariableUse::Unused)
            {
            }
            else
            {
            }

            variable_index++;
        }
        else
        {
            variable_index++;
        }
    }

    return histograms;
}


BoxPlot DataSet::calculate_single_box_plot(Tensor<type,1>& values) const
{
    const Index n = values.size();

    Tensor<Index, 1> indices(n);

    for(Index i = 0; i < n; i++)
    {
        indices(i) = i;
    }

    return box_plot(values, indices);
}


Tensor<BoxPlot, 1> DataSet::calculate_data_raw_variables_box_plot(Tensor<type,2>& data) const
{
    const Index raw_variables_number = data.dimension(1);

    Tensor<BoxPlot, 1> box_plots(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        box_plots(i) = box_plot(data.chip(i, 1));
    }

    return box_plots;
}


/// Returns a vector of subvectors with the values of a box and whiskers plot.
/// The size of the vector is equal to the number of used variables.
/// The size of the subvectors is 5 and they consist on:
/// <ul>
/// <li> Minimum
/// <li> First quartile
/// <li> Second quartile
/// <li> Third quartile
/// <li> Maximum
/// </ul>

Tensor<BoxPlot, 1> DataSet::calculate_raw_variables_box_plots() const
{
    Index raw_variables_number = get_raw_variables_number();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<BoxPlot, 1> box_plots(raw_variables_number);

//    Index used_raw_variable_index = 0;
    Index variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Numeric || raw_variables(i).type == RawVariableType::Binary)
        {
            if(raw_variables(i).raw_variable_use != VariableUse::Unused)
            {
                box_plots(i) = box_plot(data.chip(variable_index, 1), used_samples_indices);

//                used_raw_variable_index++;
            }
            else
            {
                box_plots(i) = BoxPlot();
            }

            variable_index++;
        }
        else if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();

            box_plots(i) = BoxPlot();
        }
        else
        {
            variable_index++;
            box_plots(i) = BoxPlot();
        }
    }

    return box_plots;
}


/// Counts the number of used negatives of the selected target.
/// @param target_index Index of the target to evaluate.

Index DataSet::calculate_used_negatives(const Index& target_index)
{
    Index negatives = 0;

    const Tensor<Index, 1> used_indices = get_used_samples_indices();

    const Index used_samples_number = used_indices.size();

    for(Index i = 0; i < used_samples_number; i++)
    {
        const Index training_index = used_indices(i);

        if(!isnan(data(training_index, target_index)))
        {
            if(abs(data(training_index, target_index)) < type(NUMERIC_LIMITS_MIN))
            {
                negatives++;
            }
            else if(abs(data(training_index, target_index) - type(1)) > type(NUMERIC_LIMITS_MIN)
                    || data(training_index, target_index) < type(0))
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: DataSet class.\n"
                       << "Index calculate_used_negatives(const Index&) const method.\n"
                       << "Training sample is neither a positive nor a negative: " << training_index << "-" << target_index << "-" << data(training_index, target_index) << endl;

                throw runtime_error(buffer.str());
            }
        }
    }

    return negatives;
}


/// Counts the number of negatives of the selected target in the training data.
/// @param target_index Index of the target to evaluate.

Index DataSet::calculate_training_negatives(const Index& target_index) const
{
    Index negatives = 0;

    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Index training_samples_number = training_indices.size();

    for(Index i = 0; i < training_samples_number; i++)
    {
        const Index training_index = training_indices(i);

        if(abs(data(training_index, target_index)) < type(NUMERIC_LIMITS_MIN))
        {
            negatives++;
        }
        else if(abs(data(training_index, target_index) - type(1)) > type(1.0e-3))
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Index calculate_training_negatives(const Index&) const method.\n"
                   << "Training sample is neither a positive nor a negative: " << data(training_index, target_index) << endl;

            throw runtime_error(buffer.str());
        }
    }

    return negatives;
}


/// Counts the number of negatives of the selected target in the selection data.
/// @param target_index Index of the target to evaluate.

Index DataSet::calculate_selection_negatives(const Index& target_index) const
{
    Index negatives = 0;

    const Index selection_samples_number = get_selection_samples_number();

    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    for(Index i = 0; i < Index(selection_samples_number); i++)
    {
        const Index selection_index = selection_indices(i);

        if(abs(data(selection_index, target_index)) < type(NUMERIC_LIMITS_MIN))
        {
            negatives++;
        }
        else if(abs(data(selection_index, target_index) - type(1)) > type(NUMERIC_LIMITS_MIN))
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Index calculate_testing_negatives(const Index&) const method.\n"
                   << "Selection sample is neither a positive nor a negative: " << data(selection_index, target_index) << endl;

            throw runtime_error(buffer.str());
        }
    }

    return negatives;
}


/// Counts the number of negatives of the selected target in the testing data.
/// @param target_index Index of the target to evaluate.

Index DataSet::calculate_testing_negatives(const Index& target_index) const
{
    Index negatives = 0;

    const Index testing_samples_number = get_testing_samples_number();

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    for(Index i = 0; i < Index(testing_samples_number); i++)
    {
        const Index testing_index = testing_indices(i);

        if(data(testing_index, target_index) < type(NUMERIC_LIMITS_MIN))
        {
            negatives++;
        }
    }

    return negatives;
}


/// Returns a vector of vectors containing some basic descriptives of all the variables in the data set.
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation.
/// </ul>

Tensor<Descriptives, 1> DataSet::calculate_variables_descriptives() const
{
    return descriptives(data);
}


/// Returns a vector of vectors containing some basic descriptives of the used variables and samples
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Minimum.
/// <li> Maximum.
/// <li> Mean.
/// <li> Standard deviation.
/// </ul>

Tensor<Descriptives, 1> DataSet::calculate_used_variables_descriptives() const
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();

    return descriptives(data, used_samples_indices, used_variables_indices);
}


/// Calculate the descriptives of the samples with positive targets in binary classification problems.

Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_positive_samples() const
{
    const Index target_index = get_target_variables_indices()(0);

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used positive samples

    Index positive_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, target_index) - type(1)) < type(NUMERIC_LIMITS_MIN)) positive_samples_number++;
    }

    // Get used positive samples indices

    Tensor<Index, 1> positive_used_samples_indices(positive_samples_number);
    Index positive_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, target_index) - type(1)) < type(NUMERIC_LIMITS_MIN))
        {
            positive_used_samples_indices(positive_sample_index) = sample_index;
            positive_sample_index++;
        }
    }

    return descriptives(data, positive_used_samples_indices, input_variables_indices);
}


/// Calculate the descriptives of the samples with neagtive targets in binary classification problems.

Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_negative_samples() const
{
    const Index target_index = get_target_variables_indices()(0);

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used negative samples

    Index negative_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_samples_indices(i);

        if(data(sample_index, target_index) < type(NUMERIC_LIMITS_MIN)) negative_samples_number++;
    }

    // Get used negative samples indices

    Tensor<Index, 1> negative_used_samples_indices(negative_samples_number);
    Index negative_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(data(sample_index, target_index) < type(NUMERIC_LIMITS_MIN))
        {
            negative_used_samples_indices(negative_sample_index) = sample_index;
            negative_sample_index++;
        }
    }

    return descriptives(data, negative_used_samples_indices, input_variables_indices);
}


/// Returns a matrix with the data set descriptive statistics.
/// @param class_index Data set index number to make the descriptive statistics.

Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_categories(const Index& class_index) const
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used class samples

    Index class_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, class_index) - type(1)) < type(NUMERIC_LIMITS_MIN)) class_samples_number++;
    }

    // Get used class samples indices

    Tensor<Index, 1> class_used_samples_indices(class_samples_number);
    class_used_samples_indices.setZero();
    Index class_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, class_index) - type(1)) < type(NUMERIC_LIMITS_MIN))
        {
            class_used_samples_indices(class_sample_index) = sample_index;
            class_sample_index++;
        }
    }

    return descriptives(data, class_used_samples_indices, input_variables_indices);
}


/// Returns a vector of vectors containing some basic descriptives of all variables on the training
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Training data minimum.
/// <li> Training data maximum.
/// <li> Training data mean.
/// <li> Training data standard deviation.
/// </ul>

Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_training_samples() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1> used_indices = get_used_raw_variables_indices();

    return descriptives(data, training_indices, used_indices);
}


/// Returns a vector of vectors containing some basic descriptives of all variables on the selection
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Selection data minimum.
/// <li> Selection data maximum.
/// <li> Selection data mean.
/// <li> Selection data standard deviation.
/// </ul>

Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_selection_samples() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> used_indices = get_used_raw_variables_indices();

    return descriptives(data, selection_indices, used_indices);
}


/// Returns a vector of Descriptives structures with some basic statistics of the input variables on the used
/// This includes the minimum, maximum, mean and standard deviation.
/// The size of this vector is the number of inputs.

Tensor<Descriptives, 1> DataSet::calculate_input_variables_descriptives() const
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return descriptives(data, used_samples_indices, input_variables_indices);
}


/// Returns a vector of vectors with some basic descriptives of the target variables on all
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Target variables minimum.
/// <li> Target variables maximum.
/// <li> Target variables mean.
/// <li> Target variables standard deviation.
/// </ul>

Tensor<Descriptives, 1> DataSet::calculate_target_variables_descriptives() const
{
    const Tensor<Index, 1> used_indices = get_used_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return descriptives(data, used_indices, target_variables_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_testing_target_variables_descriptives() const
{
    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return descriptives(data, testing_indices, target_variables_indices);
}


/// Returns a vector containing the minimums of the input variables.

Tensor<type, 1> DataSet::calculate_input_variables_minimums() const
{
    return raw_variables_minimums(data, get_used_samples_indices(), get_input_variables_indices());
}


/// Returns a vector containing the minimums of the target variables.

Tensor<type, 1> DataSet::calculate_target_variables_minimums() const
{
    return raw_variables_minimums(data, get_used_samples_indices(), get_target_variables_indices());
}



/// Returns a vector containing the maximums of the input variables.

Tensor<type, 1> DataSet::calculate_input_variables_maximums() const
{
    return raw_variables_maximums(data, get_used_samples_indices(), get_input_variables_indices());
}


/// Returns a vector containing the maximums of the target variables.

Tensor<type, 1> DataSet::calculate_target_variables_maximums() const
{
    return raw_variables_maximums(data, get_used_samples_indices(), get_target_variables_indices());
}


/// Returns a vector containing the maximum of the used variables.

Tensor<type, 1> DataSet::calculate_used_variables_minimums() const
{
    return raw_variables_minimums(data, get_used_samples_indices(), get_used_variables_indices());
}


/// Returns a vector containing the means of a set of given variables.
/// @param variables_indices Indices of the variables.

Tensor<type, 1> DataSet::calculate_variables_means(const Tensor<Index, 1>& variables_indices) const
{
    const Index variables_number = variables_indices.size();

    Tensor<type, 1> means(variables_number);
    means.setZero();

#pragma omp parallel for

    for(Index i = 0; i < variables_number; i++)
    {
        const Index variable_index = variables_indices(i);

        Tensor<type, 0> mean = data.chip(variable_index, 1).mean();

        means(i) = mean(0);
    }

    return means;
}


Tensor<type, 1> DataSet::calculate_used_targets_mean() const
{
    const Tensor<Index, 1> used_indices = get_used_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return mean(data, used_indices, target_variables_indices);
}


/// Returns the mean values of the target variables on the selection

Tensor<type, 1> DataSet::calculate_selection_targets_mean() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return mean(data, selection_indices, target_variables_indices);
}


/// Returns the value of the gmt that has the data set, by default it is 0.
/// This is recommended to use in forecasting problems.

Index DataSet::get_gmt() const
{
    return gmt;
}


/// Sets the value of the gmt, by default it is 0.
/// This is recommended to use in forecasting problems.

void DataSet::set_gmt(Index& new_gmt)
{
    gmt = new_gmt;
}


/// Calculates the correlations between all outputs and all inputs.
/// It returns a matrix with the data stored in CorrelationsResults format, where the number of rows is the input number
/// and number of raw_variables is the target number.
/// Each element contains the correlation between a single input and a single target.

Tensor<Correlation, 2> DataSet::calculate_input_target_raw_variables_correlations() const
{
    const int number_of_thread = omp_get_max_threads();
    ThreadPool* correlations_thread_pool = new ThreadPool(number_of_thread);
    ThreadPoolDevice* correlations_thread_pool_device = new ThreadPoolDevice(correlations_thread_pool, number_of_thread);

    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index target_raw_variables_number = get_target_raw_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();
    const Tensor<Index, 1> target_raw_variables_indices = get_target_raw_variables_indices();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

#pragma omp parallel for

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_index = input_raw_variables_indices(i);

        const Tensor<type, 2> input_raw_variable_data = get_raw_variable_data(input_index, used_samples_indices);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_index = target_raw_variables_indices(j);

            const Tensor<type, 2> target_raw_variable_data = get_raw_variable_data(target_index, used_samples_indices);

            cout << "input_raw_variable_data: " << input_raw_variable_data << endl;
            cout << "target_raw_variable_data: " << target_raw_variable_data << endl;

            correlations(i,j) = opennn::correlation(correlations_thread_pool_device, input_raw_variable_data, target_raw_variable_data);
        }
    }

    delete correlations_thread_pool;
    delete correlations_thread_pool_device;

    return correlations;
}


Tensor<Correlation, 2> DataSet::calculate_relevant_input_target_raw_variables_correlations(const Tensor<Index, 1>& input_raw_variables_indices,
                                                                                     const Tensor<Index, 1>& target_raw_variables_indices) const
{
    const int number_of_thread = omp_get_max_threads();
    ThreadPool* correlations_thread_pool = new ThreadPool(number_of_thread);
    ThreadPoolDevice* correlations_thread_pool_device = new ThreadPoolDevice(correlations_thread_pool, number_of_thread);

    const Index input_raw_variables_number = input_raw_variables_indices.dimension(0);
    const Index target_raw_variables_number = target_raw_variables_indices.dimension(0);

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

#pragma omp parallel for

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_index = input_raw_variables_indices(i);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_index = target_raw_variables_indices(j);

            const Tensor<type, 2> input_raw_variable_data = get_raw_variable_data(input_index, get_used_samples_indices());
            const Tensor<type, 2> target_raw_variable_data = get_raw_variable_data(target_index, get_used_samples_indices());

            correlations(i, j) = opennn::correlation(correlations_thread_pool_device, input_raw_variable_data, target_raw_variable_data);
        }
    }

    delete correlations_thread_pool;
    delete correlations_thread_pool_device;

    return correlations;
}




Tensor<Correlation, 2> DataSet::calculate_input_target_raw_variables_correlations_spearman() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index target_raw_variables_number = get_target_raw_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();
    const Tensor<Index, 1> target_raw_variables_indices = get_target_raw_variables_indices();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_index = input_raw_variables_indices(i);

        const Tensor<type, 2> input_raw_variable_data = get_raw_variable_data(input_index, used_samples_indices);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_index = target_raw_variables_indices(j);

            const Tensor<type, 2> target_raw_variable_data = get_raw_variable_data(target_index, used_samples_indices);

            correlations(i,j) = opennn::correlation_spearman(thread_pool_device, input_raw_variable_data, target_raw_variable_data);
        }
    }

    return correlations;
}


/// Returns true if the data contain missing values.

bool DataSet::has_nan() const
{
    const Index rows_number = data.dimension(0);

    for(Index i = 0; i < rows_number; i++)
    {
        if(samples_uses(i) != SampleUse::Unused)
        {
            if(has_nan_row(i)) return true;
        }
    }

    return false;
}


/// Returns true if the given row contains missing values.

bool DataSet::has_nan_row(const Index& row_index) const
{
    for(Index j = 0; j < data.dimension(1); j++)
    {
        if(isnan(data(row_index,j))) return true;
    }

    return false;
}


/// Print on screen the information about the missing values in the data set.
/// <ul>
/// <li> Total number of missing values.
/// <li> Number of variables with missing values.
/// <li> Number of samples with missing values.
/// </ul>

void DataSet::print_missing_values_information() const
{
    const Index missing_values_number = count_nan();

    cout << "Missing values number: " << missing_values_number << " (" << missing_values_number*100/data.size() << "%)" << endl;

    const Tensor<Index, 0> raw_variables_with_missing_values = count_nan_raw_variables().sum();

    cout << "raw_variables with missing values: " << raw_variables_with_missing_values(0)
         << " (" << raw_variables_with_missing_values(0)*100/data.dimension(1) << "%)" << endl;

    const Index samples_with_missing_values = count_rows_with_nan();

    cout << "Samples with missing values: "
         << samples_with_missing_values << " (" << samples_with_missing_values*100/data.dimension(0) << "%)" << endl;
}


/// Print on screen the correlation between targets and inputs.

void DataSet::print_input_target_raw_variables_correlations() const
{
    const Index inputs_number = get_input_variables_number();
    const Index targets_number = get_target_raw_variables_number();

    const Tensor<string, 1> inputs_names = get_input_raw_variables_names();
    const Tensor<string, 1> targets_name = get_target_raw_variables_names();

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variables_correlations();

    for(Index j = 0; j < targets_number; j++)
    {
        for(Index i = 0; i < inputs_number; i++)
        {
            cout << targets_name(j) << " - " << inputs_names(i) << ": " << correlations(i,j).r << endl;
        }
    }
}


/// This method print on screen the corretaliont between inputs and targets.
/// @param number Number of variables to be printed.

void DataSet::print_top_input_target_raw_variables_correlations() const
{
    const Index inputs_number = get_input_raw_variables_number();
    const Index targets_number = get_target_raw_variables_number();

    const Tensor<string, 1> inputs_names = get_input_variables_names();
    const Tensor<string, 1> targets_name = get_target_variables_names();

    const Tensor<type, 2> correlations = get_correlation_values(calculate_input_target_raw_variables_correlations());

    Tensor<type, 1> target_correlations(inputs_number);

    Tensor<string, 2> top_correlations(inputs_number, 2);

    map<type,string> top_correlation;

    for(Index i = 0 ; i < inputs_number; i++)
    {
        for(Index j = 0 ; j < targets_number ; j++)
        {
            top_correlation.insert(pair<type,string>(correlations(i,j), inputs_names(i) + " - " + targets_name(j)));
        }
    }

    map<type,string>::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
    {
        cout << "Correlation:  " << (*it).first << "  between  " << (*it).second << "" << endl;
    }
}


/// Calculate the correlation between each input in the data set.
/// Returns a matrix with the correlation values between variables in the data set.


Tensor<Tensor<Correlation, 2>, 1> DataSet::calculate_input_raw_variables_correlations(const bool& calculate_pearson_correlations, const bool& calculate_spearman_correlations) const
{
    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();

    const Index input_raw_variables_number = get_input_raw_variables_number();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, input_raw_variables_number);
    Tensor<Correlation, 2> correlations_spearman(input_raw_variables_number, input_raw_variables_number);

    // list to return
    Tensor<Tensor<Correlation, 2>, 1> correlations_list(2);

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index current_input_index_i = input_raw_variables_indices(i);

        const Tensor<type, 2> input_i = get_raw_variable_data(current_input_index_i);

        if(display) cout << "Calculating " << raw_variables(current_input_index_i).name << " correlations. " << endl;

        for(Index j = i; j < input_raw_variables_number; j++)
        {
            if(j == i)
            {
                if(calculate_pearson_correlations)
                {
                    correlations(i,j).r = type(1);
                    correlations(i,j).b = type(1);
                    correlations(i,j).a = type(0);

                    correlations(i,j).upper_confidence = type(1);
                    correlations(i,j).lower_confidence = type(1);
                    correlations(i,j).form = Correlation::Form::Linear;
                    correlations(i,j).method = Correlation::Method::Pearson;
                }

                if(calculate_spearman_correlations)
                {
                    correlations_spearman(i,j).r = type(1);
                    correlations_spearman(i,j).b = type(1);
                    correlations_spearman(i,j).a = type(0);

                    correlations_spearman(i,j).upper_confidence = type(1);
                    correlations_spearman(i,j).lower_confidence = type(1);
                    correlations_spearman(i,j).form = Correlation::Form::Linear;
                    correlations_spearman(i,j).method = Correlation::Method::Spearman;
                }
            }
            else
            {
                const Index current_input_index_j = input_raw_variables_indices(j);

                const Tensor<type, 2> input_j = get_raw_variable_data(current_input_index_j);

                if(calculate_pearson_correlations)
                {
                    correlations(i,j) = opennn::correlation(thread_pool_device, input_i, input_j);
                    if(correlations(i,j).r > (type(1) - NUMERIC_LIMITS_MIN))
                        correlations(i,j).r = type(1);
                }

                if(calculate_spearman_correlations)
                {
                    correlations_spearman(i,j) = opennn::correlation_spearman(thread_pool_device, input_i, input_j);

                    if(correlations_spearman(i,j).r > (type(1) - NUMERIC_LIMITS_MIN))
                        correlations_spearman(i,j).r = type(1);
                }
            }
        }
    }

    if(calculate_pearson_correlations)
    {
        for(Index i = 0; i < input_raw_variables_number; i++)
        {
            for(Index j = 0; j < i; j++)
            {
                correlations(i,j) = correlations(j,i);
            }
        }
    }

    if(calculate_spearman_correlations)
    {
        for(Index i = 0; i < input_raw_variables_number; i++)
        {
            for(Index j = 0; j < i; j++)
            {
                correlations_spearman(i,j) = correlations_spearman(j,i);
            }
        }
    }

    correlations_list(0) = correlations;
    correlations_list(1) = correlations_spearman;
    return correlations_list;
}


/// Print on screen the correlation between variables in the data set.

void DataSet::print_inputs_correlations() const
{
    const Tensor<type, 2> inputs_correlations = get_correlation_values(calculate_input_raw_variables_correlations()(0));

    cout << inputs_correlations << endl;
}


void DataSet::print_data_file_preview() const
{
    const Index size = data_file_preview.size();

    for(Index i = 0;  i < size; i++)
    {
        for(Index j = 0; j < data_file_preview(i).size(); j++)
        {
            cout << data_file_preview(i)(j) << " ";
        }

        cout << endl;
    }
}


/// This method print on screen the corretaliont between variables.
/// @param number Number of variables to be printed.

void DataSet::print_top_inputs_correlations() const
{
    const Index variables_number = get_input_variables_number();

    const Tensor<string, 1> variables_name = get_input_variables_names();

    const Tensor<type, 2> variables_correlations = get_correlation_values(calculate_input_raw_variables_correlations()(0));

    const Index correlations_number = variables_number*(variables_number-1)/2;

    Tensor<string, 2> top_correlations(correlations_number, 3);

    map<type, string> top_correlation;

    for(Index i = 0; i < variables_number; i++)
    {
        for(Index j = i; j < variables_number; j++)
        {
            if(i == j) continue;

            top_correlation.insert(pair<type,string>(variables_correlations(i,j), variables_name(i) + " - " + variables_name(j)));
        }
    }

    map<type,string> ::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
    {
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << "" << endl;
    }
}


/// Returns a vector of strings containing the scaling method that best fits each of the input variables.
/// @todo Takes too long in big files.

void DataSet::set_default_raw_variables_scalers()
{
    if(model_type == ModelType::ImageClassification)
    {
        set_raw_variables_scalers(Scaler::MinimumMaximum);
    }
    else
    {
        const Index raw_variables_number = raw_variables.size();

        for(Index i = 0; i < raw_variables_number; i++)
        {
            if(raw_variables(i).type == RawVariableType::Numeric)
            {
                raw_variables(i).scaler = Scaler::MeanStandardDeviation;
            }
            else
            {
                raw_variables(i).scaler = Scaler::MinimumMaximum;
            }
        }
    }
}


Tensor<Descriptives, 1> DataSet::scale_data()
{
    const Index variables_number = get_variables_number();

    const Tensor<Descriptives, 1> variables_descriptives = calculate_variables_descriptives();

    Index raw_variable_index;

    for(Index i = 0; i < variables_number; i++)
    {
        raw_variable_index = get_raw_variable_index(i);

        switch(raw_variables(raw_variable_index).scaler)
        {
        case Scaler::NoScaling:
            // Do nothing
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, i, variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, i);
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void scale_data() method.\n"
                   << "Unknown scaler: " << int(raw_variables(i).scaler) << "\n";

            throw runtime_error(buffer.str());
        }
        }
    }

    return variables_descriptives;
}


void DataSet::unscale_data(const Tensor<Descriptives, 1>& variables_descriptives)
{
    const Index variables_number = get_variables_number();

    for(Index i = 0; i < variables_number; i++)
    {
        switch(raw_variables(i).scaler)
        {
        case Scaler::NoScaling:
            // Do nothing
            break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(data, i, variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            unscale_logarithmic(data, i);
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void unscale_data() method.\n"
                   << "Unknown scaler: " << int(raw_variables(i).scaler) << "\n";

            throw runtime_error(buffer.str());
        }
        }
    }
}


/// It scales every input variable with the given method.
/// The method to be used is that in the scaling and unscaling method variable.

Tensor<Descriptives, 1> DataSet::scale_input_variables()
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Scaler, 1> input_variables_scalers = get_input_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives = calculate_input_variables_descriptives();

    for(Index i = 0; i < input_variables_number; i++)
    {
        switch(input_variables_scalers(i))
        {
        case Scaler::NoScaling:
            // Do nothing
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, input_variables_indices(i));
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void scale_input_variables(const Tensor<string, 1>&, const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown scaling and unscaling method: " << int(input_variables_scalers(i)) << "\n";

            throw runtime_error(buffer.str());
        }
        }
    }

    return input_variables_descriptives;
}


/// Calculates the input and target variables descriptives.
/// Then it scales the target variables with those values.
/// The method to be used is that in the scaling and unscaling method variable.
/// Finally, it returns the descriptives.

Tensor<Descriptives, 1> DataSet::scale_target_variables()
{
    const Index target_variables_number = get_target_variables_number();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Tensor<Scaler, 1> target_variables_scalers = get_target_variables_scalers();

    const Tensor<Descriptives, 1> target_variables_descriptives = calculate_target_variables_descriptives();

    for(Index i = 0; i < target_variables_number; i++)
    {
        switch(target_variables_scalers(i))
        {
        case Scaler::NoScaling:
            // Do nothing
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, target_variables_indices(i), target_variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, target_variables_indices(i), target_variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, target_variables_indices(i), target_variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, target_variables_indices(i));
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void scale_input_variables(const Tensor<string, 1>&, const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown scaling and unscaling method: " << int(target_variables_scalers(i)) << "\n";

            throw runtime_error(buffer.str());
        }
        }
    }

    return target_variables_descriptives;
}


/// It unscales every input variable with the given method.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::unscale_input_variables(const Tensor<Descriptives, 1>& input_variables_descriptives)
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Tensor<Scaler, 1> input_variables_scalers = get_input_variables_scalers();

    for(Index i = 0; i < input_variables_number; i++)
    {
        switch(input_variables_scalers(i))
        {
        case Scaler::NoScaling:
            // Do nothing
            break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            unscale_logarithmic(data, input_variables_indices(i));
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void unscale_input_variables(const Tensor<string, 1>&, const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown unscaling and unscaling method: " << int(input_variables_scalers(i)) << "\n";

            throw runtime_error(buffer.str());
        }
        }
    }
}


/// It unscales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::unscale_target_variables(const Tensor<Descriptives, 1>& targets_descriptives)
{
    const Index target_variables_number = get_target_variables_number();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Tensor<Scaler, 1> target_variables_scalers = get_target_variables_scalers();

    for(Index i = 0; i < target_variables_number; i++)
    {
        switch(target_variables_scalers(i))
        {
        case Scaler::NoScaling:
            break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(data, target_variables_indices(i), targets_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(data, target_variables_indices(i), targets_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(data, target_variables_indices(i), targets_descriptives(i));
            break;

        case Scaler::Logarithm:
            unscale_logarithmic(data, target_variables_indices(i));
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void unscale_targets(const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown unscaling and unscaling method.\n";

            throw runtime_error(buffer.str());
        }
        }
    }
}


/// Initializes the data matrix with a given value.
/// @param new_value Initialization value.

void DataSet::set_data_constant(const type& new_value)
{
    data.setConstant(new_value);
    data.dimensions();
}


type DataSet::round_to_precision(type x, const int& precision){

    const type factor = type(pow(10, precision));

    return round(factor*x)/factor;
}


Tensor<type,2> DataSet::round_to_precision_matrix(Tensor<type,2> matrix,const int& precision)
{
    Tensor<type, 2> matrix_rounded(matrix.dimension(0), matrix.dimension(1));

    const type factor = type(pow(10, precision));

    for(int i = 0; i < matrix.dimension(0); i++)
    {
        for(int j = 0; j < matrix.dimension(1); j++)
        {
            matrix_rounded(i,j) = (round(factor*matrix(i,j)))/factor;
        }
    }

    return matrix_rounded;
}

Tensor<type, 1> DataSet::round_to_precision_tensor(Tensor<type, 1> tensor, const int& precision)
{
    Tensor<type, 1> tensor_rounded(tensor.size());

    const type factor = type(pow(10, precision));

    for(Index i = 0; i < tensor.size(); i++)
    {
        tensor_rounded(i) = (round(factor*tensor(i)))/factor;
    }

    return tensor_rounded;
}


/// Initializes the data matrix with random values chosen from a uniform distribution
/// with given minimum and maximum.

void DataSet::set_data_random()
{
    data.setRandom();
}


/// Initializes the data matrix with random values chosen from a uniform distribution
/// with given minimum and maximum. The targets will be binary randoms.

void DataSet::set_data_binary_random()
{
    data.setRandom();

    const Index samples_number = data.dimension(0);
    const Index variables_number = data.dimension(1);

    const Index input_variables_number = get_input_variables_number();
    const Index target_variables_number = variables_number - input_variables_number;

    Index target_variable_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(target_variables_number == 1) 
            target_variable_index = rand()%2;
        else 
            target_variable_index = rand()%(variables_number-input_variables_number)+input_variables_number;

        for(Index j = input_variables_number; j < variables_number; j++)
        {
            if(target_variables_number == 1) 
                data(i,j) = type(target_variable_index);
            else 
                data(i,j) = (j == target_variable_index) ? type(1) : type(0);
        }
    }
}


/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void DataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
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

    // Close preview data

    file_stream.CloseElement();

    // Close data set

    file_stream.CloseElement();

    time(&finish);
}


void DataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    ostringstream buffer;

    // Data set element

    cout << " -- Data set element --" << endl;

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data set element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    cout << " -- Data set element --" << endl;



    // Data file

    cout << " -- Data File --" << endl;

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data file element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    cout << " -- Data File --" << endl;



    // Data file name

    cout << " -- Data File Name --" << endl;

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataSourcePath");

    cout << "-- wall -- 1" << endl;

    if(!data_file_name_element)
    {
        cout << "-- inside not(data_file_name_element) --" << endl;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataSourcePath element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    cout << "-- wall -- 2" << endl;

    if(data_file_name_element->GetText())
    {
        cout << "-- inside data_file_name_element->GetText() --" << endl;

        const string new_data_file_name = data_file_name_element->GetText();

        set_data_source_path(new_data_file_name);
    }

    cout << "-- wall -- 3" << endl;

    cout << " -- Data File Name --" << endl;



    // Separator

    cout << " -- Separator --" << endl;

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

    cout << " -- Separator --" << endl;



    // Has raw_variables names

    cout << " -- Has raw_variables names --" << endl;

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

    cout << " -- Has raw_variables names --" << endl;



    // Rows labels

    cout << " -- Rows labels --" << endl;

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

    cout << " -- Rows labels --" << endl;



    // Missing values label

    cout << " -- Missing values label --" << endl;

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

    cout << " -- Missing values label --" << endl;



    // Codification

    cout << " -- Codification --" << endl;

    const tinyxml2::XMLElement* codification_element = data_file_element->FirstChildElement("Codification");

    if(codification_element)
    {
        if(codification_element->GetText())
        {
            const string new_codification = codification_element->GetText();

            set_codification(new_codification);
        }
    }

    cout << " -- Codification --" << endl;



    // raw_variables

    cout << " -- raw_variables --" << endl;

    const tinyxml2::XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "raw_variables element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    cout << " -- raw_variables --" << endl;



    // raw_variables number

    cout << " -- raw_variables number --" << endl;

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

    cout << " -- raw_variables number --" << endl;



    // raw_variables

    cout << " -- raw_variables --" << endl;

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

    cout << " -- raw_variables --" << endl;


    /**
//    // Time series raw_variables

//    const tinyxml2::XMLElement* time_series_raw_variables_element = data_set_element->FirstChildElement("TimeSeriesraw_variables");

//    if(!time_series_raw_variables_element)
//    {
//        // do nothing
//    }
//    else
//    {
//        // Time series raw_variables number

//        const tinyxml2::XMLElement* time_series_raw_variables_number_element = time_series_raw_variables_element->FirstChildElement("TimeSeriesraw_variablesNumber");

//        if(!time_series_raw_variables_number_element)
//        {
//            buffer << "OpenNN Exception: DataSet class.\n"
//                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//                   << "Time seires raw_variables number element is nullptr.\n";

//            throw runtime_error(buffer.str());
//        }

//        Index time_series_new_raw_variables_number = 0;

//        if(time_series_raw_variables_number_element->GetText())
//        {
//            time_series_new_raw_variables_number = Index(atoi(time_series_raw_variables_number_element->GetText()));

//            set_time_series_raw_variables_number(time_series_new_raw_variables_number);
//        }

//        // Time series raw_variables

//        const tinyxml2::XMLElement* time_series_start_element = time_series_raw_variables_number_element;

//        if(time_series_new_raw_variables_number > 0)
//        {
//            for(Index i = 0; i < time_series_new_raw_variables_number; i++)
//            {
//                const tinyxml2::XMLElement* time_series_raw_variable_element = time_series_start_element->NextSiblingElement("TimeSeriesColumn");
//                time_series_start_element = time_series_raw_variable_element;

//                if(time_series_raw_variable_element->Attribute("Item") != to_string(i+1))
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series raw_variable item number (" << i+1 << ") does not match (" << time_series_raw_variable_element->Attribute("Item") << ").\n";

//                    throw runtime_error(buffer.str());
//                }

//                // Name

//                const tinyxml2::XMLElement* time_series_name_element = time_series_raw_variable_element->FirstChildElement("Name");

//                if(!time_series_name_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series name element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_name_element->GetText())
//                {
//                    const string time_series_new_name = time_series_name_element->GetText();

//                    time_series_raw_variables(i).name = time_series_new_name;
//                }

//                // Scaler

//                const tinyxml2::XMLElement* time_series_scaler_element = time_series_raw_variable_element->FirstChildElement("Scaler");

//                if(!time_series_scaler_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series scaler element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_scaler_element->GetText())
//                {
//                    const string time_series_new_scaler = time_series_scaler_element->GetText();

//                    time_series_raw_variables(i).set_scaler(time_series_new_scaler);
//                }

//                // raw_variable use

//                const tinyxml2::XMLElement* time_series_raw_variable_use_element = time_series_raw_variable_element->FirstChildElement("RawVariableUse");

//                if(!time_series_raw_variable_use_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series raw_variable use element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_raw_variable_use_element->GetText())
//                {
//                    const string time_series_new_raw_variable_use = time_series_raw_variable_use_element->GetText();

//                    time_series_raw_variables(i).set_use(time_series_new_raw_variable_use);
//                }

//                // Type

//                const tinyxml2::XMLElement* time_series_type_element = time_series_raw_variable_element->FirstChildElement("Type");

//                if(!time_series_type_element)
//                {
//                    buffer << "OpenNN Exception: DataSet class.\n"
//                           << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                           << "Time series type element is nullptr.\n";

//                    throw runtime_error(buffer.str());
//                }

//                if(time_series_type_element->GetText())
//                {
//                    const string time_series_new_type = time_series_type_element->GetText();
//                    time_series_raw_variables(i).set_type(time_series_new_type);
//                }

//                if(time_series_raw_variables(i).type == ColumnType::Categorical || time_series_raw_variables(i).type == ColumnType::Binary)
//                {
//                    // Categories

//                    const tinyxml2::XMLElement* time_series_categories_element = time_series_raw_variable_element->FirstChildElement("Categories");

//                    if(!time_series_categories_element)
//                    {
//                        buffer << "OpenNN Exception: DataSet class.\n"
//                               << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                               << "Time series categories element is nullptr.\n";

//                        throw runtime_error(buffer.str());
//                    }

//                    if(time_series_categories_element->GetText())
//                    {
//                        const string time_series_new_categories = time_series_categories_element->GetText();

//                        time_series_raw_variables(i).categories = get_tokens(time_series_new_categories, ';');
//                    }

//                    // Categories uses

//                    const tinyxml2::XMLElement* time_series_categories_uses_element = time_series_raw_variable_element->FirstChildElement("CategoriesUses");

//                    if(!time_series_categories_uses_element)
//                    {
//                        buffer << "OpenNN Exception: DataSet class.\n"
//                               << "void raw_variable::from_XML(const tinyxml2::XMLDocument&) method.\n"
//                               << "Time series categories uses element is nullptr.\n";

//                        throw runtime_error(buffer.str());
//                    }

//                    if(time_series_categories_uses_element->GetText())
//                    {
//                        const string time_series_new_categories_uses = time_series_categories_uses_element->GetText();

//                        time_series_raw_variables(i).set_categories_uses(get_tokens(time_series_new_categories_uses, ';'));
//                    }
//                }
//            }
//        }
//    }
    */



    // Rows label

    cout << " -- Rows label --" << endl;

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

    cout << " -- Rows label --" << endl;



    // Samples

    cout << " -- Samples --" << endl;

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    cout << " -- Samples --" << endl;



    // Samples number

    cout << " -- Samples number --" << endl;

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

    cout << " -- Samples number --" << endl;



    // Samples uses

    cout << " -- Samples uses --" << endl;

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

    cout << " -- Samples uses --" << endl;



    // Missing values

    cout << " -- Missing values --" << endl;

    const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    cout << " -- Missing values --" << endl;



    // Missing values method

    cout << " -- Missing method --" << endl;

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

    cout << " -- Missing method --" << endl;



    // Missing values number

    cout << " -- Missing values number --" << endl;

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

    cout << " -- Missing values number --" << endl;



    // Preview data

    cout << " -- Preview data --" << endl;

    const tinyxml2::XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview data element is nullptr.\n";

        throw runtime_error(buffer.str());
    }

    cout << " -- Preview data --" << endl;



    // Preview size

    cout << " -- Preview size --" << endl;

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
    }

    cout << " -- Preview size --" << endl;



    // Preview data

    cout << " -- Preview data --" << endl;

    start_element = preview_size_element;

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

    cout << " -- Preview data --" << endl;



    // Display

    cout << " -- Display --" << endl;

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

    cout << " -- Display --" << endl;
}


/// Prints to the screen in text format the main numbers from the data set object.

void DataSet::print() const
{
    if(display)
    {
        const Index variables_number = get_variables_number();
        const Index samples_number = get_samples_number();

        cout << "Data set object summary:\n"
             << "Number of variables: " << variables_number << "\n"
             << "Number of samples: " << samples_number << "\n";
    }
}


/// Saves the members of a data set object to a XML-type file in an XML-type format.
/// @param file_name Name of data set XML-type file.

void DataSet::save(const string& file_name) const
{
    FILE* pFile = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter document(pFile);

    write_XML(document);

    fclose(pFile);
}


/// Loads the members of a data set object from a XML-type file:
/// <ul>
/// <li> Samples number.
/// <li> Training samples number.
/// <li> Training samples indices.
/// <li> Selection samples number.
/// <li> Selection samples indices.
/// <li> Testing samples number.
/// <li> Testing samples indices.
/// <li> Input variables number.
/// <li> Input variables indices.
/// <li> Target variables number.
/// <li> Target variables indices.
/// <li> Input variables name.
/// <li> Target variables name.
/// <li> Input variables description.
/// <li> Target variables description.
/// <li> Display.
/// <li> Data.
/// </ul>
/// Please mind about the file format. This is specified in the User's Guide.
/// @param file_name Name of data set XML-type file.

void DataSet::load(const string& file_name)
{
    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw runtime_error(buffer.str());
    }

    from_XML(document);
}


void DataSet::print_raw_variables() const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables(i).print();
        cout << endl;
    }

    cout << endl;

}

void DataSet::print_raw_variables_types() const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Numeric) cout << "Numeric ";
        else if(raw_variables(i).type == RawVariableType::Binary) cout << "Binary ";
        else if(raw_variables(i).type == RawVariableType::Categorical) cout << "Categorical ";
        else if(raw_variables(i).type == RawVariableType::DateTime) cout << "DateTime ";
        else if(raw_variables(i).type == RawVariableType::Constant) cout << "Constant ";
    }

    cout << endl;
}


void DataSet::print_raw_variables_uses() const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Input) cout << "Input ";
        else if(raw_variables(i).raw_variable_use == VariableUse::Target) cout << "Target ";
        else if(raw_variables(i).raw_variable_use == VariableUse::Unused) cout << "Unused ";
    }

    cout << endl;
}


void DataSet::print_raw_variables_scalers() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Tensor<Scaler, 1> scalers = get_raw_variables_scalers();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(scalers[i] == Scaler::NoScaling)
            cout << "NoScaling" << endl;
        else if(scalers[i] == Scaler::MinimumMaximum)
            cout << "MinimumMaximum" << endl;
        else if(scalers[i] == Scaler::MeanStandardDeviation)
            cout << "MeanStandardDeviation" << endl;
        else if(scalers[i] == Scaler::StandardDeviation)
            cout << "StandardDeviation" << endl;
        else if(scalers[i] == Scaler::Logarithm)
            cout << "Logarithm" << endl;
    }

    cout << endl;
}


/// Prints to the screen the values of the data matrix.

void DataSet::print_data() const
{
    if(display) cout << data << endl;
}


/// Prints to the screen a preview of the data matrix, i.e. the first, second and last samples.

void DataSet::print_data_preview() const
{
    if(!display) return;

    const Index samples_number = get_samples_number();

    if(samples_number > 0)
    {
        const Tensor<type, 1> first_sample = data.chip(0, 0);

        cout << "First sample:  \n";

        for(int i = 0; i< first_sample.dimension(0); i++)
        {
            cout  << first_sample(i) << "  ";
        }

        cout << endl;
    }

    if(samples_number > 1)
    {
        const Tensor<type, 1> second_sample = data.chip(1, 0);

        cout << "Second sample:  \n";

        for(int i = 0; i< second_sample.dimension(0); i++)
        {
            cout  << second_sample(i) << "  ";
        }

        cout << endl;
    }

    if(samples_number > 2)
    {
        const Tensor<type, 1> last_sample = data.chip(samples_number-1, 0);

        cout << "Last sample:  \n";

        for(int i = 0; i< last_sample.dimension(0); i++)
        {
            cout  << last_sample(i) << "  ";
        }

        cout << endl;
    }
}


/// Saves to the data file the values of the data matrix.

void DataSet::save_data() const
{
    std::ofstream file(data_source_path.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Matrix template." << endl
               << "void save_csv(const string&, const char&, const Vector<string>&, const Vector<string>&) method." << endl
               << "Cannot open matrix data file: " << data_source_path << endl;

        throw runtime_error(buffer.str());
    }

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    const Tensor<string, 1> variables_names = get_variables_names();

    char separator_char = ',';//get_separator_char();

    if(has_rows_labels)
    {
        file << "id" << separator_char;
    }
    for(Index j = 0; j < variables_number; j++)
    {
        file << variables_names[j];

        if(j != variables_number-1)
        {
            file << separator_char;
        }
    }

    file << endl;

    for(Index i = 0; i < samples_number; i++)
    {
        if(has_rows_labels)
        {
            file << rows_labels(i) << separator_char;
        }
        for(Index j = 0; j < variables_number; j++)
        {
            file << data(i,j);

            if(j != variables_number-1)
            {
                file << separator_char;
            }
        }

        file << endl;
    }

    file.close();
}


/// Saves to the data file the values of the data matrix in binary format.

void DataSet::save_data_binary(const string& binary_data_file_name) const
{
    regex accent_regex("[\\xC0-\\xFF]");
    std::ofstream file;

    #ifdef _WIN32

    if(regex_search(binary_data_file_name, accent_regex))
    {
        file.open(string_to_wide_string(binary_data_file_name), ios::binary);
    }
    else
    {
        file.open(binary_data_file_name.c_str(), ios::binary);
    }

    #else
        file.open(binary_data_file_name.c_str(), ios::binary);
    #endif

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class." << endl
               << "void save_data_binary() method." << endl
               << "Cannot open data binary file." << endl;

        throw runtime_error(buffer.str());
    }

    // Write data

    streamsize size = sizeof(Index);

    Index raw_variables_number = data.dimension(1);
    Index rows_number = data.dimension(0);

    cout << "Saving binary data file..." << endl;

    file.write(reinterpret_cast<char*>(&raw_variables_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    for(int i = 0; i < raw_variables_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
            value = data(j,i);

            file.write(reinterpret_cast<char*>(&value), size);
        }
    }

    file.close();

    cout << "Binary data file saved." << endl;
}


/// This method loads the data from a binary data file.
/// @todo Can load block of data instead of element by element?

void DataSet::load_data_binary()
{
    const regex accent_regex("[\\xC0-\\xFF]");
    ifstream file;

    #ifdef _WIN32

    if(regex_search(data_source_path, accent_regex))
    {
        file.open(string_to_wide_string(data_source_path), ios::binary);
    }
    else
    {
        file.open(data_source_path.c_str(), ios::binary);
    }
    #else
        file.open(data_source_path.c_str(), ios::binary);
    #endif

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load_data_binary() method.\n"
               << "Cannot open binary file: " << data_source_path << "\n";

        throw runtime_error(buffer.str());
    }

    streamsize size = sizeof(Index);

    Index raw_variables_number = 0;
    Index rows_number = 0;

    file.read(reinterpret_cast<char*>(&raw_variables_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value = type(0);

    data.resize(rows_number, raw_variables_number);

    for(Index i = 0; i < rows_number*raw_variables_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);
        data(i) = value;
    }

    file.close();
}


/// Returns a vector containing the number of samples of each class in the data set.
/// If the number of target variables is one then the number of classes is two.
/// If the number of target variables is greater than one then the number of classes is equal to the number of target variables.

Tensor<Index, 1> DataSet::calculate_target_distribution() const
{
    const Index samples_number = get_samples_number();
    const Index targets_number = get_target_variables_number();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    Tensor<Index, 1> class_distribution;

    if(targets_number == 1) // Two classes
    {
        class_distribution = Tensor<Index, 1>(2);

        Index target_index = target_variables_indices(0);

        Index positives = 0;
        Index negatives = 0;

        for(Index sample_index = 0; sample_index < Index(samples_number); sample_index++)
        {
            if(!isnan(data(Index(sample_index),target_index)))
            {
                if(data(Index(sample_index), target_index) < type(0.5))
                {
                    negatives++;
                }
                else
                {
                    positives++;
                }
            }
        }

        class_distribution(0) = negatives;
        class_distribution(1) = positives;
    }
    else // More than two classes
    {
        class_distribution = Tensor<Index, 1>(targets_number);

        class_distribution.setZero();

        for(Index i = 0; i < samples_number; i++)
        {
            if(get_sample_use(i) != SampleUse::Unused)
            {
                for(Index j = 0; j < targets_number; j++)
                {
                    if(isnan(data(i,target_variables_indices(j)))) continue;

                    if(data(i,target_variables_indices(j)) > type(0.5)) class_distribution(j)++;
                }
            }
        }
    }

    return class_distribution;
}


/// Calculate the outliers from the data set using Tukey's test.
/// @param cleaning_parameter Parameter used to detect outliers.

Tensor<Tensor<Index, 1>, 1> DataSet::calculate_Tukey_outliers(const type& cleaning_parameter) const
{
    const Index samples_number = get_used_samples_number();
    const Tensor<Index, 1> samples_indices = get_used_samples_indices();

    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const Tensor<Index, 1> used_raw_variables_indices = get_used_raw_variables_indices();

    Tensor<Tensor<Index, 1>, 1> return_values(2);

    return_values(0) = Tensor<Index, 1>(samples_number);
    return_values(1) = Tensor<Index, 1>(used_raw_variables_number);

    return_values(0).setZero();
    return_values(1).setZero();

    const Tensor<BoxPlot, 1> box_plots = calculate_raw_variables_box_plots();

    Index variable_index = 0;
    Index used_variable_index = 0;

#pragma omp parallel for
    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Unused && raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            continue;
        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Unused) // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            used_variable_index++;
            continue;
        }
        else if(raw_variables(i).type == RawVariableType::Binary || raw_variables(i).type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots(i).third_quartile - box_plots(i).first_quartile;

            if(interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(samples_indices(Index(j)));

                if(sample(variable_index) < (box_plots(i).first_quartile - cleaning_parameter * interquartile_range) ||
                    sample(variable_index) > (box_plots(i).third_quartile + cleaning_parameter * interquartile_range))
                {
                    return_values(0)(Index(j)) = 1;

                    raw_variables_outliers++;
                }
            }

            return_values(1)(used_variable_index) = raw_variables_outliers;

            variable_index++;
            used_variable_index++;
        }
    }

    return return_values;
}


/// Calculate the outliers from the data set using Tukey's test and sets in samples object.
/// @param cleaning_parameter Parameter used to detect outliers

Tensor<Tensor<Index, 1>, 1> DataSet::replace_Tukey_outliers_with_NaN(const type& cleaning_parameter)
{
    const Index samples_number = get_used_samples_number();
    const Tensor<Index, 1> samples_indices = get_used_samples_indices();

    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const Tensor<Index, 1> used_raw_variables_indices = get_used_raw_variables_indices();

    Tensor<Tensor<Index, 1>, 1> return_values(2);

    return_values(0) = Tensor<Index, 1>(samples_number);
    return_values(1) = Tensor<Index, 1>(used_raw_variables_number);

    return_values(0).setZero();
    return_values(1).setZero();

    Tensor<BoxPlot, 1> box_plots = calculate_raw_variables_box_plots();

    Index variable_index = 0;
    Index used_variable_index = 0;

#pragma omp parallel for
    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).raw_variable_use == VariableUse::Unused && raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            continue;
        }
        else if(raw_variables(i).raw_variable_use == VariableUse::Unused) // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            used_variable_index++;
            continue;
        }
        else if(raw_variables(i).type == RawVariableType::Binary || raw_variables(i).type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots(i).third_quartile - box_plots(i).first_quartile;

            if(interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(samples_indices(Index(j)));

                if(sample(variable_index) < (box_plots(i).first_quartile - cleaning_parameter * interquartile_range) ||
                    sample(variable_index) > (box_plots(i).third_quartile + cleaning_parameter * interquartile_range))
                {
                    return_values(0)(Index(j)) = 1;

                    raw_variables_outliers++;

                    data(samples_indices(Index(j)), variable_index) = numeric_limits<type>::quiet_NaN();
                }
            }

            return_values(1)(used_variable_index) = raw_variables_outliers;

            variable_index++;
            used_variable_index++;
        }
    }

    return return_values;
}

void DataSet::unuse_Tukey_outliers(const type& cleaning_parameter)
{
    const Tensor<Tensor<Index, 1>, 1> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    const Tensor<Index, 1> outliers_samples = get_elements_greater_than(outliers_indices, 0);

    set_samples_uses(outliers_samples, DataSet::SampleUse::Unused);
}


/// Generates an artificial data_set with a given number of samples and number of variables
/// by constant data.
/// @param samples_number Number of samples in the data_set.
/// @param variables_number Number of variables in the data_set.

void DataSet::generate_constant_data(const Index& samples_number, const Index& variables_number, const type& value)
{
    set(samples_number, variables_number);

    data.setConstant(value);

    set_default_raw_variables_uses();
}


/// Generates an artificial data_set with a given number of samples and number of variables
/// using random data.
/// @param samples_number Number of samples in the data_set.
/// @param variables_number Number of variables in the data_set.

void DataSet::generate_random_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number, variables_number);

    data.setRandom();
}


/// Generates an artificial data_set with a given number of samples and number of variables
/// using a sequential data.
/// @param samples_number Number of samples in the data_set.
/// @param variables_number Number of variables in the data_set.

void DataSet::generate_sequential_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number, variables_number);

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < variables_number; j++)
        {
            data(i,j) = type(j);
        }
    }
}


/// Generates an artificial data_set with a given number of samples and number of variables
/// using the Rosenbrock function.
/// @param samples_number Number of samples in the data_set.
/// @param variables_number Number of variables in the data_set.

void DataSet::generate_Rosenbrock_data(const Index& samples_number, const Index& variables_number)
{
    const Index inputs_number = variables_number-1;

    set(samples_number, variables_number);

    data.setRandom();

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        type rosenbrock(0);

        for(Index j = 0; j < inputs_number-1; j++)
        {
            const type value = data(i,j);
            const type next_value = data(i,j+1);

            rosenbrock += (type(1) - value)*(type(1) - value) + type(100)*(next_value-value*value)*(next_value-value*value);
        }

        data(i, inputs_number) = rosenbrock;
    }

    set_default_raw_variables_uses();

}

/// Generates an artifical data_set with a given number of samples, a number of features and a number of classes.
/// @param samples_numer Number of samples in the data_set.
/// @param variables_number Number of features to take into account in the classification problem.
/// @param classes_number Number of classes in the data_set.

void DataSet::generate_classification_data(const Index& samples_number, const Index& variables_number, const Index& classes_number)
{
    cout << "Generating Classification Data..." << endl;

    set(samples_number, variables_number + classes_number);

    data.setRandom();
    /*
    data.setConstant(0.0);

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < variables_number; j++)
        {

            data(i, j) = rand(); // rand();

        }
    }


#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        const Index random_class = rand() % classes_number;
        data(i, variables_number + random_class) = 1;
    }
*/
    cout << "Done." << endl;
}


void DataSet::generate_sum_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number,variables_number);

    data.setRandom();

    for(Index i = 0; i < samples_number; i++)
    {
        data(i,variables_number-1) = type(0);

        for(Index j = 0; j < variables_number-1; j++)
        {
            data(i,variables_number-1) += data(i,j);
        }
    }

    set(data);
}


/// Unuses those samples with values outside a defined range.
/// @param minimums vector of minimum values in the range.
/// The size must be equal to the number of variables.
/// @param maximums vector of maximum values in the range.
/// The size must be equal to the number of variables.

Tensor<Index, 1> DataSet::filter_data(const Tensor<type, 1>& minimums, const Tensor<type, 1>& maximums)
{
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();

    const Index used_variables_number = used_variables_indices.size();

#ifdef OPENNN_DEBUG

    if(minimums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Size of minimums(" << minimums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw runtime_error(buffer.str());
    }

    if(maximums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Size of maximums(" << maximums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw runtime_error(buffer.str());
    }

#endif

    const Index samples_number = get_samples_number();

    Tensor<type, 1> filtered_indices(samples_number);
    filtered_indices.setZero();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Index used_samples_number = used_samples_indices.size();

    Index sample_index = 0;

    for(Index i = 0; i < used_variables_number; i++)
    {
        const Index variable_index = used_variables_indices(i);

        for(Index j = 0; j < used_samples_number; j++)
        {
            sample_index = used_samples_indices(j);

//            if(minimums(i)) {

//            }
//            else
//            {
                if(get_sample_use(sample_index) == SampleUse::Unused)
                    continue;

                if(isnan(data(sample_index, variable_index)))
                    continue;

                if(abs(data(sample_index, variable_index) - minimums(i)) <= type(NUMERIC_LIMITS_MIN)
                    || abs(data(sample_index, variable_index) - maximums(i)) <= type(NUMERIC_LIMITS_MIN))
                    continue;

                if(minimums(i) == maximums(i))
                {
                    if(data(sample_index, variable_index) != minimums(i))
                    {
                        filtered_indices(sample_index) = type(1);
                        set_sample_use(sample_index, SampleUse::Unused);
                    }
                }
                else if(data(sample_index, variable_index) < minimums(i) || data(sample_index, variable_index) > maximums(i))
                {
                    filtered_indices(sample_index) = type(1);
                    set_sample_use(sample_index, SampleUse::Unused);
//                }

            }
        }
    }

    const Index filtered_samples_number =
            Index(count_if(filtered_indices.data(),
                                        filtered_indices.data()+filtered_indices.size(), [](type value)
                               {return value > type(0.5);}));

    Tensor<Index, 1> filtered_samples_indices(filtered_samples_number);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(filtered_indices(i) > type(0.5))
        {
            filtered_samples_indices(index) = i;
            index++;
        }
    }

    return filtered_samples_indices;
}


/// Sets all the samples with missing values to "Unused".

void DataSet::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

#pragma omp parallel for

    for(Index i = 0; i <samples_number; i++)
    {
        if(has_nan_row(i)) set_sample_use(i, "Unused");
    }
}


/// Substitutes all the missing values by the mean of the corresponding variable.

void DataSet::impute_missing_values_mean()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Tensor<type, 1> means = mean(data, used_samples_indices, used_variables_indices);

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    Index current_variable;
    Index current_sample;

#pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
            {
                data(current_sample,current_variable) = means(j);
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
            {
                set_sample_use(i, "Unused");
            }
        }
    }
}


/// Substitutes all the missing values by the median of the corresponding variable.

void DataSet::impute_missing_values_median()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Tensor<type, 1> medians = median(data, used_samples_indices, used_variables_indices);

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    Index current_variable;
    Index current_sample;

#pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
            {
                data(current_sample,current_variable) = medians(j);
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
            {
                set_sample_use(i, "Unused");
            }
        }
    }
}


/// Substitutes all the missing values by the interpolation of the corresponding variable.

void DataSet::impute_missing_values_interpolate()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    Index current_variable;
    Index current_sample;

#pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
            {
                type x1 = type(0);
                type x2 = type(0);
                type y1 = type(0);
                type y2 = type(0);
                type x = type(0);
                type y = type(0);

                for(Index k = i - 1; k >= 0; k--)
                {
                    if(!isnan(data(used_samples_indices(k), current_variable)))
                    {
                        x1 = type(used_samples_indices(k));
                        y1 = data(x1, current_variable);
                        break;
                    }
                }
                for(Index k = i + 1; k < samples_number; k++)
                {
                    if(!isnan(data(used_samples_indices(k), current_variable)))
                    {
                        x2 = type(used_samples_indices(k));
                        y2 = data(x2, current_variable);
                        break;
                    }
                }
                if(x2 != x1)
                {
                    x = type(current_sample);
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
                }
                else
                {
                    y = y1;
                }

                data(current_sample,current_variable) = y;
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
            {
                set_sample_use(i, "Unused");
            }
        }
    }
}


/// General method for dealing with missing values.
/// It switches among the different scrubbing methods available,
/// according to the corresponding value in the missing values object.

void DataSet::scrub_missing_values()
{
    switch(missing_values_method)
    {
    case MissingValuesMethod::Unuse:

        impute_missing_values_unuse();

        break;

    case MissingValuesMethod::Mean:

        impute_missing_values_mean();

        break;

    case MissingValuesMethod::Median:

        impute_missing_values_median();

        break;

    case MissingValuesMethod::Interpolation:

        impute_missing_values_interpolate();

        break;
    }
}


void DataSet::load_data()
{
    read_csv_1();

    if(!has_time_raw_variables() && !has_categorical_raw_variables())
    {
        read_csv_2_simple();

        read_csv_3_simple();
    }
    else
    {
        read_csv_2_complete();

        read_csv_3_complete();
    }
}


void DataSet::read_csv()
{
    read_csv_1();

    if(!has_time_raw_variables() && !has_categorical_raw_variables())
    {
        read_csv_2_simple();

        read_csv_3_simple();
    }
    else
    {
        read_csv_2_complete();

        read_csv_3_complete();
    }
}


Tensor<string, 1> DataSet::get_default_raw_variables_names(const Index& raw_variables_number)
{
    Tensor<string, 1> raw_variables_names(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        ostringstream buffer;

        buffer << "column_" << i+1;

        raw_variables_names(i) = buffer.str();
    }

    return raw_variables_names;
}

string DataSet::get_raw_variable_type_string(const RawVariableType& raw_variable_type)
{
    switch(raw_variable_type)
    {
    case RawVariableType::Numeric:
        return "Numeric";

    case RawVariableType::Constant:
        return "Constant";

    case RawVariableType::Binary:
        return "Binary";

    case RawVariableType::Categorical:
        return "Categorical";

    case RawVariableType::DateTime:
        return "DateTime";

    default:
        return "";
    }
}


string DataSet::get_raw_variable_use_string(const VariableUse& raw_variable_use)
{

    switch(raw_variable_use)
    {
    case VariableUse::Input:
        return "Input";

    case VariableUse::Target:
        return "Target";

    case VariableUse::Time:
        return "Time";

    case VariableUse::Unused:
        return "Unused";

    default:
        return "";
    }
}

void DataSet::read_csv_1()
{
    if(display) cout << "Path: " << data_source_path << endl;

    if(data_source_path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv() method.\n"
               << "Data file name is empty.\n";

        throw runtime_error(buffer.str());
    }

    std::regex accent_regex("[\\xC0-\\xFF]");
    std::ifstream file;

#ifdef _WIN32

    if(std::regex_search(data_source_path, accent_regex))
    {
        file.open(string_to_wide_string(data_source_path));
    }else
    {
        file.open(data_source_path.c_str());
    }
#else
    file.open(data_source_path.c_str());
#endif

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv() method.\n"
               << "Cannot open data file: " << data_source_path << "\n";

        throw runtime_error(buffer.str());
    }

    const char separator_char = get_separator_char();

    if(display) cout << "Setting data file preview..." << endl;

    Index lines_number = has_raw_variables_names ? 4 : 3;

    data_file_preview.resize(lines_number);

    string line;

    Index lines_count = 0;

    while(file.good())
    {
        getline(file, line);

        line = decode(line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        check_separators(line);

        data_file_preview(lines_count) = get_tokens(line, separator_char);

        lines_count++;

        if(lines_count == lines_number) break;
    }

    file.close();

    // Check empty file

    if(data_file_preview(0).size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "File " << data_source_path << " is empty.\n";

        throw runtime_error(buffer.str());
    }

    // Set rows labels and raw_variables names

    if(display) cout << "Setting rows labels..." << endl;

    string first_name = data_file_preview(0)(0);
    transform(first_name.begin(), first_name.end(), first_name.begin(), ::tolower);

    const Index raw_variables_number = has_rows_labels ? data_file_preview(0).size()-1 : data_file_preview(0).size();

    raw_variables.resize(raw_variables_number);

    // Check if header has numeric value

    if(has_raw_variables_names && has_numbers(data_file_preview(0)))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "Some raw_variables names are numeric.\n";

        throw runtime_error(buffer.str());
    }

    // raw_variables names

    if(display) cout << "Setting raw_variables names..." << endl;

    if(has_raw_variables_names)
    {
        has_rows_labels ? set_raw_variables_names(data_file_preview(0).slice(Eigen::array<Eigen::Index, 1>({1}),
                                                                       Eigen::array<Eigen::Index, 1>({data_file_preview(0).size()-1})))
                        : set_raw_variables_names(data_file_preview(0));
    }
    else
    {
        set_raw_variables_names(get_default_raw_variables_names(raw_variables_number));
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
            if(has_rows_labels && i == 0) continue;

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

                file.open(data_source_path.c_str());

                if(!file.is_open())
                {
                    ostringstream buffer;

                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void read_csv() method.\n"
                           << "Cannot open data file: " << data_source_path << "\n";

                    throw runtime_error(buffer.str());
                }

                while(file.good())
                {
                    getline(file, line);
                    line = decode(line);
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
        if(has_rows_labels && i == 0) continue;
        
        string data_file_preview_1 = data_file_preview(1)(i);
        string data_file_preview_2 = data_file_preview(2)(i);
        string data_file_preview_3 = data_file_preview(lines_number-2)(i);
        string data_file_preview_4 = data_file_preview(lines_number-1)(i);
        
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
            raw_variables(raw_variable_index).type = RawVariableType::DateTime;
//            time_column = raw_variables(raw_variable_index).name;
            raw_variable_index++;
        }
        else if(((is_numeric_string(data_file_preview_1) && data_file_preview_1 != missing_values_label) || data_file_preview_1.empty())
                || ((is_numeric_string(data_file_preview_2) && data_file_preview_2 != missing_values_label) || data_file_preview_2.empty())
                || ((is_numeric_string(data_file_preview_3) && data_file_preview_3 != missing_values_label) || data_file_preview_3.empty())
                || ((is_numeric_string(data_file_preview_4) && data_file_preview_4 != missing_values_label) || data_file_preview_4.empty()))
        {
            raw_variables(raw_variable_index).type = RawVariableType::Numeric;
            raw_variable_index++;
        }
        else
        {
            raw_variables(raw_variable_index).type = RawVariableType::Categorical;
            raw_variable_index++;
        }
    }

    // Resize data file preview to original

    if(data_file_preview.size() > 4)
    {
        lines_number = has_raw_variables_names ? 4 : 3;

        Tensor<Tensor<string, 1>, 1> data_file_preview_copy(data_file_preview);

        data_file_preview.resize(lines_number);

        data_file_preview(0) = data_file_preview_copy(1);
        data_file_preview(1) = data_file_preview_copy(1);
        data_file_preview(2) = data_file_preview_copy(2);
        data_file_preview(lines_number - 2) = data_file_preview_copy(data_file_preview_copy.size()-2);
        data_file_preview(lines_number - 1) = data_file_preview_copy(data_file_preview_copy.size()-1);
    }
    
}



void DataSet::read_csv_2_simple()
{   
    regex accent_regex("[\\xC0-\\xFF]");
    std::ifstream file;

    #ifdef _WIN32

    if(regex_search(data_source_path, accent_regex))
    {
        file.open(string_to_wide_string(data_source_path));
    }else
    {
        file.open(data_source_path.c_str());
    }

    #else
        file.open(data_source_path.c_str());
    #endif

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_2_simple() method.\n"
               << "Cannot open data file: " << data_source_path << "\n";

        throw runtime_error(buffer.str());
    }

    string line;
    Index line_number = 0;

    if(has_raw_variables_names)
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

    const char separator_char = get_separator_char();

    const Index raw_variables_number = get_raw_variables_number();
    const Index raw_raw_variables_number = has_rows_labels ? raw_variables_number + 1 : raw_variables_number;

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens_count = count_tokens(line, separator_char);

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

    samples_uses.resize(samples_count);
    samples_uses.setConstant(SampleUse::Training);

    split_samples_random();
}


void DataSet::read_csv_3_simple()
{
    const regex accent_regex("[\\xC0-\\xFF]");
    ifstream file;

    #ifdef _WIN32

    if(std::regex_search(data_source_path, accent_regex))
    {
        file.open(string_to_wide_string(data_source_path));
    }else
    {
        file.open(data_source_path.c_str());
    }

    #else
        file.open(data_source_path.c_str());
    #endif

    if(!file.is_open())
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

    if(has_raw_variables_names)
    {
        while(file.good())
        {
            getline(file, line);

            line = decode(line);

            if(line.empty()) continue;

            break;
        }
    }

    // Read data

    const Index raw_raw_variables_number = has_rows_labels ? get_raw_variables_number() + 1 : get_raw_variables_number();

    Tensor<string, 1> tokens(raw_raw_variables_number);

    const Index samples_number = data.dimension(0);

    if(has_rows_labels) rows_labels.resize(samples_number);

    if(display) cout << "Reading data..." << endl;

    Index sample_index = 0;
    Index raw_variable_index = 0;

    while(file.good())
    {
        getline(file, line);

        line = decode(line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        fill_tokens(line, separator_char, tokens);

        for(Index j = 0; j < raw_raw_variables_number; j++)
        {
            trim(tokens(j));

            if(has_rows_labels && j == 0)
            {
                rows_labels(sample_index) = tokens(j);
            }
            else if(tokens(j) == missing_values_label || tokens(j).empty())
            {
                data(sample_index, raw_variable_index) = type(NAN);
                raw_variable_index++;
            }
            else if(is_float)
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

        type percentage = static_cast<type>(sample_index)/static_cast<type>(samples_number);
    }

    const Index data_file_preview_index = has_raw_variables_names ? 3 : 2;

    data_file_preview(data_file_preview_index) = tokens;

    file.close();

    if(display) cout << "Data read succesfully..." << endl;

    // Check Constant

    check_constant_raw_variables();

    // Check Binary

    if(display) cout << "Checking binary raw_variables..." << endl;

    set_binary_simple_raw_variables();
}


void DataSet::read_csv_2_complete()
{
    regex accent_regex("[\\xC0-\\xFF]");
    std::ifstream file;

    #ifdef _WIN32

    if(regex_search(data_source_path, accent_regex))
    {
        file.open(string_to_wide_string(data_source_path));
    }
    else
    {
        file.open(data_source_path.c_str());
    }
#else
    file.open(data_source_path.c_str());
#endif


    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_2_complete() method.\n"
               << "Cannot open data file: " << data_source_path << "\n";

        throw runtime_error(buffer.str());
    }

    const char separator_char = get_separator_char();

    string line;

    Tensor<string, 1> tokens;

    Index lines_count = 0;
    Index tokens_count;

    const Index raw_variables_number = raw_variables.size();

    for(Index j = 0; j < raw_variables_number; j++)
    {
        if(raw_variables(j).type != RawVariableType::Categorical)
        {
            raw_variables(j).raw_variable_use = VariableUse::Input;
        }
    }

    // Skip header

    if(has_raw_variables_names)
    {
        while(file.good())
        {
            getline(file, line);

            trim(line);

            if(line.empty()) continue;

            break;
        }
    }

    // Read data

    if(display) cout << "Setting data dimensions..." << endl;

    const Index raw_raw_variables_number = has_rows_labels ? raw_variables_number + 1 : raw_variables_number;

    Index raw_variable_index = 0;

    while(file.good())
    {
        getline(file, line);

        line = decode(line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_char);

        tokens_count = tokens.size();

        if(static_cast<unsigned>(tokens_count) != raw_raw_variables_number)
        {
            const string message =
                    "Sample " + to_string(lines_count+1) + " error:\n"
                                                           "Size of tokens (" + to_string(tokens_count) + ") is not equal to number of raw_variables (" + to_string(raw_raw_variables_number) + ").\n"
                                                                                                                                                                                    "Please check the format of the data file (e.g: Use of commas both as decimal and raw_variable separator)";

            throw runtime_error(message);
        }

        for(unsigned j = 0; j < raw_raw_variables_number; j++)
        {
            if(has_rows_labels && j == 0) continue;

            if(raw_variables(raw_variable_index).type == RawVariableType::Categorical)
            {
                if(find(raw_variables(raw_variable_index).categories.data(), raw_variables(raw_variable_index).categories.data() + raw_variables(raw_variable_index).categories.size(), tokens(j)) == (raw_variables(raw_variable_index).categories.data() + raw_variables(raw_variable_index).categories.size()))
                {
                    if(tokens(j) == missing_values_label || tokens(j).find(missing_values_label) != string::npos)
                    {
                        raw_variable_index++;
                        continue;
                    }

                    raw_variables(raw_variable_index).add_category(tokens(j));
                }
            }

            raw_variable_index++;
        }

        raw_variable_index = 0;

        lines_count++;
    }

    if(display) cout << "Setting types..." << endl;

    for(Index j = 0; j < raw_variables_number; j++)
    {
        if(raw_variables(j).type == RawVariableType::Categorical)
        {
            if(raw_variables(j).categories.size() == 2)
            {
                raw_variables(j).type = RawVariableType::Binary;
            }
        }
    }

    file.close();

    const Index samples_number = static_cast<unsigned>(lines_count);

    const Index variables_number = get_variables_number();

    data.resize(Index(samples_number), variables_number);
    data.setZero();

    if(has_rows_labels) rows_labels.resize(samples_number);

    set_default_raw_variables_uses();

    samples_uses.resize(Index(samples_number));

    samples_uses.setConstant(SampleUse::Training);

    split_samples_random();

}


void DataSet::read_csv_3_complete()
{
    const regex accent_regex("[\\xC0-\\xFF]");
    ifstream file;

    #ifdef _WIN32

    if(regex_search(data_source_path, accent_regex))
    {
        file.open(string_to_wide_string(data_source_path));
    }
    else
    {
        file.open(data_source_path.c_str());
    }

    #else
        file.open(data_source_path.c_str());
    #endif


    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_3_complete() method.\n"
               << "Cannot open data file: " << data_source_path << "\n";

        throw runtime_error(buffer.str());
    }

    const char separator_char = get_separator_char();

    const Index raw_variables_number = raw_variables.size();

    const Index raw_raw_variables_number = has_rows_labels ? raw_variables_number + 1 : raw_variables_number;

    string line;

    Tensor<string, 1> tokens;

    string token;

    unsigned sample_index = 0;
    unsigned variable_index = 0;
    unsigned raw_variable_index = 0;

    // Skip header

    if(has_raw_variables_names)
    {
        while(file.good())
        {
            getline(file, line);

            line = decode(line);

            trim(line);

            if(line.empty()) continue;

            break;
        }
    }

    // Read data

    if(display) cout << "Reading data..." << endl;

    time_t previous_timestamp = 0;
    double time_step = 60 * 60 * 24;

    while(file.good())
    {
        getline(file, line);

        line = decode(line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_char);

        variable_index = 0;
        raw_variable_index = 0;
        bool insert_nan_row = false;

        for(Index j = 0; j < raw_raw_variables_number; j++)
        {
            trim(tokens(j));

            if(has_rows_labels && j ==0)
            {
                rows_labels(sample_index) = tokens(j);
                continue;
            }
            else if(raw_variables(raw_variable_index).type == RawVariableType::Numeric)
            {
                if(tokens(j) == missing_values_label || tokens(j).empty())
                {
                    data(sample_index, variable_index) = type(NAN);
                    variable_index++;
                }
                else
                {
                    try
                    {
                        data(sample_index, variable_index) = type(stod(tokens(j)));
                        variable_index++;
                    }
                    catch(const exception& e)
                    {
                        ostringstream buffer;

                        buffer << "OpenNN Exception: DataSet class.\n"
                               << "void read_csv_3_complete() method.\n"
                               << "Sample " << sample_index << "; Invalid number: " << tokens(j) << "\n";

                        throw runtime_error(buffer.str());
                    }
                }
            }
            else if(raw_variables(raw_variable_index).type == RawVariableType::DateTime)
            {
                time_t current_timestamp = 0;

                if(!(tokens(j) == missing_values_label || tokens(j).empty()))
                {
                    current_timestamp = static_cast<time_t>(date_to_timestamp(tokens(j)));
                }

                while(previous_timestamp != 0 && difftime(current_timestamp, previous_timestamp) > time_step)
                {
                    for(Index raw_variables_index = 0; raw_variables_index < raw_variables_number; ++raw_variables_index)
                    {
                        data(sample_index, raw_variables_index) = type(NAN);
                    }
                    sample_index++;

                    previous_timestamp += time_step;
                }

                if(tokens(j).empty())
                {
                    data(sample_index, variable_index) = type(NAN);
                }
                else
                {
                    previous_timestamp = current_timestamp;
                    data(sample_index, variable_index) = type(current_timestamp);
                }

                variable_index++;
            }
            // else if(raw_variables(raw_variable_index).type == RawVariableType::DateTime)
            // {
            //     time_t current_timestamp = 0;

            //     if(!(tokens(j) == missing_values_label || tokens(j).empty()))
            //     {
            //         current_timestamp = static_cast<time_t>(date_to_timestamp(tokens(j)));
            //     }

            //     if(previous_timestamp != 0 && difftime(current_timestamp, previous_timestamp) > time_step)
            //     {
            //         insert_nan_row = true;
            //         previous_timestamp += time_step;
            //         break;
            //     }
            //     else
            //     {
            //         previous_timestamp = current_timestamp;
            //         data(sample_index, variable_index) = tokens(j).empty() ? type(NAN) : current_timestamp;
            //         variable_index++;
            //     }
            // }
            else if(raw_variables(raw_variable_index).type == RawVariableType::Categorical)
            {
                for(Index k = 0; k < raw_variables(raw_variable_index).get_categories_number(); k++)
                {
                    if(tokens(j) == missing_values_label)
                    {
                        data(sample_index, variable_index) = type(NAN);
                    }
                    else if(tokens(j) == raw_variables(raw_variable_index).categories(k))
                    {
                        data(sample_index, variable_index) = type(1);
                    }

                    variable_index++;
                }
            }
            else if(raw_variables(raw_variable_index).type == RawVariableType::Binary)
            {
                string lower_case_token = tokens(j);

                trim(lower_case_token);
                transform(lower_case_token.begin(), lower_case_token.end(), lower_case_token.begin(), ::tolower);

                Tensor<string,1> positive_words(5);
                Tensor<string,1> negative_words(5);

                positive_words.setValues({"yes", "positive", "+", "true", "si"});
                negative_words.setValues({"no", "negative", "-", "false", "no"});

                if(tokens(j) == missing_values_label || tokens(j).find(missing_values_label) != string::npos)
                {
                    data(sample_index, variable_index) = type(NAN);
                }
                else if( contains(positive_words, lower_case_token) )
                {
                    data(sample_index, variable_index) = type(1);
                }
                else if( contains(negative_words, lower_case_token) )
                {
                    data(sample_index, variable_index) = type(0);
                }
                else if(raw_variables(raw_variable_index).categories.size() > 0 && tokens(j) == raw_variables(raw_variable_index).categories(0))
                {
                    data(sample_index, variable_index) = type(1);
                }
                else if(tokens(j) == raw_variables(raw_variable_index).name)
                {
                    data(sample_index, variable_index) = type(1);
                }

                variable_index++;
            }

            raw_variable_index++;
        }

        if(insert_nan_row)
        {
            for(Index raw_variables_index = 0; raw_variables_index < raw_variables_number; ++raw_variables_index)
            {
                data(sample_index, raw_variables_index) = type(NAN);
            }
            sample_index++;

            continue;
        }

        sample_index++;

        type percentage = static_cast<type>(sample_index)/static_cast<type>(data.dimension(0));
    }

    const Index data_file_preview_index = has_raw_variables_names ? 3 : 2;

    data_file_preview(data_file_preview_index) = tokens;

    if(display) cout << "Data read succesfully..." << endl;

    file.close();

    // Check Constant and DateTime to unused

    check_constant_raw_variables();

    // Check binary

    if(display) cout << "Checking binary raw_variables..." << endl;

    ofstream myfile;
    myfile.open ("/home/artelnics/Escritorio/example.txt");
    myfile << "data: " << data << endl;
    myfile.close();

    set_binary_simple_raw_variables();
}


void DataSet::check_separators(const string& line) const
{
    if(line.find(',') == string::npos
            && line.find(';') == string::npos
            && line.find(' ') == string::npos
            && line.find('\t') == string::npos) return;

    const char separator_char = get_separator_char();

    if(line.find(separator_char) == string::npos)
    {
        const string message =
                "Error: " + get_separator_string() + " separator not found in line data file " + data_source_path + ".\n"
                                                                                                                  "Line: '" + line + "'";

        throw runtime_error(message);
    }

    if(separator == Separator::Space)
    {
        if(line.find(',') != string::npos)
        {
            const string message =
                    "Error: Found comma (',') in data file " + data_source_path + ", but separator is space (' ').";

            throw runtime_error(message);
        }
        if(line.find(';') != string::npos)
        {
            const string message =
                    "Error: Found semicolon (';') in data file " + data_source_path + ", but separator is space (' ').";

            throw runtime_error(message);
        }
    }
    else if(separator == Separator::Tab)
    {
        if(line.find(',') != string::npos)
        {
            const string message =
                    "Error: Found comma (',') in data file " + data_source_path + ", but separator is tab ('   ').";

            throw runtime_error(message);
        }
        if(line.find(';') != string::npos)
        {
            const string message =
                    "Error: Found semicolon (';') in data file " + data_source_path + ", but separator is tab ('   ').";

            throw runtime_error(message);
        }
    }
    else if(separator == Separator::Comma)
    {
        if(line.find(";") != string::npos)
        {
            const string message =
                    "Error: Found semicolon (';') in data file " + data_source_path + ", but separator is comma (',').";

            throw runtime_error(message);
        }
    }
    else if(separator == Separator::Semicolon)
    {
        if(line.find(",") != string::npos)
        {
            const string message =
                    "Error: Found comma (',') in data file " + data_source_path + ", but separator is semicolon (';'). " + line;

            throw runtime_error(message);
        }
    }
}


void DataSet::check_special_characters(const string & line) const
{
    if( line.find_first_of("|@#~€¬^*") != string::npos)
    {
        const string message =
                "Error: found special characters in line: " + line + ". Please, review the file.";
        throw runtime_error(message);
    }

    //#ifdef __unix__
    //    if(line.find("\r") != string::npos)
    //    {
    //        const string message =
    //                "Error: mixed break line characters in line: " + line + ". Please, review the file.";
    //        throw runtime_error(message);
    //    }
    //#endif

}


bool DataSet::has_binary_raw_variables() const
{
    const Index variables_number = raw_variables.size();

    for(Index i = 0; i < variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Binary) return true;
    }

    return false;
}


bool DataSet::has_categorical_raw_variables() const
{
    const Index variables_number = raw_variables.size();

    for(Index i = 0; i < variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical) return true;
    }

    return false;
}


bool DataSet::has_time_raw_variables() const
{
    const Index raw_variables_number = raw_variables.size();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::DateTime) return true;
    }

    return false;
}


//bool DataSet::has_time_time_series_raw_variables() const
//{
//    const Index time_series_raw_variables_number = time_series_raw_variables.size();
//
//    for(Index i = 0; i < time_series_raw_variables_number; i++)
//    {
//        if(time_series_raw_variables(i).type == ColumnType::DateTime) return true;
//    }
//
//    return false;
//}


bool DataSet::has_selection() const
{
    if(get_selection_samples_number() == 0) return false;

    return true;
}


Tensor<Index, 1> DataSet::count_nan_raw_variables() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index rows_number = get_samples_number();

    Tensor<Index, 1> nan_raw_variables(raw_variables_number);
    nan_raw_variables.setZero();

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        const Index current_variable_index = get_numeric_variable_indices(raw_variable_index)(0);

        for(Index row_index = 0; row_index < rows_number; row_index++)
        {
            if(isnan(data(row_index,current_variable_index)))
            {
                nan_raw_variables(raw_variable_index)++;
            }
        }
    }

    return nan_raw_variables;
}


Index DataSet::count_rows_with_nan() const
{
    Index rows_with_nan = 0;

    const Index rows_number = data.dimension(0);
    const Index raw_variables_number = data.dimension(1);

    bool has_nan = true;

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        has_nan = false;

        for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
        {
            if(isnan(data(row_index, raw_variable_index)))
            {
                has_nan = true;
                break;
            }
        }

        if(has_nan) rows_with_nan++;
    }

    return rows_with_nan;
}


Index DataSet::count_nan() const
{
    return count_NAN(data);
}


void DataSet::set_missing_values_number(const Index& new_missing_values_number)
{
    missing_values_number = new_missing_values_number;
}


void DataSet::set_missing_values_number()
{
    missing_values_number = count_nan();
}


void DataSet::set_raw_variables_missing_values_number(const Tensor<Index, 1>& new_raw_variables_missing_values_number)
{
    raw_variables_missing_values_number = new_raw_variables_missing_values_number;
}


void DataSet::set_raw_variables_missing_values_number()
{
    raw_variables_missing_values_number = count_nan_raw_variables();
}


void DataSet::set_rows_missing_values_number(const Index& new_rows_missing_values_number)
{
    rows_missing_values_number = new_rows_missing_values_number;
}


void DataSet::set_rows_missing_values_number()
{
    rows_missing_values_number = count_rows_with_nan();
}


void DataSet::fix_repeated_names()
{
    // Fix raw_variables names

    const Index raw_variables_number = raw_variables.size();

    map<string, Index> raw_variables_count_map;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        auto result = raw_variables_count_map.insert(pair<string, Index>(raw_variables(i).name, 1));

        if(!result.second) result.first->second++;
    }

    for(const auto & element : raw_variables_count_map)
    {
        if(element.second > 1)
        {
            const string repeated_name = element.first;
            Index repeated_index = 1;

            for(Index i = 0; i < raw_variables.size(); i++)
            {
                if(raw_variables(i).name == repeated_name)
                {
                    raw_variables(i).name = raw_variables(i).name + "_" + to_string(repeated_index);
                    repeated_index++;
                }
            }
        }
    }

    // Fix variables names

    if(has_categorical_raw_variables() || has_binary_raw_variables())
    {
        Tensor<string, 1> variables_names = get_variables_names();

        const Index variables_number = variables_names.size();

        map<string, Index> variables_count_map;

        for(Index i = 0; i < variables_number; i++)
        {
            auto result = variables_count_map.insert(pair<string, Index>(variables_names(i), 1));

            if(!result.second) result.first->second++;
        }

        for(const auto & element : variables_count_map)
        {
            if(element.second > 1)
            {
                const string repeated_name = element.first;

                for(Index i = 0; i < variables_number; i++)
                {
                    if(variables_names(i) == repeated_name)
                    {
                        const Index raw_variable_index = get_raw_variable_index(i);

                        if(raw_variables(raw_variable_index).type != RawVariableType::Categorical) continue;

                        variables_names(i) = variables_names(i) + "_" + raw_variables(raw_variable_index).name;
                    }
                }
            }
        }

        set_variables_names(variables_names);
    }
}


Tensor<Index, 2> DataSet::split_samples(const Tensor<Index, 1>& samples_indices, const Index& new_batch_size) const
{
    const Index samples_number = samples_indices.dimension(0);

    Index batches_number;
    Index batch_size = new_batch_size;

    if(samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
    }
    else
    {
        batches_number = samples_number / batch_size;
    }

    Tensor<Index, 2> batches(batches_number, batch_size);

    Index count = 0;

    for(Index i = 0; i < batches_number;++i)
    {
        for(Index j = 0; j < batch_size;++j)
        {
            batches(i, j) = samples_indices(count);

            count++;
        }
    }

    return batches;
}


void DataSet::shuffle()
{
    random_device rng;
    mt19937 urng(rng());

    const Index data_rows = data.dimension(0);
    const Index data_raw_variables = data.dimension(1);

    Tensor<Index, 1> indices(data_rows);

    for(Index i = 0; i < data_rows; i++) indices(i) = i;

    std::shuffle(&indices(0), &indices(data_rows-1), urng);

    Tensor<type, 2> new_data(data_rows, data_raw_variables);
    Tensor<string, 1> new_rows_labels(data_rows);

    Index index = 0;

    for(Index i = 0; i < data_rows; i++)
    {
        index = indices(i);

        new_rows_labels(i) = rows_labels(index);

        for(Index j = 0; j < data_raw_variables; j++)
        {
            new_data(i,j) = data(index,j);
        }
    }

    data = new_data;
    rows_labels = new_rows_labels;
}


bool DataSet::get_has_rows_labels() const
{
    return has_rows_labels;
}


/// This method parses a string decoded in non UTF8 decodification to UTF8
/// @param input_string String to be parsed

string DataSet::decode(const string& input_string) const
{
    switch(codification)
    {
    case DataSet::Codification::UTF8:
    {
        return input_string;
    }

    case DataSet::Codification::SHIFT_JIS:
    {
        return sj2utf8(input_string);
    }

    default:
        return input_string;
    }
}


/// This method checks if the input data file has the correct format. Returns an error message.

void DataSet::check_input_csv(const string & input_data_file_name, const char & separator_char) const
{
    std::ifstream file(input_data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void check_input_csv() method.\n"
               << "Cannot open input data file: " << input_data_file_name << "\n";

        throw runtime_error(buffer.str());
    }

    string line;
    Index line_number = 0;
    Index total_lines = 0;

    Index tokens_count;

    Index raw_variables_number = get_raw_variables_number() - get_target_raw_variables_number();
    if(model_type == ModelType::AutoAssociation)
        raw_variables_number = get_raw_variables_number() - get_target_raw_variables_number() - get_unused_raw_variables_number()/2;

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        total_lines++;

        tokens_count = count_tokens(line, separator_char);

        if(tokens_count != raw_variables_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void check_input_csv() method.\n"
                   << "Line " << line_number << ": Size of tokens in input file ("
                   << tokens_count << ") is not equal to number of raw_variables("
                   << raw_variables_number << "). \n"
                   << "Input csv must contain values for all the variables except the target. \n";

            throw runtime_error(buffer.str());
        }
    }

    file.close();

    if(total_lines == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void check_input_csv() method.\n"
               << "Input data file is empty. \n";

        throw runtime_error(buffer.str());
    }
}


/// This method loads data from a file and returns a matrix containing the input raw_variables.

Tensor<type, 2> DataSet::read_input_csv(const string& input_data_file_name,
                                        const char& separator_char,
                                        const string& missing_values_label,
                                        const bool& has_raw_variables_name,
                                        const bool& has_rows_label) const
{
    std::ifstream file(input_data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_input_csv() method.\n"
               << "Cannot open input data file: " << input_data_file_name << "\n";

        throw runtime_error(buffer.str());
    }

    // Count samples number

    Index input_samples_count = 0;

    string line;
    Index line_number = 0;

    Index tokens_count;

    Index raw_variables_number = get_raw_variables_number() - get_target_raw_variables_number();

    if(model_type == ModelType::AutoAssociation)
        raw_variables_number = get_raw_variables_number() - get_target_raw_variables_number() - get_unused_raw_variables_number()/2;

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens_count = count_tokens(line, separator_char);

        if(tokens_count != raw_variables_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void read_input_csv() method.\n"
                   << "Line " << line_number << ": Size of tokens("
                   << tokens_count << ") is not equal to number of raw_variables("
                   << raw_variables_number << ").\n";

            throw runtime_error(buffer.str());
        }

        input_samples_count++;
    }

    file.close();

    Index variables_number = get_input_variables_number();

    if(has_raw_variables_name) input_samples_count--;

    Tensor<type, 2> inputs_data(input_samples_count, variables_number);
    inputs_data.setZero();

    // Fill input data

    file.open(input_data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_input_csv() method.\n"
               << "Cannot open input data file: " << input_data_file_name << " for filling input data file. \n";

        throw runtime_error(buffer.str());
    }

    // Read first line

    if(has_raw_variables_name)
    {
        while(file.good())
        {
            getline(file, line);

            if(line.empty()) continue;

            break;
        }
    }

    // Read rest of the lines

    Tensor<string, 1> tokens;

    line_number = 0;
    Index variable_index = 0;
    Index token_index = 0;
    bool is_ID = has_rows_label;

    const bool is_float = is_same<type, float>::value;
    bool has_missing_values = false;

    while(file.good())
    {
        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_char);

        variable_index = 0;
        token_index = 0;
        is_ID = has_rows_label;

        for(Index i = 0; i < raw_variables.size(); i++)
        {
            if(is_ID)
            {
                is_ID = false;
                continue;
            }

            if(raw_variables(i).raw_variable_use == VariableUse::Unused)
            {
                token_index++;
                continue;
            }
            else if(raw_variables(i).raw_variable_use != VariableUse::Input)
            {
                continue;
            }

            if(raw_variables(i).type == RawVariableType::Numeric)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    inputs_data(line_number, variable_index) = type(NAN);
                }
                else if(is_float)
                {
                    inputs_data(line_number, variable_index) = type(strtof(tokens(token_index).data(), nullptr));
                }
                else
                {
                    inputs_data(line_number, variable_index) = type(stof(tokens(token_index)));
                }

                variable_index++;
            }
            else if(raw_variables(i).type == RawVariableType::Binary)
            {
                if(tokens(token_index) == missing_values_label)
                {
                    has_missing_values = true;
                    inputs_data(line_number, variable_index) = type(NAN);
                }
                else if(raw_variables(i).categories.size() > 0 && tokens(token_index) == raw_variables(i).categories(0))
                {
                    inputs_data(line_number, variable_index) = type(1);
                }
                else if(tokens(token_index) == raw_variables(i).name)
                {
                    inputs_data(line_number, variable_index) = type(1);
                }

                variable_index++;
            }
            else if(raw_variables(i).type == RawVariableType::Categorical)
            {
                for(Index k = 0; k < raw_variables(i).get_categories_number(); k++)
                {
                    if(tokens(token_index) == missing_values_label)
                    {
                        has_missing_values = true;
                        inputs_data(line_number, variable_index) = type(NAN);
                    }
                    else if(tokens(token_index) == raw_variables(i).categories(k))
                    {
                        inputs_data(line_number, variable_index) = type(1);
                    }

                    variable_index++;
                }
            }
            else if(raw_variables(i).type == RawVariableType::DateTime)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    inputs_data(line_number, variable_index) = type(NAN);
                }
                else
                {
                    inputs_data(line_number, variable_index) = type(date_to_timestamp(tokens(token_index), gmt));
                }

                variable_index++;
            }
            else if(raw_variables(i).type == RawVariableType::Constant)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    inputs_data(line_number, variable_index) = type(NAN);
                }
                else if(is_float)
                {
                    inputs_data(line_number, variable_index) = type(strtof(tokens(token_index).data(), nullptr));
                }
                else
                {
                    inputs_data(line_number, variable_index) = type(stof(tokens(token_index)));
                }

                variable_index++;
            }

            token_index++;
        }

        line_number++;
    }

    file.close();

    if(!has_missing_values)
    {
        return inputs_data;
    }
    else
    {
        // Scrub missing values

        const MissingValuesMethod missing_values_method = get_missing_values_method();

        if(missing_values_method == MissingValuesMethod::Unuse || missing_values_method == MissingValuesMethod::Mean)
        {
            const Tensor<type, 1> means = mean(inputs_data);

            const Index samples_number = inputs_data.dimension(0);
            const Index variables_number = inputs_data.dimension(1);

            #pragma omp parallel for schedule(dynamic)

            for(Index j = 0; j < variables_number; j++)
            {
                for(Index i = 0; i < samples_number; i++)
                {
                    if(isnan(inputs_data(i, j)))
                    {
                        inputs_data(i,j) = means(j);
                    }
                }
            }
        }
        else
        {
            const Tensor<type, 1> medians = median(inputs_data);

            const Index samples_number = inputs_data.dimension(0);
            const Index variables_number = inputs_data.dimension(1);

#pragma omp parallel for schedule(dynamic)

            for(Index j = 0; j < variables_number; j++)
            {
                for(Index i = 0; i < samples_number; i++)
                {
                    if(isnan(inputs_data(i, j)))
                    {
                        inputs_data(i,j) = medians(j);
                    }
                }
            }
        }

        return inputs_data;
    }
}

bool DataSet::get_augmentation() const { return augmentation; }
} // namespace opennn

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
