//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "data_set.h"
#include <omp.h>

using namespace  OpenNN;

namespace OpenNN
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


/// File and separator constructor. It creates a data set object by loading the object members from a data file.
/// It also sets a separator.
/// Please mind about the file format. This is specified in the User's Guide.
/// @param data_file_name Data file file name.
/// @param separator Data file file name.

DataSet::DataSet(const string& data_file_name, const char& separator, const bool& new_has_columns_names)
{
    set();

    set_default();

    set_data_file_name(data_file_name);

    set_separator(separator);

    set_has_columns_names(new_has_columns_names);

    read_csv();
}


/// Destructor.

DataSet::~DataSet()
{
    delete non_blocking_thread_pool;
    delete thread_pool_device;
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& DataSet::get_display() const
{
    return display;
}


/// Column default constructor

DataSet::Column::Column()
{
    name = "";
    column_use = Input;
    type = Numeric;
    categories.resize(0);
    categories_uses.resize(0);
}


/// Column default constructor

DataSet::Column::Column(const string& new_name,
                        const VariableUse& new_column_use,
                        const ColumnType& new_type,
                        const Tensor<string, 1>& new_categories,
                        const Tensor<VariableUse, 1>& new_categories_uses)
{
    name = new_name;
    column_use = new_column_use;
    type = new_type;
    categories = new_categories;
    categories_uses = new_categories_uses;
}

/// Column destructor.

DataSet::Column::~Column()
{}


/// Sets the use of the column and of the categories.
/// @param new_column_use New use of the column.

void DataSet::Column::set_use(const VariableUse& new_column_use)
{
    column_use = new_column_use;

    for(Index i = 0; i < categories_uses.size(); i ++)
    {
        categories_uses(i) = new_column_use;
    }
}


/// Sets the use of the column and of the categories.
/// @param new_column_use New use of the column in string format.

void DataSet::Column::set_use(const string& new_column_use)
{
    if(new_column_use == "Input")
    {
        set_use(Input);
    }
    else if(new_column_use == "Target")
    {
        set_use(Target);
    }
    else if(new_column_use == "Time")
    {
        set_use(Time);
    }
    else if(new_column_use == "Unused")
    {
        set_use(UnusedVariable);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception DataSet class.\n"
               << "void set_use(const string&) method.\n"
               << "Unknown use: " << new_column_use << "\n";

        throw logic_error(buffer.str());
    }
}


/// Sets the column type.
/// @param new_column_type Column type in string format.

void DataSet::Column::set_type(const string& new_column_type)
{
    if(new_column_type == "Numeric")
    {
        type = Numeric;
    }
    else if(new_column_type == "Binary")
    {
        type = Binary;
    }
    else if(new_column_type == "Categorical")
    {
        type = Categorical;
    }
    else if(new_column_type == "DateTime")
    {
        type = DateTime;
    }
    else if(new_column_type == "Constant")
    {
        type = Constant;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void Column::set_type(const string&) method.\n"
               << "Column type not valid (" << new_column_type << ").\n";

        throw logic_error(buffer.str());

    }
}


/// Adds a category to the categories vector of this column.
/// It also adds a default use for the category
/// @param new_category String that contains the name of the new category

void DataSet::Column::add_category(const string & new_category)
{
    const Index old_categories_number = categories.size();

    Tensor<string, 1> old_categories = categories;
    Tensor<VariableUse, 1> old_categories_uses = categories_uses;

    categories.resize(old_categories_number+1);
    categories_uses.resize(old_categories_number+1);

    for(Index category_index = 0; category_index < old_categories_number; category_index++)
    {
        categories(category_index) = old_categories(category_index);
        categories_uses(category_index) = column_use;
    }

    categories(old_categories_number) = new_category;
    categories_uses(old_categories_number) = column_use;
}


/// Sets the categories uses in the data set.
/// @param new_categories_uses String vector that contains the new categories of the data set.

void DataSet::Column::set_categories_uses(const Tensor<string, 1>& new_categories_uses)
{
    const Index new_categories_uses_number = new_categories_uses.size();

    categories_uses.resize(new_categories_uses_number);

    for(Index i = 0; i < new_categories_uses.size(); i++)
    {
        if(new_categories_uses(i) == "Input")
        {
            categories_uses(i) = Input;
        }
        else if(new_categories_uses(i) == "Target")
        {
            categories_uses(i) = Target;
        }
        else if(new_categories_uses(i) == "Time")
        {
            categories_uses(i) = Time;
        }
        else if(new_categories_uses(i) == "Unused"
                || new_categories_uses(i) == "UnusedVariable")
        {
            categories_uses(i) = UnusedVariable;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void Column::set_categories_uses(const Tensor<string, 1>&) method.\n"
                   << "Category use not valid (" << new_categories_uses(i) << ").\n";

            throw logic_error(buffer.str());

        }
    }
}


/// Sets the categories uses in the data set.
/// @param new_categories_use New categories use

void DataSet::Column::set_categories_uses(const VariableUse& new_categories_use)
{
    categories_uses.setConstant(new_categories_use);
}


void DataSet::Column::from_XML(const tinyxml2::XMLDocument& column_document)
{
    ostringstream buffer;

    // Name

    const tinyxml2::XMLElement* name_element = column_document.FirstChildElement("Name");

    if(!name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Name element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(name_element->GetText())
    {
        const string new_name = name_element->GetText();

        name = new_name;
    }

    // Column use

    const tinyxml2::XMLElement* column_use_element = column_document.FirstChildElement("ColumnUse");

    if(!column_use_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Column use element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(column_use_element->GetText())
    {
        const string new_column_use = column_use_element->GetText();

        set_use(new_column_use);
    }

    // Type

    const tinyxml2::XMLElement* type_element = column_document.FirstChildElement("Type");

    if(!type_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Type element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(type_element->GetText())
    {
        const string new_type = type_element->GetText();
        set_type(new_type);
    }

    if(type == Categorical)
    {
        // Categories

        const tinyxml2::XMLElement* categories_element = column_document.FirstChildElement("Categories");

        if(!categories_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Categories element is nullptr.\n";

            throw logic_error(buffer.str());
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
                   << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Categories uses element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(categories_uses_element->GetText())
        {
            const string new_categories_uses = categories_uses_element->GetText();

            set_categories_uses(get_tokens(new_categories_uses, ';'));
        }
    }
}


void DataSet::Column::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Name

    file_stream.OpenElement("Name");

    file_stream.PushText(name.c_str());

    file_stream.CloseElement();

    // Column use

    file_stream.OpenElement("ColumnUse");

    if(column_use == Input)
    {
        file_stream.PushText("Input");
    }
    else if (column_use == Target)
    {
        file_stream.PushText("Target");
    }
    else if (column_use == UnusedVariable)
    {
        file_stream.PushText("Unused");
    }
    else
    {
        file_stream.PushText("Time");
    }

    file_stream.CloseElement();

    // Type

    file_stream.OpenElement("Type");

    if(type == Numeric)
    {
        file_stream.PushText("Numeric");
    }
    else if (type == Binary)
    {
        file_stream.PushText("Binary");
    }
    else if (type == Categorical)
    {
        file_stream.PushText("Categorical");
    }
    else if(type == Constant)
    {
        file_stream.PushText("Constant");
    }
    else
    {
        file_stream.PushText("DateTime");
    }

    file_stream.CloseElement();

    if(type == Categorical || type == Binary)
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
            if(categories_uses(i) == Input)
            {
                file_stream.PushText("Input");
            }
            else if (categories_uses(i) == Target)
            {
                file_stream.PushText("Target");
            }
            else if (categories_uses(i) == Time)
            {
                file_stream.PushText("Time");
            }
            else
            {
                file_stream.PushText("Unused");
            }

            if(i != categories_uses.size()-1)
            {
                file_stream.PushText(";");
            }
        }

        file_stream.CloseElement();
    }
    /*else if(type == Binary)
    {
        if(categories.size() > 0)
        {
            // Categories

            file_stream.OpenElement("Categories");
            file_stream.PushText(categories(0).c_str());
            file_stream.PushText(";");
            file_stream.PushText(categories(1).c_str());
            file_stream.CloseElement();

            // Categories uses

            file_stream.OpenElement("CategoriesUses");

            if(categories_uses(0) == Input)
            {
                file_stream.PushText("Input");
            }
            else if (categories_uses(0) == Target)
            {
                file_stream.PushText("Target");
            }
            else if (categories_uses(0) == Time)
            {
                file_stream.PushText("Time");
            }
            else
            {
                file_stream.PushText("Unused");
            }

            file_stream.PushText(";");

            if(categories_uses(1) == Input)
            {
                file_stream.PushText("Input");
            }
            else if (categories_uses(1) == Target)
            {
                file_stream.PushText("Target");
            }
            else if (categories_uses(1) == Time)
            {
                file_stream.PushText("Time");
            }
            else
            {
                file_stream.PushText("Unused");
            }

            file_stream.CloseElement();
        }
    }*/
}


/// Returns the number of categories.

Index DataSet::Column::get_categories_number() const
{
    return categories.size();
}


/// Returns the number of used categories.

Index DataSet::Column::get_used_categories_number() const
{
    Index used_categories_number = 0;

    for(Index i = 0; i < categories.size(); i++)
    {
        if(categories_uses(i) != UnusedVariable) used_categories_number++;
    }

    return used_categories_number;
}


/// Returns a string vector that contains the names of the used variables in the data set.

Tensor<string, 1> DataSet::Column::get_used_variables_names() const
{
    Tensor<string, 1> used_variables_names;

    if(type != Categorical && column_use != UnusedVariable)
    {
        used_variables_names.resize(1);
        used_variables_names.setConstant(name);
    }
    else if(type == Categorical)
    {
        used_variables_names.resize(get_used_categories_number());

        Index category_index = 0;

        for(Index i = 0; i < categories.size(); i++)
        {
            if(categories_uses(i) != UnusedVariable)
            {
                used_variables_names(category_index) = categories(i);

                category_index++;
            }
        }
    }

    return used_variables_names;
}


/// This method transforms the columns into time series for forecasting problems.

void DataSet::transform_time_series_columns()
{
    const Index columns_number = get_columns_number();

    Tensor<Column, 1> new_columns;

    if(has_time_columns())
    {
        new_columns.resize((columns_number-1)*(lags_number+steps_ahead));
    }
    else
    {
        new_columns.resize(columns_number*(lags_number+steps_ahead));
    }

    Index lag_index = lags_number - 1;
    Index ahead_index = 0;
    Index column_index = 0;
    Index new_column_index = 0;

    for(Index i = 0; i < columns_number*(lags_number+steps_ahead); i++)
    {
        column_index = i%columns_number;

        if(time_series_columns(column_index).type == DateTime)
        {
            continue;
        }

        if(i < lags_number*columns_number)
        {
            new_columns(new_column_index).name = columns(column_index).name + "_lag_" + to_string(lag_index);
            new_columns(new_column_index).set_use(Input);

            new_columns(new_column_index).type = columns(column_index).type;
            new_columns(new_column_index).categories = columns(column_index).categories;
            new_columns(new_column_index).categories_uses = columns(column_index).categories_uses;

            new_column_index++;
        }
        else
        {
            new_columns(new_column_index).name = columns(column_index).name + "_ahead_" + to_string(ahead_index);
            new_columns(new_column_index).set_use(Target);

            new_columns(new_column_index).type = columns(column_index).type;
            new_columns(new_column_index).categories = columns(column_index).categories;
            new_columns(new_column_index).categories_uses = columns(column_index).categories_uses;

            new_column_index++;
        }

        if(lag_index > 0 && column_index == columns_number - 1)
        {
            lag_index--;
        }
        else if(column_index == columns_number - 1)
        {
            ahead_index++;
        }
    }

    columns = new_columns;
}


/// Returns true if a given sample is to be used for training, selection or testing,
/// and false if it is to be unused.
/// @param index Sample index.

bool DataSet::is_sample_used(const Index& index) const
{
    if(samples_uses(index) == UnusedSample)
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
    if(samples_uses(index) == UnusedSample)
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
        if(samples_uses(i) == Training)
        {
            count(0)++;
        }
        else if(samples_uses(i) == Selection)
        {
            count(1)++;
        }
        else if(samples_uses(i) == Testing)
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

    const type training_samples_percentage = training_samples_number*100/static_cast<type>(samples_number);
    const type selection_samples_percentage = selection_samples_number*100/static_cast<type>(samples_number);
    const type testing_samples_percentage = testing_samples_number*100/static_cast<type>(samples_number);
    const type unused_samples_percentage = unused_samples_number*100/static_cast<type>(samples_number);

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

    const Index columns_number = get_columns_number();

    Index variable_index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Numeric)
        {
            if(::isnan(data(sample_index, variable_index))) sample_string += missing_values_label;
            else sample_string += std::to_string(data(sample_index, variable_index));

            variable_index++;
        }
        else if(columns(i).type == Binary)
        {
            if(::isnan(data(sample_index, variable_index))) sample_string += missing_values_label;
            else sample_string += columns(i).categories(static_cast<Index>(data(sample_index, variable_index)));

            variable_index++;
        }
        else if(columns(i).type == DateTime)
        {
            // @todo do something

            if(::isnan(data(sample_index, variable_index))) sample_string += missing_values_label;
            else sample_string += std::to_string(data(sample_index, variable_index));

            variable_index++;
        }
        else if(columns(i).type == Categorical)
        {
            if(::isnan(data(sample_index, variable_index)))
            {
                sample_string += missing_values_label;
            }
            else
            {
                const Index categories_number = columns(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                {
                    if(abs(data(sample_index, variable_index+j) - static_cast<type>(1)) < std::numeric_limits<type>::min())
                    {
                        sample_string += columns(i).categories(j);
                        break;
                    }
                }

                variable_index += categories_number;
            }
        }

        if(i != columns_number-1) sample_string += separator + " ";
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
        if(samples_uses(i) == Training)
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
        if(samples_uses(i) == Selection)
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
        if(samples_uses(i) == Testing)
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
        if(samples_uses(i) != UnusedSample)
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
        if(samples_uses(i) == UnusedSample)
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
/// @todo In forecasting must be false.

Tensor<Index, 2> DataSet::get_batches(const Tensor<Index,1>& samples_indices,
                                      const Index& batch_samples_number,
                                      const bool& shuffle,
                                      const Index& new_buffer_size) const
{
    if(!shuffle) return split_samples(samples_indices, batch_samples_number);

    const Index samples_number = samples_indices.size();

    Index buffer_size = new_buffer_size;
    Index batches_number;
    Index batch_size = batch_samples_number;

    // Check batch size and samples number

    if(samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
        buffer_size = batch_size;

        Tensor<Index,1> samples_copy(samples_indices);

        Tensor<Index, 2> batches(batches_number, batch_size);

        // Shuffle

        random_shuffle(samples_copy.data(), samples_copy.data() + samples_copy.size());

        for(Index i = 0; i < batch_size; i++)
            batches(0,i) = samples_copy(i);

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
                random_index = static_cast<Index>(rand()%buffer_size);

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
                random_shuffle(buffer.data(), buffer.data() +  buffer.size());

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
                random_index = static_cast<Index>(rand()%buffer_size);

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
        if(samples_uses(i) == Training)
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
        if(samples_uses(i) == Selection)
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
        if(samples_uses(i) == Testing)
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
        if(samples_uses(i) == UnusedSample)
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
        samples_uses(i) = Training;
    }
}


/// Sets all the samples in the data set for selection.

void DataSet::set_selection()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = Selection;
    }
}


/// Sets all the samples in the data set for testing.

void DataSet::set_testing()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = Testing;
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

        samples_uses(index) = Training;
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

        samples_uses(index) = Selection;
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

        samples_uses(index) = Testing;
    }
}


/// Sets all the samples in the data set for unused.

void DataSet::set_samples_unused()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        samples_uses(i) = UnusedSample;
    }
}


/// Sets samples with given indices in the data set for unused.
/// @param indices Indices vector with the index of samples in the data set for unused.

void DataSet::set_samples_unused(const Tensor<Index, 1>& indices)
{
    for(Index i = 0; i < static_cast<Index>(indices.size()); i++)
    {
        const Index index = indices(i);

        samples_uses(index) = UnusedSample;
    }
}


/// Sets the use of a single sample.
/// @param index Index of sample.
/// @param new_use Use for that sample.

void DataSet::set_sample_use(const Index& index, const SampleUse& new_use)
{
    samples_uses(index) = new_use;

}


/// Sets the use of a single sample from a string.
/// @param index Index of sample.
/// @param new_use String with the use name("Training", "Selection", "Testing" or "Unused")

void DataSet::set_sample_use(const Index& index, const string& new_use)
{
    if(new_use == "Training")
    {
        samples_uses(index) = Training;
    }
    else if(new_use == "Selection")
    {
        samples_uses(index) = Selection;
    }
    else if(new_use == "Testing")
    {
        samples_uses(index) = Testing;
    }
    else if(new_use == "Unused")
    {
        samples_uses(index) = UnusedSample;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception DataSet class.\n"
               << "void set_sample_use(const string&) method.\n"
               << "Unknown use: " << new_use << "\n";

        throw logic_error(buffer.str());
    }
}


/// Sets new uses to all the samples from a single vector.
/// @param new_uses vector of use structures.
/// The size of given vector must be equal to the number of samples.

void DataSet::set_samples_uses(const Tensor<SampleUse, 1>& new_uses)
{
    const Index samples_number = get_samples_number();

#ifdef __OPENNN_DEBUG__

    const Index new_uses_size = new_uses.size();

    if(new_uses_size != samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_samples_uses(const Tensor<SampleUse, 1>&) method.\n"
               << "Size of uses(" << new_uses_size << ") must be equal to number of samples(" << samples_number << ").\n";

        throw logic_error(buffer.str());
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

#ifdef __OPENNN_DEBUG__

    const Index new_uses_size = new_uses.size();

    if(new_uses_size != samples_number)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_samples_uses(const Tensor<string, 1>&) method.\n"
               << "Size of uses(" << new_uses_size << ") must be equal to number of samples(" << samples_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    for(Index i = 0; i < samples_number; i++)
    {
        if(new_uses(i).compare("Training") == 0 || new_uses(i).compare("0") == 0)
        {
            samples_uses(i) = Training;
        }
        else if(new_uses(i).compare("Selection") == 0 || new_uses(i).compare("1") == 0)
        {
            samples_uses(i) = Selection;
        }
        else if(new_uses(i).compare("Testing") == 0 || new_uses(i).compare("2") == 0)
        {
            samples_uses(i) = Testing;
        }
        else if(new_uses(i).compare("Unused") == 0 || new_uses(i).compare("3") == 0)
        {
            samples_uses(i) = UnusedSample;
        }
        else
        {
            buffer << "OpenNN Exception DataSet class.\n"
                   << "void set_samples_uses(const Tensor<string, 1>&) method.\n"
                   << "Unknown use: " << new_uses(i) << ".\n";

            throw logic_error(buffer.str());
        }
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

    const Index used_samples_number = get_used_samples_number();

    if(used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    // Get number of samples for training, selection and testing

    const Index selection_samples_number = static_cast<Index>(selection_samples_ratio*used_samples_number/total_ratio);
    const Index testing_samples_number = static_cast<Index>(testing_samples_ratio*used_samples_number/total_ratio);
    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Warning: DataSet class.\n"
               << "void split_samples_random(const type&, const type&, const type&) method.\n"
               << "Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n";

        throw logic_error(buffer.str());
    }

    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;

    initialize_sequential_eigen_tensor(indices, 0, 1, samples_number-1);

    random_shuffle(indices.data(), indices.data() + indices.size());

    Index count = 0;

    for(Index i = 0; i < samples_uses.size(); i++)
    {
        if(samples_uses(i) == UnusedSample) count ++;
    }

    Index i = 0;
    Index index;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != UnusedSample)
        {
            samples_uses(index)= Training;
            count_training++;
        }

        i++;
    }

    // Selection

    Index count_selection = 0;

    while(count_selection != selection_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != UnusedSample)
        {
            samples_uses(index) = Selection;
            count_selection++;
        }

        i++;
    }

    // Testing


    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != UnusedSample)
        {
            samples_uses(index) = Testing;
            count_testing++;
        }

        i++;
    }

    for(Index i = 0; i < samples_uses.size(); i++)
    {
        if(samples_uses(i) == UnusedSample)
        {
            cout << "Sample " << i << " is unused" << endl;
        }
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

    const Index selection_samples_number = static_cast<Index>(selection_samples_ratio*used_samples_number/total_ratio);
    const Index testing_samples_number = static_cast<Index>(testing_samples_ratio*used_samples_number/total_ratio);
    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Warning: Samples class.\n"
               << "void split_samples_sequential(const type&, const type&, const type&) method.\n"
               << "Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n";

        throw logic_error(buffer.str());
    }

    Index i = 0;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        if(samples_uses(i) != UnusedSample)
        {
            samples_uses(i) = Training;
            count_training++;
        }

        i++;
    }

    // Selection

    Index count_selection = 0;

    while(count_selection != selection_samples_number)
    {
        if(samples_uses(i) != UnusedSample)
        {
            samples_uses(i) = Selection;
            count_selection++;
        }

        i++;
    }

    // Testing

    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        if(samples_uses(i) != UnusedSample)
        {
            samples_uses(i) = Testing;
            count_testing++;
        }
        i++;
    }
}


/// This method separates the dataset into n-groups to validate a model with limited data.
/// @param k Number of folds that a given data sample is given to be split into.
/// @param fold_index.
/// @todo Low priority

void DataSet::set_k_fold_cross_validation_samples_uses(const Index& k, const Index& fold_index)
{
    const Index samples_number = get_samples_number();

    const Index fold_size = samples_number/k;

    const Index start = fold_index*fold_size;
    const Index end = start + fold_size;

    split_samples_random(1, 0, 0);

    for(Index i = start; i < end; i++)
    {
        samples_uses(i) = Testing;
    }
}


/// This method sets the n columns of the dataset by default,
/// i.e. until column n-1 are Input and column n is Target.

void DataSet::set_default_columns_uses()
{
    const Index size = columns.size();

    if(size == 0)
    {
        return;
    }
    else if(size == 1)
    {
        columns(0).set_use(UnusedVariable);
    }
    else
    {
        set_input();

        for(Index i = columns.size()-1; i >= 0; i--)
        {
            if(columns(i).type == Constant) continue;
            if(columns(i).type == Binary) continue;
            if(columns(i).type == Categorical) continue;

            columns(i).set_use(Target);
            break;
        }

        input_variables_dimensions.resize(1);
    }
}


/// This method sets the n columns of the dataset by default,
/// i.e. until column n-1 are Input and column n is Target.

void DataSet::set_default_classification_columns_uses()
{
    const Index size = columns.size();

    if(size == 0)
    {
        return;
    }
    else if(size == 1)
    {
        columns(0).set_use(UnusedVariable);
    }
    else
    {
        set_input();

        for(Index i = columns.size()-1; i >= 0; i--)
        {
            if(columns(i).type == Constant) continue;

            if(columns(i).type == Binary)
            {
                columns(i).set_use(Target);
                break;
            }
            else if(columns(i).type == Categorical)
            {
                columns(i).set_use(Target);
                break;
            }
        }

        input_variables_dimensions.resize(1);
    }
}


/// This method puts the names of the columns in the dataset.
/// This is used when the dataset does not have a header,
/// the default names are: column_0, column_1, ..., column_n.

void DataSet::set_default_columns_names()
{
    const Index size = columns.size();

    if(size == 0)
    {
        return;
    }
    else if(size == 1)
    {
        return;
    }
    else
    {
        Index input_index = 1;
        Index target_index = 2;

        for(Index i = 0; i < size; i++)
        {
            if(columns(i).column_use == Input)
            {
                columns(i).name = "input_" + std::to_string(input_index+1);
                input_index++;
            }
            else if(columns(i).column_use == Target)
            {
                columns(i).name = "target_" + std::to_string(target_index+1);
                target_index++;
            }
        }
    }
}


/// Sets the name of a single column.
/// @param index Index of column.
/// @param new_use Use for that column.

void DataSet::set_column_name(const Index& column_index, const string& new_name)
{
    columns(column_index).name = new_name;
}


/// Returns the use of a single variable.
/// @param index Index of variable.

DataSet::VariableUse DataSet::get_variable_use(const Index& index) const
{
    return get_variables_uses()(index);
}


/// Returns a vector containing the use of the column, without taking into account the categories.

DataSet::VariableUse DataSet::get_column_use(const Index & index) const
{
    return columns(index).column_use;
}


/// Returns the uses of each columns of the data set.

Tensor<DataSet::VariableUse, 1> DataSet::get_columns_uses() const
{
    const Index columns_number = get_columns_number();

    Tensor<DataSet::VariableUse, 1> columns_uses(columns_number);

    for (Index i = 0; i < columns_number; i++)
    {
        columns_uses(i) = columns(i).column_use;
    }

    return columns_uses;
}


/// Returns a vector containing the use of each column, including the categories.
/// The size of the vector is equal to the number of variables.

Tensor<DataSet::VariableUse, 1> DataSet::get_variables_uses() const
{
    const Index columns_number = get_columns_number();
    const Index variables_number = get_variables_number();

    Tensor<VariableUse, 1> variables_uses(variables_number);

    Index index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index i = 0; i < (columns(i).categories_uses).size(); i++)
            {
                variables_uses(i + index) = (columns(i).categories_uses)(i);
            }
            index += columns(i).categories.size();
        }
        else
        {
            variables_uses(index) = columns(i).column_use;
            index++;
        }
    }

    return variables_uses;
}


/// Returns the name of a single variable in the data set.
/// @param index Index of variable.

string DataSet::get_variable_name(const Index& variable_index) const
{
#ifdef __OPENNN_DEBUG__

    const Index variables_number = get_variables_number();

    if(variable_index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "string& get_variable_name(const Index) method.\n"
               << "Index of variable("<<variable_index<<") must be less than number of variables("<<variables_number<<").\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index columns_number = get_columns_number();

    Index index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index j = 0; j < columns(i).get_categories_number(); j++)
            {
                if(index == variable_index)
                {
                    return columns(i).categories(j);
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
                return columns(i).name;
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

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index j = 0; j < columns(i).categories.size(); j++)
            {
                variables_names(index) = columns(i).categories(j);

                index++;
            }
        }
        else
        {
            variables_names(index) = columns(i).name;
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

    const Tensor<Index, 1> input_columns_indices = get_input_columns_indices();

    Tensor<string, 1> input_variables_names(input_variables_number);

    Index index = 0;

    for(Index i = 0; i < input_columns_indices.size(); i++)
    {
        Index input_index = input_columns_indices(i);

        const Tensor<string, 1> current_used_variables_names = columns(input_index).get_used_variables_names();

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

    const Tensor<Index, 1> target_columns_indices = get_target_columns_indices();

    Tensor<string, 1> target_variables_names(target_variables_number);

    Index index = 0;

    for(Index i = 0; i < target_columns_indices.size(); i++)
    {
        Index target_index = target_columns_indices(i);

        const Tensor<string, 1> current_used_variables_names = columns(target_index).get_used_variables_names();

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


/// Returns the number of variables which are either input nor target.

Index DataSet::get_used_variables_number() const
{
    const Index variables_number = get_variables_number();

    const Index unused_variables_number = get_unused_variables_number();

    return (variables_number - unused_variables_number);
}


/// Returns a indices vector with the positions of the inputs.

Tensor<Index, 1> DataSet::get_input_columns_indices() const
{
    const Index input_columns_number = get_input_columns_number();

    Tensor<Index, 1> input_columns_indices(input_columns_number);

    Index index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Input)
        {
            input_columns_indices(index) = i;
            index++;
        }
    }

    return input_columns_indices;
}


/// Returns a indices vector with the positions of the targets.

Tensor<Index, 1> DataSet::get_target_columns_indices() const
{
    const Index target_columns_number = get_target_columns_number();

    Tensor<Index, 1> target_columns_indices(target_columns_number);

    Index index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Target)
        {
            target_columns_indices(index) = i;
            index++;
        }
    }

    return target_columns_indices;
}


/// Returns a indices vector with the positions of the unused columns.

Tensor<Index, 1> DataSet::get_unused_columns_indices() const
{
    const Index unused_columns_number = get_unused_columns_number();

    Tensor<Index, 1> unused_columns_indices(unused_columns_number);

    Index index = 0;

    for(Index i = 0; i < unused_columns_number; i++)
    {

        if(columns(i).column_use == UnusedVariable)
        {
            unused_columns_indices(index) = i;
            index++;
        }
    }

    return unused_columns_indices;
}


/// Returns a indices vector with the positions of the used columns.

Tensor<Index, 1> DataSet::get_used_columns_indices() const
{
    const Index variables_number = get_variables_number();

    const Index used_variables_number = get_used_variables_number();

    Tensor<Index, 1> used_indices(used_variables_number);

    Index index = 0;

    for(Index i = 0; i < variables_number; i++)
    {
        if(columns(i).column_use  == Input
                || columns(i).column_use  == Target
                || columns(i).column_use  == Time)
        {
            used_indices(index) = i;
            index++;
        }
    }

    return used_indices;
}


/// Returns a string vector that contains the names of the columns.

Tensor<string, 1> DataSet::get_columns_names() const
{
    const Index columns_number = get_columns_number();

    Tensor<string, 1> columns_names(columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        columns_names(i) = columns(i).name;
    }

    return columns_names;
}


Tensor<string, 1> DataSet::get_time_series_columns_names() const
{
    const Index columns_number = get_time_series_columns_number();

    Tensor<string, 1> columns_names(columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        columns_names(i) = time_series_columns(i).name;
    }

    return columns_names;
}

/// Returns a string vector that contains the names of the columns whose uses are Input.

Tensor<string, 1> DataSet::get_input_columns_names() const
{
    const Index input_columns_number = get_input_columns_number();

    Tensor<string, 1> input_columns_names(input_columns_number);

    Index index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Input)
        {
            input_columns_names(index) = columns(i).name;
            index++;
        }
    }

    return input_columns_names;
}


/// Returns a string vector which contains the names of the columns whose uses are Target.

Tensor<string, 1> DataSet::get_target_columns_names() const
{
    const Index target_columns_number = get_target_columns_number();

    Tensor<string, 1> target_columns_names(target_columns_number);

    Index index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Target)
        {
            target_columns_names(index) = columns(i).name;
            index++;
        }
    }

    return target_columns_names;

}


/// Returns a string vector which contains the names of the columns used whether Input, Target or Time.

Tensor<string, 1> DataSet::get_used_columns_names() const
{
    const Index columns_number = get_columns_number();
    const Index used_columns_number = get_used_columns_number();

    Tensor<string, 1> names(used_columns_number);

    Index index = 0 ;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).column_use != UnusedVariable)
        {
            names(index) = columns(i).name;
            index++;
        }
    }

    return names;
}


/// Returns the number of columns whose uses are Input.

Index DataSet::get_input_columns_number() const
{
    Index input_columns_number = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Input)
        {
            input_columns_number++;
        }
    }

    return input_columns_number;
}


/// Returns the number of columns whose uses are Target.

Index DataSet::get_target_columns_number() const
{
    Index target_columns_number = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Target)
        {
            target_columns_number++;
        }
    }

    return target_columns_number;
}


/// Returns the number of columns whose uses are Time

Index DataSet::get_time_columns_number() const
{
    Index time_columns_number = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Time)
        {
            time_columns_number++;
        }
    }

    return time_columns_number;
}


/// Returns the number of columns that are not used.

Index DataSet::get_unused_columns_number() const
{
    Index unused_columns_number = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == UnusedVariable)
        {
            unused_columns_number++;
        }
    }

    return unused_columns_number;
}


/// Returns the number of columns that are used.

Index DataSet::get_used_columns_number() const
{
    Index used_columns_number = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use != UnusedVariable)
        {
            used_columns_number++;
        }
    }

    return used_columns_number;
}


/// Returns the columns of the data set.

Tensor<DataSet::Column, 1> DataSet::get_columns() const
{
    return columns;
}

Tensor<DataSet::Column, 1> DataSet::get_time_series_columns() const
{
    return time_series_columns;
}

/// Returns the input columns of the data set.

Tensor<DataSet::Column, 1> DataSet::get_input_columns() const
{
    const Index inputs_number = get_input_columns_number();

    Tensor<Column, 1> input_columns(inputs_number);
    Index input_index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Input)
        {
            input_columns(input_index) = columns(i);
            input_index++;
        }
    }

    return input_columns;
}


/// Returns the target columns of the data set.

Tensor<DataSet::Column, 1> DataSet::get_target_columns() const
{
    const Index targets_number = get_target_columns_number();

    Tensor<Column, 1> target_columns(targets_number);
    Index target_index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).column_use == Target)
        {
            target_columns(target_index) = columns(i);
            target_index++;
        }
    }

    return target_columns;
}


/// Returns the used columns of the data set.
/// @todo

Tensor<DataSet::Column, 1> DataSet::get_used_columns() const
{
    const Tensor<Index, 1> used_columns_indices = get_used_columns_indices();

//    return columns.get_subvector(used_columns_indices);

    return Tensor<DataSet::Column, 1>();
}


/// Returns the number of columns in the data set.

Index DataSet::get_columns_number() const
{
    return columns.size();
}

/// Returns the number of columns in the time series.

Index DataSet::get_time_series_columns_number() const
{
    return time_series_columns.size();
}

/// Returns the number of variables in the data set.

Index DataSet::get_variables_number() const
{
    Index variables_number = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            variables_number += columns(i).categories.size();
        }
        else
        {
            variables_number++;
        }
    }

    return variables_number;
}


/// Returns the number of input variables of the data set.
/// Note that the number of variables does not have to equal the number of columns in the data set,
/// because OpenNN recognizes the categorical columns, separating these categories into variables of the data set.

Index DataSet::get_input_variables_number() const
{
    Index inputs_number = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index j = 0; j < columns(i).categories_uses.size(); j++)
            {
                if(columns(i).categories_uses(j) == Input) inputs_number++;
            }
        }
        else if(columns(i).column_use == Input)
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

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index j = 0; j < columns(i).categories_uses.size(); j++)
            {
                if(columns(i).categories_uses(j) == Target) targets_number++;
            }

        }
        else if(columns(i).column_use == Target)
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

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index j = 0; j < columns(i).categories_uses.size(); j++)
            {
                if(columns(i).categories_uses(j) == UnusedVariable) unused_number++;
            }

        }
        else if(columns(i).column_use == UnusedVariable)
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

//    throw exception("Exception: Index DataSet::get_variable_index(const string& name) const");
}


/// Returns the indices of the unused variables.

Tensor<Index, 1> DataSet::get_unused_variables_indices() const
{
    const Index unused_number = get_unused_variables_number();

    const Tensor<Index, 1> unused_columns_indices = get_unused_columns_indices();

    Tensor<Index, 1> unused_indices(unused_number);

    Index unused_index = 0;
    Index unused_variable_index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            const Index current_categories_number = columns(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(columns(i).categories_uses(j) == UnusedVariable)
                {
                    unused_indices(unused_index) = unused_variable_index;
                    unused_index++;
                }

                unused_variable_index++;
            }
        }
        else if(columns(i).column_use == UnusedVariable)
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

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            const Index current_categories_number = columns(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(columns(i).categories_uses(j) != UnusedVariable)
                {
                    used_indices(used_index) = used_variable_index;
                    used_index++;
                }

                used_variable_index++;
            }
        }
        else if(columns(i).column_use != UnusedVariable)
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

    const Tensor<Index, 1> input_columns_indices = get_input_columns_indices();

    Tensor<Index, 1> input_variables_indices(inputs_number);

    Index input_index = 0;
    Index input_variable_index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {

        if(columns(i).type == Categorical)
        {
            const Index current_categories_number = columns(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(columns(i).categories_uses(j) == Input)
                {
                    input_variables_indices(input_index) = input_variable_index;
                    input_index++;
                }

                input_variable_index++;
            }
        }
        else if(columns(i).column_use == Input) // Binary, numeric
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


/// Returns the indices of the target variables.

Tensor<Index, 1> DataSet::get_target_variables_indices() const
{
    const Index targets_number = get_target_variables_number();

    const Tensor<Index, 1> target_columns_indices = get_target_columns_indices();

    Tensor<Index, 1> target_variables_indices(targets_number);

    Index target_index = 0;
    Index target_variable_index = 0;

    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Categorical)
        {
            const Index current_categories_number = columns(i).get_categories_number();

            for(Index j = 0; j < current_categories_number; j++)
            {
                if(columns(i).categories_uses(j) == Target)
                {
                    target_variables_indices(target_index) = target_variable_index;
                    target_index++;
                }

                target_variable_index++;
            }
        }
        else if(columns(i).column_use == Target) // Binary, numeric
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


/// Sets the uses of the data set columns.
/// @param new_columns_uses String vector that contains the new uses to be set,
/// note that this vector needs to be the size of the number of columns in the data set.

void DataSet::set_columns_uses(const Tensor<string, 1>& new_columns_uses)
{
    const Index new_columns_uses_size = new_columns_uses.size();

    if(new_columns_uses_size != columns.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception DataSet class.\n"
               << "void set_columns_uses(const Tensor<string, 1>&) method.\n"
               << "Size of columns uses ("
               << new_columns_uses_size << ") must be equal to columns size ("
               << columns.size() << "). \n";

        throw logic_error(buffer.str());
    }

    for(Index i = 0; i < new_columns_uses.size(); i++)
    {
        columns(i).set_use(new_columns_uses(i));
    }

    input_variables_dimensions.resize(1);
    input_variables_dimensions.setConstant(get_input_variables_number());
}


/// Sets the uses of the data set columns.
/// @param new_columns_uses DataSet::VariableUse vector that contains the new uses to be set,
/// note that this vector needs to be the size of the number of columns in the data set.

void DataSet::set_columns_uses(const Tensor<VariableUse, 1>& new_columns_uses)
{
    const Index new_columns_uses_size = new_columns_uses.size();

    if(new_columns_uses_size != columns.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception DataSet class.\n"
               << "void set_columns_uses(const Tensor<string, 1>&) method.\n"
               << "Size of columns uses (" << new_columns_uses_size << ") must be equal to columns size (" << columns.size() << "). \n";

        throw logic_error(buffer.str());
    }

    for(Index i = 0; i < new_columns_uses.size(); i++)
    {
        columns(i).set_use(new_columns_uses(i));
    }

    input_variables_dimensions.resize(1);
    input_variables_dimensions.setConstant(get_input_variables_number());
}


/// Sets all columns in the dataset as unused columns.

void DataSet::set_columns_unused()
{
    const Index columns_number = get_columns_number();

    for(Index i = 0; i < columns_number; i++)
    {
        set_column_use(i, UnusedVariable);
    }
}


/// Sets all input columns in the dataset as unused columns.

void DataSet::set_input_columns_unused()
{
    const Index columns_number = get_columns_number();

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).column_use == DataSet::Input) set_column_use(i, UnusedVariable);
    }
}


/// Sets the use of a single column.
/// @param index Index of column.
/// @param new_use Use for that column.

void DataSet::set_column_use(const Index& index, const VariableUse& new_use)
{
    columns(index).column_use = new_use;

    if(columns(index).type == Categorical)
    {
        columns(index).set_categories_uses(new_use);
    }
}


/// Sets the use of a single column.
/// @param name Name of column.
/// @param new_use Use for that column.

void DataSet::set_column_use(const string& name, const VariableUse& new_use)
{
    const Index index = get_column_index(name);

    set_column_use(index, new_use);
}


/// This method set the name of a single variable.
/// @param index Index of variable.
/// @param new_name Name of variable.

void DataSet::set_variable_name(const Index& variable_index, const string& new_variable_name)
{
#ifdef __OPENNN_DEBUG__

    const Index variables_number = get_variables_number();

    if(variable_index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Variables class.\n"
               << "void set_name(const Index&, const string&) method.\n"
               << "Index of variable must be less than number of variables.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index columns_number = get_columns_number();

    Index index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index j = 0; j < columns(i).get_categories_number(); j++)
            {
                if(index == variable_index)
                {
                    columns(i).categories(j) = new_variable_name;
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
                columns(i).name = new_variable_name;
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
#ifdef __OPENNN_DEBUG__

    const Index variables_number = get_variables_number();

    const Index size = new_variables_names.size();

    if(size != variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Variables class.\n"
               << "void set_names(const Tensor<string, 1>&) method.\n"
               << "Size (" << size << ") must be equal to number of variables (" << variables_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index columns_number = get_columns_number();

    Index index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Categorical)
        {
            for(Index j = 0; j < columns(i).get_categories_number(); j++)
            {
                columns(i).categories(j) = new_variables_names(index);
                index++;
            }
        }
        else
        {
            columns(i).name = new_variables_names(index);
            index++;
        }
    }
}


/// Sets new names for the columns in the data set from a vector of strings.
/// The size of that vector must be equal to the total number of variables.
/// @param new_names Name of variables.

void DataSet::set_columns_names(const Tensor<string, 1>& new_names)
{
    const Index new_names_size = new_names.size();
    const Index columns_number = get_columns_number();

    if(new_names_size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_columns_names(const Tensor<string, 1>&).\n"
               << "Size of names (" << new_names.size() << ") is not equal to columns number (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

    for(Index i = 0; i < columns_number; i++)
    {
        columns(i).name = new_names(i);
    }
}


/// Sets all the variables in the data set as input variables.

void DataSet::set_input()
{
    for(Index i = 0; i < columns.size(); i++)
    {
        if(columns(i).type == Constant) continue;

        columns(i).set_use(Input);
    }
}


/// Sets all the variables in the data set as target variables.

void DataSet::set_target()
{
    for(Index i = 0; i < columns.size(); i++)
    {
        columns(i).set_use(Target);
    }
}


/// Sets all the variables in the data set as unused variables.

void DataSet::set_variables_unused()
{
    for(Index i = 0; i < columns.size(); i++)
    {
        columns(i).set_use(UnusedVariable);
    }
}


/// Sets a new number of variables in the variables object.
/// All variables are set as inputs but the last one, which is set as targets.
/// @param new_variables_number Number of variables.

void DataSet::set_columns_number(const Index& new_variables_number)
{
    columns.resize(new_variables_number);

    set_default_columns_uses();
}


void DataSet::binarize_input_data(const type& threshold)
{
    const Index samples_number = get_samples_number();

    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < input_variables_number; i++)
        {
            const Index input_variable_index = input_variables_indices[j];

            data(i,input_variable_index) < threshold
                    ? data(i,input_variable_index) = 0
                    : data(i,input_variable_index) = 1;
        }
    }
}


Tensor<type,2> DataSet::transform_binary_column(const Tensor<type,1>& column) const
{
    const Index rows_number = column.dimension(0);

    Tensor<type, 2> new_column(rows_number , 2);
    new_column.setZero();

    for(Index i = 0; i < rows_number; i++)
    {
        if(abs(column(i) - static_cast<type>(1)) < std::numeric_limits<type>::min())
        {
            new_column(i,1) = static_cast<type>(1);
        }
        else if(abs(column(i) - static_cast<type>(0)) < std::numeric_limits<type>::min())
        {
            new_column(i,0) = static_cast<type>(1);
        }
        else
        {
            new_column(i,0) = NAN;
            new_column(i,1) = NAN;
        }
    }

    return new_column;
}


void DataSet::set_binary_simple_columns()
{
    bool is_binary = true;

    Index variable_index = 0;

    Index different_values = 0;

    for(Index column_index = 0; column_index < columns.size(); column_index++)
    {
        if(columns(column_index).type == Numeric)
        {
            Tensor<type, 1> values(3);
            values.setRandom();
            different_values = 0;
            is_binary = true;

            for(Index row_index = 0; row_index < data.dimension(0); row_index++)
            {
                if(!::isnan(data(row_index, variable_index))
                && data(row_index, variable_index) != values(0)
                && data(row_index, variable_index) != values(1))
                {
                    values(different_values) = data(row_index, variable_index);

                    different_values++;
                }

                if(row_index == (data.dimension(0)-1)){
                    if(different_values==1){
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
                columns(column_index).type = Binary;
                scale_minimum_maximum_binary(values(0), values(1), column_index);
                columns(column_index).categories.resize(2);

                if(values(0) == 0 && values(1) == 1)
                {
                    columns(column_index).categories(0) = "Negative (0)";
                    columns(column_index).categories(1) = "Positive (1)";
                }
                else if(values(0) == 1 && values(1) == 0)
                {
                    columns(column_index).categories(0) = "Positive (1)";
                    columns(column_index).categories(1) = "Negative (0)";
                }
                else
                {
                    columns(column_index).categories(0) = "Class_1";// + std::to_string(values(0));
                    columns(column_index).categories(1) = "Class_2";// + std::to_string(values(1));
                }

                const VariableUse column_use = columns(column_index).column_use;
                columns(column_index).categories_uses.resize(2);
                columns(column_index).categories_uses(0) = column_use;
                columns(column_index).categories_uses(1) = column_use;
            }

            variable_index++;
        }
        else if(columns(column_index).type == Categorical)
        {
            variable_index += columns(column_index).get_categories_number();
        }
        else
        {
            variable_index++;
        }
    }
}


/// Sets new input dimensions in the data set.

void DataSet::set_input_variables_dimensions(const Tensor<Index, 1>& new_inputs_dimensions)
{
    input_variables_dimensions = new_inputs_dimensions;
}


/// Returns true if the data set is a binary classification problem, false otherwise.
/// @todo

bool DataSet::is_binary_classification() const
{
    if(get_target_variables_number() != 1)
    {
        return false;
    }

    return true;
}


/// Returns true if the data set is a multiple classification problem, false otherwise.
/// @todo

bool DataSet::is_multiple_classification() const
{
    return true;
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


/// Returns true if any value is less or equal than a given value, and false otherwise.

bool DataSet::is_less_than(const Tensor<type, 1>& column, const type& value) const
{
    Tensor<bool, 1> if_sentence = column <= column.constant(value);

    Tensor<bool, 1> sentence(column.size());
    sentence.setConstant(true);

    Tensor<bool, 1> else_sentence(column.size());
    else_sentence.setConstant(false);

    Tensor<bool, 0> is_less = (if_sentence.select(sentence, else_sentence)).any();

    return is_less(0);
}


/// Returns a reference to the data matrix in the data set.
/// The number of rows is equal to the number of samples.
/// The number of columns is equal to the number of variables.

const Tensor<type, 2>& DataSet::get_data() const
{
    return data;
}


Tensor<type, 2>* DataSet::get_data_pointer()
{
    return &data;
}


/// Returns a reference to the time series data matrix in the data set.
/// Only for time series problems.

const Tensor<type, 2>& DataSet::get_time_series_data() const
{
    return time_series_data;
}


/// Returns a string with the method used.

DataSet::MissingValuesMethod DataSet::get_missing_values_method() const
{
    return missing_values_method;
}


/// Returns the name of the data file.

const string& DataSet::get_data_file_name() const
{
    return data_file_name;
}


/// Returns true if the first line of the data file has a header with the names of the variables, and false otherwise.

const bool& DataSet::get_header_line() const
{
    return has_columns_names;
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
    case Space:
        return ' ';

    case Tab:
        return '\t';

    case Comma:
        return ',';

    case Semicolon:
        return ';';
    }

    return char();
}


/// Returns the string which will be used as separator in the data file.

string DataSet::get_separator_string() const
{
    switch(separator)
    {
    case Space:
        return "Space";

    case Tab:
        return "Tab";

    case Comma:
        return "Comma";

    case Semicolon:
        return "Semicolon";
    }

    return string();
}


/// Returns the string which will be used as label for the missing values in the data file.

const string& DataSet::get_missing_values_label() const
{
    return missing_values_label;
}


/// Returns the number of lags to be used in a time series prediction application.

const Index& DataSet::get_lags_number() const
{
    return lags_number;
}


/// Returns the number of steps ahead to be used in a time series prediction application.

const Index& DataSet::get_steps_ahead() const
{
    return steps_ahead;
}


/// Returns the indices of the time variables in the data set.

const Index& DataSet::get_time_index() const
{
    return time_index;
}


/// Returns a value of the scaling-unscaling method enumeration from a string containing the name of that method.
/// @param scaling_unscaling_method String with the name of the scaling and unscaling method.

DataSet::ScalingUnscalingMethod DataSet::get_scaling_unscaling_method(const string& scaling_unscaling_method)
{
    if(scaling_unscaling_method == "NoScaling")
    {
        return NoScaling;
    }
    else if(scaling_unscaling_method == "NoUnscaling")
    {
        return NoUnscaling;
    }
    else if(scaling_unscaling_method == "MinimumMaximum")
    {
        return MinimumMaximum;
    }
    else if(scaling_unscaling_method == "Logarithmic")
    {
        return Logarithmic;
    }
    else if(scaling_unscaling_method == "MeanStandardDeviation")
    {
        return MeanStandardDeviation;
    }
    else if(scaling_unscaling_method == "StandardDeviation")
    {
        return StandardDeviation;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "static ScalingUnscalingMethod get_scaling_unscaling_method(const string).\n"
               << "Unknown scaling-unscaling method: " << scaling_unscaling_method << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Returns a matrix with the training samples in the data set.
/// The number of rows is the number of training
/// The number of columns is the number of variables.

Tensor<type, 2> DataSet::get_training_data() const
{

//       const Index variables_number = get_variables_number();

//       Tensor<Index, 1> variables_indices(0, 1, variables_number-1);

       Tensor<Index, 1> variables_indices = get_used_variables_indices();

       const Tensor<Index, 1> training_indices = get_training_samples_indices();

       return get_subtensor_data(training_indices, variables_indices);

//    return Tensor<type,2>();
}


/// Returns a matrix with the selection samples in the data set.
/// The number of rows is the number of selection
/// The number of columns is the number of variables.

Tensor<type, 2> DataSet::get_selection_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Index variables_number = get_variables_number();

    Tensor<Index, 1> variables_indices;
    initialize_sequential_eigen_tensor(variables_indices, 0, 1, variables_number-1);

    return get_subtensor_data(selection_indices, variables_indices);
}


/// Returns a matrix with the testing samples in the data set.
/// The number of rows is the number of testing
/// The number of columns is the number of variables.

Tensor<type, 2> DataSet::get_testing_data() const
{
    const Index variables_number = get_variables_number();

    Tensor<Index, 1> variables_indices;
    initialize_sequential_eigen_tensor(variables_indices, 0, 1, variables_number-1);

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    return get_subtensor_data(testing_indices, variables_indices);
}


/// Returns a matrix with the input variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Tensor<type, 2> DataSet::get_input_data() const
{
    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;
    initialize_sequential_eigen_tensor(indices, 0, 1, samples_number-1);

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(indices, input_variables_indices);
}


/// Returns a matrix with the target variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of target variables.

Tensor<type, 2> DataSet::get_target_data() const
{
    const Tensor<Index, 1> indices = get_used_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(indices, target_variables_indices);
}


/// Returns a tensor with the input variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Tensor<type, 2> DataSet::get_input_data(const Tensor<Index, 1>& samples_indices) const
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(samples_indices, input_variables_indices);
}


/// Returns a tensor with the target variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Tensor<type, 2> DataSet::get_target_data(const Tensor<Index, 1>& samples_indices) const
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(samples_indices, target_variables_indices);
}


/// Returns a matrix with training samples and input variables.
/// The number of rows is the number of training
/// The number of columns is the number of input variables.

Tensor<type, 2> DataSet::get_training_input_data() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(training_indices, input_variables_indices);
}


/// Returns a tensor with training samples and target variables.
/// The number of rows is the number of training
/// The number of columns is the number of target variables.

Tensor<type, 2> DataSet::get_training_target_data() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1>& target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(training_indices, target_variables_indices);
}


/// Returns a tensor with selection samples and input variables.
/// The number of rows is the number of selection
/// The number of columns is the number of input variables.

Tensor<type, 2> DataSet::get_selection_input_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return get_subtensor_data(selection_indices, input_variables_indices);
}


/// Returns a tensor with selection samples and target variables.
/// The number of rows is the number of selection
/// The number of columns is the number of target variables.

Tensor<type, 2> DataSet::get_selection_target_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(selection_indices, target_variables_indices);
}


/// Returns a tensor with testing samples and input variables.
/// The number of rows is the number of testing
/// The number of columns is the number of input variables.

Tensor<type, 2> DataSet::get_testing_input_data() const
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    return get_subtensor_data(testing_indices, input_variables_indices);
}


/// Returns a tensor with testing samples and target variables.
/// The number of rows is the number of testing
/// The number of columns is the number of target variables.

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

#ifdef __OPENNN_DEBUG__

    const Index samples_number = get_samples_number();

    if(index >= samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_sample(const Index&) const method.\n"
               << "Index of sample (" << index << ") must be less than number of samples (" << samples_number << ").\n";

        throw logic_error(buffer.str());
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
#ifdef __OPENNN_DEBUG__

    const Index samples_number = get_samples_number();

    if(sample_index >= samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_sample(const Index&, const Tensor<Index, 1>&) const method.\n"
               << "Index of sample must be less than number of \n";

        throw logic_error(buffer.str());
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

    //return data.get_row(sample_index, variables_indices);

}


/// Returns the inputs values of a single sample in the data set.
/// @param sample_index Index of the sample.

Tensor<type, 2> DataSet::get_sample_input_data(const Index & sample_index) const
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> inputs(1, input_variables_number);

    for(Index i = 0; i < input_variables_number; i++)
        inputs(0, i) = data(sample_index, input_variables_indices(i));

    return inputs;
}


/// Returns the target values of a single sample in the data set.
/// @param sample_index Index of the sample.

Tensor<type, 2> DataSet::get_sample_target_data(const Index & sample_index) const
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return get_subtensor_data(Tensor<Index, 1>(sample_index), target_variables_indices);
}


/// Returns the index of the column with the given name.
/// @param column_name Name of the column to be found.

Index DataSet::get_column_index(const string& column_name) const
{
    const Index columns_number = get_columns_number();

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).name == column_name) return i;
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: DataSet class.\n"
           << "Index get_column_index(const string&&) const method.\n"
           << "Cannot find " << column_name << "\n";

    throw logic_error(buffer.str());
}


/// Returns the index of the column to which a variable index belongs.
/// @param variable_index Index of the variable to be found.

Index DataSet::get_column_index(const Index& variable_index) const
{
    const Index columns_number = get_columns_number();

    Index total_variables_number = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Categorical)
        {
            total_variables_number += columns(i).get_categories_number();
        }
        else
        {
            total_variables_number++;
        }

        if((variable_index+1) <= total_variables_number) return i;
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: DataSet class.\n"
           << "Index get_column_index(const type&) const method.\n"
           << "Cannot find variable index: " << variable_index << ".\n";

    throw logic_error(buffer.str());
}


/// Returns the indices of a variable in the data set.
/// Note that the number of variables does not have to equal the number of columns in the data set,
/// because OpenNN recognizes the categorical columns, separating these categories into variables of the data set.

Tensor<Index, 1> DataSet::get_variable_indices(const Index& column_index) const
{
    Index index = 0;

    for(Index i = 0; i < column_index; i++)
    {
        if(columns(i).type == Categorical)
        {
            index += columns(i).categories.size();
        }
        else
        {
            index++;
        }
    }

    if(columns(column_index).type == Categorical)
    {
        Tensor<Index, 1> variable_indices(columns(column_index).categories.size());

        for (Index j = 0; j<columns(column_index).categories.size(); j++)
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


/// Returns the data from the data set of the given variables indices.
/// @param variables_indices Variable indices.
/// @todo

Tensor<type, 2> DataSet::get_column_data(const Tensor<Index, 1>& variables_indices) const
{
//    return data.get_submatrix_columns(variables_indices);

    return Tensor<type, 2>();
}


/// Returns the data from the data set column with a given index,
/// these data can be stored in a matrix or a vector depending on whether the column is categorical or not(respectively).
/// @param column_index Index of the column.

Tensor<type, 2> DataSet::get_column_data(const Index& column_index) const
{
    Index columns_number = 1;
    const Index rows_number = data.dimension(0);

    if(columns(column_index).type == Categorical)
    {
        columns_number = columns(column_index).get_categories_number();
    }

    Eigen::array<Index, 2> extents = {rows_number, columns_number};
    Eigen::array<Index, 2> offsets = {0, get_variable_indices(column_index)(0)};

    return data.slice(offsets, extents);
}

/// Returns the data from the time series column with a given index,
/// @param column_index Index of the column.

Tensor<type, 2> DataSet::get_time_series_column_data(const Index& column_index) const
{
    Index columns_number = 1;
    const Index rows_number = data.dimension(0);

    if(time_series_columns(column_index).type == Categorical)
    {
        columns_number = time_series_columns(column_index).get_categories_number();
    }

    Eigen::array<Index, 2> extents = {rows_number, columns_number};
    Eigen::array<Index, 2> offsets = {0, get_variable_indices(column_index)(0)};

    return time_series_data.slice(offsets, extents);
}


/// Returns the data from the data set column with a given index,
/// these data can be stored in a matrix or a vector depending on whether the column is categorical or not(respectively).
/// @param column_index Index of the column.
/// @param rows_indices Rows of the indices.

Tensor<type, 2> DataSet::get_column_data(const Index& column_index, Tensor<Index, 1>& rows_indices) const
{
    return get_subtensor_data(rows_indices, get_variable_indices(column_index));
}


/// Returns the data from the data set column with a given name,
/// these data can be stored in a matrix or a vector depending on whether the column is categorical or not(respectively).
/// @param column_name Name of the column.

Tensor<type, 2> DataSet::get_column_data(const string& column_name) const
{
    const Index column_index = get_column_index(column_name);

    return get_column_data(column_index);
}


/// Returns all the samples of a single variable in the data set.
/// @param index Index of the variable.

Tensor<type, 1> DataSet::get_variable_data(const Index& index) const
{

#ifdef __OPENNN_DEBUG__

    const Index variables_number = get_variables_number();

    if(index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const Index&) const method.\n"
               << "Index of variable must be less than number of \n";

        throw logic_error(buffer.str());
    }

#endif

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

#ifdef __OPENNN_DEBUG__

    const Index variables_size = variable_index.size();

    if(variables_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&) const method.\n"
               << "Variable: " << variable_name << " does not exist.\n";

        throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&) const method.\n"
               << "Variable: " << variable_name << " appears more than once in the data set.\n";

        throw logic_error(buffer.str());
    }

#endif

    return data.chip(variable_index(0), 1);
}


/// Returns a given set of samples of a single variable in the data set.
/// @param variable_index Index of the variable.
/// @param samples_indices Indices of the

Tensor<type, 1> DataSet::get_variable_data(const Index& variable_index, const Tensor<Index, 1>& samples_indices) const
{

#ifdef __OPENNN_DEBUG__

    const Index variables_number = get_variables_number();

    if(variable_index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const Index&, const Tensor<Index, 1>&) const method.\n"
               << "Index of variable must be less than number of \n";

        throw logic_error(buffer.str());
    }

#endif

    const Index samples_indices_size = samples_indices.size();

    Tensor<type, 1 > column(samples_indices_size);

    for(Index i = 0; i < samples_indices_size; i++)
    {
        Index sample_index = samples_indices(i);

        column(i) = data(sample_index, variable_index);
    }

    return column;
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

#ifdef __OPENNN_DEBUG__

    const Index variables_size = variable_index.size();

    if(variables_size == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&) const method.\n"
               << "Variable: " << variable_name << " does not exist.\n";

        throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 1> get_variable(const string&, const Tensor<Index, 1>&) const method.\n"
               << "Variable: " << variable_name << " appears more than once in the data set.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Index samples_indices_size = samples_indices.size();

    Tensor<type, 1 > column(samples_indices_size);

    for(Index i = 0; i < samples_indices_size; i++)
    {
        Index sample_index = samples_indices(i);

        column(i) = data(sample_index, variable_index(0));
    }

    return column;
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
    data_file_name = "";

    data.resize(0,0);
}


/// Sets all variables from a data matrix.
/// @param new_data Data matrix.

void DataSet::set(const Tensor<type, 2>& new_data)
{
    data_file_name = "";

    const Index variables_number = new_data.dimension(1);
    const Index samples_number = new_data.dimension(0);

    set(samples_number, variables_number);

    data = new_data;

    set_default_columns_uses();
}


/// Sets new numbers of samples and variables in the inputs targets data set.
/// All the samples are set for training.
/// All the variables are set as inputs.
/// @param new_samples_number Number of
/// @param new_variables_number Number of variables.

void DataSet::set(const Index& new_samples_number, const Index& new_variables_number)
{
#ifdef __OPENNN_DEBUG__

    if(new_samples_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set(const Index&, const Index&) method.\n"
               << "Number of samples must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

    if(new_variables_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set(const Index&, const Index&) method.\n"
               << "Number of variables must be greater than zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    data.resize(new_samples_number, new_variables_number);

    columns.resize(new_variables_number);

    for(Index index = 0; index < new_variables_number-1; index++)
    {
        columns(index).name = "column_" + to_string(index+1);
        columns(index).column_use = Input;
        columns(index).type = Numeric;
    }

    columns(new_variables_number-1).name = "column_" + to_string(new_variables_number);
    columns(new_variables_number-1).column_use = Target;
    columns(new_variables_number-1).type = Numeric;

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

    data_file_name = "";

    const Index new_variables_number = new_inputs_number + new_targets_number;

    data.resize(new_samples_number, new_variables_number);

    columns.resize(new_variables_number);

    for(Index i = 0; i < new_variables_number; i++)
    {
        if(i < new_inputs_number)
        {
            columns(i).name = "column_" + to_string(i+1);
            columns(i).column_use = Input;
            columns(i).type = Numeric;
        }
        else
        {
            columns(i).name = "column_" + to_string(i+1);
            columns(i).column_use = Target;
            columns(i).type = Numeric;
        }
    }

    input_variables_dimensions.resize(new_inputs_number);

    samples_uses.resize(new_samples_number);
    split_samples_random();
}


/// Sets the members of this data set object with those from another data set object.
/// @param other_data_set Data set object to be copied.

void DataSet::set(const DataSet& other_data_set)
{
    data_file_name = other_data_set.data_file_name;

    has_columns_names = other_data_set.has_columns_names;

    separator = other_data_set.separator;

    missing_values_label = other_data_set.missing_values_label;

    data = other_data_set.data;

    columns = other_data_set.columns;

    display = other_data_set.display;
}


/// Sets the data set members from a XML document.
/// @param data_set_document TinyXML document containing the member data.

void DataSet::set(const tinyxml2::XMLDocument& data_set_document)
{
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
    delete non_blocking_thread_pool;
    delete thread_pool_device;

    const int n = omp_get_max_threads();
    non_blocking_thread_pool = new NonBlockingThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

    has_columns_names = false;

    separator = Comma;

    missing_values_label = "NA";

    lags_number = 0;

    steps_ahead = 0;

    set_default_columns_uses();

    set_default_columns_names();

    input_variables_dimensions.resize(1);

    input_variables_dimensions.setConstant(get_input_variables_number());

}


/// Sets a new data matrix.
/// The number of rows must be equal to the number of
/// The number of columns must be equal to the number of variables.
/// Indices of all training, selection and testing samples and inputs and target variables do not change.
/// @param new_data Data matrix.

void DataSet::set_data(const Tensor<type, 2>& new_data)
{

    const Index samples_number = new_data.dimension(0);
    const Index variables_number = new_data.dimension(1);

    set(samples_number, variables_number);

    data = new_data;
}

void DataSet::set_time_series_data(const Tensor<type, 2>& new_data)
{
    time_series_data = new_data;
}

/// Sets the name of the data file.
/// It also loads the data from that file.
/// Moreover, it sets the variables and samples objects.
/// @param new_data_file_name Name of the file containing the data.

void DataSet::set_data_file_name(const string& new_data_file_name)
{
    data_file_name = new_data_file_name;
}


/// Sets if the data file contains a header with the names of the columns.

void DataSet::set_has_columns_names(const bool& new_has_columns_names)
{
    has_columns_names = new_has_columns_names;
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
        separator = Space;
    }
    else if(new_separator == '\t')
    {
        separator = Tab;
    }
    else if(new_separator == ',')
    {
        separator = Comma;
    }
    else if(new_separator == ';')
    {
        separator = Semicolon;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_separator(const char&) method.\n"
               << "Unknown separator: " << new_separator << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new separator from a string.
/// @param new_separator Char with the separator value.

void DataSet::set_separator(const string& new_separator_string)
{
    if(new_separator_string == "Space")
    {
        separator = Space;
    }
    else if(new_separator_string == "Tab")
    {
        separator = Tab;
    }
    else if(new_separator_string == "Comma")
    {
        separator = Comma;
    }
    else if(new_separator_string == "Semicolon")
    {
        separator = Semicolon;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_separator(const string&) method.\n"
               << "Unknown separator: " << new_separator_string << ".\n";

        throw logic_error(buffer.str());
    }
}



/// Sets a new label for the missing values.
/// @param new_missing_values_label Label for the missing values.

void DataSet::set_missing_values_label(const string& new_missing_values_label)
{
#ifdef __OPENNN_DEBUG__

    if(get_trimmed(new_missing_values_label).empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set_missing_values_label(const string&) method.\n"
              << "Missing values label cannot be empty.\n";

       throw logic_error(buffer.str());
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
        missing_values_method = Unuse;
    }
    else if(new_missing_values_method == "Mean")
    {
        missing_values_method = Mean;
    }
    else if(new_missing_values_method == "Median")
    {
        missing_values_method = Median;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_missing_values_method(const string & method.\n"
               << "Not known method type.\n";

        throw logic_error(buffer.str());
    }
}


/// Sets a new number of lags to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_lags_number Number of lags(x-1, ..., x-l) to be used.

void DataSet::set_lags_number(const Index& new_lags_number)
{
    lags_number = new_lags_number;
}


/// Sets a new number of steps ahead to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_steps_ahead_number Number of steps ahead to be used.

void DataSet::set_steps_ahead_number(const Index& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


/// Sets the new position where the time data is located in the data set.
/// @param new_time_index Position where the time data is located.

void DataSet::set_time_index(const Index& new_time_index)
{
    time_index = new_time_index;
}


void DataSet::set_threads_number(const int& new_threads_number)
{        
    if(non_blocking_thread_pool != nullptr) delete non_blocking_thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    non_blocking_thread_pool = new NonBlockingThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, new_threads_number);
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

Tensor<string, 1> DataSet::unuse_constant_columns()
{
    const Index columns_number = get_columns_number();

#ifdef __OPENNN_DEBUG__

    if(columns_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<string, 1> unuse_constant_columns() method.\n"
               << "Number of columns is zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<string, 1> constant_columns(0);

    Index variable_index = 0;

    for(Index i = 0; i < columns_number; i++)
    {


        if(columns(i).column_use == Input)
        {

            if(columns(i).type == Categorical)
            {

                const Index categories_number = columns(i).categories.size();

                bool is_constant = true;

                for(Index j = 0; j < categories_number; j++)
                {

                    const type column_standard_deviation = standard_deviation(data.chip(variable_index+j,1), used_samples_indices);
                    if((column_standard_deviation - 0) > numeric_limits<type>::min())
                    {
                        is_constant = false;
                        break;
                    }

                }

                if(is_constant) columns(i).set_use(UnusedVariable);

                constant_columns = push_back(constant_columns, columns(i).name);

            }
            else
            {

                const type column_standard_deviation = standard_deviation(data.chip(variable_index,1), used_samples_indices);

                if((column_standard_deviation - 0) < numeric_limits<type>::min())

                {
                    columns(i).set_use(UnusedVariable);

                    constant_columns = push_back(constant_columns, columns(i).name);

                }
            }
        }

        columns(i).type == Categorical ? variable_index += columns(i).categories.size() : variable_index++;

    }
    return constant_columns;
}


/// Removes the training, selection and testing indices of that samples which are repeated in the data matrix.
/// It might change the size of the vectors containing the training, selection and testing indices.

Tensor<Index, 1> DataSet::unuse_repeated_samples()
{
    const Index samples_number = get_samples_number();

#ifdef __OPENNN_DEBUG__

    if(samples_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> unuse_repeated_samples() method.\n"
               << "Number of samples is zero.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<Index, 1> repeated_samples;

    Tensor<type, 1> sample_i;
    Tensor<type, 1> sample_j;

    #pragma omp parallel for private(sample_i, sample_j) schedule(dynamic)

    for(Index i = 0; i < static_cast<Index>(samples_number); i++)
    {
        sample_i = get_sample_data(i);

        for(Index j = static_cast<Index>(i+1); j < samples_number; j++)
        {
            sample_j = get_sample_data(j);

            if(get_sample_use(j) != UnusedSample
                    && std::equal(sample_i.data(), sample_i.data()+sample_i.size(), sample_j.data()))
            {
                set_sample_use(j, UnusedSample);

                repeated_samples = push_back(repeated_samples, j);
            }
        }
    }

    return repeated_samples;
}


/// Return unused variables without correlation.
/// @param minimum_correlation Minimum correlation between variables.

Tensor<string, 1> DataSet::unuse_uncorrelated_columns(const type& minimum_correlation)
{
    Tensor<string, 1> unused_columns;

    const Tensor<CorrelationResults, 2> correlations = calculate_input_target_columns_correlations();

    const Index input_columns_number = get_input_columns_number();
    const Index target_columns_number = get_target_columns_number();

    const Tensor<Index, 1> input_columns_indices = get_input_columns_indices();

    for(Index i = 0; i < input_columns_number; i++)
    {
        const Index index = input_columns_indices(i);

        for(Index j = 0; j < target_columns_number; j++)
        {
            if(columns(index).column_use != UnusedVariable && abs(correlations(i,j).correlation) < minimum_correlation)
            {
                columns(index).set_use(UnusedVariable);

                unused_columns = push_back(unused_columns, columns(index).name);
            }
        }
    }

    return unused_columns;
}


/// Returns the distribution of each of the columns. In the case of numeric columns, it returns a
/// histogram, for the case of categorical columns, it returns the frequencies of each category nad for the
/// binary columns it returns the frequencies of the positives and negatives.
/// The default number of bins is 10.
/// @param bins_number Number of bins.

Tensor<Histogram, 1> DataSet::calculate_columns_distribution(const Index& bins_number) const
{
    const Index columns_number = columns.size();
    const Index used_columns_number = get_used_columns_number();
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Index used_samples_number = used_samples_indices.size();

    Tensor<Histogram, 1> histograms(used_columns_number);

    Index variable_index = 0;
    Index used_column_index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Numeric)
        {
            if(columns(i).column_use == UnusedVariable)
            {
                variable_index++;
            }
            else
            {
                Tensor<type, 1> column(used_samples_number);

                for(Index j = 0; j < used_samples_number; j++)
                {
                    column(j) = data(used_samples_indices(j), variable_index);
                }

                histograms(used_column_index) = histogram(column, bins_number);

                variable_index++;
                used_column_index++;
            }
        }
        else if(columns(i).type == Categorical)
        {
            const Index categories_number = columns(i).get_categories_number();

            if(columns(i).column_use == UnusedVariable)
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
                        if(abs(data(used_samples_indices(k), variable_index) - 1) < numeric_limits<type>::min())
                        {
                            categories_frequencies(j)++;
                        }
                    }

                    centers(j) = static_cast<type>(j);

                    variable_index++;
                }

                histograms(used_column_index).frequencies = categories_frequencies;
                histograms(used_column_index).centers = centers;

                used_column_index++;
            }
        }
        else if(columns(i).type == Binary)
        {
            if(columns(i).column_use == UnusedVariable)
            {
                variable_index++;
            }
            else
            {
                Tensor<Index, 1> binary_frequencies(2);
                binary_frequencies.setZero();

                for(Index j = 0; j < used_samples_number; j++)
                {
                    if(fabsf(data(used_samples_indices(j), variable_index) - 1) < numeric_limits<type>::min())
                    {
                        binary_frequencies(0)++;
                    }
                    else
                    {
                        binary_frequencies(1)++;
                    }
                }

                histograms(used_column_index).frequencies = binary_frequencies;
                variable_index++;
                used_column_index++;
            }
        }
        else // Time @todo
        {
            variable_index++;
        }
    }

    return histograms;
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

Tensor<BoxPlot, 1> DataSet::calculate_columns_box_plots() const
{
    Index used_columns_number = get_used_columns_number();

    Index columns_number = get_columns_number();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<BoxPlot, 1> box_plots(used_columns_number);

    Index used_column_index = 0;
    Index variable_index = 0;

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Numeric || columns(i).type == Binary)
        {
            if(columns(i).column_use != UnusedVariable)
            {
                cout << "Column: " << columns(i).name << endl;

                box_plots(used_column_index) = box_plot(data.chip(variable_index, 1), used_samples_indices);

                cout << "min: " << box_plots(used_column_index).minimum << endl;
                cout << "max: " << box_plots(used_column_index).maximum << endl;


                used_column_index++;
            }

            variable_index++;
        }
        else if(columns(i).type == Categorical)
        {
            variable_index += columns(i).get_categories_number();
        }
        else
        {
            variable_index++;
        }
    }

    return box_plots;
}


/// Counts the number of used negatives of the selected target.
/// @param target_index Index of the target to evaluate.

Index DataSet::calculate_used_negatives(const Index& target_index) const
{
    Index negatives = 0;

    const Tensor<Index, 1> used_indices = get_used_samples_indices();

    const Index used_samples_number = used_indices.size();

    for(Index i = 0; i < used_samples_number; i++)
    {
        const Index training_index = used_indices(i);

        if(fabsf(data(training_index, target_index)) < numeric_limits<type>::min())
        {
            negatives++;
        }
        else if(fabsf(data(training_index, target_index) - static_cast<type>(1)) > static_cast<type>(1.0e-3))
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Index calculate_used_negatives(const Index&) const method.\n"
                   << "Training sample is neither a positive nor a negative: " << data(training_index, target_index) << endl;

            throw logic_error(buffer.str());
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

        if(fabsf(data(training_index, target_index)) < numeric_limits<type>::min())
        {
            negatives++;
        }
        else if(fabsf(data(training_index, target_index) - static_cast<type>(1)) > static_cast<type>(1.0e-3))
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Index calculate_training_negatives(const Index&) const method.\n"
                   << "Training sample is neither a positive nor a negative: " << data(training_index, target_index) << endl;

            throw logic_error(buffer.str());
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

    for(Index i = 0; i < static_cast<Index>(selection_samples_number); i++)
    {
        const Index selection_index = selection_indices(i);

        if(fabsf(data(selection_index, target_index)) < numeric_limits<type>::min())
        {
            negatives++;
        }
        else if(fabsf(data(selection_index, target_index) - 1) > numeric_limits<type>::min())
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "Index calculate_testing_negatives(const Index&) const method.\n"
                   << "Selection sample is neither a positive nor a negative: " << data(selection_index, target_index) << endl;

            throw logic_error(buffer.str());
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

    for(Index i = 0; i < static_cast<Index>(testing_samples_number); i++)
    {
        const Index testing_index = testing_indices(i);

        if(data(testing_index, target_index) < numeric_limits<type>::min())
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
/// @todo Low priority.

Tensor<Descriptives, 1> DataSet::calculate_columns_descriptives_positive_samples() const
{

#ifdef __OPENNN_DEBUG__

    const Index targets_number = get_target_variables_number();

    if(targets_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 2> calculate_columns_descriptives_positive_samples() const method.\n"
               << "Number of targets muste be 1.\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index target_index = get_target_variables_indices()(0);


    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used positive samples

    Index positive_samples_number = 0;

    for (Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, target_index) - 1) < numeric_limits<type>::min()) positive_samples_number++;
    }

        // Get used positive samples indices

    Tensor<Index, 1> positive_used_samples_indices(positive_samples_number);
    Index positive_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, target_index) - 1) < numeric_limits<type>::min())
        {
            positive_used_samples_indices(positive_sample_index) = sample_index;
            positive_sample_index++;
        }
    }
    return descriptives(data, positive_used_samples_indices, input_variables_indices);
}


/// Calculate the descriptives of the samples with neagtive targets in binary classification problems.
/// @todo Low priority.

Tensor<Descriptives, 1> DataSet::calculate_columns_descriptives_negative_samples() const
{

#ifdef __OPENNN_DEBUG__

    const Index targets_number = get_target_variables_number();

    if(targets_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 2> calculate_columns_descriptives_positive_samples() const method.\n"
               << "Number of targets muste be 1.\n";

        throw logic_error(buffer.str());
    }
#endif

    const Index target_index = get_target_variables_indices()(0);

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used negative samples

    Index negative_samples_number = 0;

    for (Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(data(sample_index, target_index) < numeric_limits<type>::min()) negative_samples_number++;
    }

    // Get used negative samples indices

    Tensor<Index, 1> negative_used_samples_indices(negative_samples_number);
    Index negative_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(data(sample_index, target_index) < numeric_limits<type>::min())
        {
            negative_used_samples_indices(negative_sample_index) = sample_index;
            negative_sample_index++;
        }

    }

    return descriptives(data, negative_used_samples_indices, input_variables_indices);
}


/// Returns a matrix with the data set descriptive statistics.
/// @param class_index Data set index number to make the descriptive statistics.

Tensor<Descriptives, 1> DataSet::calculate_columns_descriptives_categories(const Index& class_index) const
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used class samples

    Index class_samples_number = 0;

    for (Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, class_index) - 1) < numeric_limits<type>::min()) class_samples_number++;
    }

    // Get used class samples indices

    Tensor<Index, 1> class_used_samples_indices(class_samples_number);
    class_used_samples_indices.setZero();
    Index class_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, class_index) - 1) < numeric_limits<type>::min())
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

Tensor<Descriptives, 1> DataSet::calculate_columns_descriptives_training_samples() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1> used_indices = get_used_columns_indices();

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

Tensor<Descriptives, 1> DataSet::calculate_columns_descriptives_selection_samples() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> used_indices = get_used_columns_indices();

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


/// Returns a vector containing the minimums of the input variables.

Tensor<type, 1> DataSet::calculate_input_variables_minimums() const
{
    return columns_minimums(data, get_used_samples_indices(), get_input_variables_indices());
}


/// Returns a vector containing the minimums of the target variables.

Tensor<type, 1> DataSet::calculate_target_variables_minimums() const
{
    return columns_minimums(data, get_used_samples_indices(), get_target_variables_indices());
}



/// Returns a vector containing the maximums of the input variables.

Tensor<type, 1> DataSet::calculate_input_variables_maximums() const
{
    return columns_maximums(data, get_used_samples_indices(), get_input_variables_indices());
}


/// Returns a vector containing the maximums of the target variables.

Tensor<type, 1> DataSet::calculate_target_variables_maximums() const
{
    return columns_maximums(data, get_used_samples_indices(), get_target_variables_indices());
}


/// Returns a vector containing the maximum of the used variables.

Tensor<type, 1> DataSet::calculate_used_variables_minimums() const
{
    return columns_minimums(data, get_used_samples_indices(), get_used_variables_indices());
}

/// Returns a vector containing the means of a set of given variables.
/// @param variables_indices Indices of the variables.

Tensor<type, 1> DataSet::calculate_variables_means(const Tensor<Index, 1>& variables_indices) const
{
    const Index variables_number = variables_indices.size();

    Tensor<type, 1> means(variables_number);

    #pragma omp parallel for

    for(Index i = 0; i < variables_number; i++)
    {
        const Index variable_index = variables_indices(i);

        const Tensor<type, 0> mean = data.chip(variable_index,1).mean();

        means(i) = mean(0);
    }

    return means;
}


/// Returns a vector with some basic descriptives of the given input variable on all
/// The size of this vector is four:
/// <ul>
/// <li> Input variable minimum.
/// <li> Input variable maximum.
/// <li> Input variable mean.
/// <li> Input variable standard deviation.
/// </ul>
/// @todo

Descriptives DataSet::calculate_input_descriptives(const Index& input_index) const
{
//    return descriptives_missing_values(data.chip(input_index,1));

    return Descriptives();
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
/// and number of columns is the target number.
/// Each element contains the correlation between a single input and a single target.

Tensor<CorrelationResults, 2> DataSet::calculate_input_target_columns_correlations() const
{
    const Index input_columns_number = get_input_columns_number();
    const Index target_columns_number = get_target_columns_number();

    const Tensor<Index, 1> input_columns_indices = get_input_columns_indices();
    Tensor<Index, 1> target_columns_indices = get_target_columns_indices();

    Tensor<CorrelationResults, 2> correlations(input_columns_number, target_columns_number);

//    #pragma omp parallel for

    for(Index i = 0; i < input_columns_number; i++)
    {
        const Index input_index = input_columns_indices(i);

        Tensor<type, 2> input = get_column_data(input_index);

        const ColumnType input_type = columns(input_index).type;

        for(Index j = 0; j < target_columns_number; j++)
        {
            const Index target_index = target_columns_indices(j);

            Tensor<type, 2> target = get_column_data(target_index);

            const ColumnType target_type = columns(target_index).type;

            cout << "Calculating " << columns(input_index).name << " - " << columns(target_index).name << " correlations. \n" ;

            if(input_type == Numeric && target_type == Numeric)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                const CorrelationResults linear_correlation = linear_correlations(thread_pool_device, input_column, target_column);
                const CorrelationResults exponential_correlation = exponential_correlations(thread_pool_device, input_column, target_column);
                const CorrelationResults logarithmic_correlation = logarithmic_correlations(thread_pool_device, input_column, target_column);
                const CorrelationResults power_correlation = power_correlations(thread_pool_device, input_column, target_column);

                CorrelationResults strongest_correlation = linear_correlation;

                if(abs(exponential_correlation.correlation) > abs(strongest_correlation.correlation)) strongest_correlation = exponential_correlation;
                if(abs(logarithmic_correlation.correlation) > abs(strongest_correlation.correlation)) strongest_correlation = logarithmic_correlation;
                if(abs(power_correlation.correlation) > abs(strongest_correlation.correlation)) strongest_correlation = power_correlation;

                correlations(i,j) = strongest_correlation;
            }
            else if(input_type == Binary && target_type == Binary)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                correlations(i,j) = linear_correlations(thread_pool_device, input_column, target_column);
            }
            else if(input_type == Categorical && target_type == Categorical)
            {
                // @todo
                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input, target);

//                correlations(i,j) = karl_pearson_correlations(thread_pool_device, input, target);
            }
            else if(input_type == Numeric && target_type == Binary)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                correlations(i,j) = logistic_correlations(thread_pool_device, input_column, target_column);
            }
            else if(input_type == Binary && target_type == Numeric)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                correlations(i,j) = logistic_correlations(thread_pool_device, input_column, target_column);
            }
            else if(input_type == Categorical && target_type == Numeric)
            {
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input, target/*target_column*/);
            }
            else if(input_type == Numeric && target_type == Categorical)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));

                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, target, input/*input_column*/);
            }
            else if(input_type == Binary && target_type == Categorical)
            {
                const TensorMap<Tensor<type, 1>> input_column(input.data(), input.dimension(0));

                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, target, input/*input_column*/);

//                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input, target);

//                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));

//                Tensor<type, 2> new_input = transform_binary_column(input_column);

//                correlations(i,j) = karl_pearson_correlations(thread_pool_device, new_input, target);
            }
            else if(input_type == Categorical && target_type == Binary)
            {
                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input, target);

//                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

//                Tensor<type, 2> new_target = transform_binary_column(target_column);

//                correlations(i,j) = karl_pearson_correlations(thread_pool_device, input, new_target);
            }
            else if(input_type == DateTime || target_type == DateTime)
            {
                correlations(i,j).correlation = 0;
            }
            else
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: DataSet class.\n"
                       << "Tensor<type, 2> calculate_input_target_columns_correlations() const method.\n"
                       << "Case not found: Column " << columns(input_index).name << " and Column " << columns(target_index).name << ".\n";

                throw logic_error(buffer.str());
            }

            cout << "Correlation: " << correlations(i,j).correlation << endl;

        }
    }

    return correlations;
}


/// Calculates the correlations between all outputs and all inputs.
/// It returns a matrix with the number of rows is the input number
/// and number of columns is the target number.
/// Each element contains the correlation between a single input and a single target.

Tensor<type, 2> DataSet::calculate_input_target_columns_correlations_values() const
{
    Tensor<CorrelationResults, 2> correlations = calculate_input_target_columns_correlations();

    const Index rows_number = correlations.dimension(0);
    const Index columns_number = correlations.dimension(1);

    Tensor<type, 2> correlations_values(rows_number, columns_number);

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            correlations_values(i,j) = correlations(i,j).correlation;
        }
    }

    return correlations_values;
}


/// Returns true if the data contain missing values.

bool DataSet::has_nan() const
{
    for(Index i = 0; i < data.size(); i++) if(::isnan(data(i))) return true;

    return false;
}


/// Returns true if the given row contains missing values.

bool DataSet::has_nan_row(const Index& row_index) const
{
    for(Index j = 0; j < data.dimension(1); j++)
    {
        if(::isnan(data(row_index,j))) return true;
    }

    return false;
}


/// Print on screen the information about the missing values in the data set.
/// <ul>
/// <li> Total number of missing values.
/// <li> Number of variables with missing values.
/// <li> Number of samples with missing values.
/// </ul>
/// @todo implement with indices of variables and samples?

void DataSet::print_missing_values_information() const
{
//    const Index missing_values_number = data.count_nan();

//    cout << "Missing values number: " << missing_values_number << " (" << missing_values_number*100/data.size() << "%)" << endl;

//    const Index variables_with_missing_values = data.count_columns_with_nan();

//    cout << "Variables with missing values: " << variables_with_missing_values << " (" << variables_with_missing_values*100/data.dimension(1) << "%)" << endl;

//    const Index samples_with_missing_values = data.count_rows_with_nan();

//    cout << "Samples with missing values: " << samples_with_missing_values << " (" << samples_with_missing_values*100/data.dimension(0) << "%)" << endl;
}


/// Print on screen the correlation between targets and inputs.

void DataSet::print_input_target_columns_correlations() const
{
    const Index inputs_number = get_input_variables_number();
    const Index targets_number = get_target_variables_number();

    const Tensor<string, 1> inputs_names = get_input_variables_names();
    const Tensor<string, 1> targets_name = get_target_variables_names();

    const Tensor<RegressionResults, 2> correlations;// = calculate_input_target_columns_correlations();

    for(Index j = 0; j < targets_number; j++)
    {
        for(Index i = 0; i < inputs_number; i++)
        {
            cout << targets_name(j) << " - " << inputs_names(i) << ": " << correlations(i,j).correlation << endl;
        }
    }
}


/// This method print on screen the corretaliont between inputs and targets.
/// @param number Number of variables to be printed.
/// @todo

void DataSet::print_top_input_target_columns_correlations(const Index& number) const
{
    const Index inputs_number = get_input_columns_number();
    const Index targets_number = get_target_columns_number();

    const Tensor<string, 1> inputs_names = get_input_variables_names();
    const Tensor<string, 1> targets_name = get_target_variables_names();

    const Tensor<RegressionResults, 2> correlations;// = calculate_input_target_columns_correlations();

    Tensor<type, 1> target_correlations(inputs_number);

    Tensor<string, 2> top_correlations(inputs_number, 2);

    map<type,string> top_correlation;

    for(Index i = 0 ; i < inputs_number; i++)
    {
        for(Index j = 0 ; j < targets_number ; j++)
        {
//            top_correlation.insert(pair<type,string>(correlations(i,j), inputs_names(i) + " - " + targets_name(j)));
        }
    }

    map<type,string>::iterator it;

    for(it = top_correlation.begin(); it!=top_correlation.end(); it++)
    {
        cout << "Correlation:  " << (*it).first << "  between  " << (*it).second << "" << endl;
    }
}


/// Calculates the regressions between all outputs and all inputs.
/// It returns a matrix with the data stored in RegressionResults format, where the number of rows is the input number
/// and number of columns is the target number.
/// Each element contains the correlation between a single input and a single target.

Tensor<RegressionResults, 2> DataSet::calculate_input_target_columns_regressions() const
{
    const Index input_columns_number = get_input_columns_number();
    const Index target_columns_number = get_target_columns_number();

    const Tensor<Index, 1> input_columns_indices = get_input_columns_indices();
    Tensor<Index, 1> target_columns_indices = get_target_columns_indices();

    Tensor<RegressionResults, 2> regressions(input_columns_number, target_columns_number);

    //@todo check pragma, if uncommented, for does not work well.
//#pragma omp parallel for

    for(Index i = 0; i < input_columns_number; i++)
    {
        cout << endl;

        const Index input_index = input_columns_indices(i);

        Tensor<type, 2> input = get_column_data(input_index);

        const ColumnType input_type = columns(input_index).type;

        cout << "Calculating " << columns(input_index).name;

        for(Index j = 0; j < target_columns_number; j++)
        {
            const Index target_index = target_columns_indices(j);

            Tensor<type, 2> target = get_column_data(target_index);

            const ColumnType target_type = columns(target_index).type;

            cout << " - " << columns(target_columns_indices(j)).name << " regression. \n" ;

            if(input_type == Numeric && target_type == Numeric)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                const RegressionResults linear_regression = OpenNN::linear_regression(thread_pool_device, input_column, target_column);
                const RegressionResults exponential_regression = OpenNN::exponential_regression(thread_pool_device, input_column, target_column);
                const RegressionResults logarithmic_regression = OpenNN::logarithmic_regression(thread_pool_device, input_column, target_column);
                const RegressionResults power_regression = OpenNN::power_regression(thread_pool_device, input_column, target_column);

                RegressionResults strongest_regression = linear_regression;

                if(abs(exponential_regression.correlation) > abs(strongest_regression.correlation)) strongest_regression = exponential_regression;
                if(abs(logarithmic_regression.correlation) > abs(strongest_regression.correlation)) strongest_regression = logarithmic_regression;
                if(abs(power_regression.correlation) > abs(strongest_regression.correlation)) strongest_regression = power_regression;

                regressions(i,j) = strongest_regression;
            }
            else if(input_type == Binary && target_type == Binary)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                regressions(i,j) = linear_regression(thread_pool_device, input_column, target_column);
            }
            else if(input_type == Numeric && target_type == Binary)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                regressions(i,j) = logistic_regression(thread_pool_device, input_column, target_column);
            }
            else if(input_type == Binary && target_type == Numeric)
            {
                const TensorMap<Tensor<type,1>> input_column(input.data(), input.dimension(0));
                const TensorMap<Tensor<type,1>> target_column(target.data(), target.dimension(0));

                regressions(i,j) = logistic_regression(thread_pool_device, input_column, target_column);
            }
            else if(input_type == Categorical && target_type == Categorical)
            {
                // Nothing

                regressions(i,j).a = 0;
                regressions(i,j).b = 0;
            }
            else if(input_type == Categorical && target_type == Numeric)
            {
                // Nothing

                regressions(i,j).a = 0;
                regressions(i,j).b = 0;
            }
            else if(input_type == Numeric && target_type == Categorical)
            {
                // Nothing

                regressions(i,j).a = 0;
                regressions(i,j).b = 0;
            }
            else if(input_type == Binary && target_type == Categorical)
            {
                // nothing

                regressions(i,j).a = 0;
                regressions(i,j).b = 0;
            }
            else if(input_type == Categorical && target_type == Binary)
            {
                // nothing

                regressions(i,j).a = 0;
                regressions(i,j).b = 0;
            }
            else
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: DataSet class.\n"
                       << "Tensor<type, 2> calculate_input_target_columns_regressions() const method.\n"
                       << "Case not found: Column " << columns(input_index).name << " and Column " << columns(target_index).name << ".\n";

                throw logic_error(buffer.str());
            }
        }
    }

    return regressions;
}


/// Calculate the correlation between each input in the data set.
/// Returns a matrix with the correlation values between variables in the data set.

Tensor<type, 2> DataSet::calculate_input_columns_correlations() const
{
    const Tensor<Index, 1> input_columns_indices = get_input_columns_indices();

    const Index input_columns_number = get_input_columns_number();

    Tensor<type, 2> correlations(input_columns_number, input_columns_number);
    correlations.setConstant(1);

    for(Index i = 0; i < input_columns_number; i++)
    {
        const Index current_input_index_i = input_columns_indices(i);

        const ColumnType type_i = columns(current_input_index_i).type;

        Tensor<type, 2> input_i = get_column_data(current_input_index_i);

        cout << "Calculating " << columns(current_input_index_i).name << " correlations. " << endl;

        #pragma omp parallel for

        for(Index j = i; j < input_columns_number; j++)
        {
            const Index current_input_index_j = input_columns_indices(j);

            const ColumnType type_j = columns(current_input_index_j).type;

            Tensor<type, 2> input_j = get_column_data(current_input_index_j);

            if(current_input_index_i == current_input_index_j)
            {
                correlations(i,j) = 1;
                continue;
            }

            if(type_i == Numeric && type_j == Numeric)
            {
                const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));
                const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

                const type linear_correlation = OpenNN::linear_correlation(thread_pool_device, current_input_i, current_input_j);
                const type exponential_correlation = OpenNN::exponential_correlation(thread_pool_device, current_input_i, current_input_j);
                const type logarithmic_correlation = OpenNN::logarithmic_correlation(thread_pool_device, current_input_i, current_input_j);
                const type power_correlation = OpenNN::power_correlation(thread_pool_device, current_input_i, current_input_j);

                type strongest_correlation = linear_correlation;

                if(fabsf(exponential_correlation) > fabsf(strongest_correlation)) strongest_correlation = exponential_correlation;
                if(fabsf(logarithmic_correlation) > fabsf(strongest_correlation)) strongest_correlation = logarithmic_correlation;
                if(fabsf(power_correlation) > fabsf(strongest_correlation)) strongest_correlation = power_correlation;

                correlations(i,j) = strongest_correlation;
            }
            else if(type_i == Binary && type_j == Binary)
            {
                const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));
                const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

                correlations(i,j) = linear_correlation(thread_pool_device, current_input_i, current_input_j);
            }
            else if(type_i == Categorical && type_j == Categorical)
            {
                correlations(i,j) = karl_pearson_correlation(thread_pool_device, input_i, input_j);
            }
            else if(type_i == Numeric && type_j == Binary)
            {
                const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));
                const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

                correlations(i,j) = logistic_correlations(thread_pool_device, current_input_i, current_input_j).correlation;
            }
            else if(type_i == Binary && type_j == Numeric)
            {
                const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));
                const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

                correlations(i,j) = logistic_correlations(thread_pool_device, current_input_i, current_input_j).correlation;
            }
            else if(type_i == Categorical && type_j == Numeric)
            {
                const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input_i, input_j/*current_input_j*/).correlation;
            }
            else if(type_i == Numeric && type_j == Categorical)
            {
                const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));

                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input_j, input_i/*current_input_i*/).correlation;
            }
            else if(type_i == Categorical && type_j == Binary)
            {
                const TensorMap<Tensor<type, 1>> current_input_j(input_j.data(), input_j.dimension(0));

                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input_i, input_j/*current_input_j*/).correlation;
            }
            else if(type_i == Binary && type_j == Categorical)
            {
                const TensorMap<Tensor<type, 1>> current_input_i(input_i.data(), input_i.dimension(0));

                correlations(i,j) = multiple_logistic_correlations(thread_pool_device, input_j, input_i/*current_input_i*/).correlation;
            }
            else
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: DataSet class.\n"
                       << "Tensor<type, 2> calculate_inputs_correlations() const method.\n"
                       << "Case not found: Column " << columns(input_columns_indices(i)).name << " and Column " << columns(input_columns_indices(j)).name << ".\n";

                throw logic_error(buffer.str());
            }

        }
    }

    for(Index i = 0; i < input_columns_number; i++)
    {
        for(Index j = 0; j < i; j++)
        {
            correlations(i,j) = correlations(j,i);
        }
    }

    return correlations;

}


/// Print on screen the correlation between variables in the data set.

void DataSet::print_inputs_correlations() const
{
    const Tensor<type, 2> inputs_correlations = calculate_input_columns_correlations();

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
/// @todo Low priority.

void DataSet::print_top_inputs_correlations(const Index& number) const
{
    const Index variables_number = get_input_variables_number();

    const Tensor<string, 1> variables_name = get_input_variables_names();

    const Tensor<type, 2> variables_correlations = calculate_input_columns_correlations();

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

    map<type,string> :: iterator it;

    for(it=top_correlation.begin(); it!=top_correlation.end(); it++)
    {
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << "" << endl;
    }
}


/// Returns the covariance matrix for the input data set.
/// The number of rows of the matrix is the number of inputs.
/// The number of columns of the matrix is the number of inputs.
/// @todo

Tensor<type, 2> DataSet::calculate_covariance_matrix() const
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    const Index inputs_number = get_input_variables_number();

    Tensor<type, 2> covariance_matrix(inputs_number, inputs_number);

    for(Index i = 0; i < static_cast<Index>(inputs_number); i++)
    {
        const Index first_input_index = input_variables_indices(i);

//        const Tensor<type, 1> first_inputs = data.get_column(first_input_index, used_samples_indices);

        for(Index j = i; j < inputs_number; j++)
        {
            const Index second_input_index = input_variables_indices(j);

//            const Tensor<type, 1> second_inputs = data.get_column(second_input_index, used_samples_indices);

//            covariance_matrix(i,j) = covariance(first_inputs, second_inputs);
            covariance_matrix(j,i) = covariance_matrix(i,j);
        }
    }

    return covariance_matrix;
}


/// Performs the principal components analysis of the inputs.
/// It returns a matrix containing the principal components getd in rows.
/// This method deletes the unused samples of the original data set.
/// @param minimum_explained_variance Minimum percentage of variance used to select a principal component.

Tensor<type, 2> DataSet::perform_principal_components_analysis(const type& minimum_explained_variance)
{
    // Subtract off the mean

    subtract_inputs_mean();

    // Calculate covariance matrix

    const Tensor<type, 2> covariance_matrix = this->calculate_covariance_matrix();

    // Calculate eigenvectors

//    const Tensor<type, 2> eigenvectors = OpenNN::eigenvectors(covariance_matrix);

    // Calculate eigenvalues

//    const Tensor<type, 2> eigenvalues = OpenNN::eigenvalues(covariance_matrix);

    // Calculate explained variance

//    const Tensor<type, 1> explained_variance = OpenNN::explained_variance(eigenvalues.chip(0,1));

    // Sort principal components

//    const Tensor<Index, 1> sorted_principal_components_indices = explained_variance.sort_descending_indices();

    // Choose eigenvectors

    const Index inputs_number = covariance_matrix.dimension(1);

    Tensor<Index, 1> principal_components_indices;

    Index index;

    for(Index i = 0; i < inputs_number; i++)
    {
//        index = sorted_principal_components_indices(i);

//        if(explained_variance(index) >= minimum_explained_variance)
        {
//            principal_components_indices.push_back(i);
        }
//        else
        {
            continue;
        }
    }

    const Index principal_components_number = principal_components_indices.size();

    // Arrange principal components matrix

    Tensor<type, 2> principal_components;

    if(principal_components_number == 0)
    {
        return principal_components;
    }
    else
    {
        principal_components.resize(principal_components_number, inputs_number);
    }

    for(Index i = 0; i < principal_components_number; i++)
    {
//        index = sorted_principal_components_indices(i);

//        principal_components.set_row(i, eigenvectors.chip(index,1));
    }

    // Return feature matrix

//    return principal_components.get_submatrix_rows(principal_components_indices);

    return Tensor<type, 2>();
}


/// Performs the principal components analysis of the inputs.
/// It returns a matrix containing the principal components arranged in rows.
/// This method deletes the unused samples of the original data set.
/// @param covariance_matrix Matrix of covariances.
/// @param explained_variance vector of the explained variances of the variables.
/// @param minimum_explained_variance Minimum percentage of variance used to select a principal component.
/// @todo

Tensor<type, 2> DataSet::perform_principal_components_analysis(const Tensor<type, 2>& covariance_matrix,
        const Tensor<type, 1>& explained_variance,
        const type& minimum_explained_variance)
{
        // Subtract off the mean

        subtract_inputs_mean();

        // Calculate eigenvectors

//        const Tensor<type, 2> eigenvectors = OpenNN::eigenvectors(covariance_matrix);

        // Sort principal components

//        const Tensor<Index, 1> sorted_principal_components_indices = explained_variance.sort_descending_indices();

        // Choose eigenvectors

        const Index inputs_number = covariance_matrix.dimension(1);

        Tensor<Index, 1> principal_components_indices;

        Index index;

        for(Index i = 0; i < inputs_number; i++)
        {
//            index = sorted_principal_components_indices(i);

//            if(explained_variance(index) >= minimum_explained_variance)
            {
//                principal_components_indices.push_back(i);
            }
//            else
            {
                continue;
            }
        }

        const Index principal_components_number = principal_components_indices.size();

        // Arrange principal components matrix

        Tensor<type, 2> principal_components;

        if(principal_components_number == 0)
        {
            return principal_components;
        }
        else
        {
            principal_components.resize(principal_components_number, inputs_number);
        }

        for(Index i = 0; i < principal_components_number; i++)
        {
//            index = sorted_principal_components_indices(i);

//            principal_components.set_row(i, eigenvectors.chip(index,1));
        }

        // Return feature matrix

//        return principal_components.get_submatrix_rows(principal_components_indices);

    return Tensor<type, 2>();
}


/// Transforms the data according to the principal components.
/// @param principal_components Matrix containing the principal components.
/// @todo

void DataSet::transform_principal_components_data(const Tensor<type, 2>& principal_components)
{
    const Tensor<type, 2> targets = get_target_data();

    subtract_inputs_mean();

    const Index principal_components_number = principal_components.dimension(0);

    // Transform data

    const Tensor<Index, 1> used_samples = get_used_samples_indices();

    const Index new_samples_number = get_used_samples_number();

    const Tensor<type, 2> inputs = get_input_data();

    Tensor<type, 2> new_data(new_samples_number, principal_components_number);

    Index sample_index;

    for(Index i = 0; i < new_samples_number; i++)
    {
        sample_index = used_samples(i);

        for(Index j = 0; j < principal_components_number; j++)
        {
            Tensor<type, 0> dot = (inputs.chip(sample_index, 0)).contract(principal_components.chip(j,0),product_vector_vector);

            new_data(i,j) = dot(0);
//            new_data(i,j) = dot(inputs.chip(sample_index, 0), principal_components.chip(j, 0));
        }
    }

//        data = new_data.assemble_columns(targets);

}


/// Scales the data matrix with given mean and standard deviation values.
/// It updates the data matrix.
/// @param data_descriptives vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.
/// @todo

void DataSet::scale_data_mean_standard_deviation(const Tensor<Descriptives, 1>& data_descriptives)
{

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const Index columns_number = data.dimension(1);

   const Index descriptives_size = data_descriptives.size();

   if(descriptives_size != columns_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void scale_data_mean_standard_deviation(const Tensor<Descriptives, 1>&) method.\n"
             << "Size of descriptives must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const Index variables_number = get_variables_number();

   for(Index i = 0; i < variables_number; i++)
   {
       if(display && abs(data_descriptives(i).standard_deviation) < numeric_limits<type>::min())
       {
          cout << "OpenNN Warning: DataSet class.\n"
                    << "void scale_data_mean_standard_deviation(const Tensor<Descriptives, 1>&) method.\n"
                    << "Standard deviation of variable " <<  i << " is zero.\n"
                    << "That variable won't be scaled.\n";
        }
    }

//   scale_mean_standard_deviation(data, data_descriptives);

}


/// Scales the data using the minimum and maximum method,
/// and the minimum and maximum values calculated from the data matrix.
/// It also returns the descriptives from all columns.

Tensor<Descriptives, 1> DataSet::scale_data_minimum_maximum()
{
    const Tensor<Descriptives, 1> data_descriptives = calculate_variables_descriptives();

    scale_data_minimum_maximum(data_descriptives);

    return data_descriptives;
}


/// Scales the data using the mean and standard deviation method,
/// and the mean and standard deviation values calculated from the data matrix.
/// It also returns the descriptives from all columns.

Tensor<Descriptives, 1> DataSet::scale_data_mean_standard_deviation()
{
    const Tensor<Descriptives, 1> data_descriptives = calculate_variables_descriptives();

    scale_data_mean_standard_deviation(data_descriptives);

    return data_descriptives;
}


void DataSet::scale_minimum_maximum_binary(const type& value_1, const type& value_2,const Index& column_index)
{
    const Index rows_number = data.dimension(0);

    type slope = 0;
    type intercept = 0;

    if(value_1>value_2){
        slope = 1/(value_1-value_2);
        intercept = -value_2/(value_1-value_2);
    }else{
        slope = 1/(value_2-value_1);
        intercept = -value_1/(value_2-value_1);
    }

    for(Index i = 0; i < rows_number; i++)
    {
        data(i, column_index) = slope*data(i, column_index)+intercept;
    }

}

/// Subtracts off the mean to every of the input variables.

void DataSet::subtract_inputs_mean()
{
    Tensor<Descriptives, 1> input_statistics = calculate_input_variables_descriptives();

    Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Index input_index;
    Index sample_index;

    type input_mean;

    for(Index i = 0; i < input_variables_indices.size(); i++)
    {
        input_index = input_variables_indices(i);

        input_mean = input_statistics(i).mean;

        for(Index j = 0; j < used_samples_indices.size(); j++)
        {
            sample_index = used_samples_indices(j);

            data(sample_index,input_index) -= input_mean;
        }
    }
}


/// Returns a vector of strings containing the scaling method that best fits each
/// of the input variables.

Tensor<string, 1> DataSet::calculate_default_scaling_methods() const
{
    const Tensor<Index, 1> used_inputs_indices = get_input_variables_indices();
    const Index used_inputs_number = used_inputs_indices.size();

    Index current_distribution;
    Tensor<string, 1> scaling_methods(used_inputs_number);

    #pragma omp parallel for private(current_distribution)

    for(Index i = 0; i < static_cast<Index>(used_inputs_number); i++)
    {
        current_distribution = perform_distribution_distance_analysis(data.chip(used_inputs_indices(i),1));

        if(current_distribution == 0) // Normal distribution
        {
            scaling_methods(i) = "MeanStandardDeviation";
        }
        else if(current_distribution == 1) // Uniform distribution
        {
            scaling_methods(i) = "MinimumMaximum";
        }
        else // Default
        {
            scaling_methods(i) = "MinimumMaximum";
        }
    }

    return scaling_methods;
}


/// Returns a vector of strings containing the scaling method that best fits each
/// of the target variables.

Tensor<string, 1> DataSet::calculate_default_unscaling_methods() const
{
    const Tensor<Index, 1> used_targets_indices = get_target_variables_indices();
    const Index used_targets_number = used_targets_indices.size();

    Index current_distribution;
    Tensor<string, 1> scaling_methods(used_targets_number);

    #pragma omp parallel for private(current_distribution)

    for(Index i = 0; i < static_cast<Index>(used_targets_number); i++)
    {
        current_distribution = perform_distribution_distance_analysis(data.chip(used_targets_indices(i),1));

        if(current_distribution == 0) // Normal distribution
        {
            scaling_methods(i) = "MeanStandardDeviation";
        }
        else if(current_distribution == 1) // Uniform distribution
        {
            scaling_methods(i) = "MinimumMaximum";
        }
        else // Default
        {
            scaling_methods(i) = "MinimumMaximum";
        }
    }

    return scaling_methods;
}


/// Scales the data matrix with given minimum and maximum values.
/// It updates the data matrix.
/// @param data_descriptives vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.
/// @todo

void DataSet::scale_data_minimum_maximum(const Tensor<Descriptives, 1>& data_descriptives)
{
    const Index variables_number = get_variables_number();

#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    const Index descriptives_size = data_descriptives.size();

    if(descriptives_size != variables_number)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void scale_data_minimum_maximum(const Tensor<Descriptives, 1>&) method.\n"
               << "Size of data descriptives must be equal to number of variables.\n";

        throw logic_error(buffer.str());
    }

#endif

    for(Index i = 0; i < variables_number; i++)
    {
        if(display
                && abs(data_descriptives(i).maximum - data_descriptives(i).minimum) < numeric_limits<type>::min())
        {
            cout << "OpenNN Warning: DataSet class.\n"
                 << "void scale_data_minimum_maximum(const Tensor<Descriptives, 1>&) method.\n"
                 << "Range of variable " <<  i << " is zero.\n"
                 << "That variable won't be scaled.\n";
        }
    }

//       scale_minimum_maximum(data, data_descriptives);
}


/// Scales the given input variables with given mean and standard deviation values.
/// It updates the input variable of the data matrix.
/// @param input_statistics vector of descriptives structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_mean_standard_deviation(const Descriptives& input_statistics, const Index& input_index)
{
    const type slope = (input_statistics.standard_deviation -0) < static_cast<type>(1e-3) ?
                0 :
                static_cast<type>(1)/input_statistics.standard_deviation;

    const type intercept = (input_statistics.standard_deviation -0) < static_cast<type>(1e-3) ?
                0 :
                -static_cast<type>(1)*input_statistics.mean/input_statistics.standard_deviation;

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, input_index) = data(i, input_index)*slope + intercept;
    }
}


/// Scales the given input variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector with the variables descriptives.
/// @param input_index Index of the input to be scaled.

Descriptives DataSet::scale_input_mean_standard_deviation(const Index& input_index)
{
#ifdef __OPENNN_DEBUG__

    if(is_empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Descriptives scale_input_mean_standard_deviation(const Index&) method.\n"
               << "Data file is not loaded.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Descriptives input_statistics = calculate_input_descriptives(input_index);

    scale_input_mean_standard_deviation(input_statistics, input_index);

    return input_statistics;
}


/// Scales the given input variables with given standard deviation values.
/// It updates the input variable of the data matrix.
/// @param inputs_statistics vector of descriptives structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_standard_deviation(const Descriptives& input_statistics, const Index& input_index)
{
    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, input_index) = static_cast<type>(2)*(data(i, input_index)) / input_statistics.standard_deviation;
    }
}


/// Scales the given input variables with the calculated standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector with the variables descriptives.
/// @param input_index Index of the input to be scaled.

Descriptives DataSet::scale_input_standard_deviation(const Index& input_index)
{
#ifdef __OPENNN_DEBUG__

    if(is_empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Descriptives scale_input_standard_deviation(const Index&) method.\n"
               << "Data file is not loaded.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Descriptives input_statistics = calculate_input_descriptives(input_index);

    scale_input_standard_deviation(input_statistics, input_index);

    return input_statistics;
}


/// Scales the given input variable with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param input_statistics vector with the descriptives of the input variable.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_minimum_maximum(const Descriptives& input_statistics, const Index& input_index)
{
    const type slope = std::abs(input_statistics.maximum-input_statistics.minimum) < static_cast<type>(1e-3) ?
                0 :
                (max_range-min_range)/(input_statistics.maximum-input_statistics.minimum);

    const type intercept = std::abs(input_statistics.maximum-input_statistics.minimum) < static_cast<type>(1e-3) ?
                0 :
                (min_range*input_statistics.maximum-max_range*input_statistics.minimum)/(input_statistics.maximum-input_statistics.minimum);

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, input_index) = data(i, input_index)*slope + intercept;
    }
}


/// Scales the given input variable with the calculated minimum and maximum values from the data matrix.
/// It updates the input variable of the data matrix.
/// It also returns a vector with the minimum and maximum values of the input variables.

Descriptives DataSet::scale_input_minimum_maximum(const Index& input_index)
{
#ifdef __OPENNN_DEBUG__

    if(is_empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Descriptives scale_input_minimum_maximum(const Index&) method.\n"
               << "Data file is not loaded.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Descriptives input_statistics = calculate_input_descriptives(input_index);

    scale_input_minimum_maximum(input_statistics, input_index);

    return input_statistics;
}


void DataSet::scale_input_variables_minimum_maximum(const Tensor<Descriptives, 1>& inputs_descriptives)
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index input_variables_number = input_variables_indices.size();

    for(Index i = 0; i < input_variables_number; i++)
    {
        scale_input_minimum_maximum(inputs_descriptives[i], input_variables_indices[i]);
    }
}


Tensor<Descriptives, 1> DataSet::scale_input_variables_minimum_maximum()
{
    const Tensor<Descriptives, 1> inputs_descriptives = calculate_input_variables_descriptives();

    scale_input_variables_minimum_maximum(inputs_descriptives);

    return inputs_descriptives;

}


void DataSet::unscale_input_variables_minimum_maximum(const Tensor<Descriptives, 1>& inputs_descriptives)
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index input_variables_number = input_variables_indices.size();

    for(Index i = 0; i < input_variables_number; i++)
    {
        unscale_input_variable_minimum_maximum(inputs_descriptives[i], input_variables_indices[i]);
    }
}


/// It scales every input variable with the given method.
/// The method to be used is that in the scaling and unscaling method variable.

Tensor<Descriptives, 1> DataSet::scale_input_variables(const Tensor<string, 1>& scaling_unscaling_methods)
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Tensor<Descriptives, 1> inputs_descriptives = calculate_input_variables_descriptives();

    for(Index i = 0; i < scaling_unscaling_methods.dimension(0); i++)
    {
        switch(get_scaling_unscaling_method(scaling_unscaling_methods(i)))
        {
        case NoScaling:
        {
            // Do nothing
        }
        break;

        case MinimumMaximum:
        {
            scale_input_minimum_maximum(inputs_descriptives(i), input_variables_indices(i));
        }
        break;

        case MeanStandardDeviation:
        {
            scale_input_mean_standard_deviation(inputs_descriptives(i), input_variables_indices(i));
        }
        break;

        case StandardDeviation:
        {
            scale_input_standard_deviation(inputs_descriptives(i), input_variables_indices(i));
        }
        break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void scale_input_variables(const Tensor<string, 1>&, const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown scaling and unscaling method: " << scaling_unscaling_methods(i) << "\n";

            throw logic_error(buffer.str());
        }
        }
    }

    return inputs_descriptives;
}


/// Scales the target variables with given mean and standard deviation values.
/// It updates the target variables of the data matrix.
/// @param targets_descriptives vector of descriptives structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_target_variables_mean_standard_deviation(const Tensor<Descriptives, 1>& targets_descriptives)
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Index target_variables_number = target_variables_indices.size();

    Index variable_index;

    for(Index i = 0; i < data.dimension(0); i++)
    {
        for(Index j = 0; j < target_variables_number; j++)
        {
            variable_index = target_variables_indices(j);

            if(!::isnan(data(i,variable_index)))
            {
                data(i, variable_index) =
                        (data(i, variable_index)-targets_descriptives(j).mean)/(targets_descriptives(j).standard_deviation);
            }
        }
    }
}


/// Scales the target variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of descriptives structures with the basic descriptives of all the variables.

Tensor<Descriptives, 1> DataSet::scale_target_variables_mean_standard_deviation()
{
#ifdef __OPENNN_DEBUG__

    if(is_empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Descriptives, 1> scale_target_variables_mean_standard_deviation() method.\n"
               << "Data file is not loaded.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Tensor<Descriptives, 1> targets_descriptives = calculate_target_variables_descriptives();

    scale_target_variables_mean_standard_deviation(targets_descriptives);

    return targets_descriptives;
}


/// Scales the target variables with given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param targets_descriptives vector of descriptives structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_target_variables_minimum_maximum(const Tensor<Descriptives, 1>& targets_descriptives)
{
#ifdef __OPENNN_DEBUG__

    if(is_empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Descriptives, 1> scale_target_variables_minimum_maximum() method.\n"
               << "Data file is not loaded.\n";

        throw logic_error(buffer.str());
    }

#endif

//    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
//    const Index target_variables_number = target_variables_indices.size();

//    Index variable_index;

//    for(Index i = 0; i < data.dimension(0); i++)
//    {
//        for(Index j = 0; j < target_variables_number; j++)
//        {
//            variable_index = target_variables_indices(j);

//            if(!::isnan(data(i,variable_index)))
//            {
//                data(i, variable_index) =
//                        static_cast<type>(2.0)*(data(i, variable_index)-targets_descriptives(j).minimum)/(targets_descriptives(j).maximum-targets_descriptives(j).minimum)-static_cast<type>(1.0);
//            }
//        }
//    }

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Index target_variables_number = target_variables_indices.size();

    for(Index i = 0; i < target_variables_number; i++)
    {
        scale_target_minimum_maximum(targets_descriptives[i], target_variables_indices[i]);
    }
}


/// Scales the target variables with the calculated minimum and maximum values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of vectors with the descriptives of the input target variables.

Tensor<Descriptives, 1> DataSet::scale_target_variables_minimum_maximum()
{
    const Tensor<Descriptives, 1> targets_descriptives = calculate_target_variables_descriptives();

    scale_target_variables_minimum_maximum(targets_descriptives);

    return targets_descriptives;
}


/// Scales the target variables with the logarithmic scale using the given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param targets_descriptives vector of descriptives structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_target_variables_logarithm(const Tensor<Descriptives, 1>& targets_descriptives)
{
#ifdef __OPENNN_DEBUG__

    if(is_empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Descriptives, 1> scale_target_variables_logarithm() method.\n"
               << "Data file is not loaded.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Index target_variables_number = target_variables_indices.size();

    Index variable_index;

    for(Index i = 0; i < data.dimension(0); i++)
    {
        for(Index j = 0; j < target_variables_number; j++)
        {
            variable_index = target_variables_indices(j);

            if(!::isnan(data(i,variable_index)))
            {
                data(i, variable_index) =
                        static_cast<type>(0.5)*(exp(data(i, variable_index)))*(targets_descriptives(j).maximum-targets_descriptives(j).minimum)+ targets_descriptives(j).minimum;
            }
        }
    }
}


/// Scales the target variables with the logarithmic scale using the calculated minimum and maximum values
/// from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of vectors with the descriptives of the input target variables.

Tensor<Descriptives, 1> DataSet::scale_target_variables_logarithm()
{
    const Tensor<Descriptives, 1> targets_descriptives = calculate_target_variables_descriptives();

    scale_target_variables_logarithm(targets_descriptives);

    return targets_descriptives;
}


/// Calculates the input and target variables descriptives.
/// Then it scales the target variables with those values.
/// The method to be used is that in the scaling and unscaling method variable.
/// Finally, it returns the descriptives.

Tensor<Descriptives, 1> DataSet::scale_target_variables(const string& scaling_unscaling_method)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
    {
    case NoUnscaling:
    {
        return calculate_target_variables_descriptives();
    }

    case MinimumMaximum:
    {
        return scale_target_variables_minimum_maximum();
    }

    case Logarithmic:
    {
        return scale_target_variables_logarithm();
    }

    case MeanStandardDeviation:
    {
        return scale_target_variables_mean_standard_deviation();
    }

    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "Tensor<Descriptives, 1> scale_target_variables(const string&) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw logic_error(buffer.str());
    }
    }
}


void DataSet::scale_target_minimum_maximum(const Descriptives& target_statistics, const Index& target_index)
{
    const type slope = std::abs(target_statistics.maximum-target_statistics.minimum) < static_cast<type>(1e-3) ?
                0 :
                (max_range-min_range)/(target_statistics.maximum-target_statistics.minimum);

    const type intercept = std::abs(target_statistics.maximum-target_statistics.minimum) < static_cast<type>(1e-3) ?
                0 :
                (min_range*target_statistics.maximum-max_range*target_statistics.minimum)/(target_statistics.maximum-target_statistics.minimum);

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, target_index) = data(i, target_index)*slope + intercept;
    }
}


void DataSet::scale_target_mean_standard_deviation(const Descriptives& target_statistics, const Index& target_index)
{
    const type slope = std::abs(target_statistics.standard_deviation-0) < static_cast<type>(1e-3) ?
                0 :
                static_cast<type>(1)/target_statistics.standard_deviation;

    const type intercept = std::abs(target_statistics.standard_deviation-0) < static_cast<type>(1e-3) ?
                0 :
                -target_statistics.mean/target_statistics.standard_deviation;

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, target_index) = data(i, target_index)*slope + intercept;
    }
}


void DataSet::scale_target_logarithmic(const Descriptives& target_statistics, const Index& target_index)
{
    for(Index i = 0; i < data.dimension(0); i++)
    {
        if(std::abs(target_statistics.standard_deviation-0) < static_cast<type>(1e-3))
        {
            data(i, target_index) = 0;
        }
        else
        {
            data(i, target_index) = static_cast<type>(0.5)*(exp(data(i,target_index)-1))*(target_statistics.maximum-target_statistics.minimum) + target_statistics.minimum;
        }
    }
}


/// It scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

Tensor<Descriptives, 1> DataSet::scale_target_variables(const Tensor<string, 1>& scaling_unscaling_methods)
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Tensor<Descriptives, 1> targets_descriptives = calculate_target_variables_descriptives();

//    Index column_index;

    for (Index i = 0; i < scaling_unscaling_methods.size(); i++)
    {
//        column_index = get_column_index(target_variables_indices(i));

//        if(columns(column_index).type == Binary || columns(column_index).type == Categorical) continue;

        switch(get_scaling_unscaling_method(scaling_unscaling_methods(i)))
        {
        case NoUnscaling:
            break;

        case MinimumMaximum:
            scale_target_minimum_maximum(targets_descriptives(i), target_variables_indices(i));
            break;

        case MeanStandardDeviation:
            scale_target_mean_standard_deviation(targets_descriptives(i), target_variables_indices(i));
            break;

        case Logarithmic:
            scale_target_logarithmic(targets_descriptives(i), target_variables_indices(i));
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void scale_target_variables(const string&, const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown scaling and unscaling method.\n";

            throw logic_error(buffer.str());
        }
        }
    }
    return targets_descriptives;
}


/// Unscales the given input variable with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param input_statistics vector with the descriptives of the input variable.
/// @param input_index Index of the input to be scaled.

void DataSet::unscale_input_variable_minimum_maximum(const Descriptives& input_statistics, const Index & input_index)
{
    const type slope = std::abs(max_range-min_range) < static_cast<type>(1e-3) ? 0 : (input_statistics.maximum-input_statistics.minimum)/(max_range-min_range);

    const type intercept = std::abs(max_range-min_range) < static_cast<type>(1e-3) ? 0 : -(min_range*input_statistics.maximum-max_range*input_statistics.minimum)/(max_range-min_range);

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, input_index) = data(i, input_index)*slope + intercept;
    }
}


/// Uncales the given input variables with given mean and standard deviation values.
/// It updates the input variable of the data matrix.
/// @param input_statistics vector of descriptives structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::unscale_input_mean_standard_deviation(const Descriptives& input_statistics, const Index& input_index)
{
    const type slope = std::abs(input_statistics.mean - 0) < static_cast<type>(1e-3) ? 0 : input_statistics.standard_deviation/static_cast<type>(2);

    const type intercept = std::abs(input_statistics.mean-0) < static_cast<type>(1e-3) ? input_statistics.minimum : input_statistics.mean;

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, input_index) = data(i, input_index)*slope + intercept;
    }
}


/// Unscales the given input variables with given standard deviation values.
/// It updates the input variable of the data matrix.
/// @param inputs_statistics vector of descriptives structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::unscale_input_variable_standard_deviation(const Descriptives& input_statistics, const Index& input_index)
{
    const type slope = std::abs(input_statistics.mean-0) < static_cast<type>(1e-3) ? 0 : input_statistics.standard_deviation/static_cast<type>(2);

    const type intercept = std::abs(input_statistics.mean-0) < static_cast<type>(1e-3) ? input_statistics.minimum : 0;

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, input_index) = data(i, input_index)*slope + intercept;
    }
}


/// It unscales every input variable with the given method.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::unscale_input_variables(const Tensor<string, 1>& scaling_unscaling_methods, const Tensor<Descriptives, 1>& inputs_descriptives)
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    for(Index i = 0; i < scaling_unscaling_methods.size(); i++)
    {
        switch(get_scaling_unscaling_method(scaling_unscaling_methods(i)))
        {
        case NoScaling:
        {
            // Do nothing
        }
        break;

        case MinimumMaximum:
        {
            unscale_input_variable_minimum_maximum(inputs_descriptives(i), input_variables_indices(i));
        }
        break;

        case MeanStandardDeviation:
        {
            unscale_input_mean_standard_deviation(inputs_descriptives(i), input_variables_indices(i));
        }
        break;

        case StandardDeviation:
        {
            unscale_input_variable_standard_deviation(inputs_descriptives(i), input_variables_indices(i));
        }
        break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void unscale_input_variables(const Tensor<string, 1>&, const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown unscaling and unscaling method: " << scaling_unscaling_methods(i) << "\n";

            throw logic_error(buffer.str());
        }
        }
    }
}


void DataSet::unscale_target_minimum_maximum(const Descriptives& target_statistics, const Index& target_index)
{
    const type slope = std::abs(max_range-min_range) < static_cast<type>(1e-3) ? 0 : (target_statistics.maximum-target_statistics.minimum)/(max_range-min_range);

    const type intercept = std::abs(max_range-min_range) < static_cast<type>(1e-3) ? 0 : -(min_range*target_statistics.maximum-max_range*target_statistics.minimum)/(max_range-min_range);

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, target_index) = data(i, target_index)*slope + intercept;
    }
}


void DataSet::unscale_target_mean_standard_deviation(const Descriptives& target_statistics, const Index& target_index)
{
    const type slope = std::abs(target_statistics.standard_deviation-0) < static_cast<type>(1e-3) ?
                0 :
                target_statistics.standard_deviation/static_cast<type>(2);

    const type intercept = target_statistics.mean;

    for(Index i = 0; i < data.dimension(0); i++)
    {
        data(i, target_index) = data(i, target_index)*slope + intercept;
    }
}


void DataSet::unscale_target_logarithmic(const Descriptives& target_statistics, const Index& target_index)
{
    for(Index i = 0; i < data.dimension(0); i++)
    {
        if(std::abs(target_statistics.maximum - target_statistics.minimum) < static_cast<type>(1e-3))
        {
            data(i, target_index) = target_statistics.minimum;
        }
        else
        {
            data(i, target_index) = log(static_cast<type>(2)*(data(i,target_index)-target_statistics.minimum)/(target_statistics.maximum-target_statistics.minimum));
        }
    }
}


/// It unscales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::unscale_target_variables(const Tensor<string, 1>& scaling_unscaling_methods, const Tensor<Descriptives, 1>& targets_descriptives)
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    for (Index i = 0; i < scaling_unscaling_methods.size(); i++)
    {
        switch(get_scaling_unscaling_method(scaling_unscaling_methods(i)))
        {
        case NoUnscaling:
            break;

        case MinimumMaximum:
            unscale_target_minimum_maximum(targets_descriptives(i), target_variables_indices(i));
            break;

        case MeanStandardDeviation:
            unscale_target_mean_standard_deviation(targets_descriptives(i), target_variables_indices(i));
            break;

        case Logarithmic:
            unscale_target_logarithmic(targets_descriptives(i), target_variables_indices(i));
            break;

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "void unscale_targets(const string&, const Tensor<Descriptives, 1>&) method.\n"
                   << "Unknown unscaling and unscaling method.\n";

            throw logic_error(buffer.str());
        }
        }
    }
}




/// Initializes the data matrix with a given value.
/// @param new_value Initialization value.

void DataSet::initialize_data(const type& new_value)
{
    data.setConstant(new_value);
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

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = input_variables_number; j < variables_number; j++)
        {
            data(i,j) = (1+static_cast<type>(pow((-1),rand())))/2;
        }
    }
}


/// Sets max and min scaling range for minmaxscaling.
/// @param min and max for scaling range.

void DataSet::set_min_max_range(const type min, const type max)
{
    min_range = min;
    max_range = max;
}

/// Serializes the data set object into a XML document of the TinyXML library without keep the DOM tree in memory.

void DataSet::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataFile");

    // File type ?

    {
        file_stream.OpenElement("FileType");

        file_stream.PushText("csv");

        file_stream.CloseElement();
    }

    // Data file name
    {
        file_stream.OpenElement("DataFileName");

        file_stream.PushText(data_file_name.c_str());

        file_stream.CloseElement();
    }

    // Separator
    {
        file_stream.OpenElement("Separator");

        file_stream.PushText(get_separator_string().c_str());

        file_stream.CloseElement();
    }

    // Columns names
    {
        file_stream.OpenElement("ColumnsNames");

        buffer.str("");
        buffer << has_columns_names;

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

    // Lags number
    {
        file_stream.OpenElement("LagsNumber");

        buffer.str("");
        buffer << get_lags_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Steps Ahead
    {
        file_stream.OpenElement("StepsAhead");

        buffer.str("");
        buffer << get_steps_ahead();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Time Index
    {
        file_stream.OpenElement("TimeIndex");

        buffer.str("");
        buffer << get_time_index();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }
    // Close DataFile

    file_stream.CloseElement();

    // Columns

    file_stream.OpenElement("Columns");

    // Columns number
    {
        file_stream.OpenElement("ColumnsNumber");

        buffer.str("");
        buffer << get_columns_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Columns items

    {
        const Index columns_number = get_columns_number();

        for(Index i = 0; i < columns_number; i++)
        {
            file_stream.OpenElement("Column");

            file_stream.PushAttribute("Item", to_string(i+1).c_str());

            columns(i).write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close columns

    file_stream.CloseElement();

    // Rows labels

    if(has_rows_labels)
    {
        const Index rows_labels_number = rows_labels.dimension(0);

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
            buffer << samples_uses(i);

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

        if(missing_values_method == Mean)
        {
            file_stream.PushText("Mean");
        }
        else if(missing_values_method == Median)
        {
            file_stream.PushText("Median");
        }
        else
        {
            file_stream.PushText("Unuse");
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
        // Columns missing values number

        {
            file_stream.OpenElement("ColumnsMissingValuesNumber");

            cout << "count nan columns" << endl;
            const Index columns_number = columns_missing_values_number.size();

            buffer.str("");

            for (Index i = 0; i < columns_number; i++)
            {
                buffer << columns_missing_values_number(i);

                if(i != (columns_number-1)) buffer << " ";
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
}


void DataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    ostringstream buffer;

    // Data set element

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data set element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Data file

    const tinyxml2::XMLElement* data_file_element = data_set_element->FirstChildElement("DataFile");

    if(!data_file_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Data file element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Data file name

    const tinyxml2::XMLElement* data_file_name_element = data_file_element->FirstChildElement("DataFileName");

    if(!data_file_name_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "DataFileName element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(data_file_name_element->GetText())
    {
        const string new_data_file_name = data_file_name_element->GetText();

        set_data_file_name(new_data_file_name);
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

    // Has columns names

    const tinyxml2::XMLElement* columns_names_element = data_file_element->FirstChildElement("ColumnsNames");

    if(columns_names_element)
    {
        const string new_columns_names_string = columns_names_element->GetText();

        try
        {
            set_has_columns_names(new_columns_names_string == "1");
        }
        catch(const logic_error& e)
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
        catch(const logic_error& e)
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

    // Forecasting

    // Lags number

    const tinyxml2::XMLElement* lags_number_element = data_file_element->FirstChildElement("LagsNumber");

    if(!lags_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Lags number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(lags_number_element->GetText())
    {
        const Index new_lags_number = static_cast<Index>(atoi(lags_number_element->GetText()));

        set_lags_number(new_lags_number);
    }

    // Steps ahead

    const tinyxml2::XMLElement* steps_ahead_element = data_file_element->FirstChildElement("StepsAhead");

    if(!steps_ahead_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Steps ahead element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(steps_ahead_element->GetText())
    {
        const Index new_steps_ahead = static_cast<Index>(atoi(steps_ahead_element->GetText()));

        set_steps_ahead_number(new_steps_ahead);
    }

    // Time index

    const tinyxml2::XMLElement* time_index_element = data_file_element->FirstChildElement("TimeIndex");

    if(!time_index_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Time index element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(time_index_element->GetText())
    {
        const Index new_time_index = static_cast<Index>(atoi(time_index_element->GetText()));

        set_time_index(new_time_index);
    }

    // Columns

    const tinyxml2::XMLElement* columns_element = data_set_element->FirstChildElement("Columns");

    if(!columns_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Columns element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Columns number

    const tinyxml2::XMLElement* columns_number_element = columns_element->FirstChildElement("ColumnsNumber");

    if(!columns_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Columns number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    Index new_columns_number = 0;

    if(columns_number_element->GetText())
    {
        new_columns_number = static_cast<Index>(atoi(columns_number_element->GetText()));

        set_columns_number(new_columns_number);
    }

    // Columns

    const tinyxml2::XMLElement* start_element = columns_number_element;

    if(new_columns_number > 0)
    {
        for(Index i = 0; i < new_columns_number; i++)
        {
            const tinyxml2::XMLElement* column_element = start_element->NextSiblingElement("Column");
            start_element = column_element;

            if(column_element->Attribute("Item") != std::to_string(i+1))
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet:from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column item number (" << i+1 << ") does not match (" << column_element->Attribute("Item") << ").\n";

                throw logic_error(buffer.str());
            }

            // Name

            const tinyxml2::XMLElement* name_element = column_element->FirstChildElement("Name");

            if(!name_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Name element is nullptr.\n";

                throw logic_error(buffer.str());
            }

            if(name_element->GetText())
            {
                const string new_name = name_element->GetText();

                columns(i).name = new_name;
            }

            // Column use

            const tinyxml2::XMLElement* column_use_element = column_element->FirstChildElement("ColumnUse");

            if(!column_use_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void DataSet::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Column use element is nullptr.\n";

                throw logic_error(buffer.str());
            }

            if(column_use_element->GetText())
            {
                const string new_column_use = column_use_element->GetText();

                columns(i).set_use(new_column_use);
            }

            // Type

            const tinyxml2::XMLElement* type_element = column_element->FirstChildElement("Type");

            if(!type_element)
            {
                buffer << "OpenNN Exception: DataSet class.\n"
                       << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                       << "Type element is nullptr.\n";

                throw logic_error(buffer.str());
            }

            if(type_element->GetText())
            {
                const string new_type = type_element->GetText();
                columns(i).set_type(new_type);
            }

            if(columns(i).type == Categorical || columns(i).type == Binary)
            {
                // Categories

                const tinyxml2::XMLElement* categories_element = column_element->FirstChildElement("Categories");

                if(!categories_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories element is nullptr.\n";

                    throw logic_error(buffer.str());
                }

                if(categories_element->GetText())
                {
                    const string new_categories = categories_element->GetText();

                    columns(i).categories = get_tokens(new_categories, ';');
                }

                // Categories uses

                const tinyxml2::XMLElement* categories_uses_element = column_element->FirstChildElement("CategoriesUses");

                if(!categories_uses_element)
                {
                    buffer << "OpenNN Exception: DataSet class.\n"
                           << "void Column::from_XML(const tinyxml2::XMLDocument&) method.\n"
                           << "Categories uses element is nullptr.\n";

                    throw logic_error(buffer.str());
                }

                if(categories_uses_element->GetText())
                {
                    const string new_categories_uses = categories_uses_element->GetText();

                    columns(i).set_categories_uses(get_tokens(new_categories_uses, ';'));
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

            throw logic_error(buffer.str());
        }

        // Rows labels

        if(rows_labels_element->GetText())
        {
            const string new_rows_labels = rows_labels_element->GetText();

            rows_labels = get_tokens(new_rows_labels, ',');
        }


    }

    // Samples

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(samples_number_element->GetText())
    {
        const Index new_samples_number = static_cast<Index>(atoi(samples_number_element->GetText()));

        samples_uses.resize(new_samples_number);
    }

    // Samples uses

    const tinyxml2::XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Samples uses element is nullptr.\n";

        throw logic_error(buffer.str());
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

        throw logic_error(buffer.str());
    }

    // Missing values method

    const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Missing values method element is nullptr.\n";

        throw logic_error(buffer.str());
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

        throw logic_error(buffer.str());
    }

    if(missing_values_number_element->GetText())
    {
        missing_values_number = static_cast<Index>(atoi(missing_values_number_element->GetText()));
    }

    if(missing_values_number > 0)
    {
        // Columns Missing values number

        const tinyxml2::XMLElement* columns_missing_values_number_element = missing_values_element->FirstChildElement("ColumnsMissingValuesNumber");

        if(!columns_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Columns missing values number element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(columns_missing_values_number_element->GetText())
        {
            Tensor<string, 1> new_columns_missing_values_number = get_tokens(columns_missing_values_number_element->GetText(), ' ');

            columns_missing_values_number.resize(new_columns_missing_values_number.size());

            for(Index i = 0; i < new_columns_missing_values_number.size(); i++)
            {
                columns_missing_values_number(i) = atoi(new_columns_missing_values_number(i).c_str());
            }
        }

        // Rows missing values number

        const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Rows missing values number element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(rows_missing_values_number_element->GetText())
        {
            rows_missing_values_number = static_cast<Index>(atoi(rows_missing_values_number_element->GetText()));
        }
    }

    // Preview data

    const tinyxml2::XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview data element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Preview size

    const tinyxml2::XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Preview size element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    Index new_preview_size = 0;

    if(preview_size_element->GetText())
    {
        new_preview_size = static_cast<Index>(atoi(preview_size_element->GetText()));

        if(new_preview_size > 0) data_file_preview.resize(new_preview_size);
    }

    // Preview data

    start_element = preview_size_element;

    for(Index i = 0; i < new_preview_size; i++)
    {
        const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
        start_element = row_element;

        if(row_element->Attribute("Item") != std::to_string(i+1))
        {
            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Row item number (" << i+1 << ") does not match (" << row_element->Attribute("Item") << ").\n";

            throw logic_error(buffer.str());
        }

        if(row_element->GetText())
        {
            data_file_preview(i) = get_tokens(row_element->GetText(), ',');
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
        catch(const logic_error& e)
        {
            cerr << e.what() << endl;
        }
    }
}


/// Prints to the screen in text format the main numbers from the data set object.

void DataSet::print_summary() const
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
///
/// @todo

void DataSet::save(const string& file_name) const
{
    FILE *pFile;
//    int err;

//    err = fopen_s(&pFile, file_name.c_str(), "w");
    pFile = fopen(file_name.c_str(), "w");

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

        throw logic_error(buffer.str());
    }

    from_XML(document);
}


void DataSet::print_columns_types() const
{
    const Index columns_number = get_columns_number();

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).type == Numeric) cout << "Numeric ";
        else if(columns(i).type == Binary) cout << "Binary ";
        else if(columns(i).type == Categorical) cout << "Categorical ";
        else if(columns(i).type == DateTime) cout << "DateTime ";
        else if(columns(i).type == Constant) cout << "Constant ";

    }

    cout << endl;
}


/// Prints to the screen the values of the data matrix.

void DataSet::print_data() const
{
    if(display) cout << data << endl;
}


/// Prints to the sceen a preview of the data matrix,
/// i.e., the first, second and last samples

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
    ofstream file(data_file_name.c_str());

    if(!file.is_open())
    {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template." << endl
             << "void save_csv(const string&, const char&, const Vector<string>&, const Vector<string>&) method." << endl
             << "Cannot open matrix data file: " << data_file_name << endl;

      throw logic_error(buffer.str());
    }

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    const Tensor<string, 1> variables_names = get_variables_names();

    char separator_char = ',';//get_separator_char();

    if(this->has_rows_labels)
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
        if(this->has_rows_labels)
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
    ofstream file(binary_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet template." << endl
               << "void save_data_binary(const string) method." << endl
               << "Cannot open data binary file." << endl;

        throw logic_error(buffer.str());
    }

    // Write data

    streamsize size = sizeof(Index);

    Index columns_number = data.dimension(1);
    Index rows_number = data.dimension(0);

    cout << "Rows number: " << rows_number << endl;
    cout << "Columns number: " << columns_number << endl;

    cout << "Saving binary data file..." << endl;

    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    for(int i = 0; i < columns_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
            value = data(j,i);

            file.write(reinterpret_cast<char*>(&value), size);
        }
    }

    file.close();


/*
    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    for(int i = 0; i < columns_number*rows_number; i++)
    {
//        for(int j = 0; j < rows_number; j++)
//        {
            value = data(i);

            file.write(reinterpret_cast<char*>(&value), size);
//        }
    }

    file.close();
*/


    cout << "Binary data file saved." << endl;
}


/// Arranges an input-target DataSet from a time series matrix, according to the number of lags.

void DataSet::transform_time_series()
{
    if(lags_number == 0) return;

    const Index variables_number = get_variables_number();
    const Index samples_number = get_samples_number();

    time_series_data = data;

    time_series_columns = columns;

    transform_time_series_columns();

    const Index time_series_samples_number = get_samples_number()-(lags_number-1+steps_ahead);
    const Index time_series_variables_number = get_columns_number();

    data.resize(time_series_samples_number, time_series_variables_number);

    Tensor<type, 2> new_data(time_series_samples_number, time_series_variables_number);
    Tensor<type, 1> variable_data;

    Index new_data_variable = 0;

    Index time_series_variable= 0;


// lags

    for(Index lag = lags_number; lag > 0; lag--)
    {

        for(Index variable = 0; variable < variables_number; variable++)
            {

            variable_data = time_series_data.chip(variable, 1);

            for(Index j = 0; j <= time_series_samples_number; j++)
            {

                new_data(j, time_series_variable) = variable_data(j+lags_number-lag);
            }
            time_series_variable++;
        }
    }

// steps ahead
    for(Index ahead = 1; ahead <= steps_ahead; ahead++)
    {
        for(Index variable = 0; variable < variables_number; variable++)
            {
            variable_data = time_series_data.chip(variable, 1);

            for(Index j = 0; j < time_series_samples_number; j++)
            {
                new_data(j, time_series_variable) = variable_data(j+ahead+lags_number-1);
            }

            time_series_variable++;
        }
    }

    set_data(new_data);
}


/// Arranges the data set for association.
/// @todo Low priority. Variables and samples.

void DataSet::transform_association()
{
// OpenNN::transform_association(data);
}


/// @todo

void DataSet::fill_time_series(const Index& period )
{
    Index rows = static_cast<Index>((data(data.dimension(0)- 1, 0)- data(0,0)) / period) + 1 ;

    Tensor<type, 2> new_data(rows, data.dimension(1));

    new_data.setConstant(static_cast<type>(NAN));

    Index j = 1;

//    new_data.set_row(0, data.chip(0, 0));

    cout.precision(20);

    for (Index i = 1; i < rows ; i++)
    {
      if(static_cast<Index>(data(j, 0)) == static_cast<Index>(data(j - 1, 0)))
      {

          j = j + 1;
      }
      if(static_cast<Index>(data(j, 0)) == static_cast<Index>(data(0,0) + i * period))
      {
//          new_data.set_row(i, data.chip(j, 0));

          j = j + 1;
      }
      else
      {
          new_data(i,0) = data(0,0) + i * period;
      }
    }

    time_series_data = new_data;

    data = new_data;
}


/// This method loads the data from a binary data file.

void DataSet::load_data_binary()
{
    ifstream file;

    file.open(data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet template.\n"
               << "void load_binary(const string&) method.\n"
               << "Cannot open binary file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    streamsize size = sizeof(Index);

    Index columns_number;
    Index rows_number;

    file.read(reinterpret_cast<char*>(&columns_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    data.resize(rows_number, columns_number);

//    Index row_index = 0;
//    Index column_index = 0;

    for(Index i = 0; i < rows_number*columns_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        data(i) = value;
/*
        data(row_index, column_index) = value;

        row_index++;

        if((i+1)%rows_number == 0)
        {
            row_index = 0;
            column_index++;
        }
*/
    }

    file.close();
}


/// This method loads data from a binary data file for time series prediction methodata_set.
/// @todo

void DataSet::load_time_series_data_binary()
{
//    time_series_data.load_binary(data_file_name);
}


/// This method checks if the input data file has the correct format. Returns an error message.

void DataSet::check_input_csv(const string & input_data_file_name, const char & separator_char) const
{
    ifstream file(input_data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void check_input_csv() method.\n"
               << "Cannot open input data file: " << input_data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    string line;
    Index line_number = 0;
    Index total_lines = 0;

    Index tokens_count;

    const Index columns_number = get_columns_number() - get_target_columns_number();

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        total_lines++;

        tokens_count = count_tokens(line, separator_char);

        if(tokens_count != columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void check_input_csv() method.\n"
                   << "Line " << line_number << ": Size of tokens in input file ("
                   << tokens_count << ") is not equal to number of columns("
                   << columns_number << "). \n"
                   << "Input csv must contain values for all the variables except the target. \n";

            throw logic_error(buffer.str());
        }
    }

    file.close();

    if(total_lines == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void check_input_csv() method.\n"
               << "Input data file is empty. \n";

        throw logic_error(buffer.str());
    }
}


/// This method loads data from a file and returns a matrix containing the input columns.

Tensor<type, 2> DataSet::read_input_csv(const string& input_data_file_name,
                                        const char& separator_char,
                                        const string& missing_values_label,
                                        const bool& has_columns_name,
                                        const bool& has_rows_label) const
{
    ifstream file(input_data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_input_csv() method.\n"
               << "Cannot open input data file: " << input_data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    // Count samples number

    Index input_samples_count = 0;

    string line;
    Index line_number = 0;

    Index tokens_count;

    const Index columns_number = get_columns_number() - get_target_columns_number();

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens_count = count_tokens(line, separator_char);

        if(tokens_count != columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void read_input_csv() method.\n"
                   << "Line " << line_number << ": Size of tokens("
                   << tokens_count << ") is not equal to number of columns("
                   << columns_number << ").\n";

            throw logic_error(buffer.str());
        }

        input_samples_count++;
    }

    file.close();

    Index variables_number = get_input_variables_number();

    if(has_columns_name) input_samples_count--;

    Tensor<type, 2> input_data(input_samples_count, variables_number);

    // Fill input data

    file.open(input_data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_input_csv() method.\n"
               << "Cannot open input data file: " << input_data_file_name << " for filling input data file. \n";

        throw logic_error(buffer.str());
    }

    // Read first line

    if(has_columns_name)
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

        for(Index i = 0; i < columns.size(); i++)
        {
            if(is_ID)
            {
                is_ID = false;
                continue;
            }

            if(columns(i).column_use == UnusedVariable)
            {
                token_index++;
                continue;
            }
            else if(columns(i).column_use != Input)
            {
                continue;
            }

            if(columns(i).type == Numeric)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = static_cast<type>(NAN);
                }
                else if(is_float)
                {
                    input_data(line_number, variable_index) = strtof(tokens(token_index).data(), NULL);
                }
                else
                {
                    input_data(line_number, variable_index) = stof(tokens(token_index));
                }

                variable_index++;
            }
            else if(columns(i).type == Binary)
            {
                if(tokens(token_index) == missing_values_label)
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = static_cast<type>(NAN);
                }
                else if(columns(i).categories.size() > 0 && tokens(token_index) == columns(i).categories(0))
                {
                    input_data(line_number, variable_index) = 1.0;
                }
                else if(tokens(token_index) == columns(i).name)
                {
                    input_data(line_number, variable_index) = 1.0;
                }

                variable_index++;
            }
            else if(columns(i).type == Categorical)
            {
                for(Index k = 0; k < columns(i).get_categories_number(); k++)
                {
                    if(tokens(token_index) == missing_values_label)
                    {
                        has_missing_values = true;
                        input_data(line_number, variable_index) = static_cast<type>(NAN);
                    }
                    else if(tokens(token_index) == columns(i).categories(k))
                    {
                        input_data(line_number, variable_index) = 1.0;
                    }

                    variable_index++;
                }
            }
            else if(columns(i).type == DateTime)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = static_cast<type>(NAN);
                }
                else
                {
                    input_data(line_number, variable_index) = static_cast<type>(date_to_timestamp(tokens(token_index), gmt));
                }

                variable_index++;
            }
            else if(columns(i).type == Constant)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = static_cast<type>(NAN);
                }
                else if(is_float)
                {
                    input_data(line_number, variable_index) = strtof(tokens(token_index).data(), NULL);
                }
                else
                {
                    input_data(line_number, variable_index) = stof(tokens(token_index));
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
        return input_data;
    }
    else
    {
        // Scrub missing values

        const MissingValuesMethod missing_values_method = get_missing_values_method();

        if(missing_values_method == MissingValuesMethod::Unuse || missing_values_method == MissingValuesMethod::Mean)
        {
            const Tensor<type, 1> means = mean(input_data);

            const Index samples_number = input_data.dimension(0);
            const Index variables_number = input_data.dimension(1);

        #pragma omp parallel for schedule(dynamic)

            for(Index j = 0; j < variables_number; j++)
            {
                for(Index i = 0; i < samples_number; i++)
                {
                    if(::isnan(input_data(i, j)))
                    {
                        input_data(i,j) = means(j);
                    }
                }
            }
        }
        else
        {
            const Tensor<type, 1> medians = median(input_data);

            const Index samples_number = input_data.dimension(0);
            const Index variables_number = input_data.dimension(1);

        #pragma omp parallel for schedule(dynamic)

            for(Index j = 0; j < variables_number; j++)
            {
                for(Index i = 0; i < samples_number; i++)
                {
                    if(::isnan(input_data(i, j)))
                    {
                        input_data(i,j) = medians(j);
                    }
                }
            }
        }

        return input_data;
    }
}


/// Returns a vector containing the number of samples of each class in the data set.
/// If the number of target variables is one then the number of classes is two.
/// If the number of target variables is greater than one then the number of classes is equal to the number
/// of target variables.
/// @todo Low priority. Return class_distribution is wrong

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

        for(Index sample_index = 0; sample_index < static_cast<Index>(samples_number); sample_index++)
        {
            if(!::isnan(data(static_cast<Index>(sample_index),target_index)))
            {
                if(data(static_cast<Index>(sample_index),target_index) < static_cast<type>(0.5))
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

        for(Index i = 0; i < samples_number; i++)
        {
            if(get_sample_use(i) != UnusedSample)
            {
                for(Index j = 0; j < targets_number; j++)
                {
                    if(data(i,target_variables_indices(j)) == static_cast<type>(NAN)) continue;

                    if(data(i,target_variables_indices(j)) > 0.5) class_distribution(j)++;
                }
            }
        }
    }

    return class_distribution;
}


/// Calculate the outliers from the data set using the Tukey's test.
/// @param cleaning_parameter Parameter used to detect outliers.
/// @todo Low priority.

Tensor<Tensor<Index, 1>, 1> DataSet::calculate_Tukey_outliers(const type& cleaning_parameter) const
{
    const Index samples_number = get_used_samples_number();
    const Tensor<Index, 1> samples_indices = get_used_samples_indices();

    const Index columns_number = get_columns_number();
    const Index used_columns_number = get_used_columns_number();
    const Tensor<Index, 1> used_columns_indices = get_used_columns_indices();

    Tensor<Tensor<Index, 1>, 1> return_values(2);

    return_values(0) = Tensor<Index, 1>(samples_number);
    return_values(1) = Tensor<Index, 1>(used_columns_number);

    return_values(0).setZero();
    return_values(1).setZero();

    Tensor<BoxPlot, 1> box_plots = calculate_columns_box_plots();

    Index used_column_index = 0;
    Index variable_index = 0;

    #pragma omp parallel for

    for(Index i = 0; i < columns_number; i++)
    {
        if(columns(i).column_use == UnusedVariable && columns(i).type == Categorical)
        {
            variable_index += columns(i).get_categories_number();
            continue;
        }
        else if(columns(i).column_use == UnusedVariable) // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if(columns(i).type == Categorical || columns(i).type == Binary || columns(i).type == DateTime)
        {
            used_column_index++;
            columns(i).get_categories_number() == 0 ? variable_index++ : variable_index += columns(i).get_categories_number();
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots(used_column_index).third_quartile - box_plots(used_column_index).first_quartile;

            if(interquartile_range < numeric_limits<type>::epsilon())
            {
                used_column_index++;
                variable_index++;
                continue;
            }

            Index columns_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(samples_indices(static_cast<Index>(j)));

                if(sample(variable_index) <(box_plots(used_column_index).first_quartile - cleaning_parameter*interquartile_range) ||
                        sample(variable_index) >(box_plots(used_column_index).third_quartile + cleaning_parameter*interquartile_range))
                {
                    return_values(0)(static_cast<Index>(j)) = 1;

                    columns_outliers++;
                }
            }

            return_values(1)(used_column_index) = columns_outliers;

            used_column_index++;
            variable_index++;
        }
    }

    return return_values;
}


/// Calculate the outliers from the data set using the Tukey's test and sets in samples object.
/// @param cleaning_parameter Parameter used to detect outliers
/// @todo

void DataSet::unuse_Tukey_outliers(const type& cleaning_parameter)
{
    const Tensor<Tensor<Index, 1>, 1> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

//    const Tensor<Index, 1> outliers_samples = outliers_indices(0).get_indices_greater_than(0);

//    set_samples_unused(outliers_samples);

}


/// Returns a matrix with the values of autocorrelation for every variable in the data set.
/// The number of rows is equal to the number of
/// The number of columns is the maximum lags number.
/// @param maximum_lags_number Maximum lags number for which autocorrelation is calculated.
/// @todo

Tensor<type, 2> DataSet::calculate_autocorrelations(const Index& maximum_lags_number) const
{
    if(maximum_lags_number > get_used_samples_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 2> autocorrelations(const Index&) method.\n"
               << "Maximum lags number(" << maximum_lags_number << ") is greater than the number of samples("
               << get_used_samples_number() <<") \n";

        throw logic_error(buffer.str());
    }

    const Index variables_number = data.dimension(1);

    Tensor<type, 2> autocorrelations(variables_number, maximum_lags_number);

    for(Index j = 0; j < variables_number; j++)
    {
//        autocorrelations.set_row(j, OpenNN::autocorrelations(data.chip(j,1), maximum_lags_number));
    }

    return autocorrelations;
}


/// Calculates the cross-correlation between all the variables in the data set.

Tensor<Tensor<type, 1>, 2> DataSet::calculate_cross_correlations(const Index& lags_number) const
{
    const Index variables_number = get_variables_number();

    Tensor<Tensor<type, 1>, 2> cross_correlations(variables_number, variables_number);

    Tensor<type, 1> actual_column;

    for(Index i = 0; i < variables_number; i++)
    {
        actual_column = data.chip(i,1);

        for(Index j = 0; j < variables_number; j++)
        {
            cross_correlations(i,j) = OpenNN::cross_correlations(actual_column, data.chip(j,1), lags_number);
        }
    }

    return cross_correlations;
}


/// @todo

Tensor<type, 2> DataSet::calculate_lag_plot() const
{
    const Index samples_number = get_used_samples_number();

    const Index columns_number = data.dimension(1) - 1;

    Tensor<type, 2> lag_plot(samples_number, columns_number);

//    lag_plot = data.get_submatrix_columns(columns_indices);

    return lag_plot;
}


/// @todo, check

Tensor<type, 2> DataSet::calculate_lag_plot(const Index& maximum_lags_number)
{
    const Index samples_number = get_used_samples_number();

    if(maximum_lags_number > samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<type, 2> calculate_lag_plot(const Index&) method.\n"
               << "Maximum lags number(" << maximum_lags_number
               << ") is greater than the number of samples("
               << samples_number << ") \n";

        throw logic_error(buffer.str());
    }

    //const Tensor<type, 2> lag_plot = time_series_data.calculate_lag_plot(maximum_lags_number, time_index);

//    return lag_plot;

    return Tensor<type, 2>();
}


/// Generates an artificial dataset with a given number of samples and number of variables
/// by constant data.
/// @param samples_number Number of samples in the dataset.
/// @param variables_number Number of variables in the dataset.
/// @todo

void DataSet::generate_constant_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number, variables_number);

//    data.setRandom(-5.12, 5.12);

    for(Index i = 0; i < samples_number; i++)
    {
        data(i, variables_number-1) = 0;
    }

    scale_minimum_maximum(data);

    set_default_columns_uses();
}


/// Generates an artificial dataset with a given number of samples and number of variables
/// using random data.
/// @param samples_number Number of samples in the dataset.
/// @param variables_number Number of variables in the dataset.
/// @todo

void DataSet::generate_random_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number, variables_number);

    data.setRandom();

//        data.setRandom(0.0, 1.0);

}


/// Generates an artificial dataset with a given number of samples and number of variables
/// using a sequential data.
/// @param samples_number Number of samples in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_sequential_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number, variables_number);

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < variables_number; j++)
        {
            data(i,j) = static_cast<type>(j);
        }
    }
}


/// Generates an artificial dataset with a given number of samples and number of variables
/// using a paraboloid data.
/// @param samples_number Number of samples in the dataset.
/// @param variables_number Number of variables in the dataset.
/// @todo

void DataSet::generate_paraboloid_data(const Index& samples_number, const Index& variables_number)
{
    const Index inputs_number = variables_number-1;

    set(samples_number, variables_number);

//    data.setRandom();

    data.setRandom();

    for(Index i = 0; i < samples_number; i++)
    {
//        const type norm = l2_norm(data.chip(i, 0).delete_last(1));

//        data(i, inputs_number) = norm*norm;
    }

    scale_minimum_maximum(data);
}


/// Generates an artificial dataset with a given number of samples and number of variables
/// using the Rosenbrock function.
/// @param samples_number Number of samples in the dataset.
/// @param variables_number Number of variables in the dataset.
/// @todo

void DataSet::generate_Rosenbrock_data(const Index& samples_number, const Index& variables_number)
{
    const Index inputs_number = variables_number-1;

    set(samples_number, variables_number);

    data.setRandom();

    #pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        type rosenbrock = 0;

        for(Index j = 0; j < inputs_number-1; j++)
        {
            const type value = data(i,j);
            const type next_value = data(i,j+1);

            rosenbrock += (1 - value)*(1 - value)
                + 100*(next_value-value*value)*(next_value-value*value);
        }

        data(i, inputs_number) = rosenbrock;
    }

    set_default_columns_uses();
}


/// @todo

void DataSet::generate_inputs_selection_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number,variables_number);

    data.setRandom();

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < variables_number-2; j++)
        {
            data(i,variables_number-1) += data(i,j);
        }
    }

    set_default_columns_uses();
}


void DataSet::generate_sum_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number,variables_number);

    data.setRandom();

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < variables_number-1; j++)
        {
            data(i,variables_number-1) += data(i,j);
        }
    }

    set_default();

    scale_data_mean_standard_deviation();

}


/// Generate artificial data for a binary classification problem with a given number of samples and inputs.
/// @param samples_number Number of the samples to generate.
/// @param inputs_number Number of the variables that the data set will have.
/// @todo

void DataSet::generate_data_binary_classification(const Index& samples_number, const Index& inputs_number)
{
    const Index negatives = samples_number/2;
    const Index positives = samples_number - negatives;

    // Negatives data

    Tensor<type, 1> target_0(negatives);

    Tensor<type, 2> class_0(negatives, inputs_number+1);

//        class_0.setRandom(-0.5, 1.0);

//        class_0.set_column(inputs_number, target_0, "");

        // Positives data

//        Tensor<type, 1> target_1(positives, 1.0);

//        Tensor<type, 2> class_1(positives, inputs_number+1);

//        class_1.setRandom(0.5, 1.0);

//        class_1.set_column(inputs_number, target_1, "");

        // Assemble

//        set(class_0.assemble_rows(class_1));
}


/// @todo Low priority.

void DataSet::generate_data_multiple_classification(const Index& samples_number, const Index& inputs_number, const Index& outputs_number)
{
    Tensor<type, 2> new_data(samples_number, inputs_number);

    new_data.setRandom();

    Tensor<type, 2> targets(samples_number, outputs_number);

    Index target_index = 0;

    for(Index i = 0; i < samples_number; i ++)
    {
        target_index = static_cast<unsigned>(rand())%outputs_number;

        targets(i, target_index) = 1.0;
    }

//        set(new_data.assemble_columns(targets));
}


/// Returns true if the data matrix is not empty(it has not been loaded),
/// and false otherwise.

bool DataSet::has_data() const
{
    if(is_empty())
    {
        return false;
    }
    else
    {
        return true;
    }
}


/// Unuses those samples with values outside a defined range.
/// @param minimums vector of minimum values in the range.
/// The size must be equal to the number of variables.
/// @param maximums vector of maximum values in the range.
/// The size must be equal to the number of variables.
/// @todo Low priority.

Tensor<Index, 1> DataSet::filter_data(const Tensor<type, 1>& minimums, const Tensor<type, 1>& maximums)
{
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();

    const Index used_variables_number = used_variables_indices.size();

#ifdef __OPENNN_DEBUG__

    if(minimums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Size of minimums(" << minimums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    if(maximums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> filter_data(const Tensor<type, 1>&, const Tensor<type, 1>&) method.\n"
               << "Size of maximums(" << maximums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw logic_error(buffer.str());
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

            if(get_sample_use(sample_index) == UnusedSample) continue;

            if(isnan(data(sample_index, variable_index))) continue;

            if(fabsf(data(sample_index, variable_index) - minimums(i)) <= static_cast<type>(1e-3)
                    || fabsf(data(sample_index, variable_index) - maximums(i)) <= static_cast<type>(1e-3)) continue;

            if(data(sample_index,variable_index) < minimums(i)
                    || data(sample_index,variable_index) > maximums(i))
            {
                filtered_indices(sample_index) = 1.0;

                set_sample_use(sample_index, UnusedSample);
            }
        }
    }

    Index filtered_samples_number =
            static_cast<Index>(std::count_if(filtered_indices.data(), filtered_indices.data()+filtered_indices.size(), [](type value) {return value > static_cast<type>(0.5);}));

    Tensor<Index, 1> filtered_samples_indices(filtered_samples_number);
    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(filtered_indices(i) > static_cast<type>(0.5))
        {
            filtered_samples_indices(index) = i;
            index++;
        }
    }

    return filtered_samples_indices;
}


/// Filter data set variable using a rank.
/// The values within the variable must be between minimum and maximum.
/// @param variable_index Index number where the variable to be filtered is located.
/// @param minimum Value that determine the lower limit.
/// @param maximum Value that determine the upper limit.
/// Returns a indices vector.
/// @todo

Tensor<Index, 1> DataSet::filter_column(const Index& variable_index, const type& minimum, const type& maximum)
{
    const Index samples_number = get_samples_number();

    Tensor<type, 1> filtered_indices(samples_number);

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    const Tensor<Index, 1> current_samples_indices = used_samples_indices;

    const Index current_samples_number = current_samples_indices.size();

    for(Index i = 0; i < current_samples_number; i++)
    {
        const Index index = current_samples_indices(i);

        if(data(index,variable_index) < minimum || data(index,variable_index) > maximum)
        {
            filtered_indices(index) = 1.0;

            set_sample_use(index, UnusedSample);
        }
    }

//        return filtered_indices.get_indices_greater_than(0.5);

    return Tensor<Index, 1>();
}


/// Filter data set variable using a rank.
/// The values within the variable must be between minimum and maximum.
/// @param variable_name String name where the variable to be filtered is located.
/// @param minimum Value that determine the lower limit.
/// @param maximum Value that determine the upper limit.
/// Returns a indices vector.
/// @todo

Tensor<Index, 1> DataSet::filter_column(const string& variable_name, const type& minimum, const type& maximum)
{
    const Index variable_index = get_variable_index(variable_name);

    const Index samples_number = get_samples_number();

    Tensor<type, 1> filtered_indices(samples_number);

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    const Index current_samples_number = used_samples_indices.size();

    for(Index i = 0; i < current_samples_number; i++)
    {
        const Index index = used_samples_indices(i);

        if(data(index,variable_index) < minimum || data(index,variable_index) > maximum)
        {
            filtered_indices(index) = 1.0;

            set_sample_use(index, UnusedSample);
        }
    }

//        return filtered_indices.get_indices_greater_than(0.5);

    return Tensor<Index, 1>();
}


/// This method converts a numerical variable into categorical.
/// Note that this method resizes the dataset.
/// @param variable_index Index of the variable to be converted.

void DataSet::numeric_to_categorical(const Index& variable_index)
{
#ifdef __OPENNN_DEBUG__

    const Index variables_number = get_variables_number();

    if(variable_index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void convert_categorical_variable(const Index&) method.\n"
               << "Index of variable(" << variable_index << ") must be less than number of variables (" << variables_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

//    const Tensor<type, 1> categories = data.get_column(variable_index).get_unique_elements();

//    data = data.to_categorical(variable_index);

//    columns(variable_index).categories_uses = Tensor<VariableUse, 1>(categories.size(), columns(variable_index).column_use);
//    columns(variable_index).type = Categorical;
//    columns(variable_index).categories = categories.to_string_vector();
}


/// Sets all the samples with missing values to "Unused".

void DataSet::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for(Index i = 0; i <samples_number; i++)
    {
        if(has_nan_row(i))
        {
            set_sample_use(i, "Unused");
        }
    }
}

/// Substitutes all the missing values by the mean of the corresponding variable.

void DataSet::impute_missing_values_mean()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();

    const Tensor<type, 1> means = mean(data, used_samples_indices, used_variables_indices);

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();

    Index current_variable;
    Index current_sample;

#pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < variables_number; j++)
    {
        current_variable = used_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(::isnan(data(current_sample, current_variable)))
            {
                data(current_sample,current_variable) = means(j);
            }
        }
    }
}


/// Substitutes all the missing values by the median of the corresponding variable.

void DataSet::impute_missing_values_median()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_columns_indices();

    const Tensor<type, 1> medians = median(data, used_samples_indices, used_variables_indices);

    const Index variables_number = used_variables_indices.size();
    const Index samples_number = used_samples_indices.size();

#pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < variables_number; j++)
    {
        for(Index i = 0 ; i < samples_number ; i++)
        {
            if(::isnan(data(used_samples_indices(i),used_variables_indices(j)))) data(used_samples_indices(i),used_variables_indices(j)) = medians(j);
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
    case Unuse:
    {
        impute_missing_values_unuse();
    }
        break;

    case Mean:
    {
        impute_missing_values_mean();
    }
        break;

    case Median:
    {
        impute_missing_values_median();
    }
        break;
    }
}


/// @todo Time series stuff?

void DataSet::read_csv()
{
    read_csv_1();

    if(!has_time_columns() && !has_categorical_columns())
    {
        read_csv_2_simple();

        read_csv_3_simple();
    }
    else
    {

    //  categorical data

        read_csv_2_complete();

        read_csv_3_complete();
    }
}


Tensor<string, 1> DataSet::get_default_columns_names(const Index& columns_number)
{
    Tensor<string, 1> columns_names(columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        ostringstream buffer;

        buffer << "column_" << i+1;

        columns_names(i) = buffer.str();
    }

    return columns_names;
}


void DataSet::read_csv_1()
{
    ifstream file(data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    const char separator_char = get_separator_char();

    cout << "Setting data file preview..." << endl;

    Index lines_number = has_columns_names ? 4 : 3;

    data_file_preview.resize(lines_number);

    string line;

    Index lines_count = 0;

    while(file.good())
    {
        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        check_separators(line);

        check_special_characters(line);

        data_file_preview(lines_count) = get_tokens(line, separator_char);

        lines_count++;

        if(lines_count == lines_number) break;
    }

    file.close();

    // Check empty file    @todo, size() methods returns 0

    if(data_file_preview(0).size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "File " << data_file_name << " is empty.\n";

        throw logic_error(buffer.str());
    }

    // Set rows labels and columns names

    cout << "Setting rows labels..." << endl;

    string first_name = data_file_preview(0)(0);
    transform(first_name.begin(), first_name.end(), first_name.begin(), ::tolower);

    if(contains_substring(first_name, "id"))
    {
        has_rows_labels = true;
    }

    const Index columns_number = has_rows_labels ? data_file_preview(0).size()-1 : data_file_preview(0).size();

    columns.resize(columns_number);

    // Check if header has numeric value

    if(has_columns_names && has_numbers(data_file_preview(0)))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "Some columns names are numeric.\n";

        throw logic_error(buffer.str());
    }

    // Columns names

    cout << "Setting columns names..." << endl;

    if(has_columns_names)
    {
        has_rows_labels ? set_columns_names(data_file_preview(0).slice(Eigen::array<Eigen::Index, 1>({1}), Eigen::array<Eigen::Index, 1>({data_file_preview(0).size()-1})))
                        : set_columns_names(data_file_preview(0));
    }
    else
    {
        set_columns_names(get_default_columns_names(columns_number));
    }

    // Columns types

    cout << "Setting columns types..." << endl;

    Index column_index = 0;

    for(Index i = 0; i < data_file_preview(0).dimension(0); i++)
    {
        if(has_rows_labels && i == 0) continue;

        if(((is_numeric_string(data_file_preview(1)(i)) && data_file_preview(1)(i) != missing_values_label) || data_file_preview(1)(i).empty())
        || ((is_numeric_string(data_file_preview(2)(i)) && data_file_preview(2)(i) != missing_values_label) || data_file_preview(1)(i).empty())
        || ((is_numeric_string(data_file_preview(lines_number-2)(i)) && data_file_preview(lines_number-2)(i) != missing_values_label) || data_file_preview(1)(i).empty())
        || ((is_numeric_string(data_file_preview(lines_number-1)(i)) && data_file_preview(lines_number-1)(i) != missing_values_label) || data_file_preview(1)(i).empty()))
        {
            columns(column_index).type = Numeric;
            column_index++;
        }
        else if((is_date_time_string(data_file_preview(1)(i)) && data_file_preview(1)(i) != missing_values_label)
             || (is_date_time_string(data_file_preview(2)(i)) && data_file_preview(2)(i) != missing_values_label)
             || (is_date_time_string(data_file_preview(lines_number-2)(i)) && data_file_preview(lines_number-2)(i) != missing_values_label)
             || (is_date_time_string(data_file_preview(lines_number-1)(i)) && data_file_preview(lines_number-1)(i) != missing_values_label))
        {
            columns(column_index).type = DateTime;
            column_index++;
        }
        else
        {
            columns(column_index).type = Categorical;
            column_index++;
        }
    }


}


void DataSet::read_csv_2_simple()
{
    ifstream file(data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_2_simple() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    string line;
    Index line_number = 0;

    if(has_columns_names)
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

    cout << "Setting data dimensions..." << endl;

    const char separator_char = get_separator_char();

    const Index columns_number = get_columns_number();
    const Index raw_columns_number = has_rows_labels ? columns_number + 1 : columns_number;

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        //erase(line, '"');

        if(line.empty()) continue;

        tokens_count = count_tokens(line, separator_char);

        if(tokens_count != raw_columns_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void read_csv_2_simple() method.\n"
                   << "Line " << line_number << ": Size of tokens("
                   << tokens_count << ") is not equal to number of columns("
                   << raw_columns_number << ").\n";

            throw logic_error(buffer.str());
        }

        samples_count++;
    }

    file.close();

    data.resize(samples_count, columns_number);

    set_default_columns_uses();

    samples_uses.resize(samples_count);
    samples_uses.setConstant(Training);

    split_samples_random();
}


void DataSet::read_csv_3_simple()
{
    ifstream file(data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_2_simple() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    const bool is_float = is_same<type, float>::value;

    const char separator_char = get_separator_char();

    string line;

    // Read header

    if(has_columns_names)
    {
        while(file.good())
        {
            getline(file, line);

            if(line.empty()) continue;

            break;
        }
    }


    // Read data

    Index j = 0;

    //???

    const Index raw_columns_number = has_rows_labels ? get_columns_number() + 1 : get_columns_number();

    Tensor<string, 1> tokens(raw_columns_number);

    const Index samples_number = data.dimension(0);

    if(has_rows_labels) rows_labels.resize(samples_number);

    cout << "Reading data..." << endl;

    Index sample_index = 0;
    Index column_index = 0;


    while(file.good())
    {
        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        fill_tokens(line, separator_char, tokens);

        for(j = 0; j < raw_columns_number; j++)
        {
            trim(tokens(j));

            if(has_rows_labels && j == 0)
            {
                rows_labels(sample_index) = tokens(j);
            }
            else if(tokens(j) == missing_values_label || tokens(j).empty())
            {
                data(sample_index, column_index) = static_cast<type>(NAN);
                column_index++;
            }
            else if(is_float)
            {
                data(sample_index, column_index) = strtof(tokens(j).data(), NULL);
                column_index++;
            }
            else
            {
                data(sample_index, column_index) = stof(tokens(j));
                column_index++;
            }
        }

        column_index = 0;
        sample_index++;
    }

    const Index data_file_preview_index = has_columns_names ? 3 : 2;

    data_file_preview(data_file_preview_index) = tokens;

    file.close();

    cout << "Data read succesfully..." << endl;

    // Check Constant


    cout << "Checking constant columns..." << endl;

    Index variable_index = 0;

    for(Index column = 0; column < get_columns_number(); column++)
    {
        if(columns(column).type == Numeric)
        {
            // @todo avoid chip

//            if(is_constant_numeric(data.chip(variable_index, 1)))
//            {
//                columns(column).type = Constant;
//                columns(column).column_use = UnusedVariable;
//            }

            const type a = data(0, variable_index);

            bool constant = true;

            for (int i = 1; i < data.dimension(0); i++)
            {
                if (abs(data(i, variable_index)-a) > 1e-3 || ::isnan(data(i, variable_index)) || ::isnan(a))
                    constant = false;
            }

            if(constant)
            {
                columns(column).type = Constant;
                columns(column).column_use = UnusedVariable;
            }

            variable_index++;
        }
        else if(columns(column).type == DateTime)
        {
            columns(column).column_use = UnusedVariable;
            variable_index++;
        }
        else if(columns(column).type == Constant)
        {
            variable_index++;
        }
        else if(columns(column).type == Binary)
        {
            if(columns(column).get_categories_number() == 1)
            {
                columns(column).type = Constant;
                columns(column).column_use = UnusedVariable;
            }

            variable_index++;
        }
        else if(columns(column).type == Categorical)
        {
            if(columns(column).get_categories_number() == 1)
            {
                columns(column).type = Constant;
                columns(column).column_use = UnusedVariable;
            }

            variable_index += columns(column).get_categories_number();
        }

//        if(is_constant_numeric(data.chip(column, 1)) && columns(column).type!=DateTime)
//        {
//            columns(column).type = Constant;
//            columns(column).column_use = UnusedVariable;
//        }
    }
    // Check Binary

    cout << "Checking binary columns..." << endl;

    set_binary_simple_columns();


}


void DataSet::read_csv_2_complete()
{
    ifstream file(data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_2_complete() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    const char separator_char = get_separator_char();

    string line;

    Tensor<string, 1> tokens;

    Index lines_count = 0;
    Index tokens_count;

    const Index columns_number = columns.size();

    for(unsigned j = 0; j < columns_number; j++)
    {
        if(columns(j).type != Categorical)
        {
            columns(j).column_use = Input;
        }
    }

    // Skip header

    if(has_columns_names)
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

    cout << "Setting data dimensions..." << endl;

    const Index raw_columns_number = has_rows_labels ? columns_number + 1 : columns_number;

    Index column_index = 0;

    while(file.good())
    {
        getline(file, line);

        trim(line);

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_char);

        tokens_count = tokens.size();

        if(static_cast<unsigned>(tokens_count) != raw_columns_number)
        {
            const string message =
                "Sample " + to_string(lines_count+1) + " error:\n"
                "Size of tokens (" + to_string(tokens_count) + ") is not equal to number of columns (" + to_string(raw_columns_number) + ").\n"
                "Please check the format of the data file (e.g: Use of commas both as decimal and column separator)";

            throw logic_error(message);
        }

        for(unsigned j = 0; j < raw_columns_number; j++)
        {
            if(has_rows_labels && j == 0)
            {
                continue;
            }

            trim(tokens(j));

            if(columns(column_index).type == Categorical)
            {
                if(find(columns(column_index).categories.data(), columns(column_index).categories.data() + columns(column_index).categories.size(), tokens(j)) == (columns(column_index).categories.data() + columns(column_index).categories.size()))
                {
                    if(tokens(j) == missing_values_label)
                    {
                        column_index++;
                        continue;
                    }

                    columns(column_index).add_category(tokens(j));
                }
            }

            column_index++;
        }

        column_index = 0;

        lines_count++;
    }


    cout << "Setting types..." << endl;

    for(Index j = 0; j < columns_number; j++)
    {
        if(columns(j).type == Categorical)
        {
            if(columns(j).categories.size() == 2)
            {
                columns(j).type = Binary;
            }
        }
    }

    file.close();

    const Index samples_number = static_cast<unsigned>(lines_count);

    const Index variables_number = get_variables_number();

    data.resize(static_cast<Index>(samples_number), variables_number);
    data.setZero();

    if(has_rows_labels) rows_labels.resize(samples_number);

    set_default_columns_uses();

    samples_uses.resize(static_cast<Index>(samples_number));

    samples_uses.setConstant(Training);

    split_samples_random();
}


void DataSet::read_csv_3_complete()
{
    ifstream file(data_file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_3_complete() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }



    const char separator_char = get_separator_char();

    const Index columns_number = columns.size();

    const Index raw_columns_number = has_rows_labels ? columns_number+1 : columns_number;

    string line;

    Tensor<string, 1> tokens;

    string token;

    unsigned sample_index = 0;
    unsigned variable_index = 0;
    unsigned column_index = 0;

    // Skip header

    if(has_columns_names)
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

    cout << "Reading data..." << endl;

    while(file.good())
    {
        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_char);

        variable_index = 0;
        column_index = 0;

        for(Index j = 0; j < raw_columns_number; j++)
        {

            trim(tokens(j));

            if(has_rows_labels && j ==0)
            {
                rows_labels(sample_index) = tokens(j);
                continue;
            }
            else if(columns(column_index).type == Numeric)
            {

                if(tokens(j) == missing_values_label || tokens(j).empty())
                {
                    data(sample_index, variable_index) = static_cast<type>(NAN);
                    variable_index++;
                }
                else
                {
                    try
                    {

                        data(sample_index, variable_index) = static_cast<type>(stod(tokens(j)));
                        variable_index++;
                    }
                    catch (invalid_argument)
                    {
                        ostringstream buffer;

                        buffer << "OpenNN Exception: DataSet class.\n"
                               << "void read_csv_3_complete() method.\n"
                               << "Sample " << sample_index << "; Invalid number: " << tokens(j) << "\n";

                        throw logic_error(buffer.str());
                    }
                }
            }
            else if(columns(column_index).type == DateTime)
            {
                if(tokens(j) == missing_values_label || tokens(j).empty())
                {
                    data(sample_index, variable_index) = static_cast<type>(NAN);
                    variable_index++;
                }
                else
                {
                    data(sample_index, variable_index) = static_cast<type>(date_to_timestamp(tokens(j), gmt));
                    variable_index++;
                }
            }
            else if(columns(column_index).type == Categorical)
            {
                for(Index k = 0; k < columns(column_index).get_categories_number(); k++)
                {
                    if(tokens(j) == missing_values_label)
                    {
                        data(sample_index, variable_index) = static_cast<type>(NAN);
                    }
                    else if(tokens(j) == columns(column_index).categories(k))
                    {
                        data(sample_index, variable_index) = 1.0;
                    }

                    variable_index++;
                }
            }
            else if(columns(column_index).type == Binary)
            {
                if(tokens(j) == missing_values_label)
                {
                    data(sample_index, variable_index) = static_cast<type>(NAN);
                }
                else if(columns(column_index).categories.size() > 0 && tokens(j) == columns(column_index).categories(0))
                {
                    data(sample_index, variable_index) = 1.0;
                }
                else if(tokens(j) == columns(column_index).name)
                {
                    data(sample_index, variable_index) = 1.0;
                }

                variable_index++;
            }

            column_index++;
        }

        sample_index++;
    }

    const Index data_file_preview_index = has_columns_names ? 3 : 2;

    data_file_preview(data_file_preview_index) = tokens;

    cout << "Data read succesfully..." << endl;

    file.close();

    // Check binary
    cout << "Checking binary columns..." << endl;

    set_binary_simple_columns();

    // Check Constant and DateTime to unused

    cout << "Checking constant columns..." << endl;

    variable_index = 0;

    for(Index column = 0; column < get_columns_number(); column++)
    {
        if(columns(column).type == Numeric)
        {
            const Tensor<type, 1> numeric_column = data.chip(variable_index, 1);

            if(standard_deviation(numeric_column) - static_cast<type>(0) < static_cast<type>(1.0-3))
            {

                columns(column).type = Constant;
                columns(column).column_use = UnusedVariable;
            }

            variable_index++;
        }
        else if(columns(column).type == DateTime)
        {
            columns(column).column_use = UnusedVariable;
            variable_index++;
        }
        else if(columns(column).type == Constant)
        {
            columns(column).column_use = UnusedVariable;

            variable_index++;
        }
        else if(columns(column).type == Binary)
        {
            if(columns(column).get_categories_number() == 1)
            {
                columns(column).type = Constant;
                columns(column).column_use = UnusedVariable;
                columns(column).set_categories_uses(UnusedVariable);
            }

            variable_index++;
        }
        else if(columns(column).type == Categorical)
        {
            if(columns(column).get_categories_number() == 1)
            {
                columns(column).type = Constant;
                columns(column).column_use = UnusedVariable;
                columns(column).set_categories_uses(UnusedVariable);
            }

            variable_index += columns(column).get_categories_number();
        }
    }
}


void DataSet::check_separators(const string& line) const
{
    if(line.find(',') == string::npos
            && line.find(';') == string::npos
            && line.find(' ') == string::npos
            && line.find('\t') == string::npos)
    {
        return;
    }

    const char separator_char = get_separator_char();

    if(line.find(separator_char) == string::npos)
    {
        const string message =
            "Error: " + get_separator_string() + " separator not found in data file " + data_file_name + ".";

        throw logic_error(message);
    }

    if(separator == Space)
    {
        if(line.find(',') != string::npos)
        {
            const string message =
                "Error: Found comma (',') in data file " + data_file_name + ", but separator is space (' ').";

            throw logic_error(message);
        }
        if(line.find(';') != string::npos)
        {
            const string message =
                "Error: Found semicolon (';') in data file " + data_file_name + ", but separator is space (' ').";

            throw logic_error(message);
        }
    }
    else if(separator == Tab)
    {
        if(line.find(',') != string::npos)
        {
            const string message =
                "Error: Found comma (',') in data file " + data_file_name + ", but separator is tab ('   ').";

            throw logic_error(message);
        }
        if(line.find(';') != string::npos)
        {
            const string message =
                "Error: Found semicolon (';') in data file " + data_file_name + ", but separator is tab ('   ').";

            throw logic_error(message);
        }
    }
    else if(separator == Comma)
    {
        if(line.find(";") != string::npos)
        {
            const string message =
                "Error: Found semicolon (';') in data file " + data_file_name + ", but separator is comma (',').";

            throw logic_error(message);
        }
    }
    else if(separator == Semicolon)
    {
        if(line.find(",") != string::npos)
        {
            const string message =
                "Error: Found comma (',') in data file " + data_file_name + ", but separator is semicolon (';'). " + line;

            throw logic_error(message);
        }
    }
}


void DataSet::check_special_characters(const string & line) const
{
    if(line.find_first_of("|@#~€¬^*") != std::string::npos)
    {
        const string message =
            "Error: found special characters in line: " + line + ". Please, review the document.";

        throw logic_error(message);
    }

#ifdef __unix__
    if(line.find("\r") != std::string::npos)
    {
        const string message =
                "Error: mixed break line characters in line: " + line + ". Please, review the document.";
        throw logic_error(message);
    }
#endif
}


bool DataSet::has_binary_columns() const
{
    const Index variables_number = columns.size();

    for(Index i = 0; i < variables_number; i++)
    {
        if(columns(i).type == Binary) return true;
    }

    return false;
}


bool DataSet::has_categorical_columns() const
{
    const Index variables_number = columns.size();

    for(Index i = 0; i < variables_number; i++)
    {
        if(columns(i).type == Categorical) return true;
    }

    return false;
}


bool DataSet::has_time_columns() const
{
    const Index variables_number = columns.size();

    for(Index i = 0; i < variables_number; i++)
    {
        if(columns(i).type == DateTime) return true;
    }

    return false;
}


bool DataSet::has_selection() const
{
    if(get_selection_samples_number() == 0) return false;

    return true;
}


Tensor<Index, 1> DataSet::count_nan_columns() const
{
    const Index columns_number = get_columns_number();
    const Index rows_number = get_samples_number();

    Tensor<Index, 1> nan_columns(get_columns_number());
    nan_columns.setZero();

    for(Index column_index = 0; column_index < columns_number; column_index++)
    {
        const Index current_variable_index = get_variable_indices(column_index)(0);

        for(Index row_index = 0; row_index < rows_number; row_index++)
        {
            if(isnan(data(row_index,current_variable_index)))
            {
                nan_columns(column_index) = nan_columns(column_index) + 1;
            }
        }
    }

    return nan_columns;
}


Index DataSet::count_rows_with_nan() const
{
    Index rows_with_nan = 0;

    const Index rows_number = data.dimension(0);
    const Index columns_number = data.dimension(1);

    bool has_nan = true;

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        has_nan = false;

        for(Index column_index = 0; column_index < columns_number; column_index++)
        {
            if(isnan(data(row_index, column_index)))
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
    const Index rows_number = data.dimension(0);
    const Index columns_number = data.dimension(1);

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        for(Index column_index = 0; column_index < columns_number; column_index++)
        {
            if(isnan(data(row_index, column_index))) count++;
        }
    }

    return count;
}


void DataSet::set_missing_values_number(const Index& new_missing_values_number)
{
    missing_values_number = new_missing_values_number;
}


void DataSet::set_missing_values_number()
{
    missing_values_number = count_nan();
}


void DataSet::set_columns_missing_values_number(const Tensor<Index, 1>& new_columns_missing_values_number)
{
    columns_missing_values_number = new_columns_missing_values_number;
}


void DataSet::set_columns_missing_values_number()
{
    columns_missing_values_number = count_nan_columns();
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
    // Fix columns names

    const Index columns_number = columns.size();

    std::map<std::string, Index> columns_count_map;

    for(Index i = 0; i < columns_number; i++)
    {
        auto result = columns_count_map.insert(std::pair<std::string, Index>(columns(i).name, 1));

        if (!result.second) result.first->second++;
    }

    for (auto & element : columns_count_map)
    {
        if(element.second > 1)
        {
            const string repeated_name = element.first;
            Index repeated_index = 1;

            for(Index i = 0; i < columns.size(); i++)
            {
                if(columns(i).name == repeated_name)
                {
                    columns(i).name = columns(i).name + "_" + std::to_string(repeated_index);
                    repeated_index++;
                }
            }
        }
    }

    // Fix variables names

    if(has_categorical_columns() || has_binary_columns())
    {
        Tensor<string, 1> variables_names = get_variables_names();

        const Index variables_number = variables_names.size();

        std::map<std::string, Index> variables_count_map;

        for(Index i = 0; i < variables_number; i++)
        {
            auto result = variables_count_map.insert(std::pair<std::string, Index>(variables_names(i), 1));

            if (!result.second) result.first->second++;
        }

        for (auto & element : variables_count_map)
        {
            if(element.second > 1)
            {
                const string repeated_name = element.first;

                for(Index i = 0; i < variables_number; i++)
                {
                    if(variables_names(i) == repeated_name)
                    {
                        const Index column_index = get_column_index(i);

                        if(columns(column_index).type != Categorical) continue;

                        variables_names(i) = variables_names(i) + "_" + columns(column_index).name;
                    }
                }
            }
        }

        set_variables_names(variables_names);
    }
}


Tensor<Index, 1> DataSet::push_back(const Tensor<Index, 1>& old_vector, const Index& new_string) const
{
    const Index old_size = old_vector.size();

    const Index new_size = old_size+1;

    Tensor<Index, 1> new_vector(new_size);

    for(Index i = 0; i < old_size; i++) new_vector(i) = old_vector(i);

    new_vector(new_size-1) = new_string;

    return new_vector;
}


Tensor<string, 1> DataSet::push_back(const Tensor<string, 1>& old_vector, const string& new_string) const
{
    const Index old_size = old_vector.size();

    const Index new_size = old_size+1;

    Tensor<string, 1> new_vector(new_size);

    for(Index i = 0; i < old_size; i++) new_vector(i) = old_vector(i);

    new_vector(new_size-1) = new_string;

    return new_vector;
}


void DataSet::initialize_sequential_eigen_tensor(Tensor<Index, 1>& new_tensor,
        const Index& start, const Index& step, const Index& end) const
{
    const Index new_size = (end-start)/step+1;

    new_tensor.resize(new_size);
    new_tensor(0) = start;

    for(Index i = 1; i < new_size-1; i++)
    {
        new_tensor(i) = new_tensor(i-1)+step;
    }

    new_tensor(new_size-1) = end;
}


void DataSet::intialize_sequential_eigen_type_tensor(Tensor<type, 1>& new_tensor,
        const type& start, const type& step, const type& end) const
{
    const Index new_size = (end-start)/step+1;

    new_tensor.resize(new_size);
    new_tensor(0) = start;

    for(Index i = 1; i < new_size-1; i++)
    {
        new_tensor(i) = new_tensor(i-1)+step;
    }

    new_tensor(new_size-1) = end;
}


Tensor<Index, 2> DataSet::split_samples(const Tensor<Index, 1>& samples_indices, const Index & new_batch_size) const
{
    const Index samples_number = samples_indices.dimension(0);

    Index batches_number;
    Index batch_size = new_batch_size;

//    const Index batches_number =  samples_number / batch_size;
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

    for(Index i = 0; i < batches_number; ++i)
    {
        for(Index j = 0; j < batch_size; ++j)
        {
            batches(i,j) = samples_indices(count);

            count++;
        }
    }

    return batches;
}

void DataSet::fill_submatrix(const Tensor<type, 2>& matrix,
          const Tensor<Index, 1>& rows_indices,
          const Tensor<Index, 1>& columns_indices, type* submatrix_pointer)
{
    const Index rows_number = rows_indices.size();
    const Index columns_number = columns_indices.size();

    const type* matrix_pointer = matrix.data();

#pragma omp parallel for

    for(Index j = 0; j < columns_number; j++)
    {
        const type* matrix_column_pointer = matrix_pointer + matrix.dimension(0)*columns_indices[j];
        type* submatrix_column_pointer = submatrix_pointer + rows_number*j;

        const type* value_pointer = nullptr;
        const Index* rows_indices_pointer = rows_indices.data();
        for(Index i = 0; i < rows_number; i++)
        {
            value_pointer = matrix_column_pointer + *rows_indices_pointer;
            rows_indices_pointer++;
            *submatrix_column_pointer = *value_pointer;
            submatrix_column_pointer++;
        }
    }
}


void DataSet::Batch::fill(const Tensor<Index, 1>& samples,
                          const Tensor<Index, 1>& inputs,
                          const Tensor<Index, 1>& targets)
{
    const Tensor<type, 2>& data = data_set_pointer->get_data();

    const Tensor<Index, 1>& input_variables_dimensions = data_set_pointer->get_input_variables_dimensions();

    if(input_variables_dimensions.size() == 1)
    {
        data_set_pointer->fill_submatrix(data, samples, inputs, inputs_2d.data());
    }
    else if(input_variables_dimensions.size() == 3)
    {
/*
        const Index channels_number = input_variables_dimensions(0);
        const Index rows_number = input_variables_dimensions(1);
        const Index columns_number = input_variables_dimensions(2);
        inputs_4d.resize(samples_number, channels_number, rows_number, columns_number);
        Index index = 0;
        for(Index image = 0; image < samples_number; image++)
        {
            index = 0;
            for(Index channel = 0; channel < channels_number; channel++)
            {
                for(Index row = 0; row < rows_number; row++)
                {
                    for(Index column = 0; column < columns_number; column++)
                    {
                        inputs_4d(image, channel, row, column) = data(image, index);
                        index++;
                    }
                }
            }
        }
*/
    }
    data_set_pointer->fill_submatrix(data, samples, targets, targets_2d.data());
}


DataSet::Batch::Batch(const Index& new_samples_number, DataSet* new_data_set_pointer)
{
    samples_number = new_samples_number;

    data_set_pointer = new_data_set_pointer;

    const Index input_variables_number = data_set_pointer->get_input_variables_number();
    const Index target_variables_number = data_set_pointer->get_target_variables_number();

    const Tensor<Index, 1> input_variables_dimensions = data_set_pointer->get_input_variables_dimensions();

    if(input_variables_dimensions.rank() == 1)
    {
        inputs_2d.resize(samples_number, input_variables_number);
    }
    else if(input_variables_dimensions.rank() == 3)
    {
        const Index channels_number = input_variables_dimensions(0);
        const Index rows_number = input_variables_dimensions(1);
        const Index columns_number = input_variables_dimensions(2);

        inputs_4d.resize(samples_number, channels_number, rows_number, columns_number);
    }


    targets_2d.resize(samples_number, target_variables_number);
}


Index DataSet::Batch::get_samples_number() const
{
    return samples_number;
}


void DataSet::Batch::print()
{
    cout << "Batch structure" << endl;

    cout << "Inputs:" << endl;
    cout << inputs_2d << endl;

    cout << "Targets:" << endl;
    cout << targets_2d << endl;
}


void DataSet::shuffle()
{
    const Index data_rows = data.dimension(0);
    const Index data_columns = data.dimension(1);

    Tensor<Index, 1> indices(data_rows);

    for(Index i = 0; i < data_rows; i++) indices(i) = i;

    random_shuffle(&indices(0), &indices(data_rows-1));

    Tensor<type, 2> new_data(data_rows, data_columns);
    Tensor<string, 1> new_rows_labels(data_rows);

    Index index = 0;

    for(Index i = 0; i < data_rows; i++)
    {
        index = indices(i);

        new_rows_labels(i) = rows_labels(index);

        for(Index j = 0; j < data_columns; j++)
        {
            new_data(i,j) = data(index,j);
        }
    }

    data = new_data;
    rows_labels = new_rows_labels;
}


bool DataSet::get_has_rows_labels() const
{
    return this->has_rows_labels;
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
