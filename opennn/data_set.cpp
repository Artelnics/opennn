//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "data_set.h"


using namespace  OpenNN;

namespace OpenNN
{

/// Default constructor.
/// It creates a data set object with zero instances and zero inputs and target variables.
/// It also initializes the rest of class members to their default values.

DataSet::DataSet()
{
   set();

   set_default();
}


/// Default constructor. It creates a data set object from data Eigen Matrix.
/// It also initializes the rest of class members to their default values.
/// @param data Data MatrixXd.

DataSet::DataSet(const Eigen::MatrixXd& data)
{
    set(data);

    set_default();
}


/// Data constructor. It creates a data set object from a data matrix.
/// It also initializes the rest of class members to their default values.
/// @param data Data matrix.

DataSet::DataSet(const Matrix<double>& data)
{
   set(data);

   set_default();
}


/// Instances and variables number constructor.
/// It creates a data set object with given instances and variables numbers.
/// All the variables are set as inputs.
/// It also initializes the rest of class members to their default values.
/// @param new_instances_number Number of instances in the data set.
/// @param new_variables_number Number of variables.

DataSet::DataSet(const size_t& new_instances_number, const size_t& new_variables_number)
{
   set(new_instances_number, new_variables_number);

   set_default();
}


/// Instances number, input variables number and target variables number constructor.
/// It creates a data set object with given instances and inputs and target variables numbers.
/// It also initializes the rest of class members to their default values.
/// @param new_instances_number Number of instances in the data set.
/// @param new_inputs_number Number of input variables.
/// @param new_targets_number Number of target variables.

DataSet::DataSet(const size_t& new_instances_number, const size_t& new_inputs_number, const size_t& new_targets_number)
{
   set(new_instances_number, new_inputs_number, new_targets_number);

   set_default();
}


/// Sets the data set members from a XML document.
/// @param data_set_document TinyXML document containing the member data.

DataSet::DataSet(const tinyxml2::XMLDocument& data_set_document)
{
   set_default();

   from_XML(data_set_document);
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


/// Copy constructor.
/// It creates a copy of an existing inputs targets data set object.
/// @param other_data_set Data set object to be copied.

DataSet::DataSet(const DataSet& other_data_set)
{
   set_default();

   set(other_data_set);
}


/// Destructor.

DataSet::~DataSet()
{
}


/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& DataSet::get_display() const
{
   return display;
}


// Columns methods

/// Column default constructor

DataSet::Column::Column()
{
    name = "";
    column_use = Input;
    type = Numeric;
    categories.set();
    categories_uses.set();
}


/// Column default constructor

DataSet::Column::Column(const string& new_name, const VariableUse& new_column_use, const ColumnType& new_type, const Vector<string>& new_categories, const Vector<VariableUse>& new_categories_uses)
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


/// Sets the column use.
/// @param new_column_use Column use.

void DataSet::Column::set_use(const VariableUse& new_column_use)
{
    column_use = new_column_use;
    categories_uses.initialize(new_column_use);
}


/// Sets the column use.
/// @param new_column_use Column use in string format.

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
    else if(new_column_use == "UnusedVariable")
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
/// @param new_column_type Column type.

void DataSet::Column::set_type(const ColumnType& new_column_type)
{
    type = new_column_type;
}


/// Sets the column type.
/// @param new_column_type Column type in string format.

void DataSet::Column::set_type(const string& new_column_type)
{
    if(new_column_type == "Numerical")
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
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void Column::set_type(const string&) method.\n"
               << "Column type not valid (" << new_column_type << ").\n";

        throw logic_error(buffer.str());

    }
}


/// Sets the new categories uses.
/// @param new_categories_uses Vector which contains the categories uses of the columns of the data set.

void DataSet::Column::set_categories_uses(const Vector<VariableUse>& new_categories_uses)
{
    const size_t new_categories_uses_number = new_categories_uses.size();

    categories_uses.set(new_categories_uses_number);

    for(size_t i = 0; i < new_categories_uses.size(); i++)
    {
        if(new_categories_uses[i] == OpenNN::DataSet::VariableUse::Input)
        {
            categories_uses[i] = Input;
        }
        else if(new_categories_uses[i] == OpenNN::DataSet::VariableUse::Target)
        {
            categories_uses[i] = Target;
        }
        else if(new_categories_uses[i] == OpenNN::DataSet::VariableUse::Time)
        {
            categories_uses[i] = Time;
        }
        else if(new_categories_uses[i] == OpenNN::DataSet::VariableUse::UnusedVariable)
        {
            categories_uses[i] = UnusedVariable;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void Column::set_categories_uses(const Vector<string>&) method.\n"
                   << "Category use not valid (" << new_categories_uses[i] << ").\n";

            throw logic_error(buffer.str());

        }
    }
}

/// Sets the new categories uses.
/// @param new_categories_uses String vector which contains the categories uses of the columns of the data set.

void DataSet::Column::set_categories_uses(const Vector<string>& new_categories_uses)
{
    const size_t new_categories_uses_number = new_categories_uses.size();

    categories_uses.set(new_categories_uses_number);

    for(size_t i = 0; i < new_categories_uses.size(); i++)
    {
        if(new_categories_uses[i] == "Input")
        {
            categories_uses[i] = Input;
        }
        else if(new_categories_uses[i] == "Target")
        {
            categories_uses[i] = Target;
        }
        else if(new_categories_uses[i] == "Time")
        {
            categories_uses[i] = Time;
        }
        else if(new_categories_uses[i] == "Unused"
             || new_categories_uses[i] == "UnusedVariable")
        {
            categories_uses[i] = UnusedVariable;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class.\n"
                   << "void Column::set_categories_uses(const Vector<string>&) method.\n"
                   << "Category use not valid (" << new_categories_uses[i] << ").\n";

            throw logic_error(buffer.str());

        }
    }
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

            categories = get_tokens(new_categories, ' ');
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

            set_categories_uses(get_tokens(new_categories_uses, ' '));
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
    else if (column_use == Time)
    {
        file_stream.PushText("Time");
    }
    else
    {
        file_stream.PushText("Unused");
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
    else
    {
        file_stream.PushText("DateTime");
    }

    file_stream.CloseElement();

    if(type == Categorical)
    {
        // Categories

        file_stream.OpenElement("Categories");

        for(size_t i = 0; i < categories.size(); i++)
        {
            file_stream.PushText(categories[i].c_str());
        }

        file_stream.CloseElement();

        // Categories uses

        file_stream.OpenElement("CategoriesUses");

        for(size_t i = 0; i < categories_uses.size(); i++)
        {
            if(categories_uses[i] == Input)
            {
                file_stream.PushText("Input");
            }
            else if (categories_uses[i] == Target)
            {
                file_stream.PushText("Target");
            }
            else if (categories_uses[i] == Time)
            {
                file_stream.PushText("Time");
            }
            else
            {
                file_stream.PushText("Unused");
            }
        }

        file_stream.CloseElement();
    }
}


/// Returns the number of categories contained in the data set.

size_t DataSet::Column::get_categories_number() const
{
    return categories.size();
}


/// Returns the name of the used variables in the dataset.

Vector<string> DataSet::Column::get_used_variables_names() const
{
    Vector<string> used_variables_names;

    if(type != Categorical && column_use != UnusedVariable)
    {
        used_variables_names.set(1, name);
    }
    else if(type == Categorical)
    {
        for(size_t i = 0; i < categories.size(); i++)
        {
            if(categories_uses[i] != UnusedVariable)
            {
                used_variables_names.push_back(categories[i]);
            }
        }
    }

    return used_variables_names;
}


void DataSet::transform_columns_time_series()
{
    const size_t columns_number = get_columns_number();

    Vector<Column> new_columns;

    if(has_time_variables())
    {
        new_columns.set((columns_number-1)*(lags_number+steps_ahead));
    }
    else
    {
        new_columns.set(columns_number*(lags_number+steps_ahead));
    }

    size_t lag_index = lags_number - 1;
    size_t ahead_index = 0;
    size_t column_index = 0;
    size_t new_column_index = 0;

    for(size_t i = 0; i < columns_number*(lags_number+steps_ahead); i++)
    {
        column_index = i%columns_number;

        if(columns[column_index].type == DateTime)
        {
            continue;
        }

        if(i < lags_number*columns_number)
        {
            new_columns[new_column_index].name = columns[column_index].name + "_lag_" + to_string(lag_index);
            new_columns[new_column_index].set_use(Input);

            new_columns[new_column_index].type = columns[column_index].type;
            new_columns[new_column_index].categories = columns[column_index].categories;
            new_columns[new_column_index].categories_uses = columns[column_index].categories_uses;

            new_column_index++;
        }
        else
        {
            new_columns[new_column_index].name = columns[column_index].name + "_ahead_" + to_string(ahead_index);
            new_columns[new_column_index].set_use(Target);

            new_columns[new_column_index].type = columns[column_index].type;
            new_columns[new_column_index].categories = columns[column_index].categories;
            new_columns[new_column_index].categories_uses = columns[column_index].categories_uses;

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



/// Returns true if a given instance is to be used for training, selection or testing,
/// and false if it is to be unused.
/// @param index Instance index.

bool DataSet::is_instance_used(const size_t& index) const
{
    if(instances_uses[index] == UnusedInstance)
    {
        return false;
    }
    else
    {
        return true;
    }
}


/// Returns true if a given instance is to be unused and false in other case.
/// @param index Instance index.

bool DataSet::is_instance_unused(const size_t& index) const
{
    if(instances_uses[index] == UnusedInstance)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns a vector with the number of training, selection, testing
/// and unused instances.
/// The size of that vector is therefore four.

Vector<size_t> DataSet::get_instances_uses_numbers() const
{
    Vector<size_t> count(4, 0);

    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        if(instances_uses[i] == Training)
        {
           count[0]++;
        }
        else if(instances_uses[i] == Selection)
        {
           count[1]++;
        }
        else if(instances_uses[i] == Testing)
        {
           count[2]++;
        }
        else
        {
           count[3]++;
        }
    }

    return count;
}


Vector<double> DataSet::get_instances_uses_percentages() const
{
    const size_t instances_number = get_instances_number();
    const size_t training_instances_number = get_training_instances_number();
    const size_t selection_instances_number = get_selection_instances_number();
    const size_t testing_instances_number = get_testing_instances_number();
    const size_t unused_instances_number = get_unused_instances_number();

    const double training_instances_percentage = static_cast<double>(training_instances_number)*100.0/static_cast<double>(instances_number);
    const double selection_instances_percentage = static_cast<double>(selection_instances_number)*100.0/static_cast<double>(instances_number);
    const double testing_instances_percentage = static_cast<double>(testing_instances_number)*100.0/static_cast<double>(instances_number);
    const double unused_instances_percentage = static_cast<double>(unused_instances_number)*100.0/static_cast<double>(instances_number);

    return Vector<double>({training_instances_percentage, selection_instances_percentage, testing_instances_percentage, unused_instances_percentage});
}


/// Returns the indices of the instances which will be used for training.

Vector<size_t> DataSet::get_training_instances_indices() const
{
    const size_t instances_number = get_instances_number();

    const size_t training_instances_number = get_training_instances_number();

    Vector<size_t> training_indices(training_instances_number);

    size_t count = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
       if(instances_uses[i] == Training)
       {
          training_indices[count] = static_cast<size_t>(i);
          count++;
       }
    }
    return training_indices;
}


/// Returns the indices of the instances which will be used for selection.

Vector<size_t> DataSet::get_selection_instances_indices() const
{
    const size_t instances_number = get_instances_number();

    const size_t selection_instances_number = get_selection_instances_number();

    Vector<size_t> selection_indices(selection_instances_number);

    size_t count = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
       if(instances_uses[i] == Selection)
       {
          selection_indices[count] = i;
          count++;
       }
    }

    return selection_indices;
}


/// Returns the indices of the instances which will be used for testing.

Vector<size_t> DataSet::get_testing_instances_indices() const
{
   const size_t instances_number = get_instances_number();

   const size_t testing_instances_number = get_testing_instances_number();

   Vector<size_t> testing_indices(testing_instances_number);

   size_t count = 0;

   for(size_t i = 0; i < instances_number; i++)
   {
      if(instances_uses[i] == Testing)
      {
         testing_indices[count] = i;
         count++;
      }
   }

   return testing_indices;
}


/// Returns the indices of the used instances(those which are not set unused).

Vector<size_t> DataSet::get_used_instances_indices() const
{
    const size_t instances_number = get_instances_number();

    const size_t used_instances_number = instances_number - get_unused_instances_number();

    Vector<size_t> used_indices(used_instances_number);

    size_t index = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
        if(instances_uses[i] != UnusedInstance)
        {
            used_indices[index] = i;
            index++;
        }
    }

    return used_indices;
}


/// Returns the indices of the instances set unused.

Vector<size_t> DataSet::get_unused_instances_indices() const
{
    const size_t instances_number = get_instances_number();

    const size_t unused_instances_number = get_unused_instances_number();

    Vector<size_t> unused_indices(unused_instances_number);

    size_t count = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
        if(instances_uses[i] == UnusedInstance)
        {
            unused_indices[count] = static_cast<size_t>(i);
            count++;
        }
    }

    return unused_indices;
}


/// Returns the use of a single instance.
/// @param index Instance index.

DataSet::InstanceUse DataSet::get_instance_use(const size_t& index) const
{
    return instances_uses[index];
}


/// Returns the use of every instance (training, selection, testing or unused) in a vector.

const Vector<DataSet::InstanceUse>& DataSet::get_instances_uses() const
{
    return instances_uses;
}


/// Returns a vector, where each element is a vector that contains the indices of the different batches of the training instances.
/// @param shuffle Is a boleean.
/// If shuffle is true, then the indices are shuffled into batches, and false otherwise
/// @todo In forecasting must be false.

Vector<Vector<size_t>> DataSet::get_training_batches(const bool& shuffle_batches_instances) const
{
    Vector<size_t> training_indices = get_training_instances_indices();

    if(shuffle_batches_instances) random_shuffle(training_indices.begin(), training_indices.end());

    return training_indices.split(batch_instances_number);
}


/// Returns a vector, where each element is a vector that contains the indices of the different batches of the selection instances.
/// @param shuffle Is a boleean.
/// If shuffle is true, then the indices are shuffled into batches, and false otherwise

Vector<Vector<size_t>> DataSet::get_selection_batches(const bool& shuffle_batches_instances) const
{
    Vector<size_t> selection_indices = get_selection_instances_indices();

    if(shuffle_batches_instances) random_shuffle(selection_indices.begin(), selection_indices.end());

    return selection_indices.split(batch_instances_number);
}


/// Returns a vector, where each element is a vector that contains the indices of the different batches of the testing instances.
/// If shuffle is true, then the indices within batches are shuffle, and false otherwise
/// @param shuffle_batches_instances Is a boleean.

Vector<Vector<size_t>> DataSet::get_testing_batches(const bool& shuffle_batches_instances) const
{
    Vector<size_t> testing_indices = get_testing_instances_indices();

    if(shuffle_batches_instances) random_shuffle(testing_indices.begin(), testing_indices.end());

    return testing_indices.split(batch_instances_number);
}


/// Returns the number of instances in the data set which will be used for training.

size_t DataSet::get_training_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t training_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
       if(instances_uses[i] == Training)
       {
          training_instances_number++;
       }
    }

    return training_instances_number;
}


/// Returns the number of instances in the data set which will be used for selection.

size_t DataSet::get_selection_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t selection_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
       if(instances_uses[i] == Selection)
       {
          selection_instances_number++;
       }
    }

    return selection_instances_number;
}


/// Returns the number of instances in the data set which will be used for testing.

size_t DataSet::get_testing_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t testing_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
       if(instances_uses[i] == Testing)
       {
          testing_instances_number++;
       }
    }

    return testing_instances_number;
}


/// Returns the total number of training, selection and testing instances,
/// i.e. those which are not "Unused".

size_t DataSet::get_used_instances_number() const
{
    const size_t instances_number = get_instances_number();
    const size_t unused_instances_number = get_unused_instances_number();

    return (instances_number - unused_instances_number);
}


/// Returns the number of instances in the data set which will neither be used
/// for training, selection or testing.

size_t DataSet::get_unused_instances_number() const
{
    const size_t instances_number = get_instances_number();

    size_t unused_instances_number = 0;

    for(size_t i = 0; i < instances_number; i++)
    {
       if(instances_uses[i] == UnusedInstance)
       {
          unused_instances_number++;
       }
    }

    return unused_instances_number;
}


/// Sets all the instances in the data set for training.

void DataSet::set_training()
{
   const size_t instances_number = get_instances_number();

   for(size_t i = 0; i < instances_number; i++)
   {
       instances_uses[i] = Training;
   }
}


/// Sets all the instances in the data set for selection.

void DataSet::set_selection()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        instances_uses[i] = Selection;
    }
}


/// Sets all the instances in the data set for testing.

void DataSet::set_testing()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        instances_uses[i] = Testing;
    }
}


/// Sets instances with given indices in the data set for training.
/// @param indices Indices vector with the index of instances in the data set for training.

void DataSet::set_training(const Vector<size_t>& indices)
{
    size_t index = 0;

    for(size_t i = 0; i < indices.size(); i++)
    {
        index = indices[i];

        instances_uses[index] = Training;
    }
}


/// Sets instances with given indices in the data set for selection.
/// @param indices Indices vector with the index of instances in the data set for selection.

void DataSet::set_selection(const Vector<size_t>& indices)
{
    size_t index = 0;

    for(size_t i = 0; i < indices.size(); i++)
    {
        index = indices[i];

        instances_uses[index] = Selection;
    }
}


/// Sets instances with given indices in the data set for testing.
/// @param indices Indices vector with the index of instances in the data set for testing.

void DataSet::set_testing(const Vector<size_t>& indices)
{
    size_t index = 0;

    for(size_t i = 0; i < indices.size(); i++)
    {
        index = indices[i];

        instances_uses[index] = Testing;
    }
}


/// Sets all the instances in the data set for unused.

void DataSet::set_instances_unused()
{
    const size_t instances_number = get_instances_number();

    for(size_t i = 0; i < instances_number; i++)
    {
        instances_uses[i] = UnusedInstance;
    }
}


/// Sets instances with given indices in the data set for unused.
/// @param indices Indices vector with the index of instances in the data set for unused.

void DataSet::set_instances_unused(const Vector<size_t>& indices)
{
    for(int i = 0; i < static_cast<int>(indices.size()); i++)
    {
        const size_t index = indices[static_cast<size_t>(i)];

        instances_uses[index] = UnusedInstance;
    }
}


/// Sets the use of a single instance.
/// @param index Index of instance.
/// @param new_use Use for that instance.

void DataSet::set_instance_use(const size_t& index, const InstanceUse& new_use)
{
    instances_uses[index] = new_use;

}


/// Sets the use of a single instance from a string.
/// @param index Index of instance.
/// @param new_use String with the use name("Training", "Selection", "Testing" or "Unused")

void DataSet::set_instance_use(const size_t& index, const string& new_use)
{
    if(new_use == "Training")
    {
       instances_uses[index] = Training;
    }
    else if(new_use == "Selection")
    {
       instances_uses[index] = Selection;
    }
    else if(new_use == "Testing")
    {
       instances_uses[index] = Testing;
    }
    else if(new_use == "Unused")
    {
       instances_uses[index] = UnusedInstance;
    }
    else
    {
       ostringstream buffer;

       buffer << "OpenNN Exception DataSet class.\n"
              << "void set_instance_use(const string&) method.\n"
              << "Unknown use: " << new_use << "\n";

       throw logic_error(buffer.str());
    }
}


/// Sets new uses to all the instances from a single vector.
/// @param new_uses Vector of use structures.
/// The size of given vector must be equal to the number of instances.

void DataSet::set_instances_uses(const Vector<InstanceUse>& new_uses)
{
    const size_t instances_number = get_instances_number();

   #ifdef __OPENNN_DEBUG__

   const size_t new_uses_size = new_uses.size();

   if(new_uses_size != instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instances_uses(const Vector<Use>&) method.\n"
             << "Size of uses(" << new_uses_size << ") must be equal to number of instances(" << instances_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < instances_number; i++)
   {
       instances_uses[i] = new_uses[i];
   }
}


/// Sets new uses to all the instances from a single vector of strings.
/// @param new_uses Vector of use strings.
/// Possible values for the elements are "Training", "Selection", "Testing" and "Unused".
/// The size of given vector must be equal to the number of instances.

void DataSet::set_instances_uses(const Vector<string>& new_uses)
{
    const size_t instances_number = get_instances_number();

    ostringstream buffer;

   #ifdef __OPENNN_DEBUG__

   const size_t new_uses_size = new_uses.size();

   if(new_uses_size != instances_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instances_uses(const Vector<string>&) method.\n"
             << "Size of uses(" << new_uses_size << ") must be equal to number of instances(" << instances_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < instances_number; i++)
   {
      if(new_uses[i] == "Unused")
      {
         instances_uses[i] = UnusedInstance;
      }
      else if(new_uses[i] == "Training")
      {
         instances_uses[i] = Training;
      }
      else if(new_uses[i] == "Selection")
      {
         instances_uses[i] = Selection;
      }
      else if(new_uses[i] == "Testing")
      {
         instances_uses[i] = Testing;
      }
      else
      {
         buffer << "OpenNN Exception DataSet class.\n"
                << "void set_instances_uses(const Vector<string>&) method.\n"
                << "Unknown use: " << new_uses[i] << ".\n";

         throw logic_error(buffer.str());
      }
   }
}


/// Creates new training, selection and testing indices at random.
/// @param training_instances_ratio Ratio of training instances in the data set.
/// @param selection_instances_ratio Ratio of selection instances in the data set.
/// @param testing_instances_ratio Ratio of testing instances in the data set.

void DataSet::split_instances_random(const double& training_instances_ratio,
                           const double& selection_instances_ratio,
                           const double& testing_instances_ratio)
{
   const size_t used_instances_number = get_used_instances_number();

   if(used_instances_number == 0) return;

   const double total_ratio = training_instances_ratio + selection_instances_ratio + testing_instances_ratio;

   // Get number of instances for training, selection and testing

   const size_t selection_instances_number = static_cast<size_t>(selection_instances_ratio*used_instances_number/total_ratio);
   const size_t testing_instances_number = static_cast<size_t>(testing_instances_ratio*used_instances_number/total_ratio);
   const size_t training_instances_number = used_instances_number - selection_instances_number - testing_instances_number;

   const size_t sum_instances_number = training_instances_number + selection_instances_number + testing_instances_number;

   if(sum_instances_number != used_instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Warning: DataSet class.\n"
             << "void split_instances_random(const double&, const double&, const double&) method.\n"
             << "Sum of numbers of training, selection and testing instances is not equal to number of used instances.\n";

      throw logic_error(buffer.str());
   }

   const size_t instances_number = get_instances_number();

   Vector<size_t> indices(0, 1, instances_number-1);
   random_shuffle(indices.begin(), indices.end());

   size_t i = 0;
   size_t index;

   // Training

   size_t count_training = 0;

   while(count_training != training_instances_number)
   {
      index = indices[i];

      if(instances_uses[index] != UnusedInstance)
      {
        instances_uses[index]= Training;
        count_training++;
      }

      i++;
   }

   // Selection

   size_t count_selection = 0;

   while(count_selection != selection_instances_number)
   {
      index = indices[i];

      if(instances_uses[index] != UnusedInstance)
      {
        instances_uses[index] = Selection;
        count_selection++;
      }

      i++;
   }

   // Testing

   size_t count_testing = 0;

   while(count_testing != testing_instances_number)
   {
      index = indices[i];

      if(instances_uses[index] != UnusedInstance)
      {
            instances_uses[index] = Testing;
            count_testing++;
      }

      i++;
   }
}


/// Creates new training, selection and testing indices with sequential indices.
/// @param training_instances_ratio Ratio of training instances in the data set.
/// @param selection_instances_ratio Ratio of selection instances in the data set.
/// @param testing_instances_ratio Ratio of testing instances in the data set.

void DataSet::split_instances_sequential(const double& training_instances_ratio,
                                         const double& selection_instances_ratio,
                                         const double& testing_instances_ratio)
{
   const size_t used_instances_number = get_used_instances_number();

   if(used_instances_number == 0) return;

   const double total_ratio = training_instances_ratio + selection_instances_ratio + testing_instances_ratio;

   // Get number of instances for training, selection and testing

   const size_t selection_instances_number = static_cast<size_t>(selection_instances_ratio*used_instances_number/total_ratio);
   const size_t testing_instances_number = static_cast<size_t>(testing_instances_ratio*used_instances_number/total_ratio);
   const size_t training_instances_number = used_instances_number - selection_instances_number - testing_instances_number;

   const size_t sum_instances_number = training_instances_number + selection_instances_number + testing_instances_number;

   if(sum_instances_number != used_instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Warning: Instances class.\n"
             << "void split_instances_sequential(const double&, const double&, const double&) method.\n"
             << "Sum of numbers of training, selection and testing instances is not equal to number of used instances.\n";

      throw logic_error(buffer.str());
   }

   size_t i = 0;

   // Training

   size_t count_training = 0;

   while(count_training != training_instances_number)
   {
      if(instances_uses[i] != UnusedInstance)
      {
        instances_uses[i] = Training;
        count_training++;
      }

      i++;
   }

   // Selection

   size_t count_selection = 0;

   while(count_selection != selection_instances_number)
   {
      if(instances_uses[i] != UnusedInstance)
      {
        instances_uses[i] = Selection;
        count_selection++;
      }

      i++;
   }

   // Testing

   size_t count_testing = 0;

   while(count_testing != testing_instances_number)
   {
      if(instances_uses[i] != UnusedInstance)
      {
            instances_uses[i] = Testing;
            count_testing++;
      }
      i++;
   }
}


void DataSet::set_batch_instances_number(const size_t& new_batch_instances_number)
{
    batch_instances_number = new_batch_instances_number;
}


/// Changes instances for selection by instances for testing.

void DataSet::set_selection_to_testing_instances()
{
    instances_uses.replace_value(Selection, Testing);
}


/// Changes instances for testing by instances for selection.

void DataSet::set_testing_to_selection_instances()
{
    instances_uses.replace_value(Testing, Selection);
}


/// This method separates the dataset into n-groups to validate a model with limited data.
/// @param k Number of folds that a given data sample is given to be split into.
/// @param fold_index.
/// @todo Low priority

void DataSet::set_k_fold_cross_validation_instances_uses(const size_t& k, const size_t& fold_index)
{
    const size_t instances_number = get_instances_number();

    const size_t fold_size = instances_number/k;

    const size_t start = fold_index*fold_size;
    const size_t end = start + fold_size;

    split_instances_random(1, 0, 0);

    for(size_t i = start; i < end; i++)
    {
        instances_uses[i] = Testing;
    }
}



void DataSet::set_default_columns_uses()
{
   const size_t size = columns.size();

   if(size == 0)
   {
        return;
   }
   else if(size == 1)
   {
        columns[0].set_use(UnusedVariable);
   }
   else
   {
       set_input();

       columns[size-1].set_use(Target);

       const size_t inputs_number = get_input_variables_number();
       const size_t targets_number = get_target_variables_number();

       inputs_dimensions.set(Vector<size_t>({inputs_number}));

       targets_dimensions.set(Vector<size_t>({targets_number}));
   }
}



void DataSet::set_default_columns_names()
{
   const size_t size = columns.size();

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
       size_t input_index = 1;
       size_t target_index = 2;

       for(size_t i = 0; i < size; i++)
       {
           if(columns[i].column_use == Input)
           {
               columns[i].name = "input_" + std::to_string(input_index);
               input_index++;
           }
           else if(columns[i].column_use == Target)
           {
               columns[i].name = "target_" + std::to_string(target_index);
               target_index++;
           }
       }
   }
}




/// Returns the use of a single variable.
/// @param index Index of variable.

DataSet::VariableUse DataSet::get_variable_use(const size_t& index) const
{
    return get_variables_uses()[index];
}


/// Returns a vector containing the use of the column, without taking into account the categories.

DataSet::VariableUse DataSet::get_column_use(const size_t & index) const
{
    return columns[index].column_use;
}


Vector<DataSet::VariableUse> DataSet::get_columns_uses() const
{
    const size_t columns_number = get_columns_number();

    Vector<DataSet::VariableUse> columns_uses(columns_number);

    for (size_t i = 0; i < columns_number; i++)
    {
        columns_uses[i] = columns[i].column_use;
    }

    return columns_uses;
}


/// Returns a vector containing the use of each column, including the categories.
/// The size of the vector is equal to the number of variables.

Vector<DataSet::VariableUse> DataSet::get_variables_uses() const
{
    const size_t columns_number = get_columns_number();
    const size_t variables_number = get_variables_number();

    Vector<VariableUse> variables_uses(variables_number);

    size_t index = 0;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(columns[i].type == Categorical)
        {
            variables_uses.embed(index, columns[i].categories_uses);
            index += columns[i].categories.size();
        }
        else
        {
            variables_uses[index] = columns[i].column_use;
            index++;
        }
    }

    return variables_uses;
}


/// Returns the name of a single variable in the data set.
/// @param index Index of variable.

string DataSet::get_variable_name(const size_t& variable_index) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = get_variables_number();

   if(variable_index >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "string& get_variable_name(const size_t) method.\n"
             << "Index of variable("<<variable_index<<") must be less than number of variables("<<variables_number<<").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t columns_number = get_columns_number();

   size_t index = 0;

   for(size_t i = 0; i < columns_number; i++)
   {
       if(columns[i].type == Categorical)
       {
           for(size_t j = 0; j < columns[i].get_categories_number(); j++)
           {
               if(index == variable_index)
               {
                   return columns[i].categories[j];
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
               return columns[i].name;
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

Vector<string> DataSet::get_variables_names() const
{
    const size_t variables_number = get_variables_number();

    Vector<string> variables_names(variables_number);

    size_t index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].type == Categorical)
        {
            variables_names.embed(index, columns[i].categories);
            index += columns[i].categories.size();
        }
        else
        {
            variables_names[index] = columns[i].name;
            index++;
        }
    }

    return variables_names;
}


/// Returns the names of the input variables in the data set.
/// The size of the vector is the number of input variables.

Vector<string> DataSet::get_input_variables_names() const
{
   const size_t input_variables_number = get_input_variables_number();

   const Vector<size_t> input_columns_indices = get_input_columns_indices();

   Vector<string> input_variables_names(input_variables_number);

   size_t index = 0;

   for(size_t i = 0; i < input_columns_indices.size(); i++)
   {
       size_t input_index = input_columns_indices[i];

       const Vector<string> current_used_variables_names = columns[input_index].get_used_variables_names();

       input_variables_names.embed(index, current_used_variables_names);

       index += current_used_variables_names.size();
   }

   return input_variables_names;
}


/// Returns the names of the target variables in the data set.
/// The size of the vector is the number of target variables.

Vector<string> DataSet::get_target_variables_names() const
{
    const size_t target_variables_number = get_target_variables_number();

    const Vector<size_t> target_columns_indices = get_target_columns_indices();

    Vector<string> target_variables_names(target_variables_number);

    size_t index = 0;

    for(size_t i = 0; i < target_columns_indices.size(); i++)
    {
        size_t target_index = target_columns_indices[i];

        const Vector<string> current_used_variables_names = columns[target_index].get_used_variables_names();

        target_variables_names.embed(index, current_used_variables_names);

        index += current_used_variables_names.size();
    }

    return target_variables_names;
}


/// Returns the dimensions of the input variables.

Vector<size_t> DataSet::get_input_variables_dimensions() const
{
    return inputs_dimensions;
}


/// Returns the dimesions of the target variables.

Vector<size_t> DataSet::get_target_variables_dimensions() const
{
    return targets_dimensions;
}


/// Returns the number of variables which are either input nor target.

size_t DataSet::get_used_variables_number() const
{
    const size_t variables_number = get_variables_number();

    const size_t unused_variables_number = get_unused_variables_number();

    return (variables_number - unused_variables_number);
}


/// Returns a indices vector with the positions of the inputs.

Vector<size_t> DataSet::get_input_columns_indices() const
{ 
    const size_t input_columns_number = get_input_columns_number();

    Vector<size_t> input_columns_indices(input_columns_number);

    size_t index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == Input)
        {
            input_columns_indices[index] = i;
            index++;
        }
    }

    return input_columns_indices;
}


/// Returns a indices vector with the positions of the targets.

Vector<size_t> DataSet::get_target_columns_indices() const
{
    const size_t target_columns_number = get_target_columns_number();

    Vector<size_t> target_columns_indices(target_columns_number);

    size_t index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == Target)
        {
            target_columns_indices[index] = i;
            index++;
        }
    }

    return target_columns_indices;
}


/// Returns a indices vector with the positions of the unused columns.

Vector<size_t> DataSet::get_unused_columns_indices() const
{
    const size_t unused_columns_number = get_unused_columns_number();

    Vector<size_t> unused_columns_indices(unused_columns_number);

    size_t index = 0;

    for(size_t i = 0; i < unused_columns_number; i++)
    {

        if(columns[i].column_use == UnusedVariable)
        {
            unused_columns_indices[index] = i;
            index++;
        }
    }

    return unused_columns_indices;
}


/// Returns a indices vector with the positions of the used columns.

Vector<size_t> DataSet::get_used_columns_indices() const
{

    const size_t variables_number = get_variables_number();

    const size_t used_variables_number = get_used_variables_number();

    Vector<size_t> used_indices(used_variables_number);

    size_t index = 0;

    for(size_t i = 0; i < variables_number; i++)
    {
        if(columns[i].column_use  == Input
        || columns[i].column_use  == Target
        || columns[i].column_use  == Time)
        {
            used_indices[index] = i;
            index++;
        }
    }

    return used_indices;
}


/// Returns a string vector that contains the names of the columns.

Vector<string> DataSet::get_columns_names() const
{
    const size_t columns_number = get_columns_number();

    Vector<string> columns_names(columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        columns_names[i] = columns[i].name;
    }

    return columns_names;
}


Vector<string> DataSet::get_input_columns_names() const
{
    const size_t input_columns_number = get_input_columns_number();

    Vector<string> input_columns_names(input_columns_number);

    size_t index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == Input)
        {
            input_columns_names[index] = columns[i].name;
            index++;
        }
    }

    return input_columns_names;
}


Vector<string> DataSet::get_target_columns_names() const
{
    const size_t target_columns_number = get_target_columns_number();

    Vector<string> target_columns_names(target_columns_number);

    size_t index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == Target)
        {
            target_columns_names[index] = columns[i].name;
            index++;
        }
    }

    return target_columns_names;

}


Vector<string> DataSet::get_used_columns_names() const
{
    const size_t columns_number = get_columns_number();
    const size_t used_columns_number = get_used_columns_number();

    Vector<string> names(used_columns_number);

    size_t index = 0 ;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(columns[i].column_use != UnusedVariable)
        {
            names[index] = columns[i].name;
            index++;
        }
    }

    return names;
}

size_t DataSet::get_input_columns_number() const
{
    size_t input_columns_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == Input)
        {
            input_columns_number++;
        }
    }

    return input_columns_number;
}


size_t DataSet::get_target_columns_number() const
{
    size_t target_columns_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == Target)
        {
            target_columns_number++;
        }
    }

    return target_columns_number;
}


size_t DataSet::get_time_columns_number() const
{
    size_t time_columns_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == Time)
        {
            time_columns_number++;
        }
    }

    return time_columns_number;
}


size_t DataSet::get_unused_columns_number() const
{
    size_t unused_columns_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use == UnusedVariable)
        {
            unused_columns_number++;
        }
    }

    return unused_columns_number;
}


size_t DataSet::get_used_columns_number() const
{
    size_t used_columns_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].column_use != UnusedVariable)
        {
            used_columns_number++;
        }
    }

    return used_columns_number;
}


Vector<DataSet::Column> DataSet::get_columns() const
{
    return columns;
}


Vector<DataSet::Column> DataSet::get_used_columns() const
{
    const Vector<size_t> used_columns_indices = get_used_columns_indices();

    return columns.get_subvector(used_columns_indices);
}


size_t DataSet::get_columns_number() const
{
    return columns.size();
}


size_t DataSet::get_variables_number() const
{
    size_t variables_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].type == Categorical)
        {
            variables_number += columns[i].categories.size();
        }
        else
        {
            variables_number++;
        }
    }

    return variables_number;
}


/// Returns the number of input variables of the data set.

size_t DataSet::get_input_variables_number() const
{
   size_t inputs_number = 0;

   for(size_t i = 0; i < columns.size(); i++)
   {
       if(columns[i].type == Categorical)
       {
           for(size_t j = 0; j < columns[i].categories_uses.size(); j++)
           {
               if(columns[i].categories_uses[j] == Input) inputs_number++;
           }
       }
       else if(columns[i].column_use == Input)
       {
            inputs_number++;            
       }
   }

   return inputs_number;
}


/// Returns the number of target variables of the data set.

size_t DataSet::get_target_variables_number() const
{
    size_t targets_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].type == Categorical)
        {
            for(size_t j = 0; j < columns[i].categories_uses.size(); j++)
            {
                if(columns[i].categories_uses[j] == Target) targets_number++;
            }

        }
        else if(columns[i].column_use == Target)
        {
             targets_number++;
        }
    }

    return targets_number;
}


/// Returns the number of variables which will neither be used as input nor as target.

size_t DataSet::get_unused_variables_number() const
{
    size_t unused_number = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        if(columns[i].type == Categorical)
        {
            for(size_t j = 0; j < columns[i].categories_uses.size(); j++)
            {
                if(columns[i].categories_uses[j] == UnusedVariable) unused_number++;
            }

        }
        else if(columns[i].column_use == UnusedVariable)
        {
             unused_number++;
        }
    }

    return unused_number;
}


/// Returns a variable index in the data set with given name.
/// @param name Name of variable.

size_t DataSet::get_variable_index(const string& name) const
{
    const Vector<string> names = get_variables_names();

    const size_t index = names.get_first_index(name);

    return index;
}


/// Returns the indices of the unused variables.

Vector<size_t> DataSet::get_unused_variables_indices() const
{
    const size_t unused_number = get_unused_variables_number();

    const Vector<size_t> unused_columns_indices = get_unused_columns_indices();

    Vector<size_t> unused_indices(unused_number);

    size_t unused_index = 0;
    size_t unused_variable_index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        const size_t current_categories_number = columns[i].get_categories_number();

        if(current_categories_number == 0 && columns[i].column_use == UnusedVariable)
        {
            unused_indices[unused_index] = i;
            unused_index++;
            unused_variable_index++;
        }
        else
        {
            for(size_t j = 0; j < current_categories_number; j++)
            {
                if(columns[i].categories_uses[j] == UnusedVariable)
                {
                    unused_indices[unused_index] = unused_variable_index;
                    unused_index++;
                }

                unused_variable_index++;
            }
        }
    }

    return unused_indices;
}


/// Returns the indices of the input variables.

Vector<size_t> DataSet::get_input_variables_indices() const
{
    const size_t inputs_number = get_input_variables_number();

    const Vector<size_t> input_columns_indices = get_input_columns_indices();

    Vector<size_t> inputs_indices(inputs_number);

    size_t input_index = 0;
    size_t input_variable_index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        const size_t current_categories_number = columns[i].get_categories_number();

        if(current_categories_number == 0 && columns[i].column_use == Input)
        {
            inputs_indices[input_index] = input_variable_index;
            input_index++;
            input_variable_index++;
        }
        else if(current_categories_number > 0)
        {
            for(size_t j = 0; j < current_categories_number; j++)
            {
                if(columns[i].categories_uses[j] == Input)
                {
                    inputs_indices[input_index] = input_variable_index;
                    input_index++;
                }

                input_variable_index++;
            }
        }
        else
        {
            input_variable_index++;
        }
    }

    return inputs_indices;
}


/// Returns the indices of the target variables.

Vector<size_t> DataSet::get_target_variables_indices() const
{
    const size_t targets_number = get_target_variables_number();

    const Vector<size_t> target_columns_indices = get_target_columns_indices();

    Vector<size_t> targets_indices(targets_number);

    size_t target_index = 0;
    size_t target_variable_index = 0;

    for(size_t i = 0; i < columns.size(); i++)
    {
        const size_t current_categories_number = columns[i].get_categories_number();

        if(current_categories_number == 0 && columns[i].column_use == Target)
        {
            targets_indices[target_index] = i;
            target_index++;
            target_variable_index++;
        }
        else if(current_categories_number > 0)
        {
            for(size_t j = 0; j < current_categories_number; j++)
            {
                if(columns[i].categories_uses[j] == Target)
                {
                    targets_indices[target_index] = target_variable_index;
                    target_index++;
                }

                target_variable_index++;
            }
        }
        else
        {
            target_variable_index++;
        }
    }

    return targets_indices;
}


void DataSet::set_columns_uses(const Vector<string>& new_columns_uses)
{
    const size_t new_columns_uses_size = new_columns_uses.size();

    if(new_columns_uses_size != columns.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception DataSet class.\n"
               << "void set_columns_uses(const Vector<string>&) method.\n"
               << "Size of columns uses (" << new_columns_uses_size << ") must be equal to columns size (" << columns.size() << "). \n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < new_columns_uses.size(); i++)
    {
        columns[i].set_use(new_columns_uses[i]);
    }

    inputs_dimensions.set(1, get_input_variables_number());
    targets_dimensions.set(1, get_target_variables_number());
}


void DataSet::set_columns_uses(const Vector<VariableUse>& new_columns_uses)
{
    const size_t new_columns_uses_size = new_columns_uses.size();

    if(new_columns_uses_size != columns.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception DataSet class.\n"
               << "void set_columns_uses(const Vector<string>&) method.\n"
               << "Size of columns uses (" << new_columns_uses_size << ") must be equal to columns size (" << columns.size() << "). \n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < new_columns_uses.size(); i++)
    {
        columns[i].set_use(new_columns_uses[i]);
    }

    inputs_dimensions.set(1, get_input_variables_number());
    targets_dimensions.set(1, get_target_variables_number());
}


void DataSet::set_columns_unused()
{
    const size_t columns_number = get_columns_number();

    for(size_t i = 0; i < columns_number; i++)
    {
        set_column_use(i, UnusedVariable);
    }
}


void DataSet::set_input_columns_unused()
{
    const size_t columns_number = get_columns_number();

    for(size_t i = 0; i < columns_number; i++)
    {
        if(columns[i].column_use == DataSet::Input) set_column_use(i, UnusedVariable);
    }
}


/// Sets the use of a single column.
/// @param index Index of column.
/// @param new_use Use for that column.

void DataSet::set_column_use(const size_t& index, const VariableUse& new_use)
{
   columns[index].column_use = new_use;
}


void DataSet::set_column_use(const string& name, const VariableUse& new_use)
{
   const size_t index = get_column_index(name);

   set_column_use(index, new_use);
}


/// This method set the name of a single variable.
/// @param index Index of variable.
/// @param new_name Name of variable.

void DataSet::set_variable_name(const size_t& variable_index, const string& new_variable_name)
{
   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = get_variables_number();

   if(variable_index >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Variables class.\n"
             << "void set_name(const size_t&, const string&) method.\n"
             << "Index of variable must be less than number of variables.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t columns_number = get_columns_number();

   size_t index = 0;

   for(size_t i = 0; i < columns_number; i++)
   {
       if(columns[i].type == Categorical)
       {
           for(size_t j = 0; j < columns[i].get_categories_number(); j++)
           {
               if(index == variable_index)
               {
                   columns[i].categories[j] = new_variable_name;
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
               columns[i].name = new_variable_name;
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

void DataSet::set_variables_names(const Vector<string>& new_variables_names)
{
#ifdef __OPENNN_DEBUG__

    const size_t variables_number = get_variables_number();

   const size_t size = new_variables_names.size();

   if(size != variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Variables class.\n"
             << "void set_names(const Vector<string>&) method.\n"
             << "Size (" << size << ") must be equal to number of variables (" << variables_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t columns_number = get_columns_number();

   size_t index = 0;

   for(size_t i = 0; i < columns_number; i++)
   {
       if(columns[i].type == Categorical)
       {
           for(size_t j = 0; j < columns[i].get_categories_number(); j++)
           {
               columns[i].categories[j] = new_variables_names[index];
               index++;
           }
       }
       else
       {
           columns[i].name = new_variables_names[index];
           index++;
       }
   }
}


/// Sets new names for the columns in the data set from a vector of strings.
/// The size of that vector must be equal to the total number of variables.
/// @param new_names Name of variables.

void DataSet::set_columns_names(const Vector<string>& new_names)
{
    const size_t new_names_size = new_names.size();
    const size_t columns_number = get_columns_number();

    if(new_names_size != columns_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void set_columns_names(const Vector<string>&).\n"
               << "Size of names (" << new_names.size() << ") is not equal to columns number (" << columns_number << ").\n";

        throw logic_error(buffer.str());
    }

    for(size_t i = 0; i < columns_number; i++)
    {
        columns[i].name = new_names[i];
    }
}


/// Sets all the variables in the data set as input variables.

void DataSet::set_input()
{
    for(size_t i = 0; i < columns.size(); i++)
    {
        columns[i].set_use(Input);
    }
}


/// Sets all the variables in the data set as target variables.

void DataSet::set_target()
{
    for(size_t i = 0; i < columns.size(); i++)
    {
        columns[i].set_use(Target);
    }
}


/// Sets all the variables in the data set as unused variables.

void DataSet::set_variables_unused()
{
    for(size_t i = 0; i < columns.size(); i++)
    {
        columns[i].set_use(UnusedVariable);
    }
}


/// Sets a new number of variables in the variables object.
/// All variables are set as inputs but the last one, which is set as targets.
/// @param new_variables_number Number of variables.

void DataSet::set_columns_number(const size_t& new_variables_number)
{
    columns.set(new_variables_number);

    set_default_columns_uses();
}


/// Sets new input dimensions in the data set.

void DataSet::set_input_variables_dimensions(const Vector<size_t>& new_inputs_dimensions)
{
    inputs_dimensions = new_inputs_dimensions;
}


/// Sets new target dimensions in the data set.

void DataSet::set_target_variables_dimensions(const Vector<size_t>& new_targets_dimensions)
{
    targets_dimensions = new_targets_dimensions;
}


/// Returns true if the data set is a binary classification problem, false otherwise.

bool DataSet::is_binary_classification() const
{
    if(get_target_variables_number() != 1)
    {
        return false;
    }

    if(!get_target_data().is_binary())
    {
        return false;
    }

    return true;
}


/// Returns true if the data set is a multiple classification problem, false otherwise.

bool DataSet::is_multiple_classification() const
{
    const Matrix<double> targets = get_target_data();

    if(!targets.is_binary())
    {
        return false;
    }

    for(size_t i = 0; i < targets.get_rows_number(); i++)
    {
        if(targets.get_row(i).calculate_sum() == 0.0)
        {
            return false;
        }
    }

    return true;
}


/// Returns true if the data matrix is empty, and false otherwise.

bool DataSet::is_empty() const
{
   return data.empty();
}


/// Returns a reference to the data matrix in the data set.
/// The number of rows is equal to the number of
/// The number of columns is equal to the number of variables.

const Matrix<double>& DataSet::get_data() const
{
   return data;
}


/// Returns a reference to the data matrixXd in the data set.
/// The number of rows is equal to the number of
/// The number of columns is equal to the number of variables

const Eigen::MatrixXd DataSet::get_data_eigen() const
{
    const size_t variables_number = data.get_columns_number();
    const size_t instances_number = data.get_rows_number();

    Matrix<double> copy(data);

    const Eigen::Map<Eigen::MatrixXd> matrix_eigen(static_cast<double*>(copy.data()), static_cast<int>(instances_number), static_cast<int>(variables_number));

    return matrix_eigen;
}


/// Returns a reference to the time series data matrix in the data set.
/// Only for time series problems.

const Matrix<double>& DataSet::get_time_series_data() const
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
        case Space: return ' ';

        case Tab: return '\t';

        case Comma: return ',';

        case Semicolon: return ';';
    }

    return char();
}


/// Returns the string which will be used as separator in the data file.

string DataSet::get_separator_string() const
{
    switch(separator)
    {
        case Space: return "Space";

        case Tab: return "Tab";

        case Comma: return "Comma";

        case Semicolon: return "Semicolon";
    }

    return string();
}


/// Returns the string which will be used as label for the missing values in the data file.

const string& DataSet::get_missing_values_label() const
{
    return missing_values_label;
}


/// Returns the number of lags to be used in a time series prediction application.

const size_t& DataSet::get_lags_number() const
{
    return(lags_number);
}


/// Returns the number of steps ahead to be used in a time series prediction application.

const size_t& DataSet::get_steps_ahead() const
{
    return(steps_ahead);
}


/// Returns the indices of the time variables in the data set.

const size_t& DataSet::get_time_index() const
{
    return time_index;
}


/// Returns a value of the scaling-unscaling method enumeration from a string containing the name of that method.
/// @param scaling_unscaling_method String with the name of the scaling and unscaling method.

DataSet::ScalingUnscalingMethod DataSet::get_scaling_unscaling_method(const string& scaling_unscaling_method)
{
    if(scaling_unscaling_method == "NoScaling")
    {
        return(NoScaling);
    }
    else if(scaling_unscaling_method == "NoUnscaling")
    {
        return(NoUnscaling);
    }
    else if(scaling_unscaling_method == "MinimumMaximum")
    {
        return(MinimumMaximum);
    }
    else if(scaling_unscaling_method == "Logarithmic")
    {
        return(Logarithmic);
    }
    else if(scaling_unscaling_method == "MeanStandardDeviation")
    {
        return(MeanStandardDeviation);
    }
    else if(scaling_unscaling_method == "StandardDeviation")
    {
        return(StandardDeviation);
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


/// Returns a matrix with the training instances in the data set.
/// The number of rows is the number of training
/// The number of columns is the number of variables.

Matrix<double> DataSet::get_training_data() const
{
   const size_t variables_number = get_variables_number();

   Vector<size_t> variables_indices(0, 1,variables_number-1);

   const Vector<size_t> training_indices = get_training_instances_indices();

   return(data.get_submatrix(training_indices, variables_indices));
}


/// Returns a eigen matrixXd with the training instances in the data set.
/// The number of rows is the number of training
/// The number of columns is the number of variables.

Eigen::MatrixXd DataSet::get_training_data_eigen() const
{
    Matrix<double> data = get_training_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a matrix with the selection instances in the data set.
/// The number of rows is the number of selection
/// The number of columns is the number of variables.

Matrix<double> DataSet::get_selection_data() const
{
   const size_t variables_number = get_variables_number();

   const Vector<size_t> selection_indices = get_selection_instances_indices();

   Vector<size_t> variables_indices(0, 1,variables_number-1);

   return(data.get_submatrix(selection_indices, variables_indices));
}


/// Returns a eigen matrixXd with the selection instances in the data set.
/// The number of rows is the number of selection
/// The number of columns is the number of variables.

Eigen::MatrixXd DataSet::get_selection_data_eigen() const
{
    Matrix<double> data = get_selection_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a matrix with the testing instances in the data set.
/// The number of rows is the number of testing
/// The number of columns is the number of variables.

Matrix<double> DataSet::get_testing_data() const
{
   const size_t variables_number = get_variables_number();
   Vector<size_t> variables_indices(0, 1,variables_number-1);

   const Vector<size_t> testing_indices = get_testing_instances_indices();

   return(data.get_submatrix(testing_indices, variables_indices));
}


/// Returns a eigen matrixXd with the testing instances in the data set.
/// The number of rows is the number of testing
/// The number of columns is the number of variables.

Eigen::MatrixXd DataSet::get_testing_data_eigen() const
{
    Matrix<double> data = get_testing_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a matrix with the input variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Matrix<double> DataSet::get_input_data() const
{
   const size_t instances_number = get_instances_number();

   const Vector<size_t> indices(0, 1,instances_number-1);

   const Vector<size_t> inputs_indices = get_input_variables_indices();

   return(data.get_submatrix(indices, inputs_indices));
}


/// Returns a eigen matrixXd with the input variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Eigen::MatrixXd DataSet::get_input_data_eigen() const
{
    Matrix<double> data = get_input_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a matrix with the target variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of target variables.

Matrix<double> DataSet::get_target_data() const
{
   const size_t instances_number = get_instances_number();
   const Vector<size_t> indices(0, 1, instances_number-1);

   const Vector<size_t> targets_indices = get_target_variables_indices();

   return(data.get_submatrix(indices, targets_indices));
}


/// Returns a eigen matrixXd with the target variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of target variables.

Eigen::MatrixXd DataSet::get_target_data_eigen() const
{
    Matrix<double> data = get_target_data();
    const size_t rows_number = data.get_rows_number();
    const size_t columns_number = data.get_columns_number();

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a tensor with the input variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Tensor<double> DataSet::get_input_data(const Vector<size_t>& instances_indices) const
{
    const Vector<size_t> inputs_dimensions = get_input_variables_dimensions();

    const Vector<size_t> inputs_indices = get_input_variables_indices();

    return data.get_tensor(instances_indices, inputs_indices, inputs_dimensions);
}


/// Returns a tensor with the target variables in the data set.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Tensor<double> DataSet::get_target_data(const Vector<size_t>& instances_indices) const
{
    const Vector<size_t> targets_dimensions = get_target_variables_dimensions();

    const Vector<size_t> targets_indices = get_target_variables_indices();

    return data.get_tensor(instances_indices, targets_indices, targets_dimensions);
}


/// Returns a matrix with the input variables in the data set is float type.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Matrix<float> DataSet::get_input_data_float(const Vector<size_t>& instances_indices) const
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();

    const size_t instances_number = instances_indices.size();
    const size_t inputs_number = inputs_indices.size();

    Matrix<float> inputs_float(instances_number, inputs_number);

    size_t instance_index;
    size_t input_index;

    for(size_t i = 0; i < instances_number; i++)
    {
       instance_index = instances_indices[i];

       for(size_t j = 0; j < inputs_number; j++)
       {
          input_index = inputs_indices[j];
          inputs_float(i,j) = static_cast<float>(data(instance_index,input_index));
       }
    }

    return(inputs_float);
}


/// Returns a matrix with the target variables in the data set is float type.
/// The number of rows is the number of
/// The number of columns is the number of input variables.

Matrix<float> DataSet::get_target_data_float(const Vector<size_t>& instances_indices) const
{
    const Vector<size_t> targets_indices = get_target_variables_indices();

    const size_t instances_number = instances_indices.size();

    const size_t targets_number = targets_indices.size();

    Matrix<float> targets_float(instances_number, targets_number);

    size_t instance_index;
    size_t target_index;

    for(size_t i = 0; i < instances_number; i++)
    {
       instance_index = instances_indices[i];

       for(size_t j = 0; j < targets_number; j++)
       {
          target_index = targets_indices[j];

          targets_float(i,j) = static_cast<float>(data(instance_index,target_index));
       }
    }

    return targets_float;
}


/// Returns a matrix with training instances and input variables.
/// The number of rows is the number of training
/// The number of columns is the number of input variables.

Tensor<double> DataSet::get_training_input_data() const
{
    const Vector<size_t> training_indices = get_training_instances_indices();

    const Vector<size_t> inputs_indices = get_input_variables_indices();

    const Vector<size_t> inputs_dimensions = get_input_variables_dimensions();

    return data.get_tensor(training_indices, inputs_indices, inputs_dimensions);
}


/// Returns a eigen matrixXd with training instances and input variables.
/// The number of rows is the number of training
/// The number of columns is the number of input variables.

Eigen::MatrixXd DataSet::get_training_input_data_eigen() const
{
    Tensor<double> data = get_training_input_data();

    const size_t rows_number = data.get_dimension(0);
    const size_t columns_number = data.get_dimension(1);

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a tensor with training instances and target variables.
/// The number of rows is the number of training
/// The number of columns is the number of target variables.

Tensor<double> DataSet::get_training_target_data() const
{
   const Vector<size_t> training_indices = get_training_instances_indices();

   const Vector<size_t> targets_indices = get_target_variables_indices();

   return data.get_tensor(training_indices, targets_indices, get_target_variables_dimensions());
}


/// Returns a eigen matrixXd with training instances and target variables.
/// The number of rows is the number of training
/// The number of columns is the number of target variables.

Eigen::MatrixXd DataSet::get_training_target_data_eigen() const
{
    Tensor<double> data = get_training_target_data();
    const size_t rows_number = data.get_dimension(0);
    const size_t columns_number = data.get_dimension(1);

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a tensor with selection instances and input variables.
/// The number of rows is the number of selection
/// The number of columns is the number of input variables.

Tensor<double> DataSet::get_selection_input_data() const
{
   const Vector<size_t> selection_indices = get_selection_instances_indices();

   const Vector<size_t> inputs_indices = get_input_variables_indices();

   return data.get_tensor(selection_indices, inputs_indices, get_input_variables_dimensions());
}


/// Returns a eigen matrixXd with selection instances and input variables.
/// The number of rows is the number of selection
/// The number of columns is the number of input variables.

Eigen::MatrixXd DataSet::get_selection_input_data_eigen() const
{
    Tensor<double> data = get_selection_input_data();
    const size_t rows_number = data.get_dimension(0);
    const size_t columns_number = data.get_dimension(0);

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a tensor with selection instances and target variables.
/// The number of rows is the number of selection
/// The number of columns is the number of target variables.

Tensor<double> DataSet::get_selection_target_data() const
{
   const Vector<size_t> selection_indices = get_selection_instances_indices();

   const Vector<size_t> targets_indices = get_target_variables_indices();

   return data.get_tensor(selection_indices, targets_indices, get_target_variables_dimensions());
}


/// Returns a eigen matrixXd with selection instances and target variables.
/// The number of rows is the number of selection
/// The number of columns is the number of target variables.

Eigen::MatrixXd DataSet::get_selection_target_data_eigen() const
{
    Tensor<double> data = get_selection_target_data();
    const size_t rows_number = data.get_dimension(0);
    const size_t columns_number = data.get_dimension(1);

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a tensor with testing instances and input variables.
/// The number of rows is the number of testing
/// The number of columns is the number of input variables.

Tensor<double> DataSet::get_testing_input_data() const
{
   const Vector<size_t> inputs_indices = get_input_variables_indices();

   const Vector<size_t> testing_indices = get_testing_instances_indices();

   return data.get_tensor(testing_indices, inputs_indices, get_input_variables_dimensions());
}


/// Returns a eigen matrixXd with testing instances and input variables.
/// The number of rows is the number of testing
/// The number of columns is the number of input variables.

Eigen::MatrixXd DataSet::get_testing_input_data_eigen() const
{
    Tensor<double> data = get_testing_input_data();
    const size_t rows_number = data.get_dimension(0);
    const size_t columns_number = data.get_dimension(1);

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns a tensor with testing instances and target variables.
/// The number of rows is the number of testing
/// The number of columns is the number of target variables.

Tensor<double> DataSet::get_testing_target_data() const
{
   const Vector<size_t> targets_indices = get_target_variables_indices();

   const Vector<size_t> testing_indices = get_testing_instances_indices();

   return data.get_tensor(testing_indices, targets_indices, get_target_variables_dimensions());
}


/// Returns a eigen matrixXd with testing instances and target variables.
/// The number of rows is the number of testing
/// The number of columns is the number of target variables.

Eigen::MatrixXd DataSet::get_testing_target_data_eigen() const
{
    Tensor<double> data = get_testing_target_data();
    const size_t rows_number = data.get_dimension(0);
    const size_t columns_number = data.get_dimension(1);

    const Eigen::Map<Eigen::MatrixXd> data_eigen(static_cast<double*>(data.data()), static_cast<int>(rows_number), static_cast<int>(columns_number));

    return data_eigen;
}


/// Returns the inputs and target values of a single instance in the data set.
/// @param index Index of the instance.

Vector<double> DataSet::get_instance_data(const size_t& index) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = get_instances_number();

   if(index >= instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_instance(const size_t&) const method.\n"
             << "Index of instance (" << index << ") must be less than number of instances (" << instances_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Get instance

   return data.get_row(index);
}


/// Returns the inputs and target values of a single instance in the data set.
/// @param instance_index Index of the instance.
/// @param variables_indices Indices of the variables.

Vector<double> DataSet::get_instance_data(const size_t& instance_index, const Vector<size_t>& variables_indices) const
{
    #ifdef __OPENNN_DEBUG__

   const size_t instances_number = get_instances_number();

   if(instance_index >= instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_instance(const size_t&, const Vector<size_t>&) const method.\n"
             << "Index of instance must be less than number of \n";

      throw logic_error(buffer.str());
   }

   #endif

   return data.get_row(instance_index, variables_indices);
}


/// Returns the inputs values of a single instance in the data set.
/// @param instance_index Index of the instance.

Tensor<double> DataSet::get_instance_input_data(const size_t & instance_index) const
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();
    const Vector<size_t> inputs_dimension = get_input_variables_dimensions();

    return data.get_tensor(Vector<size_t>({instance_index}),inputs_indices,inputs_dimension);
}


/// Returns the target values of a single instance in the data set.
/// @param instance_index Index of the instance.

Tensor<double> DataSet::get_instance_target_data(const size_t & instance_index) const
{
    const Vector<size_t> targets_indices = get_target_variables_indices();
    const Vector<size_t> targets_dimension = get_target_variables_dimensions();

    return data.get_tensor(Vector<size_t>({instance_index}),targets_indices,targets_dimension);
}


size_t DataSet::get_column_index(const string& column_name) const
{
    const size_t columns_number = get_columns_number();

    for(size_t i = 0; i < columns_number; i++)
    {
        if(columns[i].name == column_name)
        {
            return i;
        }
    }

    ostringstream buffer;

    buffer << "OpenNN Exception: DataSet class.\n"
           << "size_t get_column_index(const string&&) const method.\n"
           << "Cannot find " << column_name << "\n";

    throw logic_error(buffer.str());

}


Vector<size_t> DataSet::get_variable_indices(const size_t& column_index) const
{
    size_t index = 0;

    for(size_t i = 0; i < column_index; i++)
    {
        if(columns[i].categories.size() == 0)
        {
            index++;
        }
        else
        {
            index += columns[i].categories.size();
        }
    }

    if(columns[column_index].categories.size() > 0)
    {
        Vector<size_t> variable_indices(columns[column_index].categories.size());

        for (size_t j = 0; j<columns[column_index].categories.size(); j++)
        {
            variable_indices[j] = index+j;
        }

        return variable_indices;
    }
    else
    {
        return Vector<size_t>({index});
    }
}


Matrix<double> DataSet::get_column_data(const size_t& column_index) const
{
    const Vector<size_t> variable_indices = get_variable_indices(column_index);

    return data.get_submatrix_columns(variable_indices);
}


Matrix<double> DataSet::get_column_data(const Vector<size_t>& variables_indices) const
{
    return data.get_submatrix_columns(variables_indices);
}


Matrix<double> DataSet::get_column_data(const string& column_name) const
{
    const size_t column_index = get_column_index(column_name);

    return get_column_data(column_index);
}


/// Returns all the instances of a single variable in the data set.
/// @param index Index of the variable.

Vector<double> DataSet::get_variable_data(const size_t& index) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = get_variables_number();

   if(index >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_variable(const size_t&) const method.\n"
             << "Index of variable must be less than number of \n";

      throw logic_error(buffer.str());
   }

   #endif

   return data.get_column(index);
}


/// Returns all the instances of a single variable in the data set.
/// @param variable_name Name of the variable.

Vector<double> DataSet::get_variable_data(const string& variable_name) const
{
    const Vector<string> variable_names = get_variables_names();

    const Vector<size_t> variable_index = variable_names.get_indices_equal_to(variable_name);

#ifdef __OPENNN_DEBUG__

    const size_t variables_size = variable_index.size();

    if(variables_size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " does not exist.\n";

       throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " appears more than once in the data set.\n";

       throw logic_error(buffer.str());
    }

#endif

    return(data.get_column(variable_index[0]));
}


/// Returns a given set of instances of a single variable in the data set.
/// @param variable_index Index of the variable.
/// @param instances_indices Indices of the

Vector<double> DataSet::get_variable_data(const size_t& variable_index, const Vector<size_t>& instances_indices) const
{
   #ifdef __OPENNN_DEBUG__

   const size_t variables_number = get_variables_number();

   if(variable_index >= variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<double> get_variable(const size_t&, const Vector<size_t>&) const method.\n"
             << "Index of variable must be less than number of \n";

      throw logic_error(buffer.str());
   }

   #endif

   return(data.get_column(variable_index, instances_indices));
}


/// Returns a given set of instances of a single variable in the data set.
/// @param variable_name Name of the variable.
/// @param instances_indices Indices of the

Vector<double> DataSet::get_variable_data(const string& variable_name, const Vector<size_t>& instances_indices) const
{
    const Vector<string> variable_names = get_variables_names();

    const Vector<size_t> variable_index = variable_names.get_indices_equal_to(variable_name);

#ifdef __OPENNN_DEBUG__

    const size_t variables_size = variable_index.size();

    if(variables_size == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&) const method.\n"
              << "Variable: " << variable_name << " does not exist.\n";

       throw logic_error(buffer.str());
    }

    if(variables_size > 1)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<double> get_variable(const string&, const Vector<size_t>&) const method.\n"
              << "Variable: " << variable_name << " appears more than once in the data set.\n";

       throw logic_error(buffer.str());
    }

#endif

    return(data.get_column(variable_index[0], instances_indices));
}


/// Sets zero instances and zero variables in the data set.

void DataSet::set()
{
   data_file_name = "";

   data.set();

   display = true;
}


/// Sets all variables from a data matrix.
/// @param new_data Data matrix.

void DataSet::set(const Matrix<double>& new_data)
{
   data_file_name = "";

   const size_t variables_number = new_data.get_columns_number();
   const size_t instances_number = new_data.get_rows_number();

   set(instances_number, variables_number);

   data = new_data;

   if(get_header_line()) set_variables_names(data.get_header());

   display = true;
}


/// Sets all variables from a data matrix.
/// @param new_data Data matrix.

void DataSet::set(const Eigen::MatrixXd& new_data)
{
   data_file_name = "";

   const size_t variables_number = static_cast<size_t>(new_data.cols());
   const size_t instances_number = static_cast<size_t>(new_data.rows());

   set(instances_number, variables_number);

   data.set(instances_number, variables_number);

   Eigen::Map<Eigen::MatrixXd> auxiliar_eigen(data.data(), static_cast<int>(instances_number), static_cast<int>(variables_number));

   auxiliar_eigen = new_data;

   if(!data.get_header().empty()) set_variables_names(data.get_header());

   display = true;
}


/// Sets new numbers of instances and variables in the inputs targets data set.
/// All the instances are set for training.
/// All the variables are set as inputs.
/// @param new_instances_number Number of
/// @param new_variables_number Number of variables.

void DataSet::set(const size_t& new_instances_number, const size_t& new_variables_number)
{
    #ifdef __OPENNN_DEBUG__

    if(new_instances_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set(const size_t&, const size_t&) method.\n"
              << "Number of instances must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    if(new_variables_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "void set(const size_t&, const size_t&) method.\n"
              << "Number of variables must be greater than zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

   data.set(new_instances_number, new_variables_number);

   columns.set(new_variables_number);

   for(size_t index = 0; index < new_variables_number-1; index++)
   {
       columns[index].name = "column_" + to_string(index);
       columns[index].column_use = Input;
       columns[index].type = Numeric;
   }

   columns[new_variables_number-1].name = "column_" + to_string(new_variables_number-1);
   columns[new_variables_number-1].column_use = Target;
   columns[new_variables_number-1].type = Numeric;

   instances_uses.set(new_instances_number);
   split_instances_random();

   display = true;
}


/// Sets new numbers of instances and inputs and target variables in the data set.
/// The variables in the data set are the number of inputs plus the number of targets.
/// @param new_instances_number Number of
/// @param new_inputs_number Number of input variables.
/// @param new_targets_number Number of target variables.

void DataSet::set(const size_t& new_instances_number,
                  const size_t& new_inputs_number,
                  const size_t& new_targets_number)
{

   data_file_name = "";

   const size_t new_variables_number = new_inputs_number + new_targets_number;

   data.set(new_instances_number, new_variables_number);

   columns.set(new_variables_number);

   for(size_t i = 0; i < new_variables_number; i++)
   {
       if(i < new_inputs_number)
       {
           columns[i].name = "column_" + to_string(i);
           columns[i].column_use = Input;
           columns[i].type = Numeric;
       }
       else
       {
           columns[i].name = "column_" + to_string(i);
           columns[i].column_use = Target;
           columns[i].type = Numeric;
       }
   }

   inputs_dimensions.set(Vector<size_t>({new_inputs_number}));
   targets_dimensions.set(Vector<size_t>({new_targets_number}));

   instances_uses.set(new_instances_number);
   split_instances_random();

   display = true;

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
    has_columns_names = false;

    separator = Comma;

    missing_values_label = "NA";

    lags_number = 0;

    steps_ahead = 0;

    display = true;

    set_default_columns_uses();

    set_default_columns_names();
}


/// Sets a new data matrix.
/// The number of rows must be equal to the number of
/// The number of columns must be equal to the number of variables.
/// Indices of all training, selection and testing instances and inputs and target variables do not change.
/// @param new_data Data matrix.

void DataSet::set_data(const Matrix<double>& new_data)
{
    data = new_data;

   set_instances_number(data.get_rows_number());

//   set_variables_number(data.get_columns_number());

   set_instances_number(data.get_rows_number());

    const size_t instances_number = data.get_rows_number();
    const size_t variables_number = data.get_columns_number();

    set(instances_number, variables_number);

}


/// Sets the name of the data file.
/// It also loads the data from that file.
/// Moreover, it sets the variables and instances objects.
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

void DataSet::set_lags_number(const size_t& new_lags_number)
{
    lags_number = new_lags_number;
}


/// Sets a new number of steps ahead to be defined for a time series prediction application.
/// When loading the data file, the time series data will be modified according to this number.
/// @param new_steps_ahead_number Number of steps ahead to be used.

void DataSet::set_steps_ahead_number(const size_t& new_steps_ahead_number)
{
    steps_ahead = new_steps_ahead_number;
}


/// Sets the new position where the time data is located in the data set.
/// @param new_time_index Position where the time data is located.

void DataSet::set_time_index(const size_t& new_time_index)
{
    time_index = new_time_index;
}


/// Sets a new number of instances in the data set.
/// All instances are also set for training.
/// The indices of the inputs and target variables do not change.
/// @param new_instances_number Number of instances.

void DataSet::set_instances_number(const size_t& new_instances_number)
{
   const size_t variables_number = get_variables_number();

   set(new_instances_number,variables_number);
}


/// Sets new inputs and target values of a single instance in the data set.
/// @param instance_index Index of the instance.
/// @param instance New inputs and target values of the instance.

void DataSet::set_instance(const size_t& instance_index, const Vector<double>& instance)
{
   #ifdef __OPENNN_DEBUG__

   const size_t instances_number = get_instances_number();

   if(instance_index >= instances_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instance(const size_t&, const Vector<double>&) method.\n"
             << "Index of instance must be less than number of \n";

      throw logic_error(buffer.str());
   }

   const size_t size = instance.size();
   const size_t variables_number = get_variables_number();

   if(size != variables_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "void set_instance(const size_t&, const Vector<double>&) method.\n"
             << "Size(" << size << ") must be equal to number of variables(" << variables_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   data.set_row(instance_index, instance);
}


/// Removes the input of target indices of that variables with zero standard deviation.
/// It might change the size of the vectors containing the inputs and targets indices.

Vector<string> DataSet::unuse_constant_columns()
{
   const size_t columns_number = get_columns_number();

   #ifdef __OPENNN_DEBUG__

   if(columns_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: DataSet class.\n"
             << "Vector<string> unuse_constant_columns() method.\n"
             << "Number of columns is zero.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<string> constant_columns;

   for(size_t i = 0; i < columns_number; i++)
   {
      if(get_variable_use(i) == Input && data.is_column_constant(i))
      {
         set_column_use(i, DataSet::UnusedVariable);

         constant_columns.push_back(columns[i].name);
      }
   }

    return constant_columns;
}


/// Removes the training, selection and testing indices of that instances which are repeated in the data matrix.
/// It might change the size of the vectors containing the training, selection and testing indices.

Vector<size_t> DataSet::unuse_repeated_instances()
{
    const size_t instances_number = get_instances_number();

    #ifdef __OPENNN_DEBUG__

    if(instances_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<size_t> unuse_repeated_instances() method.\n"
              << "Number of instances is zero.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Vector<size_t> repeated_instances;

    Vector<double> instance_i;
    Vector<double> instance_j;

    int i = 0;

#pragma omp parallel for private(i, instance_i, instance_j) schedule(dynamic)

    for(i = 0; i < static_cast<int>(instances_number); i++)
    {
       instance_i = get_instance_data(static_cast<size_t>(i));

       for(size_t j = static_cast<size_t>(i+1); j < instances_number; j++)
       {
          instance_j = get_instance_data(j);

          if(get_instance_use(j) != UnusedInstance && instance_j == instance_i)
          {
              set_instance_use(j, UnusedInstance);

              repeated_instances.push_back(j);
          }
       }
    }

    return repeated_instances;
}


/// Unuses those binary inputs whose positives does not correspond to any positive in the target variables.
/// @todo Low priority.

Vector<size_t> DataSet::unuse_non_significant_input_columns()
{
/*
    const Vector<size_t> inputs_indices = get_input_variables_indices();
    const size_t inputs_number = inputs_indices.size();

    const size_t target_index = get_target_variables_indices()[0];

    const size_t instances_number = get_used_instances_number();

    Vector<size_t> non_significant_variables;

    if(!is_binary_classification())
    {
        return non_significant_variables;
    }

    size_t positives = 0;

    size_t current_input_index;

    for(size_t i = 0; i < inputs_number; i++)
    {
        positives = 0;

        current_input_index = inputs_indices[i];

        if(!is_binary_variable(current_input_index)) continue;

        for(size_t j = 0; j < instances_number; j++)
        {
            if(data(j, current_input_index) == 1.0 && data(j, target_index) == 1.0)
            {
                positives++;
            }
        }

        if(positives == 0)
        {
            set_column_use(current_input_index, DataSet::UnusedVariable);
            non_significant_variables.push_back(current_input_index);
        }
    }

    return non_significant_variables;
*/

    return Vector<size_t>();
}


/// Returns a vector with the unuse variables by missing values method.
/// @param missing_ratio Ratio to find the missing variables.

Vector<string> DataSet::unuse_columns_missing_values(const double& missing_ratio)
{
    const size_t columns_number = get_columns_number();

    const size_t instances_number = get_instances_number();

    const Vector<size_t> columns_missing_values = data.count_nan_columns();

    const Vector<double> columns_missing_ratios = columns_missing_values.to_double_vector()/(static_cast<double>(instances_number)-1.0);

    Vector<string> unused_variables;

    for(size_t i = 0; i < columns_number; i++)
    {
        if(columns[i].column_use != DataSet::UnusedVariable && columns_missing_ratios[i] >= missing_ratio)
        {
            set_column_use(i, DataSet::UnusedVariable);

            unused_variables.push_back(columns[i].name);
        }
    }

    return unused_variables;
}


/// Return unused variables without correlation.
/// @param minimum_correlation Minimum correlation between variables.
/// @param nominal_variables Vector containing the classes of each categorical variable.

Vector<size_t> DataSet::unuse_uncorrelated_columns(const double& minimum_correlation)
{
    Vector<size_t> unused_columns;

    const Matrix<RegressionResults> correlations; //= calculate_input_target_columns_correlations();

    const size_t input_columns_number = get_input_columns_number();
    const size_t target_columns_number = get_target_columns_number();

    const Vector<size_t> input_columns_indices = get_input_columns_indices();

    for(size_t i = 0; i < input_columns_number; i++)
    {
        const size_t index = input_columns_indices[i];

        for(size_t j = 0; j < target_columns_number; j++)
        {
            if(columns[index].column_use != UnusedVariable && abs(correlations(i,j).correlation) < minimum_correlation)
            {
                columns[index].column_use = UnusedVariable;
                unused_columns.push_back(index);
            }
        }
    }

    return unused_columns;
}


/// Returns a histogram for each variable with a given number of bins.
/// The default number of bins is 10.
/// @param bins_number Number of bins.

Vector<Histogram> DataSet::calculate_columns_histograms(const size_t& bins_number) const
{
   const size_t used_columns_number = get_used_columns_number();
   const Vector<size_t> used_columns_indices = get_used_columns_indices();

   Vector<Histogram> histograms(used_columns_number);

   #pragma omp parallel for shared(histograms)

   for(int i = 0; i < static_cast<int>(used_columns_number); i++)
   {
       if(columns[static_cast<size_t>(i)].type == Numeric)
       {
           const size_t index = used_columns_indices[static_cast<size_t>(i)];

           const Vector<double> column = get_column_data(static_cast<size_t>(index)).to_vector();

           histograms[static_cast<size_t>(i)] = histogram(column, bins_number);
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

Vector<BoxPlot> DataSet::calculate_columns_box_plots() const
{
    const size_t used_columns_number = get_used_columns_number();
    const Vector<size_t> used_columns_indices = get_used_columns_indices();

    Vector<BoxPlot> box_plots(used_columns_number);

    #pragma omp parallel for shared(box_plots)

    for(int i = 0; i < static_cast<int>(used_columns_number); i++)
    {
        if(columns[static_cast<size_t>(i)].type == Numeric)
        {
            const size_t index = used_columns_indices[static_cast<size_t>(i)];

            const Vector<double> column = get_column_data(static_cast<size_t>(index)).to_vector();

            box_plots[static_cast<size_t>(i)] = box_plot(column);
        }
    }

    return box_plots;
}


/// Counts the number of negatives of the selected target in the training data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_training_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const Vector<size_t> training_indices = get_training_instances_indices();

    const size_t training_instances_number = training_indices.size();

    for(size_t i = 0; i < training_instances_number; i++)
    {
        const size_t training_index = training_indices[static_cast<size_t>(i)];

        if(data(training_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(training_index, target_index) != 1.0)
        {
            ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_training_negatives(const size_t&) const method.\n"
                  << "Training instance is neither a positive nor a negative: " << data(training_index, target_index) << endl;

           throw logic_error(buffer.str());
        }
    }

    return negatives;
}


/// Counts the number of negatives of the selected target in the selection data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_selection_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const size_t selection_instances_number = get_selection_instances_number();

    const Vector<size_t> selection_indices = get_selection_instances_indices();

    for(int i = 0; i < static_cast<int>(selection_instances_number); i++)
    {
        const size_t selection_index = selection_indices[static_cast<size_t>(i)];

        if(data(selection_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(selection_index, target_index) != 1.0)
        {
            ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_selection_negatives(const size_t&) const method.\n"
                  << "Selection instance is neither a positive nor a negative: " << data(selection_index, target_index) << endl;

           throw logic_error(buffer.str());
        }
    }

    return negatives;
}


/// Counts the number of negatives of the selected target in the testing data.
/// @param target_index Index of the target to evaluate.

size_t DataSet::calculate_testing_negatives(const size_t& target_index) const
{
    size_t negatives = 0;

    const size_t testing_instances_number = get_testing_instances_number();

    const Vector<size_t> testing_indices = get_testing_instances_indices();

    for(int i = 0; i < static_cast<int>(testing_instances_number); i++)
    {
        const size_t testing_index = testing_indices[static_cast<size_t>(i)];

        if(data(testing_index, target_index) == 0.0)
        {
            negatives++;
        }
        else if(data(testing_index, target_index) != 1.0)
        {
            ostringstream buffer;

           buffer << "OpenNN Exception: DataSet class.\n"
                  << "size_t calculate_selection_negatives(const size_t&) const method.\n"
                  << "Testing instance is neither a positive nor a negative: " << data(testing_index, target_index) << endl;

           throw logic_error(buffer.str());
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

Vector<Descriptives> DataSet::calculate_columns_descriptives() const
{
    return descriptives_missing_values(data);
}


/// Returns all the variables descriptives from a single matrix.
/// The number of rows is the number of used variables.
/// The number of columns is five(minimum, maximum, mean and standard deviation).

Matrix<double> DataSet::calculate_columns_descriptives_matrix() const
{
    const size_t variables_number = get_used_variables_number();

    const Vector<size_t> used_variables_indices = get_used_columns_indices();

    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    const Vector<Descriptives> data_statistics_vector = descriptives_missing_values(data, used_instances_indices, used_variables_indices);

    Matrix<double> data_statistics_matrix(variables_number, 4);

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        data_statistics_matrix.set_row(static_cast<size_t>(i), data_statistics_vector[static_cast<size_t>(i)].to_vector());
    }

    return data_statistics_matrix;
}


/// Returns all the variables descriptives from a single eigen matrixXd.
/// The number of rows is the number of used variables.
/// The number of columns is five(minimum, maximum, mean and standard deviation).
/// @todo Low priority.

Eigen::MatrixXd DataSet::calculate_columns_descriptives_eigen() const
{
    const Vector<Descriptives> columns_descriptives = calculate_columns_descriptives();

    Eigen::MatrixXd eigen_statistics;

    return eigen_statistics;
}


/// Calculate the descriptives of the instances with positive targets in binary classification problems.
/// @todo Low priority.

Vector<Descriptives> DataSet::calculate_columns_descriptives_positive_instances() const
{
/*
#ifdef __OPENNN_DEBUG__

    const size_t targets_number = get_target_variables_number();

    if(targets_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_columns_descriptives_positive_instances() const method.\n"
               << "Number of targets muste be 1.\n";

        throw logic_error(buffer.str());
    }
#endif

    const size_t target_index = get_target_variables_indices()[0];

    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    const Vector<double> targets = data.get_column(target_index, used_instances_indices);

#ifdef __OPENNN_DEBUG__

    if(!targets.is_binary())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_columns_descriptives_positive_instances() const method.\n"
               << "Targets vector must be binary.\n";

        throw logic_error(buffer.str());
    }
#endif

    const Vector<size_t> inputs_variables_indices = get_input_variables_indices();

    const size_t inputs_number = inputs_variables_indices.size();

    const Vector<size_t> positives_used_instances_indices = used_instances_indices.get_subvector(targets.get_indices_equal_to(1.0));

    Matrix<double> data_statistics_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {        
        const size_t variable_index = inputs_variables_indices[i];

        const Vector<double> variable_data = data.get_column(variable_index, positives_used_instances_indices);

        const Descriptives data_descriptives = descriptives(variable_data);

        data_statistics_matrix.set_row(i, data_descriptives.to_vector());
    }

    return data_statistics_matrix;
*/
    return Vector<Descriptives>();
}


/// Calculate the descriptives of the instances with neagtive targets in binary classification problems.
/// @todo Low priority.

Vector<Descriptives> DataSet::calculate_columns_descriptives_negative_instances() const
{
/*
#ifdef __OPENNN_DEBUG__

    const size_t targets_number = get_target_variables_number();

    if(targets_number != 1)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_columns_descriptives_positive_instances() const method.\n"
               << "Number of targets muste be 1.\n";

        throw logic_error(buffer.str());
    }
#endif

    const size_t target_index = get_target_variables_indices()[0];

    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    const Vector<double> targets = data.get_column(target_index, used_instances_indices);

#ifdef __OPENNN_DEBUG__

    if(!targets.is_binary())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_columns_descriptives_positive_instances() const method.\n"
               << "Targets vector must be binary.\n";

        throw logic_error(buffer.str());
    }
#endif

    const Vector<size_t> inputs_variables_indices = get_input_variables_indices();

    const size_t inputs_number = inputs_variables_indices.size();

    const Vector<size_t> negatives_used_instances_indices = used_instances_indices.get_subvector(targets.get_indices_equal_to(0.0));

    Matrix<double> data_statistics_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        const size_t variable_index = inputs_variables_indices[i];

        const Vector<double> variable_data = data.get_column(variable_index, negatives_used_instances_indices);

        const Descriptives data_descriptives = descriptives(variable_data);

        data_statistics_matrix.set_row(i, data_descriptives.to_vector());
    }

    return data_statistics_matrix;
*/
    return Vector<Descriptives>();
}


/// Returns a matrix with the data set descriptive statistics.
/// @param class_index Data set index number to make the descriptive statistics.
/// @todo Low priority.

Vector<Descriptives> DataSet::calculate_columns_descriptives_classes(const size_t& class_index) const
{
/*
    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    const Vector<double> targets = data.get_column(class_index, used_instances_indices);

    #ifdef __OPENNN_DEBUG__

    if(!targets.is_binary())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_columns_descriptives_classes() const method.\n"
               << "Targets vector must be binary.\n";

        throw logic_error(buffer.str());
    }

#endif

    const Vector<size_t> inputs_variables_indices = get_input_variables_indices();

    const size_t inputs_number = inputs_variables_indices.size();

    const Vector<size_t> class_used_instances_indices = used_instances_indices.get_subvector(targets.get_indices_equal_to(1.0));

    Matrix<double> data_statistics_matrix(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        const size_t variable_index = inputs_variables_indices[i];

        const Vector<double> variable_data = data.get_column(variable_index, class_used_instances_indices);

        const Descriptives data_descriptives = descriptives(variable_data);

        data_statistics_matrix.set_row(i, data_descriptives.to_vector());
    }

    return data_statistics_matrix;
*/
    return Vector<Descriptives>();
}


/// Returns a vector of vectors containing some basic descriptives of all variables on the training
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Training data minimum.
/// <li> Training data maximum.
/// <li> Training data mean.
/// <li> Training data standard deviation.
/// </ul>

Vector<Descriptives> DataSet::calculate_columns_descriptives_training_instances() const
{
   const Vector<size_t> training_indices = get_training_instances_indices();

   const Vector<size_t> used_indices = get_used_columns_indices();

   return descriptives_missing_values(data, training_indices, used_indices);
}


/// Returns a vector of vectors containing some basic descriptives of all variables on the selection
/// The size of this vector is two. The subvectors are:
/// <ul>
/// <li> Selection data minimum.
/// <li> Selection data maximum.
/// <li> Selection data mean.
/// <li> Selection data standard deviation.
/// </ul>

Vector<Descriptives> DataSet::calculate_columns_descriptives_selection_instances() const
{
    const Vector<size_t> selection_indices = get_selection_instances_indices();

    const Vector<size_t> used_indices = get_used_columns_indices();

    return descriptives_missing_values(data, selection_indices, used_indices);
}


/// Returns a vector of vectors containing some basic descriptives of all variables on the testing
/// The size of this vector is five. The subvectors are:
/// <ul>
/// <li> Testing data minimum.
/// <li> Testing data maximum.
/// <li> Testing data mean.
/// <li> Testing data standard deviation.
/// </ul>

Vector<Descriptives> DataSet::calculate_columns_descriptives_testing_instances() const
{
    const Vector<size_t> testing_indices = get_testing_instances_indices();

    const Vector<size_t> used_indices = get_used_columns_indices();

    return descriptives_missing_values(data, testing_indices, used_indices);
}


/// Returns a Vector of Descriptives structures with some basic statistics of the input variables on the used
/// This includes the minimum, maximum, mean and standard deviation.
/// The size of this vector is the number of inputs.

Vector<Descriptives> DataSet::calculate_input_variables_descriptives() const
{
    const Vector<size_t> used_indices = get_used_instances_indices();

    const Vector<size_t> inputs_indices = get_input_variables_indices();

    return descriptives_missing_values(data, used_indices, inputs_indices);
}


/// Returns a vector of vectors with some basic descriptives of the target variables on all
/// The size of this vector is four. The subvectors are:
/// <ul>
/// <li> Target variables minimum.
/// <li> Target variables maximum.
/// <li> Target variables mean.
/// <li> Target variables standard deviation.
/// </ul>

Vector<Descriptives> DataSet::calculate_target_variables_descriptives() const
{
   const Vector<size_t> used_indices = get_used_instances_indices();

   const Vector<size_t> targets_indices = get_target_variables_indices();

   return descriptives_missing_values(data, used_indices, targets_indices);
}


/// Returns a vector containing the means of a set of given variables.
/// @param variables_indices Indices of the variables.

Vector<double> DataSet::calculate_variables_means(const Vector<size_t>& variables_indices) const
{
    const size_t variables_number = variables_indices.size();

    Vector<double> means(variables_number, 0.0);

#pragma omp parallel for

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        const size_t variable_index = variables_indices[static_cast<size_t>(i)];

        means[static_cast<size_t>(i)] = mean(data.get_column(variable_index));
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

Descriptives DataSet::calculate_inputs_descriptives(const size_t& input_index) const
{
   return descriptives_missing_values(data.get_column(input_index));
}


/// Returns the mean values of the target variables on the training

Vector<double> DataSet::calculate_training_targets_mean() const
{
    const Vector<size_t> training_indices = get_training_instances_indices();

    const Vector<size_t> targets_indices = get_target_variables_indices();

    return mean_missing_values(data, training_indices, targets_indices);
}


/// Returns the mean values of the target variables on the selection

Vector<double> DataSet::calculate_selection_targets_mean() const
{
    const Vector<size_t> selection_indices = get_selection_instances_indices();

    const Vector<size_t> targets_indices = get_target_variables_indices();

    return mean_missing_values(data, selection_indices, targets_indices);
}


/// Returns the mean values of the target variables on the testing

Vector<double> DataSet::calculate_testing_targets_mean() const
{
   const Vector<size_t> testing_indices = get_testing_instances_indices();

   const Vector<size_t> targets_indices = get_target_variables_indices();

   return mean_missing_values(data, testing_indices, targets_indices);
}


int DataSet::get_gmt() const
{
    return gmt;
}


void DataSet::set_gmt(int& new_gmt)
{
    gmt = new_gmt;
}


/// Calculates the linear correlations between all outputs and all inputs.
/// It returns a matrix with number of rows the targets number and number of columns the inputs number.
/// Each element contains the linear correlation between a single target and a single output.

Matrix<CorrelationResults> DataSet::calculate_input_target_columns_correlations() const
{
   const size_t input_columns_number = get_input_columns_number();
   const size_t target_columns_number = get_target_columns_number();

   const Vector<size_t> input_columns_indices = get_input_columns_indices();
   Vector<size_t> target_columns_indices = get_target_columns_indices();

   Matrix<CorrelationResults> correlations(input_columns_number, target_columns_number);

#pragma omp parallel for

   for(size_t i = 0; i < input_columns_number; i++)
   {
       const Matrix<double> input = get_column_data(input_columns_indices[i]);

       const ColumnType input_type = columns[input_columns_indices[i]].type;

       for(size_t j = 0; j < target_columns_number; j++)
       {
           const Matrix<double> target = get_column_data(target_columns_indices[j]);

           const ColumnType target_type = columns[target_columns_indices[j]].type;

           if(input_type == Numeric && target_type == Numeric)
           {
               correlations(i,j) = linear_correlations_missing_values(input.get_column(0), target.get_column(0));

               const CorrelationResults linear_correlation = linear_correlations_missing_values(input.get_column(0), target.get_column(0));
               const CorrelationResults exponential_correlation = exponential_correlations_missing_values(input.get_column(0), target.get_column(0));
               const CorrelationResults logarithmic_correlation = logarithmic_correlations_missing_values(input.get_column(0), target.get_column(0));
               const CorrelationResults power_correlation = power_correlations_missing_values(input.get_column(0), target.get_column(0));

               CorrelationResults strongest_correlation = linear_correlation;

               if(abs(exponential_correlation.correlation) > abs(strongest_correlation.correlation)) strongest_correlation = exponential_correlation;
               else if(abs(logarithmic_correlation.correlation) > abs(strongest_correlation.correlation)) strongest_correlation = logarithmic_correlation;
               else if(abs(power_correlation.correlation) > abs(strongest_correlation.correlation)) strongest_correlation = power_correlation;

               correlations(i,j) = strongest_correlation;
           }
           else if(input_type == Binary && target_type == Binary)
           {
               correlations(i,j) = linear_correlations_missing_values(input.get_column(0), target.get_column(0));
           }
           else if(input_type == Categorical && target_type == Categorical)
           {
               correlations(i,j) = karl_pearson_correlations_missing_values(input, target);
           }
           else if(input_type == Numeric && target_type == Binary)
           {
               correlations(i,j) = logistic_correlations_missing_values(input, target);
           }
           else if(input_type == Binary && target_type == Numeric)
           {
               correlations(i,j) = logistic_correlations_missing_values(input, target);
           }
           else if(input_type == Categorical && target_type == Numeric)
           {
               correlations(i,j) = one_way_anova_correlations_missing_values(input, target.get_column(0));
           }
           else if(input_type == Numeric && target_type == Categorical)
           {
               correlations(i,j) = one_way_anova_correlations_missing_values(target, input);
           }
           else
           {
               ostringstream buffer;

               buffer << "OpenNN Exception: DataSet class.\n"
                      << "Matrix<double> calculate_inputs_correlations() const method.\n"
                      << "Case not found: Column i " << input_type << " and Column j " << target_type << ".\n";

               throw logic_error(buffer.str());
           }
       }
   }

   return correlations;
}


Matrix<double> DataSet::calculate_input_target_columns_correlations_double() const
{
    Matrix<CorrelationResults> correlations = calculate_input_target_columns_correlations();

    const size_t rows_number = correlations.get_rows_number();
    const size_t columns_number = correlations.get_columns_number();

    Matrix<double> correlations_double(rows_number, columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            correlations_double(i,j) = correlations(i,j).correlation;
        }
    }

    return correlations_double;
}

/// Calculates the linear correlations between all outputs and all inputs.
/// It returns a matrixXd with number of rows the targets number and number of columns the inputs number.
/// Each element contains the linear correlation between a single target and a single output.
/// @todo

Eigen::MatrixXd DataSet::calculate_input_target_columns_correlations_eigen() const
{
/*
    const size_t target_number = get_target_variables_number();
    const size_t input_number = get_input_variables_number();

    const Eigen::Map<Eigen::MatrixXd> correlations(static_cast<double*>(calculate_input_target_columns_correlations().data()),
                                                   static_cast<int>(target_number),
                                                   static_cast<int>(input_number));

    return correlations;
*/
    return Eigen::MatrixXd();
}


void DataSet::print_missing_values_information() const
{
    const size_t missing_values_number = data.count_nan();

    cout << "Missing values number: " << missing_values_number << " (" << missing_values_number*100/data.size() << "%)" << endl;

    const size_t variables_with_missing_values = data.count_columns_with_nan();

    cout << "Variables with missing values: " << variables_with_missing_values << " (" << variables_with_missing_values*100/data.get_columns_number() << "%)" << endl;

    const size_t instances_with_missing_values = data.count_rows_with_nan();

    cout << "Instances with missing values: " << instances_with_missing_values << " (" << instances_with_missing_values*100/data.get_rows_number() << "%)" << endl;
}


/// Print on screen the correlation between targets and inputs.

void DataSet::print_input_target_columns_correlations() const
{
    const size_t inputs_number = get_input_variables_number();
    const size_t targets_number = get_target_variables_number();

    const Vector<string> inputs_names = get_input_variables_names();
    const Vector<string> targets_name = get_target_variables_names();

    const Matrix<RegressionResults> correlations;// = calculate_input_target_columns_correlations();

    for(size_t j = 0; j < targets_number; j++)
    {
        for(size_t i = 0; i < inputs_number; i++)
        {
            cout << targets_name[j] << " - " << inputs_names[i] << ": " << correlations(i,j).correlation << endl;
        }
    }
}


/// This method print on screen the corretaliont between inputs and targets.
/// @param number Number of variables to be printed.
/// @todo

void DataSet::print_top_input_target_columns_correlations(const size_t& number) const
{
    const size_t inputs_number = get_input_columns_number();
    const size_t targets_number = get_target_columns_number();

    const Vector<string> inputs_names = get_input_variables_names();
    const Vector<string> targets_name = get_target_variables_names();

    const Matrix<RegressionResults> correlations;// = calculate_input_target_columns_correlations();

    Vector<double> target_correlations(inputs_number);

    Matrix<string> top_correlations(inputs_number, 2);

    map<double,string> top_correlation;

    for(size_t i = 0 ; i < inputs_number; i++)
    {
        for(size_t j = 0 ; j < targets_number ; j++)
        {
//            top_correlation.insert(pair<double,string>(correlations(i,j), inputs_names[i] + " - " + targets_name[j]));
        }
    }

    map<double,string>::iterator it;

    for(it = top_correlation.begin(); it!=top_correlation.end(); it++)
    {
        cout << "Correlation:  " << (*it).first << "  between  " << (*it).second << "" << endl;
    }
}


/// Calculate the correlation between each variable in the data set.
/// Returns a matrix with the correlation values between variables in the data set.

Matrix<double> DataSet::calculate_inputs_correlations() const
{
    const Vector<size_t> input_columns_indices = get_input_columns_indices();

    const size_t input_columns_number = get_input_columns_number();

    Matrix<double> correlations(input_columns_number, input_columns_number);

    correlations.initialize_identity();

    for(size_t i = 0; i < input_columns_number; i++)
    {
        const ColumnType type_i = columns[i].type;

        const Matrix<double> column_i = get_column_data(input_columns_indices[i]);

        for(size_t j = i; j < input_columns_number; j++)
        {
            const ColumnType type_j = columns[j].type;

            const Matrix<double> column_j = get_column_data(input_columns_indices[j]);

            if(type_i == Numeric && type_j == Numeric)
            {
                correlations(i,j) = linear_correlation_missing_values(column_i.get_column(0), column_j.get_column(0));

                const double linear_correlation = linear_correlation_missing_values(column_i.get_column(0), column_j.get_column(0));
                const double exponential_correlation = exponential_correlation_missing_values(column_i.get_column(0), column_j.get_column(0));
                const double logarithmic_correlation = logarithmic_correlation_missing_values(column_i.get_column(0), column_j.get_column(0));
                const double power_correlation = power_correlation_missing_values(column_i.get_column(0), column_j.get_column(0));
                const Vector<double> correlations_i_j({linear_correlation, exponential_correlation, logarithmic_correlation, power_correlation});

                correlations(i,j) = strongest(correlations_i_j);
            }
            else if(type_i == Binary && type_j == Binary)
            {
                correlations(i,j) = linear_correlation_missing_values(column_i.get_column(0), column_j.get_column(0));
            }
            else if(type_i == Categorical && type_j == Categorical)
            {
                correlations(i,j) = karl_pearson_correlation_missing_values(column_i, column_j);
            }
            else if(type_i == Numeric && type_j == Binary)
            {
                correlations(i,j) = logistic_correlation_missing_values(column_i, column_j);
            }
            else if(type_i == Binary && type_j == Numeric)
            {
                correlations(i,j) = logistic_correlation_missing_values(column_j, column_i);
            }
            else if(type_i == Categorical && type_j == Numeric)
            {
                correlations(i,j) = one_way_anova_correlation(column_i, column_j.get_column(0));
            }
            else if(type_i == Numeric && type_j == Categorical)
            {
                correlations(i,j) = one_way_anova_correlation(column_j, column_i.get_column(0));
            }
            else
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: DataSet class.\n"
                       << "Matrix<double> calculate_inputs_correlations() const method.\n"
                       << "Case not found: Column i " << type_i << " and Column j " << type_j << ".\n";

                throw logic_error(buffer.str());
            }

        }
    }

    for(size_t i = 0; i < input_columns_number; i++)
    {
        for(size_t j = 0; j < i; j++)
        {
            correlations(i,j) = correlations(j,i);
        }
    }

    return correlations;
}


/// Print on screen the correlation between variables in the data set.

void DataSet::print_inputs_correlations() const
{
    const Matrix<double> inputs_correlations = calculate_inputs_correlations();

    cout << inputs_correlations << endl;
}


void DataSet::print_data_file_preview() const
{
    const size_t size = data_file_preview.size();

    for(size_t i = 0;  i < size; i++)
    {
        cout << data_file_preview[i] << endl;
    }
}


/// This method print on screen the corretaliont between variables.
/// @param number Number of variables to be printed.
/// @todo Low priority.

void DataSet::print_top_inputs_correlations(const size_t& number) const
{
    const size_t variables_number = get_input_variables_number();

    const Vector<string> variables_name = get_input_variables_names();

    const Matrix<double> variables_correlations = calculate_inputs_correlations();

    const size_t correlations_number = variables_number*(variables_number-1)/2;

    Matrix<string> top_correlations(correlations_number, 3);

    map<double, string> top_correlation;

    for(size_t i = 0; i < variables_number; i++)
    {
        for(size_t j = i; j < variables_number; j++)
        {
            if(i == j) continue;

            top_correlation.insert(pair<double,string>(variables_correlations(i,j), variables_name[i] + " - " + variables_name[j]));
         }
     }

    map<double,string> :: iterator it;

    for(it=top_correlation.begin(); it!=top_correlation.end(); it++)
    {
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << "" << endl;
    }
}


/// Returns the covariance matrix for the input data set.
/// The number of rows of the matrix is the number of inputs.
/// The number of columns of the matrix is the number of inputs.

Matrix<double> DataSet::calculate_covariance_matrix() const
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();
    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    const size_t inputs_number = get_input_variables_number();

    Matrix<double> covariance_matrix(inputs_number, inputs_number, 0.0);

    for(int i = 0; i < static_cast<int>(inputs_number); i++)
    {
        const size_t first_input_index = inputs_indices[static_cast<size_t>(i)];

        const Vector<double> first_inputs = data.get_column(first_input_index, used_instances_indices);

        for(size_t j = static_cast<size_t>(i); j < inputs_number; j++)
        {
            const size_t second_input_index = inputs_indices[j];

            const Vector<double> second_inputs = data.get_column(second_input_index, used_instances_indices);

            covariance_matrix(static_cast<size_t>(i),j) = covariance(first_inputs, second_inputs);
            covariance_matrix(j,static_cast<size_t>(i)) = covariance_matrix(static_cast<size_t>(i),j);
        }
    }

    return covariance_matrix;
}


/// Performs the principal components analysis of the inputs.
/// It returns a matrix containing the principal components getd in rows.
/// This method deletes the unused instances of the original data set.
/// @param minimum_explained_variance Minimum percentage of variance used to select a principal component.

Matrix<double> DataSet::perform_principal_components_analysis(const double& minimum_explained_variance)
{
    // Subtract off the mean

    subtract_inputs_mean();

    // Calculate covariance matrix

    const Matrix<double> covariance_matrix = this->calculate_covariance_matrix();

    // Calculate eigenvectors

    const Matrix<double> eigenvectors = OpenNN::eigenvectors(covariance_matrix);

    // Calculate eigenvalues

    const Matrix<double> eigenvalues = OpenNN::eigenvalues(covariance_matrix);

    // Calculate explained variance

    const Vector<double> explained_variance = OpenNN::explained_variance(eigenvalues.get_column(0));

    // Sort principal components

    const Vector<size_t> sorted_principal_components_indices = explained_variance.sort_descending_indices();

    // Choose eigenvectors

    const size_t inputs_number = covariance_matrix.get_columns_number();

    Vector<size_t> principal_components_indices;

    size_t index;

    for(size_t i = 0; i < inputs_number; i++)
    {
        index = sorted_principal_components_indices[i];

        if(explained_variance[index] >= minimum_explained_variance)
        {
            principal_components_indices.push_back(i);
        }
        else
        {
            continue;
        }
    }

    const size_t principal_components_number = principal_components_indices.size();

    // Arrange principal components matrix

    Matrix<double> principal_components;

    if(principal_components_number == 0)
    {
        return principal_components;
    }
    else
    {
        principal_components.set(principal_components_number, inputs_number);
    }

    for(size_t i = 0; i < principal_components_number; i++)
    {
        index = sorted_principal_components_indices[i];

        principal_components.set_row(i, eigenvectors.get_column(index));
    }

    // Return feature matrix

    return principal_components.get_submatrix_rows(principal_components_indices);
}


/// Performs the principal components analysis of the inputs.
/// It returns a matrix containing the principal components arranged in rows.
/// This method deletes the unused instances of the original data set.
/// @param covariance_matrix Matrix of covariances.
/// @param explained_variance Vector of the explained variances of the variables.
/// @param minimum_explained_variance Minimum percentage of variance used to select a principal component.

Matrix<double> DataSet::perform_principal_components_analysis(const Matrix<double>& covariance_matrix,
                                                              const Vector<double>& explained_variance,
                                                              const double& minimum_explained_variance)
{
    // Subtract off the mean

    subtract_inputs_mean();

    // Calculate eigenvectors

    const Matrix<double> eigenvectors = OpenNN::eigenvectors(covariance_matrix);

    // Sort principal components

    const Vector<size_t> sorted_principal_components_indices = explained_variance.sort_descending_indices();

    // Choose eigenvectors

    const size_t inputs_number = covariance_matrix.get_columns_number();

    Vector<size_t> principal_components_indices;

    size_t index;

    for(size_t i = 0; i < inputs_number; i++)
    {
        index = sorted_principal_components_indices[i];

        if(explained_variance[index] >= minimum_explained_variance)
        {
            principal_components_indices.push_back(i);
        }
        else
        {
            continue;
        }
    }

    const size_t principal_components_number = principal_components_indices.size();

    // Arrange principal components matrix

    Matrix<double> principal_components;

    if(principal_components_number == 0)
    {
        return principal_components;
    }
    else
    {
        principal_components.set(principal_components_number, inputs_number);
    }

    for(size_t i = 0; i < principal_components_number; i++)
    {
        index = sorted_principal_components_indices[i];

        principal_components.set_row(i, eigenvectors.get_column(index));
    }

    // Return feature matrix

    return principal_components.get_submatrix_rows(principal_components_indices);
}


/// Transforms the data according to the principal components.
/// @param principal_components Matrix containing the principal components.

void DataSet::transform_principal_components_data(const Matrix<double>& principal_components)
{
    const Matrix<double> targets = get_target_data();

    subtract_inputs_mean();

    const size_t principal_components_number = principal_components.get_rows_number();

    // Transform data

    const Vector<size_t> used_instances = get_used_instances_indices();

    const size_t new_instances_number = get_used_instances_number();

    const Matrix<double> inputs = get_input_data();

    Matrix<double> new_data(new_instances_number, principal_components_number, 0.0);

    size_t instance_index;

    for(size_t i = 0; i < new_instances_number; i++)
    {
        instance_index = used_instances[i];

        for(size_t j = 0; j < principal_components_number; j++)
        {
            new_data(i,j) = dot(inputs.get_row(instance_index), principal_components.get_row(j));
        }
    }

    data = new_data.assemble_columns(targets);
}


/// Scales the data matrix with given mean and standard deviation values.
/// It updates the data matrix.
/// @param data_descriptives Vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::scale_data_mean_standard_deviation(const Vector<Descriptives>& data_descriptives)
{
   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const size_t columns_number = data.get_columns_number();

   const size_t descriptives_size = data_descriptives.size();

   if(descriptives_size != columns_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void scale_data_mean_standard_deviation(const Vector<Descriptives>&) method.\n"
             << "Size of descriptives must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   const size_t variables_number = get_variables_number();

   for(size_t i = 0; i < variables_number; i++)
   {
       if(display && abs(data_descriptives[i].standard_deviation) < numeric_limits<double>::min())
       {
          cout << "OpenNN Warning: DataSet class.\n"
                    << "void scale_data_mean_standard_deviation(const Vector< Descriptives<Type> >&) method.\n"
                    << "Standard deviation of variable " <<  i << " is zero.\n"
                    << "That variable won't be scaled.\n";
        }
    }

   scale_mean_standard_deviation(data, data_descriptives);
}


/// Scales the data using the minimum and maximum method,
/// and the minimum and maximum values calculated from the data matrix.
/// It also returns the descriptives from all columns.

Vector<Descriptives> DataSet::scale_data_minimum_maximum()
{
    const Vector<Descriptives> data_descriptives = calculate_columns_descriptives();

    scale_data_minimum_maximum(data_descriptives);

    return data_descriptives;
}


/// Scales the data using the mean and standard deviation method,
/// and the mean and standard deviation values calculated from the data matrix.
/// It also returns the descriptives from all columns.

Vector<Descriptives> DataSet::scale_data_mean_standard_deviation()
{
    const Vector<Descriptives> data_descriptives = calculate_columns_descriptives();

    scale_data_mean_standard_deviation(data_descriptives);

    return data_descriptives;
}


/// Subtracts off the mean to every of the input variables.

void DataSet::subtract_inputs_mean()
{
    Vector<Descriptives> input_statistics = calculate_input_variables_descriptives();

    Vector<size_t> inputs_indices = get_input_variables_indices();
    Vector<size_t> used_instances_indices = get_used_instances_indices();

    size_t input_index;
    size_t instance_index;

    double input_mean;

    for(size_t i = 0; i < inputs_indices.size(); i++)
    {
        input_index = inputs_indices[i];

        input_mean = input_statistics[i].mean;

        for(size_t j = 0; j < used_instances_indices.size(); j++)
        {
            instance_index = used_instances_indices[j];

            data(instance_index,input_index) -= input_mean;
        }
    }
}


/// Returns a vector of strings containing the scaling method that best fits each
/// of the input variables.
/// @todo Low priority.

Vector<string> DataSet::calculate_default_scaling_methods() const
{
    const Vector<size_t> used_inputs_indices = get_input_variables_indices();
    const size_t used_inputs_number = used_inputs_indices.size();

    size_t current_distribution;
    Vector<string> scaling_methods(used_inputs_number);

#pragma omp parallel for private(current_distribution)

    for(int i = 0; i < static_cast<int>(used_inputs_number); i++)
    {
        current_distribution = perform_distribution_distance_analysis(data.get_column(used_inputs_indices[static_cast<size_t>(i)]));

        if(current_distribution == 0) // Normal distribution
        {
            scaling_methods[static_cast<size_t>(i)] = "MeanStandardDeviation";
        }
        else if(current_distribution == 1) // Uniform distribution
        {
            scaling_methods[static_cast<size_t>(i)] = "MinimumMaximum";
        }
        else // Default
        {
            scaling_methods[static_cast<size_t>(i)] = "MinimumMaximum";
        }
    }

    return scaling_methods;
}


/// Scales the data matrix with given minimum and maximum values.
/// It updates the data matrix.
/// @param data_descriptives Vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::scale_data_minimum_maximum(const Vector<Descriptives>& data_descriptives)
{
    const size_t variables_number = get_variables_number();

   #ifdef __OPENNN_DEBUG__

   ostringstream buffer;

   const size_t descriptives_size = data_descriptives.size();

   if(descriptives_size != variables_number)
   {
      buffer << "OpenNN Exception: DataSet class.\n"
             << "void scale_data_minimum_maximum(const Vector<Descriptives>&) method.\n"
             << "Size of data descriptives must be equal to number of variables.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t i = 0; i < variables_number; i++)
   {
       if(display
       && abs(data_descriptives[i].maximum - data_descriptives[i].minimum) < numeric_limits<double>::min())
       {
          cout << "OpenNN Warning: DataSet class.\n"
                    << "void scale_data_minimum_maximum(const Vector< Descriptives<Type> >&) method.\n"
                    << "Range of variable " <<  i << " is zero.\n"
                    << "That variable won't be scaled.\n";
        }
    }

   scale_minimum_maximum(data, data_descriptives);
}


/// Scales the input variables with given mean and standard deviation values.
/// It updates the input variables of the data matrix.
/// @param inputs_descriptives Vector of descriptives structures for the input variables.
/// The size of that vector must be equal to the number of inputs.

void DataSet::scale_inputs_mean_standard_deviation(const Vector<Descriptives>& inputs_descriptives)
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();

    scale_columns_mean_standard_deviation(data, inputs_descriptives, inputs_indices);
}


/// Scales the input variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector of vectors with the variables descriptives.

Vector<Descriptives> DataSet::scale_inputs_mean_standard_deviation()
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<Descriptives> scale_inputs_mean_standard_deviation() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Vector<Descriptives> inputs_descriptives = calculate_input_variables_descriptives();

   scale_inputs_mean_standard_deviation(inputs_descriptives);

   return inputs_descriptives;
}


/// Scales the given input variables with given mean and standard deviation values.
/// It updates the input variable of the data matrix.
/// @param input_statistics Vector of descriptives structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_mean_standard_deviation(const Descriptives& input_statistics, const size_t& input_index)
{
    Vector<double> column = data.get_column(input_index);

    scale_mean_standard_deviation(column, input_statistics);

    data.set_column(input_index, column, "");
}


/// Scales the given input variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector with the variables descriptives.
/// @param input_index Index of the input to be scaled.

Descriptives DataSet::scale_input_mean_standard_deviation(const size_t& input_index)
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Descriptives scale_input_mean_standard_deviation(const size_t&) method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Descriptives input_statistics = calculate_inputs_descriptives(input_index);

   scale_input_mean_standard_deviation(input_statistics, input_index);

   return input_statistics;
}


/// Scales the given input variables with given standard deviation values.
/// It updates the input variable of the data matrix.
/// @param inputs_statistics Vector of descriptives structures for the input variables.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_standard_deviation(const Descriptives& input_statistics, const size_t& input_index)
{
    Vector<double> column = data.get_column(input_index);

    scale_standard_deviation(column, input_statistics);

    data.set_column(input_index, column, "");
}


/// Scales the given input variables with the calculated standard deviation values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector with the variables descriptives.
/// @param input_index Index of the input to be scaled.

Descriptives DataSet::scale_input_standard_deviation(const size_t& input_index)
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Descriptives scale_input_standard_deviation(const size_t&) method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Descriptives input_statistics = calculate_inputs_descriptives(input_index);

   scale_input_standard_deviation(input_statistics, input_index);

   return input_statistics;
}


/// Scales the input variables with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param inputs_descriptives Vector of descriptives structures for all the inputs in the data set.
/// The size of that vector must be equal to the number of input variables.

void DataSet::scale_inputs_minimum_maximum(const Vector<Descriptives>& inputs_descriptives)
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();

    scale_columns_minimum_maximum(data, inputs_descriptives, inputs_indices);
}


/// Scales the input variables with the calculated minimum and maximum values from the data matrix.
/// It updates the input variables of the data matrix.
/// It also returns a vector of vectors with the minimum and maximum values of the input variables.

Vector<Descriptives> DataSet::scale_inputs_minimum_maximum()
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<Descriptives> scale_inputs_minimum_maximum() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif


   const Vector<Descriptives> inputs_descriptives = calculate_input_variables_descriptives();

   scale_inputs_minimum_maximum(inputs_descriptives);

   return inputs_descriptives;
}


/// Scales the input variables with the calculated minimum and maximum values from the data matrixXd.
/// It updates the input variables of the data matrixXd.
/// It also returns a vector of vectors with the minimum and maximum values of the input variables.

Eigen::MatrixXd DataSet::scale_inputs_minimum_maximum_eigen()
{
    const Vector<Descriptives> inputs_descriptives = scale_inputs_minimum_maximum();

    const size_t inputs_number = inputs_descriptives.size();

    Eigen::MatrixXd eigen(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        eigen(static_cast<int>(i),0) = inputs_descriptives[i].minimum;
        eigen(static_cast<int>(i),1) = inputs_descriptives[i].maximum;
        eigen(static_cast<int>(i),2) = inputs_descriptives[i].mean;
        eigen(static_cast<int>(i),3) = inputs_descriptives[i].standard_deviation;
    }

    return eigen;
}


/// Scales the given input variable with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param input_statistics Vector with the descriptives of the input variable.
/// @param input_index Index of the input to be scaled.

void DataSet::scale_input_minimum_maximum(const Descriptives& input_statistics, const size_t & input_index)
{
    Vector<double> column = data.get_column(input_index);

    scale_minimum_maximum(column, input_statistics);

    data.set_column(input_index, column, "");
}


/// Scales the given input variable with the calculated minimum and maximum values from the data matrix.
/// It updates the input variable of the data matrix.
/// It also returns a vector with the minimum and maximum values of the input variables.

Descriptives DataSet::scale_input_minimum_maximum(const size_t& input_index)
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Descriptives scale_input_minimum_maximum(const size_t&) method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Descriptives input_statistics = calculate_inputs_descriptives(input_index);

    scale_input_minimum_maximum(input_statistics, input_index);

    return input_statistics;
}


/// Calculates the input and target variables descriptives.
/// Then it scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.
/// Finally, it returns the descriptives.

Vector<Descriptives> DataSet::scale_inputs(const string& scaling_unscaling_method)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
    {
        case NoScaling: return calculate_input_variables_descriptives();

        case MinimumMaximum: return scale_inputs_minimum_maximum();

        case MeanStandardDeviation: return scale_inputs_mean_standard_deviation();

        case StandardDeviation: return scale_inputs_mean_standard_deviation();

        default:
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet class\n"
                   << "Vector<Descriptives> scale_inputs() method.\n"
                   << "Unknown scaling and unscaling method.\n";

            throw logic_error(buffer.str());
        }
    }
}


/// Calculates the input and target variables descriptives.
/// Then it scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_inputs(const string& scaling_unscaling_method, const Vector<Descriptives>& inputs_descriptives)
{
   switch(get_scaling_unscaling_method(scaling_unscaling_method))
   {
      case NoScaling:
      {
          // Do nothing
      }
      break;

      case MinimumMaximum:
      {
         scale_inputs_minimum_maximum(inputs_descriptives);
      }
      break;

      case MeanStandardDeviation:
      {
         scale_inputs_mean_standard_deviation(inputs_descriptives);
      }
      break;

      default:
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: DataSet class\n"
                << "void scale_inputs(const string&, const Vector<Descriptives>&) method.\n"
                << "Unknown scaling and unscaling method.\n";

         throw logic_error(buffer.str());
      }
   }
}


/// It scales every input variable with the given method.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_inputs(const Vector<string>& scaling_unscaling_methods, const Vector<Descriptives>& inputs_descriptives)
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();

   for(size_t i = 0; i < scaling_unscaling_methods.size(); i++)
   {
       switch(get_scaling_unscaling_method(scaling_unscaling_methods[i]))
       {
          case NoScaling:
          {
              // Do nothing
          }
          break;

          case MinimumMaximum:
          {
             scale_input_minimum_maximum(inputs_descriptives[i], inputs_indices[i]);
          }
          break;

          case MeanStandardDeviation:
          {
              scale_input_mean_standard_deviation(inputs_descriptives[i], inputs_indices[i]);
          }
          break;

          case StandardDeviation:
          {
               scale_input_standard_deviation(inputs_descriptives[i], inputs_indices[i]);
          }
          break;

          default:
          {
             ostringstream buffer;

             buffer << "OpenNN Exception: DataSet class\n"
                    << "void scale_inputs(const Vector<string>&, const Vector<Descriptives>&) method.\n"
                    << "Unknown scaling and unscaling method: " << scaling_unscaling_methods[i] << "\n";

             throw logic_error(buffer.str());
          }
       }
   }
}


/// Scales the target variables with given mean and standard deviation values.
/// It updates the target variables of the data matrix.
/// @param targets_descriptives Vector of descriptives structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_mean_standard_deviation(const Vector<Descriptives>& targets_descriptives)
{
    const Vector<size_t> targets_indices = get_target_variables_indices();

    scale_columns_mean_standard_deviation(data, targets_descriptives, targets_indices);
}


/// Scales the target variables with the calculated mean and standard deviation values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of descriptives structures with the basic descriptives of all the variables.

Vector<Descriptives> DataSet::scale_targets_mean_standard_deviation()
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<Descriptives> scale_targets_mean_standard_deviation() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

   const Vector<Descriptives> targets_descriptives = calculate_target_variables_descriptives();

   scale_targets_mean_standard_deviation(targets_descriptives);

   return targets_descriptives;
}


/// Scales the target variables with given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param targets_descriptives Vector of descriptives structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_minimum_maximum(const Vector<Descriptives>& targets_descriptives)
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<Descriptives> scale_targets_minimum_maximum() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Vector<size_t> targets_indices = get_target_variables_indices();

    scale_columns_minimum_maximum(data, targets_descriptives, targets_indices);
}


/// Scales the target variables with the calculated minimum and maximum values from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of vectors with the descriptives of the input target variables.

Vector<Descriptives> DataSet::scale_targets_minimum_maximum()
{
   const Vector<Descriptives> targets_descriptives = calculate_target_variables_descriptives();

   scale_targets_minimum_maximum(targets_descriptives);

   return targets_descriptives;
}


/// Scales the target variables with the calculated minimum and maximum values from the data eigen matrix.
/// It updates the target variables of the data eigen matrix.
/// It also returns a vector of vectors with the descriptives of the input target variables.

Eigen::MatrixXd DataSet::scale_targets_minimum_maximum_eigen()
{
    const Vector<Descriptives> targets_descriptives = scale_targets_minimum_maximum();

    const size_t inputs_number = targets_descriptives.size();

    Eigen::MatrixXd descriptives_eigen(inputs_number, 4);

    for(size_t i = 0; i < inputs_number; i++)
    {
        descriptives_eigen(static_cast<int>(i),0) = targets_descriptives[i].minimum;
        descriptives_eigen(static_cast<int>(i),1) = targets_descriptives[i].maximum;
        descriptives_eigen(static_cast<int>(i),2) = targets_descriptives[i].mean;
        descriptives_eigen(static_cast<int>(i),3) = targets_descriptives[i].standard_deviation;
    }

    return descriptives_eigen;
}


/// Scales the target variables with the logarithmic scale using the given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param targets_descriptives Vector of descriptives structures for all the targets in the data set.
/// The size of that vector must be equal to the number of target variables.

void DataSet::scale_targets_logarithmic(const Vector<Descriptives>& targets_descriptives)
{
    #ifdef __OPENNN_DEBUG__

    if(data.empty())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: DataSet class.\n"
              << "Vector<Descriptives> scale_targets_logarithmic() method.\n"
              << "Data file is not loaded.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Vector<size_t> targets_indices = get_target_variables_indices();

    scale_columns_logarithmic(data, targets_descriptives, targets_indices);
}


/// Scales the target variables with the logarithmic scale using the calculated minimum and maximum values
/// from the data matrix.
/// It updates the target variables of the data matrix.
/// It also returns a vector of vectors with the descriptives of the input target variables.

Vector<Descriptives> DataSet::scale_targets_logarithmic()
{
   const Vector<Descriptives> targets_descriptives = calculate_target_variables_descriptives();

   scale_targets_logarithmic(targets_descriptives);

   return targets_descriptives;
}


/// Calculates the input and target variables descriptives.
/// Then it scales the target variables with those values.
/// The method to be used is that in the scaling and unscaling method variable.
/// Finally, it returns the descriptives.

Vector<Descriptives> DataSet::scale_targets(const string& scaling_unscaling_method)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
   {
    case NoUnscaling:
    {
        return calculate_target_variables_descriptives();
    }

    case MinimumMaximum:
    {
        return scale_targets_minimum_maximum();
    }

    case Logarithmic:
    {
        return scale_targets_logarithmic();
    }

    case MeanStandardDeviation:
    {
        return scale_targets_mean_standard_deviation();
    }

    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "Vector<Descriptives> scale_targets(const string&) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw logic_error(buffer.str());
    }
   }
}


/// It scales the input variables with that values.
/// The method to be used is that in the scaling and unscaling method variable.

void DataSet::scale_targets(const string& scaling_unscaling_method, const Vector<Descriptives>& targets_descriptives)
{
    switch(get_scaling_unscaling_method(scaling_unscaling_method))
   {
    case NoUnscaling:
    {
        // Do nothing
    }
    break;

    case MinimumMaximum:
    {
        scale_targets_minimum_maximum(targets_descriptives);
    }
        break;

    case MeanStandardDeviation:
    {
        scale_targets_mean_standard_deviation(targets_descriptives);
    }
        break;

    case Logarithmic:
    {
        scale_targets_logarithmic(targets_descriptives);
    }
        break;

    default:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class\n"
               << "void scale_targets(const string&, const Vector<Descriptives>&) method.\n"
               << "Unknown scaling and unscaling method.\n";

        throw logic_error(buffer.str());
    }
   }
}


/// Unscales the data matrix with given mean and standard deviation values.
/// It updates the data matrix.
/// @param data_descriptives Vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_data_mean_standard_deviation(const Vector<Descriptives>& data_descriptives)
{
   unscale_mean_standard_deviation(data, data_descriptives);
}


/// Unscales the data matrix with given minimum and maximum values.
/// It updates the data matrix.
/// @param data_descriptives Vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_data_minimum_maximum(const Vector<Descriptives>& data_descriptives)
{
   unscale_minimum_maximum(data, data_descriptives);
}


/// Unscales the input variables with given mean and standard deviation values.
/// It updates the input variables of the data matrix.
/// @param data_descriptives Vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_inputs_mean_standard_deviation(const Vector<Descriptives>& data_descriptives)
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();

    unscale_columns_mean_standard_deviation(data, data_descriptives, inputs_indices);
}


/// Unscales the input variables with given minimum and maximum values.
/// It updates the input variables of the data matrix.
/// @param data_descriptives Vector of descriptives structures for all the data in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_inputs_minimum_maximum(const Vector<Descriptives>& data_descriptives)
{
    const Vector<size_t> inputs_indices = get_input_variables_indices();

    unscale_columns_minimum_maximum(data, data_descriptives, inputs_indices);
}


/// Unscales the target variables with given mean and standard deviation values.
/// It updates the target variables of the data matrix.
/// @param targets_descriptives Vector of descriptives structures for all the variables in the data set.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_targets_mean_standard_deviation(const Vector<Descriptives>& targets_descriptives)
{    
    const Vector<size_t> targets_indices = get_target_variables_indices();

    unscale_columns_mean_standard_deviation(data, targets_descriptives, targets_indices);
}


/// Unscales the target variables with given minimum and maximum values.
/// It updates the target variables of the data matrix.
/// @param data_descriptives Vector of descriptives structures for all the variables.
/// The size of that vector must be equal to the number of variables.

void DataSet::unscale_targets_minimum_maximum(const Vector<Descriptives>& data_descriptives)
{
    const Vector<size_t> targets_indices = get_target_variables_indices();

    unscale_columns_minimum_maximum(data, data_descriptives, targets_indices);
}


/// Initializes the data matrix with a given value.
/// @param new_value Initialization value.

void DataSet::initialize_data(const double& new_value)
{
   data.initialize(new_value);
}


/// Initializes the data matrix with random values chosen from a uniform distribution
/// with given minimum and maximum.

void DataSet::randomize_data_uniform(const double& minimum, const double& maximum)
{
   data.randomize_uniform(minimum, maximum);
}


/// Initializes the data matrix with random values chosen from a normal distribution
/// with given mean and standard deviation.

void DataSet::randomize_data_normal(const double& mean, const double& standard_deviation)
{
   data.randomize_normal(mean, standard_deviation);
}


/// Serializes the data set object into a XML document of the TinyXML library.

tinyxml2::XMLDocument* DataSet::to_XML() const
{
   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   ostringstream buffer;

   // Data set

   tinyxml2::XMLElement* data_set_element = document->NewElement("DataSet");
   document->InsertFirstChild(data_set_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Data file

   tinyxml2::XMLElement* data_file_element = document->NewElement("DataFile");

   data_set_element->InsertFirstChild(data_file_element);

   // Lags number
   {
       element = document->NewElement("LagsNumber");
       data_file_element->LinkEndChild(element);

       const size_t lags_number = get_lags_number();

       buffer.str("");
       buffer << lags_number;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Steps ahead
   {
       element = document->NewElement("StepsAhead");
       data_file_element->LinkEndChild(element);

       const size_t steps_ahead = get_steps_ahead();

       buffer.str("");
       buffer << steps_ahead;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Time index
   {
       element = document->NewElement("TimeIndex");
       data_file_element->LinkEndChild(element);

       const size_t time_index = get_time_index();

       buffer.str("");
       buffer << time_index;

       text = document->NewText(buffer.str().c_str());
       element->LinkEndChild(text);
   }

   // Header line
   {
      element = document->NewElement("ColumnsNames");
      data_file_element->LinkEndChild(element);

      buffer.str("");
      buffer << has_columns_names;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Rows label
   {
      element = document->NewElement("rows_labels");
      data_file_element->LinkEndChild(element);

      buffer.str("");
      buffer << has_rows_labels;

      text = document->NewText(buffer.str().c_str());
      element->LinkEndChild(text);
   }

   // Separator
   {
      element = document->NewElement("Separator");
      data_file_element->LinkEndChild(element);

      text = document->NewText(get_separator_string().c_str());
      element->LinkEndChild(text);
   }

   // Missing values label
   {
      element = document->NewElement("missing_values_label");
      data_file_element->LinkEndChild(element);

      text = document->NewText(missing_values_label.c_str());
      element->LinkEndChild(text);
   }

   // Data file name
   {
      element = document->NewElement("data_file_name");
      data_file_element->LinkEndChild(element);

      text = document->NewText(data_file_name.c_str());
      element->LinkEndChild(text);
   }

   // Display
//   {
//      element = document->NewElement("Display");
//      data_set_element->LinkEndChild(element);

//      buffer.str("");
//      buffer << display;

//      text = document->NewText(buffer.str().c_str());
//      element->LinkEndChild(text);
//   }

   return document;
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
        file_stream.OpenElement("missing_values_label");

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
        const size_t columns_number = get_columns_number();

        for(size_t i = 0; i < columns_number; i++)
        {
            file_stream.OpenElement("Column");

            file_stream.PushAttribute("Item", to_string(i).c_str());

            columns[i].write_XML(file_stream);

            file_stream.CloseElement();
        }
    }

    // Close columns

    file_stream.CloseElement();

    // Instances

    file_stream.OpenElement("Instances");

    // Instances number

    {
        file_stream.OpenElement("InstancesNumber");

        buffer.str("");
        buffer << get_instances_number();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Instances uses

    {
        file_stream.OpenElement("InstancesUses");

        buffer.str("");
        buffer << get_instances_uses();

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();
    }

    // Close instances


    // Close DataSet

    file_stream.CloseElement();
}


/// @todo

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

    const tinyxml2::XMLElement* columns_names_element = data_file_element->FirstChildElement("ColumnsName");

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

    // Rows label

    const tinyxml2::XMLElement* rows_label_element = data_file_element->FirstChildElement("RowsLabel");

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
        const size_t new_lags_number = static_cast<size_t>(atoi(lags_number_element->GetText()));

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
        const size_t new_steps_ahead = static_cast<size_t>(atoi(steps_ahead_element->GetText()));

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
        const size_t new_time_index = static_cast<size_t>(atoi(time_index_element->GetText()));

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

    if(columns_number_element->GetText())
    {
        const size_t new_columns_number = static_cast<size_t>(atoi(columns_number_element->GetText()));

        set_columns_number(new_columns_number);
    }

    // Read and set columns..
/*
    // Instances

    const tinyxml2::XMLElement* instances_element = data_set_element->FirstChildElement("Instances");

    if(!instances_element)
    {
       buffer << "OpenNN Exception: DataSet class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Instances element is nullptr.\n";

       throw logic_error(buffer.str());
    }

    // Instances number

    const tinyxml2::XMLElement* instances_number_element = instances_element->FirstChildElement("InstancesNumber");

    if(!instances_number_element)
    {
        buffer << "OpenNN Exception: DataSet class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Instances number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(instances_number_element->GetText())
    {
        const size_t new_instances_number = static_cast<size_t>(atoi(instances_number_element->GetText()));

        instances_uses.set(new_instances_number);

        set_instances_number(new_instances_number); // ?
    }
*/
    // Display

//    const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

//    if(display_element)
//    {
//        const string new_display_string = display_element->GetText();

//        try
//        {
//            set_display(new_display_string != "0");
//        }
//        catch(const logic_error& e)
//        {
//            cerr << e.what() << endl;
//        }
//    }
}


/// Returns a string representation of the current data set object.

string DataSet::object_to_string() const
{
   ostringstream buffer;

   buffer << "Data set object\n"
          << "Data file name: " << data_file_name << "\n"
          << "Header line: " << has_columns_names << "\n"
          << "Separator: " << separator << "\n"
          << "Missing values label: " << missing_values_label << "\n"
          << "Data:\n" << data << "\n"
          << "Display: " << display << "\n";

   return buffer.str();
}


/// Prints to the screen in text format the members of the data set object.

void DataSet::print() const
{
   if(display)
   {
      cout << object_to_string();
   }
}


/// Prints to the screen in text format the main numbers from the data set object.

void DataSet::print_summary() const
{
    if(display)
    {
        const size_t variables_number = get_variables_number();
        const size_t instances_number = get_instances_number();

        cout << "Data set object summary:\n"
             << "Number of variables: " << variables_number << "\n"
             << "Number of instances: " << instances_number << "\n";
    }
}


/// Saves the members of a data set object to a XML-type file in an XML-type format.
/// @param file_name Name of data set XML-type file.
///
/// @todo

void DataSet::save(const string& file_name) const
{
    FILE* file = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter filestream(file);
    write_XML(filestream);

    fclose(file);
}


/// Loads the members of a data set object from a XML-type file:
/// <ul>
/// <li> Instances number.
/// <li> Training instances number.
/// <li> Training instances indices.
/// <li> Selection instances number.
/// <li> Selection instances indices.
/// <li> Testing instances number.
/// <li> Testing instances indices.
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
    const size_t columns_number = get_columns_number();

    for(size_t i = 0; i < columns_number; i++)
    {
        if(columns[i].type == Numeric) cout << "Numeric ";
        else if(columns[i].type == Binary) cout << "Binary ";
        else if(columns[i].type == Categorical) cout << "Categorical ";
        else if(columns[i].type == DateTime) cout << "DateTime ";
    }

    cout << endl;
}


/// Prints to the screen the values of the data matrix.

void DataSet::print_data() const
{
   if(display) cout << data << endl;
}


/// Prints to the sceen a preview of the data matrix,
/// i.e., the first, second and last instances

void DataSet::print_data_preview() const
{
   if(display)
   {
       const size_t instances_number = get_instances_number();

       if(instances_number > 0)
       {
          const Vector<double> first_instance = data.get_row(0);

          cout << "First instance:\n"
                    << first_instance << endl;
       }

       if(instances_number > 1)
       {
          const Vector<double> second_instance = data.get_row(1);

          cout << "Second instance:\n"
                    << second_instance << endl;
       }

       if(instances_number > 2)
       {
          const Vector<double> last_instance = data.get_row(instances_number-1);

          cout << "Instance " << instances_number << ":\n"
                    << last_instance << endl;
       }
    }
}


/// Saves to the data file the values of the data matrix.

void DataSet::save_data() const
{
    data.save_csv(data_file_name);
}


/// Arranges an input-target matrix from a time series matrix, according to the number of lags.

void DataSet::transform_time_series()
{
    if(lags_number == 0) return;

//    time_series_data = data;

    time_series_columns = columns;

    delete_unused_instances();

    if(has_time_variables())
    {
        OpenNN::transform_time_series(data, lags_number, steps_ahead, time_index);
    }
    else
    {
        OpenNN::transform_time_series(data, lags_number, steps_ahead);
    }

cout << "After transform time series" << endl;
    transform_columns_time_series();

    Vector<InstanceUse> new_instance_uses(data.get_rows_number());

    instances_uses = new_instance_uses;

    const size_t inputs_number = get_input_variables_number();
    const size_t targets_number = get_target_variables_number();

    inputs_dimensions.set(Vector<size_t>({inputs_number}));

    targets_dimensions.set(Vector<size_t>({targets_number}));
}


/// Arranges the data set for association.
/// @todo Low priority. Variables and instances.

void DataSet::transform_association()
{
    OpenNN::transform_association(data);
}


void DataSet::delete_unused_instances()
{
    Vector<size_t> index(get_unused_instances_number());
    size_t j = 0;

    for (size_t i = 0; i < get_instances_number(); i++)
    {
        if(get_instance_use(i) == UnusedInstance)
        {
            index[j] = i;
            j++;
        }
    }

    data = data.delete_rows(index);


}


void DataSet::fill_time_series(const size_t& period )
{
    size_t rows = static_cast<size_t>((data(data.get_rows_number()- 1, 0)- data(0,0)) / period) + 1 ;

    Matrix<double> new_data(rows, data.get_columns_number());

    new_data.initialize(static_cast<double>(NAN));

    size_t j = 1;

    new_data.set_row(0, data.get_row(0));

    cout.precision(20);

    for (size_t i = 1; i < rows ; i++)
    {
      if(static_cast<size_t>(data(j, 0)) == static_cast<size_t>(data(j - 1, 0)))
      {

          j = j + 1;
      }
      if(static_cast<size_t>(data(j, 0)) == static_cast<size_t>(data(0,0) + i * period))
      {
          new_data.set_row(i, data.get_row(j));

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
    data.load_binary(data_file_name);
}


/// This method loads data from a binary data file for time series prediction methodata_set.

void DataSet::load_time_series_data_binary()
{
    time_series_data.load_binary(data_file_name);
}


/// Returns a vector containing the number of instances of each class in the data set.
/// If the number of target variables is one then the number of classes is two.
/// If the number of target variables is greater than one then the number of classes is equal to the number
/// of target variables.
/// @todo Low priority. Return class_distribution is wrong

Vector<size_t> DataSet::calculate_target_distribution() const
{
   const size_t instances_number = get_instances_number();
   const size_t targets_number = get_target_variables_number();
   const Vector<size_t> targets_indices = get_target_variables_indices();

   Vector<size_t> class_distribution;

   if(targets_number == 1) // Two classes
   {
      class_distribution.set(2, 0);

      size_t target_index = targets_indices[0];

      size_t positives = 0;
      size_t negatives = 0;

      for(int instance_index = 0; instance_index < static_cast<int>(instances_number); instance_index++)
      {
          if(!::isnan(data(static_cast<size_t>(instance_index),target_index)))
          {
              if(data(static_cast<size_t>(instance_index),target_index) < 0.5)
              {
                  negatives++;
              }
              else
              {
                  positives++;
              }
          }
      }

      class_distribution[0] = negatives;
      class_distribution[1] = positives;
   }
   else // More than two classes
   {
      class_distribution.set(targets_number, 0);

      for(size_t i = 0; i < instances_number; i++)
      {
          if(get_instance_use(i) != UnusedInstance)
          {
             for(size_t j = 0; j < targets_number; j++)
             {
                 if(data(i,targets_indices[j]) == static_cast<double>(NAN)) continue;

                 if(data(i,targets_indices[j]) > 0.5) class_distribution[j]++;
             }
          }
      }
   }

   return class_distribution;
}


/// This method balances the targets ditribution of a data set with only one target variable by unusing
/// instances whose target variable belongs to the most populated target class.
/// It returns a vector with the indices of the instances set unused.
/// @param percentage Percentage of instances to be unused.
/// @todo Low priority. "total unbalanced instances" needs target class distribution function.

Vector<size_t> DataSet::balance_binary_targets_distribution(const double& percentage)
{
    Vector<size_t> unused_instances;

    const size_t instances_number = get_used_instances_number();

    const Vector<size_t> target_class_distribution = calculate_target_distribution();

    const Vector<size_t> maximal_indices = OpenNN::maximal_indices(target_class_distribution.to_double_vector(), 2);

    const size_t maximal_target_class_index = maximal_indices[0];
    const size_t minimal_target_class_index = maximal_indices[1];

    size_t total_unbalanced_instances_number = static_cast<size_t>((percentage/100.0)*(target_class_distribution[maximal_target_class_index] - target_class_distribution[minimal_target_class_index]));

    size_t actual_unused_instances_number;

    size_t unbalanced_instances_number = total_unbalanced_instances_number/10;

    Vector<size_t> actual_unused_instances;

    while(total_unbalanced_instances_number != 0)
    {
        if(total_unbalanced_instances_number < instances_number/10)
        {
           unbalanced_instances_number = total_unbalanced_instances_number;
        }
        else if(total_unbalanced_instances_number > 0 && unbalanced_instances_number < 1)
        {
            unbalanced_instances_number = total_unbalanced_instances_number;
        }
/*
        actual_unused_instances = unuse_most_populated_target(unbalanced_instances_number);

        actual_unused_instances_number = actual_unused_size();

        unused_instances = unused_assemble(actual_unused_instances);

        total_unbalanced_instances_number = total_unbalanced_instances_number - actual_unused_instances_number;

        actual_unused_clear();
*/
    }

    return unused_instances;
}


/// This method balances the targets ditribution of a data set with more than one target variable by unusing
/// instances whose target variable belongs to the most populated target class.
/// It returns a vector with the indices of the instances set unused.
/// @todo "total unbalanced instances" needs target class distribution function

Vector<size_t> DataSet::balance_multiple_targets_distribution()
{
    Vector<size_t> unused_instances;

    const size_t bins_number = 10;

    const Vector<size_t> target_class_distribution = calculate_target_distribution();

    const size_t targets_number = get_target_variables_number();

    const Vector<size_t> inputs_variables_indices = get_input_variables_indices();
    const Vector<size_t> targets_variables_indices = get_target_variables_indices();

    const Vector<size_t> maximal_target_class_indices = maximal_indices(target_class_distribution.to_double_vector(), targets_number);

    const size_t minimal_target_class_index = maximal_target_class_indices[targets_number - 1];

    // Target class differences

    Vector<size_t> target_class_differences(targets_number);

    for(size_t i = 0; i < targets_number; i++)
    {
        target_class_differences[i] = target_class_distribution[i] - target_class_distribution[minimal_target_class_index];
    }

    Vector<double> instance;

    size_t count_instances = 0;

    size_t unbalanced_instances_number;

    size_t instances_number;
    Vector<size_t> instances_indices;

    Vector<Histogram> data_histograms;

    Matrix<size_t> total_frequencies;
    Vector<size_t> instance_frequencies;

    size_t maximal_difference_index;
    size_t instance_index;
    size_t instance_target_index;

    Vector<size_t> unbalanced_instances_indices;

    while(!target_class_differences.is_in(0, 0))
    {
        unbalanced_instances_indices.clear();
        instances_indices.clear();

        instances_indices = get_used_instances_indices();

        instances_number = instances_indices.size();

        maximal_difference_index = maximal_index(target_class_differences.to_double_vector());

        unbalanced_instances_number = static_cast<size_t>(target_class_differences[maximal_difference_index]/10);

        if(unbalanced_instances_number < 1)
        {
            unbalanced_instances_number = 1;
        }

        data_histograms = calculate_columns_histograms(bins_number);

        total_frequencies.clear();

        total_frequencies.set(instances_number, 2);

        count_instances = 0;

        for(size_t i = 0; i < instances_number; i++)
        {
            instance_index = instances_indices[i];

            instance = get_instance_data(instance_index);

            instance_target_index = targets_variables_indices[maximal_difference_index];

            if(instance[instance_target_index] == 1.0)
            {
                //instance_frequencies = instance.total_frequencies(data_histograms);

                total_frequencies(count_instances, 0) = instance_frequencies.calculate_partial_sum(inputs_variables_indices);
                total_frequencies(count_instances, 1) = instance_index;

                count_instances++;
            }
        }

        unbalanced_instances_indices = total_frequencies.sort_descending(0).get_column(1).get_first(unbalanced_instances_number);
/*
        unused_instances = unused_assemble(unbalanced_instances_indices);

        set_unused(unbalanced_instances_indices);

        target_class_differences[maximal_difference_index] = target_class_differences[maximal_difference_index] - unbalanced_instances_number;
*/
    }

    return unused_instances;
}


/// This method unuses a given number of instances of the most populated target.
/// If the given number is greater than the number of used instances which belongs to that target,
/// it unuses all the instances in that target.
/// If the given number is lower than 1, it unuses 1 instance.
/// @param instances_to_unuse Number of instances to set unused.
/// @todo Low priority. instance frequency

Vector<size_t> DataSet::unuse_most_populated_target(const size_t& instances_to_unuse)
{
    Vector<size_t> most_populated_instances(instances_to_unuse);

    if(instances_to_unuse == 0)
    {
        return most_populated_instances;
    }

    const size_t bins_number = 10;

    // Variables

    const size_t targets_number = get_target_variables_number();

    const Vector<size_t> inputs = get_input_variables_indices();
    const Vector<size_t> targets = get_target_variables_indices();

    const Vector<size_t> unused_variables = get_unused_variables_indices();

    // Instances

    const Vector<size_t> used_instances = get_used_instances_indices();

    const size_t used_instances_number = get_used_instances_number();

    // Most populated target

    const Vector<Histogram> data_histograms = calculate_columns_histograms(bins_number);

    size_t most_populated_target = 0;
    size_t most_populated_bin = 0;

    size_t frequency;
    size_t maximum_frequency = 0;

    size_t unused = 0;

    for(size_t i = 0; i < targets_number; i++)
    {
        frequency = data_histograms[targets[i] - unused_variables.count_less_than(targets[i])].calculate_maximum_frequency();

        if(frequency > maximum_frequency)
        {
            unused = unused_variables.count_less_than(targets[i]);

            maximum_frequency = frequency;

            most_populated_target = targets[i];

            most_populated_bin = data_histograms[targets[i] - unused].calculate_most_populated_bin();
        }
    }

    // Calculates frequencies of the instances which belong to the most populated target

    size_t index;
    size_t bin;
    double value;
    Vector<double> instance;

    Vector<size_t> instance_frequencies;

    Matrix<size_t> total_instances_frequencies(maximum_frequency, 2);

    size_t count_instances = 0;

    for(size_t i = 0; i < used_instances_number; i++)
    {
        index = used_instances[static_cast<size_t>(i)];

        instance = get_instance_data(index);

        value = instance[most_populated_target];

        bin = data_histograms[most_populated_target - unused].calculate_bin(value);

        if(bin == most_populated_bin)
        {

//            instance_frequencies = instance.total_frequencies(data_histograms);
//            instance_frequencies = total_frequencies(data_histograms);

            total_instances_frequencies(count_instances, 0) = instance_frequencies.calculate_partial_sum(inputs);
            total_instances_frequencies(count_instances, 1) = used_instances[static_cast<size_t>(i)];

            count_instances++;
        }
    }

    // Unuses instances

    if(instances_to_unuse > maximum_frequency)
    {
        most_populated_instances = total_instances_frequencies.sort_descending(0).get_column(1).get_first(maximum_frequency);
    }
    else
    {
        most_populated_instances = total_instances_frequencies.sort_descending(0).get_column(1).get_first(instances_to_unuse);
    }

    set_instances_unused(most_populated_instances);

    return most_populated_instances;
}


/// This method balances the target ditribution of a data set for a function regression problem.
/// It returns a vector with the indices of the instances set unused.
/// It unuses a given percentage of the
/// @param percentage Percentage of the instances to be unused.
/// @todo Low priority.

Vector<size_t> DataSet::balance_approximation_targets_distribution(const double& percentage)
{
    Vector<size_t> unused_instances;

    const size_t instances_number = get_used_instances_number();

    const size_t instances_to_unuse = static_cast<size_t>(instances_number*percentage/100.0);

    size_t count;

/*
    while(unused_size() < instances_to_unuse)
    {
        if(instances_to_unuse - unused_size() < instances_to_unuse/10)
        {
            count = instances_to_unuse - unused_size();
        }
        else
        {
            count = instances_to_unuse/10;
        }

        if(count == 0)
        {
            count = 1;
        }

        unused_instances = unused_assemble(unuse_most_populated_target(count));
    }
*/
    return unused_instances;
}


/// Calculate the outliers from the data set using the Tukey's test for a single variable.
/// @param variable_index Index of the variable to calculate the outliers.
/// @param cleaning_parameter Parameter used to detect outliers.
/// @todo Low priority.

Vector<size_t> DataSet::calculate_Tukey_outliers(const size_t& column_index, const double& cleaning_parameter) const
{
    Vector<size_t> outliers;

    if(columns[column_index].type != Numeric) return outliers;

    const size_t instances_number = get_used_instances_number();
    const Vector<size_t> instances_indices = get_used_instances_indices();

    double interquartile_range;

    const BoxPlot box_plot = OpenNN::box_plot(get_column_data(column_index).to_vector());

    if(abs(box_plot.third_quartile - box_plot.first_quartile) < numeric_limits<double>::epsilon())
    {
        return outliers;
    }
    else
    {
        interquartile_range = abs((box_plot.third_quartile - box_plot.first_quartile));
    }
/*
    for(size_t j = 0; j < instances_number; j++)
    {
        const Vector<double> instance = get_instance(instances_indices[j]);

        if(instance[variable_index] < (box_plot[1] - cleaning_parameter*interquartile_range))
        {
            outliers.push_back(instances_indices[j]);
        }
        else if(instance[variable_index] >(box_plot[3] + cleaning_parameter*interquartile_range))
        {
            outliers.push_back(instances_indices[j]);
        }
    }
*/
    return outliers;
}


/// Calculate the outliers from the data set using the Tukey's test.
/// @param cleaning_parameter Parameter used to detect outliers.
/// @todo Low priority.

Vector<Vector<size_t>> DataSet::calculate_Tukey_outliers(const double& cleaning_parameter) const
{
    const size_t instances_number = get_used_instances_number();
    const Vector<size_t> instances_indices = get_used_instances_indices();

    const size_t variables_number = get_used_variables_number();
    const Vector<size_t> used_variables_indices = get_used_columns_indices();

    double interquartile_range;

    Vector<Vector<size_t>> return_values(2);
    return_values[0] = Vector<size_t>(instances_number, 0);
    return_values[1] = Vector<size_t>(variables_number, 0);

    size_t variable_index;

    Vector<BoxPlot> box_plots(variables_number);
/*
    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        variable_index = used_variables_indices[static_cast<size_t>(i)];

        if(is_binary_variable(variable_index)) continue;

        box_plots[static_cast<size_t>(i)] = box_plot(data.get_column(variable_index));
    }

    for(int i = 0; i < static_cast<int>(variables_number); i++)
    {
        variable_index = used_variables_indices[static_cast<size_t>(i)];

        if(is_binary_variable(variable_index)) continue;

        const BoxPlot variable_box_plot = box_plots[static_cast<size_t>(i)];

        if(abs(variable_box_plot[3] - variable_box_plot[1]) < numeric_limits<double>::epsilon())
        {
            continue;
        }
        else
        {
            interquartile_range = abs((variable_box_plot[3] - variable_box_plot[1]));
        }

        size_t variables_outliers = 0;

        for(int j = 0; j < static_cast<int>(instances_number); j++)
        {
            const Vector<double> instance = get_instance(instances_indices[static_cast<size_t>(j)]);

            if(instance[variable_index] <(variable_box_plot[1] - cleaning_parameter*interquartile_range) ||
               instance[variable_index] >(variable_box_plot[3] + cleaning_parameter*interquartile_range))
            {
                    return_values[0][static_cast<size_t>(j)] = 1;

                    variables_outliers++;
            }
        }

        return_values[1][static_cast<size_t>(i)] = variables_outliers;
    }
*/
    return return_values;
}


/// Calculate the outliers from the data set using the Tukey's test and sets in instances object.
/// @param cleaning_parameter Parameter used to detect outliers

void DataSet::unuse_Tukey_outliers(const double& cleaning_parameter)
{
    const Vector<Vector<size_t>> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    const Vector<size_t> outliers_instances = outliers_indices[0].get_indices_greater_than(0);

    set_instances_unused(outliers_instances);
}


/// Returns a matrix with the values of autocorrelation for every variable in the data set.
/// The number of rows is equal to the number of
/// The number of columns is the maximum lags number.
/// @param maximum_lags_number Maximum lags number for which autocorrelation is calculated.

Matrix<double> DataSet::calculate_autocorrelations(const size_t& maximum_lags_number) const
{
    if(maximum_lags_number > get_used_instances_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> autocorrelations(const size_t&) method.\n"
               << "Maximum lags number(" << maximum_lags_number << ") is greater than the number of instances("
               << get_used_instances_number() <<") \n";

        throw logic_error(buffer.str());
    }

    const size_t variables_number = data.get_columns_number();

    Matrix<double> autocorrelations(variables_number, maximum_lags_number);

        for(size_t j = 0; j < variables_number; j++)
        {
            autocorrelations.set_row(j, OpenNN::autocorrelations(data.get_column(j), maximum_lags_number));
        }


    return autocorrelations;
}


/// Calculates the cross-correlation between all the variables in the data set.

Matrix<Vector<double>> DataSet::calculate_cross_correlations(const size_t& lags_number) const
{
    const size_t variables_number = get_variables_number();

    Matrix<Vector<double>> cross_correlations(variables_number, variables_number);

    Vector<double> actual_column;

    for(size_t i = 0; i < variables_number; i++)
    {
        actual_column = data.get_column(i);

        for(size_t j = 0; j < variables_number; j++)
        {
            cross_correlations(i,j) = OpenNN::cross_correlations(actual_column, data.get_column(j), lags_number);
        }
    }

    return cross_correlations;
}


/// @todo, check

Matrix<double> DataSet::calculate_lag_plot() const
{
    const size_t instances_number = get_used_instances_number();

    const size_t columns_number = data.get_columns_number() - 1;

    Matrix<double> lag_plot(instances_number, columns_number);

    Vector<size_t> columns_indices(1, 1, columns_number);

    lag_plot = data.get_submatrix_columns(columns_indices);

    return lag_plot;
}


/// @todo, check

Matrix<double> DataSet::calculate_lag_plot(const size_t& maximum_lags_number)
{
    const size_t instances_number = get_used_instances_number();

    if(maximum_lags_number > instances_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Matrix<double> calculate_lag_plot(const size_t&) method.\n"
               << "Maximum lags number(" << maximum_lags_number
               << ") is greater than the number of instances("
               << instances_number << ") \n";

        throw logic_error(buffer.str());
    }

    //const Matrix<double> lag_plot = time_series_data.calculate_lag_plot(maximum_lags_number, time_index);


//    return lag_plot;

    return Matrix<double>();
}


/// Generates an artificial dataset with a given number of instances and number of variables
/// by constant data.
/// @param instances_number Number of instances in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_constant_data(const size_t& instances_number, const size_t& variables_number)
{
    set(instances_number, variables_number);

    data.randomize_uniform(-5.12, 5.12);

    for(size_t i = 0; i < instances_number; i++)
    {
        data(i, variables_number-1) = 0.0;
    }

    scale_minimum_maximum(data);

    set_default_columns_uses();
}


/// Generates an artificial dataset with a given number of instances and number of variables
/// using random data.
/// @param instances_number Number of instances in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_random_data(const size_t& instances_number, const size_t& variables_number)
{
    set(instances_number, variables_number);

    data.randomize_uniform(0.0, 1.0);
}


/// Generates an artificial dataset with a given number of instances and number of variables
/// using a sequential data.
/// @param instances_number Number of instances in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_sequential_data(const size_t& instances_number, const size_t& variables_number)
{
    set(instances_number, variables_number);

    for(size_t i = 0; i < instances_number; i++)
    {
        for(size_t j = 0; j < variables_number; j++)
        {
            data(i,j) = static_cast<double>(j);
        }
    }
}


/// Generates an artificial dataset with a given number of instances and number of variables
/// using a paraboloid data.
/// @param instances_number Number of instances in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_paraboloid_data(const size_t& instances_number, const size_t& variables_number)
{
    const size_t inputs_number = variables_number-1;

    set(instances_number, variables_number);

    data.randomize_uniform(-5.12, 5.12);

    for(size_t i = 0; i < instances_number; i++)
    {
        const double norm = l2_norm(data.get_row(i).delete_last(1));

        data(i, inputs_number) = norm*norm;
    }

    scale_minimum_maximum(data);
}


/// Generates an artificial dataset with a given number of instances and number of variables
/// using the Rosenbrock function.
/// @param instances_number Number of instances in the dataset.
/// @param variables_number Number of variables in the dataset.

void DataSet::generate_Rosenbrock_data(const size_t& instances_number, const size_t& variables_number)
{
    const size_t inputs_number = variables_number-1;

    set(instances_number, variables_number);

    data.randomize_uniform(-2.048, 2.048);

    double rosenbrock;

    for(int i = 0; i < static_cast<int>(instances_number); i++)
    {
        rosenbrock = 0.0;

        for(size_t j = 0; j < inputs_number-1; j++)
        {
            rosenbrock +=
           (1.0 - data(static_cast<size_t>(i),j))*(1.0 - data(static_cast<size_t>(i),j))
            + 100.0*(data(static_cast<size_t>(i),j+1)-data(static_cast<size_t>(i),j)*data(static_cast<size_t>(i),j))*
                    (data(static_cast<size_t>(i),j+1)-data(static_cast<size_t>(i),j)*data(static_cast<size_t>(i),j));
        }

        data(static_cast<size_t>(i), inputs_number) = rosenbrock;
    }

    scale_range(data, -1.0, 1.0);

    set_default_columns_uses();
}


void DataSet::generate_inputs_selection_data(const size_t& instances_number, const size_t& variables_number)
{
    set(instances_number,variables_number);

    data.randomize_uniform(0.0, 1.0);

    for(size_t i = 0; i < instances_number; i++)
    {
        for(size_t j = 0; j < variables_number-2; j++)
        {
            data(i,variables_number-1) += data(i,j);
        }
    }

    set_default_columns_uses();
}


void DataSet::generate_sum_data(const size_t& instances_number, const size_t& variables_number)
{
    set(instances_number,variables_number);

    data.randomize_uniform(0.0, 1.0);

    for(size_t i = 0; i < instances_number; i++)
    {
        for(size_t j = 0; j < variables_number-1; j++)
        {
            data(i,variables_number-1) += data(i,j);
        }
    }

    set_default();

    scale_data_mean_standard_deviation();

}


/// Generate artificial data for a binary classification problem with a given number of instances and inputs.
/// @param instances_number Number of the instances to generate.
/// @param inputs_number Number of the variables that the data set will have.

void DataSet::generate_data_binary_classification(const size_t& instances_number, const size_t& inputs_number)
{
    const size_t negatives = instances_number/2;
    const size_t positives = instances_number - negatives;

    // Negatives data

    Vector<double> target_0(negatives, 0.0);

    Matrix<double> class_0(negatives, inputs_number+1);

    class_0.randomize_normal(-0.5, 1.0);

    class_0.set_column(inputs_number, target_0, "");

    // Positives data

    Vector<double> target_1(positives, 1.0);

    Matrix<double> class_1(positives, inputs_number+1);

    class_1.randomize_normal(0.5, 1.0);

    class_1.set_column(inputs_number, target_1, "");

    // Assemble

    set(class_0.assemble_rows(class_1));
}


/// @todo Low priority.

void DataSet::generate_data_multiple_classification(const size_t& instances_number, const size_t& inputs_number, const size_t& outputs_number)
{
    Matrix<double> new_data(instances_number, inputs_number);
    new_data.randomize_normal();

    Matrix<double> targets(instances_number, outputs_number, 0.0);

    size_t target_index = 0;

    for(size_t i = 0; i < instances_number; i ++)
    {
        target_index = static_cast<unsigned>(rand())%outputs_number;

        targets(i, target_index) = 1.0;
    }

    set(new_data.assemble_columns(targets));
}


/// Returns true if the data matrix is not empty(it has not been loaded),
/// and false otherwise.

bool DataSet::has_data() const
{
    if(data.empty())
    {
        return false;
    }
    else
    {
        return true;
    }
}


/// Unuses those instances with values outside a defined range.
/// @param minimums Vector of minimum values in the range.
/// The size must be equal to the number of variables.
/// @param maximums Vector of maximum values in the range.
/// The size must be equal to the number of variables.
/// @todo Low priority.

Vector<size_t> DataSet::filter_data(const Vector<double>& minimums, const Vector<double>& maximums)
{
    const Vector<size_t> used_variables_indices = get_used_columns_indices();

    const size_t used_variables_number = get_used_variables_number();

    #ifdef __OPENNN_DEBUG__

    if(minimums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&) method.\n"
               << "Size of minimums(" << minimums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    if(maximums.size() != used_variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Vector<size_t> filter_data(const Vector<double>&, const Vector<double>&) method.\n"
               << "Size of maximums(" << maximums.size() << ") is not equal to number of variables(" << used_variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    #endif

    const size_t instances_number = get_instances_number();

    Vector<double> filtered_indices(instances_number, 0.0);

    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    for(size_t j = 0; j < used_variables_number; j++)
    {
        const size_t current_variable_index = used_variables_indices[j];

        const Vector<size_t> current_instances_indices = used_instances_indices;

        const size_t current_instances_number = current_instances_indices.size();

        for(size_t i = 0; i < current_instances_number; i++)
        {
            const size_t current_instance_index = current_instances_indices[i];

            if(data(current_instance_index,current_variable_index) < minimums[j]
            || data(current_instance_index,current_variable_index) > maximums[j])
            {
                filtered_indices[current_instance_index] = 1.0;

                set_instance_use(current_instance_index, UnusedInstance);
            }
        }
    }

    return filtered_indices.get_indices_greater_than(0.5);
}


/// Filter data set variable using a rank.
/// The values within the variable must be between minimum and maximum.
/// @param variable_index Index number where the variable to be filtered is located.
/// @param minimum Value that determine the lower limit.
/// @param maximum Value that determine the upper limit.
/// Returns a indices vector.

Vector<size_t> DataSet::filter_column(const size_t& variable_index, const double& minimum, const double& maximum)
{
    const size_t instances_number = get_instances_number();

    Vector<double> filtered_indices(instances_number, 0.0);

    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    const Vector<size_t> current_instances_indices = used_instances_indices;

    const size_t current_instances_number = current_instances_indices.size();

    for(size_t i = 0; i < current_instances_number; i++)
    {
        const size_t index = current_instances_indices[i];

        if(data(index,variable_index) < minimum || data(index,variable_index) > maximum)
        {
            filtered_indices[index] = 1.0;

            set_instance_use(index, UnusedInstance);
        }
    }
    return(filtered_indices.get_indices_greater_than(0.5));
}


/// Filter data set variable using a rank.
/// The values within the variable must be between minimum and maximum.
/// @param variable_name String name where the variable to be filtered is located.
/// @param minimum Value that determine the lower limit.
/// @param maximum Value that determine the upper limit.
/// Returns a indices vector.

Vector<size_t> DataSet::filter_column(const string& variable_name, const double& minimum, const double& maximum)
{
    const size_t variable_index = get_variable_index(variable_name);

    const size_t instances_number = get_instances_number();

    Vector<double> filtered_indices(instances_number, 0.0);

    const Vector<size_t> used_instances_indices = get_used_instances_indices();

    const size_t current_instances_number = used_instances_indices.size();

    for(size_t i = 0; i < current_instances_number; i++)
    {
        const size_t index = used_instances_indices[i];

        if(data(index,variable_index) < minimum || data(index,variable_index) > maximum)
        {
            filtered_indices[index] = 1.0;

            set_instance_use(index, UnusedInstance);
        }
    }

    return filtered_indices.get_indices_greater_than(0.5);
}


/// This method converts a numerical variable into categorical.
/// Note that this method resizes the dataset.
/// @param variable_index Index of the variable to be converted.

void DataSet::numeric_to_categorical(const size_t& variable_index)
{
    #ifdef __OPENNN_DEBUG__

    const size_t variables_number = get_variables_number();

    if(variable_index >= variables_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void convert_categorical_variable(const size_t&) method.\n"
               << "Index of variable(" << variable_index << ") must be less than number of variables (" << variables_number << ").\n";

        throw logic_error(buffer.str());
    }

    #endif

    const Vector<double> categories = data.get_column(variable_index).get_unique_elements();

    data = data.to_categorical(variable_index);

    columns[variable_index].set_categories_uses(Vector<VariableUse>(categories.size(), columns[variable_index].column_use));
    columns[variable_index].type = Categorical;
    columns[variable_index].categories = categories.to_string_vector();
}


/// Sets all the instances with missing values to "Unused".

void DataSet::impute_missing_values_unuse()
{
    const size_t instances_number = get_instances_number();

#pragma omp parallel for

    for(size_t i = 0; i <instances_number; i++)
    {
        if(data.has_nan_row(i))
        {
            set_instance_use(i, "Unused");
        }
    }
}

/// Substitutes all the missing values by the mean of the corresponding variable.

void DataSet::impute_missing_values_mean()
{
    const Vector<size_t> used_columns_indices = get_used_columns_indices();

    const Vector<double> means = mean_missing_values(data, Vector<size_t>(0,1,data.get_rows_number()-1),used_columns_indices);

    const size_t variables_number = used_columns_indices.size();
    const size_t instances_number = get_instances_number();

    cout<<"instances number"<< instances_number<<endl;
    cout<<"rows"<<data.get_rows_number()<<endl;

    #pragma omp parallel for schedule(dynamic)

    for(size_t j = 0; j < variables_number; j++)
    {
        for(size_t i = 0 ; i < instances_number - 1 ; i++)
        {
            if(::isnan(data(i,j))) data(i,j) = means[j];
        }
    }
}


/// Substitutes all the missing values by the median of the corresponding variable.

void DataSet::impute_missing_values_median()
{
    const Vector<size_t> used_columns_indices = get_used_columns_indices();

    const Vector<double> medians = median_missing_values(data, Vector<size_t>(0,1,data.get_rows_number()-1),used_columns_indices);

    const size_t variables_number = used_columns_indices.size();
    const size_t instances_number = get_instances_number();

    #pragma omp parallel for schedule(dynamic)

    for(size_t j = 0; j < variables_number; j++)
    {
        for(size_t i = 0 ; i < instances_number ; i++)
        {
            if(::isnan(data(i,j))) data(i,j) = medians[j];
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


void DataSet::read_csv()
{
    read_csv_1();

    if(!has_time_variables() && !has_categorical_variables())
    {
        read_csv_2_simple();
        read_csv_3_simple();
    }
    else
    {
        read_csv_2_complete();
        read_csv_3_complete();
    }


/*

        // Fill time series
cout << "hello" << endl;
        const size_t period = static_cast<size_t>(data(1, time_index) - data(0,time_index));

        if(static_cast<size_t>((data(data.get_rows_number() - 1, time_index) - data(0,time_index))/period) + 1 == data.get_rows_number())
        {
            // Do nothing
        }
        else
        {
            fill_time_series(period);
        }


         scrub_missing_values();
cout<<"hello"<<endl;
        // Transform time series

        transform_time_series();
        split_instances_random(0.75,0,0.25);
        */

}


Vector<string> DataSet::get_default_columns_names(const size_t& columns_number)
{
    Vector<string> columns_names(columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        ostringstream buffer;

        buffer << "column_" << i+1;

        columns_names[i] = buffer.str();
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

    size_t lines_number = 3;

    data_file_preview.set(lines_number);

    string line;

    size_t lines_count = 0;

    while(file.good())
    {
        getline(file, line);

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

    if(data_file_preview[0].empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv() method.\n"
               << "File " << data_file_name << " is empty.\n";

        throw logic_error(buffer.str());
    }

    // Set rows labels and columns names

    if(contains_substring(data_file_preview[0][0], "id"))
    {
        has_rows_labels = true;
    }

    const size_t columns_number = data_file_preview[0].size();

    columns.set(columns_number);

    // Check if header has numeric value

    if(has_columns_names && has_numbers(data_file_preview[0]))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void read_csv_1() method.\n"
               << "Some columns names are numeric.\n";

        throw logic_error(buffer.str());
    }

    // Columns names

    if(has_columns_names)
    {
        set_columns_names(data_file_preview[0]);
    }
    else
    {        
        set_columns_names(get_default_columns_names(columns_number));
    }

    // Columns types

    for(size_t i = 0; i < columns_number; i++)
    {
        if(is_date_time_string(data_file_preview[1][i]))
        {
            columns[i].type = DateTime;
        }
        else if(is_numeric_string(data_file_preview[1][i]))
        {
            columns[i].type = Numeric;
        }
        else
        {
            columns[i].type = Categorical;
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
              << "void read_csv() method.\n"
              << "Cannot open data file: " << data_file_name << "\n";

       throw logic_error(buffer.str());
    }

    const char separator_char = get_separator_char();
    const size_t columns_number = get_columns_number();

    string line;
    size_t line_number = 0;

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

    size_t instances_count = 0;

    size_t tokens_count;

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
                   << "void read_csv() method.\n"
                   << "Line " << line_number << ": Size of tokens(" << tokens_count << ") is not equal to number of columns(" << columns_number << ").\n";

            throw logic_error(buffer.str());
        }

        instances_count++;
    }

    file.close();

    data.set(instances_count, columns_number);

    data.set_header(get_columns_names());

    set_default_columns_uses();

    instances_uses.set(instances_count);

    split_instances_random();
}


void DataSet::read_csv_3_simple()
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

    const size_t variables_number = get_variables_number();

    string line;

    Vector<string> tokens;

    size_t instance_index = 0;

    // Read header

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

    while(file.good())
    {
        getline(file, line);

        trim(line);

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_char);

        for(size_t j = 0; j < variables_number; j++)
        {
            trim(tokens[j]);

            erase(line, '"');

            if(tokens[j] == missing_values_label || tokens[j].empty())
            {
                data(instance_index, j) = static_cast<double>(NAN);
            }
            else
            {
                data(instance_index, j) = stod(tokens[j]);
            }
        }

        instance_index++;
    }

    // Check Binary

    for(size_t k = 0; k < variables_number; k++)
    {
        if(data.is_column_binary(k))
        {
            columns[k].type = Binary;
        }
    }

    file.close();
}


void DataSet::read_csv_2_complete()
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

    string line;

    Vector<string> tokens;

    size_t lines_count = 0;
    size_t tokens_count;

    const size_t columns_number = columns.size();

    for(unsigned j = 0; j < columns_number; j++)
    {
        if(columns[j].type != Categorical)
        {
            columns[j].column_use = Input;
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

    while(file.good())
    {
        getline(file, line);

        trim(line);

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_char);

        tokens_count = tokens.size();

        if(static_cast<unsigned>(tokens_count) != columns_number)
        {
            const string message =
//                    "Instance " + to_string(lines_count+1) + " error:\n"
//                    "Size of tokens (" + string::number(tokens_count) + ") is not equal to number of columns (" + string::number(totalColumnsNumber) + ").\n"
                    "Please check the format of the data file.";

            throw logic_error(message);
        }

        for(unsigned j = 0; j < columns_number; j++)
        {
            trim(tokens[j]);

            if(columns[j].type == Categorical)
            {
                if(!columns[j].categories.contains(tokens[j]))
                {
                    if(tokens[j] == missing_values_label) continue;

                    columns[j].categories.push_back(tokens[j]);
                    columns[j].categories_uses.push_back(Input);
                }
            }
        }

        lines_count++;
    }

    for(unsigned j = 0; j < columns_number; j++)
    {
         if(columns[j].type == Categorical)
         {
             if(columns[j].categories.size() == 2)
             {
                 columns[j].type = Binary;
                 columns[j].categories.set();
                 columns[j].categories_uses.set();
             }
         }
    }

    file.close();

    const size_t instances_number = static_cast<unsigned>(lines_count);

    const size_t variables_number = get_variables_number();

    data.set(static_cast<size_t>(instances_number), variables_number);

    set_default_columns_uses();

    instances_uses.set(static_cast<size_t>(instances_number));

    split_instances_random();
}


void DataSet::read_csv_3_complete()
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

    const size_t columns_number = columns.size();

    string line;

    Vector<string> tokens;

    string token;

    unsigned instance_index = 0;

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

    while(file.good())
    {
          getline(file, line);

          trim(line);

          if(line.empty()) continue;

          tokens = get_tokens(line, separator_char);

          for(size_t j = 0; j < columns_number; j++)
          {
              trim(tokens[j]);

              erase(line, '"');

                if(columns[j].type == Numeric)
                {
                    if(tokens[j] == missing_values_label || tokens[j].empty())
                    {
                        data(instance_index, j) = static_cast<double>(NAN);
                    }
                    else
                    {
                        try
                        {
                            data(instance_index, j) = stod(tokens[j]);
                        }
                        catch (invalid_argument)
                        {
                            ostringstream buffer;

                            buffer << "OpenNN Exception: DataSet class.\n"
                                   << "void read_csv() method.\n"
                                   << "Instance " << instance_index << "; Invalid number: " << tokens[j] << "\n";

                            throw logic_error(buffer.str());
                        }
                    }
                }
                else if(columns[j].type == DateTime)
                {
                    if(tokens[j] == missing_values_label || tokens[j].empty())
                    {
                        data(instance_index, j) = static_cast<double>(NAN);
                    }
                    else
                    {
                        data(instance_index, j) = static_cast<double>(date_to_timestamp(tokens[j], gmt));
                    }
                }
                else if(columns[j].type == Categorical)
                {
                    const Vector<size_t> variable_indices = get_variable_indices(j);

                    for(size_t k = 0; k < variable_indices.size(); k++)
                    {
                        if(tokens[j] == missing_values_label)
                        {
                            data(instance_index, variable_indices[k]) = static_cast<double>(NAN);
                        }
                        else if(tokens[j] == columns[j].categories[k])
                        {
                            data(instance_index, variable_indices[k]) = 1.0;
                        }
                    }
                }
                else if(columns[j].type == Binary)
                {
                    const Vector<size_t> variable_indices = get_variable_indices(j);

                    if(tokens[j] == missing_values_label)
                    {
                        data(instance_index, variable_indices[0]) = static_cast<double>(NAN);
                    }
                    else if(tokens[j] == columns[j].name)
                    {
                        data(instance_index, variable_indices[0]) = 1.0;
                    }
                }
          }

          instance_index++;
    }

    // Read header

    for (size_t j = 0; j < columns_number; j++)
    {
        if(columns[j].type == Categorical)
        {
            const Vector<size_t> variable_indices = get_variable_indices(j);

            for(size_t k = 0; k < variable_indices.size(); k++)
            {
                data.set_header(variable_indices[k], columns[j].categories[k]);
            }
        }
        else // Binary, DateTime, Numeric
        {
            data.set_header(j,columns[j].name);
        }
    }

    file.close();
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
            "Error: Found comma (',') in data file " + data_file_name + ", but separator is semicolon (';').";

            throw logic_error(message);
        }
    }
}


bool DataSet::has_categorical_variables() const
{
    const size_t variables_number = columns.size();

    for(size_t i = 0; i < variables_number; i++)
    {
        if(columns[i].type == Categorical) return true;
    }

    return false;
}


bool DataSet::has_time_variables() const
{
    const size_t variables_number = columns.size();

    for(size_t i = 0; i < variables_number; i++)
    {
        if(columns[i].type == DateTime) return true;
    }

    return false;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
