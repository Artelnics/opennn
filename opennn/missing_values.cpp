/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M I S S I N G   V A L U E S   C L A S S                                                                    */
/*                                                                                                              */ 
/*   Roberto Lopez                                                                                              */ 
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "missing_values.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a missing values object with zero missing values.
/// It also initializes the rest of class members to their default values.

MissingValues::MissingValues(void)
{
    set();

    set_default();
}


// INSTANCES AND VARIABLES NUMBER CONSTRUCTOR

/// Missing values number constructor.
/// It creates a data set object with a given number of missing values.
/// It also initializes the rest of class members to their default values.

MissingValues::MissingValues(const size_t& new_instances_number,const size_t& new_variables_number)
{
    set(new_instances_number, new_variables_number);

    set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a missing balues object by loading the object members from a TinyXML document.
/// @param missing_values_document XML document from the TinyXML library.

MissingValues::MissingValues(const tinyxml2::XMLDocument& missing_values_document)
{   
    set(missing_values_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing MissingValues object.
/// @param other_missing_values Missing balues object to be copied.

MissingValues::MissingValues(const MissingValues& other_missing_values)
{
    instances_number = other_missing_values.instances_number;
    variables_number = other_missing_values.variables_number;

    scrubbing_method = other_missing_values.scrubbing_method;

    items = other_missing_values.items;

    display = other_missing_values.display;
}


// DESTRUCTOR

/// Destructor. 

MissingValues::~MissingValues(void)
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to the current object the members of an existing missing values object.
/// @param other_missing_values Missing values object to be assigned.

MissingValues& MissingValues::operator=(const MissingValues& other_missing_values)
{
    if(this != &other_missing_values)
    {
        instances_number = other_missing_values.instances_number;
        variables_number = other_missing_values.variables_number;

        items = other_missing_values.items;

        display = other_missing_values.display;
    }

    return(*this);
}


// EQUAL TO OPERATOR

// bool operator == (const MissingValues&) const method

/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_missing_values Missing values object to be compared with.

bool MissingValues::operator == (const MissingValues& other_missing_values) const
{
    if(items == other_missing_values.items
            && display == other_missing_values.display)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// METHODS

// size_t get_instances_number(void) const method

/// Returns the number of instances in the data set.

size_t MissingValues::get_instances_number(void) const
{
    return(instances_number);
}


// size_t get_variables_number(void) const method

/// Returns the number of variables in the data set.

size_t MissingValues::get_variables_number(void) const
{
    return(variables_number);
}


// Vector<size_t> get_missing_values_numbers(void) const method

/// Returns a vector with the number of missing values for each variable in the data set.
/// The size of the vector is the number of variables.

Vector<size_t> MissingValues::get_missing_values_numbers(void) const
{
    Vector<size_t> missing_values_numbers(variables_number, 0);

    const size_t missing_values_number = get_missing_values_number();

    size_t variable_index;

    for(size_t i = 0; i < missing_values_number; i++)
    {
        variable_index = items[i].variable_index;

        missing_values_numbers[variable_index]++;
    }

    return(missing_values_numbers);
}


// const Vector<Item>& get_items(void) const method

/// Returns a constant reference to the vector of missing value items.

const Vector<MissingValues::Item>& MissingValues::get_items(void) const
{
    return(items);
}


// const Item& get_item(const size_t&) const method

/// Returns a single missing value, with information about the instance and variable indices.
/// @param index Index of missing value.

const MissingValues::Item& MissingValues::get_item(const size_t& index) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t missing_values_number = get_missing_values_number();

    if(index >= missing_values_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "const Item& get_item(const size_t&) const method.\n"
               << "Index (" << index << ") must be less than number of missing values (" << missing_values_number << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    return(items[index]);
}


// ScrubbingMethod get_scrubbing_method(void) const

/// Returns the method to be used for dealing with the missing values.

MissingValues::ScrubbingMethod MissingValues::get_scrubbing_method(void) const
{
    return(scrubbing_method);
}


// std::string write_scrubbing_method(void) const method

/// Returns a string with the name of the method used for scrubbing.

std::string MissingValues::write_scrubbing_method(void) const
{
    if(scrubbing_method == Unuse)
    {
        return("Unuse");
    }
    else if(scrubbing_method == Mean)
    {
        return("Mean");
    }
    if(scrubbing_method == NoScrubbing)
    {
        return("NoScrubbing");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "std::string write_scrubbing_method(void) const method.\n"
               << "Unknown scrubbing method.\n";

        throw std::logic_error(buffer.str());
    }
}


// std::string write_scrubbing_method_text(void) const method

/// Returns a string with the name of the method used for scrubbing, as paragaph text.

std::string MissingValues::write_scrubbing_method_text(void) const
{
    if(scrubbing_method == Unuse)
    {
        return("unuse");
    }
    else if(scrubbing_method == Mean)
    {
        return("mean");
    }
    else if(scrubbing_method == NoScrubbing)
    {
        return("no scrubbing");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "std::string write_scrubbing_method_text(void) const method.\n"
               << "Unknown scrubbing method.\n";

        throw std::logic_error(buffer.str());
    }
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen,
/// or false if messages from this class can't be displayed on the screen.

const bool& MissingValues::get_display(void) const
{
    return(display);
}


// void set(void) method

/// Sets a missing values object with zero instances, variables and missing values.

void MissingValues::set(void)
{
    instances_number = 0;
    variables_number = 0;

    set_missing_values_number(0);

    set_default();
}


// void set(const size_t&, const size_t&) method

/// Sets a new number of missing values in the object.
/// @param new_instances_number Number of instances in the data set.
/// @param new_variables_number Number of variables in the data set.

void MissingValues::set(const size_t& new_instances_number, const size_t& new_variables_number)
{   
    instances_number = new_instances_number;
    variables_number = new_variables_number;

    set_missing_values_number(0);

    set_default();
}


// void set(const tinyxml2::XMLDocument&) method

/// Sets the members of this object from a XML document.
/// @param document Pointer to a TinyXML document containing the member data.

void MissingValues::set(const tinyxml2::XMLDocument& document)
{
    set();

    from_XML(document);
}


// void set_instances_number(const size_t&) method

/// Sets the number of data set instances in this object.

void MissingValues::set_instances_number(const size_t& new_instances_number)
{
    set_missing_values_number(0);

    instances_number = new_instances_number;
}


// void set_variables_number(const size_t&) method

/// Sets the number of data set variables in this object.

void MissingValues::set_variables_number(const size_t& new_variables_number)
{
    set_missing_values_number(0);

    variables_number = new_variables_number;
}


// void set_default(void) method

/// Sets the default values to this MissingValues object:
/// <ul>
/// <li>scrubbing_method: Unuse</li>
/// <li>display: true</li>
/// </ul>

void MissingValues::set_default(void)
{
    scrubbing_method = Unuse;

    display = true;
}


// void set_display(const bool&) method

/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void MissingValues::set_display(const bool& new_display)
{
    display = new_display;
}


// void set_items(const Vector<Item>&) method

/// Sets a new vector of missing value items.

void MissingValues::set_items(const Vector<Item>& new_items)
{
    items = new_items;
}


// void set_item(const size_t&, const size_t&, const size_t&) method

/// Sets the indices

void MissingValues::set_item(const size_t& index, const size_t& instance_index, const size_t& variable_index)
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t missing_values_number = get_missing_values_number();

    if(index >= missing_values_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void set_item(const size_t&, const size_t&, const size_t&) method.\n"
               << "Index (" << index << ") must be less than number of missing values (" << missing_values_number << ").\n";

        throw std::logic_error(buffer.str());
    }

    if(instance_index >= instances_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void set_item(const size_t&, const size_t&, const size_t&) method.\n"
               << "Index of instance (" << instance_index << ") must be less than number of instances (" << instances_number << ").\n";

        throw std::logic_error(buffer.str());
    }

    if(variable_index >= variables_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void set_item(const size_t&, const size_t&, const size_t&) method.\n"
               << "Index of variable (" << variable_index << ") must be less than number of variables (" << variables_number << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    items[index].instance_index = instance_index;
    items[index].variable_index = variable_index;
}


// void append(const size_t&, const size_t&) method

/// Appends a new item to the vector of missing values.
/// @param instance_index Instance with a missing value.
/// @param variable_index Variable with a missing value.

void MissingValues::append(const size_t& instance_index, const size_t& variable_index)
{
#ifdef __OPENNN_DEBUG__

    if(instance_index >= instances_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void append(const size_t&, const size_t&) method.\n"
               << "Index of instance (" << instance_index << ") must be less than number of instances (" << instances_number << ").\n";

        throw std::logic_error(buffer.str());
    }

    if(variable_index >= variables_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void append(const size_t&, const size_t&) method.\n"
               << "Index of variable (" << variable_index << ") must be less than number of instances (" << variables_number << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    Item item(instance_index, variable_index);

    items.push_back(item);
}


// void set_missing_values_number(const size_t&) method

/// Sets a new number of missing values in the data set.
/// @param new_missing_values_number Number of missing values.

void MissingValues::set_missing_values_number(const size_t& new_missing_values_number)
{
    items.set(new_missing_values_number);
}


// void set_scrubbing_method(const ScrubbingMethod&) method

/// Sets a new method for dealing with the missing values.
/// @param new_scrubbing_method Scrubbing method.

void MissingValues::set_scrubbing_method(const ScrubbingMethod& new_scrubbing_method)
{
    scrubbing_method = new_scrubbing_method;
}


// void set_scrubbing_method(const std::string&) method

/// Sets a new method for dealing with the missing values from a string.
/// @param new_scrubbing_method String with the name of the scrubbing method.

void MissingValues::set_scrubbing_method(const std::string& new_scrubbing_method)
{
    if(new_scrubbing_method == "Unuse")
    {
        scrubbing_method = Unuse;
    }
    else if(new_scrubbing_method == "Mean")
    {
        scrubbing_method = Mean;
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void new_scrubbing_method(const std::string&) method.\n"
               << "Unknown scrubbing method: " << new_scrubbing_method << ".\n";

        throw std::logic_error(buffer.str());
    }
}

// bool has_missing_values(void) const method

/// Returns true if there are missing values,
/// and false if the number of missing values is zero.

bool MissingValues::has_missing_values(void) const
{
    if(items.empty())
    {
        return(false);
    }
    else
    {
        return(true);
    }
}


// bool has_missing_values(const size_t&) const method

/// Returns true if a given instance has missing values,
/// and false otherwise.
/// @param instance_index Index of instance.

bool MissingValues::has_missing_values(const size_t& instance_index) const
{
    if(items.empty())
    {
        return(false);
    }

    const size_t missing_values_number = get_missing_values_number();

    for(size_t i = 0; i < missing_values_number; i++)
    {
        if(items[i].instance_index == instance_index)
        {
            return(true);
        }
    }

    return(false);
}


// bool has_missing_values(const size_t&, const Vector<size_t>&) const method

/// Returns true if a given instance has missing values for given variables,
/// and false otherwise.
/// @param instance_index Index of instance.
/// @param variables_indices Indices of variables.

bool MissingValues::has_missing_values(const size_t& instance_index, const Vector<size_t>& variables_indices) const
{
    const size_t missing_values_number = get_missing_values_number();

    const size_t variables_number = variables_indices.size();

    for(size_t i = 0; i < missing_values_number; i++)
    {
        if(items[i].instance_index == instance_index)
        {
            for(size_t j = 0; j < variables_number; j++)
            {
                if(items[i].variable_index == variables_indices[j])
                {
                    return(true);
                }
            }
        }
    }

    return(false);
}


// bool is_missing_value(const size_t&, const size_t&) const method

/// Returns true if the value with given instance and variable indices is missing.
/// @param instance_index Index of instance.
/// @param variable_index Index of variable.

bool MissingValues::is_missing_value(const size_t& instance_index, const size_t& variable_index) const
{
    if(items.empty())
    {
        return(false);
    }

    const size_t missing_values_number = get_missing_values_number();

    for(size_t i = 0; i < missing_values_number; i++)
    {
        if(items[i].instance_index == instance_index && items[i].variable_index == variable_index)
        {
            return(true);
        }
    }

    return(false);
}


// Vector<size_t> arrange_missing_instances(void) const method

/// Returns a vector with the indices of those instances with missing values.

Vector<size_t> MissingValues::arrange_missing_instances(void) const
{
    const size_t missing_values_number = get_missing_values_number();

    Vector<size_t> missing_instances;

    for(size_t i = 0; i < missing_values_number; i++)
    {
        if(!missing_instances.contains(items[i].instance_index))
        {
            missing_instances.push_back(items[i].instance_index);
        }
    }

    return(missing_instances);
}


// size_t count_missing_instances(void) const method

/// Returns the number of instances with missing values.

size_t MissingValues::count_missing_instances(void) const
{
    return(arrange_missing_instances().size());
}


// Vector<size_t> arrange_missing_variables(void) const method

/// Returns a vector with the indices of those variables with missing values.

Vector<size_t> MissingValues::arrange_missing_variables(void) const
{
    const size_t missing_values_number = get_missing_values_number();

    Vector<size_t> missing_variables;

    for(size_t i = 0; i < missing_values_number; i++)
    {
        if(!missing_variables.contains(items[i].variable_index))
        {
            missing_variables.push_back(items[i].variable_index);
        }
    }

    return(missing_variables);
}


// void convert_time_series(const size_t&) method

/// @todo Complete method.

void MissingValues::convert_time_series(const size_t& lags_number)
{
    if(lags_number == 0)
    {
        return;
    }
/*
    const size_t missing_values_number = get_missing_values_number();

    //    const size_t new_missing_values_number = missing_values_number*(1+lags_number);

    Vector<Item> new_items;

//    size_t instance_index;
//    size_t variable_index;

    Item new_item;

    for(size_t i = 0; i < missing_values_number; i++)
    {
        instance_index = items[i].instance_index;
        variable_index = items[i].variable_index;

        for(size_t j = 0; j < 1+lags_number; j++)
        {
            new_item.instance_index = items[i].instance_index+j;

            new_item.variable_index = items[i].variable_index*(1+j);

            new_items.push_back(new_item);
        }
    }

    set_items(new_items);
*/
}


// void convert_association(void) method

/// @todo Implement this method.

void MissingValues::convert_association(void)
{
    const size_t missing_values_number = get_missing_values_number();

    Vector<Item> association_items(missing_values_number);

}


// Vector< Vector<size_t> > arrange_missing_indices(void) const method

/// Returns a vector of vectors with the indices of the missing values for each variable.
/// The size of the vector is the number of variables.
/// The size of each subvector is the number of missing values for the corresponding variable.

Vector< Vector<size_t> > MissingValues::arrange_missing_indices(void) const
{
    Vector< Vector<size_t> > missing_indices(variables_number);

    const size_t missing_values_number = get_missing_values_number();

    size_t variable_index;
    size_t instance_index;

    for(size_t i = 0; i < missing_values_number; i++)
    {
        variable_index = items[i].variable_index;
        instance_index = items[i].instance_index;

        missing_indices[variable_index].push_back(instance_index);
    }

    return(missing_indices);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the MissingValues object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document. 
/// @todo

tinyxml2::XMLDocument* MissingValues::to_XML(void) const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    std::ostringstream buffer;

    // MissingValues

    tinyxml2::XMLElement* missing_values_element = document->NewElement("MissingValues");

    document->InsertFirstChild(missing_values_element);

    tinyxml2::XMLElement* element = NULL;
    tinyxml2::XMLText* text = NULL;

    const size_t missing_values_number = get_missing_values_number();

    // Instances number
    {
        element = document->NewElement("InstancesNumber");
        missing_values_element->LinkEndChild(element);

        buffer.str("");
        buffer << instances_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Variables number
    {
        element = document->NewElement("VariablesNumber");
        missing_values_element->LinkEndChild(element);

        buffer.str("");
        buffer << variables_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Scrubbing method
    {
        element = document->NewElement("ScrubbingMethod");
        missing_values_element->LinkEndChild(element);

        text = document->NewText(write_scrubbing_method().c_str());
        element->LinkEndChild(text);
    }

    // Missing values number
    {
        element = document->NewElement("MissingValuesNumber");
        missing_values_element->LinkEndChild(element);

        buffer.str("");
        buffer << missing_values_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }



    if(missing_values_number > 0)
    {
        tinyxml2::XMLElement* instances_indices_element = document->NewElement("InstancesIndices");
        missing_values_element->LinkEndChild(instances_indices_element);

        tinyxml2::XMLElement* variables_indices_element = document->NewElement("VariablesIndices");
        missing_values_element->LinkEndChild(variables_indices_element);

        std::string instances_indices_text;
        std::string variables_indices_text;

        for(size_t i = 0; i < missing_values_number; i++)
        {
            // Instance index

            const size_t instance_index = items[i].instance_index + 1;

            instances_indices_text.append(number_to_string(instance_index));

            if(i != missing_values_number - 1)
            {
                instances_indices_text.append(" ");
            }

            // Variable index

            const size_t variable_index = items[i].variable_index + 1;

            variables_indices_text.append(number_to_string(variable_index));

            if(i != missing_values_number - 1)
            {
                variables_indices_text.append(" ");
            }
        }

        // Insert instances indices

        text = document->NewText(instances_indices_text.c_str());
        instances_indices_element->LinkEndChild(text);

        // Insert variables indices

        text = document->NewText(variables_indices_text.c_str());
        variables_indices_element->LinkEndChild(text);
    }

    /*
    for(size_t i = 0; i < missing_values_number; i++)
    {
        element = document->NewElement("Item");
        element->SetAttribute("Index", (unsigned)i+1);
        missing_values_element->LinkEndChild(element);

        // Instance index

        tinyxml2::XMLElement* instance_index_element = document->NewElement("InstanceIndex");
        element->LinkEndChild(instance_index_element);

        buffer.str("");
        buffer << items[i].instance_index+1;

        text = document->NewText(buffer.str().c_str());
        instance_index_element->LinkEndChild(text);

        // Variable index

        tinyxml2::XMLElement* variable_index_element = document->NewElement("VariableIndex");
        element->LinkEndChild(variable_index_element);

        buffer.str("");
        buffer << items[i].variable_index+1;

        text = document->NewText(buffer.str().c_str());
        variable_index_element->LinkEndChild(text);
    }
*/

    // Display
//    {
//        element = document->NewElement("Display");
//        missing_values_element->LinkEndChild(element);

//        buffer.str("");
//        buffer << display;

//        text = document->NewText(buffer.str().c_str());
//        element->LinkEndChild(text);
//    }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the missing values object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void MissingValues::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    const size_t missing_values_number = get_missing_values_number();

    file_stream.OpenElement("MissingValues");

    // Instances number

    file_stream.OpenElement("InstancesNumber");

    buffer.str("");
    buffer << instances_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Variables number

    file_stream.OpenElement("VariablesNumber");

    buffer.str("");
    buffer << variables_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Scrubbing method

    file_stream.OpenElement("ScrubbingMethod");

    file_stream.PushText(write_scrubbing_method().c_str());

    file_stream.CloseElement();

    // Missing values number

    file_stream.OpenElement("MissingValuesNumber");

    buffer.str("");
    buffer << missing_values_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    if(missing_values_number <= 0)
    {
        file_stream.CloseElement();

        return;
    }

    std::string instances_indices_text;
    std::string variables_indices_text;

    for(size_t i = 0; i < missing_values_number; i++)
    {
        // Instance index

        const size_t instance_index = items[i].instance_index + 1;

        instances_indices_text.append(number_to_string(instance_index));

        if(i != missing_values_number - 1)
        {
            instances_indices_text.append(" ");
        }


        // Variable index

        const size_t variable_index = items[i].variable_index + 1;

        variables_indices_text.append(number_to_string(variable_index));

        if(i != missing_values_number - 1)
        {
            variables_indices_text.append(" ");
        }
    }

    // Instances indices

    file_stream.OpenElement("InstancesIndices");

    file_stream.PushText(instances_indices_text.c_str());

    file_stream.CloseElement();

    // Variables indices

    file_stream.OpenElement("VariablesIndices");

    file_stream.PushText(variables_indices_text.c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this MissingValues object.
/// @param document XML document containing the member data.
/// @todo Check that the method i

void MissingValues::from_XML(const tinyxml2::XMLDocument& document)
{
    std::ostringstream buffer;

    // Missing values element

    const tinyxml2::XMLElement* missing_values_element = document.FirstChildElement("MissingValues");

    if(!missing_values_element)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pointer to MissingValues element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Instances number

    const tinyxml2::XMLElement* instances_number_element = missing_values_element->FirstChildElement("InstancesNumber");

    if(!instances_number_element)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pointer to instances number is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    instances_number = atoi(instances_number_element->GetText());

    // Variables number

    const tinyxml2::XMLElement* variables_number_element = missing_values_element->FirstChildElement("VariablesNumber");

    if(!variables_number_element)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pointer to variables number is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    variables_number = atoi(variables_number_element->GetText());

    // Scrubbing method

    const tinyxml2::XMLElement* scrubbing_method_element = missing_values_element->FirstChildElement("ScrubbingMethod");

    if(!scrubbing_method_element)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pointer to scrubbing method element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const std::string scrubbing_method_string = scrubbing_method_element->GetText();

    set_scrubbing_method(scrubbing_method_string);

    // Missing values number

    const tinyxml2::XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pointer to missing values number is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const size_t new_missing_values_number = atoi(missing_values_number_element->GetText());

    set_missing_values_number(new_missing_values_number);

    if(new_missing_values_number <= 0)
    {
        return;
    }

    // Items

    const tinyxml2::XMLElement* instances_indices_element = missing_values_element->FirstChildElement("InstancesIndices");

    if(!instances_indices_element)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pointer to instances indices element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const std::string instances_indices_text(instances_indices_element->GetText());

    Vector<size_t> instances_indices;
    instances_indices.parse(instances_indices_text);

    if(instances_indices.size() != new_missing_values_number)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Instances indices size (" << instances_indices.size() << ") must be the same than missing values number (" << new_missing_values_number << ").\n";

        throw std::logic_error(buffer.str());
    }

    const tinyxml2::XMLElement* variables_indices_element = missing_values_element->FirstChildElement("VariablesIndices");

    if(!variables_indices_element)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Pointer to variables indices element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const std::string variables_indices_text(variables_indices_element->GetText());

    Vector<size_t> variables_indices;
    variables_indices.parse(variables_indices_text);

    if(variables_indices.size() != new_missing_values_number)
    {
        buffer << "OpenNN Exception: MissingValues class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Variables indices size (" << variables_indices.size() << ") must be the same than missing values number (" << new_missing_values_number << ").\n";

        throw std::logic_error(buffer.str());
    }

    for(size_t i = 0; i < new_missing_values_number; i++)
    {
        items[i].instance_index = instances_indices.at(i) - 1;

        items[i].variable_index = variables_indices.at(i) - 1;
    }

/*
    unsigned index = 0;

    const tinyxml2::XMLElement* start_element = missing_values_number_element;

    for(size_t i = 0; i < new_missing_values_number; i++)
    {
        const tinyxml2::XMLElement* item_element = start_element->NextSiblingElement("Item");
        start_element = item_element;

        if(!item_element)
        {
            buffer << "OpenNN Exception: MissingValues class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Item " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        item_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
        {
            buffer << "OpenNN Exception: MissingValues class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Index " << index << " is not correct.\n";

            throw std::logic_error(buffer.str());
        }

        // Instance index

        const tinyxml2::XMLElement* instance_index_element = item_element->FirstChildElement("InstanceIndex");

        if(!instance_index_element)
        {
            buffer << "OpenNN Exception: MissingValues class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Pointer to instance index element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        const size_t new_instance_index = atoi(instance_index_element->GetText())-1;

        items[i].instance_index = new_instance_index;

        // Variable index

        const tinyxml2::XMLElement* variable_index_element = item_element->FirstChildElement("VariableIndex");

        if(!instance_index_element)
        {
            buffer << "OpenNN Exception: MissingValues class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Pointer to variable index element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        const size_t new_variable_index = atoi(variable_index_element->GetText())-1;

        items[i].variable_index = new_variable_index;
    }
*/
}


// std::string to_string(void) const method

/// Returns a string representation of the current MissingValues object.

std::string MissingValues::to_string(void) const
{
    std::ostringstream buffer;

    const size_t missing_values_number = get_missing_values_number();

    buffer << "Missing values object\n"
           << "Instances number: " << instances_number << "\n"
           << "Variables number: " << variables_number << "\n"
           << "Missing values number: " << missing_values_number << "\n";

    for(size_t i = 0; i < missing_values_number; i++)
    {
        buffer << "Missing value " << i+1 << ":\n"
               << "Instance index: " << items[i].instance_index << "\n"
               << "Variable index: " << items[i].variable_index << "\n";
    }

    buffer << "Scrubbing method: " << write_scrubbing_method() << "\n";

    //buffer << "Display: " << display << "\n";

    return(buffer.str());
}


// void print(void) const method

/// Prints to the screen information about the missing values object.

void MissingValues::print(void) const
{
    std::cout << to_string();
}



}


// OpenNN: Open Neural Networks Library.
// Copyright (c) 2005-2016 Roberto Lopez.
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
