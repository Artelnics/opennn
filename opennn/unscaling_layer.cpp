//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R    C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "unscaling_layer.h"
#include "strings_utilities.h"

namespace opennn
{

/// Default constructor.

UnscalingLayer::UnscalingLayer() : Layer()
{
    set();
}


/// Outputs number constructor.

UnscalingLayer::UnscalingLayer(const Index& new_neurons_number) : Layer()
{
    set(new_neurons_number);
}


/// Outputs descriptives constructor.

UnscalingLayer::UnscalingLayer(const Tensor<Descriptives, 1>& new_descriptives) : Layer()
{
    set(new_descriptives);
}


/// Returns the number of inputs

Index UnscalingLayer::get_inputs_number() const
{
    return descriptives.size();
}


/// Returns the number of unscaling neurons in this layer.

Index UnscalingLayer::get_neurons_number() const
{
    return descriptives.size();
}


dimensions UnscalingLayer::get_output_dimensions() const
{
    Index neurons_number = get_neurons_number();

    return { neurons_number };
}


/// Returns all the available descriptives of the inputs and output variables.
/// The format is a vector of pointers to vectors of size ten:
/// <ul>
/// <li> Mean of variables.
/// <li> Standard deviation of variables.
/// <li> Minimum of variables.
/// <li> Maximum of variables.
/// </ul>

Tensor<Descriptives, 1> UnscalingLayer::get_descriptives() const
{
    return descriptives;
}

/// Returns a vector with the minimum values of all unscaling neurons.
/// The size is the number of neurons in the layer.

Tensor<type, 1> UnscalingLayer::get_minimums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> minimums(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        minimums[i] = descriptives[i].minimum;
    }

    return minimums;
}


/// Returns a vector with the maximum values of all unscaling neurons.
/// The size is the number of neurons in the layer.

Tensor<type, 1> UnscalingLayer::get_maximums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> maximums(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        maximums[i] = descriptives[i].maximum;
    }

    return maximums;
}


/// Returns the method used for unscaling
/// (no unscaling, minimum and maximum or mean and standard deviation).

Tensor<Scaler, 1> UnscalingLayer::get_unscaling_method() const
{
    return scalers;
}


/// Returns a string with the expression of the inputs scaling process.

string UnscalingLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers(i) == Scaler::None)
        {
            buffer << outputs_names(i) << " = " << inputs_names(i) << ";\n";
        }
        else if(scalers(i) == Scaler::MinimumMaximum)
        {
            if(abs(descriptives(i).minimum - descriptives(i).maximum) < type(NUMERIC_LIMITS_MIN))
            {
                buffer << outputs_names[i] << "=" << descriptives(i).minimum <<";\n";
            }
            else
            {
                const type slope = (descriptives(i).maximum-descriptives(i).minimum)/(max_range-min_range);

                const type intercept = descriptives(i).minimum - min_range*(descriptives(i).maximum-descriptives(i).minimum)/(max_range-min_range);

                buffer << outputs_names[i] << "=" << inputs_names[i] << "*" << slope << "+" << intercept<<";\n";
            }
        }
        else if(scalers(i) == Scaler::MeanStandardDeviation)
        {
            const type standard_deviation = descriptives(i).standard_deviation;

            const type mean = descriptives(i).mean;

            buffer << outputs_names[i] << "=" << inputs_names[i] << "*" << standard_deviation<<"+"<<mean<<";\n";
        }
        else if(scalers(i) == Scaler::StandardDeviation)
        {
            const type standard_deviation = descriptives(i).standard_deviation;

            buffer << outputs_names[i] <<  "=" <<  inputs_names(i) << "*" << standard_deviation<<";\n";
        }
        else if(scalers(i) == Scaler::Logarithm)
        {
            buffer << outputs_names[i] << "=" << "exp(" << inputs_names[i] << ");\n";
        }
        else
        {
            throw runtime_error("Unknown inputs scaling method.\n");
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}


/// Returns a vector of strings with the name of the method used for each unscaling neuron.

Tensor<string, 1> UnscalingLayer::write_unscaling_methods() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers[i] == Scaler::None)
        {
            scaling_methods_strings[i] = "None";
        }
        else if(scalers[i] == Scaler::MinimumMaximum)
        {
            scaling_methods_strings[i] = "MinimumMaximum";
        }
        else if(scalers[i] == Scaler::MeanStandardDeviation)
        {
            scaling_methods_strings[i] = "MeanStandardDeviation";
        }
        else if(scalers[i] == Scaler::StandardDeviation)
        {
            scaling_methods_strings[i] = "StandardDeviation";
        }
        else if(scalers[i] == Scaler::Logarithm)
        {
            scaling_methods_strings[i] = "Logarithm";
        }
        else
        {
            throw runtime_error("Unknown unscaling method.\n");
        }
    }

    return scaling_methods_strings;
}


/// Returns a string with the name of the method used for unscaling,
/// as paragraph text.

Tensor<string, 1> UnscalingLayer::write_unscaling_method_text() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers[i] == Scaler::None)
        {
            scaling_methods_strings[i] = "no unscaling";
        }
        else if(scalers[i] == Scaler::MinimumMaximum)
        {
            scaling_methods_strings[i] = "minimum and maximum";
        }
        else if(scalers[i] == Scaler::MeanStandardDeviation)
        {
            scaling_methods_strings[i] = "mean and standard deviation";
        }
        else if(scalers[i] == Scaler::StandardDeviation)
        {
            scaling_methods_strings[i] = "standard deviation";
        }
        else if(scalers[i] == Scaler::Logarithm)
        {
            scaling_methods_strings[i] = "logarithm";
        }
        else
        {
            throw runtime_error("Unknown unscaling method.\n");
        }
    }

    return scaling_methods_strings;
}


/// Returns true if messages from this class are displayed on the screen, or false if messages
/// from this class are not displayed on the screen.

const bool& UnscalingLayer::get_display() const
{
    return display;
}


/// Sets the unscaling layer to be empty.

void UnscalingLayer::set()
{
    descriptives.resize(0);

    scalers.resize(0);

    set_default();
}


void UnscalingLayer::set_inputs_number(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);
}


void UnscalingLayer::set_neurons_number(const Index& new_neurons_number)
{
    descriptives.resize(new_neurons_number);
}


/// Sets a new size in the unscaling layer.
/// It also sets the members to their default values.

void UnscalingLayer::set(const Index& new_neurons_number)
{
    descriptives.resize(new_neurons_number);

    scalers.resize(new_neurons_number);

    scalers.setConstant(Scaler::MinimumMaximum);

    set_default();
}


/// Sets the size of the unscaling layer and the descriptives values.
/// @param new_descriptives Vector of vectors containing the minimums, maximums, means, and standard deviations for the unscaling layer.
/// The size of this vector must be 4.
/// The size of each subvector will be the size of the unscaling layer.

void UnscalingLayer::set(const Tensor<Descriptives, 1>& new_descriptives)
{
    descriptives = new_descriptives;

    scalers.resize(new_descriptives.size());

    scalers.setConstant(Scaler::MinimumMaximum);

    set_default();
}


void UnscalingLayer::set(const Tensor<Descriptives, 1>& new_descriptives, const Tensor<Scaler, 1>& new_scalers)
{
    descriptives = new_descriptives;

    scalers = new_scalers;
}


/// Sets the unscaling layer members from an XML document.
/// @param new_unscaling_layer_document Pointer to a TinyXML document containing the member data.

void UnscalingLayer::set(const tinyxml2::XMLDocument& new_unscaling_layer_document)
{
    set_default();

    from_XML(new_unscaling_layer_document);
}


/// Sets the members of this object to be the members of another object of the same class.

void UnscalingLayer::set(const UnscalingLayer& new_unscaling_layer)
{
    descriptives = new_unscaling_layer.descriptives;

    scalers = new_unscaling_layer.scalers;

    display = new_unscaling_layer.display;
}


/// This member sets the default values for the unscaling layer:
/// <ul>
/// <li> Minimus: -1 for all unscaling neurons.
/// <li> Maximums: 1 for al unscaling neurons.
/// <li> Means: 0 for all unscaling neurons.
/// <li> Standard deviations 1 for all unscaling neurons.
/// <li> Unscaling method: Minimum and maximum.
/// <li> Display: True.
/// </ul>

void UnscalingLayer::set_default()
{
    layer_name = "unscaling_layer";

    set_scalers(Scaler::MinimumMaximum);

    set_min_max_range(type(-1), type(1));

    set_display(true);

    layer_type = Type::Unscaling;
}

/// Sets max and min scaling range for minmaxscaling.
/// @param min and max for scaling range.

void UnscalingLayer::set_min_max_range(const type min, const type max)
{
    min_range = min;
    max_range = max;
}

/// Sets the descriptives for all the neurons in the unscaling layer from a vector.
/// The size of this vector must be equal to the number of unscaling neurons.
/// @param new_descriptives Unscaling neurons descriptives.

void UnscalingLayer::set_descriptives(const Tensor<Descriptives, 1>& new_descriptives)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    const Index new_descriptives_size = new_descriptives.size();

    if(new_descriptives_size != neurons_number)
        throw runtime_error("Size of descriptives (" + to_string(new_descriptives_size) + ") "
                            "must be equal to number of unscaling neurons (" + to_string(neurons_number) + ").\n");

#endif

    // Set all descriptives

    descriptives = new_descriptives;
}


/// Sets the minimum, maximum, mean and standard deviation values of a single unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param item_descriptives  Descriptives values for that neuron.

void UnscalingLayer::set_item_descriptives(const Index& i, const Descriptives& item_descriptives)
{
    descriptives[i] = item_descriptives;
}


/// Sets the minimum value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_minimum Minimum value.

void UnscalingLayer::set_minimum(const Index& i, const type& new_minimum)
{
    descriptives[i].set_minimum(new_minimum);
}


/// Sets the maximum value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_maximum Maximum value.

void UnscalingLayer::set_maximum(const Index& i, const type& new_maximum)
{
    descriptives[i].set_maximum(new_maximum);
}


/// Sets the mean value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_mean Mean value.

void UnscalingLayer::set_mean(const Index& i, const type& new_mean)
{
    descriptives[i].set_mean(new_mean);
}


/// Sets the standard deviation value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_standard_deviation Standard deviation value.

void UnscalingLayer::set_standard_deviation(const Index& i, const type& new_standard_deviation)
{
    descriptives[i].set_standard_deviation(new_standard_deviation);
}


/// Sets the method to be used for unscaling the outputs from the neural network
/// @param new_unscaling_method New unscaling method for the output variables.

void UnscalingLayer::set_scalers(const Tensor<Scaler,1>& new_unscaling_method)
{
    scalers = new_unscaling_method;
}


/// Sets the method to be used for unscaling the outputs from the neural network
/// The argument is a string containing the name of the method("None", "MeanStandardDeviation", "MinimumMaximum" or "Logarithm").
/// @param new_unscaling_method New unscaling method for the output variables.

void UnscalingLayer::set_scalers(const string& new_scaling_methods_string)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 0)
        throw runtime_error("Neurons number (" + to_string(neurons_number) + ") must be greater than 0.\n");

#endif

    if(new_scaling_methods_string == "None")
    {
        set_scalers(Scaler::None);
    }
    else if(new_scaling_methods_string == "MinimumMaximum")
    {
        set_scalers(Scaler::MinimumMaximum);
    }
    else if(new_scaling_methods_string == "MeanStandardDeviation")
    {
        set_scalers(Scaler::MeanStandardDeviation);
    }
    else if(new_scaling_methods_string == "StandardDeviation")
    {
        set_scalers(Scaler::StandardDeviation);
    }
    else if(new_scaling_methods_string == "Logarithm")
    {
        set_scalers(Scaler::Logarithm);
    }
    else
    {
        throw runtime_error("set_scalers(const string& new_scaling_methods_string) method.\n");
    }
}


/// Sets the methods to be used for unscaling each variable.
/// The argument is a vector string containing the name of the methods("None", "MeanStandardDeviation" or "Logarithm").
/// @param new_unscaling_methods_string New unscaling methods for the variables.

void UnscalingLayer::set_scalers(const Tensor<string, 1>& new_unscaling_methods_string)
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    if(neurons_number == 0)
        throw runtime_error("Neurons number (" + to_string(neurons_number) + ") must be greater than 0.\n");

#endif

    Tensor<Scaler, 1> new_unscaling_methods(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(new_unscaling_methods_string(i) == "None")
        {
            new_unscaling_methods(i) = Scaler::None;
        }
        else if(new_unscaling_methods_string(i) == "MeanStandardDeviation")
        {
            new_unscaling_methods(i) = Scaler::MeanStandardDeviation;
        }
        else if(new_unscaling_methods_string(i) == "StandardDeviation")
        {
            new_unscaling_methods(i) = Scaler::StandardDeviation;
        }
        else if(new_unscaling_methods_string(i) == "MinimumMaximum")
        {
            new_unscaling_methods(i) = Scaler::MinimumMaximum;
        }
        else if(new_unscaling_methods_string(i) == "Logarithm")
        {
            new_unscaling_methods(i) = Scaler::Logarithm;
        }
        else
        {
            throw runtime_error("Unknown scaling method: " + new_unscaling_methods_string(i) + ".\n");
        }
    }

    set_scalers(new_unscaling_methods);
}


/// Sets the method to be used for unscaling the variables.
/// @param new_unscaling_method New unscaling method for the variables.

void UnscalingLayer::set_scalers(const Scaler& new_unscaling_method)
{
    const Index neurons_number = get_neurons_number();
    for(Index i = 0; i < neurons_number; i++)
    {
        scalers(i) = new_unscaling_method;
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void UnscalingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Checks whether the outptus from the unscaling layer are inside the range defined by the minimums and maximum values.
/// It displays a warning message if they are outside.
/// @param outputs Set of outptus from the unscaling layer.

void UnscalingLayer::check_range(const Tensor<type, 1>& outputs) const
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    const Index size = outputs.size();

    if(size != neurons_number)
        throw runtime_error("Size of outputs must be equal to number of unscaling neurons.\n");

#endif

    // Check outputs

    if(display)
    {
        for(Index i = 0; i < neurons_number; i++)
        {
            if(outputs[i] < descriptives[i].minimum)
            {
                cout << "OpenNN Warning: UnscalingLayer class.\n"
                     << "void check_range(const Tensor<type, 1>&) const method.\n"
                     << "Output variable " << i << " is less than outputs.\n";
            }

            if(outputs[i] > descriptives[i].maximum)
            {
                cout << "OpenNN Warning: UnscalingLayer class.\n"
                     << "void check_range(const Tensor<type, 1>&) const method.\n"
                     << "Output variable " << i << " is greater than maximum.\n";
            }
        }
    }
}


/// Returns true if the number of unscaling neurons is zero, and false otherwise.

bool UnscalingLayer::is_empty() const
{
    const Index neurons_number = get_neurons_number();

    if(neurons_number == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


void UnscalingLayer::forward_propagate(const Tensor<pair<type*, dimensions>, 1>& inputs_pair,
                                       LayerForwardPropagation* forward_propagation,
                                       const bool& is_training)
{
    const Index samples_number = inputs_pair(0).second[0];
    const Index neurons_number = get_neurons_number();

    UnscalingLayerForwardPropagation* unscaling_layer_forward_propagation
            = static_cast<UnscalingLayerForwardPropagation*>(forward_propagation);

    const TensorMap<Tensor<type,2>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1]);

    Tensor<type,2>& outputs = unscaling_layer_forward_propagation->outputs;

    Scaler scaler;

    for(Index i = 0; i < neurons_number; i++)
    {
        scaler = scalers(i);

        const TensorMap<Tensor<type, 1>> input_column(inputs.data() + i * samples_number, samples_number);

        TensorMap<Tensor<type, 1>> output_column(outputs.data() + i * samples_number, samples_number);

        if(abs(descriptives(i).standard_deviation) < type(NUMERIC_LIMITS_MIN))
        {
            if(display)
            {
                cout << "OpenNN Warning: ScalingLayer2D class.\n"
                     << "void forward_propagate\n"
                     << "Standard deviation of variable " << i << " is zero.\n"
                     << "Those variables won't be scaled.\n";
            }
        }
        else
        {
            if(scaler == Scaler::None)
            {
            }
            else if(scaler == Scaler::MinimumMaximum)
            {
                const type slope = (descriptives(i).maximum-descriptives(i).minimum)/(max_range-min_range);

                const type intercept = -(min_range*descriptives(i).maximum-max_range*descriptives(i).minimum)/(max_range-min_range);

                output_column.device(*thread_pool_device) = intercept + slope * input_column;
            }
            else if(scaler == Scaler::MeanStandardDeviation)
            {
                const type slope = descriptives(i).standard_deviation;

                const type intercept = descriptives(i).mean;

                output_column.device(*thread_pool_device) = intercept + slope*input_column;
            }
            else if(scaler == Scaler::StandardDeviation)
            {
                const type standard_deviation = descriptives(i).standard_deviation;

                output_column.device(*thread_pool_device) = standard_deviation*input_column;
            }
            else if(scaler == Scaler::Logarithm)
            {
                output_column.device(*thread_pool_device) = input_column.exp();
            }
            else
            {
                throw runtime_error("Unknown scaling method.\n");
            }
        }
    }
}


/// Serializes the unscaling layer object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void UnscalingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    // Unscaling layer

    file_stream.OpenElement("UnscalingLayer");

    // Unscaling neurons number

    file_stream.OpenElement("UnscalingNeuronsNumber");

    buffer.str("");
    buffer << neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Descriptives

    const Tensor<string, 1> scalers = write_unscaling_methods();

    for(Index i = 0; i < neurons_number; i++)
    {
        file_stream.OpenElement("Descriptives");

        file_stream.PushAttribute("Index", int(i+1));

        // Minimum

        file_stream.OpenElement("Minimum");

        buffer.str("");
        buffer << descriptives[i].minimum;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Maximum

        file_stream.OpenElement("Maximum");

        buffer.str("");
        buffer << descriptives[i].maximum;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Mean

        file_stream.OpenElement("Mean");

        buffer.str("");
        buffer << descriptives[i].mean;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Standard deviation

        file_stream.OpenElement("StandardDeviation");

        buffer.str("");
        buffer << descriptives[i].standard_deviation;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Unscaling method

        file_stream.OpenElement("Scaler");

        buffer.str("");
        buffer << scalers(i);

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Unscaling neuron (end tag)

        file_stream.CloseElement();
    }

    // Unscaling layer (end tag)

    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this unscaling layer object.
/// @param document XML document containing the member data.

void UnscalingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("UnscalingLayer");

    if(!root_element)
        throw runtime_error("Unscaling layer element is nullptr.\n");

    // Unscaling neurons number

    const tinyxml2::XMLElement* neurons_number_element = root_element->FirstChildElement("UnscalingNeuronsNumber");

    if(!neurons_number_element)
        throw runtime_error("Unscaling neurons number element is nullptr.\n");

    const Index neurons_number = Index(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // Index does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;


    for(Index i = 0; i < neurons_number; i++)
    {
        const tinyxml2::XMLElement* descriptives_element = start_element->NextSiblingElement("Descriptives");
        start_element = descriptives_element;

        if(!descriptives_element)
            throw runtime_error("Descriptives of unscaling neuron " + to_string(i+1) + " is nullptr.\n");

        descriptives_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
            throw runtime_error("Index " + to_string(index) + " is not correct.\n");

        // Minimum

        const tinyxml2::XMLElement* minimum_element = descriptives_element->FirstChildElement("Minimum");

        if(!minimum_element)
            throw runtime_error("Minimum element " + to_string(i+1) + " is nullptr.\n");

        if(minimum_element->GetText())
        {
            descriptives(i).minimum = type(atof(minimum_element->GetText()));
        }

        // Maximum

        const tinyxml2::XMLElement* maximum_element = descriptives_element->FirstChildElement("Maximum");

        if(!maximum_element)
            throw runtime_error("Maximum element " + to_string(i+1) + " is nullptr.\n");

        if(maximum_element->GetText())
        {
            descriptives(i).maximum = type(atof(maximum_element->GetText()));
        }

        // Mean

        const tinyxml2::XMLElement* mean_element = descriptives_element->FirstChildElement("Mean");

        if(!mean_element)
            throw runtime_error("Mean element " + to_string(i+1) + " is nullptr.\n");

        if(mean_element->GetText())
        {
            descriptives(i).mean = type(atof(mean_element->GetText()));
        }

        // Standard deviation

        const tinyxml2::XMLElement* standard_deviation_element = descriptives_element->FirstChildElement("StandardDeviation");

        if(!standard_deviation_element)
            throw runtime_error("Standard deviation element " + to_string(i+1) + " is nullptr.\n");

        if(standard_deviation_element->GetText())
        {
            descriptives(i).standard_deviation = type(atof(standard_deviation_element->GetText()));
        }

        // Unscaling method

        const tinyxml2::XMLElement* unscaling_method_element = descriptives_element->FirstChildElement("Scaler");

        if(!unscaling_method_element)
            throw runtime_error("Unscaling method element " + to_string(i+1) + " is nullptr.\n");

        const string new_method = unscaling_method_element->GetText();

        if(new_method == "None")
        {
            scalers[i] = Scaler::None;
        }
        else if(new_method == "MinimumMaximum")
        {
            scalers[i] = Scaler::MinimumMaximum;
        }
        else if(new_method == "MeanStandardDeviation")
        {
            scalers[i] = Scaler::MeanStandardDeviation;
        }
        else if(new_method == "StandardDeviation")
        {
            scalers[i] = Scaler::StandardDeviation;
        }
        else if(new_method == "Logarithm")
        {
            scalers[i] = Scaler::Logarithm;
        }
    }

    // Display

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

    if(element)
    {
        string new_display_string = element->GetText();

        try
        {
            set_display(new_display_string != "0");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }
}


pair<type*, dimensions> UnscalingLayerForwardPropagation::get_outputs_pair() const
{
    const Index neurons_number = layer->get_neurons_number();

    return pair<type*, dimensions>(outputs_data, { batch_samples_number, neurons_number });
}


void UnscalingLayerForwardPropagation::set(const Index& new_batch_samples_number, Layer* new_layer)
{
    layer = new_layer;

    const Index neurons_number = static_cast<UnscalingLayer*>(layer)->get_neurons_number();

    batch_samples_number = new_batch_samples_number;

    outputs.resize(batch_samples_number, neurons_number);

    outputs_data = outputs.data();
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
