//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   2 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer_2d.h"
#include "opennn_strings.h"

namespace opennn
{

/// Default constructor.
/// It creates a scaling layer object with no scaling neurons.

ScalingLayer2D::ScalingLayer2D() : Layer()
{
    set();
}


/// Scaling neurons number constructor.
/// This constructor creates a scaling layer with a given size.
/// It initializes the members of this object with the default values.
/// @param new_neurons_number Number of scaling neurons in the layer.

ScalingLayer2D::ScalingLayer2D(const Index& new_neurons_number) : Layer()
{
    set(new_neurons_number);
}


ScalingLayer2D::ScalingLayer2D(const Tensor<Index, 1>& new_inputs_dimensions) : Layer()
{
    set(new_inputs_dimensions);
}


/// Descriptives constructor.
/// This constructor creates a scaling layer with given minimums, maximums, means, and standard deviations.
/// It also initializes the rest of the members of this object with the default values.
/// @param new_descriptives Vector of vectors with the variables descriptives.

ScalingLayer2D::ScalingLayer2D(const Tensor<Descriptives, 1>& new_descriptives) : Layer()
{
    set(new_descriptives);
}


Tensor<Index, 1> ScalingLayer2D::get_outputs_dimensions() const
{
    return inputs_dimensions;
}


Index ScalingLayer2D::get_inputs_number() const
{
    return descriptives.size();
}


Tensor<Index, 1> ScalingLayer2D::get_inputs_dimensions() const
{
    return inputs_dimensions;
}


Index ScalingLayer2D::get_neurons_number() const
{
    return descriptives.size();
}


/// Returns all the scaling layer descriptives.
/// The format is a vector of descriptives structures of size the number of scaling neurons.

Tensor<Descriptives, 1> ScalingLayer2D::get_descriptives() const
{
    return descriptives;
}


/// Returns the descriptives structure of a single scaling neuron.
/// @param index Neuron index.

Descriptives ScalingLayer2D::get_descriptives(const Index& index) const
{
    return descriptives(index);
}


/// Returns a single matrix with the minimums of all scaling neurons.

Tensor<type, 1> ScalingLayer2D::get_minimums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> minimums(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        minimums[i] = descriptives[i].minimum;
    }

    return minimums;
}


/// Returns a single matrix with the maximums of all scaling neurons.

Tensor<type, 1> ScalingLayer2D::get_maximums() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> maximums(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        maximums[i] = descriptives[i].maximum;
    }

    return maximums;
}


/// Returns a single matrix with the means of all scaling neurons.

Tensor<type, 1> ScalingLayer2D::get_means() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> means(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        means[i] = descriptives[i].mean;
    }

    return means;
}


/// Returns a single matrix with the standard deviations of all scaling neurons.

Tensor<type, 1> ScalingLayer2D::get_standard_deviations() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 1> standard_deviations(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        standard_deviations[i] = descriptives[i].standard_deviation;
    }

    return standard_deviations;
}


/// Returns the methods used for scaling.

Tensor<Scaler, 1> ScalingLayer2D::get_scaling_methods() const
{
    return scalers;
}


/// Returns a vector of strings with the name of the method used for each scaling neuron.

Tensor<string, 1> ScalingLayer2D::write_scalers() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers[i] == Scaler::NoScaling)
        {
            scaling_methods_strings[i] = "NoScaling";
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
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "Tensor<string, 1> write_scalers() const method.\n"
                   << "Unknown " << i << " scaling method.\n";

            throw invalid_argument(buffer.str());
        }
    }

    return scaling_methods_strings;
}


/// Returns a vector of strings with the name of the methods used for scaling,
/// as paragaph text.

Tensor<string, 1> ScalingLayer2D::write_scalers_text() const
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer2D class.\n"
               << "Tensor<string, 1> write_scalers() const method.\n"
               << "Neurons number must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers[i] == Scaler::NoScaling)
        {
            scaling_methods_strings[i] = "no scaling";
        }
        else if(scalers[i] == Scaler::MeanStandardDeviation)
        {
            scaling_methods_strings[i] = "mean and standard deviation";
        }
        else if(scalers[i] == Scaler::StandardDeviation)
        {
            scaling_methods_strings[i] = "standard deviation";
        }
        else if(scalers[i] == Scaler::MinimumMaximum)
        {
            scaling_methods_strings[i] = "minimum and maximum";
        }
        else if(scalers[i] == Scaler::Logarithm)
        {
            scaling_methods_strings[i] = "Logarithm";
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "Tensor<string, 1> write_scalers_text() const method.\n"
                   << "Unknown " << i << " scaling method.\n";

            throw invalid_argument(buffer.str());
        }
    }

    return scaling_methods_strings;
}

// const bool& get_display() const method

/// Returns true if messages from this class are displayed on the screen, or false if messages
/// from this class are not displayed on the screen.

const bool& ScalingLayer2D::get_display() const
{
    return display;
}


/// Sets the scaling layer to be empty.

void ScalingLayer2D::set()
{
    descriptives.resize(0);

    scalers.resize(0);

    set_default();
}


/// Sets a new size in the scaling layer.
/// It also sets the members to their default values.

void ScalingLayer2D::set(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);

    scalers.resize(new_inputs_number);

    scalers.setConstant(Scaler::MeanStandardDeviation);

    set_default();
}


void ScalingLayer2D::set(const Tensor<Index, 1>& new_inputs_dimensions)
{
    const Tensor<Index,0> dimension_product = new_inputs_dimensions.prod();

    descriptives.resize(dimension_product(0));

    scalers.resize(dimension_product(0));
    scalers.setConstant(Scaler::MeanStandardDeviation);

    inputs_dimensions.resize(new_inputs_dimensions.size());

    inputs_dimensions = new_inputs_dimensions;

    set_default();
}


/// Sets the size of the scaling layer and the descriptives values.
/// @param new_descriptives Vector of vectors containing the minimums, maximums, means, and standard deviations for the scaling layer.
/// The size of this vector must be 4.
/// The size of each subvector will be the size of the scaling layer.

void ScalingLayer2D::set(const Tensor<Descriptives, 1>& new_descriptives)
{
    descriptives = new_descriptives;

    scalers.resize(new_descriptives.size());

    scalers.setConstant(Scaler::MeanStandardDeviation);

    set_neurons_number(new_descriptives.size());

    set_default();
}


void ScalingLayer2D::set(const Tensor<Descriptives, 1>& new_descriptives, const Tensor<Scaler, 1>& new_scalers)
{
    descriptives = new_descriptives;

    scalers = new_scalers;
}


/// Sets the scaling layer members from an XML document.
/// @param new_scaling_layer_document Pointer to a TinyXML document containing the member data.

void ScalingLayer2D::set(const tinyxml2::XMLDocument& new_scaling_layer_document)
{
    set_default();

    from_XML(new_scaling_layer_document);
}


void ScalingLayer2D::set_inputs_number(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);

    scalers.resize(new_inputs_number);

    scalers.setConstant(Scaler::MeanStandardDeviation);
}


void ScalingLayer2D::set_neurons_number(const Index& new_neurons_number)
{
    descriptives.resize(new_neurons_number);

    scalers.resize(new_neurons_number);

    scalers.setConstant(Scaler::MeanStandardDeviation);
}


/// Sets the members to their default value:
/// <ul>
/// <li> Minimus: -1 for all unscaling neurons.
/// <li> Maximums: 1 for all unscaling neurons.
/// <li> Means: 0 for all unscaling neurons.
/// <li> Standard deviations 1 for all unscaling neurons.
/// <li> Scaling method: Minimum and maximum.
/// <li> Display: True.
/// </ul>

void ScalingLayer2D::set_default()
{
    layer_name = "scaling_layer";

    set_scalers(Scaler::MeanStandardDeviation);

    set_min_max_range(type(-1), type(1));

    set_display(true);

    layer_type = Type::Scaling2D;
}


/// Sets max and min scaling range for minmaxscaling.
/// @param min and max for scaling range.

void ScalingLayer2D::set_min_max_range(const type& min, const type& max)
{
    min_range = min;
    max_range = max;
}


/// Sets all the scaling layer descriptives from a vector descriptives structures.
/// The size of the vector must be equal to the number of scaling neurons in the layer.
/// @param new_descriptives Scaling layer descriptives.

void ScalingLayer2D::set_descriptives(const Tensor<Descriptives, 1>& new_descriptives)
{

#ifdef OPENNN_DEBUG

    const Index new_descriptives_size = new_descriptives.size();

    const Index neurons_number = get_neurons_number();

    if(new_descriptives_size != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer2D class.\n"
               << "void set_descriptives(const Tensor<Descriptives, 1>&) method.\n"
               << "Size of descriptives (" << new_descriptives_size << ") is not equal to number of scaling neurons (" << neurons_number << ").\n";

        throw invalid_argument(buffer.str());
    }

#endif

    descriptives = new_descriptives;
}


/// Sets the descriptives of a single scaling neuron.
/// @param i Index of neuron.
/// @param item_descriptives Descriptives structure for that neuron.

void ScalingLayer2D::set_item_descriptives(const Index& i, const Descriptives& item_descriptives)
{
    descriptives(i) = item_descriptives;
}


/// Sets the minimum value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_minimum Minimum value.

void ScalingLayer2D::set_minimum(const Index& i, const type& new_minimum)
{
    descriptives(i).set_minimum(new_minimum);
}


/// Sets the maximum value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_maximum Maximum value.

void ScalingLayer2D::set_maximum(const Index& i, const type& new_maximum)
{
    descriptives(i).set_maximum(new_maximum);
}


/// Sets the mean value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_mean Mean value.

void ScalingLayer2D::set_mean(const Index& i, const type& new_mean)
{
    descriptives(i).set_mean(new_mean);
}


/// Sets the standard deviation value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_standard_deviation Standard deviation value.

void ScalingLayer2D::set_standard_deviation(const Index& i, const type& new_standard_deviation)
{
    descriptives(i).set_standard_deviation(new_standard_deviation);
}


/// Sets the methods to be used for scaling each variable.
/// @param new_scaling_methods New scaling methods for the variables.

void ScalingLayer2D::set_scalers(const Tensor<Scaler, 1>& new_scaling_methods)
{
#ifdef OPENNN_DEBUG

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer2D class.\n"
               << "void set_scalers(const Tensor<Scaler, 1>&) method.\n"
               << "Neurons number (" << neurons_number << ") must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    scalers = new_scaling_methods;
}


/// Sets the methods to be used for scaling each variable.
/// The argument is a vector string containing the name of the methods("NoScaling", "MeanStandardDeviation" or "MinimumMaximum").
/// @param new_scaling_methods_string New scaling methods for the variables.

void ScalingLayer2D::set_scalers(const Tensor<string, 1>& new_scaling_methods_string)
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer2D class.\n"
               << "void set_scalers(const Tensor<string, 1>&) method.\n"
               << "Neurons number (" << neurons_number << ") must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<Scaler, 1> new_scaling_methods(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string(i) == "NoScaling")
        {
            new_scaling_methods(i) = Scaler::NoScaling;
        }
        else if(new_scaling_methods_string(i) == "MinimumMaximum")
        {
            new_scaling_methods(i) = Scaler::MinimumMaximum;
        }
        else if(new_scaling_methods_string(i) == "MeanStandardDeviation")
        {
            new_scaling_methods(i) = Scaler::MeanStandardDeviation;
        }
        else if(new_scaling_methods_string(i) == "StandardDeviation")
        {
            new_scaling_methods(i) = Scaler::StandardDeviation;
        }
        else if(new_scaling_methods_string(i) == "Logarithm")
        {
            new_scaling_methods(i) = Scaler::Logarithm;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "void set_scalers(const Tensor<string, 1>&) method.\n"
                   << "Unknown scaling method: " << new_scaling_methods_string[i] << ".\n";

            throw invalid_argument(buffer.str());
        }
    }

    set_scalers(new_scaling_methods);
}


void ScalingLayer2D::set_scaler(const Index& variable_index, const Scaler& new_scaler)
{
    scalers(variable_index) = new_scaler;
}


/// Sets all the methods to be used for scaling with the given method.
/// The argument is a string containing the name of the method("NoScaling", "MeanStandardDeviation" or "MinimumMaximum").
/// @param new_scaling_methods_string New scaling methods for the variables.

void ScalingLayer2D::set_scalers(const string& new_scaling_methods_string)
{
    const Index neurons_number = get_neurons_number();

#ifdef OPENNN_DEBUG

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer2D class.\n"
               << "void set_scalers(const Tensor<string, 1>&) method.\n"
               << "Neurons number (" << neurons_number << ")must be greater than 0.\n";

        throw invalid_argument(buffer.str());
    }

#endif

    Tensor<Scaler, 1> new_scaling_methods(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string == "NoScaling")
        {
            new_scaling_methods(i) = Scaler::NoScaling;
        }
        else if(new_scaling_methods_string == "MeanStandardDeviation")
        {
            new_scaling_methods(i) = Scaler::MeanStandardDeviation;
        }
        else if(new_scaling_methods_string == "MinimumMaximum")
        {
            new_scaling_methods(i) = Scaler::MinimumMaximum;
        }
        else if(new_scaling_methods_string == "StandardDeviation")
        {
            new_scaling_methods(i) = Scaler::StandardDeviation;
        }
        else if(new_scaling_methods_string == "Logarithm")
        {
            new_scaling_methods(i) = Scaler::Logarithm;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "void set_scalers(const Tensor<string, 1>&) method.\n"
                   << "Unknown scaling method: " << new_scaling_methods_string[i] << ".\n";

            throw invalid_argument(buffer.str());
        }
    }

    set_scalers(new_scaling_methods);
}


/// Sets the method to be used for scaling the variables.
/// @param new_scaling_method New scaling method for the variables.

void ScalingLayer2D::set_scalers(const Scaler& new_scaling_method)
{
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        scalers(i) = new_scaling_method;
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are displayed on the screen;
/// if it is set to false messages from this class are not displayed on the screen.
/// @param new_display Display value.

void ScalingLayer2D::set_display(const bool& new_display)
{
    display = new_display;
}


/// Returns true if the number of scaling neurons is zero, and false otherwise.

bool ScalingLayer2D::is_empty() const
{
    const Index inputs_number = get_neurons_number();

    if(inputs_number == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// This method chechs whether the inputs to the scaling layer have the right size.
/// If not, it displays an error message and exits the program.
/// It also checks whether the input values are inside the range defined by the minimums and maximum values, and
/// displays a v fg warning message if they are outside.
/// @param inputs Set of inputs to the scaling layer.

void ScalingLayer2D::check_range(const Tensor<type, 1>& inputs) const
{
    const Index inputs_number = get_neurons_number();

    // Check inputs

    if(display)
    {
        for(Index i = 0; i < inputs_number; i++)
        {
            if(inputs(i) < descriptives(i).minimum)
            {
                cout << "OpenNN Warning: ScalingLayer2D class.\n"
                     << "void check_range(const Tensor<type, 1>&) const method.\n"
                     << "Input value " << i << " is less than corresponding minimum.\n";
            }

            if(inputs(i) > descriptives(i).maximum)
            {
                cout << "OpenNN Warning: ScalingLayer2D class.\n"
                     << "void check_range(const Tensor<type, 1>&) const method.\n"
                     << "Input value " << i << " is greater than corresponding maximum.\n";
            }
        }
    }
}


void ScalingLayer2D::forward_propagate(const pair<type*, dimensions>& inputs,
                                     LayerForwardPropagation* forward_propagation,
                                     const bool& is_training)
{
    ScalingLayer2DForwardPropagation* scaling_layer_forward_propagation
            = static_cast<ScalingLayer2DForwardPropagation*>(forward_propagation);

    const Index neurons_number = scaling_layer_forward_propagation->layer_pointer->get_neurons_number();

    const Tensor<Index, 0> input_size = inputs_dimensions.prod();

    const TensorMap<Tensor<type, 2>> inputs_map(inputs.first, inputs.second[0][0], inputs.second[0][1]);

    Tensor<type, 2>& outputs = scaling_layer_forward_propagation->outputs;

    for(Index i = 0; i < neurons_number; i++)
    {
        const Scaler scaler = scalers(i);

        /// @todo to inputs map

        Tensor<type, 1> column = inputs_map.chip(i, 1);

        if(abs(descriptives(i).standard_deviation) < type(NUMERIC_LIMITS_MIN))
        {
            if(display)
            {
                cout << "display: " << display << endl;
                cout << "OpenNN Warning: ScalingLayer2D class.\n"
                     << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
                     << "Standard deviation of variable " << i << " is zero.\n"
                     << "Those variables won't be scaled.\n";
            }
        }
        else
        {
            if(scaler == Scaler::NoScaling)
            {

            }
            else if(scaler == Scaler::MinimumMaximum)
            {
                const type slope =
                        (max_range-min_range)/(descriptives(i).maximum-descriptives(i).minimum);

                const type intercept =
                        (min_range*descriptives(i).maximum-max_range*descriptives(i).minimum)/(descriptives(i).maximum-descriptives(i).minimum);

                column = intercept + slope*inputs_map.chip(i, 1);
            }
            else if(scaler == Scaler::MeanStandardDeviation)
            {
                const type slope = type(1)/descriptives(i).standard_deviation;

                const type intercept = -descriptives(i).mean/descriptives(i).standard_deviation;

                column = intercept + slope*inputs_map.chip(i, 1);
            }
            else if(scaler == Scaler::StandardDeviation)
            {
                column = type(1/descriptives(i).standard_deviation) * inputs_map.chip(i, 1);
                //column/type(descriptives(i).standard_deviation);
            }
            else if(scaler == Scaler::Logarithm)
            {
                column = inputs_map.chip(i,1).log();
            }
            else
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: ScalingLayer2D class\n"
                       << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
                       << "Unknown scaling method.\n";

                throw invalid_argument(buffer.str());
            }

        }

        outputs.chip(i, 1) = column;
    }
}


/// Returns a string with the expression of the scaling process when the none method is used.
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons.
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons.

string ScalingLayer2D::write_no_scaling_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names(i) << " = " << inputs_names(i) << ";\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the scaling process with the minimum and maximum method.
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons.
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons.

string ScalingLayer2D::write_minimum_maximum_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names(i) << " = 2*(" << inputs_names(i) << "-(" << descriptives(i).minimum << "))/(" << descriptives(i).maximum << "-(" << descriptives(i).minimum << "))-1;\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the scaling process with the mean and standard deviation method.
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons.
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons.

string ScalingLayer2D::write_mean_standard_deviation_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names(i) << " = (" << inputs_names(i) << "-(" << descriptives(i).mean << "))/" << descriptives(i).standard_deviation << ";\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the scaling process with the standard deviation method.
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons.
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons.

string ScalingLayer2D::write_standard_deviation_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names(i) << " = " << inputs_names(i) << "/(" << descriptives(i).standard_deviation << ");\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the inputs scaling process.

string ScalingLayer2D::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>&) const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scalers(i) == Scaler::NoScaling)
        {
            buffer << "scaled_" << inputs_names(i) << " = " << inputs_names(i) << ";\n";
        }
        else if(scalers(i) == Scaler::MinimumMaximum)
        {
            buffer << "scaled_" << inputs_names(i) << " = " << inputs_names(i) << "*(" << max_range << "-" << min_range << ")/(" << descriptives(i).maximum << "-(" << descriptives(i).minimum << "))-" << descriptives(i).minimum << "*(" << max_range << "-" << min_range << ")/(" << descriptives(i).maximum << "-" << descriptives(i).minimum << ")+" << min_range << ";\n";
        }
        else if(scalers(i) == Scaler::MeanStandardDeviation)
        {
            buffer << "scaled_" << inputs_names(i) << " = (" << inputs_names(i) << "-" << descriptives(i).mean << ")/" << descriptives(i).standard_deviation << ";\n";
        }
        else if(scalers(i) == Scaler::StandardDeviation)
        {
            buffer << "scaled_" << inputs_names(i) << " = " << inputs_names(i) << "/(" << descriptives(i).standard_deviation << ");\n";
        }
        else if(scalers(i) == Scaler::Logarithm)
        {
            buffer << "scaled_" << inputs_names(i) << " = log(" << inputs_names(i) << ");\n";
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "string write_expression() const method.\n"
                   << "Unknown inputs scaling method.\n";

            throw invalid_argument(buffer.str());
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;

}


void ScalingLayer2D::print() const
{
    cout << "Scaling layer" << endl;

    const Index inputs_number = get_inputs_number();

    const Tensor<string, 1> scalers_text = write_scalers_text();

    for(Index i = 0; i < inputs_number; i++)
    {
        cout << "Neuron " << i << endl;

        cout << "Scaler " << scalers_text(i) << endl;

        descriptives(i).print();
    }
}



/// Serializes the scaling layer object into an XML document of the TinyXML library without keeping the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ScalingLayer2D::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    // Scaling layer

    file_stream.OpenElement("ScalingLayer2D");

    // Scaling neurons number

    file_stream.OpenElement("ScalingNeuronsNumber");

    buffer.str("");
    buffer << neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Scaling neurons

    const Tensor<string, 1> scaling_methods_string = write_scalers();

    for(Index i = 0; i < neurons_number; i++)
    {
        // Scaling neuron

        file_stream.OpenElement("ScalingNeuron");

        file_stream.PushAttribute("Index", int(i+1));

        //Descriptives

        file_stream.OpenElement("Descriptives");

        buffer.str(""); buffer << descriptives(i).minimum;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << descriptives(i).maximum;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << descriptives(i).mean;
        file_stream.PushText(buffer.str().c_str());
        file_stream.PushText("\\");

        buffer.str(""); buffer << descriptives(i).standard_deviation;
        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Scaler

        file_stream.OpenElement("Scaler");

        buffer.str("");
        buffer << scaling_methods_string(i);

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Scaling neuron (end tag)

        file_stream.CloseElement();
    }

    // Scaling layer (end tag)

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this scaling layer object.
/// @param document XML document containing the member data.

void ScalingLayer2D::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* scaling_layer_element = document.FirstChildElement("ScalingLayer2D");

    if(!scaling_layer_element)
    {
        buffer << "OpenNN Exception: ScalingLayer2D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Scaling layer element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    // Scaling neurons number

    const tinyxml2::XMLElement* neurons_number_element = scaling_layer_element->FirstChildElement("ScalingNeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: ScalingLayer2D class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Scaling neurons number element is nullptr.\n";

        throw invalid_argument(buffer.str());
    }

    const Index neurons_number = Index(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // Index does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;

    for(Index i = 0; i < neurons_number; i++)
    {
        const tinyxml2::XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
        start_element = scaling_neuron_element;

        if(!scaling_neuron_element)
        {
            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Scaling neuron " << i+1 << " is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        scaling_neuron_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
        {
            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Index " << index << " is not correct.\n";

            throw invalid_argument(buffer.str());
        }

        // Descriptives

        const tinyxml2::XMLElement* descriptives_element = scaling_neuron_element->FirstChildElement("Descriptives");

        if(!descriptives_element)
        {
            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Descriptives element " << i+1 << " is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        if(descriptives_element->GetText())
        {
            const char* new_descriptives_element = descriptives_element->GetText();
            Tensor<string,1> splitted_descriptives = get_tokens(new_descriptives_element, '\\');
            descriptives[i].minimum = type(stof(splitted_descriptives[0]));
            descriptives[i].maximum = type(stof(splitted_descriptives[1]));
            descriptives[i].mean = type(stof(splitted_descriptives[2]));
            descriptives[i].standard_deviation = type(stof(splitted_descriptives[3]));
        }

        // Scaling method

        const tinyxml2::XMLElement* scaling_method_element = scaling_neuron_element->FirstChildElement("Scaler");

        if(!scaling_method_element)
        {
            buffer << "OpenNN Exception: ScalingLayer2D class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Scaling method element " << i+1 << " is nullptr.\n";

            throw invalid_argument(buffer.str());
        }

        const string new_method = scaling_method_element->GetText();

        if(new_method == "NoScaling" || new_method == "No Scaling")
        {
            scalers[i] = Scaler::NoScaling;
        }
        else if(new_method == "MinimumMaximum" || new_method == "Minimum - Maximum")
        {
            scalers[i] = Scaler::MinimumMaximum;
        }
        else if(new_method == "MeanStandardDeviation" || new_method == "Mean - Standard deviation")
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
        else
        {
            scalers[i] = Scaler::NoScaling;
        }
    }

    // Display
    {
        const tinyxml2::XMLElement* display_element = scaling_layer_element->FirstChildElement("Display");

        if(display_element)
        {
            string new_display_string = display_element->GetText();

            try
            {
                set_display(new_display_string != "0");
            }
            catch(const invalid_argument& e)
            {
                cerr << e.what() << endl;
            }
        }
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
