//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "scaling_layer.h"

namespace OpenNN
{

/// Default constructor.
/// It creates a scaling layer object with no scaling neurons.

ScalingLayer::ScalingLayer() : Layer()
{
    set();
}


/// Scaling neurons number constructor.
/// This constructor creates a scaling layer with a given size.
/// The members of this object are initialized with the default values.
/// @param new_neurons_number Number of scaling neurons in the layer.

ScalingLayer::ScalingLayer(const Index& new_neurons_number) : Layer()
{
    set(new_neurons_number);
}


ScalingLayer::ScalingLayer(const Tensor<Index, 1>& new_inputs_dimensions) : Layer()
{
    set(new_inputs_dimensions);
}


/// Descriptives constructor.
/// This constructor creates a scaling layer with given minimums, maximums, means and standard deviations.
/// The rest of members of this object are initialized with the default values.
/// @param new_descriptives Vector of vectors with the variables descriptives.

ScalingLayer::ScalingLayer(const Tensor<Descriptives, 1>& new_descriptives) : Layer()
{
    set(new_descriptives);
}


/// Destructor.

ScalingLayer::~ScalingLayer()
{
}


Tensor<Index, 1> ScalingLayer::get_outputs_dimensions() const
{
    return input_variables_dimensions;
}


Index ScalingLayer::get_inputs_number() const
{
    return descriptives.size();
}


Index ScalingLayer::get_neurons_number() const
{
    return descriptives.size();
}


/// Returns all the scaling layer descriptives.
/// The format is a vector of descriptives structures of size the number of scaling neurons.

Tensor<Descriptives, 1> ScalingLayer::get_descriptives() const
{
    return descriptives;
}


/// Returns the descriptives structure of a single scaling neuron.
/// @param index Neuron index.

Descriptives ScalingLayer::get_descriptives(const Index& index) const
{
    return descriptives(index);
}


/// Returns a single matrix with the descriptives of all scaling neurons.
/// The number of rows is the number of scaling neurons.
/// The number of columns is four(minimum, maximum, mean and standard deviation).

Tensor<type, 2> ScalingLayer::get_descriptives_matrix() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<type, 2> statistics_matrix(neurons_number, 4);

    for(Index i = 0; i < neurons_number; i++)
    {
        statistics_matrix(i,0) = descriptives(i).minimum;
        statistics_matrix(i,1) = descriptives(i).maximum;
        statistics_matrix(i,2) = descriptives(i).mean;
        statistics_matrix(i,3) = descriptives(i).standard_deviation;
    }

    return statistics_matrix;
}


/// Returns a single matrix with the minimums of all scaling neurons.

Tensor<type, 1> ScalingLayer::get_minimums() const
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

Tensor<type, 1> ScalingLayer::get_maximums() const
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

Tensor<type, 1> ScalingLayer::get_means() const
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

Tensor<type, 1> ScalingLayer::get_standard_deviations() const
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

const Tensor<ScalingLayer::ScalingMethod, 1> ScalingLayer::get_scaling_methods() const
{
    return scaling_methods;
}


/// Returns a vector of strings with the name of the method used for each scaling neuron.

Tensor<string, 1> ScalingLayer::write_scaling_methods() const
{
    const Index neurons_number = get_neurons_number();

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scaling_methods[i] == NoScaling)
        {
            scaling_methods_strings[i] = "NoScaling";
        }
        else if(scaling_methods[i] == MeanStandardDeviation)
        {
            scaling_methods_strings[i] = "MeanStandardDeviation";
        }
        else if(scaling_methods[i] == MinimumMaximum)
        {
            scaling_methods_strings[i] = "MinimumMaximum";
        }
        else if(scaling_methods[i] == StandardDeviation)
        {
            scaling_methods_strings[i] = "StandardDeviation";
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "Tensor<string, 1> write_scaling_methods() const method.\n"
                   << "Unknown " << i << " scaling method.\n";

            throw logic_error(buffer.str());
        }
    }

    return scaling_methods_strings;
}


/// Returns a vector of strings with the name of the methods used for scaling,
/// as paragaph text.

Tensor<string, 1> ScalingLayer::write_scaling_methods_text() const
{
    const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "Tensor<string, 1> write_scaling_methods() const method.\n"
               << "Neurons number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<string, 1> scaling_methods_strings(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scaling_methods[i] == NoScaling)
        {
            scaling_methods_strings[i] = "no scaling";
        }
        else if(scaling_methods[i] == MeanStandardDeviation)
        {
            scaling_methods_strings[i] = "mean and standard deviation";
        }
        else if(scaling_methods[i] == StandardDeviation)
        {
            scaling_methods_strings[i] = "standard deviation";
        }
        else if(scaling_methods[i] == MinimumMaximum)
        {
            scaling_methods_strings[i] = "minimum and maximum";
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "Tensor<string, 1> write_scaling_methods_text() const method.\n"
                   << "Unknown " << i << " scaling method.\n";

            throw logic_error(buffer.str());
        }
    }

    return scaling_methods_strings;
}

// const bool& get_display() const method

/// Returns true if messages from this class are to be displayed on the screen, or false if messages
/// from this class are not to be displayed on the screen.

const bool& ScalingLayer::get_display() const
{
    return display;
}


/// Sets the scaling layer to be empty.

void ScalingLayer::set()
{
    descriptives.resize(0);

    scaling_methods.resize(0);

    set_default();
}


/// Sets a new size in the scaling layer.
/// It also sets the members to their default values.

void ScalingLayer::set(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);

    scaling_methods.resize(new_inputs_number);

    scaling_methods.setConstant(MinimumMaximum);

    set_default();
}


void ScalingLayer::set(const Tensor<Index, 1>& new_inputs_dimensions)
{
    const Tensor<Index,0> dimension_product = new_inputs_dimensions.prod();

    descriptives.resize(dimension_product(0));

    scaling_methods.resize(dimension_product(0));
    scaling_methods.setConstant(MinimumMaximum);

    input_variables_dimensions.resize(new_inputs_dimensions.size());

    input_variables_dimensions = new_inputs_dimensions;

    set_default();
}


/// Sets the size of the scaling layer and the descriptives values.
/// @param new_descriptives Vector of vectors containing the minimums, maximums, means and standard deviations for the scaling layer.
/// The size of this vector must be 4.
/// The size of each subvector will be the size of the scaling layer.

void ScalingLayer::set(const Tensor<Descriptives, 1>& new_descriptives)
{
    descriptives = new_descriptives;

    scaling_methods.resize(new_descriptives.size());

    scaling_methods.setConstant(MinimumMaximum);

    set_default();
}


/// Sets the scaling layer members from a XML document.
/// @param new_scaling_layer_document Pointer to a TinyXML document containing the member data.

void ScalingLayer::set(const tinyxml2::XMLDocument& new_scaling_layer_document)
{
    set_default();

    from_XML(new_scaling_layer_document);
}


/// Sets the members of this object to be the members of another object of the same class.
/// @param new_scaling_layer Object to be copied.

void ScalingLayer::set(const ScalingLayer& new_scaling_layer)
{
    descriptives = new_scaling_layer.descriptives;

    scaling_methods = new_scaling_layer.scaling_methods;

    layer_type = new_scaling_layer.layer_type;

    display = new_scaling_layer.display;

    layer_type = Scaling;
}


//void ScalingLayer::set(const Tensor<bool, 1>& new_uses)
//{
//    /*
//        const Tensor<Index, 1> indices = new_uses.get_indices_equal_to(true);
//    */
////    descriptives = descriptives.get_subvector(indices);

////    scaling_methods = scaling_methods.get_subvector(indices);
//}


void ScalingLayer::set_inputs_number(const Index& new_inputs_number)
{
    descriptives.resize(new_inputs_number);

    scaling_methods.resize(new_inputs_number);

    scaling_methods.setConstant(MinimumMaximum);
}


void ScalingLayer::set_neurons_number(const Index& new_neurons_number)
{
    descriptives.resize(new_neurons_number);

    scaling_methods.resize(new_neurons_number);

    scaling_methods.setConstant(MinimumMaximum);
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

void ScalingLayer::set_default()
{
    layer_name = "scaling_layer";

    set_scaling_methods(MinimumMaximum);

    set_min_max_range(-1, 1);

    set_display(true);

    layer_type = Scaling;
}


/// Sets max and min scaling range for minmaxscaling.
/// @param min and max for scaling range.

void ScalingLayer::set_min_max_range(const type min, const type max)
{
    min_range = min;
    max_range = max;
}

/// Sets all the scaling layer descriptives from a vector descriptives structures.
/// The size of the vector must be equal to the number of scaling neurons in the layer.
/// @param new_descriptives Scaling layer descriptives.

void ScalingLayer::set_descriptives(const Tensor<Descriptives, 1>& new_descriptives)
{

#ifdef __OPENNN_DEBUG__

    const Index new_descriptives_size = new_descriptives.size();

    const Index neurons_number = get_neurons_number();

    if(new_descriptives_size != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_descriptives(const Tensor<Descriptives, 1>&) method.\n"
               << "Size of descriptives (" << new_descriptives_size << ") is not equal to number of scaling neurons (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    descriptives = new_descriptives;
}


void ScalingLayer::set_descriptives_eigen(const Tensor<type, 2>& descriptives_eigen)
{
    const Index neurons_number = get_neurons_number();

    Tensor<Descriptives, 1> descriptives(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        descriptives(i).set_minimum(descriptives_eigen(i, 0));
        descriptives(i).set_maximum(descriptives_eigen(i, 1));
        descriptives(i).set_mean(descriptives_eigen(i, 2));
        descriptives(i).set_standard_deviation(descriptives_eigen(i, 3));
    }

    set_descriptives(descriptives);
}


/// Sets the descriptives of a single scaling neuron.
/// @param i Index of neuron.
/// @param item_descriptives Descriptives structure for that neuron.

void ScalingLayer::set_item_descriptives(const Index& i, const Descriptives& item_descriptives)
{
    descriptives(i) = item_descriptives;
}


/// Sets the minimum value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_minimum Minimum value.

void ScalingLayer::set_minimum(const Index& i, const type& new_minimum)
{
    descriptives(i).set_minimum(new_minimum);
}


/// Sets the maximum value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_maximum Maximum value.

void ScalingLayer::set_maximum(const Index& i, const type& new_maximum)
{
    descriptives(i).set_maximum(new_maximum);
}


/// Sets the mean value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_mean Mean value.

void ScalingLayer::set_mean(const Index& i, const type& new_mean)
{
    descriptives(i).set_mean(new_mean);
}


/// Sets the standard deviation value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_standard_deviation Standard deviation value.

void ScalingLayer::set_standard_deviation(const Index& i, const type& new_standard_deviation)
{
    descriptives(i).set_standard_deviation(new_standard_deviation);
}


/// Sets the methods to be used for scaling each variable.
/// @param new_scaling_methods New scaling methods for the variables.

void ScalingLayer::set_scaling_methods(const Tensor<ScalingLayer::ScalingMethod, 1>& new_scaling_methods)
{
#ifdef __OPENNN_DEBUG__

    const Index neurons_number = get_neurons_number();

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_scaling_methods(const Tensor<ScalingMethod, 1>&) method.\n"
               << "Neurons number (" << neurons_number << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    scaling_methods = new_scaling_methods;
}


/// Sets the methods to be used for scaling each variable.
/// The argument is a vector string containing the name of the methods("NoScaling", "MeanStandardDeviation" or "MinimumMaximum").
/// @param new_scaling_methods_string New scaling methods for the variables.

void ScalingLayer::set_scaling_methods(const Tensor<string, 1>& new_scaling_methods_string)
{
    const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_scaling_methods(const Tensor<string, 1>&) method.\n"
               << "Neurons number (" << neurons_number << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<ScalingMethod, 1> new_scaling_methods(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string(i) == "NoScaling")
        {
            new_scaling_methods(i) = NoScaling;
        }
        else if(new_scaling_methods_string(i) == "MeanStandardDeviation")
        {
            new_scaling_methods(i) = MeanStandardDeviation;
        }
        else if(new_scaling_methods_string(i) == "MinimumMaximum")
        {
            new_scaling_methods(i) = MinimumMaximum;
        }
        else if(new_scaling_methods_string(i) == "StandardDeviation")
        {
            new_scaling_methods(i) = StandardDeviation;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void set_scaling_methods(const Tensor<string, 1>&) method.\n"
                   << "Unknown scaling method: " << new_scaling_methods_string[i] << ".\n";

            throw logic_error(buffer.str());
        }
    }

    set_scaling_methods(new_scaling_methods);
}


/// Sets all the methods to be used for scaling with the given method.
/// The argument is a string containing the name of the method("NoScaling", "MeanStandardDeviation" or "MinimumMaximum").
/// @param new_scaling_methods_string New scaling methods for the variables.

void ScalingLayer::set_scaling_methods(const string& new_scaling_methods_string)
{
    const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_scaling_methods(const Tensor<string, 1>&) method.\n"
               << "Neurons number (" << neurons_number << ")must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Tensor<ScalingMethod, 1> new_scaling_methods(neurons_number);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string == "NoScaling")
        {
            new_scaling_methods(i) = NoScaling;
        }
        else if(new_scaling_methods_string == "MeanStandardDeviation")
        {
            new_scaling_methods(i) = MeanStandardDeviation;
        }
        else if(new_scaling_methods_string == "MinimumMaximum")
        {
            new_scaling_methods(i) = MinimumMaximum;
        }
        else if(new_scaling_methods_string == "StandardDeviation")
        {
            new_scaling_methods(i) = StandardDeviation;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void set_scaling_methods(const Tensor<string, 1>&) method.\n"
                   << "Unknown scaling method: " << new_scaling_methods_string[i] << ".\n";

            throw logic_error(buffer.str());
        }
    }

    set_scaling_methods(new_scaling_methods);
}


/// Sets the method to be used for scaling the variables.
/// @param new_scaling_method New scaling method for the variables.

void ScalingLayer::set_scaling_methods(const ScalingLayer::ScalingMethod& new_scaling_method)
{
    const Index neurons_number = get_neurons_number();

    for(Index i = 0; i < neurons_number; i++)
    {
        scaling_methods(i) = new_scaling_method;
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void ScalingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


/// Returns true if the number of scaling neurons is zero, and false otherwise.

bool ScalingLayer::is_empty() const
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

void ScalingLayer::check_range(const Tensor<type, 1>& inputs) const
{
    const Index inputs_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    const Index size = inputs.size();

    if(size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void check_range(const Tensor<type, 1>&) const method.\n"
               << "Size of inputs must be equal to number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Check inputs

    if(display)
    {
        for(Index i = 0; i < inputs_number; i++)
        {
            if(inputs(i) < descriptives(i).minimum)
            {
                cout << "OpenNN Warning: ScalingLayer class.\n"
                     << "void check_range(const Tensor<type, 1>&) const method.\n"
                     << "Input value " << i << " is less than corresponding minimum.\n";
            }

            if(inputs(i) > descriptives(i).maximum)
            {
                cout << "OpenNN Warning: ScalingLayer class.\n"
                     << "void check_range(const Tensor<type, 1>&) const method.\n"
                     << "Input value " << i << " is greater than corresponding maximum.\n";
            }
        }
    }
}


/// Scales some values to produce some scaled values.
/// @param inputs Set of inputs to the scaling layer.

Tensor<type, 2> ScalingLayer::calculate_outputs(const Tensor<type, 2>& inputs)
{
    Tensor<type, 2> outputs;

    if(inputs.rank() == 2)
    {
        const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        const Index columns_number = inputs.dimension(1);

        if(columns_number != neurons_number)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
                   << "Size of inputs (" << columns_number << ") must be equal to number of scaling neurons (" << neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

#endif

        const Index points_number = inputs.dimension(0);

        outputs.resize(points_number, neurons_number);

        for(Index i = 0; i < points_number; i++)
        {
            for(Index j = 0; j < neurons_number; j++)
            {
                if(abs(descriptives(j).minimum - descriptives(j).maximum) < numeric_limits<type>::min())
                {
                    if(display)
                    {
                        cout << "OpenNN Warning: ScalingLayer class.\n"
                             << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
                             << "Standard deviation of variable " << i << " is zero.\n"
                             << "Those variables won't be scaled.\n";
                    }

                    outputs(j) = inputs(j);
                }
                else
                {
                    if(scaling_methods(j) == NoScaling)
                    {
                        outputs(i,j) = inputs(i,j);
                    }
                    else if(scaling_methods(j) == MinimumMaximum)
                    {
                        const type slope =
                                    (max_range-min_range)/(descriptives(j).maximum-descriptives(j).minimum);

                        const type intercept =
                                    (min_range*descriptives(j).maximum-max_range*descriptives(j).minimum)/(descriptives(j).maximum-descriptives(j).minimum);

                        outputs(i,j) = inputs(i,j)*slope + intercept;
                    }
                    else if(scaling_methods(j) == MeanStandardDeviation)
                    {
                        const type slope = static_cast<type>(1)/descriptives(j).standard_deviation;

                        const type intercept = -descriptives(j).mean/descriptives(j).standard_deviation;

                        outputs(i,j) = inputs(i,j)*slope + intercept;

                    }
                    else if(scaling_methods(j) == StandardDeviation)
                    {
                        outputs(i,j) = inputs(i,j)/descriptives(j).standard_deviation;
                    }
                    else
                    {
                        ostringstream buffer;

                        buffer << "OpenNN Exception: ScalingLayer class\n"
                               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
                               << "Unknown scaling method.\n";

                        throw logic_error(buffer.str());
                    }
                }
            }
        }
    }
    else if(inputs.rank() == 4)
    {
        /*

        const Index neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        const Index columns_number = inputs.dimension(1) * inputs.dimension(2) * inputs.dimension(3);

        if(columns_number != neurons_number)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
                   << "Size of inputs (" << columns_number << ") must be equal to number of scaling neurons (" << neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

#endif

        const Index points_number = inputs.dimension(0);

//        outputs.set(points_number, inputs.dimension(1), inputs.dimension(2), inputs.dimension(3));
        outputs.resize(points_number, inputs.dimension(1));

        for(Index i = 0; i < points_number; i++)
        {
            for(Index j = 0; j < neurons_number; j++)
            {
                Index channel_index = j%(inputs.dimension(1));
                Index row_index = (j/(inputs.dimension(1)))%(inputs.dimension(2));
                Index column_index = (j/(inputs.dimension(1) * inputs.dimension(2)))%(inputs.dimension(3));

                if(abs(descriptives(j).minimum - descriptives(j).maximum) < numeric_limits<type>::min())
                {
                    if(display)
                    {
                        cout << "OpenNN Warning: ScalingLayer class.\n"
                             << "Tensor<type, 2> calculate_mean_standard_deviation_outputs(const Tensor<type, 2>&) const method.\n"
                             << "Standard deviation of variable " << i << " is zero.\n"
                             << "Those variables won't be scaled.\n";
                    }

                    outputs(j) = inputs(j);
                }
                else
                {

                    if(scaling_methods(j) == NoScaling)
                    {
                        outputs(i, channel_index, row_index, column_index) = inputs(i, channel_index, row_index, column_index);
                    }
                    else if(scaling_methods(j) == MinimumMaximum)
                    {
                        outputs(i, channel_index, row_index, column_index) = static_cast<type>(2)*(inputs(i, channel_index, row_index, column_index) - descriptives(j).minimum)/(descriptives(j).maximum-descriptives(j).minimum) - static_cast<type>(1);
                    }
                    else if(scaling_methods(j) == MeanStandardDeviation)
                    {
                        outputs(i, channel_index, row_index, column_index) = (inputs(i, channel_index, row_index, column_index) - descriptives(j).mean)/descriptives(j).standard_deviation;
                    }
                    else if(scaling_methods(j) == StandardDeviation)
                    {
                        outputs(i, channel_index, row_index, column_index) = inputs(i, channel_index, row_index, column_index)/descriptives(j).standard_deviation;
                    }
                    else
                    {
                        ostringstream buffer;

                        buffer << "OpenNN Exception: ScalingLayer class\n"
                               << "Tensor<type, 2> calculate_outputs(const Tensor<type, 2>&) const method.\n"
                               << "Unknown scaling method.\n";

                        throw logic_error(buffer.str());
                    }

                }
            }
        }
        */
    }

    return outputs;
}


/// Returns a string with the expression of the scaling process when the none method is used.
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons.
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons.

string ScalingLayer::write_no_scaling_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
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

string ScalingLayer::write_minimum_maximum_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
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

string ScalingLayer::write_mean_standard_deviation_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
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

string ScalingLayer::write_standard_deviation_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
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

string ScalingLayer::write_expression(const Tensor<string, 1>& inputs_names, const Tensor<string, 1>& outputs_names) const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scaling_methods(i) == NoScaling)
        {
            buffer << "scaled_" << inputs_names(i) << " = " << inputs_names(i) << ";\n";
        }
        else if(scaling_methods(i) == MinimumMaximum)
        {
            buffer << "scaled_" << inputs_names(i) << " = " << inputs_names(i) << "*(" << max_range << "-" << min_range << ")/(" << descriptives(i).maximum << "-(" << descriptives(i).minimum << "))-" << descriptives(i).minimum << "*(" << max_range << "-" << min_range << ")/(" << descriptives(i).maximum << "-" << descriptives(i).minimum << ")+" << min_range << ";\n";
        }
        else if(scaling_methods(i) == MeanStandardDeviation)
        {
            buffer << "scaled_" << inputs_names(i) << " = (" << inputs_names(i) << "-(" << descriptives(i).mean << "))/" << descriptives(i).standard_deviation << ";\n";
        }
        else if(scaling_methods(i) == StandardDeviation)
        {
            buffer << "scaled_" << inputs_names(i) << " = " << inputs_names(i) << "/(" << descriptives(i).standard_deviation << ");\n";
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "string write_expression() const method.\n"
                   << "Unknown inputs scaling method.\n";

            throw logic_error(buffer.str());
        }
    }

    string expression = buffer.str();

    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;

}


/// \brief write_expression_c
/// \return

string ScalingLayer::write_expression_c() const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    buffer << "vector<float> " << layer_name << "(const vector<float>& inputs)\n{" << endl;

    buffer << "\tvector<float> outputs(" << neurons_number << ");\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scaling_methods(i) == NoScaling)
        {
            buffer << "\toutputs[" << i << "] = inputs[" << i << "];" << endl;
        }
        else if(scaling_methods(i) == MinimumMaximum)
        {
            const type slope = (max_range-min_range)/(descriptives(i).maximum-descriptives(i).minimum);

            const type intercept = -(descriptives(i).minimum*(max_range-min_range))/(descriptives(i).maximum - descriptives(i).minimum) + min_range;

            buffer << "\toutputs[" << i << "] = inputs[" << i << "]*"<<slope<<"+"<<intercept<<";\n";
        }
        else if(scaling_methods(i) == MeanStandardDeviation)
        {
            const type slope = static_cast<type>(2)/descriptives(i).standard_deviation;

            const type intercept = -static_cast<type>(2)*descriptives(i).mean/descriptives(i).standard_deviation;

            buffer << "\toutputs[" << i << "] = inputs[" << i << "]*"<<slope<<"+"<<intercept<<";\n";
        }
        else if(scaling_methods(i) == StandardDeviation)
        {
            buffer << "\toutputs[" << i << "] = inputs[" << i << "]/" << descriptives(i).standard_deviation << " ;" << endl;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "string write_expression() const method.\n"
                   << "Unknown inputs scaling method.\n";

            throw logic_error(buffer.str());
        }
    }

    buffer << "\n\treturn outputs;\n}" << endl;

    return buffer.str();
}


string ScalingLayer::write_expression_python() const
{
    const Index neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    buffer << "def " << layer_name << "(inputs):\n" << endl;

    buffer << "\toutputs = [None] * "<<neurons_number<<"\n" << endl;

    for(Index i = 0; i < neurons_number; i++)
    {
        if(scaling_methods(i) == NoScaling)
        {
            buffer << "\toutputs[" << i << "] = inputs[" << i << "]" << endl;
        }
        else if(scaling_methods(i) == MinimumMaximum)
        {
            const type slope = (max_range-min_range)/(descriptives(i).maximum-descriptives(i).minimum);

            const type intercept = -(descriptives(i).minimum*(max_range-min_range))/(descriptives(i).maximum - descriptives(i).minimum) + min_range;

            buffer << "\toutputs[" << i << "] = inputs[" << i << "]*"<<slope<<"+"<<intercept<<"\n";
        }
        else if(scaling_methods(i) == MeanStandardDeviation)
        {
            const type slope = static_cast<type>(2)/descriptives(i).standard_deviation;

            const type intercept = -static_cast<type>(2)*descriptives(i).mean/descriptives(i).standard_deviation;

            buffer << "\toutputs[" << i << "] = inputs[" << i << "]*"<<slope<<"+"<<intercept<<"\n";
        }
        else if(scaling_methods(i) == StandardDeviation)
        {
            buffer << "\toutputs[" << i << "] = inputs[" << i << "]/" << descriptives(i).standard_deviation << " " << endl;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "string write_expression() const method.\n"
                   << "Unknown inputs scaling method.\n";

            throw logic_error(buffer.str());
        }
    }

    buffer << "\n\treturn outputs;\n" << endl;

    return buffer.str();
}


/// Serializes the scaling layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ScalingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    const Index neurons_number = get_neurons_number();

    // Scaling layer

    file_stream.OpenElement("ScalingLayer");

    // Scaling neurons number

    file_stream.OpenElement("ScalingNeuronsNumber");

    buffer.str("");
    buffer << neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    const Tensor<string, 1> scaling_methods_string = write_scaling_methods();

    // Scaling neurons

    for(Index i = 0; i < neurons_number; i++)
    {
        // Scaling neuron

        file_stream.OpenElement("ScalingNeuron");

        file_stream.PushAttribute("Index",int(i+1));

        // Minimum

        file_stream.OpenElement("Minimum");

        buffer.str("");
        buffer << descriptives(i).minimum;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Maximum

        file_stream.OpenElement("Maximum");

        buffer.str("");
        buffer << descriptives(i).maximum;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Mean

        file_stream.OpenElement("Mean");

        buffer.str("");
        buffer << descriptives(i).mean;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Standard deviation

        file_stream.OpenElement("StandardDeviation");

        buffer.str("");
        buffer << descriptives(i).standard_deviation;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Scaling Method

        file_stream.OpenElement("ScalingMethod");

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

void ScalingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* scaling_layer_element = document.FirstChildElement("ScalingLayer");

    if(!scaling_layer_element)
    {
        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Scaling layer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Scaling neurons number

    const tinyxml2::XMLElement* neurons_number_element = scaling_layer_element->FirstChildElement("ScalingNeuronsNumber");

    if(!neurons_number_element)
    {
        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Scaling neurons number element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const Index neurons_number = static_cast<Index>(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // Index does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;

    for(Index i = 0; i < neurons_number; i++)
    {
        const tinyxml2::XMLElement* scaling_neuron_element = start_element->NextSiblingElement("ScalingNeuron");
        start_element = scaling_neuron_element;

        if(!scaling_neuron_element)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Scaling neuron " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        scaling_neuron_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Index " << index << " is not correct.\n";

            throw logic_error(buffer.str());
        }

        // Minimum

        const tinyxml2::XMLElement* minimum_element = scaling_neuron_element->FirstChildElement("Minimum");

        if(!minimum_element)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Minimum element " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(minimum_element->GetText())
        {
            descriptives[i].minimum = static_cast<type>(atof(minimum_element->GetText()));
        }

        // Maximum

        const tinyxml2::XMLElement* maximum_element = scaling_neuron_element->FirstChildElement("Maximum");

        if(!maximum_element)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Maximum element " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(maximum_element->GetText())
        {
            descriptives[i].maximum = static_cast<type>(atof(maximum_element->GetText()));
        }

        // Mean

        const tinyxml2::XMLElement* mean_element = scaling_neuron_element->FirstChildElement("Mean");

        if(!mean_element)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Mean element " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(mean_element->GetText())
        {
            descriptives[i].mean = static_cast<type>(atof(mean_element->GetText()));
        }

        // Standard deviation

        const tinyxml2::XMLElement* standard_deviation_element = scaling_neuron_element->FirstChildElement("StandardDeviation");

        if(!standard_deviation_element)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Standard deviation element " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        if(standard_deviation_element->GetText())
        {
            descriptives[i].standard_deviation = static_cast<type>(atof(standard_deviation_element->GetText()));
        }

        // Scaling method

        const tinyxml2::XMLElement* scaling_method_element = scaling_neuron_element->FirstChildElement("ScalingMethod");

        if(!scaling_method_element)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
                   << "Scaling method element " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        string new_method = scaling_method_element->GetText();

        if(new_method == "MinimumMaximum" || new_method == "Minimum - Maximum")
        {
            scaling_methods[i] = MinimumMaximum;
        }
        else if(new_method == "MeanStandardDeviation" || new_method == "Mean - Standard deviation")
        {
            scaling_methods[i] = MeanStandardDeviation;
        }
        else if(new_method == "StandardDeviation")
        {
            scaling_methods[i] = StandardDeviation;
        }
        else
        {
            scaling_methods[i] = NoScaling;
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
            catch(const logic_error& e)
            {
                cerr << e.what() << endl;
            }
        }
    }
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
