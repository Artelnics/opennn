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

ScalingLayer::ScalingLayer(const size_t& new_neurons_number) : Layer()
{
    set(new_neurons_number);
}


ScalingLayer::ScalingLayer(const Vector<size_t>& new_inputs_dimensions) : Layer()
{
    set(new_inputs_dimensions);
}


/// Descriptives constructor.
/// This constructor creates a scaling layer with given minimums, maximums, means and standard deviations.
/// The rest of members of this object are initialized with the default values.
/// @param new_descriptives Vector of vectors with the variables descriptives.

ScalingLayer::ScalingLayer(const Vector<Descriptives>& new_descriptives) : Layer()
{
    set(new_descriptives);
}


/// Copy constructor. 

ScalingLayer::ScalingLayer(const ScalingLayer& new_scaling_layer) : Layer()
{
    set(new_scaling_layer);
}


/// Destructor.

ScalingLayer::~ScalingLayer()
{
}


Vector<size_t> ScalingLayer::get_input_variables_dimensions() const
{
    return inputs_dimensions;
}


Vector<size_t> ScalingLayer::get_outputs_dimensions() const
{
    return inputs_dimensions;
}


size_t ScalingLayer::get_inputs_number() const
{
    return descriptives.size();
}


size_t ScalingLayer::get_neurons_number() const
{
    return descriptives.size();
}


/// Returns all the scaling layer descriptives.
/// The format is a vector of descriptives structures of size the number of scaling neurons.

Vector<Descriptives> ScalingLayer::get_descriptives() const
{
    return descriptives;
}


/// Returns the descriptives structure of a single scaling neuron.
/// @param index Neuron index.

Descriptives ScalingLayer::get_descriptives(const size_t& index) const
{
    return(descriptives[index]);
}


/// Returns a single matrix with the descriptives of all scaling neurons.
/// The number of rows is the number of scaling neurons.
/// The number of columns is four(minimum, maximum, mean and standard deviation).

Matrix<double> ScalingLayer::get_descriptives_matrix() const
{
    const size_t neurons_number = get_neurons_number();

    Matrix<double> statistics_matrix(neurons_number, 4);

    for(size_t i = 0; i < neurons_number; i++)
    {
        statistics_matrix.set_row(i, descriptives[i].to_vector());
    }

    return(statistics_matrix);
}


/// Returns a single matrix with the minimums of all scaling neurons.

Vector<double> ScalingLayer::get_minimums() const
{
    const size_t neurons_number = get_neurons_number();

    Vector<double> minimums(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
    {
        minimums[i] = descriptives[i].minimum;
    }

    return(minimums);
}


/// Returns a single matrix with the maximums of all scaling neurons.

Vector<double> ScalingLayer::get_maximums() const
{
    const size_t neurons_number = get_neurons_number();

    Vector<double> maximums(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
    {
        maximums[i] = descriptives[i].maximum;
    }

    return(maximums);
}


/// Returns a single matrix with the means of all scaling neurons.

Vector<double> ScalingLayer::get_means() const
{
    const size_t neurons_number = get_neurons_number();

    Vector<double> means(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
    {
        means[i] = descriptives[i].mean;
    }

    return means;
}


/// Returns a single matrix with the standard deviations of all scaling neurons.

Vector<double> ScalingLayer::get_standard_deviations() const
{
    const size_t neurons_number = get_neurons_number();

    Vector<double> standard_deviations(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
    {
        standard_deviations[i] = descriptives[i].standard_deviation;
    }

    return(standard_deviations);
}


/// Returns the methods used for scaling.

const Vector<ScalingLayer::ScalingMethod> ScalingLayer::get_scaling_methods() const
{
    return(scaling_methods);
}


/// Returns a vector of strings with the name of the method used for each scaling neuron.

Vector<string> ScalingLayer::write_scaling_methods() const
{
    const size_t neurons_number = get_neurons_number();

    Vector<string> scaling_methods_strings(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
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
                   << "Vector<string> write_scaling_methods() const method.\n"
                   << "Unknown " << i << " scaling method.\n";

            throw logic_error(buffer.str());
        }
    }

    return scaling_methods_strings;
}


/// Returns a vector of strings with the name of the methods used for scaling,
/// as paragaph text.

Vector<string> ScalingLayer::write_scaling_methods_text() const
{
    const size_t neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "Vector<string> write_scaling_methods() const method.\n"
               << "Neurons number must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<string> scaling_methods_strings(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
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
                   << "Vector<string> write_scaling_methods_text() const method.\n"
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
    descriptives.set();

    set_default();
}


/// Sets a new size in the scaling layer. 
/// It also sets the members to their default values. 

void ScalingLayer::set(const size_t& new_inputs_number)
{
    descriptives.set(new_inputs_number);

    scaling_methods.set(new_inputs_number, MinimumMaximum);

    set_default();
}


void ScalingLayer::set(const Vector<size_t>& new_inputs_dimensions)
{
    descriptives.set(new_inputs_dimensions.calculate_product());

    scaling_methods.set(new_inputs_dimensions.calculate_product(), MinimumMaximum);

    inputs_dimensions.set(new_inputs_dimensions);

    set_default();
}


/// Sets the size of the scaling layer and the descriptives values.
/// @param new_descriptives Vector of vectors containing the minimums, maximums, means and standard deviations for the scaling layer.
/// The size of this vector must be 4. 
/// The size of each subvector will be the size of the scaling layer. 

void ScalingLayer::set(const Vector<Descriptives>& new_descriptives)
{
    descriptives = new_descriptives;

    scaling_methods.set(new_descriptives.size(), MinimumMaximum);

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


void ScalingLayer::set(const Vector<bool>& new_uses)
{
    const Vector<size_t> indices = new_uses.get_indices_equal_to(true);

//    descriptives = descriptives.get_subvector(indices);

//    scaling_methods = scaling_methods.get_subvector(indices);
}


void ScalingLayer::set_inputs_number(const size_t& new_inputs_number)
{
    descriptives.set(new_inputs_number);
    scaling_methods.set(new_inputs_number, MinimumMaximum);
}


void ScalingLayer::set_neurons_number(const size_t& new_neurons_number)
{
    descriptives.set(new_neurons_number);
    scaling_methods.set(new_neurons_number, MinimumMaximum);
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
    set_scaling_methods(MinimumMaximum);

    set_display(true);

    layer_type = Scaling;
}


/// Sets all the scaling layer descriptives from a vector descriptives structures.
/// The size of the vector must be equal to the number of scaling neurons in the layer.
/// @param new_descriptives Scaling layer descriptives.

void ScalingLayer::set_descriptives(const Vector<Descriptives>& new_descriptives)
{
#ifdef __OPENNN_DEBUG__

    const size_t new_descriptives_size = new_descriptives.size();

    const size_t neurons_number = get_neurons_number();

    if(new_descriptives_size != neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_descriptives(const Vector<Descriptives>&) method.\n"
               << "Size of descriptives (" << new_descriptives_size << ") is not equal to number of scaling neurons (" << neurons_number << ").\n";

        throw logic_error(buffer.str());
    }

#endif

    // Set all descriptives

    descriptives = new_descriptives;
}


void ScalingLayer::set_descriptives_eigen(const Eigen::MatrixXd& descriptives_eigen)
{
    const size_t neurons_number = get_neurons_number();

    Vector<Descriptives> descriptives(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
    {
        descriptives[i].set_minimum(descriptives_eigen(static_cast<int>(i), 0));
        descriptives[i].set_maximum(descriptives_eigen(static_cast<int>(i), 1));
        descriptives[i].set_mean(descriptives_eigen(static_cast<int>(i), 2));
        descriptives[i].set_standard_deviation(descriptives_eigen(static_cast<int>(i), 3));
    }

    set_descriptives(descriptives);
}


/// Sets the descriptives of a single scaling neuron.
/// @param i Index of neuron.
/// @param item_descriptives Descriptives structure for that neuron.

void ScalingLayer::set_item_descriptives(const size_t& i, const Descriptives& item_descriptives)
{
    descriptives[i] = item_descriptives;
}


/// Sets the minimum value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_minimum Minimum value.

void ScalingLayer::set_minimum(const size_t& i, const double& new_minimum)
{
    descriptives[i].set_minimum(new_minimum);
}


/// Sets the maximum value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_maximum Maximum value.

void ScalingLayer::set_maximum(const size_t& i, const double& new_maximum)
{
    descriptives[i].set_maximum(new_maximum);
}


/// Sets the mean value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_mean Mean value.

void ScalingLayer::set_mean(const size_t& i, const double& new_mean)
{
    descriptives[i].set_mean(new_mean);
}


/// Sets the standard deviation value of a given scaling neuron.
/// @param i Index of scaling neuron.
/// @param new_standard_deviation Standard deviation value.

void ScalingLayer::set_standard_deviation(const size_t& i, const double& new_standard_deviation)
{
    descriptives[i].set_standard_deviation(new_standard_deviation);
}


/// Sets the methods to be used for scaling each variable.
/// @param new_scaling_methods New scaling methods for the variables.

void ScalingLayer::set_scaling_methods(const Vector<ScalingLayer::ScalingMethod>& new_scaling_methods)
{
#ifdef __OPENNN_DEBUG__

    const size_t neurons_number = get_neurons_number();

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_scaling_methods(const Vector<ScalingMethod>&) method.\n"
               << "Neurons number (" << neurons_number << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    scaling_methods = new_scaling_methods;
}


/// Sets the methods to be used for scaling each variable.
/// The argument is a vector string containing the name of the methods("NoScaling", "MeanStandardDeviation" or "MinimumMaximum").
/// @param new_scaling_methods_string New scaling methods for the variables.

void ScalingLayer::set_scaling_methods(const Vector<string>& new_scaling_methods_string)
{
    const size_t neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_scaling_methods(const Vector<string>&) method.\n"
               << "Neurons number (" << neurons_number << ") must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<ScalingMethod> new_scaling_methods(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string[i] == "NoScaling")
        {
            new_scaling_methods[i] = NoScaling;
        }
        else if(new_scaling_methods_string[i] == "MeanStandardDeviation")
        {
            new_scaling_methods[i] = MeanStandardDeviation;
        }
        else if(new_scaling_methods_string[i] == "MinimumMaximum")
        {
            new_scaling_methods[i] = MinimumMaximum;
        }
        else if(new_scaling_methods_string[i] == "StandardDeviation")
        {
            new_scaling_methods[i] = StandardDeviation;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void set_scaling_methods(const Vector<string>&) method.\n"
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
    const size_t neurons_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    if(neurons_number == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void set_scaling_methods(const Vector<string>&) method.\n"
               << "Neurons number (" << neurons_number << ")must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<ScalingMethod> new_scaling_methods(neurons_number);

    for(size_t i = 0; i < neurons_number; i++)
    {
        if(new_scaling_methods_string == "NoScaling")
        {
            new_scaling_methods[i] = NoScaling;
        }
        else if(new_scaling_methods_string == "MeanStandardDeviation")
        {
            new_scaling_methods[i] = MeanStandardDeviation;
        }
        else if(new_scaling_methods_string == "MinimumMaximum")
        {
            new_scaling_methods[i] = MinimumMaximum;
        }
        else if(new_scaling_methods_string == "StandardDeviation")
        {
            new_scaling_methods[i] = StandardDeviation;
        }
        else
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "void set_scaling_methods(const Vector<string>&) method.\n"
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
    const size_t neurons_number = get_neurons_number();

    for(size_t i = 0; i < neurons_number; i++)
    {
        scaling_methods[i] = new_scaling_method;
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


/// Add a scaling neuron from the scaling layer and asociate new descriptives.
/// @param new_descriptives Value of the descriptives of the new neuron added. The default value is an empty vector.

void ScalingLayer::grow_neuron(const Descriptives& new_descriptives)
{
    descriptives.push_back(new_descriptives);
}


/// Removes a given scaling neuron from the scaling layer.
/// @param index Index of neuron to be removed.

void ScalingLayer::prune_neuron(const size_t& index)
{
#ifdef __OPENNN_DEBUG__

    const size_t neurons_number = get_neurons_number();

    if(index >= neurons_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void prune_neuron(const size_t&) method.\n"
               << "Index of scaling neuron is equal or greater than number of scaling neurons.\n";

        throw logic_error(buffer.str());
    }

#endif

    descriptives.erase(descriptives.begin() + static_cast<long long>(index));
}


/// Returns true if the number of scaling neurons is zero, and false otherwise. 

bool ScalingLayer::is_empty() const
{
    const size_t inputs_number = get_neurons_number();

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
/// displays a warning message if they are outside.
/// @param inputs Set of inputs to the scaling layer.

void ScalingLayer::check_range(const Vector<double>& inputs) const
{
    const size_t inputs_number = get_neurons_number();

#ifdef __OPENNN_DEBUG__

    const size_t size = inputs.size();

    if(size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: ScalingLayer class.\n"
               << "void check_range(const Vector<double>&) const method.\n"
               << "Size of inputs must be equal to number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    // Check inputs

    if(display)
    {
        for(size_t i = 0; i < inputs_number; i++)
        {
            if(inputs[i] < descriptives[i].minimum)
            {
                cout << "OpenNN Warning: ScalingLayer class.\n"
                          << "void check_range(const Vector<double>&) const method.\n"
                          << "Input value " << i << " is less than corresponding minimum.\n";
            }

            if(inputs[i] > descriptives[i].maximum)
            {
                cout << "OpenNN Warning: ScalingLayer class.\n"
                          << "void check_range(const Vector<double>&) const method.\n"
                          << "Input value " << i << " is greater than corresponding maximum.\n";
            }
        }
    }
}


/// Scales some values to produce some scaled values. 
/// @param inputs Set of inputs to the scaling layer.

Tensor<double> ScalingLayer::calculate_outputs(const Tensor<double>& inputs)
{   
    Tensor<double> outputs;

    if(inputs.get_dimensions_number() == 2)
    {
        const size_t neurons_number = get_neurons_number();

    #ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        const size_t columns_number = inputs.get_dimension(1);

        if(columns_number != neurons_number)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "Tensor<double> calculate_outputs(const Tensor<double>&) const method.\n"
                   << "Size of inputs (" << columns_number << ") must be equal to number of scaling neurons (" << neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

    #endif

        const size_t points_number = inputs.get_dimension(0);

        outputs.set(Vector<size_t>({points_number, neurons_number}));

        for(size_t i = 0; i < points_number; i++)
        {
            for(size_t j = 0; j < neurons_number; j++)
            {
                if(abs(descriptives[j].minimum - descriptives[j].maximum) < numeric_limits<double>::min())
                {
                    if(display)
                    {
                        cout << "OpenNN Warning: ScalingLayer class.\n"
                                  << "Tensor<double> calculate_mean_standard_deviation_outputs(const Tensor<double>&) const method.\n"
                                  << "Standard deviation of variable " << i << " is zero.\n"
                                  << "Those variables won't be scaled.\n";
                    }

                    outputs[j] = inputs[j];
                }
                else
                {
                    if(scaling_methods[j] == NoScaling)
                    {
                        outputs(i,j) = inputs(i,j);
                    }
                    else if(scaling_methods[j] == MinimumMaximum)
                    {
                        outputs(i,j) = 2.0*(inputs(i,j) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum) - 1.0;
                    }
                    else if(scaling_methods[j] == MeanStandardDeviation)
                    {
                        outputs(i,j) = (inputs(i,j) - descriptives[j].mean)/descriptives[j].standard_deviation;
                    }
                    else if(scaling_methods[j] == StandardDeviation)
                    {
                        outputs(i,j) = inputs(i,j)/descriptives[j].standard_deviation;
                    }
                    else
                    {
                        ostringstream buffer;

                        buffer << "OpenNN Exception: ScalingLayer class\n"
                               << "Tensor<double> calculate_outputs(const Tensor<double>&) const method.\n"
                               << "Unknown scaling method.\n";

                        throw logic_error(buffer.str());
                    }
                }
            }
        }
    }
    else if(inputs.get_dimensions_number() == 4)
    {
        const size_t neurons_number = get_neurons_number();

    #ifdef __OPENNN_DEBUG__

        ostringstream buffer;

        const size_t columns_number = inputs.get_dimension(1) * inputs.get_dimension(2) * inputs.get_dimension(3);

        if(columns_number != neurons_number)
        {
            buffer << "OpenNN Exception: ScalingLayer class.\n"
                   << "Tensor<double> calculate_outputs(const Tensor<double>&) const method.\n"
                   << "Size of inputs (" << columns_number << ") must be equal to number of scaling neurons (" << neurons_number << ").\n";

            throw logic_error(buffer.str());
        }

    #endif

        const size_t points_number = inputs.get_dimension(0);

        outputs.set(Vector<size_t>({points_number, inputs.get_dimension(1), inputs.get_dimension(2), inputs.get_dimension(3)}));

        for(size_t i = 0; i < points_number; i++)
        {
            for(size_t j = 0; j < neurons_number; j++)
            {
                size_t channel_index = j%(inputs.get_dimension(1));
                size_t row_index = (j/(inputs.get_dimension(1)))%(inputs.get_dimension(2));
                size_t column_index = (j/(inputs.get_dimension(1) * inputs.get_dimension(2)))%(inputs.get_dimension(3));

                if(abs(descriptives[j].minimum - descriptives[j].maximum) < numeric_limits<double>::min())
                {
                    if(display)
                    {
                        cout << "OpenNN Warning: ScalingLayer class.\n"
                                  << "Tensor<double> calculate_mean_standard_deviation_outputs(const Tensor<double>&) const method.\n"
                                  << "Standard deviation of variable " << i << " is zero.\n"
                                  << "Those variables won't be scaled.\n";
                    }

                    outputs[j] = inputs[j];
                }
                else
                {
                    if(scaling_methods[j] == NoScaling)
                    {
                        outputs(i, channel_index, row_index, column_index) = inputs(i, channel_index, row_index, column_index);
                    }
                    else if(scaling_methods[j] == MinimumMaximum)
                    {
                        outputs(i, channel_index, row_index, column_index) = 2.0*(inputs(i, channel_index, row_index, column_index) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum) - 1.0;
                    }
                    else if(scaling_methods[j] == MeanStandardDeviation)
                    {
                        outputs(i, channel_index, row_index, column_index) = (inputs(i, channel_index, row_index, column_index) - descriptives[j].mean)/descriptives[j].standard_deviation;
                    }
                    else if(scaling_methods[j] == StandardDeviation)
                    {
                        outputs(i, channel_index, row_index, column_index) = inputs(i, channel_index, row_index, column_index)/descriptives[j].standard_deviation;
                    }
                    else
                    {
                        ostringstream buffer;

                        buffer << "OpenNN Exception: ScalingLayer class\n"
                               << "Tensor<double> calculate_outputs(const Tensor<double>&) const method.\n"
                               << "Unknown scaling method.\n";

                        throw logic_error(buffer.str());
                    }
                }
            }
        }
    }

    return outputs;
}


/// Calculates the outputs from the scaling layer with the minimum and maximum method for a set of inputs.
/// @param inputs Vector of input values to the scaling layer. The size must be equal to the number of scaling neurons. 

Tensor<double> ScalingLayer::calculate_minimum_maximum_outputs(const Tensor<double>& inputs) const
{
    const size_t points_number = inputs.get_dimension(0);
    const size_t neurons_number = get_neurons_number();

    Tensor<double> outputs(points_number, neurons_number);

    for(size_t j = 0; j < neurons_number; j++)
    {
        if(abs(descriptives[j].maximum-descriptives[j].minimum) < numeric_limits<double>::min())
        {
            if(display)
            {
                cout << "OpenNN Warning: ScalingLayer class\n"
                     << "Vector<double> calculate_minimum_maximum_outputs(Vector<double>&) const method.\n"
                     << "Minimum and maximum values of variable " << j << " are equal.\n"
                     << "Those inputs won't be scaled.\n";
            }

            outputs[j] = inputs[j];
        }
        else
        {
            outputs[j] = 2.0*(inputs[j] - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum) - 1.0;
        }
    }

    return outputs;
}


/// Calculates the outputs from the scaling layer with the mean and standard deviation method for a set of inputs.
/// @param inputs Vector of input values to the scaling layer. The size must be equal to the number of scaling neurons. 

Tensor<double> ScalingLayer::calculate_mean_standard_deviation_outputs(const Tensor<double>& inputs) const
{
    const size_t points_number = inputs.get_dimension(0);
    const size_t neurons_number = get_neurons_number();

    Tensor<double> outputs(points_number, neurons_number);

    for(size_t j = 0; j < neurons_number; j++)
    {
        if(abs(descriptives[j].standard_deviation) < numeric_limits<double>::min())
        {
            if(display)
            {
                cout << "OpenNN Warning: ScalingLayer class.\n"
                          << "Vector<double> calculate_mean_standard_deviation_outputs(const Vector<double>&) const method.\n"
                          << "Standard deviation of variable " << j << " is zero.\n"
                          << "Those variables won't be scaled.\n";
            }

            outputs[j] = inputs[j];
        }
        else
        {
            outputs[j] = (inputs[j] - descriptives[j].mean)/descriptives[j].standard_deviation;
        }
    }

    return outputs;
}


/// Returns a string with the expression of the scaling process when the none method is used.
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons.
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons.

string ScalingLayer::write_no_scaling_expression(const Vector<string>& inputs_names, const Vector<string>& outputs_names) const
{
    const size_t inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names[i] << "=" << inputs_names[i] << ";\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the scaling process with the minimum and maximum method. 
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons. 
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons. 

string ScalingLayer::write_minimum_maximum_expression(const Vector<string>& inputs_names, const Vector<string>& outputs_names) const
{
    const size_t inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names[i] << "=2*(" << inputs_names[i] << "-" << descriptives[i].minimum << ")/(" << descriptives[i].maximum << "-" << descriptives[i].minimum << ")-1;\n";
    }

    return buffer.str();
}   


/// Returns a string with the expression of the scaling process with the mean and standard deviation method. 
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons. 
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons. 

string ScalingLayer::write_mean_standard_deviation_expression(const Vector<string>& inputs_names, const Vector<string>& outputs_names) const
{
    const size_t inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names[i] << "= (" << inputs_names[i] << "-" << descriptives[i].mean << ")/" << descriptives[i].standard_deviation << ";\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the scaling process with the standard deviation method.
/// @param inputs_names Name of inputs to the scaling layer. The size of this vector must be equal to the number of scaling neurons.
/// @param outputs_names Name of outputs from the scaling layer. The size of this vector must be equal to the number of scaling neurons.

string ScalingLayer::write_standard_deviation_expression(const Vector<string>& inputs_names, const Vector<string>& outputs_names) const
{
    const size_t inputs_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer << outputs_names[i] << "=" << inputs_names[i] << "/" << descriptives[i].standard_deviation << ";\n";
    }

    return buffer.str();
}


/// Returns a string with the expression of the inputs scaling process. 

string ScalingLayer::write_expression(const Vector<string>& inputs_names, const Vector<string>& outputs_names) const
{
    const size_t neurons_number = get_neurons_number();

    ostringstream buffer;

    buffer.precision(10);

    for(size_t i = 0; i < neurons_number; i++)
    {
        if(scaling_methods[i] == NoScaling)
        {
            buffer << outputs_names[i] << " = " << inputs_names[i] << ";\n";
        }
        else if(scaling_methods[i] == MinimumMaximum)
        {
            buffer << outputs_names[i] << " = 2*(" << inputs_names[i] << "-" << descriptives[i].minimum << ")/(" << descriptives[i].maximum << "-" << descriptives[i].minimum << ")-1;\n";
        }
        else if(scaling_methods[i] == MeanStandardDeviation)
        {
            buffer << outputs_names[i] << " = (" << inputs_names[i] << "-" << descriptives[i].mean << ")/" << descriptives[i].standard_deviation << ";\n";
        }
        else if(scaling_methods[i] == StandardDeviation)
        {
            buffer << outputs_names[i] << " = " << inputs_names[i] << "/" << descriptives[i].standard_deviation << ";\n";
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

    return buffer.str();
}


/// Returns a string representation of the current scaling layer object. 

string ScalingLayer::object_to_string() const
{
    ostringstream buffer;

    const size_t neurons_number = get_neurons_number();

    buffer << "Scaling layer\n";

    for(size_t i = 0; i < neurons_number; i++)
    {
        buffer << "Descriptives " << i+1 << "\n"
               << "Minimum: " << descriptives[i].minimum << "\n"
               << "Maximum: " << descriptives[i].maximum << "\n"
               << "Mean: " << descriptives[i].mean << "\n"
               << "Standard deviation: " << descriptives[i].standard_deviation << "\n";
    }

    buffer << "Scaling methods: " << write_scaling_methods() << "\n";

    return buffer.str();
}


/// Serializes the scaling layer object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* ScalingLayer::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    ostringstream buffer;

    tinyxml2::XMLElement* scaling_layer_element = document->NewElement("ScalingLayer");

    document->InsertFirstChild(scaling_layer_element);

    // Scaling neurons number

    tinyxml2::XMLElement* size_element = document->NewElement("ScalingNeuronsNumber");
    scaling_layer_element->LinkEndChild(size_element);

    const size_t neurons_number = get_neurons_number();

    buffer.str("");
    buffer << neurons_number;

    tinyxml2::XMLText* size_text = document->NewText(buffer.str().c_str());
    size_element->LinkEndChild(size_text);

    const Vector<string> scaling_methods_string = write_scaling_methods();

    for(size_t i = 0; i < neurons_number; i++)
    {
        tinyxml2::XMLElement* scaling_neuron_element = document->NewElement("ScalingNeuron");
        scaling_neuron_element->SetAttribute("Index", static_cast<unsigned>(i+1));

        scaling_layer_element->LinkEndChild(scaling_neuron_element);

        // Minimum

        tinyxml2::XMLElement* minimum_element = document->NewElement("Minimum");
        scaling_neuron_element->LinkEndChild(minimum_element);

        buffer.str("");
        buffer << descriptives[i].minimum;

        tinyxml2::XMLText* minimum_text = document->NewText(buffer.str().c_str());
        minimum_element->LinkEndChild(minimum_text);

        // Maximum

        tinyxml2::XMLElement* maximum_element = document->NewElement("Maximum");
        scaling_neuron_element->LinkEndChild(maximum_element);

        buffer.str("");
        buffer << descriptives[i].maximum;

        tinyxml2::XMLText* maximum_text = document->NewText(buffer.str().c_str());
        maximum_element->LinkEndChild(maximum_text);

        // Mean

        tinyxml2::XMLElement* mean_element = document->NewElement("Mean");
        scaling_neuron_element->LinkEndChild(mean_element);

        buffer.str("");
        buffer << descriptives[i].mean;

        tinyxml2::XMLText* mean_text = document->NewText(buffer.str().c_str());
        mean_element->LinkEndChild(mean_text);

        // Standard deviation

        tinyxml2::XMLElement* standard_deviation_element = document->NewElement("StandardDeviation");
        scaling_neuron_element->LinkEndChild(standard_deviation_element);

        buffer.str("");
        buffer << descriptives[i].standard_deviation;

        tinyxml2::XMLText* standard_deviation_text = document->NewText(buffer.str().c_str());
        standard_deviation_element->LinkEndChild(standard_deviation_text);

        // Scaling method

        tinyxml2::XMLElement* scaling_method_element = document->NewElement("ScalingMethod");
        scaling_neuron_element->LinkEndChild(scaling_method_element);

        buffer.str("");
        buffer << scaling_methods_string[i];

        tinyxml2::XMLText* scaling_method_text = document->NewText(buffer.str().c_str());
        scaling_method_element->LinkEndChild(scaling_method_text);
    }

    // Scaling method

//    tinyxml2::XMLElement* method_element = document->NewElement("ScalingMethod");
//    scaling_layer_element->LinkEndChild(method_element);

//    tinyxml2::XMLText* method_text = document->NewText(write_scaling_method().c_str());
//    method_element->LinkEndChild(method_text);

    // Display warnings

    //   tinyxml2::XMLElement* display_element = document->NewElement("Display");
    //   scaling_layer_element->LinkEndChild(display_element);

    //   buffer.str("");
    //   buffer << display;

    //   tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
    //   display_element->LinkEndChild(display_text);

    return document;
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the scaling layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ScalingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    const size_t neurons_number = get_neurons_number();

    file_stream.OpenElement("ScalingLayer");

    // Scaling neurons number

    file_stream.OpenElement("ScalingNeuronsNumber");

    buffer.str("");
    buffer << neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    const Vector<string> scaling_methods_string = write_scaling_methods();

    // Scaling neurons

    for(size_t i = 0; i < neurons_number; i++)
    {
        file_stream.OpenElement("ScalingNeuron");

        file_stream.PushAttribute("Index",static_cast<unsigned>(i)+1);

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

//        file_stream.CloseElement();

        file_stream.CloseElement();

        // Scaling Method

        file_stream.OpenElement("ScalingMethod");

        buffer.str("");
        buffer << scaling_methods_string[i];

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        file_stream.CloseElement();
    }

    // Scaling method

//    file_stream.OpenElement("ScalingMethod");

//    file_stream.PushText(write_scaling_method().c_str());

//    file_stream.CloseElement();


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

    const size_t neurons_number = static_cast<size_t>(atoi(neurons_number_element->GetText()));

    set(neurons_number);

    unsigned index = 0; // size_t does not work

    const tinyxml2::XMLElement* start_element = neurons_number_element;

    for(size_t i = 0; i < neurons_number; i++)
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
            descriptives[i].minimum = atof(minimum_element->GetText());
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
            descriptives[i].maximum = atof(maximum_element->GetText());
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
            descriptives[i].mean = atof(mean_element->GetText());
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
            descriptives[i].standard_deviation = atof(standard_deviation_element->GetText());
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

        if(new_method == "NoScaling")
        {
            scaling_methods[i] = NoScaling;
        }
        else if(new_method == "MinimumMaximum")
        {
            scaling_methods[i] = MinimumMaximum;
        }
        else if(new_method == "MeanStandardDeviation")
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

//            buffer << "OpenNN Exception: ScalingLayer class.\n"
//                   << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
//                   << "Unknown scaling method element " << i+1 << " (" << new_method << ").\n";

//            throw logic_error(buffer.str());
        }
    }

    // Scaling method
//    {
//        const tinyxml2::XMLElement* scaling_method_element = scaling_layer_element->FirstChildElement("ScalingMethod");

//        if(scaling_method_element)
//        {
//            string new_method = scaling_method_element->GetText();

//            try
//            {
//                set_scaling_method(new_method);
//            }
//            catch(const logic_error& e)
//            {
//                cerr << e.what() << endl;
//            }
//        }
//    }

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
