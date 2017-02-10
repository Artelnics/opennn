/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   U N S C A L I N G   L A Y E R    C L A S S                                                                 */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "unscaling_layer.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 

UnscalingLayer::UnscalingLayer(void)
{
    set();
}


/// Outputs number constructor. 

UnscalingLayer::UnscalingLayer(const size_t& new_unscaling_neurons_number)
{
    set(new_unscaling_neurons_number);
}


/// Outputs statistics constructor. 

UnscalingLayer::UnscalingLayer(const Vector< Statistics<double> > & new_statistics)
{
    set(new_statistics);
}


/// XML constructor. 

UnscalingLayer::UnscalingLayer(const tinyxml2::XMLDocument& unscaling_layer_document)
{
    set(unscaling_layer_document);
}


/// Copy constructor. 

UnscalingLayer::UnscalingLayer(const UnscalingLayer& other_unscaling_layer)
{
    set(other_unscaling_layer);
}


// DESTRUCTOR

/// Destructor.

UnscalingLayer::~UnscalingLayer(void)
{
}


// ASSIGNMENT OPERATOR

// UnscalingLayer& operator = (const UnscalingLayer&) method

/// Assignment operator.
/// @param other_unscaling_layer Object to be copied. 

UnscalingLayer& UnscalingLayer::operator = (const UnscalingLayer& other_unscaling_layer)
{
    if(this != &other_unscaling_layer)
    {
        //statistics = other_unscaling_layer.statistics;

        unscaling_method = other_unscaling_layer.unscaling_method;

        display = other_unscaling_layer.display;
    }

    return(*this);
}


// EQUAL TO OPERATOR

// bool operator == (const UnscalingLayer&) const method

/// Equal to operator. 
/// If compares this object with another object of the same class, and returns true if they are equal, and false otherwise. 
/// @param other_unscaling_layer Object to be compared with. 

bool UnscalingLayer::operator == (const UnscalingLayer& other_unscaling_layer) const
{
    if(
            //  statistics == other_unscaling_layer.statistics
            unscaling_method == other_unscaling_layer.unscaling_method
            && display == other_unscaling_layer.display)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// size_t get_unscaling_neurons_number(void) const method

/// Returns the number of unscaling neurons in this layer. 

size_t UnscalingLayer::get_unscaling_neurons_number(void) const
{
    return(statistics.size());
}


// Vector< Statistics<double> > get_statistics(void) method

/// Returns all the available statistics of the inputs and output variables. 
/// The format is a vector of pointers to vectors of size ten: 
/// <ul>
/// <li> Mean of variables.
/// <li> Standard deviation of variables.
/// <li> Minimum of variables.
/// <li> Maximum of variables.
/// </ul>

Vector< Statistics<double> > UnscalingLayer::get_statistics(void) const
{
    return(statistics);
}



// Matrix<double> arrange_statistics(void) const method

/// Returns a single matrix with the statistics of all unscaling neurons.
/// The number of rows is the number of unscaling neurons,
/// and the number of columns is 4 (minimum, maximum, mean and standard deviation).

Matrix<double> UnscalingLayer::arrange_statistics(void) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Matrix<double> statistics_matrix(unscaling_neurons_number, 4);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        statistics_matrix.set_row(i, statistics[i].to_vector());
    }

    return(statistics_matrix);
}


// Vector<double> arrange_minimums(void) const method

/// Returns a vector with the minimum values of all unscaling neurons.
/// The size is the number of neurons in the layer.

Vector<double> UnscalingLayer::arrange_minimums(void) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Vector<double> minimums(unscaling_neurons_number, 4);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        minimums[i] = statistics[i].minimum;
    }

    return(minimums);
}


// Vector<double> arrange_maximums(void) const method

/// Returns a vector with the maximum values of all unscaling neurons.
/// The size is the number of neurons in the layer.

Vector<double> UnscalingLayer::arrange_maximums(void) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Vector<double> maximums(unscaling_neurons_number, 4);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        maximums[i] = statistics[i].maximum;
    }

    return(maximums);
}


// const UnscalingMethod& get_unscaling_method(void) const method

/// Returns the method used for unscaling
/// (no unscaling, minimum and maximum or mean and standard deviation).

const UnscalingLayer::UnscalingMethod& UnscalingLayer::get_unscaling_method(void) const
{
    return(unscaling_method);
}


// std::string write_unscaling_method(void) const method

/// Returns a string with the name of the method used for unscaling
/// ("MinimumMaximum", "MeanStandardDeviation" or "NoUnscaling").

std::string UnscalingLayer::write_unscaling_method(void) const
{
    if(unscaling_method == NoUnscaling)
    {
        return("NoUnscaling");
    }
    else if(unscaling_method == MinimumMaximum)
    {
        return("MinimumMaximum");
    }
    else if(unscaling_method == MeanStandardDeviation)
    {
        return("MeanStandardDeviation");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "std::string write_unscaling_method(void) const method.\n"
               << "Unknown unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
}


// std::string write_unscaling_method_text(void) const method

/// Returns a string with the name of the method used for unscaling,
/// as paragraph text.

std::string UnscalingLayer::write_unscaling_method_text(void) const
{
    if(unscaling_method == NoUnscaling)
    {
        return("no unscaling");
    }
    else if(unscaling_method == MeanStandardDeviation)
    {
        return("mean and standard deviation");
    }
    else if(unscaling_method == MinimumMaximum)
    {
        return("minimum and maximum");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "std::string write_unscaling_method_text(void) const method.\n"
               << "Unknown unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
}


// const bool& get_display(void) const method

/// Returns true if messages from this class are to be displayed on the screen, or false if messages 
/// from this class are not to be displayed on the screen.

const bool& UnscalingLayer::get_display(void) const
{
    return(display);
}


// void set(void) method

/// Sets the unscaling layer to be empty. 

void UnscalingLayer::set(void)
{
    statistics.set();

    set_default();
}


// void set(const size_t&) method

/// Sets a new size in the unscaling layer. 
/// It also sets the members to their default values. 

void UnscalingLayer::set(const size_t& new_unscaling_neurons_number)
{
    statistics.set(new_unscaling_neurons_number);

    set_default();
}


// void set(const Vector< Statistics<double> >&) method

/// Sets the size of the unscaling layer and the statistics values. 
/// @param new_statistics Vector of vectors containing the minimums, maximums, means and standard deviations for the unscaling layer. 
/// The size of this vector must be 4. 
/// The size of each subvector will be the size of the unscaling layer. 

void UnscalingLayer::set(const Vector< Statistics<double> >& new_statistics)
{
    statistics = new_statistics;

    set_default();
}


// void set(const tinyxml2::XMLDocument&) method

/// Sets the unscaling layer members from a XML document. 
/// @param new_unscaling_layer_document Pointer to a TinyXML document containing the member data.

void UnscalingLayer::set(const tinyxml2::XMLDocument& new_unscaling_layer_document)
{
    set_default();

    from_XML(new_unscaling_layer_document);
}


// void set(const UnscalingLayer&) method

/// Sets the members of this object to be the members of another object of the same class. 

void UnscalingLayer::set(const UnscalingLayer& new_unscaling_layer)
{
    statistics = new_unscaling_layer.statistics;

    unscaling_method = new_unscaling_layer.unscaling_method;

    display = new_unscaling_layer.display;
}


// void set_default(void) method

/// This member sets the default values for the unscaling layer:
/// <ul>
/// <li> Minimus: -1 for all unscaling neurons.
/// <li> Maximums: 1 for al unscaling neurons. 
/// <li> Means: 0 for all unscaling neurons. 
/// <li> Standard deviations 1 for all unscaling neurons. 
/// <li> Unscaling method: Minimum and maximum. 
/// <li> Display: True. 
/// </ul>

void UnscalingLayer::set_default(void)
{
    set_unscaling_method(MinimumMaximum);

    set_display(true);
}


// void set_statistics(const Vector< Statistics<double> >&) method

/// Sets the statistics for all the neurons in the unscaling layer from a vector.
/// The size of this vector must be equal to the number of unscaling neurons.
/// @param new_statistics Unscaling neurons statistics.


void UnscalingLayer::set_statistics(const Vector< Statistics<double> >& new_statistics)
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    const size_t new_statistics_size = new_statistics.size();

    if(new_statistics_size != unscaling_neurons_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "void set_statistics(const Vector< Statistics<double> >&) method.\n"
               << "Size of statistics must be equal to number of unscaling neurons.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    // Set all statistics

    statistics = new_statistics;
}


// void set_item_statistics(const size_t&, const Statistics<double>&) method

/// Sets the minimum, maximum, mean and standard deviation values of a single unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param item_statistics  Statistics values for that neuron.

void UnscalingLayer::set_item_statistics(const size_t& i, const Statistics<double>& item_statistics)
{
    statistics[i] = item_statistics;
}


// void set_minimum(const size_t&, const double&) method

/// Sets the minimum value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_minimum Minimum value.

void UnscalingLayer::set_minimum(const size_t& i, const double& new_minimum)
{
    statistics[i].set_minimum(new_minimum);
}


// void set_maximum(const size_t&, const double&) method

/// Sets the maximum value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_maximum Maximum value.

void UnscalingLayer::set_maximum(const size_t& i, const double& new_maximum)
{
    statistics[i].set_maximum(new_maximum);
}


// void set_mean(const size_t&, const double&) method

/// Sets the mean value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_mean Mean value.

void UnscalingLayer::set_mean(const size_t& i, const double& new_mean)
{
    statistics[i].set_mean(new_mean);
}


// void set_standard_deviation(const size_t&, const double&) method

/// Sets the standard deviation value of a given unscaling neuron.
/// @param i Index of unscaling neuron.
/// @param new_standard_deviation Standard deviation value.

void UnscalingLayer::set_standard_deviation(const size_t& i, const double& new_standard_deviation)
{
    statistics[i].set_standard_deviation(new_standard_deviation);
}


// void set_unscaling_method(const UnscalingMethod&) method

/// Sets the method to be used for unscaling the outputs from the multilayer perceptron
/// @param new_unscaling_method New unscaling method for the output variables.

void UnscalingLayer::set_unscaling_method(const UnscalingLayer::UnscalingMethod& new_unscaling_method)
{
    unscaling_method = new_unscaling_method;
}


// void set_unscaling_method(const std::string&) method

/// Sets the method to be used for unscaling the outputs from the multilayer perceptron
/// The argument is a string containing the name of the method ("NoUnscaling", "MeanStandardDeviation" or "MinimumMaximum").
/// @param new_unscaling_method New unscaling method for the output variables.

void UnscalingLayer::set_unscaling_method(const std::string& new_unscaling_method)
{
    if(new_unscaling_method == "NoUnscaling")
    {
        set_unscaling_method(NoUnscaling);
    }
    else if(new_unscaling_method == "MeanStandardDeviation")
    {
        set_unscaling_method(MeanStandardDeviation);
    }
    else if(new_unscaling_method == "MinimumMaximum")
    {
        set_unscaling_method(MinimumMaximum);
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "void set_unscaling_method(const std::string&) method.\n"
               << "Unknown unscaling method: " << new_unscaling_method << ".\n";

        throw std::logic_error(buffer.str());
    }
}


// void set_display(const bool&) method

/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void UnscalingLayer::set_display(const bool& new_display)
{
    display = new_display;
}


// void prune_unscaling_neuron(const size_t&) method

/// Removes a single unscaling neuron from the unscaling layer.
/// @param index Index of neuron to be removed.

void UnscalingLayer::prune_unscaling_neuron(const size_t& index)
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    if(index >= unscaling_neurons_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "void prune_unscaling_neuron(const size_t&) method.\n"
               << "Index of unscaling neuron is equal or greater than number of unscaling neurons.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    statistics.erase(statistics.begin() + index);
}


// void check_range(const Vector<double>&) const method

/// Checks whether the outptus from the unscaling layer are inside the range defined by the minimums and maximum values. 
/// It displays a warning message if they are outside.
/// @param outputs Set of outptus from the unscaling layer.

void UnscalingLayer::check_range(const Vector<double>& outputs) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = outputs.size();

    if(size != unscaling_neurons_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "void check_range(const Vector<double>&) const method.\n"
               << "Size of outputs must be equal to number of unscaling neurons.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    // Check outputs

    if(display)
    {
        for(size_t i = 0; i < unscaling_neurons_number; i++)
        {
            if(outputs[i] < statistics[i].minimum)
            {
                std::cout << "OpenNN Warning: UnscalingLayer class.\n"
                          << "void check_range(const Vector<double>&) const method.\n"
                          << "Output variable " << i << " is less than outputs.\n";
            }

            if(outputs[i] > statistics[i].maximum)
            {
                std::cout << "OpenNN Warning: UnscalingLayer class.\n"
                          << "void check_range(const Vector<double>&) const method.\n"
                          << "Output variable " << i << " is greater than maximum.\n";
            }
        }
    }
}


// bool is_empty(void) const method

/// Returns true if the number of unscaling neurons is zero, and false otherwise. 

bool UnscalingLayer::is_empty(void) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    if(unscaling_neurons_number == 0)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// void initialize_random(void) method

/// Initializes at random the statistics of all neurons in the layer
/// and the unscaling method.

void UnscalingLayer::initialize_random(void)
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    // Statistics

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        statistics[i].initialize_random();
    }

    // Unscaling method

    switch(rand()%2)
    {
    case 0:
    {
        unscaling_method = MinimumMaximum;
    }
        break;

    case 1:
    {
        unscaling_method = MeanStandardDeviation;
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "void initialize_random(void) method.\n"
               << "Unknown unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }
}


// Vector<double> calculate_outputs(const Vector<double>&) const method

/// Calculates the outputs from the unscaling layer for a given set of inputs to that layer.  
/// @param inputs Set of inputs to the unscaling layer.

Vector<double> UnscalingLayer::calculate_outputs(const Vector<double>& inputs) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    const size_t size = inputs.size();

    if(size != unscaling_neurons_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
               << "Size must be equal to number of unscaling neurons.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    // Unscale

    switch(unscaling_method)
    {
    case MinimumMaximum:
    {
        return(calculate_minimum_maximum_outputs(inputs));
    }
        break;

    case MeanStandardDeviation:
    {
        return(calculate_mean_standard_deviation_outputs(inputs));
    }
        break;

    case NoUnscaling:
    {
        return(inputs);
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
               << "Unknown unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }
}  


// Vector<double> calculate_derivatives(const Vector<double>&) const method

/// This method retuns the derivatives of the unscaled outputs with respect to the scaled outputs.
/// That derivatives depend on the method for unscaling the outputs to be used. 

Vector<double> UnscalingLayer::calculate_derivatives(const Vector<double>& inputs) const
{
    // Control sentence (if debug)

    //   #ifdef __OPENNN_DEBUG__
    //
    //  const size_t size = scaled_output.size();
    //
    //   if(size != unscaling_neurons_number)
    //   {
    //      std::ostringstream buffer;

    //      buffer << "OpenNN Exception: UnscalingLayer class.\n"
    //             << "Vector<double> calculate_derivatives(const Vector<double>&) const method.\n"
    //             << "Size must be equal to number of unscaling neurons.\n";
    //
    //	  throw std::logic_error(buffer.str());
    //   }
    //
    //   #endif

    switch(unscaling_method)
    {
    case MinimumMaximum:
    {
        return(calculate_minimum_maximum_derivatives(inputs));
    }
        break;

    case MeanStandardDeviation:
    {
        return(calculate_mean_standard_deviation_derivatives(inputs));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "Vector<double> calculate_derivatives(const Vector<double>&) const.\n"
               << "Unknown scaling and unscaling method.\n";

        throw std::logic_error(buffer.str());
    }
        break;

    }// end switch
}


// Vector<double> calculate_second_derivatives(const Vector<double>&) const

/// This method retuns the second derivatives of the unscaled outputs with respect to the scaled outputs.
/// That second derivatives depend on the method for unscaling the outputs to be used. 

Vector<double> UnscalingLayer::calculate_second_derivatives(const Vector<double>& inputs) const
{
    switch(unscaling_method)
    {
    case MinimumMaximum:
    {
        return(calculate_minimum_maximum_second_derivatives(inputs));
    }
        break;

    case MeanStandardDeviation:
    {
        return(calculate_mean_standard_deviation_second_derivatives(inputs));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "Vector<double> calculate_second_derivatives(const Vector<double>&) const.\n"
               << "Unknown scaling and unscaling method.\n";

        throw std::logic_error(buffer.str());
    }// end default
        break;

    }// end switch
}


// Vector<double> calculate_minimum_maximum_outputs(const Vector<double>&) const method

/// Calculates the outputs from the unscaling layer with the minimum and maximum method for a set of inputs.
/// @param inputs Vector of input values to the unscaling layer. The size must be equal to the number of unscaling neurons. 

Vector<double> UnscalingLayer::calculate_minimum_maximum_outputs(const Vector<double>& inputs) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Vector<double> outputs(unscaling_neurons_number);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        if(statistics[i].maximum - statistics[i].minimum < 1e-99)
        {
            if(display)
            {
                std::cout << "OpenNN Warning: UnscalingLayer class.\n"
                          << "Vector<double> calculate_minimum_maximum_outputs(Vector<double>&) const method.\n"
                          << "Minimum and maximum values of output variable " << i << " are equal.\n"
                          << "Those outputs won't be unscaled.\n";
            }

            outputs[i] = inputs[i];
        }
        else
        {
            outputs[i] = 0.5*(inputs[i] + 1.0)*(statistics[i].maximum-statistics[i].minimum) + statistics[i].minimum;
        }
    }

    return(outputs);
}


// Vector<double> calculate_minimum_maximum_derivatives(const Vector<double>&) const method

/// Calculates the derivatives of the outputs from the unscaling layer with the minimum and maximum method.
/// As the minimum and maximum method is a linear method, the derivatives will not depend on the inputs. 

Vector<double> UnscalingLayer::calculate_minimum_maximum_derivatives(const Vector<double>&) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Vector<double> derivative(unscaling_neurons_number);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        if(statistics[i].maximum-statistics[i].minimum < 1e-99)
        {
            if(display)
            {
                std::cout << "OpenNN Warning: UnscalingLayer class.\n"
                          << "Vector<double> calculate_minimum_maximum_derivatives(const Vector<double>&) const.\n"
                          << "Minimum and maximum values of output variable " << i << " are equal.\n"
                          << "Those derivatives won't be unscaled.\n";
            }

            derivative[i] = 1.0;
        }
        else
        {
            derivative[i] = 0.5*(statistics[i].maximum-statistics[i].minimum);
        }
    }

    return(derivative);
}


// Vector<double> calculate_minimum_maximum_second_derivatives(const Vector<double>&) const method

/// Calculates the second derivatives of the outputs from the unscaling layer with the minimum and maximum method.
/// As the minimum and maximum method is a linear method, the second derivatives will be always zero. 

Vector<double> UnscalingLayer::calculate_minimum_maximum_second_derivatives(const Vector<double>&) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    const Vector<double> unscaled_second_derivative(unscaling_neurons_number, 0.0);

    return(unscaled_second_derivative);
}


// Vector<double> calculate_mean_standard_deviation_outputs(const Vector<double>&) const method

/// Calculates the outputs from the unscaling layer with the mean and standard deviation method for a set of inputs.
/// @param inputs Vector of input values to the unscaling layer. The size must be equal to the number of unscaling neurons. 

Vector<double> UnscalingLayer::calculate_mean_standard_deviation_outputs(const Vector<double>& inputs) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Vector<double> outputs(unscaling_neurons_number);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        if(statistics[i].standard_deviation < 1e-99)
        {
            if(display)
            {
                std::cout << "OpenNN Warning: UnscalingLayer class.\n"
                          << "Vector<double> calculate_mean_standard_deviation_outputs(const Vector<double>&) const method.\n"
                          << "Standard deviation of output variable " << i << " is zero.\n"
                          << "Those outputs won't be unscaled.\n";
            }

            outputs[i] = inputs[i];
        }
        else
        {
            outputs[i] = inputs[i]*statistics[i].standard_deviation + statistics[i].mean;
        }
    }

    return(outputs);
}


// Vector<double> calculate_mean_standard_deviation_derivatives(const Vector<double>&) const method

/// Calculates the derivatives of the outputs from the unscaling layer with the mean and standard deviation method.
/// As the minimum and maximum method is a linear method, the derivatives will not depend on the inputs. 

Vector<double> UnscalingLayer::calculate_mean_standard_deviation_derivatives(const Vector<double>&) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Vector<double> unscaled_derivative(unscaling_neurons_number, 0.0);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        if(statistics[i].standard_deviation < 1e-99)
        {
            if(display)
            {
                std::cout << "OpenNN Warning: UnscalingLayer class.\n"
                          << "Vector<double> calculate_mean_standard_deviation_derivatives(const Vector<double>&) const.\n"
                          << "Standard deviation of output variable " << i << " is zero.\n"
                          << "Those derivatives won't be unscaled.\n";
            }

            unscaled_derivative[i] = 1.0;
        }
        else
        {
            unscaled_derivative[i] = statistics[i].standard_deviation;
        }
    }

    return(unscaled_derivative);
}


// Vector<double> calculate_mean_standard_deviation_second_derivatives(const Vector<double>&) const method

/// Calculates the second derivatives of the outputs from the unscaling layer with the mean and standard deviation method.
/// As the minimum and maximum method is a linear method, the second derivatives will be always zero. 

Vector<double> UnscalingLayer::calculate_mean_standard_deviation_second_derivatives(const Vector<double>&) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    const Vector<double> unscaled_second_derivative(unscaling_neurons_number, 0.0);

    return(unscaled_second_derivative);
}


// Matrix<double> arrange_Jacobian(const Vector<double>&) const method

/// Arranges a "Jacobian" matrix from the vector of derivatives. 

Matrix<double> UnscalingLayer::arrange_Jacobian(const Vector<double>& derivatives) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Matrix<double> Jacobian(unscaling_neurons_number, unscaling_neurons_number, 0.0);

    Jacobian.set_diagonal(derivatives);

    return(Jacobian);
}


// Vector< Matrix<double> > arrange_Hessian_form(const Vector<double>&) const method

/// Arranges a "Hessian form" vector of matrices from the vector of second derivatives. 

Vector< Matrix<double> > UnscalingLayer::arrange_Hessian_form(const Vector<double>& second_derivative) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    Vector< Matrix<double> > Hessian_form(unscaling_neurons_number);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        Hessian_form[i].set(unscaling_neurons_number, unscaling_neurons_number, 0.0);

        Hessian_form[i](i,i) = second_derivative[i];
    }
    return(Hessian_form);
}


// std::string to_string(void) const method

/// Returns a string representation of the current unscaling layer object. 

std::string UnscalingLayer::to_string(void) const
{
    std::ostringstream buffer;

    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    buffer << "Unscaling layer\n";

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        buffer << "Statistics " << i+1 << ":\n"
               << "Minimum: " << statistics[i].minimum << "\n"
               << "Maximum: " << statistics[i].maximum << "\n"
               << "Mean: " << statistics[i].mean << "\n"
               << "Standard deviation: " << statistics[i].standard_deviation << "\n";
    }

    buffer << "Unscaling method: " << write_unscaling_method() << "\n";
    //<< "Display: " << display << "\n";

    return(buffer.str());
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes this unscaling layer object into a TinyXML document-> 
/// Please read the OpenNN manual for more information about this. 

tinyxml2::XMLDocument* UnscalingLayer::to_XML(void) const
{
    std::ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    tinyxml2::XMLElement* unscaling_layer_element = document->NewElement("UnscalingLayer");

    document->InsertFirstChild(unscaling_layer_element);

    tinyxml2::XMLElement* element = NULL;
    tinyxml2::XMLText* text = NULL;

    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    // Unscaling neurons number
    {
        element = document->NewElement("UnscalingNeuronsNumber");
        unscaling_layer_element->LinkEndChild(element);

        buffer.str("");
        buffer << unscaling_neurons_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        tinyxml2::XMLElement* statistics_element = document->NewElement("Statistics");
        statistics_element->SetAttribute("Index", (unsigned)i+1);

        unscaling_layer_element->LinkEndChild(statistics_element);

        // Minimum

        tinyxml2::XMLElement* minimum_element = document->NewElement("Minimum");
        statistics_element->LinkEndChild(minimum_element);

        buffer.str("");
        buffer << statistics[i].minimum;

        tinyxml2::XMLText* minimum_text = document->NewText(buffer.str().c_str());
        minimum_element->LinkEndChild(minimum_text);

        // Maximum

        tinyxml2::XMLElement* maximum_element = document->NewElement("Maximum");
        statistics_element->LinkEndChild(maximum_element);

        buffer.str("");
        buffer << statistics[i].maximum;

        tinyxml2::XMLText* maximum_text = document->NewText(buffer.str().c_str());
        maximum_element->LinkEndChild(maximum_text);

        // Mean

        tinyxml2::XMLElement* mean_element = document->NewElement("Mean");
        statistics_element->LinkEndChild(mean_element);

        buffer.str("");
        buffer << statistics[i].mean;

        tinyxml2::XMLText* mean_text = document->NewText(buffer.str().c_str());
        mean_element->LinkEndChild(mean_text);

        // Standard deviation

        tinyxml2::XMLElement* standard_deviation_element = document->NewElement("StandardDeviation");
        statistics_element->LinkEndChild(standard_deviation_element);

        buffer.str("");
        buffer << statistics[i].standard_deviation;

        tinyxml2::XMLText* standard_deviation_text = document->NewText(buffer.str().c_str());
        standard_deviation_element->LinkEndChild(standard_deviation_text);
    }

    // Unscaling method
    {
        element = document->NewElement("UnscalingMethod");
        unscaling_layer_element->LinkEndChild(element);

        text = document->NewText(write_unscaling_method().c_str());
        element->LinkEndChild(text);
    }

    // Display
    //   {
    //      element = document->NewElement("Display");
    //      unscaling_layer_element->LinkEndChild(element);

    //      buffer.str("");
    //      buffer << display;

    //      text = document->NewText(buffer.str().c_str());
    //      element->LinkEndChild(text);
    //   }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the unscaling layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void UnscalingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    file_stream.OpenElement("UnscalingLayer");

    // Unscaling neurons number

    file_stream.OpenElement("UnscalingNeuronsNumber");

    buffer.str("");
    buffer << unscaling_neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Statistics

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        file_stream.OpenElement("Statistics");

        file_stream.PushAttribute("Index", (unsigned)i+1);

        // Minimum

        file_stream.OpenElement("Minimum");

        buffer.str("");
        buffer << statistics[i].minimum;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Maximum

        file_stream.OpenElement("Maximum");

        buffer.str("");
        buffer << statistics[i].maximum;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Mean

        file_stream.OpenElement("Mean");

        buffer.str("");
        buffer << statistics[i].mean;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();

        // Standard deviation

        file_stream.OpenElement("StandardDeviation");

        buffer.str("");
        buffer << statistics[i].standard_deviation;

        file_stream.PushText(buffer.str().c_str());

        file_stream.CloseElement();


        file_stream.CloseElement();
    }

    // Unscaling method

    file_stream.OpenElement("UnscalingMethod");

    file_stream.PushText(write_unscaling_method().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this unscaling layer object.
/// @param document XML document containing the member data.

void UnscalingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    std::ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("UnscalingLayer");

    if(!root_element)
    {
        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Unscaling layer element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Unscaling neurons number

    const tinyxml2::XMLElement* unscaling_neurons_number_element = root_element->FirstChildElement("UnscalingNeuronsNumber");

    if(!unscaling_neurons_number_element)
    {
        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Unscaling neurons number element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const size_t unscaling_neurons_number = atoi(unscaling_neurons_number_element->GetText());

    set(unscaling_neurons_number);

    unsigned index = 0; // size_t does not work

    const tinyxml2::XMLElement* start_element = unscaling_neurons_number_element;

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        const tinyxml2::XMLElement* statistics_element = start_element->NextSiblingElement("Statistics");
        start_element = statistics_element;

        if(!statistics_element)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Statistics of unscaling neuron " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        statistics_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Index " << index << " is not correct.\n";

            throw std::logic_error(buffer.str());
        }

        // Minimum

        const tinyxml2::XMLElement* minimum_element = statistics_element->FirstChildElement("Minimum");

        if(!minimum_element)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Minimum element " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(minimum_element->GetText())
        {
            statistics[i].minimum = atof(minimum_element->GetText());
        }

        // Maximum

        const tinyxml2::XMLElement* maximum_element = statistics_element->FirstChildElement("Maximum");

        if(!maximum_element)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Maximum element " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(maximum_element->GetText())
        {
            statistics[i].maximum = atof(maximum_element->GetText());
        }

        // Mean

        const tinyxml2::XMLElement* mean_element = statistics_element->FirstChildElement("Mean");

        if(!mean_element)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Mean element " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(mean_element->GetText())
        {
            statistics[i].mean = atof(mean_element->GetText());
        }

        // Standard deviation

        const tinyxml2::XMLElement* standard_deviation_element = statistics_element->FirstChildElement("StandardDeviation");

        if(!standard_deviation_element)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Standard deviation element " << i+1 << " is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(standard_deviation_element->GetText())
        {
            statistics[i].standard_deviation = atof(standard_deviation_element->GetText());
        }
    }

    // Unscaling method

    const tinyxml2::XMLElement* unscaling_method_element = root_element->FirstChildElement("UnscalingMethod");

    if(unscaling_method_element)
    {
        const std::string new_method = unscaling_method_element->GetText();

        try
        {
            set_unscaling_method(new_method);
        }
        catch(const std::logic_error& e)
        {
            std::cout << e.what() << std::endl;
        }
    }

    // Display

    const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

    if(element)
    {
        std::string new_display_string = element->GetText();

        try
        {
            set_display(new_display_string != "0");
        }
        catch(const std::logic_error& e)
        {
            std::cout << e.what() << std::endl;
        }
    }
}



// void UnscalingLayer::to_PMML(tinyxml2::XMLElement*, const Vector<std::string>& ) method.

/// Serializes the unscaling layer object into a PMML document.
/// @param element XML element to append the unscaling layer object.
/// @param outputs_names Names of the outputs variables.

void UnscalingLayer::to_PMML(tinyxml2::XMLElement* element, const Vector<std::string>& outputs_names) const
{
    tinyxml2::XMLDocument* pmml_document = element->GetDocument();

    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    std::stringstream double_precision_stream ;

    // Check unscaling neurons number error
    if(unscaling_neurons_number != outputs_names.size())
    {
        return;
    }

    if(unscaling_method == NoUnscaling)
    {
        return;
    }

    const Vector<double> inputs_to_unscale_range_begin(unscaling_neurons_number,0.0f);
    const Vector<double> inputs_to_unscale_range_end(unscaling_neurons_number,1.0f);

    const Vector<double> unscaled_outputs_range_begin = calculate_outputs(inputs_to_unscale_range_begin);
    const Vector<double> unscaled_outputs_range_end = calculate_outputs(inputs_to_unscale_range_end);


    for(size_t i = 0 ; i < unscaling_neurons_number; i++)
    {
        const std::string current_output_display_name(outputs_names[i]);
        const std::string current_output_name(current_output_display_name + "*");

        tinyxml2::XMLElement* derived_field = pmml_document->NewElement("DerivedField");
        element->LinkEndChild(derived_field);

        derived_field->SetAttribute("displayName",current_output_display_name.c_str());
        derived_field->SetAttribute("name",current_output_name.c_str());
        derived_field->SetAttribute("dataType","double");
        derived_field->SetAttribute("optype","continuous");


        tinyxml2::XMLElement* norm_continuous = pmml_document->NewElement("NormContinuous");
        derived_field->LinkEndChild(norm_continuous);

        norm_continuous->SetAttribute("field",current_output_display_name.c_str());


        tinyxml2::XMLElement* linear_norm_begin = pmml_document->NewElement("LinearNorm");
        norm_continuous->LinkEndChild(linear_norm_begin);

        linear_norm_begin->SetAttribute("norm", "0.0");

        double_precision_stream.str(std::string());
        double_precision_stream << std::setprecision(15) << unscaled_outputs_range_begin[i];

        linear_norm_begin->SetAttribute("orig",double_precision_stream.str().c_str());


        tinyxml2::XMLElement* linear_norm_end = pmml_document->NewElement("LinearNorm");
        norm_continuous->LinkEndChild(linear_norm_end);

        linear_norm_end->SetAttribute("norm", "1.0");

        double_precision_stream.str(std::string());
        double_precision_stream << std::setprecision(15) << unscaled_outputs_range_end[i];

        linear_norm_end->SetAttribute("orig",double_precision_stream.str().c_str());
    }
}


// void write_PMML(tinyxml2::XMLPrinter&, const Vector<std::string>&) const;

/// Serializes the unscaling layer object into a PMML document.
/// @param file_stream XML file where the unscaling layer object will be serialized.
/// @param outputs_names Names of the outputs variables.

void UnscalingLayer::write_PMML(tinyxml2::XMLPrinter& file_stream, const Vector<std::string>& outputs_names) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    std::stringstream double_precision_stream;

    // Check unscaling neurons number error

    if(unscaling_neurons_number != outputs_names.size())
    {
        return;
    }

    if(unscaling_method == NoUnscaling)
    {
        return;
    }

    const Vector<double> inputs_to_unscale_range_begin(unscaling_neurons_number,0.0f);
    const Vector<double> inputs_to_unscale_range_end(unscaling_neurons_number,1.0f);

    const Vector<double> unscaled_outputs_range_begin = calculate_outputs(inputs_to_unscale_range_begin);
    const Vector<double> unscaled_outputs_range_end = calculate_outputs(inputs_to_unscale_range_end);

    for(size_t i = 0 ; i < unscaling_neurons_number; i++)
    {
        const std::string current_output_display_name(outputs_names[i]);
        const std::string current_output_name(current_output_display_name + "*");

        file_stream.OpenElement("DerivedField");

        file_stream.PushAttribute("displayName",current_output_display_name.c_str());
        file_stream.PushAttribute("name",current_output_name.c_str());
        file_stream.PushAttribute("dataType","double");
        file_stream.PushAttribute("optype","continuous");


        file_stream.OpenElement("NormContinuous");

        file_stream.PushAttribute("field",current_output_display_name.c_str());

        // Normalization range begin

        file_stream.OpenElement("LinearNorm");

        file_stream.PushAttribute("norm", "0.0");

        double_precision_stream.str(std::string());
        double_precision_stream << std::setprecision(15) << unscaled_outputs_range_begin[i];

        file_stream.PushAttribute("orig",double_precision_stream.str().c_str());

        file_stream.CloseElement();

        // Normalization range end

        file_stream.OpenElement("LinearNorm");

        file_stream.PushAttribute("norm", "1.0");

        double_precision_stream.str(std::string());
        double_precision_stream << std::setprecision(15) << unscaled_outputs_range_end[i];

        file_stream.PushAttribute("orig",double_precision_stream.str().c_str());

        file_stream.CloseElement();

        // Close NormContinuous
        file_stream.CloseElement();

        // Close DerivedField
        file_stream.CloseElement();
    }
}


// void from_PMML(const tinyxml2::XMLElement*,const Vector<std::string>&) method

/// Deserializes a PMML document into this unscaling layer object.

void UnscalingLayer::from_PMML(const tinyxml2::XMLElement* element,const Vector<std::string>& outputs_names)
{
    std::stringstream buffer;

    const tinyxml2::XMLElement* derived_field = element->FirstChildElement("DerivedField");

    UnscalingMethod new_unscaling_method = NoUnscaling;

    int i = 0;
    while(derived_field)
    {
        std::string derived_field_name;

        const tinyxml2::XMLAttribute* attribute_display_name = derived_field->FindAttribute("displayName");

        if(!attribute_display_name)
        {
            const tinyxml2::XMLAttribute* attribute_name = derived_field->FindAttribute("name");

            if(!attribute_name)
            {
                buffer << "OpenNN Exception: UnscalingLayer class.\n"
                       << "void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>&) method.\n"
                       << "Attribute \"name\" in DerivedField element is NULL.\n";

                throw std::logic_error(buffer.str());
            }

            derived_field_name = std::string(attribute_name->Value());
        }
        else
        {
            derived_field_name = std::string(attribute_display_name->Value());
        }

        if( ! derived_field->FindAttribute("optype"))
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>&) method.\n"
                   << "Attribute \"optype\" in DerivedField element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        //not supported

        if( std::string(derived_field->FindAttribute("optype")->Value()) != "continuous")
        {
            break;
        }

        const tinyxml2::XMLElement* norm_continuous = derived_field->FirstChildElement("NormContinuous");

        if(!norm_continuous)
        {
            break;
        }

        const tinyxml2::XMLAttribute* norm_continuous_field = norm_continuous->FindAttribute("field");

        if(!norm_continuous_field)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>&) method.\n"
                   << "Attribute \"field\" in NormContinuous_field element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        if(!outputs_names.contains(std::string(norm_continuous_field->Value())))
        {
            derived_field = derived_field->NextSiblingElement("DerivedField");
            continue;
        }

        const tinyxml2::XMLElement* linear_norm_begin = norm_continuous->FirstChildElement("LinearNorm");
        const tinyxml2::XMLElement* linear_norm_end = norm_continuous->LastChildElement("LinearNorm");

        if(!linear_norm_begin || !linear_norm_end)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>&) method.\n"
                   << "LinearNorm in NormContinuous element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        const tinyxml2::XMLAttribute* attribute_orig_begin = linear_norm_begin->FindAttribute("orig");
        const tinyxml2::XMLAttribute* attribute_orig_end = linear_norm_end->FindAttribute("orig");

        if(!attribute_orig_begin || !attribute_orig_end)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>&) method.\n"
                   << "Attribute \"orig\" in LinearNorm element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        const tinyxml2::XMLAttribute* attribute_norm_begin = linear_norm_begin->FindAttribute("norm");
        const tinyxml2::XMLAttribute* attribute_norm_end = linear_norm_end->FindAttribute("norm");

        if(!attribute_norm_begin || !attribute_norm_end)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>&) method.\n"
                   << "Attribute \"norm\" in LinearNorm element is NULL.\n";

            throw std::logic_error(buffer.str());
        }

        const std::string string_orig_begin(attribute_orig_begin->Value());
        const std::string string_orig_end(attribute_orig_end->Value());
        const std::string string_normalization_range_begin(attribute_norm_begin->Value());
        const std::string string_normalization_range_end(attribute_norm_end->Value());

        if(string_orig_begin == ""
          || string_orig_end == ""
          || string_normalization_range_begin == ""
          || string_normalization_range_end == "" )
        {
            break;
        }

        double orig_begin = atof(string_orig_begin.c_str());
        double orig_end = atof(string_orig_end.c_str());
        double normalization_range_begin = atof(string_normalization_range_begin.c_str());
        double normalization_range_end = atof(string_normalization_range_end.c_str());

        if(orig_begin > orig_end)
        {
            double swap = orig_begin;
            orig_begin = orig_end;
            orig_end = swap;

            swap = normalization_range_begin;
            normalization_range_begin = normalization_range_end;
            normalization_range_end = swap;
        }

        if(orig_begin == orig_end)
        {
            buffer << "OpenNN Exception: UnscalingLayer class.\n"
                   << "void from_PMML(const tinyxml2::XMLElement*, const Vector<std::string>&) method.\n"
                   << "Attributes \"orig\" in LinearNorm elements are equals.\n";

            throw std::logic_error(buffer.str());
        }

        if(normalization_range_begin != normalization_range_end)
        {
            // Set mean and standard deviation

            const double new_data_standard_deviation = (orig_begin - orig_end) / (normalization_range_begin - normalization_range_end);
            const double new_data_mean = orig_begin - normalization_range_begin * new_data_standard_deviation;

            set_mean(i,new_data_mean);
            set_standard_deviation(i,new_data_standard_deviation);


            const double new_min = ((2 * normalization_range_end * orig_begin) + (2 * orig_begin) - (2 * normalization_range_begin * orig_end) - (2 * orig_end)) / (2 * (normalization_range_end - normalization_range_begin));
            double new_max;

            if((normalization_range_begin + 1) != 0)
            {
                new_max = ( (2 * (orig_begin - new_min)) / (normalization_range_begin + 1) ) + new_min;
            }
            else
            {
                new_max = ( (2 * (orig_end - new_min)) / (normalization_range_end + 1) ) + new_min;
            }

            if (fabs(new_min - get_statistics().at(i).minimum) < 1e-5
               && fabs(new_max - get_statistics().at(i).maximum) < 1e-5)
            {
                if(new_unscaling_method == NoUnscaling)
                {
                    new_unscaling_method = MinimumMaximum;
                }
            }
            else
            {
                new_unscaling_method = MeanStandardDeviation;
            }
        }

        derived_field = derived_field->NextSiblingElement("DerivedField");

        i++;
    }

    set_unscaling_method(new_unscaling_method);
}


// std::string write_none_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the unscaling process with the none method.
/// @param inputs_name Name of inputs to the unscaling layer. The size of this vector must be equal to the number of unscaling neurons.
/// @param outputs_name Name of outputs from the unscaling layer. The size of this vector must be equal to the number of unscaling neurons.

std::string UnscalingLayer::write_none_expression(const Vector<std::string>& inputs_name, const Vector<std::string>& outputs_name) const
{

    std::ostringstream buffer;

    buffer.str("");

    buffer << "(" << outputs_name.to_string(",") << ") = (" << inputs_name.to_string(",") << ");\n";

    return(buffer.str());
}


// std::string write_minimum_maximum_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the unscaling process with the minimum and maximum method. 
/// @param inputs_name Name of inputs to the unscaling layer. The size of this vector must be equal to the number of unscaling neurons. 
/// @param outputs_name Name of outputs from the unscaling layer. The size of this vector must be equal to the number of unscaling neurons. 

std::string UnscalingLayer::write_minimum_maximum_expression(const Vector<std::string>& inputs_name, const Vector<std::string>& outputs_name) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    std::ostringstream buffer;

    buffer.precision(10);

    Vector<std::string> expressions(unscaling_neurons_number);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        buffer.str("");
        buffer << "0.5*(" << inputs_name[i] << "+1.0)*(" << statistics[i].maximum << "-" << statistics[i].minimum << ")+" << statistics[i].minimum;

        expressions[i] = buffer.str();
    }

    buffer.str("");

    buffer << "(" << outputs_name.to_string(",") << ") = (" << expressions.to_string(",") << ");\n";

    return(buffer.str());
}


// std::string write_mean_stadard_deviation_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the unscaling process with the mean and standard deviation method. 
/// @param inputs_name Name of inputs to the unscaling layer. The size of this vector must be equal to the number of unscaling neurons. 
/// @param outputs_name Name of outputs from the unscaling layer. The size of this vector must be equal to the number of unscaling neurons. 

std::string UnscalingLayer::write_mean_stadard_deviation_expression(const Vector<std::string>& inputs_name, const Vector<std::string>& outputs_name) const
{
    const size_t unscaling_neurons_number = get_unscaling_neurons_number();

    std::ostringstream buffer;

    buffer.precision(10);

    Vector<std::string> expressions(unscaling_neurons_number);

    for(size_t i = 0; i < unscaling_neurons_number; i++)
    {
        buffer.str("");
        buffer <<   statistics[i].mean << "+" << statistics[i].standard_deviation << "*" << inputs_name[i];

        expressions[i] = buffer.str();
    }

    buffer.str("");

    buffer << "(" << outputs_name.to_string(",") << ") = (" << expressions.to_string(",") << ");\n";

    return(buffer.str());
}


// std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the unscaling process in this layer. 
/// @param inputs_name Name of inputs to the unscaling layer. The size of this vector must be equal to the number of unscaling neurons. 
/// @param outputs_name Name of outputs from the unscaling layer. The size of this vector must be equal to the number of unscaling neurons. 

std::string UnscalingLayer::write_expression(const Vector<std::string>& inputs_name, const Vector<std::string>& outputs_name) const
{
    switch(unscaling_method)
    {
    case NoUnscaling:
    {
        return(write_none_expression(inputs_name, outputs_name));
    }
        break;

    case MinimumMaximum:
    {
        return(write_minimum_maximum_expression(inputs_name, outputs_name));
    }
        break;

    case MeanStandardDeviation:
    {
        return(write_mean_stadard_deviation_expression(inputs_name, outputs_name));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: UnscalingLayer class.\n"
               << "std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const method.\n"
               << "Unknown unscaling method.\n";

        throw std::logic_error(buffer.str());
    }// end default
        break;
    }
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
