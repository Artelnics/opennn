/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   P R O B A B I L I S T I C   L A Y E R   C L A S S                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "probabilistic_layer.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a probabilistic layer object with zero probabilistic neurons.

ProbabilisticLayer::ProbabilisticLayer(void)
{
    set();
}


// PROBABILISTIC NEURONS NUMBER CONSTRUCTOR

/// Probabilistic neurons number constructor. 
/// It creates a probabilistic layer with a given size.
/// @param new_probabilistic_neurons_number Number of neurons in the layer. 

ProbabilisticLayer::ProbabilisticLayer(const size_t& new_probabilistic_neurons_number)
{
    set(new_probabilistic_neurons_number);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing probabilistic layer object. 
/// @param other_probabilistic_layer Probabilistic layer to be copied.

ProbabilisticLayer::ProbabilisticLayer(const ProbabilisticLayer& other_probabilistic_layer)
{
    set(other_probabilistic_layer);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any pointer. 

ProbabilisticLayer::~ProbabilisticLayer(void)
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to this object the members of an existing probabilistic layer object.
/// @param other_probabilistic_layer Probabilistic layer object to be assigned.

ProbabilisticLayer& ProbabilisticLayer::operator = (const ProbabilisticLayer& other_probabilistic_layer)
{
    if(this != &other_probabilistic_layer)
    {
        probabilistic_neurons_number = other_probabilistic_layer.probabilistic_neurons_number;

        probabilistic_method = other_probabilistic_layer.probabilistic_method;

        decision_threshold = other_probabilistic_layer.decision_threshold;

        display = other_probabilistic_layer.display;
    }

    return(*this);
}


// EQUAL TO OPERATOR

// bool operator == (const ProbabilisticLayer&) const method

/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_probabilistic_layer Probabilistic layer to be compared with.

bool ProbabilisticLayer::operator == (const ProbabilisticLayer& other_probabilistic_layer) const
{
    if(probabilistic_neurons_number == other_probabilistic_layer.probabilistic_neurons_number
            && probabilistic_method == other_probabilistic_layer.probabilistic_method
            && decision_threshold == other_probabilistic_layer.decision_threshold
            && display == other_probabilistic_layer.display)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// METHODS

// const size_t& get_outputs_number(void) const method

/// Returns the number of probabilistic neurons in the layer. 

const size_t& ProbabilisticLayer::get_probabilistic_neurons_number(void) const
{
    return(probabilistic_neurons_number);
}


// const double& get_decision_threshold(void) const method

/// Returns the decision threshold.

const double& ProbabilisticLayer::get_decision_threshold(void) const
{
    return(decision_threshold);
}


// const ProbabilisticMethod& get_probabilistic_method(void) const method

/// Returns the method to be used for interpreting the outputs as probabilistic values. 
/// The methods available for that are Binary, Probability, Competitive and Softmax.

const ProbabilisticLayer::ProbabilisticMethod& ProbabilisticLayer::get_probabilistic_method(void) const
{
    return(probabilistic_method);
}


// std::string write_probabilistic_method(void) const method

/// Returns a string with the probabilistic method for the outputs
/// ("Competitive", "Softmax" or "NoProbabilistic").

std::string ProbabilisticLayer::write_probabilistic_method(void) const
{
    if(probabilistic_method == Binary)
    {
        return("Binary");
    }
    else if(probabilistic_method == Probability)
    {
        return("Probability");
    }
    else if(probabilistic_method == Competitive)
    {
        return("Competitive");
    }
    else if(probabilistic_method == Softmax)
    {
        return("Softmax");
    }
    else if(probabilistic_method == NoProbabilistic)
    {
        return("NoProbabilistic");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "std::string write_probabilistic_method(void) const method.\n"
               << "Unknown probabilistic method.\n";

        throw std::logic_error(buffer.str());
    }
}


// std::string write_probabilistic_method_text(void) const method

/// Returns a string with the probabilistic method for the outputs to be included in some text
/// ("competitive", "softmax" or "no probabilistic").

std::string ProbabilisticLayer::write_probabilistic_method_text(void) const
{
    if(probabilistic_method == Binary)
    {
        return("binary");
    }
    else if(probabilistic_method == Probability)
    {
        return("probability");
    }
    else if(probabilistic_method == Competitive)
    {
        return("competitive");
    }
    else if(probabilistic_method == Softmax)
    {
        return("softmax");
    }
    else if(probabilistic_method == NoProbabilistic)
    {
        return("no probabilistic");
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "std::string write_probabilistic_method_text(void) const method.\n"
               << "Unknown probabilistic method.\n";

        throw std::logic_error(buffer.str());
    }
}


// const bool& get_display(void) const method

/// Returns true if messages from this class are to be displayed on the screen, or false if messages 
/// from this class are not to be displayed on the screen.

const bool& ProbabilisticLayer::get_display(void) const
{
    return(display);
}


// void set(void) method

/// Sets a probabilistic layer with zero probabilistic neurons.
/// It also sets the rest of members to their default values. 

void ProbabilisticLayer::set(void)
{
    probabilistic_neurons_number = 0;

    set_default();
}


// void set(const size_t&) method

/// Resizes the size of the probabilistic layer. 
/// It also sets the rest of class members to their default values.
/// @param new_probabilistic_neurons_number New size for the probabilistic layer. 

void ProbabilisticLayer::set(const size_t& new_probabilistic_neurons_number)
{
    probabilistic_neurons_number = new_probabilistic_neurons_number;

    set_default();
}


// void set(const ProbabilisticLayer&) method

/// Sets this object to be equal to another object of the same class.
/// @param other_probabilistic_layer Probabilistic layer object to be copied. 

void ProbabilisticLayer::set(const ProbabilisticLayer& other_probabilistic_layer)
{
    set_default();

    probabilistic_neurons_number = other_probabilistic_layer.probabilistic_neurons_number;

    probabilistic_method = other_probabilistic_layer.probabilistic_method;

    decision_threshold = other_probabilistic_layer.decision_threshold;

    display = other_probabilistic_layer.display;
}


// void set_probabilistic_neurons_number(const size_t&) method

/// Resizes the size of the probabilistic layer. 
/// @param new_probabilistic_neurons_number New size for the probabilistic layer. 

void ProbabilisticLayer::set_probabilistic_neurons_number(const size_t& new_probabilistic_neurons_number)
{
    probabilistic_neurons_number = new_probabilistic_neurons_number;
}


// void set_decision_threshold(const double&) method

/// Sets a new threshold value for discriminating between two classes.
/// @param new_decision_threshold New discriminating value. It must be comprised between 0 and 1.

void ProbabilisticLayer::set_decision_threshold(const double& new_decision_threshold)
{
#ifdef __OPENNN_DEBUG__

    if(new_decision_threshold <= 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_decision_threshold(const double&) method.\n"
               << "Decision threshold (" << decision_threshold << ") must be greater than zero.\n";

        throw std::logic_error(buffer.str());
    }
    else if(new_decision_threshold >= 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_decision_threshold(const double&) method.\n"
               << "Decision threshold (" << decision_threshold << ") must be less than one.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    decision_threshold = new_decision_threshold;
}


// void set_default(void) method

/// Sets the members to their default values:
/// <ul>
/// <li> Probabilistic method: Softmax.
/// <li> Display: True. 
/// </ul>

void ProbabilisticLayer::set_default(void)
{
    probabilistic_method = Softmax;

    decision_threshold = 0.5;

    display = true;
}


// void set_probablistic_method(const ProbabilisticMethod&) method

/// Sets the chosen method for probabilistic postprocessing. 
/// Current probabilistic methods include Binary, Probability, Competitive and Softmax.
/// @param new_probabilistic_method Method for interpreting the outputs as probabilistic values. 

void ProbabilisticLayer::set_probabilistic_method(const ProbabilisticMethod& new_probabilistic_method)
{
    probabilistic_method = new_probabilistic_method;
}


// void set_probabilistic_method(const std::string&) method

/// Sets a new method for probabilistic processing from a string with the name. 
/// Current probabilistic methods include Competitive and Softmax. 
/// @param new_probabilistic_method Method for interpreting the outputs as probabilistic values. 

void ProbabilisticLayer::set_probabilistic_method(const std::string& new_probabilistic_method)
{
    if(new_probabilistic_method == "Binary")
    {
        set_probabilistic_method(Binary);
    }
    else if(new_probabilistic_method == "Probability")
    {
        set_probabilistic_method(Probability);
    }
    else if(new_probabilistic_method == "Competitive")
    {
        set_probabilistic_method(Competitive);
    }
    else if(new_probabilistic_method == "Softmax")
    {
        set_probabilistic_method(Softmax);
    }
    else if(new_probabilistic_method == "NoProbabilistic")
    {
        set_probabilistic_method(NoProbabilistic);
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void set_probabilistic_method(const std::string&) method.\n"
               << "Unknown probabilistic method: " << new_probabilistic_method << ".\n";

        throw std::logic_error(buffer.str());
    }
}


// void set_display(const bool&) method

/// Sets a new display value. 
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void ProbabilisticLayer::set_display(const bool& new_display)
{
    display = new_display;
}


// void prune_probabilistic_neuron(void) method

/// Removes a probabilistic neuron from the probabilistic layer.
/// As probabilistic neurons do not have any parameter, it does not matter which one is pruned.

void ProbabilisticLayer::prune_probabilistic_neuron(void)
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    if(probabilistic_neurons_number == 0)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void prune_probabilistic_neuron(void) method.\n"
               << "Number of probabilistic neurons is zero.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    probabilistic_neurons_number--;
}


// void initialize_random(void) method

/// Initializes at random the probabilistic method.

void ProbabilisticLayer::initialize_random(void)
{
    // Probabilistic method

    switch(rand()%5)
    {
    case 0:
    {
        probabilistic_method = Binary;
    }
        break;

    case 1:
    {
        probabilistic_method = Probability;
    }
        break;

    case 2:
    {
        probabilistic_method = Competitive;
    }
        break;

    case 3:
    {
        probabilistic_method = Softmax;
    }
        break;

    case 4:
    {
        probabilistic_method = NoProbabilistic;
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void initialize_random(void) method.\n"
               << "Unknown probabilistic method.\n";

        throw std::logic_error(buffer.str());
    }
        break;
    }
}


// Vector<double> calculate_outputs(const Vector<double>&) const method

/// This method processes the input to the probabilistic layer in order to obtain a set of outputs which can be interpreted as probabilities. 
/// This posprocessing is performed according to the probabilistic method to be used. 
/// @param inputs Set of inputs to the probabilistic layer.

Vector<double> ProbabilisticLayer::calculate_outputs(const Vector<double>& inputs) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = inputs.size();

    if(size != probabilistic_neurons_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
               << "Size must be equal to number of probabilistic neurons.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    switch(probabilistic_method)
    {
    case Binary:
    {
        return(calculate_binary_output(inputs));
    }
        break;

    case Probability:
    {
        return(calculate_probability_output(inputs));
    }
        break;

    case Competitive:
    {
        return(calculate_competitive_output(inputs));
    }
        break;

    case Softmax:
    {
        return(calculate_softmax_output(inputs));
    }
        break;

    case NoProbabilistic:
    {
        return(calculate_no_probabilistic_output(inputs));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
               << "Unknown probabilistic method.\n";

        throw std::logic_error(buffer.str());
    }// end default
        break;

    }// end switch
}  


// Matrix<double> calculate_Jacobian(const Vector<double>&) const method

/// Returns the partial derivatives of the outputs from the probabilistic layer with respect to its inputs,
/// depending on the probabilistic method to be used.
/// This quantity is the Jacobian matrix of the probabilistic function. 
/// @param inputs Inputs to the probabilistic layer.

Matrix<double> ProbabilisticLayer::calculate_Jacobian(const Vector<double>& inputs) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = inputs.size();

    if(size != probabilistic_neurons_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Matrix<double> calculate_Jacobian(const Vector<double>&) const method.\n"
               << "Size must be equal to number of probabilistic neurons.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    switch(probabilistic_method)
    {
    case Binary:
    {
        return(calculate_binary_Jacobian(inputs));
    }
        break;

    case Probability:
    {
        return(calculate_probability_Jacobian(inputs));
    }
        break;

    case Competitive:
    {
        return(calculate_competitive_Jacobian(inputs));
    }
        break;

    case Softmax:
    {
        return(calculate_softmax_Jacobian(inputs));
    }
        break;

    case NoProbabilistic:
    {
        return(calculate_no_probabilistic_Jacobian(inputs));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Matrix<double> calculate_Jacobian(const Vector<double>&) const method.\n"
               << "Unknown probabilistic method.\n";

        throw std::logic_error(buffer.str());
    }// end default
        break;

    }// end switch
}


// Vector< Matrix<double> > calculate_Hessian_form(const Vector<double>&) const method

/// Calculates the Hessian form of the probabilistic layer. 
/// This is a vector of matrices. 
/// The elements contain second partial derivatives of the outputs from the layer with resptect to the inputs to it.

Vector< Matrix<double> > ProbabilisticLayer::calculate_Hessian_form(const Vector<double>& inputs) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = inputs.size();

    if(size != probabilistic_neurons_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Matrix<double> calculate_Hessian_form(const Vector<double>&) const method.\n"
               << "Size must be equal to number of probabilistic neurons.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    switch(probabilistic_method)
    {
    case Binary:
    {
        return(calculate_binary_Hessian_form(inputs));
    }
        break;

    case Probability:
    {
        return(calculate_probability_Hessian_form(inputs));
    }
        break;

    case Competitive:
    {
        return(calculate_competitive_Hessian_form(inputs));
    }
        break;

    case Softmax:
    {
        return(calculate_softmax_Hessian_form(inputs));
    }
        break;

    case NoProbabilistic:
    {
        return(calculate_no_probabilistic_Hessian_form(inputs));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Matrix<double> calculate_Hessian_form(const Vector<double>&) const method.\n"
               << "Unknown probabilistic method.\n";

        throw std::logic_error(buffer.str());
    }// end default
        break;

    }// end switch
}


// Vector<double> calculate_binary_output(const Vector<double>&) const method

/// Returns the output value from this layer as a binary value (0 or 1).
/// The size of the probabilistic layer must be 1.
/// @param inputs Vector of input values. The size here must be also 1.

Vector<double> ProbabilisticLayer::calculate_binary_output(const Vector<double>& inputs) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    if(probabilistic_neurons_number != 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Vector<double> calculate_binary_output(const Vector<double>&) const method.\n"
               << "The number of probabilistic neurons number must be 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    Vector<double> outputs(1);

    if(inputs[0] < decision_threshold)
    {
        outputs[0] = 0.0;
    }
    else
    {
        outputs[0] = 1.0;
    }

    return(outputs);
}


// Matrix<double> calculate_binary_Jacobian(const Vector<double>&) const method

/// This method throws an exception, since the threshold function is not derivable.

Matrix<double> ProbabilisticLayer::calculate_binary_Jacobian(const Vector<double>&) const
{
    std::ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Matrix<double> calculate_binary_Jacobian(const Vector<double>&) const method.\n"
           << "The binary function is not derivable.\n";

    throw std::logic_error(buffer.str());
}


// Vector< Matrix<double> > calculate_binary_Hessian_form(const Vector<double>&) const method

/// This method throws an exception, since the threshold function is not derivable.

Vector< Matrix<double> > ProbabilisticLayer::calculate_binary_Hessian_form(const Vector<double>&) const
{
    std::ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Vector< Matrix<double> > calculate_binary_Hessian_form(const Vector<double>&) const method.\n"
           << "The binary function is not derivable.\n";

    throw std::logic_error(buffer.str());
}


// Vector<double> calculate_probability_output(const Vector<double>&) const method

/// @todo Check that the input values fall between 0 and 1.
/// @param inputs Vector of input values to the probabilistic layer.

Vector<double> ProbabilisticLayer::calculate_probability_output(const Vector<double>& inputs) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    if(probabilistic_neurons_number != 1)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "Vector<double> calculate_probability_output(const Vector<double>&) const method.\n"
               << "The number of probabilistic neurons number must be 1.\n";

        throw std::logic_error(buffer.str());
    }

#endif

    const Vector<double> outputs(inputs);

    return(outputs);
}


// Matrix<double> calculate_probability_Jacobian(const Vector<double>&) const method

/// @todo Check that the input values fall between 0 and 1.

Matrix<double> ProbabilisticLayer::calculate_probability_Jacobian(const Vector<double>&) const
{
    const Matrix<double> Jacobian(1, 1, 1.0);

    return(Jacobian);
}


// Vector< Matrix<double> > calculate_probability_Hessian_form(const Vector<double>&) const method

/// @todo Check that the input values fall between 0 and 1.

Vector< Matrix<double> > ProbabilisticLayer::calculate_probability_Hessian_form(const Vector<double>&) const
{
    Vector< Matrix<double> > Hessian_form(1);

    Hessian_form[0].set(1, 1, 0.0);

    return(Hessian_form);
}


// Vector<double> calculate_competitive_output(const Vector<double>&) const method

/// Returns the outputs from the layer for given inputs when the probabilistic method is the competitive. 
/// @param inputs Vector of input values to the probabilistic layer. 

Vector<double> ProbabilisticLayer::calculate_competitive_output(const Vector<double>& inputs) const
{
    return(inputs.calculate_competitive());
}


// Matrix<double> calculate_competitive_Jacobian(const Vector<double>&) const method

/// This method throws an exception, since the competitive function is not derivable. 

Matrix<double> ProbabilisticLayer::calculate_competitive_Jacobian(const Vector<double>&) const
{
    std::ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Matrix<double> calculate_competitive_Jacobian(const Vector<double>&) const method.\n"
           << "The competitive function is not derivable.\n";

    throw std::logic_error(buffer.str());
}


// Vector< Matrix<double> > calculate_competitive_Hessian_form(const Vector<double>&) const method

/// This method throws an exception, since the competitive function is not derivable. 

Vector< Matrix<double> > ProbabilisticLayer::calculate_competitive_Hessian_form(const Vector<double>&) const
{
    std::ostringstream buffer;

    buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
           << "Vector< Matrix<double> > calculate_competitive_Hessian_form(const Vector<double>&) const method.\n"
           << "The competitive function is not derivable.\n";

    throw std::logic_error(buffer.str());
}


// Vector<double> calculate_softmax_output(const Vector<double>&) const method

/// Returns the outputs of the softmax function for given inputs. 
/// @param inputs Input values to the probabilistic layer. 

Vector<double> ProbabilisticLayer::calculate_softmax_output(const Vector<double>& inputs) const
{
    return(inputs.calculate_softmax());
}


// Matrix<double> calculate_softmax_Jacobian(const Vector<double>&) const method

/// Returns the partial derivatives of the softmax outputs with respect to the inputs. 
/// @param inputs Input values to the probabilistic layer. 

Matrix<double> ProbabilisticLayer::calculate_softmax_Jacobian(const Vector<double>& inputs) const
{
    Matrix<double> probabilistic_Jacobian(probabilistic_neurons_number, probabilistic_neurons_number);

    const Vector<double> outputs = inputs.calculate_softmax();

    for(size_t i = 0; i < probabilistic_neurons_number; i++)
    {
        for(size_t j = 0; j < probabilistic_neurons_number; j++)
        {
            if(i == j)
            {
                probabilistic_Jacobian(i,i) = outputs[i]*(1.0 - outputs[i]);
            }
            else
            {
                probabilistic_Jacobian(i,j) = -outputs[i]*outputs[j];
            }
        }
    }

    return(probabilistic_Jacobian);
}


// Vector< Matrix<double> > calculate_softmax_Hessian_form(const Vector<double>&) const method

/// Returns the second partial derivatives of the softmax outputs with respect to the inputs,
/// in the so called Hessian form. 
/// @todo

Vector< Matrix<double> > ProbabilisticLayer::calculate_softmax_Hessian_form(const Vector<double>& inputs) const
{
    Vector< Matrix<double> > Hessian_form(probabilistic_neurons_number);

    for(size_t i = 0; i < probabilistic_neurons_number; i++)
    {
        Hessian_form[i].set(probabilistic_neurons_number, probabilistic_neurons_number);
    }

    const Vector<double> outputs = inputs.calculate_softmax();

    for(size_t i = 0; i < probabilistic_neurons_number; i++)
    {
        for(size_t j = 0; j < probabilistic_neurons_number; j++)
        {
            for(size_t k = 0; k < probabilistic_neurons_number; k++)
            {
                if(j == i && j == k && i == k)
                {
                    Hessian_form[i](j, k) = outputs[i]*(1 - outputs[i] - 2*outputs[i]*(1 - outputs[i]));
                }
                else if(j == i && j != k && i != k)
                {
                    Hessian_form[i](j, k) = -outputs[i]*outputs[k]*(1 - 2*outputs[i]);
                }
                else if(j != i && j == k && i != k)
                {
                    Hessian_form[i](j, k) = outputs[i]*outputs[j]*outputs[k] - outputs[i]*outputs[j]*(1 - outputs[j]);
                }
                else if(j != i && i == k && j != k)
                {
                    Hessian_form[i](j, k) = outputs[i]*outputs[j]*outputs[k] - outputs[i]*outputs[j]*(1 - outputs[i]);
                }
                else if(j != i && i != k && j != k)
                {
                    Hessian_form[i](j, k) = 2*outputs[i]*outputs[j]*outputs[k];
                }
            }
        }
    }

    return(Hessian_form);
}


// Vector<double> calculate_no_probabilistic_output(const Vector<double>&) const method

/// Returns the outputs of the no probabilistic function for given inputs.
/// This is just the identity function.
/// @param inputs Input values to the probabilistic layer.

Vector<double> ProbabilisticLayer::calculate_no_probabilistic_output(const Vector<double>& inputs) const
{
    return(inputs);
}


// Matrix<double> calculate_no_probabilistic_Jacobian(const Vector<double>&) const method

/// Returns the partial derivatives of the no probabilistic outputs with respect to the inputs.
/// This is just the identity matrix of size the number of probabilistic neurons.

Matrix<double> ProbabilisticLayer::calculate_no_probabilistic_Jacobian(const Vector<double>&) const
{
    Matrix<double> Jacobian(probabilistic_neurons_number, probabilistic_neurons_number);

    Jacobian.initialize_identity();

    return(Jacobian);
}


// Vector< Matrix<double> > calculate_no_probabilistic_Hessian_form(const Vector<double>&) const method

/// Returns the second partial derivatives of the no probabilistic outputs with respect to the inputs,
/// in the so called Hessian form.
/// @todo

Vector< Matrix<double> > ProbabilisticLayer::calculate_no_probabilistic_Hessian_form(const Vector<double>&) const
{
    Vector< Matrix<double> > Hessian_form;

    return(Hessian_form);
}


// std::string to_string(void) const method

/// Returns a string representation of the current probabilistic layer object. 

std::string ProbabilisticLayer::to_string(void) const
{
    std::ostringstream buffer;

    buffer << "Probabilistic layer\n"
           << "Probabilistic neurons number: " << probabilistic_neurons_number << "\n"
           << "Probabilistic method: " << write_probabilistic_method() << "\n";
    //<< "Display: "<< display << "\n";

    return(buffer.str());
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Serializes the probabilistic layer object into a XML document of the TinyXML library. 
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* ProbabilisticLayer::to_XML(void) const
{
    std::ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    tinyxml2::XMLElement* root_element = document->NewElement("ProbabilisticLayer");

    document->InsertFirstChild(root_element);

    tinyxml2::XMLElement* element = NULL;
    tinyxml2::XMLText* text = NULL;

    // Probabilistic neurons number
    {
        element = document->NewElement("ProbabilisticNeuronsNumber");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << probabilistic_neurons_number;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Probabilistic method
    {
        element = document->NewElement("ProbabilisticMethod");
        root_element->LinkEndChild(element);

        text = document->NewText(write_probabilistic_method().c_str());
        element->LinkEndChild(text);
    }

    // Probabilistic neurons number
    {
        element = document->NewElement("DecisionThreshold");
        root_element->LinkEndChild(element);

        buffer.str("");
        buffer << decision_threshold;

        text = document->NewText(buffer.str().c_str());
        element->LinkEndChild(text);
    }

    // Display
    //   {
    //      element = document->NewElement("Display");
    //      root_element->LinkEndChild(element);

    //      buffer.str("");
    //      buffer << display;

    //      text = document->NewText(buffer.str().c_str());
    //      element->LinkEndChild(text);
    //   }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the probabilistic layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void ProbabilisticLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("ProbabilisticLayer");

    // Probabilistic neurons number

    file_stream.OpenElement("ProbabilisticNeuronsNumber");

    buffer.str("");
    buffer << probabilistic_neurons_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Probabilistic method

    file_stream.OpenElement("ProbabilisticMethod");

    file_stream.PushText(write_probabilistic_method().c_str());

    file_stream.CloseElement();

    // Probabilistic neurons number

    file_stream.OpenElement("DecisionThreshold");

    buffer.str("");
    buffer << decision_threshold;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this probabilistic layer object.
/// @param document XML document containing the member data.

void ProbabilisticLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    std::ostringstream buffer;

    const tinyxml2::XMLElement* probabilistic_layer_element = document.FirstChildElement("ProbabilisticLayer");

    if(!probabilistic_layer_element)
    {
        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Probabilistic layer element is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Probabilistic neurons number
    {
        const tinyxml2::XMLElement* element = probabilistic_layer_element->FirstChildElement("ProbabilisticNeuronsNumber");

        if(element)
        {
            const char* text = element->GetText();

            if(text)
            {
                try
                {
                    set_probabilistic_neurons_number(atoi(text));
                }
                catch(const std::logic_error& e)
                {
                    std::cout << e.what() << std::endl;
                }
            }
        }
    }

    // Probabilistic method
    {
        const tinyxml2::XMLElement* element = probabilistic_layer_element->FirstChildElement("ProbabilisticMethod");

        if(element)
        {
            const char* text = element->GetText();

            if(text)
            {
                try
                {
                    std::string new_probabilistic_method(text);

                    set_probabilistic_method(new_probabilistic_method);
                }
                catch(const std::logic_error& e)
                {
                    std::cout << e.what() << std::endl;
                }
            }
        }
    }

    // Decision threshold
    {
        const tinyxml2::XMLElement* element = probabilistic_layer_element->FirstChildElement("DecisionThreshold");

        if(element)
        {
            const char* text = element->GetText();

            if(text)
            {
                try
                {
                    set_decision_threshold(atof(text));
                }
                catch(const std::logic_error& e)
                {
                    std::cout << e.what() << std::endl;
                }
            }
        }
    }

    // Display
    {
        const tinyxml2::XMLElement* display_element = probabilistic_layer_element->FirstChildElement("Display");

        if(display_element)
        {
            std::string new_display_string = display_element->GetText();

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
}

// std::string write_binary_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the binary probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

std::string ProbabilisticLayer::write_binary_expression(const Vector<std::string>& inputs_names, const Vector<std::string>& outputs_names) const
{
    std::ostringstream buffer;

    buffer.str("");

    buffer << "(" << outputs_names.to_string(",") << ") = Binary(" << inputs_names.to_string(",") << ");\n";

    return(buffer.str());
}


// std::string write_probability_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the probability outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

std::string ProbabilisticLayer::write_probability_expression(const Vector<std::string>& inputs_names, const Vector<std::string>& outputs_names) const
{
    std::ostringstream buffer;

    buffer << "(" << outputs_names.to_string(",") << ") = Probability(" << inputs_names.to_string(",") << ");\n";

    return(buffer.str());
}



// std::string write_competitive_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the competitive probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

std::string ProbabilisticLayer::write_competitive_expression(const Vector<std::string>& inputs_names, const Vector<std::string>& outputs_names) const
{
    std::ostringstream buffer;

    buffer << "(" << outputs_names.to_string(",") << ") = Competitive(" << inputs_names.to_string(",") << ");\n";

    return(buffer.str());
}


// std::string write_softmax_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the softmax probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

std::string ProbabilisticLayer::write_softmax_expression(const Vector<std::string>& inputs_names, const Vector<std::string>& outputs_names) const
{
    std::ostringstream buffer;

    buffer << "(" << outputs_names.to_string(",") << ") = Softmax(" << inputs_names.to_string(",") << ");\n";

    return(buffer.str());
}


// std::string write_no_probabilistic_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the no probabilistic outputs function.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

std::string ProbabilisticLayer::write_no_probabilistic_expression(const Vector<std::string>& inputs_names, const Vector<std::string>& outputs_names) const
{
    std::ostringstream buffer;

    buffer << "(" << outputs_names.to_string(",") << ") = (" << inputs_names.to_string(",") << ");\n";

    return(buffer.str());
}


// std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const method

/// Returns a string with the expression of the probabilistic outputs function,
/// depending on the probabilistic method to be used.
/// @param inputs_names Names of inputs to the probabilistic layer.
/// @param outputs_names Names of outputs to the probabilistic layer.

std::string ProbabilisticLayer::write_expression(const Vector<std::string>& inputs_names, const Vector<std::string>& outputs_names) const
{
    switch(probabilistic_method)
    {
    case Binary:
    {
        return(write_binary_expression(inputs_names, outputs_names));
    }
        break;

    case Probability:
    {
        return(write_probability_expression(inputs_names, outputs_names));
    }
        break;

    case Competitive:
    {
        return(write_competitive_expression(inputs_names, outputs_names));
    }
        break;

    case Softmax:
    {
        return(write_softmax_expression(inputs_names, outputs_names));
    }
        break;

    case NoProbabilistic:
    {
        return(write_no_probabilistic_expression(inputs_names, outputs_names));
    }
        break;

    default:
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: ProbabilisticLayer class.\n"
               << "std::string write_expression(const Vector<std::string>&, const Vector<std::string>&) const method.\n"
               << "Unknown probabilistic method.\n";

        throw std::logic_error(buffer.str());
    }// end default
        break;
    }// end switch
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
