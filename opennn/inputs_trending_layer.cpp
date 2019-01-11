/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   I N P U T S   T R E N D I N G   L A Y E R   C L A S S                                                      */
/*                                                                                                              */
/*   Patricia Garcia                                                                                            */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   patriciagarcia@artelnics.com                                                                               */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "inputs_trending_layer.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.
/// It creates an inputs trending layer object with zero inputs trending neurons.

InputsTrendingLayer::InputsTrendingLayer()
{
   set();
}


// INPUTS TRENDING NEURONS NUMBER CONSTRUCTOR

/// Inputs trending neurons number constructor.
/// It creates an inputs trending layer with a given size.
/// @param inputs_trending_neurons_number Number of inputs trending neurons in the layer.

InputsTrendingLayer::InputsTrendingLayer(const size_t& inputs_trending_neurons_number)
{
   set(inputs_trending_neurons_number);

   set_default();
}


// XML CONSTRUCTOR

/// XML constructor.
/// It creates an inputs trending layer and loads its members from a XML document.
/// @param inputs_trending_layer_document TinyXML document with the member data.

InputsTrendingLayer::InputsTrendingLayer(const tinyxml2::XMLDocument& inputs_trending_layer_document)
{
   set(inputs_trending_layer_document);
}


// COPY CONSTRUCTOR

/// Copy constructor.
/// It creates a copy of an existing inputs trending layer object.
/// @param other_inputs_trending_layer Inputs trending layer to be copied.

InputsTrendingLayer::InputsTrendingLayer(const InputsTrendingLayer& other_inputs_trending_layer)
{
   set(other_inputs_trending_layer);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any pointer.

InputsTrendingLayer::~InputsTrendingLayer()
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator.
/// It assigns to this object the members of an existing inputs trending layer object.
/// @param other_inputs_trending_layer Inputs trending layer object to be assigned.

InputsTrendingLayer& InputsTrendingLayer::operator = (const InputsTrendingLayer& other_inputs_trending_layer)
{
   if(this != &other_inputs_trending_layer)
   {
      display = other_inputs_trending_layer.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator.
/// It compares this object with another object of the same class.
/// It returns true if the members of the two objects have the same value, and false otherwise.
/// @param other_inputs_trending_layer Inputs trending layer to be compared with.

bool InputsTrendingLayer::operator == (const InputsTrendingLayer& other_inputs_trending_layer) const
{
    if(get_inputs_trending_neurons_number() != other_inputs_trending_layer.get_inputs_trending_neurons_number())
    {
       return(false);
    }
    else if(get_inputs_trending_neurons_number() == other_inputs_trending_layer.get_inputs_trending_neurons_number())
    {
        if(display != other_inputs_trending_layer.display)
        {
            return(false);
        }

        for(size_t i = 0; i < get_inputs_trending_neurons_number(); i++)
        {
            if(fabs(get_intercept(i) - other_inputs_trending_layer.get_intercept(i)) > numeric_limits<double>::epsilon()
                    || fabs(get_slope(i) - other_inputs_trending_layer.get_slope(i)) > numeric_limits<double>::epsilon()
                    || fabs(get_correlation(i) - other_inputs_trending_layer.get_correlation(i)) > numeric_limits<double>::epsilon())
            {
                return(false);
            }
        }
    }

    return(true);
}


/// Returns true if the size of the layer is zero, and false otherwise.

bool InputsTrendingLayer::is_empty() const
{
   if(get_inputs_trending_neurons_number() == 0)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


/// Returns the method used for inputs trending layer.

const InputsTrendingLayer::InputsTrendingMethod& InputsTrendingLayer::get_inputs_trending_method() const
{
    return(inputs_trending_method);
}


/// Returns a string with the name of the method used for inputs trending layer.

string InputsTrendingLayer::write_inputs_trending_method() const
{
    if(inputs_trending_method == NoTrending)
    {
        return("NoTrending");

    }
    else if(inputs_trending_method == Linear)
    {
        return("Linear");
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
               << "string write_inputs_trending_method() const method.\n"
               << "Unknown inputs trending method.\n";

        throw logic_error(buffer.str());
    }
}


/// Returns the number of inputs trending neurons in the layer.

size_t InputsTrendingLayer::get_inputs_trending_neurons_number() const
{
   return(inputs_trends.size());
}


/// Returns the intercepts values of all the inputs trending neurons in the layer.

Vector<double> InputsTrendingLayer::get_intercepts() const
{
    const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   Vector<double> intercepts(inputs_trending_neurons_number);

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
       intercepts[i] = inputs_trends[i].intercept;
   }

   return(intercepts);
}


/// Returns the intercept of a single inputs trending neuron.
/// @param i Index of inputs trending neuron.

double InputsTrendingLayer::get_intercept(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   if(i >= inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "double get_intercept(const size_t&) const method.\n"
             << "Index must be less than number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   return(inputs_trends[i].intercept);
}


/// Returns the slopes values of all the inputs trending neurons in the layer.

Vector<double> InputsTrendingLayer::get_slopes() const
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   Vector<double> slopes(inputs_trending_neurons_number);

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
       slopes[i] = inputs_trends[i].slope;
   }

   return(slopes);
}


/// Returns the slope value of a single inputs trending neuron.
/// @param i Index of inputs trending neuron.

double InputsTrendingLayer::get_slope(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   if(inputs_trending_neurons_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "double get_slope(const size_t&) const method.\n"
             << "Number of inputs trending neurons is zero.\n";

      throw logic_error(buffer.str());
   }
   else if(i >= inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "double get_slope(const size_t&) const method.\n"
             << "Index must be less than number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   return(inputs_trends[i].slope);
}


/// Returns the correlations values of all the inputs trending neurons in the layer.

Vector<double> InputsTrendingLayer::get_correlations() const
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   Vector<double> correlations(inputs_trending_neurons_number);

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
       correlations[i] = inputs_trends[i].correlation;
   }

   return(correlations);
}


/// Returns the correlation value of a single inputs trending neuron.
/// @param i Index of inputs trending neuron.

double InputsTrendingLayer::get_correlation(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   if(inputs_trending_neurons_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "double get_correlation(const size_t&) const method.\n"
             << "Number of inputs trending neurons is zero.\n";

      throw logic_error(buffer.str());
   }
   else if(i >= inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "double get_correlation(const size_t&) const method.\n"
             << "Index must be less than number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   return(inputs_trends[i].correlation);
}


/// Returns the trend value of each inputs trending neuron.

Vector< LinearRegressionParameters<double> > InputsTrendingLayer::get_inputs_trends() const
{
    return(inputs_trends);
}


/// Sets the number of inputs trending neurons to be zero.
/// It also sets the rest of members to their default values.

void InputsTrendingLayer::set()
{
   inputs_trends.set();

   set_default();
}


/// Resizes the inputs trending layer.
/// It also sets the rest of members to their default values.
/// @param new_trending_neurons_number Size of the inputs trending layer.

void InputsTrendingLayer::set(const size_t& new_inputs_trending_neurons_number)
{
   inputs_trends.set(new_inputs_trending_neurons_number);

   set_default();
}


/// Sets the inputs trending layer members from a XML document.
/// @param inputs_trending_layer_document Pointer to a TinyXML document containing the member data.

void InputsTrendingLayer::set(const tinyxml2::XMLDocument& inputs_trending_layer_document)
{
    set_default();

   from_XML(inputs_trending_layer_document);
}


/// Sets the members of this object to be the members of another object of the same class.
/// @param other_inputs_trending_layer Object to be copied.

void InputsTrendingLayer::set(const InputsTrendingLayer& other_inputs_trending_layer)
{
   inputs_trends = other_inputs_trending_layer.inputs_trends;

   display = other_inputs_trending_layer.display;
}


/// Sets a new inputs trending method.
/// @param new_method New inputs trending method.

void InputsTrendingLayer::set_inputs_trending_method(const InputsTrendingMethod& new_method)
{
    inputs_trending_method = new_method;
}


/// Sets a new inputs trending method.
/// @param new_method_string New inputs trending method string.

void InputsTrendingLayer::set_inputs_trending_method(const string& new_method_string)
{
    if(new_method_string == "NoTrending")
    {
        inputs_trending_method = NoTrending;
    }
    else if(new_method_string == "Linear")
    {
        inputs_trending_method = Linear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
               << "void set_inputs_trending_method(const string&) method.\n"
               << "Unknown inputs trending method: " << new_method_string << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets the inputs trending layer members to the given values.

void InputsTrendingLayer::set_inputs_trends(const Vector< LinearRegressionParameters<double> >& new_inputs_trends)
{
     const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t size = new_inputs_trends.size();

    if(size != inputs_trending_neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
              << "void set_inputs_trends(const Vector< Vector<double> >&) method.\n"
              << "Size of the new inputs trends must be equal to the number of inputs trending neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Set inputs trends

    for(size_t i = 0; i < inputs_trending_neurons_number; i++)
    {
        inputs_trends[i] = new_inputs_trends[i];
    }
}


/// Sets new intercepts for all the neurons in the layer.
/// @param new_intercepts New set of intercepts for the inputs trending neurons.

void InputsTrendingLayer::set_intercepts(const Vector<double>& new_intercepts)
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_intercepts.size() != inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "void set_intercepts(const Vector<double>&) method.\n"
             << "Size must be equal to number of inputs trending neurons number.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set intercepts

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
       inputs_trends[i].intercept = new_intercepts[i];
   }
}


/// Sets a new intercept for a single inputs trending neuron.
/// @param index Index of inputs trending neuron.
/// @param new_intercept New intercept for the neuron with index i.

void InputsTrendingLayer::set_intercept(const size_t& index, const double& new_intercept)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   if(index >= inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "void set_intercept(const size_t&, const double&) method.\n"
             << "Index of inputs trending neuron must be less than number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set intercept of a single neuron

   inputs_trends[index].intercept = new_intercept;
}


/// Sets new slopes for all the inputs trending neurons.
/// @param new?slopes New set of slopes for the layer.

void InputsTrendingLayer::set_slopes(const Vector<double>& new_slopes)
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_slopes.size() != inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "void set_slopes(const Vector<double>&) method.\n"
             << "Size must be equal to number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set slopes

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
       inputs_trends[i].slope = new_slopes[i];
   }
}


/// Sets a new slope for a single inputs trending neuron.
/// @param index Index of inputs trending neuron.
/// @param new_slope New slope for the inputs trending neuron with that index.

void InputsTrendingLayer::set_slope(const size_t& index, const double& new_slope)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   if(index >= inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "void set_slope(const size_t&, const double&) method.\n"
             << "Index of inputs trending neuron must be less than number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set slope of a single neuron

   inputs_trends[index].slope = new_slope;
}


/// Sets new correlations for all the inputs trending neurons.
/// @param new_correlations New set of correlations for the layer.

void InputsTrendingLayer::set_correlations(const Vector<double>& new_correlations)
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_correlations.size() != inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "void set_correlations(const Vector<double>&) method.\n"
             << "Size must be equal to number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set correlations

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
       inputs_trends[i].correlation = new_correlations[i];
   }
}


/// Sets a new correlation for a single inputs trending neuron.
/// @param index Index of inputs trending neuron.
/// @param new_correlation New correlation for the inputs trending neuron with that index.

void InputsTrendingLayer::set_correlation(const size_t& index, const double& new_correlation)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   if(index >= inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "void set_correlation(const size_t&, const double&) method.\n"
             << "Index of inputs trending neuron must be less than number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set correlation of a single neuron

   inputs_trends[index].correlation = new_correlation;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void InputsTrendingLayer::set_display(const bool& new_display)
{
   display = new_display;
}


/// Sets the members to their default values:
/// <ul>
/// <li> Display: True.
/// </ul>

void InputsTrendingLayer::set_default()
{
   display = true;
}


/// Removes a given inputs trending neuron from the inputs trending layer.
/// @param index Index of neuron to be pruned.

void InputsTrendingLayer::prune_input_trending_neuron(const size_t& index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

    if(index >= inputs_trending_neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
              << "void prune_input_trending_neuron(const size_t&) method.\n"
              << "Index of inputs trending neuron is equal or greater than number of inputs trending neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    inputs_trends.erase(inputs_trends.begin() + static_cast<unsigned>(index));
}


/// Initializes the linear regression parameters of all the inputs trending neurons with random values.

void InputsTrendingLayer::initialize_random()
{
    size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

    for(size_t i = 0; i < inputs_trending_neurons_number; i++)
    {
        inputs_trends[i].initialize_random();
    }

    if(rand()%2)
    {
        set_inputs_trending_method("NoTrending");
    }
    else
    {
        set_inputs_trending_method("Linear");
    }
}


/// Calculates the outputs from the inputs trending layer for a set of inputs to that layer.

Matrix<double> InputsTrendingLayer::calculate_outputs(const Matrix<double>& inputs, const double& time) const
{
    const size_t inputs_number = get_inputs_trending_neurons_number();

    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_size = inputs.size();

    if(inputs_size != inputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
              << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
              << "Size of inputs must be equal to number of inputs inputs trending neuron.\n";

       throw logic_error(buffer.str());
    }

 #endif

    const size_t points_number = inputs.get_rows_number();

    Matrix<double> outputs(points_number, inputs_number);

   if(inputs_trending_method == NoTrending)
   {
       return(inputs);
   }
   else if(inputs_trending_method == Linear)
   {
        for(size_t i  = 0; i < points_number; i++)
        {
            for(size_t j = 0; j < inputs_number; j++)
            {
               outputs(i,j) = inputs(i,j) -(inputs_trends[j].intercept + inputs_trends[j].slope * time);
            }
        }

       return(outputs);
   }
   else
   {
       ostringstream buffer;

       buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
              << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
              << "Unknown inputs trending method.\n";

       throw logic_error(buffer.str());
    }
}


/// Returns the derivatives of the outputs with respect to the inputs.

Matrix<double> InputsTrendingLayer::calculate_derivatives(const Matrix<double>&) const
{
/*
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   Vector<double> derivatives(inputs_trending_neurons_number);

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
         derivatives[i] = 1.0;
   }

   return(derivatives);
*/
    return Matrix<double>();
}


/// Returns the second derivatives of the outputs with respect to the inputs.

Matrix<double> InputsTrendingLayer::calculate_second_derivatives(const Matrix<double>&) const
{
/*
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   Vector<double> second_derivatives(inputs_trending_neurons_number, 0.0);

   return(second_derivative);
*/
    return Matrix<double>();
}


/// Arranges a "Jacobian matrix" from a vector of derivatives.
/// The Jacobian matrix is composed of the partial derivatives of the layer outputs with respect to the layer inputs.
/// @param derivatives Vector of outputs-inputs derivatives of each inputs trending neuron.

Matrix<double> InputsTrendingLayer::calculate_Jacobian(const Vector<double>& derivatives) const
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t derivatives_size = derivatives.size();

   if(derivatives_size != inputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
             << "Matrix<double> calculate_Jacobian(const Vector<double>&) method.\n"
             << "Size of derivatives must be equal to number of inputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<double> Jacobian(inputs_trending_neurons_number, inputs_trending_neurons_number, 0.0);
   Jacobian.set_diagonal(derivatives);

   return(Jacobian);
}


/// Arranges a "Hessian form" vector of matrices from a vector of derivatives.
/// The Hessian form is composed of the second partial derivatives of the layer outputs with respect to the layer inputs.

Vector< Matrix<double> > InputsTrendingLayer::calculate_Hessian(const Vector<double>&) const
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   Vector< Matrix<double> > trended_Hessian(inputs_trending_neurons_number);

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
      trended_Hessian[i].set(inputs_trending_neurons_number, inputs_trending_neurons_number, 0.0);
   }

   return(trended_Hessian);
}


/// Returns a string with the expression of the trend functions.

string InputsTrendingLayer::write_expression(const Vector<string>& inputs_name, const Vector<string>& outputs_name) const
{
   ostringstream buffer;

   buffer.precision(10);

   if(inputs_trending_method == Linear)
   {
       const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

       for(size_t i = 0; i < inputs_trending_neurons_number; i++)
       {
           buffer << outputs_name[i] << " = " << inputs_name[i] << " -(" << inputs_trends[i].intercept << " + "
                  << inputs_trends[i].slope << " * time) \n";
       }
   }
   else
   {
       buffer << "";
   }

   return(buffer.str());
}


/// Returns a string representation of the current inputs trending layer object.

string InputsTrendingLayer::object_to_string() const
{
   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   ostringstream buffer;

   buffer << "Inputs trending layer\n";

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
          buffer << "Inputs trending neuron: " << i << "\n"
                 << "Intercept: " << inputs_trends[i].intercept << "\n"
                 << "Slope: " << inputs_trends[i].slope << "\n"
                 << "Corelation: " << inputs_trends[i].correlation << "\n";
   }

   buffer << "Display: " << display << "\n";

   return(buffer.str());
}


/// Serializes the inputs trending layer object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* InputsTrendingLayer::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    ostringstream buffer;

    tinyxml2::XMLElement* inputs_trending_layer_element = document->NewElement("InputsTrendingLayer");

    document->InsertFirstChild(inputs_trending_layer_element);

    // Inputs trending neurons number

    tinyxml2::XMLElement* size_element = document->NewElement("InputsTrendingNeuronsNumber");
    inputs_trending_layer_element->LinkEndChild(size_element);

    const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

    buffer.str("");
    buffer << inputs_trending_neurons_number;

    tinyxml2::XMLText* size_text = document->NewText(buffer.str().c_str());
    size_element->LinkEndChild(size_text);

    for(size_t i = 0; i < inputs_trending_neurons_number; i++)
    {
        tinyxml2::XMLElement* item_element = document->NewElement("Item");
        item_element->SetAttribute("Index",static_cast<unsigned>(i)+1);

        inputs_trending_layer_element->LinkEndChild(item_element);

        // Intercept

        tinyxml2::XMLElement* intercept_element = document->NewElement("Intercept");
        item_element->LinkEndChild(intercept_element);

        buffer.str("");
        buffer << inputs_trends[i].intercept;

        tinyxml2::XMLText* intercept_text = document->NewText(buffer.str().c_str());
        intercept_element->LinkEndChild(intercept_text);

        // Slope

        tinyxml2::XMLElement* slope_element = document->NewElement("Slope");
        slope_element->LinkEndChild(slope_element);

        buffer.str("");
        buffer << inputs_trends[i].slope;

        tinyxml2::XMLText* slope_text = document->NewText(buffer.str().c_str());
        slope_element->LinkEndChild(slope_text);

        // Correlation

        tinyxml2::XMLElement* correlation_element = document->NewElement("Correlation");
        correlation_element->LinkEndChild(correlation_element);

        buffer.str("");
        buffer << inputs_trends[i].correlation;

        tinyxml2::XMLText* correlation_text = document->NewText(buffer.str().c_str());
        correlation_element->LinkEndChild(correlation_text);
    }

    // Inputs trending method

    tinyxml2::XMLElement* method_element = document->NewElement("UseInputsTrendingLayer");
    inputs_trending_layer_element->LinkEndChild(method_element);

    if(inputs_trending_method == NoTrending)
    {
        buffer.str("");
        buffer << 0;
    }
    else if(inputs_trending_method == Linear)
    {
        buffer.str("");
        buffer << 1;
    }
    else
    {
        buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
               << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
               << "Unknown inputs trending method type.\n";

        throw logic_error(buffer.str());
    }

    tinyxml2::XMLText* method_text = document->NewText(buffer.str().c_str());
    method_element->LinkEndChild(method_text);

   // Display

   {
      tinyxml2::XMLElement* display_element = document->NewElement("Display");
      inputs_trending_layer_element->LinkEndChild(display_element);

      buffer.str("");
      buffer << display;

      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
      display_element->LinkEndChild(display_text);
   }

   return(document);
}


/// Serializes the inputs trending layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void InputsTrendingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
   ostringstream buffer;

   file_stream.OpenElement("InputsTrendingLayer");

   // Inputs trending neurons number

   file_stream.OpenElement("InputsTrendingNeuronsNumber");

   const size_t inputs_trending_neurons_number = get_inputs_trending_neurons_number();

   buffer.str("");
   buffer << inputs_trending_neurons_number;

   file_stream.PushText(buffer.str().c_str());

   file_stream.CloseElement();

   for(size_t i = 0; i < inputs_trending_neurons_number; i++)
   {
       file_stream.OpenElement("Item");

       file_stream.PushAttribute("Index",static_cast<unsigned>(i)+1);

       // Intercept

       file_stream.OpenElement("Intercept");

       buffer.str("");
       buffer << inputs_trends[i].intercept;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();

       // Slope

       file_stream.OpenElement("Slope");

       buffer.str("");
       buffer << inputs_trends[i].slope;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();

       // Correlation

       file_stream.OpenElement("Correlation");

       buffer.str("");
       buffer << inputs_trends[i].correlation;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Inputs trending method

   file_stream.OpenElement("UseInputsTrendingLayer");

   if(inputs_trending_method == NoTrending)
   {
       buffer.str("");
       buffer << 0;
   }
   else if(inputs_trending_method == Linear)
   {
       buffer.str("");
       buffer << 1;
   }
   else
   {
       file_stream.CloseElement();

       buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
              << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
              << "Unknown inputs trending method type.\n";

       throw logic_error(buffer.str());
   }

   file_stream.PushText(buffer.str().c_str());

   file_stream.CloseElement();

//   // Display

//   {
//      file_stream.OpenElement("Display");

//      buffer.str("");
//      buffer << display;

//      file_stream.PushText(buffer.str().c_str());

//      file_stream.CloseElement();
//   }

//   file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this inputs trending layer object.
/// @param document TinyXML document containing the member data.

void InputsTrendingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* inputs_trending_layer_element = document.FirstChildElement("InputsTrendingLayer");

    if(!inputs_trending_layer_element)
    {
        buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsTrendingLayer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inputs trending neurons number

    const tinyxml2::XMLElement* inputs_trending_neurons_number_element = inputs_trending_layer_element->FirstChildElement("InputsTrendingNeuronsNumber");

    if(!inputs_trending_neurons_number_element)
    {
        buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "InputsTrendingNeuronsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const size_t inputs_trending_neurons_number = static_cast<size_t>(atoi(inputs_trending_neurons_number_element->GetText()));

    set(inputs_trending_neurons_number);

    unsigned index = 0; // size_t does not work

    const tinyxml2::XMLElement* start_element = inputs_trending_neurons_number_element;

    for(size_t i = 0; i < inputs_trends.size(); i++)
    {
        const tinyxml2::XMLElement* item_element = start_element->NextSiblingElement("Item");
        start_element = item_element;

        if(!item_element)
        {
            buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Item " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        item_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
        {
            buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Index " << index << " is not correct.\n";

            throw logic_error(buffer.str());
        }

        // Intercept

        const tinyxml2::XMLElement* intercept_element = item_element->FirstChildElement("Intercept");

        if(intercept_element)
        {
            if(intercept_element->GetText())
            {
                inputs_trends[index-1].intercept = atof(intercept_element->GetText());
            }
        }

        // Slope

        const tinyxml2::XMLElement* slope_element = item_element->FirstChildElement("Slope");

        if(slope_element)
        {
            if(slope_element->GetText())
            {
                inputs_trends[index-1].slope = atof(slope_element->GetText());
            }
        }

        // Correlation

        const tinyxml2::XMLElement* correlation_element = item_element->FirstChildElement("Correlation");

        if(correlation_element)
        {
            if(correlation_element->GetText())
            {
                inputs_trends[index-1].correlation = atof(correlation_element->GetText());
            }
        }
    }

    // Use inputs trending layer

    {
        const tinyxml2::XMLElement* use_inputs_trending_layer_element = inputs_trending_layer_element->FirstChildElement("UseInputsTrendingLayer");

        if(use_inputs_trending_layer_element)
        {
            size_t new_method = static_cast<size_t>(atoi(use_inputs_trending_layer_element->GetText()));

            if(new_method == 0)
            {
                inputs_trending_method = NoTrending;
            }
            else if(new_method == 1)
            {
                inputs_trending_method = Linear;
            }
            else
            {
                buffer << "OpenNN Exception: InputsTrendingLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                       << "Unknown inputs trending method.\n";

                throw logic_error(buffer.str());
            }
        }
    }

}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
