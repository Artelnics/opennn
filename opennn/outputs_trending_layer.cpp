/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   O U T P U T S   T R E N D I N G   L A Y E R   C L A S S                                                    */
/*                                                                                                              */
/*   Patricia Garcia                                                                                            */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   patriciagarcia@artelnics.com                                                                               */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "outputs_trending_layer.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates an outputs trending layer object with zero outputs trending neurons.

OutputsTrendingLayer::OutputsTrendingLayer()
{
   set();
}


// OUTPUTS TRENDING NEURONS NUMBER CONSTRUCTOR

/// Outputs trending neurons number constructor.
/// It creates an outputs trending layer with a given size.
/// @param outputs_trending_neurons_number Number of outputs trending neurons in the layer.

OutputsTrendingLayer::OutputsTrendingLayer(const size_t& outputs_trending_neurons_number)
{
   set(outputs_trending_neurons_number);

   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates an outputs trending layer and loads its members from a XML document.
/// @param outputs_trending_layer_document TinyXML document with the member data.

OutputsTrendingLayer::OutputsTrendingLayer(const tinyxml2::XMLDocument& outputs_trending_layer_document)
{
   set(outputs_trending_layer_document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing outputs trending layer object.
/// @param other_outputs_trending_layer Outputs trending layer to be copied.

OutputsTrendingLayer::OutputsTrendingLayer(const OutputsTrendingLayer& other_outputs_trending_layer)
{
   set(other_outputs_trending_layer);
}


// DESTRUCTOR

/// Destructor.
/// This destructor does not delete any pointer. 

OutputsTrendingLayer::~OutputsTrendingLayer()
{
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to this object the members of an existing trending layer object.
/// @param other_outputs_trending_layer Outputs trending layer object to be assigned.

OutputsTrendingLayer& OutputsTrendingLayer::operator = (const OutputsTrendingLayer& other_outputs_trending_layer)
{
   if(this != &other_outputs_trending_layer)
   {
      display = other_outputs_trending_layer.display;
   }

   return(*this);
}


// EQUAL TO OPERATOR


/// Equal to operator. 
/// It compares this object with another object of the same class. 
/// It returns true if the members of the two objects have the same values, and false otherwise.
/// @ param other_outputs_trending_layer Outputs trending layer to be compared with.

bool OutputsTrendingLayer::operator == (const OutputsTrendingLayer& other_outputs_trending_layer) const
{
    if(get_outputs_trending_neurons_number() != other_outputs_trending_layer.get_outputs_trending_neurons_number())
    {
       return(false);
    }
    else if(get_outputs_trending_neurons_number() == other_outputs_trending_layer.get_outputs_trending_neurons_number())
    {
        if(display != other_outputs_trending_layer.display)
        {
            return(false);
        }

        for(size_t i = 0; i < get_outputs_trending_neurons_number(); i++)
        {
            if(fabs(get_intercept(i) - other_outputs_trending_layer.get_intercept(i)) > numeric_limits<double>::epsilon()
                    || fabs(get_slope(i) - other_outputs_trending_layer.get_slope(i)) > numeric_limits<double>::epsilon()
                    || fabs(get_correlation(i) - other_outputs_trending_layer.get_correlation(i)) > numeric_limits<double>::epsilon())
            {
                return(false);
            }
        }
    }

    return(true);
}


/// Returns true if the size of the layer is zero, and false otherwise.

bool OutputsTrendingLayer::is_empty() const
{
   if(get_outputs_trending_neurons_number() == 0)
   {
      return(true);
   }
   else
   {
      return(false);
   }
}


/// Returns the method used for outputs trending layer.

const OutputsTrendingLayer::OutputsTrendingMethod& OutputsTrendingLayer::get_outputs_trending_method() const
{
    return(outputs_trending_method);
}


/// Returns a string with the name of the method used for outputs trending layer.

string OutputsTrendingLayer::write_outputs_trending_method() const
{
    if(outputs_trending_method == NoTrending)
    {
        return("NoTrending");
    }
    else if(outputs_trending_method == Linear)
    {
        return("Linear");
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
               << "string write_outputs_trending_method() const method.\n"
               << "Unknown outputs trending method.\n";

        throw logic_error(buffer.str());
    }
}


/// Returns the number of outputs trending neurons in the layer.

size_t OutputsTrendingLayer::get_outputs_trending_neurons_number() const
{
   return(outputs_trends.size());
}


/// Returns the intercepts values of all the outputs trending neurons in the layer.

Vector<double> OutputsTrendingLayer::get_intercepts() const
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   Vector<double> intercepts(outputs_trending_neurons_number);

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
       intercepts[i] = outputs_trends[i].intercept;
   }

   return(intercepts);
}


/// Returns the intercept of a single outputs trending neuron.
/// @param i Index of outputs trending neuron.

double OutputsTrendingLayer::get_intercept(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   if(i >= outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "double get_intercept(const size_t&) const method.\n"
             << "Index must be less than number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   return(outputs_trends[i].intercept);
}


/// Returns the slopes values of all the outputs trending neurons in the layer.

Vector<double> OutputsTrendingLayer::get_slopes() const
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   Vector<double> slopes(outputs_trending_neurons_number);

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
       slopes[i] = outputs_trends[i].slope;
   }

   return(slopes);
}


/// Returns the slope value of a single outputs trending neuron.
/// @param i Index of outputs trending neuron.

double OutputsTrendingLayer::get_slope(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   if(outputs_trending_neurons_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "double get_slope(const size_t&) const method.\n"
             << "Number of outputs trending neurons is zero.\n";

      throw logic_error(buffer.str());
   }
   else if(i >= outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "double get_slope(const size_t&) const method.\n"
             << "Index must be less than number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   return(outputs_trends[i].slope);
}


/// Returns the correlations values of all the outputs trending neurons in the layer.

Vector<double> OutputsTrendingLayer::get_correlations() const
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   Vector<double> correlations(outputs_trending_neurons_number);

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
       correlations[i] = outputs_trends[i].correlation;
   }

   return(correlations);
}


/// Returns the correlation value of a single outputs trending neuron.
/// @param i Index of outputs trending neuron.

double OutputsTrendingLayer::get_correlation(const size_t& i) const
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   if(outputs_trending_neurons_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "double get_correlation(const size_t&) const method.\n"
             << "Number of outputs trending neurons is zero.\n";

      throw logic_error(buffer.str());
   }
   else if(i >= outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "double get_correlation(const size_t&) const method.\n"
             << "Index must be less than number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   return(outputs_trends[i].correlation);
}


/// Returns the trend value of each outputs trending neuron.

Vector< LinearRegressionParameters<double> > OutputsTrendingLayer::get_outputs_trends() const
{
    return(outputs_trends);
}


/// Sets the number of outputs trending neurons to be zero.
/// It also sets the rest of members to their default values.

void OutputsTrendingLayer::set()
{
   outputs_trends.set();

   set_default();
}


/// Resizes the outputs trending layer.
/// It also sets the rest of members to their default values.
/// @param new_outputs_trending_neurons_number Size of the outputs trending layer.

void OutputsTrendingLayer::set(const size_t& new_outputs_trending_neurons_number)
{
   outputs_trends.set(new_outputs_trending_neurons_number);

   set_default();
}


/// Sets the outputs trending layer members from a XML document.
/// @param outputs_trending_layer_document Pointer to a TinyXML document containing the member data.

void OutputsTrendingLayer::set(const tinyxml2::XMLDocument& outputs_trending_layer_document)
{
   set_default();

   from_XML(outputs_trending_layer_document);
}


/// Sets the members of this object to be the members of another object of the same class.
/// @param other_outputs_trending_layer Object to be copied.

void OutputsTrendingLayer::set(const OutputsTrendingLayer& other_outputs_trending_layer)
{
   outputs_trends = other_outputs_trending_layer.outputs_trends;

   display = other_outputs_trending_layer.display;
}


/// Sets a new outputs trending method.
/// @param new_method New trending method.

void OutputsTrendingLayer::set_outputs_trending_method(const OutputsTrendingMethod& new_method)
{
    outputs_trending_method = new_method;
}


/// Sets a new outputs trending method.
/// @param new_method_string New outputs trending method string.

void OutputsTrendingLayer::set_outputs_trending_method(const string& new_method_string)
{
    if(new_method_string == "NoTrending")
    {
        outputs_trending_method = NoTrending;
    }
    else if(new_method_string == "Linear")
    {
        outputs_trending_method = Linear;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
               << "void set_outputs_trending_method(const string&) method.\n"
               << "Unknown outputs trending method: " << new_method_string << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Sets the outputs trending layer members to the given values.

void OutputsTrendingLayer::set_outputs_trends(const Vector< LinearRegressionParameters<double> >& new_outputs_trends)
{
     const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t size = new_outputs_trends.size();

    if(size != outputs_trending_neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
              << "void set_outuputs_trends(const Vector< Vector<double> >&) method.\n"
              << "Size of the new outputs trends must be equal to the number of outputs trending neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Set outputs trends

    for(size_t i = 0; i < outputs_trending_neurons_number; i++)
    {
        outputs_trends[i] = new_outputs_trends[i];
    }
}


/// Sets new intercepts for all the neurons in the layer.
/// @param new_intercepts New set of intercepts for the outputs trending neurons.

void OutputsTrendingLayer::set_intercepts(const Vector<double>& new_intercepts)
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_intercepts.size() != outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "void set_intercepts(const Vector<double>&) method.\n"
             << "Size must be equal to number of outputs trending neurons number.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set intercepts

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
       outputs_trends[i].intercept = new_intercepts[i];
   }
}


/// Sets a new intercept for a single neuron.
/// @param index Index of outputs trending neuron.
/// @param new_intercept New intercept for the neuron with index i.

void OutputsTrendingLayer::set_intercept(const size_t& index, const double& new_intercept)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   if(index >= outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "void set_intercept(const size_t&, const double&) method.\n"
             << "Index of trending neurons must be less than number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set intercept of a single neuron

   outputs_trends[index].intercept = new_intercept;
}


/// Sets new slopes for all the outputs trending neurons.
/// @param new_slopes New set of slopes for the layer.

void OutputsTrendingLayer::set_slopes(const Vector<double>& new_slopes)
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_slopes.size() != outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "void set_slopes(const Vector<double>&) method.\n"
             << "Size must be equal to number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set slopes

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
       outputs_trends[i].slope = new_slopes[i];
   }
}


/// Sets a new slope for a single neuron.
/// @param index Index of outputs trending neuron.
/// @param new_slope New slope for the outputs trending neuron with that index.

void OutputsTrendingLayer::set_slope(const size_t& index, const double& new_slope)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   if(index >= outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "void set_slope(const size_t&, const double&) method.\n"
             << "Index of outputs trending neuron must be less than number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set slope of a single neuron

   outputs_trends[index].slope = new_slope;
}


/// Sets new correlations for all the outputs trending neurons.
/// @param new_correlations New set of correlations for the layer.

void OutputsTrendingLayer::set_correlations(const Vector<double>& new_correlations)
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   if(new_correlations.size() != outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "void set_correlations(const Vector<double>&) method.\n"
             << "Size must be equal to number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set correlations

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
       outputs_trends[i].correlation = new_correlations[i];
   }
}


/// Sets a new correlation for a single neuron.
/// @param index Index of outputs trending neuron.
/// @param new_correlation New correlation for the outputs trending neuron with that index.

void OutputsTrendingLayer::set_correlation(const size_t& index, const double& new_correlation)
{
   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   if(index >= outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "void set_correlation(const size_t&, const double&) method.\n"
             << "Index of outputs trending neuron must be less than number of outputs trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set correlation of a single neuron

   outputs_trends[index].correlation = new_correlation;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void OutputsTrendingLayer::set_display(const bool& new_display)
{
   display = new_display;
}


/// Sets the members to their default values:
/// <ul>
/// <li> Display: True. 
/// </ul>

void OutputsTrendingLayer::set_default()
{
   display = true;        
}


/// Removes a given outputs trending neuron from the outputs trending layer.
/// @param index Index of neuron to be pruned.

void OutputsTrendingLayer::prune_output_trending_neuron(const size_t& index)
{
    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

    if(index >= outputs_trending_neurons_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
              << "void prune_outputs_trending_neuron(const size_t&) method.\n"
              << "Index of outputs trending neuron is equal or greater than number of outputs trending neurons.\n";

       throw logic_error(buffer.str());
    }

    #endif

    outputs_trends.erase(outputs_trends.begin() + static_cast<unsigned>(index));
}


/// Initializes the linear regression parameters of all the outputs trending neurons with random values.

void OutputsTrendingLayer::initialize_random()
{
    size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

    for(size_t i = 0; i < outputs_trending_neurons_number; i++)
    {
        outputs_trends[i].initialize_random();
    }

    if(rand()%2)
    {
        set_outputs_trending_method("NoTrending");
    }
    else
    {
        set_outputs_trending_method("Linear");
    }
}


/// Calculates the outputs from the outputs trending layer for a set of inputs to that layer.

Matrix<double> OutputsTrendingLayer::calculate_outputs(const Matrix<double>& inputs, const double& time) const
{
    const size_t points_number = inputs.get_rows_number();
    const size_t outputs_number = get_outputs_trending_neurons_number();

    // Control sentence(if debug)

    #ifdef __OPENNN_DEBUG__

    const size_t inputs_size = inputs.size();

    if(inputs_size != outputs_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
              << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
              << "Size of inputs must be equal to number of outputs trending neurons.\n";

       throw logic_error(buffer.str());
    }

 #endif

    Matrix<double> outputs(points_number, outputs_number);

   if(outputs_trending_method == NoTrending)
   {
       return(inputs);
   }
   else if(outputs_trending_method == Linear)
   {
       for(size_t i = 0; i < points_number; i++)
       {
           for(size_t j = 0; j < outputs_number; j++)
           {
               outputs[j] = inputs[j] + outputs_trends[j].intercept + outputs_trends[j].slope * time;
           }
        }
       return(outputs);
   }
   else
   {
       ostringstream buffer;

       buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
              << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
              << "Unknown outputs trending method.\n";

       throw logic_error(buffer.str());
    }
}


/// Returns the derivatives of the outputs with respect to the inputs.

Matrix<double> OutputsTrendingLayer::calculate_derivatives(const Matrix<double>& inputs) const
{
    const size_t points_number = inputs.get_rows_number();
    const size_t outputs_number = get_outputs_trending_neurons_number();

   return Matrix<double>(points_number, outputs_number, 1.0);
}


/// Returns the second derivatives of the outputs with respect to the inputs.

Matrix<double> OutputsTrendingLayer::calculate_second_derivatives(const Matrix<double>& inputs) const
{
    const size_t points_number = inputs.get_rows_number();
    const size_t outputs_number = get_outputs_trending_neurons_number();

   return Matrix<double>(points_number, outputs_number, 0.0);
}


/// Arranges a "Jacobian matrix" from a vector of derivatives.
/// The Jacobian matrix is composed of the partial derivatives of the layer outputs with respect to the layer inputs. 
/// @param derivatives Vector of outputs-inputs derivatives of each outputs trending neuron.

Matrix<double> OutputsTrendingLayer::calculate_Jacobian(const Vector<double>& derivatives) const
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   // Control sentence(if debug)

   #ifdef __OPENNN_DEBUG__

   const size_t derivatives_size = derivatives.size();

   if(derivatives_size != outputs_trending_neurons_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
             << "Matrix<double> calculate_Jacobian(const Vector<double>&) method.\n"
             << "Size of derivatives must be equal to number of trending neurons.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Matrix<double> Jacobian(outputs_trending_neurons_number, outputs_trending_neurons_number, 0.0);
   Jacobian.set_diagonal(derivatives);

   return(Jacobian);
}


/// Arranges a "Hessian form" vector of matrices from a vector of derivatives.
/// The Hessian form is composed of the second partial derivatives of the layer outputs with respect to the layer inputs. 

Vector< Matrix<double> > OutputsTrendingLayer::calculate_Hessian(const Vector<double>&) const
{
   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   Vector< Matrix<double> > trended_Hessian(outputs_trending_neurons_number);

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
      trended_Hessian[i].set(outputs_trending_neurons_number, outputs_trending_neurons_number, 0.0);
   }

   return(trended_Hessian);
}


/// Returns a string with the expression of the the trend functions.

string OutputsTrendingLayer::write_expression(const Vector<string>& inputs_name, const Vector<string>& outputs_name) const
{
   ostringstream buffer;

   buffer.precision(10);

   if(outputs_trending_method == Linear)
   {
       const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

       for(size_t i = 0; i < outputs_trending_neurons_number; i++)
       {
           buffer << outputs_name[i] << " = " << inputs_name[i] << " + (" << outputs_trends[i].intercept << " + "
                  << outputs_trends[i].slope << " * time) \n";
       }
   }
   else
   {
       buffer << "";
   }

   return(buffer.str());
}


/// Returns a string representation of the current outputs trending layer object.

string OutputsTrendingLayer::object_to_string() const
{
   size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   ostringstream buffer;

   buffer << "Outputs trending layer\n";

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
          buffer << "Outputs trending neuron: " << i << "\n"
                 << "Intercept: " << outputs_trends[i].intercept << "\n"
                 << "Slope: " << outputs_trends[i].slope << "\n"
                 << "Corelation: " << outputs_trends[i].correlation << "\n";
   }

   buffer << "Display: " << display << "\n";

   return(buffer.str());
}


/// Serializes the outputs trending layer object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document.

tinyxml2::XMLDocument* OutputsTrendingLayer::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    ostringstream buffer;

    tinyxml2::XMLElement* outputs_trending_layer_element = document->NewElement("OutputsTrendingLayer");

    document->InsertFirstChild(outputs_trending_layer_element);

    // Outputs trending neurons number

    tinyxml2::XMLElement* size_element = document->NewElement("OutputsTrendingNeuronsNumber");
    outputs_trending_layer_element->LinkEndChild(size_element);

    const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

    buffer.str("");
    buffer << outputs_trending_neurons_number;

    tinyxml2::XMLText* size_text = document->NewText(buffer.str().c_str());
    size_element->LinkEndChild(size_text);

    for(size_t i = 0; i < outputs_trending_neurons_number; i++)
    {
        tinyxml2::XMLElement* item_element = document->NewElement("Item");
        item_element->SetAttribute("Index",static_cast<unsigned>(i)+1);

        outputs_trending_layer_element->LinkEndChild(item_element);

        // Intercept

        tinyxml2::XMLElement* intercept_element = document->NewElement("Intercept");
        item_element->LinkEndChild(intercept_element);

        buffer.str("");
        buffer << outputs_trends[i].intercept;

        tinyxml2::XMLText* intercept_text = document->NewText(buffer.str().c_str());
        intercept_element->LinkEndChild(intercept_text);

        // Slope

        tinyxml2::XMLElement* slope_element = document->NewElement("Slope");
        slope_element->LinkEndChild(slope_element);

        buffer.str("");
        buffer << outputs_trends[i].slope;

        tinyxml2::XMLText* slope_text = document->NewText(buffer.str().c_str());
        slope_element->LinkEndChild(slope_text);

        // Correlation

        tinyxml2::XMLElement* correlation_element = document->NewElement("Correlation");
        correlation_element->LinkEndChild(correlation_element);

        buffer.str("");
        buffer << outputs_trends[i].correlation;

        tinyxml2::XMLText* correlation_text = document->NewText(buffer.str().c_str());
        correlation_element->LinkEndChild(correlation_text);
    }

    // Outputs trending method

    tinyxml2::XMLElement* method_element = document->NewElement("UseOutputsTrendingLayer");
    outputs_trending_layer_element->LinkEndChild(method_element);

    if(outputs_trending_method == NoTrending)
    {
        buffer.str("");
        buffer << 0;
    }
    else if(outputs_trending_method == Linear)
    {
        buffer.str("");
        buffer << 1;
    }
    else
    {
        buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
               << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
               << "Unknown outputs trending method type.\n";

        throw logic_error(buffer.str());
    }

    tinyxml2::XMLText* method_text = document->NewText(buffer.str().c_str());
    method_element->LinkEndChild(method_text);

   // Display

   {
      tinyxml2::XMLElement* display_element = document->NewElement("Display");
      outputs_trending_layer_element->LinkEndChild(display_element);

      buffer.str("");
      buffer << display;

      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
      display_element->LinkEndChild(display_text);
   }

   return(document);
}


/// Serializes the outputs trending layer object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void OutputsTrendingLayer::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
   ostringstream buffer;

   file_stream.OpenElement("OutputsTrendingLayer");

   // Outputs trending neurons number

   file_stream.OpenElement("TrendingNeuronsNumber");

   const size_t outputs_trending_neurons_number = get_outputs_trending_neurons_number();

   buffer.str("");
   buffer << outputs_trending_neurons_number;

   file_stream.PushText(buffer.str().c_str());

   file_stream.CloseElement();

   for(size_t i = 0; i < outputs_trending_neurons_number; i++)
   {
       file_stream.OpenElement("Item");

       file_stream.PushAttribute("Index",static_cast<unsigned>(i)+1);

       // Intercept

       file_stream.OpenElement("Intercept");

       buffer.str("");
       buffer << outputs_trends[i].intercept;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();

       // Slope

       file_stream.OpenElement("Slope");

       buffer.str("");
       buffer << outputs_trends[i].slope;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();

       // Correlation

       file_stream.OpenElement("Correlation");

       buffer.str("");
       buffer << outputs_trends[i].correlation;

       file_stream.PushText(buffer.str().c_str());

       file_stream.CloseElement();
   }

   // Outputs trending method

   file_stream.OpenElement("UseOutputsTrendingLayer");

   if(outputs_trending_method == NoTrending)
   {
       buffer.str("");
       buffer << 0;
   }
   else if(outputs_trending_method == Linear)
   {
       buffer.str("");
       buffer << 1;
   }
   else
   {
       file_stream.CloseElement();

       buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
              << "void write_XML(tinyxml2::XMLPrinter&) const method.\n"
              << "Unknown outputs trending method type.\n";

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


/// Deserializes a TinyXML document into this outputs trending layer object.
/// @param document TinyXML document containing the member data.

void OutputsTrendingLayer::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* outputs_trending_layer_element = document.FirstChildElement("OutputsTrendingLayer");

    if(!outputs_trending_layer_element)
    {
        buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "OutputsTrendingLayer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Outputs trending neurons number

    const tinyxml2::XMLElement* outputs_trending_neurons_number_element = outputs_trending_layer_element->FirstChildElement("OutputsTrendingNeuronsNumber");

    if(!outputs_trending_neurons_number_element)
    {
        buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "OutputsTrendingNeuronsNumber element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const size_t outputs_trending_neurons_number = static_cast<size_t>(atoi(outputs_trending_neurons_number_element->GetText()));

    set(outputs_trending_neurons_number);

    unsigned index = 0; // size_t does not work

    const tinyxml2::XMLElement* start_element = outputs_trending_neurons_number_element;

    for(size_t i = 0; i < outputs_trends.size(); i++)
    {
        const tinyxml2::XMLElement* item_element = start_element->NextSiblingElement("Item");
        start_element = item_element;

        if(!item_element)
        {
            buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
                   << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                   << "Item " << i+1 << " is nullptr.\n";

            throw logic_error(buffer.str());
        }

        item_element->QueryUnsignedAttribute("Index", &index);

        if(index != i+1)
        {
            buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
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
                outputs_trends[index-1].intercept = atof(intercept_element->GetText());
            }
        }

        // Slope

        const tinyxml2::XMLElement* slope_element = item_element->FirstChildElement("Slope");

        if(slope_element)
        {
            if(slope_element->GetText())
            {
                outputs_trends[index-1].slope = atof(slope_element->GetText());
            }
        }

        // Correlation

        const tinyxml2::XMLElement* correlation_element = item_element->FirstChildElement("Correlation");

        if(correlation_element)
        {
            if(correlation_element->GetText())
            {
                outputs_trends[index-1].correlation = atof(correlation_element->GetText());
            }
        }
    }

    // Use trending layer
    {
        const tinyxml2::XMLElement* use_outputs_trending_layer_element = outputs_trending_layer_element->FirstChildElement("UseOutputsTrendingLayer");

        if(use_outputs_trending_layer_element)
        {
            size_t new_method = static_cast<size_t>(atoi(use_outputs_trending_layer_element->GetText()));

            if(new_method == 0)
            {
                outputs_trending_method = NoTrending;
            }
            else if(new_method == 1)
            {
                outputs_trending_method = Linear;
            }
            else
            {
                buffer << "OpenNN Exception: OutputsTrendingLayer class.\n"
                       << "void from_XML(const tinyxml2::XMLElement*) method.\n"
                       << "Unknown outputs trending method.\n";

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
