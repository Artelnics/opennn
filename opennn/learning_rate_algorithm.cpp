//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L E A R N I N G   R A T E   A L G O R I T H M   C L A S S             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "learning_rate_algorithm.h"

namespace OpenNN
{

/// Default constructor. 
/// It creates a learning rate algorithm object not associated to any loss index object.
/// It also initializes the class members to their default values. 

LearningRateAlgorithm::LearningRateAlgorithm()
 : loss_index_pointer(nullptr)
{ 
   set_default();
}


/// Destructor.
/// It creates a learning rate algorithm associated to a loss index.
/// It also initializes the class members to their default values. 
/// @param new_loss_index_pointer Pointer to a loss index object.

LearningRateAlgorithm::LearningRateAlgorithm(LossIndex* new_loss_index_pointer)
 : loss_index_pointer(new_loss_index_pointer)
{
   set_default();
}


/// XML constructor. 
/// It creates a learning rate algorithm object not associated to any loss index object.
/// It also loads the class members from a XML document. 
/// @param document Pointer to a TinyXML document->

LearningRateAlgorithm::LearningRateAlgorithm(const tinyxml2::XMLDocument& document)
 : loss_index_pointer(nullptr)
{ 
   from_XML(document);
}


/// Destructor.

LearningRateAlgorithm::~LearningRateAlgorithm()
{ 
}


/// Returns a pointer to the loss index object
/// to which the learning rate algorithm is associated.
/// If the loss index pointer is nullptr, this method throws an exception.

LossIndex* LearningRateAlgorithm::get_loss_index_pointer() const
{
    #ifdef __OPENNN_DEBUG__

    if(!loss_index_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
               << "LossIndex* get_loss_index_pointer() const method.\n"
               << "Loss index pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

    #endif

   return(loss_index_pointer);
}


/// Returns true if this learning rate algorithm has an associated loss index,
/// and false otherwise.

bool LearningRateAlgorithm::has_loss_index() const
{
    if(loss_index_pointer)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Returns the training rate method used for training.

const LearningRateAlgorithm::LearningRateMethod& LearningRateAlgorithm::get_learning_rate_method() const
{
   return(learning_rate_method);
}


/// Returns a string with the name of the training rate method to be used.

string LearningRateAlgorithm::write_learning_rate_method() const
{
   switch(learning_rate_method)
   {
      case Fixed:
      {
         return "Fixed";
	   }

      case GoldenSection:
      {
         return "GoldenSection";
	   }

      case BrentMethod:
      {
         return "BrentMethod";
	   }
   }

   return string();
}


const double& LearningRateAlgorithm::get_loss_tolerance() const
{
   return loss_tolerance;
}


/// Returns the training rate value at wich a warning message is written to the screen during line
/// minimization.

const double& LearningRateAlgorithm::get_warning_learning_rate() const
{
   return(warning_learning_rate);
}


/// Returns the training rate value at wich the line minimization algorithm is assumed to fail when
/// bracketing a minimum.

const double& LearningRateAlgorithm::get_error_learning_rate() const
{
   return(error_learning_rate);
}


/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& LearningRateAlgorithm::get_display() const
{
   return display;
}


/// Sets the loss index pointer to nullptr.
/// It also sets the rest of members to their default values. 

void LearningRateAlgorithm::set()
{
   loss_index_pointer = nullptr;
   set_default();
}


/// Sets a new loss index pointer.
/// It also sets the rest of members to their default values. 
/// @param new_loss_index_pointer Pointer to a loss index object.

void LearningRateAlgorithm::set(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;

   set_default();
}


/// Sets the members of the learning rate algorithm to their default values.

void LearningRateAlgorithm::set_default()
{
   // TRAINING OPERATORS

   learning_rate_method = BrentMethod;

   // TRAINING PARAMETERS

   loss_tolerance = 1.0e-3;

   warning_learning_rate = 1.0e6;

   error_learning_rate = 1.0e9;

   // UTILITIES

   display = true;
}


/// Sets a pointer to a loss index object to be associated to the optimization algorithm.
/// @param new_loss_index_pointer Pointer to a loss index object.

void LearningRateAlgorithm::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;
}


/// Sets a new training rate method to be used for training.
/// @param new_learning_rate_method Training rate method.

void LearningRateAlgorithm::set_learning_rate_method(const LearningRateAlgorithm::LearningRateMethod& new_learning_rate_method)
{
   learning_rate_method = new_learning_rate_method;
}


/// Sets the method for obtaining the training rate from a string with the name of the method.
/// @param new_learning_rate_method Name of training rate method("Fixed", "GoldenSection", "BrentMethod").

void LearningRateAlgorithm::set_learning_rate_method(const string& new_learning_rate_method)
{
   if(new_learning_rate_method == "Fixed")
   {
      learning_rate_method = Fixed;
   }
   else if(new_learning_rate_method == "GoldenSection")
   {
      learning_rate_method = GoldenSection;
   }
   else if(new_learning_rate_method == "BrentMethod")
   {
      learning_rate_method = BrentMethod;
   }
   else
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
             << "void set_method(const string&) method.\n"
			 << "Unknown training rate method: " << new_learning_rate_method << ".\n";
   
      throw logic_error(buffer.str());
   }
}


/// Sets a new tolerance value to be used in line minimization.
/// @param new_loss_tolerance Tolerance value in line minimization.

void LearningRateAlgorithm::set_loss_tolerance(const double& new_loss_tolerance)
{  
   #ifdef __OPENNN_DEBUG__ 
                                      
   if(new_loss_tolerance <= 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
             << "void set_loss_tolerance(const double&) method.\n"
             << "Tolerance must be greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set loss tolerance

   loss_tolerance = new_loss_tolerance;
}


/// Sets a new training rate value at wich a warning message is written to the screen during line
/// minimization.
/// @param new_warning_learning_rate Warning training rate value.

void LearningRateAlgorithm::set_warning_learning_rate(const double& new_warning_learning_rate)
{  
   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_learning_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
             << "void set_warning_learning_rate(const double&) method.\n"
             << "Warning training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   warning_learning_rate = new_warning_learning_rate;
}


/// Sets a new training rate value at wich a the line minimization algorithm is assumed to fail when
/// bracketing a minimum.
/// @param new_error_learning_rate Error training rate value.

void LearningRateAlgorithm::set_error_learning_rate(const double& new_error_learning_rate)
{  
   #ifdef __OPENNN_DEBUG__ 

   if(new_error_learning_rate < 0.0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
             << "void set_error_learning_rate(const double&) method.\n"
             << "Error training rate must be equal or greater than 0.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Set error training rate

   error_learning_rate = new_error_learning_rate;
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void LearningRateAlgorithm::set_display(const bool& new_display)
{
   display = new_display;
}


/// Returns a vector with two elements:
///(i) the training rate calculated by means of the corresponding algorithm, and
///(ii) the loss for that training rate.
/// @param loss Initial Performance value.
/// @param training_direction Initial training direction.
/// @param initial_learning_rate Initial training rate to start the algorithm. 

pair<double,double> LearningRateAlgorithm::calculate_directional_point(const double& loss,
                                                                  const Vector<double>& training_direction,
                                                                  const double& initial_learning_rate) const
{
   #ifdef __OPENNN_DEBUG__ 

   if(loss_index_pointer == nullptr)
   {
      ostringstream buffer;

      buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
             << "Vector<double> calculate_directional_point(const double&, const Vector<double>&, const double&) const method.\n"
             << "Pointer to loss index is nullptr.\n";

      throw logic_error(buffer.str());
   }

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   if(neural_network_pointer == nullptr)
   {
      ostringstream buffer;

      buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
             << "Vector<double> calculate_directional_point(const double&, const Vector<double>&, const double&) const method.\n"
             << "Pointer to neural network is nullptr.\n";

      throw logic_error(buffer.str());
   }

   #endif
   
   switch(learning_rate_method)
   {
      case Fixed:
      {
         return calculate_fixed_directional_point(loss, training_direction, initial_learning_rate);
      }

      case GoldenSection:
      {
         return calculate_golden_section_directional_point(loss, training_direction, initial_learning_rate);
      }

      case BrentMethod:
      {
         return calculate_Brent_method_directional_point(loss, training_direction, initial_learning_rate);
      }
   }

   return pair<double,double>();
}


/// Returns bracketing triplet.
/// This algorithm is used by line minimization algorithms. 
/// @param loss Initial Performance value.
/// @param training_direction Initial training direction.
/// @param initial_learning_rate Initial training rate to start the algorithm. 

LearningRateAlgorithm::Triplet LearningRateAlgorithm::calculate_bracketing_triplet(
        const double& loss,
        const Vector<double>& training_direction,
        const double& initial_learning_rate) const
{    
    Triplet triplet;

    // Left point

    triplet.A.first = 0.0;
    triplet.A.second = loss;

   if(training_direction == 0.0 || initial_learning_rate == 0.0)
   {
       triplet.U = triplet.A;
       triplet.B = triplet.A;

       return triplet;
   }

   size_t count = 0;

   // Right point

   triplet.B.first = initial_learning_rate;

   triplet.B.second = loss_index_pointer->calculate_training_loss(training_direction, triplet.B.first);
   count++;

   if(triplet.A.second > triplet.B.second)
   {
       triplet.U = triplet.B;

       triplet.B.first *= golden_ratio;
       triplet.B.second = loss_index_pointer->calculate_training_loss(training_direction, triplet.B.first);
       count++;

       while(triplet.U.second > triplet.B.second)
       {
           triplet.A = triplet.U;
           triplet.U = triplet.B;

           triplet.B.first *= golden_ratio;
           triplet.B.second = loss_index_pointer->calculate_training_loss(training_direction, triplet.B.first);
           count++;
       }
   }
   else if(triplet.A.second < triplet.B.second)
   {
       triplet.U.first = triplet.A.first + (triplet.B.first - triplet.A.first)*0.382;
       triplet.U.second = loss_index_pointer->calculate_training_loss(training_direction, triplet.U.first);
       count++;

       while(triplet.A.second < triplet.U.second)
       {
          triplet.B = triplet.U;

          triplet.U.first = triplet.A.first + (triplet.B.first-triplet.A.first)*0.382;
          triplet.U.second = loss_index_pointer->calculate_training_loss(training_direction, triplet.U.first);

          if(triplet.U.first - triplet.A.first <= loss_tolerance)
          {
              triplet.U = triplet.A;
              triplet.B = triplet.A;
              triplet.check();

              return triplet;
          }
       }
   }

   triplet.check();

   return triplet;
}


/// Returns a vector with two elements, a fixed training rate,
/// and the loss for that training rate. 
/// @param training_direction Training direction for the directional point.
/// @param initial_learning_rate Training rate for the directional point.

pair<double,double> LearningRateAlgorithm::calculate_fixed_directional_point(const double&, const Vector<double>& training_direction, const double& initial_learning_rate) const
{
   pair<double,double> directional_point;

   directional_point.first = initial_learning_rate;
   directional_point.second = loss_index_pointer->calculate_training_loss(training_direction, initial_learning_rate);

   return(directional_point);
}


/// Returns the training rate by searching in a given direction to locate the minimum of the error
/// function in that direction. It uses the golden section method.
/// @param loss Neural neural_network_pointer's loss value.
/// @param training_direction Training direction vector.
/// @param initial_learning_rate Initial training rate in line minimization.

pair<double,double> LearningRateAlgorithm:: calculate_golden_section_directional_point(const double& loss,
                                                                                 const Vector<double>& training_direction,
                                                                                 const double& initial_learning_rate) const
{
   ostringstream buffer;

   // Bracket minimum

   try
   {
      Triplet triplet = calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

      if(triplet.has_length_zero())
	  {
         return(triplet.A);
	  }

      pair<double,double> V;

      // Reduce the interval

      do
      {
         V.first = calculate_golden_section_learning_rate(triplet);

         V.second = loss_index_pointer->calculate_training_loss(training_direction, V.first);

         // Update points
 
         if(V.first < triplet.U.first && V.second >= triplet.U.second)
	      {
            triplet.A = V;
            //U = U;
            //B = B;
          }
         else if(V.first < triplet.U.first && V.second <= triplet.U.second)
         {
            //A = A;
            triplet.U = V;
            triplet.B = triplet.U;
          }
          else if(V.first > triplet.U.first && V.second >= triplet.U.second)
	      {
            //A = A;
            triplet.B = V;
            //U = U;
         }
         else if(V.first > triplet.U.first && V.second <= triplet.U.second)
         {
            triplet.A = triplet.U;
            triplet.U = V;
            //B = B;
         }
         else if(abs(V.first - triplet.U.first) < numeric_limits<double>::min())
		 {
            buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                   << "Vector<double> calculate_golden_section_directional_point(double, const Vector<double>, double) const method.\n"
                   << "Both interior points have the same ordinate.\n";

            cout << buffer.str() << endl;

	        break;
		 }
	     else
	     {
            buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                   << "Vector<double> calculate_golden_section_directional_point(double, const Vector<double>, double) const method.\n" 
                   << "Unknown set:\n" 
                     << "A = (" << triplet.A.first << "," << triplet.A.second << ")\n"
                     << "B = (" << triplet.B.first << "," << triplet.B.second << ")\n"
                     << "U = (" << triplet.U.first << "," << triplet.U.second << ")\n"
                     << "V = (" << V.first << "," << V.second << ")\n";
         
	        throw logic_error(buffer.str());
	     }

		 // Check triplet

         triplet.check();

      }while(triplet.B.second - triplet.A.second > loss_tolerance);

      return(triplet.U);
   }
   catch(const logic_error& e)
   {  
      cerr << e.what() << endl;

      pair<double,double> X;
      X.first = initial_learning_rate;
      X.second = loss_index_pointer->calculate_training_loss(training_direction, X.first);

       if(X.second > loss)
	   {
          X.first = 0.0;
          X.second = 0.0;
	   }

      return X;
   }
}


/// Returns the training rate by searching in a given direction to locate the minimum of the loss
/// function in that direction. It uses the Brent's method.
/// @param loss Neural network loss value.
/// @param training_direction Training direction vector.
/// @param initial_learning_rate Initial training rate in line minimization.

pair<double, double> LearningRateAlgorithm::calculate_Brent_method_directional_point(const double& loss,
                                                                               const Vector<double>& training_direction,
                                                                               const double& initial_learning_rate) const
{
   ostringstream buffer;

   // Bracket minimum

   try
   {
      Triplet triplet = calculate_bracketing_triplet(loss, training_direction, initial_learning_rate);

      size_t count = 0;

      if(triplet.A == triplet.B)
      {
         return(triplet.A);
	  }

      pair<double, double> V;

      // Reduce the interval

      while(abs(triplet.B.second - triplet.A.second) > loss_tolerance)
      {
          try
	      {
            V.first = calculate_Brent_method_learning_rate(triplet);
	      }
         catch(const logic_error&)
	      {
              return triplet.minimum();
	      }

         // Calculate loss for V

         V.second = loss_index_pointer->calculate_training_loss(training_direction, V.first);
         count++;

         // Update points
 
        if(V.first <= triplet.U.first)
        {
            if(V.second >= triplet.U.second)
             {
               triplet.A = V;
             }
            else if(V.second <= triplet.U.second)
            {
               triplet.B = triplet.U;
               triplet.U = V;
            }
        }
        else if(V.first >= triplet.U.first)
        {
            if(V.second >= triplet.U.second)
            {
               triplet.B = V;
            }
            else if(V.second <= triplet.U.second)
            {
               triplet.A = triplet.U;
               triplet.U = V;
            }
        }
	     else
	     {
            buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
                   << "Vector<double> calculate_Brent_method_directional_point(double, const Vector<double>, double) const method.\n" 
                   << "Unknown set:\n" 
                     << "A = (" << triplet.A.first << "," << triplet.A.second << ")\n"
                     << "B = (" << triplet.B.first << "," << triplet.B.second << ")\n"
                     << "U = (" << triplet.U.first << "," << triplet.U.second << ")\n"
                     << "V = (" << V.first << "," << V.second << ")\n";
         
	        throw logic_error(buffer.str());
	     }

		 // Check triplet

         triplet.check();
      }

      return(triplet.U);
   }
   catch(range_error& e) // Interval is of length 0
   {  
      cerr << e.what() << endl;

      pair<double, double> A;
      A.first = 0.0;
      A.second = loss;

      return(A);
   }
   catch(const logic_error& e)
   {  
      cerr << e.what() << endl;

      pair<double, double> X;
      X.first = initial_learning_rate;
      X.second = loss_index_pointer->calculate_training_loss(training_direction, X.first);

      if(X.second > loss)
	  {
         X.first = 0.0;
         X.second = 0.0;
	  }

      return X;
   }
}


/// Calculates the golden section point within a minimum interval defined by three points.
/// @param triplet Triplet containing a minimum.

double LearningRateAlgorithm::calculate_golden_section_learning_rate(const Triplet& triplet) const
{
    double learning_rate;

   if(triplet.U.first < triplet.A.first + 0.5*(triplet.B.first - triplet.A.first))
   {
      learning_rate = triplet.A.first + 0.618*(triplet.B.first - triplet.A.first);
   }
   else
   {
      learning_rate = triplet.A.first + 0.382*(triplet.B.first - triplet.A.first);
   }

    #ifdef __OPENNN_DEBUG__

    if(learning_rate < triplet.A.first)
    {
       ostringstream buffer;

       buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
              << "double calculate_golden_section_learning_rate(const Triplet&) const method.\n"
              << "Training rate(" << learning_rate << ") is less than triplet left point(" << triplet.A.first << ").\n";

       throw logic_error(buffer.str());
    }

    if(learning_rate > triplet.B.first)
    {
       ostringstream buffer;

       buffer << "OpenNN Error: LearningRateAlgorithm class.\n"
              << "double calculate_golden_section_learning_rate(const Triplet&) const method.\n"
              << "Training rate(" << learning_rate << ") is greater than triplet right point(" << triplet.B.first << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    return(learning_rate);
}


/// Returns the minimimal training rate of a parabola defined by three directional points.
/// @param triplet Triplet containing a minimum.

double LearningRateAlgorithm::calculate_Brent_method_learning_rate(const Triplet& triplet) const
{
  const double c = -(triplet.A.second*(triplet.U.first-triplet.B.first)
  + triplet.U.second*(triplet.B.first-triplet.A.first)
  + triplet.B.second*(triplet.A.first-triplet.U.first))/((triplet.A.first-triplet.U.first)*(triplet.U.first-triplet.B.first)*(triplet.B.first-triplet.A.first));

   if(abs(c) < numeric_limits<double>::min())
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
             << "double calculate__method_learning_rate(Vector<double>&, Vector<double>&, Vector<double>&) const method.\n"
             << "Parabola cannot be constructed.\n";

      throw logic_error(buffer.str());
   }
   else if(c < 0) 
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
             << "double calculate_Brent_method_learning_rate(Vector<double>&, Vector<double>&, Vector<double>&) const method.\n"
             << "Parabola does not have a minimum but a maximum.\n";

      throw logic_error(buffer.str());
   }

   const double b = (triplet.A.second*(triplet.U.first*triplet.U.first-triplet.B.first*triplet.B.first)
   + triplet.U.second*(triplet.B.first*triplet.B.first-triplet.A.first*triplet.A.first)
   + triplet.B.second*(triplet.A.first*triplet.A.first-triplet.U.first*triplet.U.first))/((triplet.A.first-triplet.U.first)*(triplet.U.first-triplet.B.first)*(triplet.B.first-triplet.A.first));

   const double Brent_method_learning_rate = -b/(2.0*c);

   if(Brent_method_learning_rate < triplet.A.first || Brent_method_learning_rate > triplet.B.first)
   {
       ostringstream buffer;

      buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
             << "double calculate_parabola_minimal_learning_rate(Vector<double>&, Vector<double>&, Vector<double>&) const method.\n"
             << "Brent method training rate is not inside interval.\n"
             << "Interval:(" << triplet.A.first << "," << triplet.B.first << ")\n"
	         << "Brent method training rate: " << Brent_method_learning_rate << endl;

      throw logic_error(buffer.str());
   }

   return(Brent_method_learning_rate);
}


/// Returns a default string representation in XML-type format of the optimization algorithm object.
/// This containts the training operators, the training parameters, stopping criteria and other stuff.

tinyxml2::XMLDocument* LearningRateAlgorithm::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Optimization algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("LearningRateAlgorithm");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Training rate method
   {
   element = document->NewElement("LearningRateMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_learning_rate_method().c_str());
   element->LinkEndChild(text);
   }

   // Bracketing factor
//   {
//   element = document->NewElement("BracketingFactor");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << bracketing_factor;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Loss tolerance
   {
   element = document->NewElement("LossTolerance");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << loss_tolerance;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Warning training rate 
//   {
//   element = document->NewElement("WarningLearningRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_learning_rate;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Error training rate
//   {
//   element = document->NewElement("ErrorLearningRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_learning_rate;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Display warnings
//   {
//   element = document->NewElement("Display");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << display;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   return document;
}


//void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the learning rate algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void LearningRateAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    file_stream.OpenElement("LearningRateAlgorithm");

    // Training rate method

    file_stream.OpenElement("LearningRateMethod");

    file_stream.PushText(write_learning_rate_method().c_str());

    file_stream.CloseElement();

    // Training rate tolerance

    file_stream.OpenElement("LearningRateTolerance");

    buffer.str("");
    buffer << loss_tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


/// Loads a learning rate algorithm object from a XML-type file.
/// Please mind about the file format, wich is specified in the manual. 
/// @param document TinyXML document with the learning rate algorithm members.

void LearningRateAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
   const tinyxml2::XMLElement* root_element = document.FirstChildElement("LearningRateAlgorithm");

   if(!root_element)
   {
       ostringstream buffer;

       buffer << "OpenNN Exception: LearningRateAlgorithm class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Training rate algorithm element is nullptr.\n";

       throw logic_error(buffer.str());
   }

   // Training rate method
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("LearningRateMethod");

       if(element)
       {
          string new_learning_rate_method = element->GetText();

          try
          {
             set_learning_rate_method(new_learning_rate_method);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Loss tolerance
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossTolerance");

       if(element)
       {
          const double new_loss_tolerance = atof(element->GetText());

          try
          {
             set_loss_tolerance(new_loss_tolerance);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Warning training rate 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningLearningRate");

       if(element)
       {
          const double new_warning_learning_rate = atof(element->GetText());

          try
          {
             set_warning_learning_rate(new_warning_learning_rate);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Error training rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorLearningRate");

       if(element)
       {
          const double new_error_learning_rate = atof(element->GetText());

          try
          {
             set_error_learning_rate(new_error_learning_rate);
          }
          catch(const logic_error& e)
          {
             cerr << e.what() << endl;
          }
       }
   }

   // Display warnings
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

       if(element)
       {
          const string new_display = element->GetText();

          try
          {
             set_display(new_display != "0");
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
