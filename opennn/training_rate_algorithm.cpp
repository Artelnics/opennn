/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T R A I N I N G   R A T E   A L G O R I T H M   C L A S S                                                  */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "training_rate_algorithm.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a training rate algorithm object not associated to any loss functional object.  
/// It also initializes the class members to their default values. 

TrainingRateAlgorithm::TrainingRateAlgorithm(void)
 : loss_index_pointer(NULL)
{ 
   set_default();
}


// GENERAL CONSTRUCTOR

/// General constructor. 
/// It creates a training rate algorithm associated to a loss functional.
/// It also initializes the class members to their default values. 
/// @param new_loss_index_pointer Pointer to a loss functional object.

TrainingRateAlgorithm::TrainingRateAlgorithm(LossIndex* new_loss_index_pointer)
 : loss_index_pointer(new_loss_index_pointer)
{
   set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a training rate algorithm object not associated to any loss functional object.  
/// It also loads the class members from a XML document. 
/// @param document Pointer to a TinyXML document->

TrainingRateAlgorithm::TrainingRateAlgorithm(const tinyxml2::XMLDocument& document)
 : loss_index_pointer(NULL)
{ 
   from_XML(document);
}


// DESTRUCTOR 

/// Destructor

TrainingRateAlgorithm::~TrainingRateAlgorithm(void)
{ 
}


// METHODS

// LossIndex* get_loss_index_pointer(void) const method

/// Returns a pointer to the loss functional object
/// to which the training rate algorithm is associated.
/// If the loss functional pointer is NULL, this method throws an exception.

LossIndex* TrainingRateAlgorithm::get_loss_index_pointer(void) const
{
    #ifdef __OPENNN_DEBUG__

    if(!loss_index_pointer)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
               << "LossIndex* get_loss_index_pointer(void) const method.\n"
               << "Loss index pointer is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    #endif

   return(loss_index_pointer);
}


// bool has_loss_index(void) const method

/// Returns true if this training rate algorithm has an associated loss functional,
/// and false otherwise.

bool TrainingRateAlgorithm::has_loss_index(void) const
{
    if(loss_index_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// const TrainingRateMethod& get_training_rate_method(void) const method

/// Returns the training rate method used for training.

const TrainingRateAlgorithm::TrainingRateMethod& TrainingRateAlgorithm::get_training_rate_method(void) const
{
   return(training_rate_method);
}


// std::string write_training_rate_method(void) const method

/// Returns a string with the name of the training rate method to be used.

std::string TrainingRateAlgorithm::write_training_rate_method(void) const
{
   switch(training_rate_method)
   {
      case Fixed:
      {
         return("Fixed");
	   }
      break;

      case GoldenSection:
      {
         return("GoldenSection");
	   }
      break;

      case BrentMethod:
      {
         return("BrentMethod");
	   }
      break;

      default:
      {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
                << "std::string get_training_rate_method(void) const method.\n"
                << "Unknown training rate method.\n";
 
         throw std::logic_error(buffer.str());
	   }
      break;
   }
}


// const double& get_bracketing_factor(void) const method

/// Returns the increase factor when bracketing a minimum in line minimization.

const double& TrainingRateAlgorithm::get_bracketing_factor(void) const
{
   return(bracketing_factor);       
}


// const double& get_training_rate_tolerance(void) const method

/// Returns the tolerance value in line minimization.

const double& TrainingRateAlgorithm::get_training_rate_tolerance(void) const
{
   return(training_rate_tolerance);
}


// const double& get_warning_training_rate(void) const method

/// Returns the training rate value at wich a warning message is written to the screen during line
/// minimization.

const double& TrainingRateAlgorithm::get_warning_training_rate(void) const
{
   return(warning_training_rate);
}


// const double& get_error_training_rate(void) const method

/// Returns the training rate value at wich the line minimization algorithm is assumed to fail when
/// bracketing a minimum.

const double& TrainingRateAlgorithm::get_error_training_rate(void) const
{
   return(error_training_rate);
}


// const bool& get_display(void) const method

/// Returns true if messages from this class can be displayed on the screen, or false if messages from
/// this class can't be displayed on the screen.

const bool& TrainingRateAlgorithm::get_display(void) const
{
   return(display);
}


// void set(void) method

/// Sets the loss functional pointer to NULL.
/// It also sets the rest of members to their default values. 

void TrainingRateAlgorithm::set(void)
{
   loss_index_pointer = NULL;
   set_default();
}


// void set(LossIndex*) method

/// Sets a new loss functional pointer.
/// It also sets the rest of members to their default values. 
/// @param new_loss_index_pointer Pointer to a loss functional object. 

void TrainingRateAlgorithm::set(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;
   set_default();
}


// void set_default(void) method 

/// Sets the members of the training rate algorithm to their default values.

void TrainingRateAlgorithm::set_default(void)
{
   // TRAINING OPERATORS

   training_rate_method = BrentMethod;

   // TRAINING PARAMETERS

   bracketing_factor = 1.5;

   training_rate_tolerance = 1.0e-6;

   warning_training_rate = 1.0e3;

   error_training_rate = 1.0e6;

   // UTILITIES

   display = true;
}


// void set_loss_index_pointer(LossIndex*) method

/// Sets a pointer to a loss functional object to be associated to the training algorithm.
/// @param new_loss_index_pointer Pointer to a loss functional object.

void TrainingRateAlgorithm::set_loss_index_pointer(LossIndex* new_loss_index_pointer)
{
   loss_index_pointer = new_loss_index_pointer;
}


// void set_training_rate_method(const TrainingRateMethod&) method

/// Sets a new training rate method to be used for training.
/// @param new_training_rate_method Training rate method.

void TrainingRateAlgorithm::set_training_rate_method(const TrainingRateAlgorithm::TrainingRateMethod& new_training_rate_method)
{
   training_rate_method = new_training_rate_method;
}


// void set_training_rate_method(const std::string&) method

/// Sets the method for obtaining the training rate from a string with the name of the method.
/// @param new_training_rate_method Name of training rate method ("Fixed", "GoldenSection", "BrentMethod" or "UserTrainingRate"). 

void TrainingRateAlgorithm::set_training_rate_method(const std::string& new_training_rate_method)
{
   if(new_training_rate_method == "Fixed")
   {
      training_rate_method = Fixed;
   }
   else if(new_training_rate_method == "GoldenSection")
   {
      training_rate_method = GoldenSection;
   }
   else if(new_training_rate_method == "BrentMethod")
   {
      training_rate_method = BrentMethod;
   }
   else
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
             << "void set_method(const std::string&) method.\n"
			 << "Unknown training rate method: " << new_training_rate_method << ".\n";
   
      throw std::logic_error(buffer.str());
   }
}


// void set_bracketing_factor(const double&) method

/// Sets a new increase factor value to be used for line minimization when bracketing a minimum.
/// @param new_bracketing_factor Bracketing factor value.

void TrainingRateAlgorithm::set_bracketing_factor(const double& new_bracketing_factor)
{ 
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_bracketing_factor < 0.0) 
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
             << "void set_bracketing_factor(const double&) method.\n"
             << "Bracketing factor must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   bracketing_factor = new_bracketing_factor;
}


// void set_training_rate_tolerance(const double&) method

/// Sets a new tolerance value to be used in line minimization.
/// @param new_training_rate_tolerance Tolerance value in line minimization.

void TrainingRateAlgorithm::set_training_rate_tolerance(const double& new_training_rate_tolerance)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 
                                      
   if(new_training_rate_tolerance < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
             << "void set_training_rate_tolerance(const double&) method.\n"
             << "Tolerance must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set training rate tolerance

   training_rate_tolerance = new_training_rate_tolerance;
}


// void set_warning_training_rate(const double&) method

/// Sets a new training rate value at wich a warning message is written to the screen during line
/// minimization.
/// @param new_warning_training_rate Warning training rate value.

void TrainingRateAlgorithm::set_warning_training_rate(const double& new_warning_training_rate)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_warning_training_rate < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n" 
             << "void set_warning_training_rate(const double&) method.\n"
             << "Warning training rate must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   warning_training_rate = new_warning_training_rate;
}


// void set_error_training_rate(const double&) method

/// Sets a new training rate value at wich a the line minimization algorithm is assumed to fail when
/// bracketing a minimum.
/// @param new_error_training_rate Error training rate value.

void TrainingRateAlgorithm::set_error_training_rate(const double& new_error_training_rate)
{
   // Control sentence (if debug)

   #ifdef __OPENNN_DEBUG__ 

   if(new_error_training_rate < 0.0)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
             << "void set_error_training_rate(const double&) method.\n"
             << "Error training rate must be equal or greater than 0.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   // Set error training rate

   error_training_rate = new_error_training_rate;
}


// void set_display(const bool&) method

/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void TrainingRateAlgorithm::set_display(const bool& new_display)
{
   display = new_display;
}


// Vector<double> calculate_directional_point(const double&, const Vector<double>&, const double&) const method

/// Returns a vector with two elements:
/// (i) the training rate calculated by means of the corresponding algorithm, and
/// (ii) the loss for that training rate.
/// @param loss Initial Performance value.
/// @param training_direction Initial training direction.
/// @param initial_training_rate Initial training rate to start the algorithm. 

Vector<double> TrainingRateAlgorithm::calculate_directional_point(const double& loss, const Vector<double>& training_direction, const double& initial_training_rate) const 
{
   #ifdef __OPENNN_DEBUG__ 

   if(loss_index_pointer == NULL)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Error: TrainingRateAlgorithm class.\n"
             << "Vector<double> calculate_directional_point(const double&, const Vector<double>&, const double&) const method.\n"
             << "Pointer to loss functional is NULL.\n";

      throw std::logic_error(buffer.str());
   }

   #endif

   #ifdef __OPENNN_DEBUG__ 

   const NeuralNetwork* neural_network_pointer = loss_index_pointer->get_neural_network_pointer();

   if(neural_network_pointer == NULL)
   {
      std::ostringstream buffer;

      buffer << "OpenNN Error: TrainingRateAlgorithm class.\n"
             << "Vector<double> calculate_directional_point(const double&, const Vector<double>&, const double&) const method.\n"
             << "Pointer to neural network is NULL.\n";

      throw std::logic_error(buffer.str());
   }

   #endif
   
   switch(training_rate_method)
   {
      case TrainingRateAlgorithm::Fixed:
      {
         return(calculate_fixed_directional_point(loss, training_direction, initial_training_rate));
      }
      break;

      case TrainingRateAlgorithm::GoldenSection:
      {
         return(calculate_golden_section_directional_point(loss, training_direction, initial_training_rate));
      }
      break;

      case TrainingRateAlgorithm::BrentMethod:
      {
         return(calculate_Brent_method_directional_point(loss, training_direction, initial_training_rate));
      }
      break;

	  default:
	  {
         std::ostringstream buffer;

         buffer << "OpenNN Exception: TrainingRateAlgorithm class\n"
                << "Vector<double> calculate_directional_point(const double&, const Vector<double>&, const double&) const method.\n"
                << "Unknown training rate method.\n";

         throw std::logic_error(buffer.str());
	   }
   }
}


// Triplet calculate_bracketing_triplet(const double&, const Vector<double>&, const double&) const method

/// Returns bracketing triplet.
/// This algorithm is used by line minimization algorithms. 
/// @param loss Initial Performance value.
/// @param training_direction Initial training direction.
/// @param initial_training_rate Initial training rate to start the algorithm. 

TrainingRateAlgorithm::Triplet TrainingRateAlgorithm::calculate_bracketing_triplet(
        const double& loss,
        const Vector<double>& training_direction,
        const double& initial_training_rate) const
{    
    Triplet triplet;

   if(training_direction == 0.0)
   {
       triplet.A[0] = 0.0;
       triplet.A[1] = loss;

       triplet.U = triplet.A;

       triplet.B = triplet.A;

       return(triplet);
   }

   if(initial_training_rate == 0.0)
   {
       triplet.A[0] = 0.0;
       triplet.A[1] = loss;

       triplet.U = triplet.A;

       triplet.B = triplet.A;

       return(triplet);
   }

   // Left point

   triplet.A[0] = 0.0;
   triplet.A[1] = loss;

   // Right point

   triplet.B[0] = initial_training_rate;
   triplet.B[1] = loss_index_pointer->calculate_loss(training_direction, triplet.B[0]);

   while(triplet.A[1] > triplet.B[1])
   {
      triplet.A = triplet.B;

      triplet.B[0] *= bracketing_factor;
      triplet.B[1] = loss_index_pointer->calculate_loss(training_direction, triplet.B[0]);

      if(triplet.B[0] > error_training_rate)
      {
          std::ostringstream buffer;

          buffer << "OpenNN Warning: TrainingRateAlgorithm class.\n"
                 << "Vector<double> calculate_bracketing_triplet(double, const Vector<double>&, double) const method\n."
                 << "Right point is " << triplet.B[0] << "." << std::endl;

          buffer << "Training loss: " << loss << "\n"
                 << "Training direction: " << training_direction << "\n"
                 << "Initial training rate: " << initial_training_rate << std::endl;

          throw std::logic_error(buffer.str());

          return(triplet);
      }
   }

    // Interior point

   triplet.U[0] = triplet.A[0] + (triplet.B[0] - triplet.A[0])/2.0;
   triplet.U[1] = loss_index_pointer->calculate_loss(training_direction, triplet.U[0]);

   while(triplet.A[1] < triplet.U[1])
   {
      triplet.U[0] = triplet.A[0] + (triplet.U[0]-triplet.A[0])/bracketing_factor;
      triplet.U[1] = loss_index_pointer->calculate_loss(training_direction, triplet.U[0]);

      if(triplet.U[0] - triplet.A[0] <= training_rate_tolerance)
      {
          triplet.U = triplet.A;
          triplet.B = triplet.A;

          triplet.check();

         return(triplet);
      }
   }

   triplet.check();

   return(triplet);
}


// Vector<double> calculate_fixed_directional_point(const double&, const Vector<double>&, const double&) const method

/// Returns a vector with two elements, a fixed training rate,
/// and the loss for that training rate. 
/// @param training_direction Training direction for the directional point.
/// @param initial_training_rate Training rate for the directional point.


Vector<double> TrainingRateAlgorithm::calculate_fixed_directional_point(const double&, const Vector<double>& training_direction, const double& initial_training_rate) const
{
   Vector<double> directional_point(2);

   directional_point[0] = initial_training_rate;
   directional_point[1] = loss_index_pointer->calculate_loss(training_direction, initial_training_rate);

   return(directional_point);
}


// Vector<double> calculate_golden_section_directional_point(double, Vector<double>, double) const method

/// Returns the training rate by searching in a given direction to locate the minimum of the objective
/// function in that direction. It uses the golden section method.
/// @param loss Neural multilayer_perceptron_pointer's loss value.
/// @param training_direction Training direction vector.
/// @param initial_training_rate Initial training rate in line minimization.

Vector<double> TrainingRateAlgorithm::calculate_golden_section_directional_point
(const double& loss, const Vector<double>& training_direction, const double& initial_training_rate) const
{
   std::ostringstream buffer;

   // Bracket minimum

   try
   {
      Triplet triplet = calculate_bracketing_triplet(loss, training_direction, initial_training_rate);

      if(triplet.has_length_zero())
	  {
         return(triplet.A);
	  }

      Vector<double> V(2);

      // Reduce the interval

      do
      {
         V[0] = calculate_golden_section_training_rate(triplet);
         V[1] = loss_index_pointer->calculate_loss(training_direction, V[0]);

         // Update points
 
         if(V[0] < triplet.U[0] && V[1] >= triplet.U[1])
	      {
            triplet.A = V;
            //U = U;
            //B = B;
          }
         else if(V[0] < triplet.U[0] && V[1] <= triplet.U[1])
         {
            //A = A;
            triplet.U = V;
            triplet.B = triplet.U;
          }
          else if(V[0] > triplet.U[0] && V[1] >= triplet.U[1])
	      {
            //A = A;
            triplet.B = V;
            //U = U;
         }
         else if(V[0] > triplet.U[0] && V[1] <= triplet.U[1])
         {
            triplet.A = triplet.U;
            triplet.U = V;
            //B = B;
         }
         else if(V[0] == triplet.U[0])
		 {
            buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
                   << "Vector<double> calculate_golden_section_directional_point(double, const Vector<double>, double) const method.\n"
                   << "Both interior points have the same ordinate.\n";

            std::cout << buffer.str() << std::endl;

	        break;
		 }
	     else
	     {
            buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n" 
                   << "Vector<double> calculate_golden_section_directional_point(double, const Vector<double>, double) const method.\n" 
                   << "Unknown set:\n" 
                     << "A = (" << triplet.A[0] << "," << triplet.A[1] << ")\n"
                     << "B = (" << triplet.B[0] << "," << triplet.B[1] << ")\n"
                     << "U = (" << triplet.U[0] << "," << triplet.U[1] << ")\n"
                     << "V = (" << V[0] << "," << V[1] << ")\n";
         
	        throw std::logic_error(buffer.str());
	     }

		 // Check triplet

         triplet.check();

      }while(triplet.B[0] - triplet.A[0] > training_rate_tolerance);

      return(triplet.U);
   }
   catch(const std::logic_error& e)
   {  
      std::cerr << e.what() << std::endl;

	  Vector<double> X(2);
      X[0] = initial_training_rate;
      X[1] = loss_index_pointer->calculate_loss(training_direction, X[0]);

	   if(X[1] > loss)
	   {
	      X[0] = 0.0;
	      X[1] = 0.0;
	   }

      return(X);
   }
}


// Vector<double> calculate_Brent_method_directional_point(const double&, const Vector<double>, const double&) const method

/// Returns the training rate by searching in a given direction to locate the minimum of the loss
/// function in that direction. It uses the Brent's method.
/// @param loss Neural network loss value.
/// @param training_direction Training direction vector.
/// @param initial_training_rate Initial training rate in line minimization.

Vector<double> TrainingRateAlgorithm::calculate_Brent_method_directional_point
(const double& loss, const Vector<double>& training_direction, const double& initial_training_rate) const 
{
   std::ostringstream buffer;

   // Bracket minimum

   try
   {
      Triplet triplet = calculate_bracketing_triplet(loss, training_direction, initial_training_rate);

      if(triplet.A == triplet.B)
	  {
         return(triplet.A);
	  }

      Vector<double> V(2);

      // Reduce the interval

      while(triplet.B[0] - triplet.A[0] > training_rate_tolerance)
      {
          try
	      {
            V[0] = calculate_Brent_method_training_rate(triplet);
	      }
         catch(const std::logic_error&)
	      {
            V[0] = calculate_golden_section_training_rate(triplet);
	      }

         // Calculate loss for V

         V[1] = loss_index_pointer->calculate_loss(training_direction, V[0]);

         // Update points
 
         if(V[0] < triplet.U[0] && V[1] >= triplet.U[1])
	      {
            triplet.A = V;
            //B = B;
            //U = U;
	      }
         else if(V[0] < triplet.U[0] && V[1] <= triplet.U[1])
         {
            //A = A;
            triplet.B = triplet.U;
            triplet.U = V;
	     }
         else if(V[0] > triplet.U[0] && V[1] >= triplet.U[1])
         {
            //A = A;
            triplet.B = V;
            //U = U;
         }
         else if(V[0] > triplet.U[0] && V[1] <= triplet.U[1])
         {
            triplet.A = triplet.U;
            //B = B;
            triplet.U = V;
         }
           else if(V[0] == triplet.U[0])
		   {
            buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
                   << "Vector<double> calculate_Brent_method_directional_point(double, const Vector<double>, double) const method.\n"
                   << "Both interior points have the same ordinate.\n";

	        break;
		  }
	     else
	     {
            buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n" 
                   << "Vector<double> calculate_Brent_method_directional_point(double, const Vector<double>, double) const method.\n" 
                   << "Unknown set:\n" 
                     << "A = (" << triplet.A[0] << "," << triplet.A[1] << ")\n"
                     << "B = (" << triplet.B[0] << "," << triplet.B[1] << ")\n"
                     << "U = (" << triplet.U[0] << "," << triplet.U[1] << ")\n"
		             << "V = (" << V[0] << "," << V[1] << ")\n";
         
	        throw std::logic_error(buffer.str());
	     }

		 // Check triplet

         triplet.check();
      }

      return(triplet.U);
   }
   catch(std::range_error& e) // Interval is of length 0
   {  
      std::cerr << e.what() << std::endl;

      Vector<double> A(2);
      A[0] = 0.0;
      A[1] = loss;

      return(A);
   }
   catch(const std::logic_error& e)
   {  
      std::cerr << e.what() << std::endl;

	  Vector<double> X(2);
      X[0] = initial_training_rate;
      X[1] = loss_index_pointer->calculate_loss(training_direction, X[0]);

      if(X[1] > loss)
	  {
	     X[0] = 0.0;
	     X[1] = 0.0;
	  }

      return(X);
   }
}


// double calculate_golden_section_training_rate(const Triplet&) const method

/// Calculates the golden section point within a minimum interval defined by three points.
/// @param triplet Triplet containing a minimum.

double TrainingRateAlgorithm::calculate_golden_section_training_rate(const Triplet& triplet) const
{
//   const double tau = 0.382; // (3.0-sqrt(5.0))/2.0   

    double training_rate;

   if(triplet.U[0] < triplet.A[0] + 0.5*(triplet.B[0] - triplet.A[0]))
   {
      training_rate = triplet.A[0] + 0.618*(triplet.B[0] - triplet.A[0]);
   }
   else
   {
      training_rate = triplet.A[0] + 0.382*(triplet.B[0] - triplet.A[0]);
   }

    #ifdef __OPENNN_DEBUG__

    if(training_rate < triplet.A[0])
    {
       std::ostringstream buffer;

       buffer << "OpenNN Error: TrainingRateAlgorithm class.\n"
              << "double calculate_golden_section_training_rate(const Triplet&) const method.\n"
              << "Training rate (" << training_rate << ") is less than triplet left point (" << triplet.A[0] << ").\n";

       throw std::logic_error(buffer.str());
    }

    if(training_rate > triplet.B[0])
    {
       std::ostringstream buffer;

       buffer << "OpenNN Error: TrainingRateAlgorithm class.\n"
              << "double calculate_golden_section_training_rate(const Triplet&) const method.\n"
              << "Training rate (" << training_rate << ") is greater than triplet right point (" << triplet.B[0] << ").\n";

       throw std::logic_error(buffer.str());
    }

    #endif

    return(training_rate);
}


// double calculate_Brent_method_training_rate(const Triplet&) const method

/// Returns the minimimal training rate of a parabola defined by three directional points.
/// @param triplet Triplet containing a minimum.

double TrainingRateAlgorithm::calculate_Brent_method_training_rate(const Triplet& triplet) const
{
  const double c = -(triplet.A[1]*(triplet.U[0]-triplet.B[0])
  + triplet.U[1]*(triplet.B[0]-triplet.A[0])
  + triplet.B[1]*(triplet.A[0]-triplet.U[0]))/((triplet.A[0]-triplet.U[0])*(triplet.U[0]-triplet.B[0])*(triplet.B[0]-triplet.A[0]));

   if(c == 0) 
   {
       std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
             << "double calculate_Brent_method_training_rate(Vector<double>&, Vector<double>&, Vector<double>&) const method.\n"
             << "Parabola cannot be constructed.\n";

      throw std::logic_error(buffer.str());
   }
   else if(c < 0) 
   {
       std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
             << "double calculate_Brent_method_training_rate(Vector<double>&, Vector<double>&, Vector<double>&) const method.\n"
             << "Parabola does not have a minimum but a maximum.\n";

      throw std::logic_error(buffer.str());
   }

   const double b = (triplet.A[1]*(triplet.U[0]*triplet.U[0]-triplet.B[0]*triplet.B[0])
   + triplet.U[1]*(triplet.B[0]*triplet.B[0]-triplet.A[0]*triplet.A[0])
   + triplet.B[1]*(triplet.A[0]*triplet.A[0]-triplet.U[0]*triplet.U[0]))/((triplet.A[0]-triplet.U[0])*(triplet.U[0]-triplet.B[0])*(triplet.B[0]-triplet.A[0]));

   const double Brent_method_training_rate = -b/(2.0*c);

   if(Brent_method_training_rate <= triplet.A[0] || Brent_method_training_rate >= triplet.B[0])
   {
       std::ostringstream buffer;

      buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
             << "double calculate_parabola_minimal_training_rate(Vector<double>&, Vector<double>&, Vector<double>&) const method.\n"
             << "Brent method training rate is not inside interval.\n"
             << "Interval: (" << triplet.A[0] << "," << triplet.B[0] << ")\n"
	         << "Brent method training rate: " << Brent_method_training_rate << std::endl;

      throw std::logic_error(buffer.str());
   }

   return(Brent_method_training_rate);
}


// tinyxml2::XMLDocument* to_XML(void) const method

/// Returns a default string representation in XML-type format of the training algorithm object.
/// This containts the training operators, the training parameters, stopping criteria and other stuff.

tinyxml2::XMLDocument* TrainingRateAlgorithm::to_XML(void) const
{
   std::ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Training algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("TrainingRateAlgorithm");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = NULL;
   tinyxml2::XMLText* text = NULL;

   // Training rate method
   {
   element = document->NewElement("TrainingRateMethod");
   root_element->LinkEndChild(element);

   text = document->NewText(write_training_rate_method().c_str());
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

   // Training rate tolerance 
   {
   element = document->NewElement("TrainingRateTolerance");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << training_rate_tolerance;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Warning training rate 
//   {
//   element = document->NewElement("WarningTrainingRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << warning_training_rate;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Error training rate
//   {
//   element = document->NewElement("ErrorTrainingRate");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << error_training_rate;

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

   return(document);
}


//void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the training rate algorithm object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void TrainingRateAlgorithm::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    file_stream.OpenElement("TrainingRateAlgorithm");

    // Training rate method

    file_stream.OpenElement("TrainingRateMethod");

    file_stream.PushText(write_training_rate_method().c_str());

    file_stream.CloseElement();

    // Training rate tolerance

    file_stream.OpenElement("TrainingRateTolerance");

    buffer.str("");
    buffer << training_rate_tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    file_stream.CloseElement();
}


// virtual void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a training rate algorithm object from a XML-type file.
/// Please mind about the file format, wich is specified in the manual. 
/// @param document TinyXML document with the training rate algorithm members.

void TrainingRateAlgorithm::from_XML(const tinyxml2::XMLDocument& document)
{
   const tinyxml2::XMLElement* root_element = document.FirstChildElement("TrainingRateAlgorithm");

   if(!root_element)
   {
       std::ostringstream buffer;

       buffer << "OpenNN Exception: TrainingRateAlgorithm class.\n"
              << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
              << "Training rate algorithm element is NULL.\n";

       throw std::logic_error(buffer.str());
   }

   // Training rate method
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrainingRateMethod");

       if(element)
       {
          std::string new_training_rate_method = element->GetText();

          try
          {
             set_training_rate_method(new_training_rate_method);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Bracketing factor
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("BracketingFactor");

       if(element)
       {
          const double new_bracketing_factor = atof(element->GetText());

          try
          {
             set_bracketing_factor(new_bracketing_factor);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }
/*
   // First training rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("FirstTrainingRate");

       if(element)
       {
          const double new_first_training_rate = atof(element->GetText());

          try
          {
             set_first_training_rate(new_first_training_rate);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }
*/
   // Training rate tolerance 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrainingRateTolerance");

       if(element)
       {
          const double new_training_rate_tolerance = atof(element->GetText());

          try
          {
             set_training_rate_tolerance(new_training_rate_tolerance);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Warning training rate 
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("WarningTrainingRate");

       if(element)
       {
          const double new_warning_training_rate = atof(element->GetText());

          try
          {
             set_warning_training_rate(new_warning_training_rate);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Error training rate
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("ErrorTrainingRate");

       if(element)
       {
          const double new_error_training_rate = atof(element->GetText());

          try
          {
             set_error_training_rate(new_error_training_rate);
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
   }

   // Display warnings
   {
       const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

       if(element)
       {
          const std::string new_display = element->GetText();

          try
          {
             set_display(new_display != "0");
          }
          catch(const std::logic_error& e)
          {
             std::cout << e.what() << std::endl;
          }
       }
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
