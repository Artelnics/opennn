/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   S I M U L A T E D   A N N E A L I N G   O R D E R   C L A S S                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/


// OpenNN includes

#include "simulated_annealing_order.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.

SimulatedAnnealingOrder::SimulatedAnnealingOrder()
    : OrderSelectionAlgorithm()
{
    set_default();
}


// TRAINING STRATEGY CONSTRUCTOR

/// Training strategy constructor.
/// @param new_training_strategy_pointer Pointer to a training strategy object.

SimulatedAnnealingOrder::SimulatedAnnealingOrder(TrainingStrategy* new_training_strategy_pointer)
    : OrderSelectionAlgorithm(new_training_strategy_pointer)
{
    set_default();
}


// FILE CONSTRUCTOR

/// File constructor.
/// @param file_name Name of XML simulated annealing order file.

SimulatedAnnealingOrder::SimulatedAnnealingOrder(const string& file_name)
    : OrderSelectionAlgorithm(file_name)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor.
/// @param simulated_annealing_order_document Pointer to a TinyXML document containing the simulated annealing order data.

SimulatedAnnealingOrder::SimulatedAnnealingOrder(const tinyxml2::XMLDocument& simulated_annealing_order_document)
    : OrderSelectionAlgorithm(simulated_annealing_order_document)
{
    from_XML(simulated_annealing_order_document);
}


// DESTRUCTOR

/// Destructor.

SimulatedAnnealingOrder::~SimulatedAnnealingOrder()
{
}

// METHODS

// const double& get_cooling_rate() const method

/// Returns the temperature reduction factor for the simulated annealing method.

const double& SimulatedAnnealingOrder::get_cooling_rate() const
{
    return(cooling_rate);
}

// const double& get_minimum_temperature() const method

/// Returns the minimum temperature reached in the simulated annealing model order selection algorithm.

const double& SimulatedAnnealingOrder::get_minimum_temperature() const
{
    return(minimum_temperature);
}

// void set_default() method 

/// Sets the members of the model selection object to their default values.

void SimulatedAnnealingOrder::set_default()
{
    cooling_rate = 0.5;

    minimum_temperature = 1.0e-3;
}

// void set_cooling_rate(const double&) method

/// Sets the cooling rate for the simulated annealing.
/// @param new_cooling_rate Temperature reduction factor.

void SimulatedAnnealingOrder::set_cooling_rate(const double& new_cooling_rate)
{
#ifdef __OPENNN_DEBUG__

    if(new_cooling_rate <= 0)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: SimulatedAnnealingOrder class.\n"
               << "void set_cooling_rate(const size_t&) method.\n"
               << "Cooling rate must be greater than 0.\n";

        throw logic_error(buffer.str());
    }

    if(new_cooling_rate >= 1)
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: SimulatedAnnealingOrder class.\n"
               << "void set_cooling_rate(const size_t&) method.\n"
               << "Cooling rate must be less than 1.\n";

        throw logic_error(buffer.str());
    }

#endif

    cooling_rate = new_cooling_rate;
}

// void set_minimum_temperature(const double&) method

/// Sets the minimum temperature for the simulated annealing order selection algorithm.
/// @param new_minimum_temperature Value of the minimum temperature.

void SimulatedAnnealingOrder::set_minimum_temperature(const double& new_minimum_temperature)
{
#ifdef __OPENNN_DEBUG__

    if(new_minimum_temperature < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: SimulatedAnnealingOrder class.\n"
               << "void set_minimum_temperature(const double&) method.\n"
               << "Minimum temperature must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    minimum_temperature = new_minimum_temperature;
}


// size_t get_optimal_selection_error_index() const method

/// Return the index of the optimal individual of the history considering the tolerance.

size_t SimulatedAnnealingOrder::get_optimal_selection_error_index() const
{
    size_t index = 0;

    size_t optimal_order = order_history[0];

    double optimum_error = selection_error_history[0];

    size_t current_order;

    double current_error;

    for(size_t i = 1; i < order_history.size(); i++)
    {
        current_order = order_history[i];
        current_error = selection_error_history[i];

        if((fabs(optimum_error-current_error) < tolerance &&
             optimal_order > current_order) ||
           (fabs(optimum_error-current_error) >= tolerance &&
             current_error < optimum_error)   )
        {
            optimal_order = current_order;
            optimum_error = current_error;

            index = i;
        }
    }

    return(index);
}

// SimulatedAnnealingOrderResults* perform_order_selection() method

/// Perform the order selection with the simulated annealing method.

SimulatedAnnealingOrder::SimulatedAnnealingOrderResults* SimulatedAnnealingOrder::perform_order_selection()
{
    SimulatedAnnealingOrderResults* results = new SimulatedAnnealingOrderResults();

    NeuralNetwork* neural_network_pointer = training_strategy_pointer->get_loss_index_pointer()->get_neural_network_pointer();
    MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    size_t optimal_order, current_order;
    Vector<double> optimum_loss(2);
    Vector<double> current_order_loss(2);
    Vector<double> optimum_parameters, current_parameters;

    double current_training_loss, current_selection_error;

    bool end = false;
    size_t iterations = 0;
    size_t random_failures = 0;
    size_t upper_bound;
    size_t lower_bound;

    time_t beginning_time, current_time;
    double elapsed_time;

    double temperature;
    double boltzmann_probability;
    double random_uniform;

    if(display)
    {
        cout << "Performing order selection with simulated annealing method..." << endl;
    }

    time(&beginning_time);

    optimal_order = (size_t)(minimum_order +
                             calculate_random_uniform(0.,1.)*(maximum_order - minimum_order));
    optimum_loss = perform_model_evaluation(optimal_order);
    optimum_parameters = get_parameters_order(optimal_order);

    current_training_loss = optimum_loss[0];
    current_selection_error = optimum_loss[1];

    temperature = current_selection_error;

    results->order_data.push_back(optimal_order);

    if(reserve_error_data)
    {
        results->loss_data.push_back(current_training_loss);
    }

    if(reserve_selection_error_data)
    {
        results->selection_error_data.push_back(current_selection_error);
    }

    if(reserve_parameters_data)
    {
        results->parameters_data.push_back(optimum_parameters);
    }

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(display)
    {
        cout << "Initial values : " << endl;
        cout << "Hidden perceptrons : " << optimal_order << endl;
        cout << "Final selection error : " << optimum_loss[1] << endl;
        cout << "Final Training loss : " << optimum_loss[0] << endl;
        cout << "Temperature : " << temperature << endl;
        cout << "Elapsed time : " << write_elapsed_time(elapsed_time) << endl;
    }

    while(!end){

        upper_bound = min(maximum_order, optimal_order + (maximum_order-minimum_order)/3);
        if(optimal_order <= (maximum_order-minimum_order)/3)
        {
            lower_bound = minimum_order;
        }
        else
        {
            lower_bound = optimal_order -(maximum_order-minimum_order)/3;
        }

        current_order = (size_t)(lower_bound + calculate_random_uniform(0.,1.)*(upper_bound - lower_bound));
        while(current_order == optimal_order)
        {
            current_order = (size_t)(lower_bound + calculate_random_uniform(0.,1.)*(upper_bound - lower_bound));
            random_failures++;

            if(random_failures >= 5 && optimal_order != minimum_order)
            {
                current_order = optimal_order - 1;
            }
            else if(random_failures >= 5 && optimal_order != maximum_order)
            {
                current_order = optimal_order + 1;
            }
        }

#ifdef __OPENNN_MPI__
        int order_int = (int)current_order;
        MPI_Bcast(&order_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        current_order = order_int;
#endif

        random_failures = 0;

        current_order_loss = perform_model_evaluation(current_order);
        current_training_loss = current_order_loss[0];
        current_selection_error = current_order_loss[1];
        current_parameters = get_parameters_order(current_order);

        boltzmann_probability = min(1.0, exp(-(current_selection_error-optimum_loss[1])/temperature));
        random_uniform = calculate_random_uniform(0.,1.);

#ifdef __OPENNN_MPI__
        MPI_Bcast(&random_uniform, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif

        if(boltzmann_probability > random_uniform)
        {
            optimal_order = current_order;
            optimum_loss = current_order_loss;
            optimum_parameters = get_parameters_order(optimal_order);
        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        results->order_data.push_back(current_order);

        if(reserve_error_data)
        {
            results->loss_data.push_back(current_training_loss);
        }

        if(reserve_selection_error_data)
        {
            results->selection_error_data.push_back(current_selection_error);
        }

        if(reserve_parameters_data)
        {
            results->parameters_data.push_back(current_parameters);
        }

        temperature = cooling_rate*temperature;

        iterations++;

        // Stopping criteria

        if(temperature <= minimum_temperature)
        {
            end = true;

            if(display)
            {
                cout << "Minimum temperature reached." << endl;
            }

            results->stopping_condition = SimulatedAnnealingOrder::MinimumTemperature;
        }
        else if(elapsed_time > maximum_time)
        {
            end = true;

            if(display)
            {
                cout << "Maximum time reached." << endl;
            }

            results->stopping_condition = SimulatedAnnealingOrder::MaximumTime;
        }
        else if(optimum_loss[1] <= selection_error_goal)
        {
            end = true;

            if(display)
            {
                cout << "Selection loss reached." << endl;
            }

            results->stopping_condition = SimulatedAnnealingOrder::SelectionErrorGoal;
        }
        else if(iterations >= maximum_iterations_number)
        {
            end = true;

            if(display)
            {
                cout << "Maximum number of iterations reached." << endl;
            }

            results->stopping_condition = SimulatedAnnealingOrder::MaximumIterations;
        }

        if(display)
        {
            cout << "Iteration : " << iterations << endl;
            cout << "Hidden neurons number : " << optimal_order << endl;
            cout << "Selection loss : " << optimum_loss[1] << endl;
            cout << "Training loss : " << optimum_loss[0] << endl;
            cout << "Current temperature : " << temperature << endl;
            cout << "Elapsed time : " << write_elapsed_time(elapsed_time) << endl;
        }
    }

    size_t optimal_index = get_optimal_selection_error_index();

    optimal_order = order_history[optimal_index];
    optimum_loss[0] = loss_history[optimal_index];
    optimum_loss[1] = selection_error_history[optimal_index];
    optimum_parameters = get_parameters_order(optimal_order);

    if(display)
    {
        cout << "Optimal order : " << optimal_order << endl;
        cout << "Optimum selection error : " << optimum_loss[1] << endl;
        cout << "Corresponding training loss : " << optimum_loss[0] << endl;
    }

    const size_t last_hidden_layer = multilayer_perceptron_pointer->get_layers_number()-2;

    multilayer_perceptron_pointer->set_layer_perceptrons_number(last_hidden_layer, optimal_order);

    multilayer_perceptron_pointer->set_parameters(optimum_parameters);

#ifdef __OPENNN_MPI__
    neural_network_pointer->set_multilayer_perceptron_pointer(multilayer_perceptron_pointer);
#endif

    if(reserve_minimal_parameters)
    {
        results->minimal_parameters = optimum_parameters;
    }

    results->optimal_order = optimal_order;
    results->final_loss = optimum_loss[0];
    results->final_selection_error = optimum_loss[1];
    results->elapsed_time = elapsed_time;
    results->iterations_number = iterations;

    return(results);
}

// Matrix<string> to_string_matrix() const method

/// Writes as matrix of strings the most representative atributes.

Matrix<string> SimulatedAnnealingOrder::to_string_matrix() const
{
    ostringstream buffer;

    Vector<string> labels;
    Vector<string> values;

   // Minimum order

   labels.push_back("Minimum order");

   buffer.str("");
   buffer << minimum_order;

   values.push_back(buffer.str());

   // Maximum order

   labels.push_back("Maximum order");

   buffer.str("");
   buffer << maximum_order;

   values.push_back(buffer.str());

   // Step

   labels.push_back("Cooling Rate");

   buffer.str("");
   buffer << cooling_rate;

   values.push_back(buffer.str());

   // Trials number

   labels.push_back("Trials number");

   buffer.str("");
   buffer << trials_number;

   values.push_back(buffer.str());

   // Tolerance

   labels.push_back("Tolerance");

   buffer.str("");
   buffer << tolerance;

   values.push_back(buffer.str());

   // selection error goal

   labels.push_back("Selection loss goal");

   buffer.str("");
   buffer << selection_error_goal;

   values.push_back(buffer.str());

   // Minimum temperature

   labels.push_back("Minimum temperature");

   buffer.str("");
   buffer << minimum_temperature;

   values.push_back(buffer.str());

   // Maximum iterations number

   labels.push_back("Maximum iterations number");

   buffer.str("");
   buffer << maximum_iterations_number;

   values.push_back(buffer.str());

   // Maximum time

   labels.push_back("Maximum time");

   buffer.str("");
   buffer << maximum_time;

   values.push_back(buffer.str());

   // Plot training loss history

   labels.push_back("Plot training error history");

   buffer.str("");

   if(reserve_error_data)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   // Plot selection error history

   labels.push_back("Plot selection error history");

   buffer.str("");

   if(reserve_selection_error_data)
   {
       buffer << "true";
   }
   else
   {
       buffer << "false";
   }

   values.push_back(buffer.str());

   const size_t rows_number = labels.size();
   const size_t columns_number = 2;

   Matrix<string> string_matrix(rows_number, columns_number);

   string_matrix.set_column(0, labels, "name");
   string_matrix.set_column(1, values, "value");

    return(string_matrix);
}

// tinyxml2::XMLDocument* to_XML() const method

/// Prints to the screen the simulated annealing order parameters, the stopping criteria
/// and other user stuff concerning the simulated annealing order object.

tinyxml2::XMLDocument* SimulatedAnnealingOrder::to_XML() const
{
   ostringstream buffer;

   tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

   // Order Selection algorithm

   tinyxml2::XMLElement* root_element = document->NewElement("SimulatedAnnealingOrder");

   document->InsertFirstChild(root_element);

   tinyxml2::XMLElement* element = nullptr;
   tinyxml2::XMLText* text = nullptr;

   // Minimum order
   {
   element = document->NewElement("MinimumOrder");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_order;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum order
   {
   element = document->NewElement("MaximumOrder");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_order;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Cooling rate
   {
   element = document->NewElement("CoolingRate");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << cooling_rate;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Trials number
   {
   element = document->NewElement("TrialsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << trials_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Tolerance
   {
   element = document->NewElement("Tolerance");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << tolerance;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // selection error goal
   {
   element = document->NewElement("SelectionErrorGoal");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << selection_error_goal;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Minimum temperature
   {
   element = document->NewElement("MinimumTemperature");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << minimum_temperature;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum iterations
   {
   element = document->NewElement("MaximumIterationsNumber");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_iterations_number;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Maximum time
   {
   element = document->NewElement("MaximumTime");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << maximum_time;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve error data
   {
   element = document->NewElement("ReserveErrorHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_error_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Reserve selection error data
   {
   element = document->NewElement("ReserveSelectionErrorHistory");
   root_element->LinkEndChild(element);

   buffer.str("");
   buffer << reserve_selection_error_data;

   text = document->NewText(buffer.str().c_str());
   element->LinkEndChild(text);
   }

   // Performance calculation method
//   {
//   element = document->NewElement("LossCalculationMethod");
//   root_element->LinkEndChild(element);

//   text = document->NewText(write_loss_calculation_method().c_str());
//   element->LinkEndChild(text);
//   }

   // Reserve parameters data
//   {
//   element = document->NewElement("ReserveParametersData");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_parameters_data;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Reserve minimal parameters
//   {
//   element = document->NewElement("ReserveMinimalParameters");
//   root_element->LinkEndChild(element);

//   buffer.str("");
//   buffer << reserve_minimal_parameters;

//   text = document->NewText(buffer.str().c_str());
//   element->LinkEndChild(text);
//   }

   // Display
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


// void write_XML(tinyxml2::XMLPrinter&) const method

/// Serializes the simulated annealing order object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void SimulatedAnnealingOrder::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    ostringstream buffer;

    //file_stream.OpenElement("SimulatedAnnealingOrder");

    // Minimum order

    file_stream.OpenElement("MinimumOrder");

    buffer.str("");
    buffer << minimum_order;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum order

    file_stream.OpenElement("MaximumOrder");

    buffer.str("");
    buffer << maximum_order;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Cooling rate

    file_stream.OpenElement("CoolingRate");

    buffer.str("");
    buffer << cooling_rate;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Trials number

    file_stream.OpenElement("TrialsNumber");

    buffer.str("");
    buffer << trials_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Tolerance

    file_stream.OpenElement("Tolerance");

    buffer.str("");
    buffer << tolerance;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // selection error goal

    file_stream.OpenElement("SelectionErrorGoal");

    buffer.str("");
    buffer << selection_error_goal;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Minimum temperature

    file_stream.OpenElement("MinimumTemperature");

    buffer.str("");
    buffer << minimum_temperature;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum iterations

    file_stream.OpenElement("MaximumIterationsNumber");

    buffer.str("");
    buffer << maximum_iterations_number;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Maximum time

    file_stream.OpenElement("MaximumTime");

    buffer.str("");
    buffer << maximum_time;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve error data

    file_stream.OpenElement("ReserveErrorHistory");

    buffer.str("");
    buffer << reserve_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Reserve selection error data

    file_stream.OpenElement("ReserveSelectionErrorHistory");

    buffer.str("");
    buffer << reserve_selection_error_data;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Deserializes a TinyXML document into this simulated annealing order object.
/// @param document TinyXML document containing the member data.

void SimulatedAnnealingOrder::from_XML(const tinyxml2::XMLDocument& document)
{
    const tinyxml2::XMLElement* root_element = document.FirstChildElement("SimulatedAnnealingOrder");

    if(!root_element)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IncrementalOrder class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "SimulatedAnnealingOrder element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Minimum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumOrder");

        if(element)
        {
           const size_t new_minimum_order = atoi(element->GetText());

           try
           {
              minimum_order = new_minimum_order;
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Maximum order
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumOrder");

        if(element)
        {
           const size_t new_maximum_order = atoi(element->GetText());

           try
           {
              maximum_order = new_maximum_order;
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Parameters assays number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("TrialsNumber");

        if(element)
        {
           const size_t new_trials_number = atoi(element->GetText());

           try
           {
              set_trials_number(new_trials_number);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Performance calculation method
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("LossCalculationMethod");

        if(element)
        {
           const string new_loss_calculation_method = element->GetText();

           try
           {
              set_loss_calculation_method(new_loss_calculation_method);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Cooling rate
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("CoolingRate");

        if(element)
        {
           const double new_cooling_rate = atof(element->GetText());

           try
           {
              set_cooling_rate(new_cooling_rate);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve parameters data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveParametersData");

        if(element)
        {
           const string new_reserve_parameters_data = element->GetText();

           try
           {
              set_reserve_parameters_data(new_reserve_parameters_data != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve loss data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveErrorHistory");

        if(element)
        {
           const string new_reserve_error_data = element->GetText();

           try
           {
              set_reserve_error_data(new_reserve_error_data != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve selection error data
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveSelectionErrorHistory");

        if(element)
        {
           const string new_reserve_selection_error_data = element->GetText();

           try
           {
              set_reserve_selection_error_data(new_reserve_selection_error_data != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Reserve minimal parameters
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ReserveMinimalParameters");

        if(element)
        {
           const string new_reserve_minimal_parameters = element->GetText();

           try
           {
              set_reserve_minimal_parameters(new_reserve_minimal_parameters != "0");
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Display
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

    // selection error goal
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("SelectionErrorGoal");

        if(element)
        {
           const double new_selection_error_goal = atof(element->GetText());

           try
           {
              set_selection_error_goal(new_selection_error_goal);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Maximum iterations number
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumIterationsNumber");

        if(element)
        {
           const size_t new_maximum_iterations_number = atoi(element->GetText());

           try
           {
              set_maximum_iterations_number(new_maximum_iterations_number);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Maximum time
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MaximumTime");

        if(element)
        {
           const double new_maximum_time = atoi(element->GetText());

           try
           {
              set_maximum_time(new_maximum_time);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Tolerance
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Tolerance");

        if(element)
        {
           const double new_tolerance = atof(element->GetText());

           try
           {
              set_tolerance(new_tolerance);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }

    // Minimum temperature
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MinimumTemperature");

        if(element)
        {
           const double new_minimum_temperature = atof(element->GetText());

           try
           {
              set_minimum_temperature(new_minimum_temperature);
           }
           catch(const logic_error& e)
           {
              cerr << e.what() << endl;
           }
        }
    }
}

// void save(const string&) const method

/// Saves to a XML-type file the members of the simulated annealing order object.
/// @param file_name Name of simulated annealing order XML-type file.

void SimulatedAnnealingOrder::save(const string& file_name) const
{
   tinyxml2::XMLDocument* document = to_XML();

   document->SaveFile(file_name.c_str());

   delete document;
}


// void load(const string&) method

/// Loads a simulated annealing order object from a XML-type file.
/// @param file_name Name of simulated annealing order XML-type file.

void SimulatedAnnealingOrder::load(const string& file_name)
{
   set_default();

   tinyxml2::XMLDocument document;

   if(document.LoadFile(file_name.c_str()))
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: SimulatedAnnealingOrder class.\n"
             << "void load(const string&) method.\n"
             << "Cannot load XML file " << file_name << ".\n";

      throw logic_error(buffer.str());
   }

   from_XML(document);
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
