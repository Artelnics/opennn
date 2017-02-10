/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   W E I G T H E D   S Q U A R E D   E R R O R   C L A S S                                                    */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "weighted_squared_error.h"

namespace OpenNN
{
// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates a weighted squared error term not associated to any
/// neural network and not measured on any data set.
/// It also initializes all the rest of class members to their default values.

WeightedSquaredError::WeightedSquaredError(void) : ErrorTerm()
{
    set_default();
}


// NEURAL NETWORK CONSTRUCTOR

/// Neural network constructor. 
/// It creates a weighted squared error term object associated to a
/// neural network object but not measured on any data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.

WeightedSquaredError::WeightedSquaredError(NeuralNetwork* new_neural_network_pointer)
    : ErrorTerm(new_neural_network_pointer)
{
    set_default();
}


// DATA SET CONSTRUCTOR

/// Data set constructor. 
/// It creates a weighted squared error term not associated to any
/// neural network but to be measured on a given data set object.
/// It also initializes all the rest of class members to their default values.
/// @param new_data_set_pointer Pointer to a data set object.

WeightedSquaredError::WeightedSquaredError(DataSet* new_data_set_pointer)
    : ErrorTerm(new_data_set_pointer)
{
    set_default();
}


// NEURAL NETWORK AND DATA SET CONSTRUCTOR

/// Neural network and data set constructor. 
/// It creates a weighted squared error term object associated to a
/// neural network and measured on a data set.
/// It also initializes all the rest of class members to their default values.
/// @param new_neural_network_pointer Pointer to a neural network object.
/// @param new_data_set_pointer Pointer to a data set object.

WeightedSquaredError::WeightedSquaredError(NeuralNetwork* new_neural_network_pointer, DataSet* new_data_set_pointer)
    : ErrorTerm(new_neural_network_pointer, new_data_set_pointer)
{
    set_default();
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a weighted squared error object with all pointers set to NULL.
/// The object members are loaded by means of a XML document.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param weighted_squared_error_document TinyXML document with the weighted squared error elements.

WeightedSquaredError::WeightedSquaredError(const tinyxml2::XMLDocument& weighted_squared_error_document)
    : ErrorTerm(weighted_squared_error_document)
{
    set_default();
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing weighted squared error object.
/// @param other_weighted_squared_error Weighted squared error object to be copied.

WeightedSquaredError::WeightedSquaredError(const WeightedSquaredError& other_weighted_squared_error)
    : ErrorTerm(other_weighted_squared_error)
{
    negatives_weight = other_weighted_squared_error.negatives_weight;
    positives_weight = other_weighted_squared_error.positives_weight;
}


// DESTRUCTOR

/// Destructor.

WeightedSquaredError::~WeightedSquaredError(void)
{
}


// METHODS

// double get_positives_weight(void) const method

/// Returns the weight of the positives.

double WeightedSquaredError::get_positives_weight(void) const
{
    return(positives_weight);
}


// double get_negatives_weight(void) const method

/// Returns the weight of the negatives.

double WeightedSquaredError::get_negatives_weight(void) const
{
    return(negatives_weight);
}


// void check(void) const method

/// Checks that there are a neural network and a data set associated to the weighted squared error,
/// and that the numbers of inputs and outputs in the neural network are equal to the numbers of inputs and targets in the data set. 
/// If some of the above conditions is not hold, the method throws an exception. 

void WeightedSquaredError::check(void) const
{
    std::ostringstream buffer;

    // Neural network stuff

    if(!neural_network_pointer)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void check(void) const method.\n"
               << "Pointer to neural network is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    if(!multilayer_perceptron_pointer)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void check(void) const method.\n"
               << "Pointer to multilayer perceptron is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    if(inputs_number == 0)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void check(void) const method.\n"
               << "Number of inputs in multilayer perceptron object is zero.\n";

        throw std::logic_error(buffer.str());
    }

    if(outputs_number == 0)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void check(void) const method.\n"
               << "Number of outputs in multilayer perceptron object is zero.\n";

        throw std::logic_error(buffer.str());
    }

    // Data set stuff

    if(!data_set_pointer)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void check(void) const method.\n"
               << "Pointer to data set is NULL.\n";

        throw std::logic_error(buffer.str());
    }

    // Sum squared error stuff

    const Variables& variables = data_set_pointer->get_variables();

    const size_t data_set_inputs_number = variables.count_inputs_number();
    const size_t data_set_targets_number = variables.count_targets_number();

    if(inputs_number != data_set_inputs_number)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void check(void) const method.\n"
               << "Number of inputs in multilayer perceptron must be equal to number of inputs in data set.\n";

        throw std::logic_error(buffer.str());
    }

    if(outputs_number != data_set_targets_number)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "void check(void) const method.\n"
               << "Number of outputs in multilayer perceptron must be equal to number of targets in data set.\n";

        throw std::logic_error(buffer.str());
    }
}


// void set_default(void) method

/// Set the default values for the object.

void WeightedSquaredError::set_default(void)
{
    negatives_weight = 1.0;
    positives_weight = 1.0;
}


// void set_positives_weight(const double&)

/// Set a new weight for the positives values.
/// @param new_positives_weight New weight for the positives.

void WeightedSquaredError::set_positives_weight(const double& new_positives_weight)
{
    positives_weight = new_positives_weight;
}


// void set_negatives_weight(const double&)

/// Set a new weight for the negatives values.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_negatives_weight(const double& new_negatives_weight)
{
    negatives_weight = new_negatives_weight;
}


// void set_weights(const double&, const double&) method

/// Set new weights for the positives and negatives values.
/// @param new_positives_weight New weight for the positives.
/// @param new_negatives_weight New weight for the negatives.

void WeightedSquaredError::set_weights(const double& new_positives_weight, const double& new_negatives_weight)
{
    positives_weight = new_positives_weight;
    negatives_weight = new_negatives_weight;
}


// void set_weights(void) method

/// Calculates of the weights for the positives and negatives values with the data of the data set.

void WeightedSquaredError::set_weights(void)
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    const Vector<size_t> target_distribution = data_set_pointer->calculate_target_distribution();

    const size_t negatives = target_distribution[0];
    const size_t positives = target_distribution[1];

    if(positives == 0 || negatives == 0)
    {
        positives_weight = 1.0;
        negatives_weight = 1.0;

        return;
    }

    negatives_weight = 1.0;
    positives_weight = (double)negatives/(double)positives;
}


// double calculate_positives_loss(void) const method

/// Returns the weighted squared error for the positive instances.

double WeightedSquaredError::calculate_positives_loss(void) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    // Weighted squared error stuff

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    int i = 0;

    double sum_squared_error = 0.0;

    const double positives = get_positives_weight();

#pragma omp parallel for private(i, training_index, inputs, outputs, targets) firstprivate(positives) reduction(+:sum_squared_error)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        outputs[0] = 0.5;

        if(targets[0] == 1.0)
        {
            sum_squared_error += positives*outputs.calculate_sum_squared_error(targets);
        }
    }

    return(sum_squared_error);
}


// double calculate_negatives_loss(void) const method

/// Returns the weighted squared error for the negative instances.

double WeightedSquaredError::calculate_negatives_loss(void) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    // Weighted squared error stuff

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    int i = 0;

    double sum_squared_error = 0.0;

    double negatives = get_negatives_weight();

#pragma omp parallel for private(i, training_index, inputs, outputs, targets) firstprivate(negatives) reduction(+:sum_squared_error)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        outputs[0] = 0.5;

        if(targets[0] == 0.0)
        {
            sum_squared_error += negatives*outputs.calculate_sum_squared_error(targets);
        }
    }

    return(sum_squared_error);
}


// double calculate_error(void) const method

/// Returns the weighted squared error of a neural network on a data set.

double WeightedSquaredError::calculate_error(void) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    // Weighted squared error stuff

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    int i = 0;

    double error = 0;

    double sum_squared_error = 0.0;    

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

#pragma omp parallel for private(i, training_index, inputs, outputs, targets, error) reduction(+:sum_squared_error)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        if(targets[0] == 1.0)
        {
            error = positives_w*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            error = negatives_w*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(void) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }

        sum_squared_error += error;
    }

    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(sum_squared_error/normalization_coefficient);
}


// double calculate_error(const Vector<double>&) const method

/// Returns which would be the error term of a neural network for an hypothetical
/// vector of parameters. It does not set that vector of parameters to the neural network. 
/// @param parameters Vector of potential parameters for the neural network associated to the error term.

double WeightedSquaredError::calculate_error(const Vector<double>& parameters) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

#ifdef __OPENNN_DEBUG__

    const size_t size = parameters.size();

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    if(size != parameters_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "double calculate_error(const Vector<double>&) const method.\n"
               << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    // Weighted squared error stuff

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    double sum_squared_error = 0.0;

    int i = 0;

    double error = 0;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

#pragma omp parallel for private(i, training_index, inputs, outputs, targets, error) reduction(+:sum_squared_error)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

        // Target vector

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        if(targets[0] == 1.0)
        {
            error = positives_w*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            error = negatives_w*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(const Vector<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }

        sum_squared_error += error;

    }

    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(sum_squared_error/normalization_coefficient);
}


// double calculate_selection_loss(void) const method

/// Returns the weighted squared error of the neural network measured on the selection instances of the
/// data set.

double WeightedSquaredError::calculate_selection_error(void) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.count_selection_instances_number();

    if(selection_instances_number == 0)
    {
        return(0.0);
    }

    const Vector<size_t> selection_indices = instances.arrange_selection_indices();

    size_t selection_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    double selection_loss = 0.0;

    double loss = 0.0;

    int i = 0;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

#pragma omp parallel for private(i, selection_index, inputs, outputs, targets, loss) reduction(+:selection_loss)

    for(i = 0; i < (int)selection_instances_number; i++)
    {
        selection_index = selection_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        targets = data_set_pointer->get_instance(selection_index, targets_indices);

        // Sum squared error

        if(targets[0] == 1.0)
        {
            loss = positives_w*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            loss = negatives_w*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(const Vector<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }

        selection_loss += loss;
    }

    const size_t negatives = data_set_pointer->calculate_selection_negatives(targets_indices[0]);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(selection_loss/normalization_coefficient);
}

// double calculate_error(const double&) const method

/// Returns the weighted squared error of a neural network on a data set.
/// @param normalization_coefficient Normalization coefficient to be used.

double WeightedSquaredError::calculate_error(const double& normalization_coefficient) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    // Weighted squared error stuff

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    int i = 0;

    double error = 0;

    double sum_squared_error = 0.0;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

#pragma omp parallel for private(i, training_index, inputs, outputs, targets, error) reduction(+:sum_squared_error)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        if(targets[0] == 1.0)
        {
            error = positives_w*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            error = negatives_w*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(void) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }

        sum_squared_error += error;
    }

    return(sum_squared_error/normalization_coefficient);
}

// double calculate_error(const Vector<double>&, const double&) const method

/// Returns which would be the error term of a neural network for an hypothetical
/// vector of parameters. It does not set that vector of parameters to the neural network.
/// @param parameters Vector of potential parameters for the neural network associated to the error term.
/// @param normalization_coefficient Normalization coefficient to be used.

double WeightedSquaredError::calculate_error(const Vector<double>& parameters, const double& normalization_coefficient) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

#ifdef __OPENNN_DEBUG__

    const size_t size = parameters.size();

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    if(size != parameters_number)
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "double calculate_error(const Vector<double>&) const method.\n"
               << "Size (" << size << ") must be equal to number of parameters (" << parameters_number << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    // Weighted squared error stuff

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    double sum_squared_error = 0.0;

    int i = 0;

    double error = 0;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

#pragma omp parallel for private(i, training_index, inputs, outputs, targets, error) reduction(+:sum_squared_error)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

        // Target vector

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Sum squared error

        if(targets[0] == 1.0)
        {
            error = positives_w*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            error = negatives_w*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(const Vector<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }

        sum_squared_error += error;

    }

    return(sum_squared_error/normalization_coefficient);
}


// double calculate_selection_loss(const double&) const method

/// Returns the weighted squared error of the neural network measured on the selection instances of the
/// data set.
/// @param normalization_coefficient Normalization coefficient to be used.

double WeightedSquaredError::calculate_selection_error(const double& normalization_coefficient) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const Instances& instances = data_set_pointer->get_instances();

    const size_t selection_instances_number = instances.count_selection_instances_number();

    if(selection_instances_number == 0)
    {
        return(0.0);
    }

    const Vector<size_t> selection_indices = instances.arrange_selection_indices();

    size_t selection_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    double selection_loss = 0.0;

    double loss = 0.0;

    int i = 0;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

#pragma omp parallel for private(i, selection_index, inputs, outputs, targets, loss) reduction(+:selection_loss)

    for(i = 0; i < (int)selection_instances_number; i++)
    {
        selection_index = selection_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(selection_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        targets = data_set_pointer->get_instance(selection_index, targets_indices);

        // Sum squared error

        if(targets[0] == 1.0)
        {
            loss = positives_w*outputs.calculate_sum_squared_error(targets);
        }
        else if(targets[0] == 0.0)
        {
            loss = negatives_w*outputs.calculate_sum_squared_error(targets);
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "double calculate_error(const Vector<double>&) const method.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }

        selection_loss += loss;
    }

    return(selection_loss/normalization_coefficient);
}

// Vector<double> calculate_output_gradient(void) const method

/// Calculates the loss output gradient by means of the back-propagation algorithm,
/// and returns it in a single vector of size the number of multilayer perceptron parameters.
/// @param output Vector of the outputs of the model.
/// @param target Vector of targets of the data set.

Vector<double> WeightedSquaredError::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target) const
{
    Vector<double> output_gradient;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

//    const size_t training_instances_number = target.size();

//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

//    Vector<double> targets(outputs_number);

    //size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

//    for(size_t i = 0; i < training_instances_number; i++)
//    {
        //targets = data_set_pointer->get_instance(training_index, targets_indices);

    if(target[0] == 1.0)
    {
        output_gradient = (output-target)*positives_w*2.0;
    }
    else if(target[0] == 0.0)
    {
        output_gradient = (output-target)*negatives_w*2.0;
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "Vector<double> calculate_output_gradient(void) const method.\n"
               << "Target is neither a positive nor a negative.\n";

        throw std::logic_error(buffer.str());
    }
//    }

    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return output_gradient/normalization_coefficient;
}


// Matrix<double> calculate_output_Hessian(void) const method

/// @todo

Matrix<double> WeightedSquaredError::calculate_output_Hessian(const Vector<double>& , const Vector<double>& target) const
{
    Matrix<double> output_Hessian;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

//    const size_t training_instances_number = target.size();

//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

//    Vector<double> targets(outputs_number);

    //size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

//    for(size_t i = 0; i < training_instances_number; i++)
//    {
        //targets = data_set_pointer->get_instance(training_index, targets_indices);

    if(target[0] == 1.0)
    {
        output_Hessian.initialize_diagonal(1, positives_w*2.0/normalization_coefficient);
    }
    else if(target[0] == 0.0)
    {
        output_Hessian.initialize_diagonal(1, negatives_w*2.0/normalization_coefficient);
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "Vector<double> calculate_output_Hessian(const Vector<double>&, const Vector<double>&) const method.\n"
               << "Target is neither a positive nor a negative.\n";

        throw std::logic_error(buffer.str());
    }
//    }

    return output_Hessian;
}

// Vector<double> calculate_output_gradient(const Vector<double>&, const Vector<double>&, const double&) const method

/// Calculates the loss output gradient by means of the back-propagation algorithm,
/// and returns it in a single vector of size the number of multilayer perceptron parameters.
/// @param output Vector of the outputs of the model.
/// @param target Vector of targets of the data set.
/// @param normalization_coefficient Coefficient of the normalization for the gradient.

Vector<double> WeightedSquaredError::calculate_output_gradient(const Vector<double>& output, const Vector<double>& target, const double& normalization_coefficient) const
{
    Vector<double> output_gradient;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

//    const size_t training_instances_number = target.size();

//    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

//    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

//    Vector<double> targets(outputs_number);

    //size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

//    for(size_t i = 0; i < training_instances_number; i++)
//    {
        //targets = data_set_pointer->get_instance(training_index, targets_indices);

    if(target[0] == 1.0)
    {
        output_gradient = (output-target)*positives_w*2.0;
    }
    else if(target[0] == 0.0)
    {
        output_gradient = (output-target)*negatives_w*2.0;
    }
    else
    {
        std::ostringstream buffer;

        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "Vector<double> calculate_output_gradient(void) const method.\n"
               << "Target is neither a positive nor a negative.\n";

        throw std::logic_error(buffer.str());
    }
//    }

    return output_gradient/normalization_coefficient;
}

// Vector<double> calculate_gradient_with_normalization(const double&) const

/// Calculates the loss output gradient by means of the back-propagation algorithm,
/// and returns it in a single vector of size the number of multilayer perceptron parameters.
/// @param normalization_coefficient Coefficient of the normalization for the gradient.

Vector<double> WeightedSquaredError::calculate_gradient_with_normalization(const double& normalization_coefficient) const
{
#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    // Neural network stuff

    const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

    const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

    Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

    Vector<double> particular_solution;
    Vector<double> homogeneous_solution;

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> targets(outputs_number);

    // Sum squared error stuff

    Vector<double> output_gradient(outputs_number);

    Vector< Matrix<double> > layers_combination_parameters_Jacobian;

    Vector< Vector<double> > layers_inputs(layers_number);
    Vector< Vector<double> > layers_delta;

    Vector<double> point_gradient(neural_parameters_number, 0.0);

    Vector<double> gradient(neural_parameters_number, 0.0);

    int i;

#pragma omp parallel for private(i, training_index, inputs, targets, first_order_forward_propagation, layers_inputs, layers_combination_parameters_Jacobian,\
    output_gradient, layers_delta, particular_solution, homogeneous_solution, point_gradient)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
        const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

        layers_inputs = multilayer_perceptron_pointer->arrange_layers_input(inputs, layers_activation);

        layers_combination_parameters_Jacobian = multilayer_perceptron_pointer->calculate_layers_combination_parameters_Jacobian(layers_inputs);

        if(!has_conditions_layer)
        {
            output_gradient = calculate_output_gradient(layers_activation[layers_number-1], targets, normalization_coefficient);

            layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);
        }
        else
        {
            particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
            homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

            output_gradient = (particular_solution+homogeneous_solution*layers_activation[layers_number-1] - targets)*2.0;

            layers_delta = calculate_layers_delta(layers_activation_derivative, homogeneous_solution, output_gradient);
        }

        point_gradient = calculate_point_gradient(layers_combination_parameters_Jacobian, layers_delta);

#pragma omp critical
        gradient += point_gradient;
    }

    return(gradient);
}

// FirstOrderPerformance calculate_first_order_loss(void) const method

/// @todo

ErrorTerm::FirstOrderPerformance WeightedSquaredError::calculate_first_order_loss(void) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    FirstOrderPerformance first_order_loss;

    first_order_loss.loss = calculate_error();
    first_order_loss.gradient = calculate_gradient();

    return(first_order_loss);
}


// SecondOrderloss calculate_second_order_loss(void) const method

/// @todo

ErrorTerm::SecondOrderPerformance WeightedSquaredError::calculate_second_order_loss(void) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    SecondOrderPerformance second_order_loss;

    second_order_loss.loss = calculate_error();
    second_order_loss.gradient = calculate_gradient();
    second_order_loss.Hessian = calculate_Hessian();

    return(second_order_loss);
}


// Vector<double> calculate_terms(void) const method

/// Returns loss vector of the error terms function for the weighted squared error.
/// It uses the error back-propagation method.

Vector<double> WeightedSquaredError::calculate_terms(void) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    // Weighted squared error stuff

    Vector<double> error_terms(training_instances_number);

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> targets(outputs_number);

    int i = 0;

    const double positives_w = positives_weight;
    const double negatives_w = negatives_weight;

#pragma omp parallel for private(i, training_index, inputs, outputs, targets)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        // Input vector

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        // Output vector

        outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

        // Target vector

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        // Error

        if(targets[0] == 1.0)
        {
            error_terms[i] = positives_w*outputs.calculate_distance(targets);
        }
        else if(targets[0] == 0.0)
        {
            error_terms[i] = negatives_w*outputs.calculate_distance(targets);
        }
        else
        {
            std::ostringstream buffer;

            buffer << "OpenNN Exception: WeightedSquaredError class.\n"
                   << "Vector<double> WeightedSquaredError::calculate_terms(void) const.\n"
                   << "Target is neither a positive nor a negative.\n";

            throw std::logic_error(buffer.str());
        }
    }

    const size_t negatives = data_set_pointer->calculate_training_negatives(targets_indices[0]);

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(error_terms/sqrt(normalization_coefficient));
}


// Vector<double> calculate_terms(const Vector<double>&) const method

/// Returns which would be the error terms loss vector of a multilayer perceptron for an hypothetical vector of multilayer perceptron parameters.
/// It does not set that vector of parameters to the multilayer perceptron. 
/// @param network_parameters Vector of a potential multilayer_perceptron_pointer parameters for the multilayer perceptron associated to the loss functional.

Vector<double> WeightedSquaredError::calculate_terms(const Vector<double>& network_parameters) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

#ifdef __OPENNN_DEBUG__

    std::ostringstream buffer;

    const size_t size = network_parameters.size();

    const size_t parameters_number = neural_network_pointer->count_parameters_number();

    if(size != parameters_number)
    {
        buffer << "OpenNN Exception: WeightedSquaredError class.\n"
               << "double calculate_terms(const Vector<double>&) const method.\n"
               << "Size (" << size << ") must be equal to number of multilayer perceptron parameters (" << parameters_number << ").\n";

        throw std::logic_error(buffer.str());
    }

#endif

    NeuralNetwork neural_network_copy(*neural_network_pointer);

    neural_network_copy.set_parameters(network_parameters);

    WeightedSquaredError weighted_squared_error_copy(*this);

    weighted_squared_error_copy.set_neural_network_pointer(&neural_network_copy);

    return(weighted_squared_error_copy.calculate_terms());
}


// Matrix<double> calculate_terms_Jacobian(void) const method

/// Returns the Jacobian matrix of the weighted squared error function, whose elements are given by the
/// derivatives of the squared errors data set with respect to the multilayer perceptron parameters.

Matrix<double> WeightedSquaredError::calculate_terms_Jacobian(void) const
{
    // Control sentence

#ifdef __OPENNN_DEBUG__

    check();

#endif

    // Neural network stuff

    const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    const size_t neural_parameters_number = multilayer_perceptron_pointer->count_parameters_number();

    Vector< Vector< Vector<double> > > first_order_forward_propagation(2);

    Vector<double> particular_solution;
    Vector<double> homogeneous_solution;

    const bool has_conditions_layer = neural_network_pointer->has_conditions_layer();

    const ConditionsLayer* conditions_layer_pointer = has_conditions_layer ? neural_network_pointer->get_conditions_layer_pointer() : NULL;

    // Data set stuff

    const Instances& instances = data_set_pointer->get_instances();

    const size_t training_instances_number = instances.count_training_instances_number();

    const Vector<size_t> training_indices = instances.arrange_training_indices();

    size_t training_index;

    const Variables& variables = data_set_pointer->get_variables();

    const Vector<size_t> inputs_indices = variables.arrange_inputs_indices();
    const Vector<size_t> targets_indices = variables.arrange_targets_indices();

    Vector<double> inputs(inputs_number);
    Vector<double> targets(outputs_number);

    // Loss index

    Vector<double> term(outputs_number);
    double term_norm;

    Vector<double> output_gradient(outputs_number);

    Vector< Vector<double> > layers_delta(layers_number);
    Vector<double> point_gradient(neural_parameters_number);

    Matrix<double> terms_Jacobian(training_instances_number, neural_parameters_number);

    // Main loop

    int i = 0;

#pragma omp parallel for private(i, training_index, inputs, targets, first_order_forward_propagation,  \
    term, term_norm, output_gradient, layers_delta, particular_solution, homogeneous_solution, point_gradient)

    for(i = 0; i < (int)training_instances_number; i++)
    {
        training_index = training_indices[i];

        inputs = data_set_pointer->get_instance(training_index, inputs_indices);

        targets = data_set_pointer->get_instance(training_index, targets_indices);

        first_order_forward_propagation = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

        const Vector< Vector<double> >& layers_activation = first_order_forward_propagation[0];
        const Vector< Vector<double> >& layers_activation_derivative = first_order_forward_propagation[1];

        if(!has_conditions_layer)
        {
            const Vector<double>& outputs = first_order_forward_propagation[0][layers_number-1];

            term = (outputs-targets);

            term_norm = term.calculate_norm();

            if(term_norm == 0.0)
            {
                output_gradient.set(outputs_number, 0.0);
            }
            else
            {
                output_gradient = term/term_norm;
            }

            layers_delta = calculate_layers_delta(layers_activation_derivative, output_gradient);
        }
        else
        {
            particular_solution = conditions_layer_pointer->calculate_particular_solution(inputs);
            homogeneous_solution = conditions_layer_pointer->calculate_homogeneous_solution(inputs);

            term = (particular_solution+homogeneous_solution*layers_activation[layers_number-1] - targets)/sqrt((double)training_instances_number);
            term_norm = term.calculate_norm();

            if(term_norm == 0.0)
            {
                output_gradient.set(outputs_number, 0.0);
            }
            else
            {
                output_gradient = term/term_norm;
            }

            layers_delta = calculate_layers_delta(layers_activation_derivative, homogeneous_solution, output_gradient);
        }

        point_gradient = calculate_point_gradient(inputs, layers_activation, layers_delta);

        terms_Jacobian.set_row(i, point_gradient);
    }

    const double negatives = training_instances_number
                           - data_set_pointer->arrange_training_target_data().arrange_column(0).calculate_sum();

    const double normalization_coefficient = negatives*negatives_weight*0.5;

    return(terms_Jacobian/sqrt(normalization_coefficient));
}


// FirstOrderTerms calculate_first_order_terms(void) const method

/// Returns a first order terms loss structure, which contains the values and the Jacobian of the error terms function.

/// @todo

WeightedSquaredError::FirstOrderTerms WeightedSquaredError::calculate_first_order_terms(void) const
{
    // Control sentence (if debug)

#ifdef __OPENNN_DEBUG__

    check();

#endif

    FirstOrderTerms first_order_terms;

    first_order_terms.terms = calculate_terms();

    first_order_terms.Jacobian = calculate_terms_Jacobian();

    return(first_order_terms);
}


// std::string write_error_term_type(void) const method

/// Returns a string with the name of the weighted squared error loss type, "WEIGHTED_SQUARED_ERROR".

std::string WeightedSquaredError::write_error_term_type(void) const
{
    return("WEIGHTED_SQUARED_ERROR");
}


// tinyxml2::XMLDocument* to_XML(void) const method 

/// Serializes the weighted squared error object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this document-> 

tinyxml2::XMLDocument* WeightedSquaredError::to_XML(void) const
{
    std::ostringstream buffer;

    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    // Weighted squared error

    tinyxml2::XMLElement* weighted_squared_error_element = document->NewElement("WeightedSquaredError");

    document->InsertFirstChild(weighted_squared_error_element);

    // Positives weight
    {
    tinyxml2::XMLElement* element = document->NewElement("PositivesWeight");
    weighted_squared_error_element->LinkEndChild(element);

    buffer.str("");
    buffer << positives_weight;

    tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);
    }

    // Negatives weight
    {
    tinyxml2::XMLElement* element = document->NewElement("NegativesWeight");
    weighted_squared_error_element->LinkEndChild(element);

    buffer.str("");
    buffer << negatives_weight;

    tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
    element->LinkEndChild(text);
    }

    // Display
    //   {
    //      tinyxml2::XMLElement* element = document->NewElement("Display");
    //      weighted_squared_error_element->LinkEndChild(element);

    //      buffer.str("");
    //      buffer << display;

    //      tinyxml2::XMLText* text = document->NewText(buffer.str().c_str());
    //      element->LinkEndChild(text);
    //   }

    return(document);
}


// void write_XML(tinyxml2::XMLPrinter&) const method

void WeightedSquaredError::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    std::ostringstream buffer;

    //file_stream.OpenElement("WeightedSquaredError");

    // Positives weight

    file_stream.OpenElement("PositivesWeight");

    buffer.str("");
    buffer << positives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    // Negatives weight

    file_stream.OpenElement("NegativesWeight");

    buffer.str("");
    buffer << negatives_weight;

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();


    //file_stream.CloseElement();
}


// void from_XML(const tinyxml2::XMLDocument&) method

/// Loads a weighted squared error object from a XML document.
/// @param document Pointer to a TinyXML document with the object data.

void WeightedSquaredError::from_XML(const tinyxml2::XMLDocument& document)
{
   const tinyxml2::XMLElement* weighted_element = document.FirstChildElement("WeightedSquaredError");

   if(!weighted_element)
   {
      return;
   }

   // Positives weight

   const tinyxml2::XMLElement* positives_weight_element = weighted_element->FirstChildElement("PositivesWeight");

   if(positives_weight_element)
   {
      const std::string string = positives_weight_element->GetText();

      try
      {
         set_positives_weight(atof(string.c_str()));
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;
      }
   }

   // Negatives weight

   const tinyxml2::XMLElement* negatives_weight_element = weighted_element->FirstChildElement("NegativesWeight");

   if(negatives_weight_element)
   {
      const std::string string = negatives_weight_element->GetText();

      try
      {
         set_negatives_weight(atof(string.c_str()));
      }
      catch(const std::logic_error& e)
      {
         std::cout << e.what() << std::endl;
      }
   }

   // Display

   const tinyxml2::XMLElement* display_element = weighted_element->FirstChildElement("Display");

   if(display_element)
   {
      const std::string new_display_string = display_element->GetText();

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


// std::string write_information(void) const method
/*
std::string WeightedSquaredError::write_information(void) const
{
    std::ostringstream buffer;

    buffer << "Weighted squared error: " << calculate_error() << "\n";

    return(buffer.str());

}*/


// std::string print_weights(void) const method

std::string WeightedSquaredError::to_string(void) const
{
    std::ostringstream buffer;

    buffer << "Weighted squared error.\n"
           << "Positives weight: " << positives_weight << "\n"
           << "Negatives weight: " << negatives_weight << std::endl;

    return(buffer.str());
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
