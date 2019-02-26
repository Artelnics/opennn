
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   N E U R A L   N E T W O R K   C L A S S                                                                    */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "neural_network.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 
/// It creates an empty neural network object.
/// All pointers in the object are initialized to nullptr. 
/// The rest of members are initialized to their default values.

NeuralNetwork::NeuralNetwork()
{
    set_default();
}


// MULTILAYER PERCEPTRON CONSTRUCTOR

/// Multilayer Perceptron constructor. 
/// It creates a neural network object from a given multilayer perceptron. 
/// The rest of pointers are initialized to nullptr. 
/// This constructor also initializes the rest of class members to their default values.

NeuralNetwork::NeuralNetwork(const MultilayerPerceptron& new_multilayer_perceptron)
{
    multilayer_perceptron_pointer = new MultilayerPerceptron(new_multilayer_perceptron);

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    inputs_pointer = new Inputs(inputs_number);
    outputs_pointer = new Outputs(outputs_number);

    set_default();
}


// MULTILAYER PERCEPTRON ARCHITECTURE CONSTRUCTOR

/// Multilayer perceptron architecture constructor. 
/// It creates a neural network object with a multilayer perceptron given by its architecture.
/// This constructor allows an arbitrary deep learning architecture.
/// The rest of pointers are initialized to nullptr.  
/// This constructor also initializes the rest of class members to their default values.
/// @param new_multilayer_perceptron_architecture Vector with the number of inputs and the numbers of perceptrons in each layer. 
/// The size of this vector must be equal to one plus the number of layers.

NeuralNetwork::NeuralNetwork(const Vector<size_t>& new_multilayer_perceptron_architecture)
{
    multilayer_perceptron_pointer = new MultilayerPerceptron(new_multilayer_perceptron_architecture);

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    inputs_pointer = new Inputs(inputs_number);
    outputs_pointer = new Outputs(outputs_number);

    set(new_multilayer_perceptron_architecture);
}


NeuralNetwork::NeuralNetwork(const vector<size_t>& new_multilayer_perceptron_architecture)
{
    multilayer_perceptron_pointer = new MultilayerPerceptron(Vector<size_t>(new_multilayer_perceptron_architecture));

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    inputs_pointer = new Inputs(inputs_number);
    outputs_pointer = new Outputs(outputs_number);

    set(new_multilayer_perceptron_architecture);
}


// ONE LAYER CONSTRUCTOR

/// One layer constructor. 
/// It creates a one-layer perceptron object. 
/// The number of independent parameters is set to zero.  
/// The multilayer perceptron parameters are initialized at random. 
/// @param new_inputs_number Number of inputs in the layer.
/// @param new_perceptrons_number Number of perceptrons in the layer.

NeuralNetwork::NeuralNetwork(const size_t& new_inputs_number, const size_t& new_perceptrons_number)
{
    multilayer_perceptron_pointer = new MultilayerPerceptron(new_inputs_number, new_perceptrons_number);

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    inputs_pointer = new Inputs(inputs_number);
    outputs_pointer = new Outputs(outputs_number);

    set_default();
}


// TWO LAYERS CONSTRUCTOR

/// Two layers constructor. 
/// It creates a neural network object with a two layers perceptron. 
/// The rest of pointers of this object are initialized to nullptr. 
/// The other members are initialized to their default values. 
/// @param new_inputs_number Number of inputs in the multilayer perceptron.
/// @param new_hidden_perceptrons_number Number of neurons in the hidden layer of the multilayer perceptron.
/// @param new_output_perceptrons_number Number of outputs neurons.

NeuralNetwork::NeuralNetwork(const size_t& new_inputs_number, const size_t& new_hidden_perceptrons_number, const size_t& new_output_perceptrons_number)
{
    multilayer_perceptron_pointer = new MultilayerPerceptron(new_inputs_number, new_hidden_perceptrons_number, new_output_perceptrons_number);

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    inputs_pointer = new Inputs(inputs_number);
    outputs_pointer = new Outputs(outputs_number);

    set_default();
}


// FILE CONSTRUCTOR

/// File constructor. 
/// It creates a neural network object by loading its members from an XML-type file.
/// Please be careful with the format of that file, which is specified in the OpenNN manual.
/// @param file_name Name of neural network file.

NeuralNetwork::NeuralNetwork(const string& file_name)
{
    load(file_name);
}


// XML CONSTRUCTOR

/// XML constructor. 
/// It creates a neural network object by loading its members from an XML document.
/// @param document TinyXML document containing the neural network data.

NeuralNetwork::NeuralNetwork(const tinyxml2::XMLDocument& document)
{
    from_XML(document);
}


// COPY CONSTRUCTOR

/// Copy constructor. 
/// It creates a copy of an existing neural network object. 
/// @param other_neural_network Neural network object to be copied.

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other_neural_network)
{
    set(other_neural_network);
}


// DESTRUCTOR

/// Destructor.

NeuralNetwork::~NeuralNetwork()
{
    delete multilayer_perceptron_pointer;
    delete inputs_trending_layer_pointer;
    delete scaling_layer_pointer;
    delete principal_components_layer_pointer;
    delete unscaling_layer_pointer;
    delete outputs_trending_layer_pointer;
    delete bounding_layer_pointer;
    delete probabilistic_layer_pointer;
    delete inputs_pointer;
    delete outputs_pointer;
}


// ASSIGNMENT OPERATOR

/// Assignment operator. 
/// It assigns to this object the members of an existing neural network object.
/// @param other_neural_network Neural network object to be assigned.

NeuralNetwork& NeuralNetwork::operator = (const NeuralNetwork& other_neural_network)
{
    set(other_neural_network);

    return(*this);
}


// EQUAL TO OPERATOR

/// Equal to operator. 
/// @param other_neural_network Neural network object to be compared with.

bool NeuralNetwork::operator == (const NeuralNetwork& other_neural_network) const
{
    if(*multilayer_perceptron_pointer == *other_neural_network.multilayer_perceptron_pointer
            && *inputs_trending_layer_pointer == *other_neural_network.inputs_trending_layer_pointer
            && *scaling_layer_pointer == *other_neural_network.scaling_layer_pointer
//            && *principal_components_layer_pointer == *other_neural_network.principal_components_layer_pointer
            && *unscaling_layer_pointer == *other_neural_network.unscaling_layer_pointer
            && *outputs_trending_layer_pointer == *other_neural_network.outputs_trending_layer_pointer
            && *bounding_layer_pointer == *other_neural_network.bounding_layer_pointer
            && *probabilistic_layer_pointer == *other_neural_network.probabilistic_layer_pointer
            && *inputs_pointer == *other_neural_network.inputs_pointer
            && *outputs_pointer == *other_neural_network.outputs_pointer
            &&  display == other_neural_network.display)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


// METHODS


/// Returns true if the neural network object has a multilayer perceptron object inside,
/// and false otherwise.

bool NeuralNetwork::has_multilayer_perceptron() const
{
    if(multilayer_perceptron_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has an inputs object inside,
/// and false otherwise.

bool NeuralNetwork::has_inputs() const
{
    if(inputs_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has an outputs object inside,
/// and false otherwise.

bool NeuralNetwork::has_outputs() const
{
    if(outputs_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has an inputs trending layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_inputs_trending_layer() const
{
    if(inputs_trending_layer_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has a scaling layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_scaling_layer() const
{
    if(scaling_layer_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has a principal components layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_principal_components_layer() const
{
    if(principal_components_layer_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has an unscaling layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_unscaling_layer() const
{
    if(unscaling_layer_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has an outputs trending layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_outputs_trending_layer() const
{
    if(outputs_trending_layer_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has a bounding layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_bounding_layer() const
{
    if(bounding_layer_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns true if the neural network object has a probabilistic layer object inside,
/// and false otherwise.

bool NeuralNetwork::has_probabilistic_layer() const
{
    if(probabilistic_layer_pointer)
    {
        return(true);
    }
    else
    {
        return(false);
    }
}


/// Returns a pointer to the multilayer perceptron composing this neural network.

MultilayerPerceptron* NeuralNetwork::get_multilayer_perceptron_pointer() const
{   
#ifdef __OPENNN_DEBUG__

    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "MultilayerPerceptron* get_multilayer_perceptron_pointer() const method.\n"
               << "Multilayer perceptron pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(multilayer_perceptron_pointer);
}


/// Returns a pointer to the inputs trending layer composing this neural network.

InputsTrendingLayer* NeuralNetwork::get_inputs_trending_layer_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!inputs_trending_layer_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "InputsTrendingLayer* get_inputs_trending_layer_pointer() const method.\n"
               << "Inputs trending layer pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(inputs_trending_layer_pointer);
}


/// Returns a pointer to the scaling layer composing this neural network.

ScalingLayer* NeuralNetwork::get_scaling_layer_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!scaling_layer_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "ScalingLayer* get_scaling_layer_pointer() const method.\n"
               << "Scaling layer pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(scaling_layer_pointer);
}


/// Returns a pointer to the principal components layer composing this neural network.

PrincipalComponentsLayer* NeuralNetwork::get_principal_components_layer_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!principal_components_layer_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "PrincipalComponentsLayer* get_principal_components_layer_pointer() const method.\n"
               << "Principal components layer pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(principal_components_layer_pointer);
}


/// Returns a pointer to the unscaling layer composing this neural network.

UnscalingLayer* NeuralNetwork::get_unscaling_layer_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!unscaling_layer_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "UnscalingLayer* get_unscaling_layer_pointer() const method.\n"
               << "Unscaling layer pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(unscaling_layer_pointer);
}


/// Returns a pointer to the outputs trending layer composing this neural network.

OutputsTrendingLayer* NeuralNetwork::get_outputs_trending_layer_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!outputs_trending_layer_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "OutputsTrendingLayer* get_outputs_trending_layer_pointer() const method.\n"
               << "Outputs trending layer pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(outputs_trending_layer_pointer);
}


/// Returns a pointer to the bounding layer composing this neural network.

BoundingLayer* NeuralNetwork::get_bounding_layer_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!bounding_layer_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "BoundingLayer* get_bounding_layer_pointer() const method.\n"
               << "Bounding layer pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(bounding_layer_pointer);
}


/// Returns a pointer to the probabilistic layer composing this neural network.

ProbabilisticLayer* NeuralNetwork::get_probabilistic_layer_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!probabilistic_layer_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "ProbabilisticLayer* get_probabilistic_layer_pointer() const method.\n"
               << "Probabilistic layer pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(probabilistic_layer_pointer);
}


/// Returns a pointer to the inputs object composing this neural network.

Inputs* NeuralNetwork::get_inputs_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!inputs_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Inputs* get_inputs_pointer() const method.\n"
               << "Inputs pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(inputs_pointer);
}


/// Returns a pointer to the outputs object composing this neural network.

Outputs* NeuralNetwork::get_outputs_pointer() const
{
#ifdef __OPENNN_DEBUG__

    if(!outputs_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Outputs* get_outputs_pointer() const method.\n"
               << "Outputs pointer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    return(outputs_pointer);
}


/// Returns true if messages from this class are to be displayed on the screen, or false if messages
/// from this class are not to be displayed on the screen.

const bool& NeuralNetwork::get_display() const
{
    return(display);
}


/// This method deletes all the pointers in the neural network.
/// It also sets the rest of members to their default values. 

void NeuralNetwork::set()
{
    delete_pointers();

    set_default();
}


/// This method deletes all the pointers in the neural network and then constructs a copy of an exisiting multilayer perceptron.
/// It also sets the rest of members to their default values. 
/// @param new_multilayer_perceptron Multilayer perceptron object to be copied. 

void NeuralNetwork::set(const MultilayerPerceptron& new_multilayer_perceptron)
{
    delete_pointers();

    multilayer_perceptron_pointer = new MultilayerPerceptron(new_multilayer_perceptron);

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    inputs_pointer = new Inputs(inputs_number);

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    outputs_pointer = new Outputs(outputs_number);

    set_default();
}


/// Sets a new neural network with a given multilayer perceptron architecture.
/// It also sets the rest of members to their default values. 
/// @param new_multilayer_perceptron_architecture Architecture of the multilayer perceptron. 

void NeuralNetwork::set(const Vector<size_t>& new_multilayer_perceptron_architecture)
{
    delete_pointers();

    multilayer_perceptron_pointer = new MultilayerPerceptron(new_multilayer_perceptron_architecture);

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    inputs_pointer = new Inputs(inputs_number);

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    outputs_pointer = new Outputs(outputs_number);

    set_default();
}


/// Sets a new multilayer_perceptron_pointer architecture with one layer and zero independent parameters.
/// It also sets the rest of members to their default values. 
/// @param new_inputs_number Number of inputs.
/// @param new_outputs_number Number of outputs.

void NeuralNetwork::set(const size_t& new_inputs_number, const size_t& new_outputs_number)
{
    delete_pointers();

    inputs_pointer = new Inputs(new_inputs_number);

    multilayer_perceptron_pointer = new MultilayerPerceptron(new_inputs_number, new_outputs_number);

    outputs_pointer = new Outputs(new_outputs_number);

    set_default();
}


/// Sets a new multilayer_perceptron_pointer architecture with one hidden layer and zero independent parameters.
/// It also sets the rest of members to their default values. 
/// @param new_inputs_number Number of inputs.
/// @param new_hidden_neurons_number Number of neurons in the hidden layer. 
/// @param new_outputs_number Number of outputs.

void NeuralNetwork::set(const size_t& new_inputs_number, const size_t& new_hidden_neurons_number, const size_t& new_outputs_number)
{
    delete_pointers();

    inputs_pointer = new Inputs(new_inputs_number);

    multilayer_perceptron_pointer = new MultilayerPerceptron(new_inputs_number, new_hidden_neurons_number, new_outputs_number);

    outputs_pointer = new Outputs(new_outputs_number);

    set_default();
}


/// Sets the neural network members by loading them from a XML file.
/// @param file_name Neural network XML file_name.

void NeuralNetwork::set(const string& file_name)
{
    delete_pointers();

    load(file_name);
}


/// Sets the members of this neural network object with those from other neural network object.
/// @param other_neural_network Neural network object to be copied. 

void NeuralNetwork::set(const NeuralNetwork& other_neural_network)
{
    // Pointers

    delete_pointers();

    if(other_neural_network.has_multilayer_perceptron())
    {
        multilayer_perceptron_pointer = new MultilayerPerceptron(*other_neural_network.multilayer_perceptron_pointer);
    }

    if(other_neural_network.has_inputs_trending_layer())
    {
        inputs_trending_layer_pointer = new InputsTrendingLayer(*other_neural_network.inputs_trending_layer_pointer);
    }

    if(other_neural_network.has_scaling_layer())
    {
        scaling_layer_pointer = new ScalingLayer(*other_neural_network.scaling_layer_pointer);
    }

    if(other_neural_network.has_principal_components_layer())
    {
        principal_components_layer_pointer = new PrincipalComponentsLayer(*other_neural_network.principal_components_layer_pointer);
    }

    if(other_neural_network.has_unscaling_layer())
    {
        unscaling_layer_pointer = new UnscalingLayer(*other_neural_network.unscaling_layer_pointer);
    }

    if(other_neural_network.has_outputs_trending_layer())
    {
        outputs_trending_layer_pointer = new OutputsTrendingLayer(*other_neural_network.outputs_trending_layer_pointer);
    }

    if(other_neural_network.has_bounding_layer())
    {
        bounding_layer_pointer = new BoundingLayer(*other_neural_network.bounding_layer_pointer);
    }

    if(other_neural_network.has_probabilistic_layer())
    {
        probabilistic_layer_pointer = new ProbabilisticLayer(*other_neural_network.probabilistic_layer_pointer);
    }

    if(other_neural_network.has_inputs())
    {
        inputs_pointer = new Inputs(*other_neural_network.inputs_pointer);
    }

    if(other_neural_network.has_outputs())
    {
        outputs_pointer = new Outputs(*other_neural_network.outputs_pointer);
    }

    // Other

    display = other_neural_network.display;
}


void NeuralNetwork::set_inputs(const Vector<bool>& new_uses)
{
    const size_t new_inputs_number = new_uses.count_equal_to(true);

    if(new_inputs_number == 0)
    {
        set();
    }

    if(inputs_pointer)
    {
        inputs_pointer->set(new_uses);
    }

    if(scaling_layer_pointer)
    {
        scaling_layer_pointer->set(new_uses);
    }

    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->set_inputs_number(new_inputs_number);
    }
}


/// Sets those members which are not pointer to their default values.

void NeuralNetwork::set_default()
{
    display = true;
}

#ifdef __OPENNN_MPI__

void NeuralNetwork::set_MPI(const NeuralNetwork* neural_network)
{

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int layers_number;
    Vector<int> architecture;

    int parameters_number;
    Vector<double> parameters;

    int inputs_number;
    int outputs_number;

    int* activation_functions;

    if(rank == 0)
    {
        // Variables to send initialization

        const MultilayerPerceptron* original_multilayer_perceptron_pointer = neural_network->get_multilayer_perceptron_pointer();

        layers_number = (int)original_multilayer_perceptron_pointer->get_layers_number();
        architecture = original_multilayer_perceptron_pointer->get_architecture_int();

        parameters = original_multilayer_perceptron_pointer->get_parameters();
        parameters_number = (int)parameters.size();

        const Vector<Perceptron::ActivationFunction> layers_activation_functions = original_multilayer_perceptron_pointer->get_layers_activation_function();

        activation_functions = (int *)malloc(layers_number*sizeof(int));

        for(int i = 0; i < layers_number; i++)
        {
            activation_functions[i] = (int)layers_activation_functions[i];
        }

        inputs_number = (int)original_multilayer_perceptron_pointer->get_inputs_number();
        outputs_number = (int)original_multilayer_perceptron_pointer->get_outputs_number();
    }

    // Send variables

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank > 0)
    {
        MPI_Request req[2];

        MPI_Irecv(&layers_number, 1, MPI_INT, rank-1, 3, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(&parameters_number, 1, MPI_INT, rank-1, 4, MPI_COMM_WORLD, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);

        architecture.set(layers_number+1);
        parameters.set(parameters_number);

        MPI_Irecv(architecture.data(),(int)layers_number+1, MPI_INT, rank-1, 9, MPI_COMM_WORLD, &req[0]);
        MPI_Irecv(parameters.data(),(int)parameters_number, MPI_DOUBLE, rank-1, 10, MPI_COMM_WORLD, &req[1]);

        MPI_Waitall(2, req, MPI_STATUS_IGNORE);

        MPI_Request* req_activations = (MPI_Request*)malloc(layers_number*sizeof(MPI_Request));

        activation_functions = (int *)malloc(layers_number*sizeof(int));

        for(int i = 0; i < layers_number; i++)
        {
            MPI_Irecv(&activation_functions[i], 1, MPI_INT, rank-1, 11+i, MPI_COMM_WORLD, &req_activations[i]);
        }

        MPI_Waitall((int)layers_number, req_activations, MPI_STATUS_IGNORE);

        free(req_activations);
    }

    if(rank < size-1)
    {
        MPI_Request req[4];

        MPI_Isend(&layers_number, 1, MPI_INT, rank+1, 3, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&parameters_number, 1, MPI_INT, rank+1, 4, MPI_COMM_WORLD, &req[1]);

        MPI_Isend(architecture.data(),(int)layers_number+1, MPI_INT, rank+1, 9, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(parameters.data(),(int)parameters_number, MPI_DOUBLE, rank+1, 10, MPI_COMM_WORLD, &req[3]);

        MPI_Waitall(4, req, MPI_STATUS_IGNORE);

        MPI_Request* req_activations = (MPI_Request*)malloc(layers_number*sizeof(MPI_Request));

        for(int i = 0; i < layers_number; i++)
        {
            MPI_Isend(&activation_functions[i], 1, MPI_INT, rank+1, 11+i, MPI_COMM_WORLD, &req_activations[i]);
        }

        MPI_Waitall((int)layers_number, req_activations, MPI_STATUS_IGNORE);

        free(req_activations);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(!has_multilayer_perceptron())
    {
        multilayer_perceptron_pointer = new MultilayerPerceptron(architecture);
    }
    else
    {
        multilayer_perceptron_pointer->set(architecture);
    }

    multilayer_perceptron_pointer->set_parameters(parameters);

    for(int i = 0; i < layers_number; i++)
    {
        multilayer_perceptron_pointer->set_layer_activation_function(i,(Perceptron::ActivationFunction)activation_functions[i]);
    }

    free(activation_functions);
}
#endif


/// Sets a new multilayer perceptron within the neural network.
/// @param new_multilayer_perceptron_pointer Pointer to a multilayer perceptron object. 
/// Note that the neural network destructror will delete this pointer. 

void NeuralNetwork::set_multilayer_perceptron_pointer(MultilayerPerceptron* new_multilayer_perceptron_pointer)
{
    if(new_multilayer_perceptron_pointer != multilayer_perceptron_pointer)
    {
        delete multilayer_perceptron_pointer;

        multilayer_perceptron_pointer = new_multilayer_perceptron_pointer;
    }
}


/// Sets a new inputs trending layer within the neural network.
/// @param new_inputs_trending_layer_pointer Pointer to an inputs trending layer object.
/// Note that the neural network destructror will delete this pointer.

void NeuralNetwork::set_inputs_trending_layer_pointer(InputsTrendingLayer* new_inputs_trending_layer_pointer)
{
    if(new_inputs_trending_layer_pointer != inputs_trending_layer_pointer)
    {
        delete inputs_trending_layer_pointer;

        inputs_trending_layer_pointer = new_inputs_trending_layer_pointer;
    }
}


/// Sets a new scaling layer within the neural network.
/// @param new_scaling_layer_pointer Pointer to a scaling layer object. 
/// Note that the neural network destructror will delete this pointer. 

void NeuralNetwork::set_scaling_layer_pointer(ScalingLayer* new_scaling_layer_pointer)
{
    if(new_scaling_layer_pointer != scaling_layer_pointer)
    {
        delete scaling_layer_pointer;

        scaling_layer_pointer = new_scaling_layer_pointer;
    }
}


/// Sets a new principal components layer within the neural network.
/// @param new_principal_components_layer_pointer Pointer to a principal components layer object.
/// Note that the neural network destructror will delete this pointer.

void NeuralNetwork::set_principal_components_layer_pointer(PrincipalComponentsLayer* new_principal_components_layer_pointer)
{
    if(new_principal_components_layer_pointer != principal_components_layer_pointer)
    {
        delete principal_components_layer_pointer;

        principal_components_layer_pointer = new_principal_components_layer_pointer;
    }
}


/// Sets a new unscaling layer within the neural network.
/// @param new_unscaling_layer_pointer Pointer to an unscaling layer object. 
/// Note that the neural network destructror will delete this pointer. 

void NeuralNetwork::set_unscaling_layer_pointer(UnscalingLayer* new_unscaling_layer_pointer)
{
    if(new_unscaling_layer_pointer != unscaling_layer_pointer)
    {
        delete unscaling_layer_pointer;

        unscaling_layer_pointer = new_unscaling_layer_pointer;
    }
}


/// Sets a new outputs trending layer within the neural network.
/// @param new_outputs_trending_layer_pointer Pointer to an outputs trending layer object.
/// Note that the neural network destructror will delete this pointer.

void NeuralNetwork::set_outputs_trending_layer_pointer(OutputsTrendingLayer* new_outputs_trending_layer_pointer)
{
    if(new_outputs_trending_layer_pointer != outputs_trending_layer_pointer)
    {
        delete outputs_trending_layer_pointer;

        outputs_trending_layer_pointer = new_outputs_trending_layer_pointer;
    }
}


/// Sets a new bounding layer within the neural network.
/// @param new_bounding_layer_pointer Pointer to a bounding layer object. 
/// Note that the neural network destructror will delete this pointer. 

void NeuralNetwork::set_bounding_layer_pointer(BoundingLayer* new_bounding_layer_pointer)
{
    if(new_bounding_layer_pointer != bounding_layer_pointer)
    {
        delete bounding_layer_pointer;

        bounding_layer_pointer = new_bounding_layer_pointer;
    }
}


/// Sets a new probabilistic layer within the neural network.
/// @param new_probabilistic_layer_pointer Pointer to a probabilistic layer object. 
/// Note that the neural network destructror will delete this pointer. 

void NeuralNetwork::set_probabilistic_layer_pointer(ProbabilisticLayer* new_probabilistic_layer_pointer)
{
    if(new_probabilistic_layer_pointer != probabilistic_layer_pointer)
    {
        delete probabilistic_layer_pointer;

        probabilistic_layer_pointer = new_probabilistic_layer_pointer;
    }
}


/// Sets a new inputs object within the neural network.
/// @param new_inputs_pointer Pointer to an inputs object.
/// Note that the neural network destructror will delete this pointer. 

void NeuralNetwork::set_inputs_pointer(Inputs* new_inputs_pointer)
{
    if(new_inputs_pointer != inputs_pointer)
    {
        delete inputs_pointer;

        inputs_pointer = new_inputs_pointer;
    }
}


/// Sets a new outputs object within the neural network.
/// @param new_outputs_pointer Pointer to an outputs object.
/// Note that the neural network destructror will delete this pointer.

void NeuralNetwork::set_outputs_pointer(Outputs* new_outputs_pointer)
{
    if(new_outputs_pointer != outputs_pointer)
    {
        delete outputs_pointer;

        outputs_pointer = new_outputs_pointer;
    }
}


/// Sets new scaling layer within the neural network.
/// @param new_scaling_layer Scaling layer to be asociated to the neural network.

void NeuralNetwork::set_scaling_layer(ScalingLayer& new_scaling_layer)
{
    delete scaling_layer_pointer;

    scaling_layer_pointer = new ScalingLayer(new_scaling_layer);
}


/// Returns the number of inputs to the neural network.

size_t NeuralNetwork::get_inputs_number() const
{
    size_t inputs_number;

    if(inputs_trending_layer_pointer)
    {
        inputs_number = inputs_trending_layer_pointer->get_inputs_trending_neurons_number();
    }
    else if(scaling_layer_pointer)
    {
        inputs_number = scaling_layer_pointer->get_scaling_neurons_number();
    }
    else if(multilayer_perceptron_pointer)
    {
        inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    }
    else if(unscaling_layer_pointer)
    {
        inputs_number = unscaling_layer_pointer->get_unscaling_neurons_number();
    }
    else if(outputs_trending_layer_pointer)
    {
        inputs_number = outputs_trending_layer_pointer->get_outputs_trending_neurons_number();
    }
    else if(probabilistic_layer_pointer)
    {
        inputs_number = probabilistic_layer_pointer->get_probabilistic_neurons_number();
    }
    else if(bounding_layer_pointer)
    {
        inputs_number = bounding_layer_pointer->get_bounding_neurons_number();
    }
    else
    {
        inputs_number = 0;
    }

    return(inputs_number);
}


/// Returns the number of outputs to the neural network.

size_t NeuralNetwork::get_outputs_number() const
{
    size_t outputs_number;

    if(inputs_trending_layer_pointer)
    {
        outputs_number = inputs_trending_layer_pointer->get_inputs_trending_neurons_number();
    }
    else if(multilayer_perceptron_pointer)
    {
        outputs_number = multilayer_perceptron_pointer->get_outputs_number();
    }
    else if(unscaling_layer_pointer)
    {
        outputs_number = unscaling_layer_pointer->get_unscaling_neurons_number();
    }
    else if(outputs_trending_layer_pointer)
    {
        outputs_number = outputs_trending_layer_pointer->get_outputs_trending_neurons_number();
    }
    else if(probabilistic_layer_pointer)
    {
        outputs_number = probabilistic_layer_pointer->get_probabilistic_neurons_number();
    }
    else if(bounding_layer_pointer)
    {
        outputs_number = bounding_layer_pointer->get_bounding_neurons_number();
    }
    else
    {
        outputs_number = 0;
    }

    return(outputs_number);
}


/// Returns a vector with the architecture of the neural network.
/// The elements of this vector are as follows;
/// <UL>
/// <LI> Number of inputs trending neurons(if there is an inputs trending layer).</LI>
/// <LI> Number of scaling neurons(if there is a scaling layer).</LI>
/// <LI> Number of principal components neurons(if there is a principal components layer).</LI>
/// <LI> Multilayer perceptron architecture(if there is a multilayer perceptron).</LI>
/// <LI> Number of conditions neurons(if there is a conditions layer).</LI>
/// <LI> Number of unscaling neurons(if there is an unscaling layer).</LI>
/// <LI> Number of outputs trending neurons(if there is an outputs trending layer).</LI>
/// <LI> Number of probabilistic neurons(if there is a probabilistic layer).</LI>
/// <LI> Number of bounding neurons(if there is a bounding layer).</LI>
/// </UL>

Vector<size_t> NeuralNetwork::get_architecture() const
{
    Vector<size_t> architecture;

    const size_t inputs_number = get_inputs_number();

    if(inputs_number == 0)
    {
        return(architecture);
    }

    architecture.push_back(inputs_number);

    // Inputs trending layer

    if(inputs_trending_layer_pointer)
    {
        architecture.push_back(inputs_trending_layer_pointer->get_inputs_trending_neurons_number());
    }

    // Scaling layer

    if(scaling_layer_pointer)
    {
        architecture.push_back(scaling_layer_pointer->get_scaling_neurons_number());
    }

    // Principal components layer

    if(principal_components_layer_pointer)
    {
        if(principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
        {
            architecture.push_back(principal_components_layer_pointer->get_principal_components_number());
        }
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        architecture = architecture.assemble(multilayer_perceptron_pointer->get_layers_perceptrons_numbers());
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        architecture.push_back(unscaling_layer_pointer->get_unscaling_neurons_number());
    }

    // Outputs trending layer

    if(outputs_trending_layer_pointer)
    {
        architecture.push_back(outputs_trending_layer_pointer->get_outputs_trending_neurons_number());
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        architecture.push_back(probabilistic_layer_pointer->get_probabilistic_neurons_number());
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        architecture.push_back(bounding_layer_pointer->get_bounding_neurons_number());
    }

    return(architecture);
}


/// Returns the number of parameters in the multilayer perceptron
/// The number of parameters is the sum of all the multilayer perceptron parameters(biases and synaptic weights) and independent parameters.

size_t NeuralNetwork::get_parameters_number() const
{
    size_t parameters_number = 0;

    if(multilayer_perceptron_pointer)
    {
        parameters_number += multilayer_perceptron_pointer->get_parameters_number();
    }

    return(parameters_number);
}


/// Returns the values of the parameters in the multilayer perceptron as a single vector.
/// This contains all the multilayer perceptron parameters(biases and synaptic weights) and preprocessed independent parameters.

Vector<double> NeuralNetwork::get_parameters() const
{
    // Only network parameters

    if(multilayer_perceptron_pointer)
    {
        return(multilayer_perceptron_pointer->get_parameters());
    }
    else
    {
        return Vector<double>();
    }
}



/// Sets all the parameters(multilayer_perceptron_pointer parameters and independent parameters) from a single vector.
/// @param new_parameters New set of parameter values. 

void NeuralNetwork::set_parameters(const Vector<double>& new_parameters)
{
    // Control sentence(if debug)

#ifdef __OPENNN_DEBUG__

    const size_t size = new_parameters.size();

    const size_t parameters_number = get_parameters_number();

    if(size != parameters_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void set_parameters(const Vector<double>&) method.\n"
               << "Size must be equal to number of parameters.\n";

        throw logic_error(buffer.str());
    }

#endif

    if(multilayer_perceptron_pointer)
    {// Only network parameters

        multilayer_perceptron_pointer->set_parameters(new_parameters);
    }
    else
    {// None neural neither independent parameters

        return;
    }
}


/// This method deletes all the pointers composing the neural network:
/// <ul>
/// <li> Inputs.
/// <li> Outputs.
/// <li> Multilayer perceptron.
/// <li> Inputs trending layer.
/// <li> Scaling layer.
/// <li> Unscaling layer.
/// <li> Outputs trending layer.
/// <li> Bounding layer.
/// <li> Probabilistic layer. 
/// <li> Conditions layer. 
/// <li> Independent parameters. 
/// </ul>

void NeuralNetwork::delete_pointers()
{
    delete multilayer_perceptron_pointer;
    delete inputs_trending_layer_pointer;
    delete scaling_layer_pointer;
    delete principal_components_layer_pointer;
    delete unscaling_layer_pointer;
    delete outputs_trending_layer_pointer;
    delete bounding_layer_pointer;
    delete probabilistic_layer_pointer;
    delete inputs_pointer;
    delete outputs_pointer;

    multilayer_perceptron_pointer = nullptr;
    inputs_trending_layer_pointer = nullptr;
    scaling_layer_pointer = nullptr;
    principal_components_layer_pointer = nullptr;
    unscaling_layer_pointer = nullptr;
    outputs_trending_layer_pointer = nullptr;
    bounding_layer_pointer = nullptr;
    probabilistic_layer_pointer = nullptr;
    inputs_pointer = nullptr;
    outputs_pointer = nullptr;
}


/// This method constructs an empty multilayer perceptron within the neural network. 

void NeuralNetwork::construct_multilayer_perceptron()
{
    if(!multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer = new MultilayerPerceptron();
    }
}


/// This method constructs an inputs trending layer within the neural network.
/// The size of the inputs trending layer is the number of inputs in the multilayer perceptron.

void NeuralNetwork::construct_inputs_trending_layer()
{
    if(!inputs_trending_layer_pointer)
    {
        size_t inputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            inputs_number = multilayer_perceptron_pointer->get_inputs_number();
        }

        inputs_trending_layer_pointer = new InputsTrendingLayer(inputs_number);
    }
}


/// This method constructs a scaling layer within the neural network. 
/// The size of the scaling layer is the number of inputs in the multilayer perceptron. 

void NeuralNetwork::construct_scaling_layer()
{
    if(!scaling_layer_pointer)
    {
        size_t inputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            inputs_number = multilayer_perceptron_pointer->get_inputs_number();
        }

        scaling_layer_pointer = new ScalingLayer(inputs_number);
    }
}


/// This method constructs a scaling layer within the neural network.
/// The size of the scaling layer is the number of inputs in the multilayer perceptron.
/// @param input_statistics Inputs Statistics vector.

void NeuralNetwork::construct_scaling_layer(const Vector< Statistics<double> >& input_statistics)
{
    construct_scaling_layer();

    scaling_layer_pointer->set_statistics(input_statistics);
}


/// This method constructs a principal_components layer within the neural network.
/// The size of the principal components layer is the number of inputs in the multilayer perceptron.

void NeuralNetwork::construct_principal_components_layer()
{
    if(!principal_components_layer_pointer)
    {
        size_t inputs_number = 0;
        size_t principal_components_number = 0;

        if(multilayer_perceptron_pointer)
        {
            inputs_number = multilayer_perceptron_pointer->get_inputs_number();
        }

        principal_components_layer_pointer = new PrincipalComponentsLayer(inputs_number, principal_components_number);
    }
}


/// This method constructs an unscaling layer within the neural network. 
/// The size of the unscaling layer is the number of outputs in the multilayer perceptron. 

void NeuralNetwork::construct_unscaling_layer()
{
    if(!unscaling_layer_pointer)
    {
        size_t outputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            outputs_number = multilayer_perceptron_pointer->get_outputs_number();
        }

        unscaling_layer_pointer = new UnscalingLayer(outputs_number);
    }
}


/// This method constructs a scaling layer within the neural network.
/// The size of the scaling layer is the number of inputs in the multilayer perceptron.
/// @param target_statistics Targets Statistics vector.

void NeuralNetwork::construct_unscaling_layer(const Vector< Statistics<double> >& target_statistics)
{
    construct_unscaling_layer();

    scaling_layer_pointer->set_statistics(target_statistics);
}


/// This method constructs an outputs trending layer within the neural network.
/// The size of the outputs trending layer is the number of outputs in the multilayer perceptron.

void NeuralNetwork::construct_outputs_trending_layer()
{
    if(!outputs_trending_layer_pointer)
    {
        size_t outputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            outputs_number = multilayer_perceptron_pointer->get_outputs_number();
        }

        outputs_trending_layer_pointer = new OutputsTrendingLayer(outputs_number);
    }
}


/// This method constructs a bounding layer within the neural network. 
/// The size of the bounding layer is the number of outputs in the multilayer perceptron. 

void NeuralNetwork::construct_bounding_layer()
{
    if(!bounding_layer_pointer)
    {
        size_t outputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            outputs_number = multilayer_perceptron_pointer->get_outputs_number();
        }

        bounding_layer_pointer = new BoundingLayer(outputs_number);
    }
}


/// This method constructs a probabilistic layer within the neural network. 
/// The size of the probabilistic layer is the number of outputs in the multilayer perceptron. 

void NeuralNetwork::construct_probabilistic_layer()
{
    if(!probabilistic_layer_pointer)
    {
        size_t outputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            outputs_number = multilayer_perceptron_pointer->get_outputs_number();
        }

        probabilistic_layer_pointer = new ProbabilisticLayer(outputs_number);
    }
}


/// This method constructs an inputs object within the neural network.
/// The number of inputs is the number of inputs in the multilayer perceptron. 

void NeuralNetwork::construct_inputs()
{
    if(!inputs_pointer)
    {
        size_t inputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            inputs_number = multilayer_perceptron_pointer->get_inputs_number();
        }

        inputs_pointer = new Inputs(inputs_number);
    }
}


/// This method constructs an outputs object within the neural network.
/// The number of outputs is the number of outputs in the multilayer perceptron.

void NeuralNetwork::construct_outputs()
{
    if(!outputs_pointer)
    {
        size_t outputs_number = 0;

        if(multilayer_perceptron_pointer)
        {
            outputs_number = multilayer_perceptron_pointer->get_outputs_number();
        }

        outputs_pointer = new Outputs(outputs_number);
    }
}


/// This method deletes the multilayer perceptron within the neural network. 

void NeuralNetwork::destruct_multilayer_perceptron()
{
    delete multilayer_perceptron_pointer;

    multilayer_perceptron_pointer = nullptr;
}


/// This method deletes the inputs trending layer within the neural network.

void NeuralNetwork::destruct_inputs_trending_layer()
{
    delete inputs_trending_layer_pointer;

    inputs_trending_layer_pointer = nullptr;
}


/// This method deletes the scaling layer within the neural network. 

void NeuralNetwork::destruct_scaling_layer()
{
    delete scaling_layer_pointer;

    scaling_layer_pointer = nullptr;
}


/// This method deletes the unscaling layer within the neural network. 

void NeuralNetwork::destruct_unscaling_layer()
{
    delete unscaling_layer_pointer;

    unscaling_layer_pointer = nullptr;
}


/// This method deletes the outputs trending layer within the neural network.

void NeuralNetwork::destruct_outputs_trending_layer()
{
    delete outputs_trending_layer_pointer;

    outputs_trending_layer_pointer = nullptr;
}


/// This method deletes the bounding layer within the neural network. 

void NeuralNetwork::destruct_bounding_layer()
{
    delete bounding_layer_pointer;

    bounding_layer_pointer = nullptr;
}


/// This method deletes the probabilistic layer within the neural network. 

void NeuralNetwork::destruct_probabilistic_layer()
{
    delete probabilistic_layer_pointer;

    probabilistic_layer_pointer = nullptr;
}


/// This method deletes the inputs object within the neural network.

void NeuralNetwork::destruct_inputs()
{
    delete inputs_pointer;

    inputs_pointer = nullptr;
}


/// This method deletes the outputs object within the neural network.

void NeuralNetwork::destruct_outputs()
{
    delete outputs_pointer;

    outputs_pointer = nullptr;
}


/// Initializes the neural network at random.
/// This is useful for testing purposes. 

void NeuralNetwork::initialize_random()
{
    size_t inputs_number;
    size_t outputs_number;

    // Multilayer perceptron

    if(rand()%5)
    {
        if(!multilayer_perceptron_pointer)
        {
            multilayer_perceptron_pointer = new MultilayerPerceptron();
        }

        multilayer_perceptron_pointer->initialize_random();

        inputs_number = multilayer_perceptron_pointer->get_inputs_number();
        outputs_number = multilayer_perceptron_pointer->get_outputs_number();
    }
    else
    {
        inputs_number =  rand()%10 + 1;
        outputs_number =  rand()%10 + 1;
    }

    // Inputs trending layer

    if(rand()%5)
    {
        if(!inputs_trending_layer_pointer)
        {
            inputs_trending_layer_pointer = new InputsTrendingLayer(inputs_number);
        }

        inputs_trending_layer_pointer->initialize_random();
    }

    // Scaling layer

    if(rand()%5)
    {
        if(!scaling_layer_pointer)
        {
            scaling_layer_pointer = new ScalingLayer(inputs_number);
        }

        scaling_layer_pointer->initialize_random();
    }

    // Unscaling layer

    if(rand()%5)
    {
        if(!unscaling_layer_pointer)
        {
            unscaling_layer_pointer = new UnscalingLayer(outputs_number);
        }

        unscaling_layer_pointer->initialize_random();
    }

    // Outputs trending layer

    if(rand()%5)
    {
        if(!outputs_trending_layer_pointer)
        {
            outputs_trending_layer_pointer = new OutputsTrendingLayer(outputs_number);
        }

        outputs_trending_layer_pointer->initialize_random();
    }

    // Bounding layer

    if(rand()%5)
    {
        if(!bounding_layer_pointer)
        {
            bounding_layer_pointer = new BoundingLayer(outputs_number);
        }

        bounding_layer_pointer->initialize_random();
    }

    // Probabilistic layer

    if(rand()%5)
    {
        if(!probabilistic_layer_pointer)
        {
            probabilistic_layer_pointer = new ProbabilisticLayer(outputs_number);
        }

        probabilistic_layer_pointer->initialize_random();
    }

    // Inputs

    if(rand()%5)
    {
        if(!inputs_pointer)
        {
            inputs_pointer = new Inputs(inputs_number);
        }
    }

    // Outputs

    if(rand()%5)
    {
        if(!outputs_pointer)
        {
            outputs_pointer = new Outputs(outputs_number);
        }
    }
}


/// Sets a new display value.
/// If it is set to true messages from this class are to be displayed on the screen;
/// if it is set to false messages from this class are not to be displayed on the screen.
/// @param new_display Display value.

void NeuralNetwork::set_display(const bool& new_display)
{
    display = new_display;
}


/// Add an input to the neural network and asociate the statistics to this input.
/// @param new_statistics Values of the statistics of the new input. The default value is an empty vector.

void NeuralNetwork::grow_input(const Statistics<double>& new_statistics)
{
    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void grow_input(const size_t&) method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    multilayer_perceptron_pointer->grow_input();

    if(scaling_layer_pointer)
    {
        scaling_layer_pointer->grow_scaling_neuron(new_statistics);
    }

    if(inputs_pointer)
    {
        inputs_pointer->grow_input();
    }
}


/// Removes a given input to the neural network.
/// This involves removing the input itself and the corresponding scaling layer,
/// conditions layer and multilayer perceptron inputs.
/// @param index Index of input to be pruned.

void NeuralNetwork::prune_input(const size_t& index)
{
    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void prune_input(const size_t&) method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    multilayer_perceptron_pointer->prune_input(index);  

    if(inputs_trending_layer_pointer)
    {
        inputs_trending_layer_pointer->prune_input_trending_neuron(index);
    }

    if(scaling_layer_pointer)
    {
        scaling_layer_pointer->prune_scaling_neuron(index);
    }

    if(inputs_pointer)
    {
        inputs_pointer->prune_input(index);
    }
}


/// Removes a given output from the neural network.
/// This involves removing the output itself and the corresponding unscaling layer,
/// conditions layer, probabilistic layer, bounding layer and multilayer perceptron outputs.
/// @param index Index of output to be pruned.

void NeuralNetwork::prune_output(const size_t& index)
{
    if(!multilayer_perceptron_pointer)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void prune_output(const size_t&) method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    multilayer_perceptron_pointer->prune_output(index);

    if(unscaling_layer_pointer)
    {
        unscaling_layer_pointer->prune_unscaling_neuron(index);
    }

    if(outputs_trending_layer_pointer)
    {
        outputs_trending_layer_pointer->prune_output_trending_neuron(index);
    }

    if(bounding_layer_pointer)
    {
        bounding_layer_pointer->prune_bounding_neuron(index);
    }

    if(probabilistic_layer_pointer)
    {
        probabilistic_layer_pointer->prune_probabilistic_neuron();
    }

    if(outputs_pointer)
    {
        outputs_pointer->prune_output(index);
    }
}


/// @todo

void NeuralNetwork::resize_inputs_number(const size_t&)
{

}


/// @todo

void NeuralNetwork::resize_outputs_number(const size_t&)
{

}


/// Returns the number of layers in the neural network.
/// That includes perceptron, scaling, unscaling, inputs trending, outputs trending, bounding, probabilistic or conditions layers.

size_t NeuralNetwork::get_layers_number() const
{
    size_t layers_number = 0;

    if(multilayer_perceptron_pointer)
    {
        layers_number += multilayer_perceptron_pointer->get_layers_number();
    }

    if(inputs_trending_layer_pointer)
    {
        layers_number += 1;
    }

    if(scaling_layer_pointer)
    {
        layers_number += 1;
    }

    if(principal_components_layer_pointer)
    {
        layers_number += 1;
    }

    if(unscaling_layer_pointer)
    {
        layers_number += 1;
    }

    if(outputs_trending_layer_pointer)
    {
        layers_number += 1;
    }

    if(bounding_layer_pointer)
    {
        layers_number += 1;
    }

    if(probabilistic_layer_pointer)
    {
        layers_number += 1;
    }

    return(layers_number);
}


/// Initializes all the neural and the independent parameters with a given value.

void NeuralNetwork::initialize_parameters(const double& value)
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->initialize_parameters(value);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent parameters)
/// at random with values comprised between -1 and +1.

void NeuralNetwork::randomize_parameters_uniform()
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_uniform();
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values comprised between a given minimum and a given maximum values.
/// @param minimum Minimum initialization value.
/// @param maximum Maximum initialization value.

void NeuralNetwork::randomize_parameters_uniform(const double& minimum, const double& maximum)
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_uniform(minimum, maximum);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values comprised between a different minimum and maximum numbers for each free 
/// parameter.
/// @param minimum Vector of minimum initialization values.
/// @param maximum Vector of maximum initialization values.

void NeuralNetwork::randomize_parameters_uniform(const Vector<double>& minimum, const Vector<double>& maximum)
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_uniform(minimum, maximum);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) values comprised between a different minimum and maximum numbers for each parameter.
/// Minimum and maximum initialization values are given from a vector of two real vectors.
/// The first element must contain the minimum initialization value for each parameter.
/// The second element must contain the maximum initialization value for each parameter.
/// @param minimum_maximum Vector of minimum and maximum initialization vectors.

void NeuralNetwork::randomize_parameters_uniform(const Vector< Vector<double> >& minimum_maximum)
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_uniform(minimum_maximum);
    }

}


/// Initializes all the parameters in the neural newtork(biases and synaptic weiths + independent
/// parameters) at random with values chosen from a normal distribution with mean 0 and standard deviation 1.

void NeuralNetwork::randomize_parameters_normal()
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_normal();
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values chosen from a normal distribution with a given mean and a given standard 
/// deviation.
/// @param mean Mean of normal distribution.
/// @param standard_deviation Standard deviation of normal distribution.

void NeuralNetwork::randomize_parameters_normal(const double& mean, const double& standard_deviation)
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_normal(mean, standard_deviation);
    }
}


/// Initializes all the parameters in the neural newtork(biases and synaptic weiths +
/// independent parameters) at random with values chosen from normal distributions with a given mean and a given 
/// standard deviation for each parameter.
/// @param mean Vector of minimum initialization values.
/// @param standard_deviation Vector of maximum initialization values.

void NeuralNetwork::randomize_parameters_normal(const Vector<double>& mean, const Vector<double>& standard_deviation)
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_normal(mean, standard_deviation);
    }
}


/// Initializes all the parameters in the newtork(biases and synaptic weiths + independent
/// parameters) at random with values chosen from normal distributions with a given mean and a given standard 
/// deviation for each parameter.
/// All mean and standard deviation values are given from a vector of two real vectors.
/// The first element must contain the mean value for each parameter.
/// The second element must contain the standard deviation value for each parameter.
/// @param mean_standard_deviation Mean and standard deviation vectors.

void NeuralNetwork::randomize_parameters_normal(const Vector< Vector<double> >& mean_standard_deviation)
{
    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->randomize_parameters_normal(mean_standard_deviation);
    }
}


/// Returns the norm of the vector of parameters.

double NeuralNetwork::calculate_parameters_norm() const
{
    const Vector<double> parameters = get_parameters();

    const double parameters_norm = parameters.calculate_L2_norm();

    return(parameters_norm);
}


/// Returns a statistics structure of the parameters vector.
/// That contains the minimum, maximum, mean and standard deviation values of the parameters.

Statistics<double> NeuralNetwork::calculate_parameters_statistics() const
{
    const Vector<double> parameters = get_parameters();

    return(parameters.calculate_statistics());
}


/// Returns a histogram structure of the parameters vector.
/// That will be used for looking at the distribution of the parameters.
/// @param bins_number Number of bins in the histogram(10 by default).

Histogram<double> NeuralNetwork::calculate_parameters_histogram(const size_t& bins_number) const
{
    const Vector<double> parameters = get_parameters();

    return(parameters.calculate_histogram(bins_number));
}


/// Perturbate parameters of the multilayer perceptron.
/// @param perturbation Maximum distance of perturbation.

void NeuralNetwork::perturbate_parameters(const double& perturbation)
{
#ifdef __OPENNN_DEBUG__

    if(perturbation < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void perturbate_parameters(const double&) method.\n"
               << "Perturbation must be equal or greater than 0.\n";

        throw logic_error(buffer.str());
    }

#endif

    Vector<double>parameters = get_parameters();

    Vector<double>parameters_perturbation(parameters);

    parameters_perturbation.randomize_uniform(-perturbation,perturbation);

    parameters = parameters + parameters_perturbation;

    set_parameters(parameters);
}


/// Calculates the inputs importance for a neural network with only one hidden layer.
/// Returns a vector containing the importance for each of the inputs with respect to a given output.
/// The size of the vector is the number of inputs of the neural network.
/// @param output_index Index of the output.

Vector<double> NeuralNetwork::calculate_inputs_importance_parameters(const size_t& output_index) const
{
    #ifdef __OPENNN_DEBUG__

    if(get_layers_number() != 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Vector<double> calculate_inputs_importance_parameters() const method.\n"
               << "Number of layers must be 2.\n";

        throw logic_error(buffer.str());
    }

    #endif

    const size_t inputs_number = get_inputs_number();

    #ifdef __OPENNN_DEBUG__

    if(output_index >= get_outputs_number() || output_index < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "Vector<double> calculate_inputs_importance_parameters() const method.\n"
               << "Not valid output index.\n";

        throw logic_error(buffer.str());
    }

    #endif

    Vector<double> inputs_importance(inputs_number, 0.0);

//    const size_t layers_number = get_layers_number();

    Vector<PerceptronLayer> layers = multilayer_perceptron_pointer->get_layers();

    const Vector<size_t> layers_size = multilayer_perceptron_pointer->get_architecture();
    const size_t layers_number = layers_size.size();

//    cout << "Architecture: " << layers_size << endl;

    const Vector< Matrix<double> > layers_synaptic_weights = multilayer_perceptron_pointer->get_layers_synaptic_weights();

    const size_t hidden_layer_neurons_number = layers_synaptic_weights[0].get_rows_number();

    // Relative weights

    //Matrix<double> products(layers_size[1], layers_number-2);

    size_t products_number = 1;

    for(size_t i = 1; i < layers_number-1; i++)
    {
        products_number *= products_number*layers_size[i];
    }

//    cout << "products number: " << products_number << endl;
//    cout << "layers number: " << layers.size() << endl;

//    cout << "layer 0 weight: " << layers[0].get_synaptic_weights() << endl;
//    cout << "layer 1 weight: " << layers[1].get_synaptic_weights() << endl;

//    Vector<double> products(products_number, 1.0);

//    for(size_t i = 1; i < layers_number; i++)
//    {
//       for(size_t j = 0; j < layers_size[i]; j++)
//       {
//           for(size_t k = 0; k < layers[i].get_synaptic_weights().size(); k++)
//           {

//           }
//       }
//    }

    // Relative importance

    double numerator = 0.0;
    double first_term_numerator = 0.0;

    double denominator = 0.0;
    double first_term_denominator = 0.0;
    double second_term_denominator = 0.0;

    for(size_t k = 0; k < inputs_number; k++)
    {
        numerator = 0.0;
        denominator = 0.0;

        // Numerator

        for(size_t i = 0; i < hidden_layer_neurons_number; i++)
        {
            first_term_numerator = 0.0;

            for(size_t j = 0; j < inputs_number; j++)
            {
                first_term_numerator += fabs(layers_synaptic_weights[0](i,j));
            }

            numerator += fabs(layers_synaptic_weights[0](i,k))/first_term_numerator*fabs(layers_synaptic_weights[1](output_index,i));
        }

        // Denominator

        for(size_t j = 0; j < inputs_number; j++)
        {
            second_term_denominator = 0.0;

            for(size_t i = 0; i < hidden_layer_neurons_number; i++)
            {
                first_term_denominator = 0.0;

                for(size_t l = 0; l < inputs_number; l++)
                {
                    first_term_denominator += fabs(layers_synaptic_weights[0](i,l));
                }

                second_term_denominator += fabs(layers_synaptic_weights[0](i,j))/first_term_denominator*fabs(layers_synaptic_weights[1](output_index,i));
            }

            denominator += second_term_denominator;
        }

        inputs_importance[k] = numerator/denominator/**100.0*/;
    }

    return inputs_importance;
}


/// Calculates the outputs vector from the multilayer perceptron in response to an inputs vector.
/// The activity for that is the following:
/// <ul>
/// <li> Check inputs range.
/// <li> Calculate scaled inputs.
/// <li> Calculate forward propagation.
/// <li> Calculate unscaled outputs.
/// <li> Apply boundary condtions.
/// <li> Calculate bounded outputs. 
/// </ul>
/// @param inputs Set of inputs to the neural network.

Matrix<double> NeuralNetwork::calculate_outputs(const Matrix<double>& inputs) const
{
    // Control sentence(if debug)

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

#ifdef __OPENNN_DEBUG__

    if(multilayer_perceptron_pointer)
    {
        const size_t inputs_size = inputs.size();

        if(inputs_size != inputs_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "Vector<double> calculate_outputs(const Vector<double>&) const method.\n"
                   << "Size of inputs must be equal to number of inputs.\n";

            throw logic_error(buffer.str());
        }
    }

#endif

    const size_t points_number = inputs.get_rows_number();

    Matrix<double> outputs(points_number, inputs_number,0.0);

    // Scaling layer

    if(scaling_layer_pointer)
    {
        outputs = scaling_layer_pointer->calculate_outputs(inputs);
    }

    // Principal components layer

    if(principal_components_layer_pointer)
    {
       outputs = principal_components_layer_pointer->calculate_outputs(outputs);
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        outputs = multilayer_perceptron_pointer->calculate_outputs(outputs);
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        outputs = unscaling_layer_pointer->calculate_outputs(outputs);
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        outputs = probabilistic_layer_pointer->calculate_outputs(outputs);
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        outputs = bounding_layer_pointer->calculate_outputs(outputs);
    }

    return(outputs);
}


Eigen::MatrixXd NeuralNetwork::calculate_outputs_eigen(const Eigen::MatrixXd& inputs_eigen) const
{
    const size_t points_number = inputs_eigen.rows();
    const size_t inputs_number = get_inputs_number();

    Matrix<double> inputs(points_number, inputs_number);

    Eigen::Map<Eigen::MatrixXd> aux((double*)inputs.data(), points_number, inputs_number);
    aux = inputs_eigen;


    const Matrix<double> outputs = calculate_outputs(inputs);

    const Eigen::Map<Eigen::MatrixXd> outputs_eigen((double*)outputs.data(), points_number, outputs.get_columns_number());


    return(outputs_eigen);
}


/// Calculates the outputs vector from the multilayer perceptron in response to an inputs vector
/// and a time value.
/// The activity for that is the following:
/// <ul>
/// <li> Check inputs range.
/// <li> Subtract the trend.
/// <li> Calculate scaled inputs.
/// <li> Calculate forward propagation.
/// <li> Calculate unscaled outputs.
/// <li> Add the trend.
/// <li> Apply boundary condtions.
/// <li> Calculate bounded outputs.
/// </ul>
/// @param inputs Set of inputs to the neural network.
/// @param time

Matrix<double> NeuralNetwork::calculate_outputs(const Matrix<double>& inputs, const double& time) const
{
    // Control sentence(if debug)

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

#ifdef __OPENNN_DEBUG__

    if(multilayer_perceptron_pointer)
    {
        const size_t inputs_size = inputs.size();

        if(inputs_size != inputs_number)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "Vector<double> calculate_outputs(const Vector<double>&, const double&) const method.\n"
                   << "Size of inputs must be equal to number of inputs.\n";

            throw logic_error(buffer.str());
        }
    }

#endif

    Matrix<double> outputs(inputs.get_rows_number(), inputs_number);

    // Inputs trending layer

    if(inputs_trending_layer_pointer)
    {
        outputs = inputs_trending_layer_pointer->calculate_outputs(inputs, time);
    }

    // Scaling layer

    if(scaling_layer_pointer)
    {
        outputs = scaling_layer_pointer->calculate_outputs(outputs);
    }

    // Principal components layer

    if(principal_components_layer_pointer)
    {
       outputs = principal_components_layer_pointer->calculate_outputs(outputs);
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        outputs = multilayer_perceptron_pointer->calculate_outputs(outputs);
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        outputs = unscaling_layer_pointer->calculate_outputs(outputs);
    }

    // Outputs trending layer

    if(outputs_trending_layer_pointer)
    {
        outputs = outputs_trending_layer_pointer->calculate_outputs(outputs, time);
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        outputs = probabilistic_layer_pointer->calculate_outputs(outputs);
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        outputs = bounding_layer_pointer->calculate_outputs(outputs);
    }

    return(outputs);
}


/// Calculates the input data which is necessary to compute the output data from the neural network in some direction.
/// @param direction Input index(must be between 0 and number of inputs - 1).
/// @param point Input point through the directional input passes.
/// @param minimum Minimum value of the input with the above index.
/// @param maximum Maximum value of the input with the above index.
/// @param points_number Number of points in the directional input data set.

Matrix<double> NeuralNetwork::calculate_directional_inputs(const size_t& direction,
                                                               const Vector<double>& point,
                                                               const double& minimum,
                                                               const double& maximum,
                                                               const size_t& points_number) const
{
    const size_t inputs_number = inputs_pointer->get_inputs_number();

    Matrix<double> directional_inputs(points_number, inputs_number);

    Vector<double> inputs(inputs_number);

    inputs = point;

    for(size_t i = 0; i < points_number; i++)
    {
        inputs[direction] = minimum + (maximum-minimum)*i/static_cast<double>(points_number-1);

        directional_inputs.set_row(i, inputs);
    }

    return(directional_inputs);
}


/// Returns the Jacobian Matrix of the neural network for a set of inputs, corresponding to the
/// point in inputs space at which the Jacobian Matrix is to be found. It uses a forward-propagation method.
/// @param inputs Set of inputs to the neural network.

Vector< Matrix<double> > NeuralNetwork::calculate_Jacobian(const Matrix<double>& inputs) const
{
#ifdef __OPENNN_DEBUG__

    const size_t size = inputs.size();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    if(size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void calculate_Jacobian(const Vector<double>&) const method.\n"
               << "Size must be equal to number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif

    Matrix<double> outputs(inputs);

    Vector< Matrix<double> > scaling_layer_Jacobian;
    Vector< Matrix<double> > principal_components_layer_Jacobian;
    Vector< Matrix<double> > unscaling_layer_Jacobian;
    Vector< Matrix<double> > multilayer_perceptron_Jacobian;
    Vector< Matrix<double> > bounding_layer_Jacobian;
    Vector< Matrix<double> > probabilistic_layer_Jacobian;

    // Scaling layer

    if(scaling_layer_pointer)
    {
        const Matrix<double> scaling_layer_derivative = scaling_layer_pointer->calculate_derivatives(outputs);

        scaling_layer_Jacobian = scaling_layer_pointer->calculate_Jacobian(scaling_layer_derivative);

        outputs = scaling_layer_pointer->calculate_outputs(inputs);
    }


    // Principal components layer

    if(principal_components_layer_pointer)
    {
        principal_components_layer_Jacobian = principal_components_layer_pointer->calculate_Jacobian(outputs);

        outputs = principal_components_layer_pointer->calculate_outputs(outputs);
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_Jacobian = multilayer_perceptron_pointer->calculate_Jacobian(outputs);

        outputs = multilayer_perceptron_pointer->calculate_outputs(outputs);
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        const Matrix<double> unscaling_layer_derivative = unscaling_layer_pointer->calculate_derivatives(outputs);

        unscaling_layer_Jacobian = unscaling_layer_pointer->calculate_Jacobian(unscaling_layer_derivative);

        outputs = unscaling_layer_pointer->calculate_outputs(outputs);
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        probabilistic_layer_Jacobian = probabilistic_layer_pointer->calculate_Jacobian(outputs);

        outputs = probabilistic_layer_pointer->calculate_outputs(outputs);
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        const Matrix<double> derivatives = bounding_layer_pointer->calculate_derivatives(outputs);

        bounding_layer_Jacobian = bounding_layer_pointer->calculate_Jacobian(derivatives);

        outputs = bounding_layer_pointer->calculate_outputs(outputs);
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t points_number = inputs.get_rows_number();

    Vector< Matrix<double> > Jacobian(points_number);

    for(size_t i = 0; i < points_number; i++)
    {
        Jacobian[i].set(outputs_number, outputs_number, 0.0);
        Jacobian[i].set_diagonal(1.0);
    }


    // Bounding layer

    if(bounding_layer_pointer && bounding_layer_pointer->write_bounding_method() != "NoBounding")
    {
        for(size_t i = 0; i < points_number; i++)
        {
            Jacobian[i] = Jacobian[i].dot(bounding_layer_Jacobian[i]);
        }
    }

    // Probabilistic outputs

    if(probabilistic_layer_pointer && probabilistic_layer_pointer->write_probabilistic_method() != "NoProbabilistic")
    {
        for(size_t i = 0; i < points_number; i++)
        {
            Jacobian[i] = Jacobian[i].dot(probabilistic_layer_Jacobian[i]);
        }
    }

    // Unscaling layer

    if(unscaling_layer_pointer && unscaling_layer_pointer->write_unscaling_method() != "NoUnscaling")
    {
        for(size_t i = 0; i < points_number; i++)
        {
            Jacobian[i] = Jacobian[i].dot(unscaling_layer_Jacobian[i]);
        }
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        for(size_t i = 0; i < points_number; i++)
        {
            Jacobian[i] = Jacobian[i].dot(multilayer_perceptron_Jacobian[i]);
        }
    }

    // Principal components layer

    if(principal_components_layer_pointer && principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
    {
        for(size_t i = 0; i < points_number; i++)
        {
            Jacobian[i] = Jacobian[i].dot(principal_components_layer_Jacobian[i]);
        }
    }

    // Scaling layer

    if(scaling_layer_pointer && scaling_layer_pointer->write_scaling_methods() != "NoScaling")
    {
        for(size_t i = 0; i < points_number; i++)
        {
            Jacobian[i] = Jacobian[i].dot(scaling_layer_Jacobian[i]);
        }
    }

    return(Jacobian);
}


/// Returns the Jacobian Matrix of the neural network for a set of inputs and a given time, corresponding to the
/// point in inputs space at which the Jacobian Matrix is to be found. It uses a forward-propagation method.
/// @param inputs Set of inputs to the neural network.
/// @param time Instant of time.
/// @todo

Matrix<double> NeuralNetwork::calculate_Jacobian(const Vector<double>& inputs, const double& time) const
{
#ifdef __OPENNN_DEBUG__

    const size_t size = inputs.size();

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

    if(size != inputs_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void calculate_Jacobian(const Vector<double>&, const double&) const method.\n"
               << "Size must be equal to number of inputs.\n";

        throw logic_error(buffer.str());
    }

#endif
/*
    Vector<double> outputs(inputs);

    Matrix<double> inputs_trending_layer_Jacobian;
    Matrix<double> scaling_layer_Jacobian;
    Matrix<double> principal_components_layer_Jacobian;
    Matrix<double> unscaling_layer_Jacobian;
    Matrix<double> outputs_trending_layer_Jacobian;
    Matrix<double> multilayer_perceptron_Jacobian;
    Matrix<double> bounding_layer_Jacobian;
    Matrix<double> probabilistic_layer_Jacobian;

    // Inputs trending layer

    if(inputs_trending_layer_pointer)
    {
        const Vector<double> derivatives = inputs_trending_layer_pointer->calculate_derivatives();

        inputs_trending_layer_Jacobian = inputs_trending_layer_pointer->calculate_Jacobian(derivatives);

        outputs = inputs_trending_layer_pointer->calculate_outputs(inputs, time);
    }

    // Scaling layer

    if(scaling_layer_pointer)
    {
        const Vector<double> scaling_layer_derivative = scaling_layer_pointer->calculate_derivatives(outputs);

        scaling_layer_Jacobian = scaling_layer_pointer->calculate_Jacobian(scaling_layer_derivative);

        outputs = scaling_layer_pointer->calculate_outputs(outputs);
    }

    // Principal components layer

    if(principal_components_layer_pointer)
    {
        principal_components_layer_Jacobian = principal_components_layer_pointer->calculate_Jacobian(outputs);

        outputs = principal_components_layer_pointer->calculate_outputs(outputs);
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        //multilayer_perceptron_Jacobian = multilayer_perceptron_pointer->calculate_Jacobian(outputs);

        //outputs = multilayer_perceptron_pointer->calculate_outputs(outputs);
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        const Vector<double> unscaling_layer_derivative = unscaling_layer_pointer->calculate_derivatives(outputs);

        unscaling_layer_Jacobian = unscaling_layer_pointer->calculate_Jacobian(unscaling_layer_derivative);

        outputs = unscaling_layer_pointer->calculate_outputs(outputs);
    }

    // Outputs trending layer

    if(outputs_trending_layer_pointer)
    {
        const Vector<double> derivatives = outputs_trending_layer_pointer->calculate_derivatives();

        outputs_trending_layer_Jacobian = outputs_trending_layer_pointer->calculate_Jacobian(derivatives);

        outputs = outputs_trending_layer_pointer->calculate_outputs(outputs, time);
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        probabilistic_layer_Jacobian = probabilistic_layer_pointer->calculate_Jacobian(outputs);

        outputs = probabilistic_layer_pointer->calculate_outputs(outputs);
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        const Vector<double>& derivatives = bounding_layer_pointer->calculate_derivatives(outputs);

        bounding_layer_Jacobian = bounding_layer_pointer->calculate_Jacobian(derivatives);

        outputs = bounding_layer_pointer->calculate_outputs(outputs);
    }

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    Matrix<double> Jacobian(outputs_number, outputs_number, 0.0);
    Jacobian.set_diagonal(1.0);

    // Bounding layer

    if(bounding_layer_pointer)
    {
        Jacobian = Jacobian.dot(bounding_layer_Jacobian);
    }

    // Probabilistic outputs

    if(probabilistic_layer_pointer)
    {
        Jacobian = Jacobian.dot(probabilistic_layer_Jacobian);
    }

    // Outputs trending layer

    if(outputs_trending_layer_pointer)
    {
        Jacobian = Jacobian.dot(outputs_trending_layer_Jacobian);
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        Jacobian = Jacobian.dot(unscaling_layer_Jacobian);
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        Jacobian = Jacobian.dot(multilayer_perceptron_Jacobian);
    }

    // Principal components layer

    if(principal_components_layer_pointer)
    {
        Jacobian = Jacobian.dot(principal_components_layer_Jacobian);
    }

    // Scaling layer

    if(scaling_layer_pointer)
    {
        Jacobian = Jacobian.dot(scaling_layer_Jacobian);
    }

    // Inputs trending layer

    if(inputs_trending_layer_pointer)
    {
        Jacobian = Jacobian.dot(inputs_trending_layer_Jacobian);
    }

    return(Jacobian);
*/
    return Matrix<double>();
}


/// Calculates a set of Jacobians from the neural network in response to a set of inputs.
/// The format is a vector of matrices, where each element is the Jacobian matrix for a single input.
/// @param inputs Matrix of inputs to the neural network.

Vector< Matrix<double> > NeuralNetwork::calculate_Jacobian_data(const Matrix<double>& inputs) const
{
/*
    const size_t inputs_number = inputs_pointer->get_inputs_number();

    const size_t inputs_size = inputs.get_rows_number();

    Vector< Matrix<double> > Jacobian_data(inputs_size);

    Vector<double> input_values(inputs_number);

    for(size_t i = 0; i < inputs_size; i++)
    {
        input_values = inputs.get_row(i);

        Jacobian_data[i] = calculate_Jacobian(input_values);
    }

    return(Jacobian_data);
*/

    return Vector<Matrix<double>>();
}


/// Calculates the histogram of the outputs with random inputs.
/// @param points_number Number of random instances to evaluate the neural network.
/// @param bins_number Number of bins for the histograms.

Vector< Histogram<double> > NeuralNetwork::calculate_outputs_histograms(const size_t& points_number, const size_t& bins_number) const
{
    const size_t inputs_number = inputs_pointer->get_inputs_number();

    Matrix<double> inputs(points_number, inputs_number);

    if(scaling_layer_pointer == nullptr)
    {
        inputs.randomize_uniform();
    }
    else
    {
        const Vector<ScalingLayer::ScalingMethod> scaling_methods = scaling_layer_pointer->get_scaling_methods();

        for(size_t i = 0; i < scaling_methods.size(); i++)
        {
            Vector<double> input_column(points_number, 0.0);

            if(scaling_methods[i] == ScalingLayer::NoScaling)
            {
                input_column.randomize_uniform();
            }
            else if(scaling_methods[i] == ScalingLayer::MinimumMaximum)
            {
                double minimum = scaling_layer_pointer->get_statistics(i).minimum;
                double maximum = scaling_layer_pointer->get_statistics(i).maximum;

                input_column.randomize_uniform(minimum, maximum);
            }
            else if(scaling_methods[i] == ScalingLayer::MeanStandardDeviation)
            {
                double mean = scaling_layer_pointer->get_statistics(i).mean;
                double standard_deviation = scaling_layer_pointer->get_statistics(i).standard_deviation;

                input_column.randomize_normal(mean, standard_deviation);
            }
            else if(scaling_methods[i] == ScalingLayer::StandardDeviation)
            {
                double mean = scaling_layer_pointer->get_statistics(i).mean;
                double standard_deviation = scaling_layer_pointer->get_statistics(i).standard_deviation;

                input_column.randomize_normal(mean, standard_deviation);
            }

            inputs.set_column(i, input_column, "");
        }
    }

    const Matrix<double> outputs = calculate_outputs(inputs);

    return(outputs.calculate_histograms(bins_number));
}


/// Calculates the histogram of the outputs with a matrix of given inputs.
/// @param inputs Matrix of the data to evaluate the neural network.
/// @param bins_number Number of bins for the histograms.

Vector< Histogram<double> > NeuralNetwork::calculate_outputs_histograms(const Matrix<double>& inputs, const size_t& bins_number) const
{
    const Matrix<double> outputs = calculate_outputs(inputs);

    return(outputs.calculate_histograms(bins_number));
}


//vector<double> NeuralNetwork::calculate_outputs_std(const vector<double>& inputs) const
//{
//    return calculate_outputs(Vector<double>(inputs)).to_std_vector();
//}


/// Returns a string representation of the current neural network object.

string NeuralNetwork::object_to_string() const
{
    ostringstream buffer;

    buffer << "NeuralNetwork\n";

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        buffer << multilayer_perceptron_pointer->object_to_string();
    }

    // Inputs trending layer

    if(inputs_trending_layer_pointer)
    {
        buffer << inputs_trending_layer_pointer->object_to_string();
    }

    // Scaling layer

    if(scaling_layer_pointer)
    {
        buffer << scaling_layer_pointer->object_to_string();
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        buffer << unscaling_layer_pointer->object_to_string();
    }

    // Outputs trending layer

    if(outputs_trending_layer_pointer)
    {
        buffer << outputs_trending_layer_pointer->object_to_string();
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        buffer << bounding_layer_pointer->object_to_string();
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        buffer << probabilistic_layer_pointer->object_to_string();
    }

    // Inputs

    if(inputs_pointer)
    {
        buffer << inputs_pointer->object_to_string();
    }

    // Outputs

    if(outputs_pointer)
    {
        buffer << outputs_pointer->object_to_string();
    }

    buffer << "Display: " <<  display << "\n";

    return(buffer.str());
}


/// Serializes the neural network object into a PMML document of the TinyXML library.

tinyxml2::XMLDocument* NeuralNetwork::to_PMML() const
{
    tinyxml2::XMLDocument* pmml_document = new tinyxml2::XMLDocument;

    tinyxml2::XMLDeclaration* pmml_declaration = pmml_document->NewDeclaration("xml version=\"1.0\" encoding=\"UTF-8\"");
    pmml_document->LinkEndChild(pmml_declaration);

    tinyxml2::XMLElement* root_element = pmml_document->NewElement("PMML");
    pmml_document->LinkEndChild(root_element);

    root_element->SetAttribute("version", 4.2);
    root_element->SetAttribute("xmlns", "http://www.dmg.org/PMML-4_2");
    root_element->SetAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance");


    // Header markup

    tinyxml2::XMLElement* header = pmml_document->NewElement("Header");
    root_element->LinkEndChild(header);

    header->SetAttribute("copyright", "Artelnics");

    tinyxml2::XMLElement* application = pmml_document->NewElement("Application");
    header->InsertFirstChild(application);

    application->SetAttribute("name", "Neural Designer");


    // Check null pointers

    if(!inputs_pointer || !outputs_pointer)
    {
        return pmml_document;
    }


    const size_t inputs_number = inputs_pointer->get_inputs_number();
    const size_t outputs_number = outputs_pointer->get_outputs_number();

    const bool is_probabilistic = has_probabilistic_layer();

    const bool is_data_scaled = has_scaling_layer() &&(scaling_layer_pointer->get_scaling_methods() != ScalingLayer::NoScaling);
    const bool is_data_unscaled = has_unscaling_layer() &&(unscaling_layer_pointer->get_unscaling_method() != UnscalingLayer::NoUnscaling);

    // Data dictionary markup

    tinyxml2::XMLElement* data_dictionary = pmml_document->NewElement("DataDictionary");
    root_element->LinkEndChild(data_dictionary);

    size_t number_of_fields;

    if(is_probabilistic)
    {
        number_of_fields = inputs_number + 1;
    }
    else
    {
        number_of_fields = inputs_number + outputs_number;
    }

    data_dictionary->SetAttribute("numberOfFields",static_cast<unsigned>(number_of_fields));

    if(is_data_scaled)
    {
        Vector< Statistics<double> > inputs_statistics = get_scaling_layer_pointer()->get_statistics();
        inputs_pointer->to_PMML(data_dictionary, is_data_scaled, inputs_statistics);
    }
    else
    {
        inputs_pointer->to_PMML(data_dictionary);
    }

    if(is_data_unscaled)
    {
        Vector< Statistics<double> > outputs_statistics = get_unscaling_layer_pointer()->get_statistics();
        outputs_pointer->to_PMML(data_dictionary,is_probabilistic, is_data_unscaled , outputs_statistics);
    }
    else
    {
        outputs_pointer->to_PMML(data_dictionary,is_probabilistic);
    }

    // Check null pointer

    if(!multilayer_perceptron_pointer)
    {
        return pmml_document;
    }

    // Transformation dictionary

    tinyxml2::XMLElement* transformation_dictionary = pmml_document->NewElement("TransformationDictionary");
    root_element->LinkEndChild(transformation_dictionary);

    if(is_data_scaled)
    {
        const Vector<string> inputs_names = inputs_pointer->get_names();

        scaling_layer_pointer->to_PMML(transformation_dictionary, inputs_names);
    }

    // Neural network markup

    tinyxml2::XMLElement* neural_network = pmml_document->NewElement("NeuralNetwork");
    root_element->LinkEndChild(neural_network);

    if(is_probabilistic)
    {
        neural_network->SetAttribute("functionName", "classification");
    }
    else
    {
        neural_network->SetAttribute("functionName", "regression");
    }

    const size_t number_of_layers = multilayer_perceptron_pointer->get_layers_perceptrons_numbers().size();
    neural_network->SetAttribute("numberOfLayers",static_cast<unsigned>(number_of_layers));


    // Neural network - mining schema markup

    tinyxml2::XMLElement* mining_schema = pmml_document->NewElement("MiningSchema");
    neural_network->LinkEndChild(mining_schema);

    inputs_pointer->to_PMML(mining_schema);

    outputs_pointer->to_PMML(mining_schema,is_probabilistic);

    // Neural network - neural inputs markup

    tinyxml2::XMLElement* neural_inputs = pmml_document->NewElement("NeuralInputs");
    neural_network->LinkEndChild(neural_inputs);

    neural_inputs->SetAttribute("numberOfInputs",static_cast<unsigned>(inputs_number));


    inputs_pointer->to_PMML(neural_inputs,is_data_scaled);

    // Neural network - neural layers markups

    multilayer_perceptron_pointer->to_PMML(neural_network);

    // Neural network - neural outputs markup

    tinyxml2::XMLElement* neural_outputs = pmml_document->NewElement("NeuralOutputs");
    neural_network->LinkEndChild(neural_outputs);

    neural_outputs->SetAttribute("numberOfOutputs",static_cast<unsigned>(outputs_number));

    outputs_pointer->to_PMML(neural_outputs,is_probabilistic,is_data_unscaled);


    if(is_probabilistic)
    {
        if(probabilistic_layer_pointer->get_probabilistic_method() == ProbabilisticLayer::Softmax)
        {
            tinyxml2::XMLElement* probabilistic_layer = neural_network->LastChildElement("NeuralLayer");

            // PMML defines softmax normalization method
            probabilistic_layer->SetAttribute("normalizationMethod","softmax");
        }
        else
        {
            // Classification network with only one output(binary output)
            if(outputs_number == 1)
            {
                const string output_display_name(outputs_pointer->get_name(0));
                const string output_name(output_display_name + "*");

                tinyxml2::XMLElement* derived_field = pmml_document->NewElement("DerivedField");
                transformation_dictionary->LinkEndChild(derived_field);

                derived_field->SetAttribute("displayName",output_display_name.c_str());
                derived_field->SetAttribute("name",output_name.c_str());
                derived_field->SetAttribute("dataType","double");
                derived_field->SetAttribute("optype","continuous");


                tinyxml2::XMLElement* norm_continuous = pmml_document->NewElement("NormContinuous");
                derived_field->LinkEndChild(norm_continuous);

                norm_continuous->SetAttribute("field",output_display_name.c_str());


                tinyxml2::XMLElement* linear_norm_begin = pmml_document->NewElement("LinearNorm");
                norm_continuous->LinkEndChild(linear_norm_begin);

                linear_norm_begin->SetAttribute("norm", "0.0");
                linear_norm_begin->SetAttribute("orig", "0.0");


                tinyxml2::XMLElement* linear_norm_end = pmml_document->NewElement("LinearNorm");
                norm_continuous->LinkEndChild(linear_norm_end);

                linear_norm_end->SetAttribute("norm", "1.0");
                linear_norm_end->SetAttribute("orig", "1.0");
            }
        }
        //probabilistic_layer_pointer->to_PMML(neural_network);
    }
    else
    {
        if(is_data_unscaled)
        {
            const Vector<string> outputs_names = outputs_pointer->get_names();

            unscaling_layer_pointer->to_PMML(transformation_dictionary,outputs_names);
        }
    }
    // End neural network markup


    return pmml_document;
}


/// Serializes the neural network object into a PMML file without memory load.
/// @todo
///
void NeuralNetwork::write_PMML(const string& file_name) const
{
/*
    ostringstream buffer;

    // Required for XMLPrinter constructor

    ofstream file(file_name.c_str());

    if(pmml_file == nullptr)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void write_PMML(const string&) method.\n"
               << "File " << file_name << " is nullptr.\n";

        throw logic_error(buffer.str());
    }

    tinyxml2::XMLPrinter file_stream(pmml_file);

    file_stream.PushDeclaration("xml version=\"1.0\" encoding=\"UTF-8\"");

    file_stream.OpenElement("PMML");

    file_stream.PushAttribute("version", 4.2);
    file_stream.PushAttribute("xmlns", "http://www.dmg.org/PMML-4_2");
    file_stream.PushAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance");


    file_stream.OpenElement("Header");

    file_stream.PushAttribute("copyright", "Artelnics");


    file_stream.OpenElement("Application");

    file_stream.PushAttribute("name", "Neural Designer");

    // Close Application
    file_stream.CloseElement();

    // Close Header
    file_stream.CloseElement();

    // Check null pointers

    if(!inputs_pointer || !outputs_pointer)
    {
        file_stream.CloseElement();
        fclose(pmml_file);
        return;
    }

    const size_t inputs_number = inputs_pointer->get_inputs_number();
    const size_t outputs_number = outputs_pointer->get_outputs_number();

    const bool is_probabilistic = has_probabilistic_layer();

    const bool is_data_scaled = has_scaling_layer() &&(scaling_layer_pointer->get_scaling_methods() != ScalingLayer::NoScaling);
    const bool is_data_unscaled = has_unscaling_layer() &&(unscaling_layer_pointer->get_unscaling_method() != UnscalingLayer::NoUnscaling);

    // Data dictionary

    file_stream.OpenElement("DataDictionary");

    size_t number_of_fields;

    if(is_probabilistic)
    {
        number_of_fields = inputs_number + 1;
    }
    else
    {
        number_of_fields = inputs_number + outputs_number;
    }

    // DataDictionary attribute
    file_stream.PushAttribute("numberOfFields",static_cast<unsigned>(number_of_fields));

    if(has_scaling_layer())
    {
        Vector< Statistics<double> > inputs_statistics = get_scaling_layer_pointer()->get_statistics();

        inputs_pointer->write_PMML_data_dictionary(file_stream, inputs_statistics);
    }
    else
    {
        inputs_pointer->write_PMML_data_dictionary(file_stream);
    }

    if(has_unscaling_layer())
    {
        Vector< Statistics<double> > outputs_statistics = get_unscaling_layer_pointer()->get_statistics();

        outputs_pointer->write_PMML_data_dictionary(file_stream,is_probabilistic, outputs_statistics);
    }
    else
    {
        outputs_pointer->write_PMML_data_dictionary(file_stream,is_probabilistic);
    }

    // Close DataDictionary
    file_stream.CloseElement();

    // Check null pointer

    if(!multilayer_perceptron_pointer)
    {
        file_stream.CloseElement();
        fclose(pmml_file);
        return;
    }

    // Transformation dictionary

    file_stream.OpenElement("TransformationDictionary");

    if(is_data_scaled)
    {
        const Vector<string> inputs_names = inputs_pointer->get_names();

        scaling_layer_pointer->write_PMML(file_stream, inputs_names);
    }

    if(is_data_unscaled)
    {
        const Vector<string> outputs_names = outputs_pointer->get_names();

        unscaling_layer_pointer->write_PMML(file_stream, outputs_names);
    }

    // Define no normalization for probabilistic binary output
    // but SPSS requires next fields
    if(is_probabilistic &&(outputs_number == 1))
    {
        const string output_display_name(outputs_pointer->get_name(0));
        const string output_name(output_display_name + "*");

        file_stream.OpenElement("DerivedField");

        file_stream.PushAttribute("displayName",output_display_name.c_str());
        file_stream.PushAttribute("name",output_name.c_str());
        file_stream.PushAttribute("dataType","double");
        file_stream.PushAttribute("optype","continuous");


        file_stream.OpenElement("NormContinuous");

        file_stream.PushAttribute("field",output_display_name.c_str());

        // Normalization range begin

        file_stream.OpenElement("LinearNorm");

        file_stream.PushAttribute("orig", "0.0");
        file_stream.PushAttribute("norm", "0.0");

        file_stream.CloseElement();

        // Normalization range end

        file_stream.OpenElement("LinearNorm");

        file_stream.PushAttribute("orig", "1.0");
        file_stream.PushAttribute("norm", "1.0");

        file_stream.CloseElement();

        // Close NormContinuous
        file_stream.CloseElement();

        // Close DerivedField
        file_stream.CloseElement();
    }

    // Close TransformationDictionary
    file_stream.CloseElement();


    // Neural network markup

    file_stream.OpenElement("NeuralNetwork");

    if(is_probabilistic)
    {
        file_stream.PushAttribute("functionName", "classification");
    }
    else
    {
        file_stream.PushAttribute("functionName", "regression");
    }

    const size_t number_of_layers = multilayer_perceptron_pointer->get_layers_perceptrons_numbers().size();

    file_stream.PushAttribute("numberOfLayers",static_cast<unsigned>(number_of_layers));

    PerceptronLayer::ActivationFunction neural_network_activation_function = multilayer_perceptron_pointer->get_layers_activation_function().at(0);

    switch(neural_network_activation_function)
    {
    case PerceptronLayer::Threshold:
        file_stream.PushAttribute("activationFunction","threshold");
        break;

    case PerceptronLayer::SymmetricThreshold:
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void write_PMML(string) const method.\n"
               << "Symmetric threshold activaton function is not supported by PMML.\n";

        throw logic_error(buffer.str());
    }

    case PerceptronLayer::Logistic:
        file_stream.PushAttribute("activationFunction","logistic");
        break;

    case PerceptronLayer::HyperbolicTangent:
        file_stream.PushAttribute("activationFunction","tanh");
        break;

    case PerceptronLayer::Linear:
        file_stream.PushAttribute("activationFunction","identity");
        break;
//    default:
//        break;
    }

    // Neural network - mining schema markup

    file_stream.OpenElement("MiningSchema");

    // Mining schema inputs

    for(size_t i = 0; i < inputs_number; i++)
    {
        file_stream.OpenElement("MiningField");

        file_stream.PushAttribute("name", inputs_pointer->get_name(i).c_str());

        file_stream.CloseElement();
    }

    // Mining schema outputs

    outputs_pointer->write_PMML_mining_schema(file_stream,is_probabilistic);

    // Close MiningSchema

    file_stream.CloseElement();

    // Neural network - neural inputs markup

    file_stream.OpenElement("NeuralInputs");

    file_stream.PushAttribute("numberOfInputs",static_cast<unsigned>(inputs_number));

    inputs_pointer->write_PMML_neural_inputs(file_stream,is_data_scaled);

    // Close NeuralInputs

    file_stream.CloseElement();

    // Neural network - neural layers markups

    const bool is_softmax_normalization_method = (is_probabilistic &&(probabilistic_layer_pointer->get_probabilistic_method() == ProbabilisticLayer::Softmax));

    multilayer_perceptron_pointer->write_PMML(file_stream, is_softmax_normalization_method);

    // Neural network - neural outputs markup

    file_stream.OpenElement("NeuralOutputs");

    file_stream.PushAttribute("numberOfOutputs",static_cast<unsigned>(outputs_number));

    outputs_pointer->write_PMML_neural_outputs(file_stream, number_of_layers ,is_probabilistic, is_data_unscaled);

    // Close NeuralOutputs
    file_stream.CloseElement();


    // Close NeuralNetwork
    file_stream.CloseElement();


    // Close PMML
    file_stream.CloseElement();

    fclose(pmml_file);
*/
}


/// Deserializes a TinyXML document into this neural network object.

void NeuralNetwork::from_PMML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    // Root element(PMML)

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("PMML");

    if(!root_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "PMML element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Neural network element

    const tinyxml2::XMLElement* neural_network = root_element->FirstChildElement("NeuralNetwork");

    if(!neural_network)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuralNetwork element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    const tinyxml2::XMLAttribute* attribute_function_name = neural_network->FindAttribute("functionName");

    if(!attribute_function_name)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "Attibute \"functionName\" in NeuralNetwork element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    bool is_probabilistic;

    if(string(attribute_function_name->Value()) == "classification")
    {
        is_probabilistic = true;
    }
    else
    {
        if(string(attribute_function_name->Value()) == "regression")
        {
            is_probabilistic = false;
        }
        else
        {
            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                   << "Function: " << string(attribute_function_name->Value()) << " for NeuralNetwork is not supported.\n";

            throw logic_error(buffer.str());
        }
    }

    // Neural network - mining schema element

    const tinyxml2::XMLElement* mining_schema = neural_network->FirstChildElement("MiningSchema");

    if(!mining_schema)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "MiningSchema element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Mining schema elements

    const tinyxml2::XMLElement* mining_field = mining_schema->FirstChildElement("MiningField");

    Vector<string> input_mining_fields(0);
    Vector<string> output_mining_fields(0);

    while(mining_field)
    {
        const tinyxml2::XMLAttribute * mining_field_name =  mining_field->FindAttribute("name");
        const tinyxml2::XMLAttribute * mining_field_usage = mining_field->FindAttribute("usageType");

        if(mining_field_name)
        {
            if(!mining_field_usage || string(mining_field_usage->Value()) == "active")
            {
                input_mining_fields.push_back(string(mining_field_name->Value()));
            }
            else
            {
                if(string(mining_field_usage->Value()) == "predicted" || string(mining_field_usage->Value()) == "target")
                {
                    output_mining_fields.push_back(string(mining_field_name->Value()));
                }
            }
        }
        else
        {
            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                   << "Attribute \"name\" in MiningField element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        mining_field = mining_field->NextSiblingElement("MiningField");
    }

    const size_t inputs_number = input_mining_fields.size();

    size_t outputs_number = output_mining_fields.size();

    if(inputs_number == 0)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "Number of inputs in MiningField element is 0.\n";

        throw logic_error(buffer.str());
    }

    if(outputs_number == 0)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "Number of outputs in MiningField element is 0.\n";

        throw logic_error(buffer.str());
    }

    // Data dictionary

    const tinyxml2::XMLElement* data_dictionary = root_element->FirstChildElement("DataDictionary");

    if(!data_dictionary)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "DataDictionary element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    Vector< Statistics<double> > inputs_statistics;
    Vector< Statistics<double> > outputs_statistics;

    Vector<string> output_classification_fields;

    const tinyxml2::XMLElement* data_field = data_dictionary->FirstChildElement("DataField");

    while(data_field)
    {
        const tinyxml2::XMLAttribute* attribute_name_data_field = data_field->FindAttribute("name");

        if(!attribute_name_data_field)
        {
            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                   << "Attribute \"name\" in DataField in DataDictionary element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        string field_name(attribute_name_data_field->Value());

        const tinyxml2::XMLAttribute* attribute_optype_data_field = data_field->FindAttribute("optype");

        if(!attribute_optype_data_field)
        {
            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                   << "Attribute \"optype\" in DataField in DataDictionary element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        string data_field_optype(attribute_optype_data_field->Value());

        // Search classification field values

        if(is_probabilistic && data_field_optype == "categorical")
        {
            if(output_mining_fields.contains(field_name))
            {
                const tinyxml2::XMLElement* data_field_value = data_field->FirstChildElement("Value");

                if(!data_field_value)
                {
                    buffer << "OpenNN Exception: NeuralNetwork class.\n"
                           << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                           << "Value in DataField in DataDictionary element is nullptr.\n";

                    throw logic_error(buffer.str());
                }

                while(data_field_value)
                {
                    const tinyxml2::XMLAttribute* attribute_value_data_field_value = data_field_value->FindAttribute("value");

                    if(!attribute_value_data_field_value)
                    {
                        buffer << "OpenNN Exception: NeuralNetwork class.\n"
                               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                               << "Attribute \"value\" in Value in DataDictionary element is nullptr.\n";

                        throw logic_error(buffer.str());
                    }

                    output_classification_fields.push_back(string(attribute_value_data_field_value->Value()));

                    data_field_value = data_field_value->NextSiblingElement("Value");
                }
            }
        }

        // Search data statistics(maximum and minimum)

        if(data_field_optype == "continuous")
        {
            const tinyxml2::XMLElement* data_field_interval = data_field->FirstChildElement("Interval");

            if(data_field_interval)
            {
                const tinyxml2::XMLAttribute* attribute_left_margin_interval = data_field_interval->FindAttribute("leftMargin");
                const tinyxml2::XMLAttribute* attribute_right_margin_interval = data_field_interval->FindAttribute("rightMargin");

                if(!attribute_left_margin_interval)
                {
                    buffer << "OpenNN Exception: NeuralNetwork class.\n"
                           << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                           << "Attribute \"leftMargin\" in Interval in DataField element is nullptr.\n";

                    throw logic_error(buffer.str());
                }

                if(!attribute_right_margin_interval)
                {
                    buffer << "OpenNN Exception: NeuralNetwork class.\n"
                           << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                           << "Attribute \"rightMargin\" in Interval in DataField element is nullptr.\n";

                    throw logic_error(buffer.str());
                }

                const string left_margin_string(attribute_left_margin_interval->Value());
                const string right_margin_string(attribute_right_margin_interval->Value());

                if(left_margin_string == "")
                {
                    buffer << "OpenNN Exception: NeuralNetwork class.\n"
                           << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                           << "Attribute \"leftMargin\" in Interval in DataField element is empty.\n";

                    throw logic_error(buffer.str());
                }

                if(right_margin_string == "")
                {
                    buffer << "OpenNN Exception: NeuralNetwork class.\n"
                           << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                           << "Attribute \"rightMargin\" in Interval in DataField element is empty.\n";

                    throw logic_error(buffer.str());
                }

                const double left_margin = atof(left_margin_string.c_str());
                const double right_margin = atof(right_margin_string.c_str());

                if(right_margin < left_margin)
                {
                    buffer << "OpenNN Exception: NeuralNetwork class.\n"
                           << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                           << "Right margin in Interval is less than left margin.\n";

                    throw logic_error(buffer.str());
                }

                Statistics<double> new_data_field_statistics;

                new_data_field_statistics.minimum = left_margin;
                new_data_field_statistics.maximum = right_margin;

                if(input_mining_fields.contains(field_name))
                {
                    inputs_statistics.push_back(new_data_field_statistics);
                }
                else
                {
                    if(output_mining_fields.contains(field_name))
                    {
                        outputs_statistics.push_back(new_data_field_statistics);
                    }
                }
            }
        }

        data_field = data_field->NextSiblingElement("DataField");
    }

    if(output_classification_fields.size() != 0)
    {
        outputs_number = output_classification_fields.size();
    }

    // Neural network - neural inputs

    //    const tinyxml2::XMLElement* neural_inputs = neural_network->FirstChildElement("NeuralInputs");

    //    if(!neural_inputs)
    //    {
    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //               << "NeuralInputs element is nullptr.\n";

    //        throw logic_error(buffer.str());
    //    }


    //    const tinyxml2::XMLElement* neural_input = neural_inputs->FirstChildElement("NeuralInput");
    //    Vector<string> inputs_IDs(inputs_number);

    // Neural input
    //    if(!neural_input)
    //    {
    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //               << "NeuralInput element is empty.\n";

    //        throw logic_error(buffer.str());
    //    }

    //    while(neural_input)
    //    {
    //        const tinyxml2::XMLAttribute * attribute_id_neural_input =  neural_input->FindAttribute("id");

    //        if(!attribute_id_neural_input)
    //        {
    //            buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //                   << "Attribute \"id\" in NeuralInput element is nullptr.\n";

    //            throw logic_error(buffer.str());
    //        }

    //        const string id_value(attribute_id_neural_input->Value());
    //        string id_string;

    //        if(id_value.find(',') != string::npos)
    //        {
    //            stringstream to_split(id_value);
    //            string id_current_number;

    //            // save the last number, which is the ID
    //            while(getline(to_split,id_current_number,','))
    //                id_string = id_current_number;
    //        }

    //        if(id_string == "")
    //        {
    //            buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //                   << "Attribute \"id\" in NeuralInput element is empty.\n";

    //            throw logic_error(buffer.str());
    //        }

    //        int input_id = atoi(id_string);

    //        if(input_id < 0 || input_id >= inputs_number)
    //        {
    //            buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //                   << "Attribute \"id\" in NeuralInput element is invalid.\n";

    //            throw logic_error(buffer.str());
    //        }

    //        const tinyxml2::XMLElement* neural_input_derived_field = neural_input->FirstChildElement("DerivedField");

    //        if(!neural_input_derived_field)
    //        {
    //            buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //                   << "DerivedField in NeuralInput element is nullptr.\n";

    //            throw logic_error(buffer.str());
    //        }

    //        const tinyxml2::XMLElement* derived_field_field_ref = neural_input_derived_field->FirstChildElement("FieldRef");

    //        if(!derived_field_field_ref)
    //        {
    //            buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //                   << "FieldRef in DerivedField in NeuralInput element is nullptr.\n";

    //            throw logic_error(buffer.str());
    //        }

    //        const tinyxml2::XMLAttribute* attribute_field_field_ref = derived_field_field_ref->FindAttribute("field");
    //        if(!attribute_field_field_ref)
    //        {
    //            buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //                   << "Attribute \"field\" in FieldRef in DerivedField in NeuralInput element is nullptr.\n";

    //            throw logic_error(buffer.str());
    //        }

    //        string field_name = string(attribute_field_field_ref->Value());

    //        if(field_name == "")
    //        {
    //            buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
    //                   << "Attribute \"field\" in FieldRef in DerivedField in NeuralInput element is empty.\n";

    //            throw logic_error(buffer.str());
    //        }

    //        inputs_IDs.at(input_id) = field_name;


    //        neural_input = neural_input->NextSiblingElement("NeuralInput");
    //    }

    // Neural layers architecture

    const tinyxml2::XMLElement* neural_layer = neural_network->FirstChildElement("NeuralLayer");

    if(!neural_layer)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
               << "NeuralLayer element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Get the neural network architecture from PMML document

    /*
     * neural_layers_architecture.size() = layers number
     * neural_layers_architecture[i] = number of neurons of layer i
     */
    Vector<size_t> neural_layers_architecture(0);

    while(neural_layer)
    {
        const tinyxml2::XMLElement* neuron = neural_layer->FirstChildElement("Neuron");

        if(!neuron)
        {
            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                   << "Neuron element is nullptr.\n";

            throw logic_error(buffer.str());
        }

        size_t neurons_number = 0;

        while(neuron)
        {
            neurons_number++;

            neuron = neuron->NextSiblingElement("Neuron");
        }

        neural_layers_architecture.push_back(neurons_number);

        neural_layer = neural_layer->NextSiblingElement("NeuralLayer");
    }

    // Set network architecture

    const size_t number_of_layers = neural_layers_architecture.size();

    Vector<size_t> architecture(1 + number_of_layers);

    architecture[0] = inputs_number;

    for( size_t i = 0; i < number_of_layers; i++ )
    {
        architecture[i+1] = neural_layers_architecture[i];
    }

    set(architecture);

    // Set inputs and outputs names

    inputs_pointer->set_names(input_mining_fields);

    if(is_probabilistic && outputs_number > 1)
    {
        outputs_pointer->set_names(output_classification_fields);
    }
    else
    {
        outputs_pointer->set_names(output_mining_fields);
    }

    // Set network parameters

    multilayer_perceptron_pointer->from_PMML(neural_network);

    // Set scaling layer

    construct_scaling_layer();

    if(inputs_statistics.size() == inputs_number)
    {
        scaling_layer_pointer->set_statistics(inputs_statistics);
    }

    const tinyxml2::XMLElement* transformation_dictionary = root_element->FirstChildElement("TransformationDictionary");
    const tinyxml2::XMLElement* local_transformations = neural_network->FirstChildElement("LocalTransformations");

    if(transformation_dictionary && !transformation_dictionary->NoChildren())
    {
        scaling_layer_pointer->from_PMML(transformation_dictionary,input_mining_fields);
    }
    else
    {
        if(local_transformations && !local_transformations->NoChildren())
        {
            scaling_layer_pointer->from_PMML(local_transformations,input_mining_fields);
        }
        else
        {
            scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);
        }
    }

    // Set probabilistic layer

    if(is_probabilistic)
    {
        construct_probabilistic_layer();

        const tinyxml2::XMLAttribute* attribute_method_probabilistic_layer = neural_network->LastChildElement("NeuralLayer")->FindAttribute("normalizationMethod");

        if(attribute_method_probabilistic_layer && outputs_number > 1)
        {
            string probabilistic_layer_normalization_method(attribute_method_probabilistic_layer->Value());

            if(probabilistic_layer_normalization_method == "softmax" )
            {
                probabilistic_layer_pointer->set_probabilistic_method(ProbabilisticLayer::Softmax);
            }
            else
            {
                buffer << "OpenNN Exception: NeuralNetwork class.\n"
                       << "void from_PMML(const tinyxml2::XMLDocument&) method.\n"
                       << "Probabilistic layer method " << probabilistic_layer_normalization_method <<" not supported.\n";

                throw logic_error(buffer.str());
            }
        }
        else
        {
            // Add binary and competitive probabilistic outputs
            probabilistic_layer_pointer->set_probabilistic_method(ProbabilisticLayer::NoProbabilistic);
        }
    }

    // Set unscaling layer

    else
    {
        construct_unscaling_layer();

        if(outputs_statistics.size() == outputs_number)
        {
            unscaling_layer_pointer->set_statistics(outputs_statistics);
        }

        if(transformation_dictionary && !transformation_dictionary->NoChildren())
        {
            unscaling_layer_pointer->from_PMML(transformation_dictionary,output_mining_fields);
        }
        else
        {
            if(local_transformations && !local_transformations->NoChildren())
            {
                unscaling_layer_pointer->from_PMML(local_transformations,output_mining_fields);
            }
            else
            {
                unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);
            }
        }
    }
}


/// Serializes the neural network object into a XML document of the TinyXML library.
/// See the OpenNN manual for more information about the format of this element. 

tinyxml2::XMLDocument* NeuralNetwork::to_XML() const
{
    tinyxml2::XMLDocument* document = new tinyxml2::XMLDocument;

    tinyxml2::XMLElement* neural_network_element = document->NewElement("NeuralNetwork");

    document->InsertFirstChild(neural_network_element);

    ostringstream buffer;

    // Inputs

    if(inputs_pointer)
    {        
        tinyxml2::XMLDocument* inputs_document = inputs_pointer->to_XML();

        const tinyxml2::XMLElement* inputs_element = inputs_document->FirstChildElement("Inputs");

        tinyxml2::XMLNode* node = inputs_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete inputs_document;
    }

    // Inputs trending layer

    if(inputs_trending_layer_pointer)
    {
        const tinyxml2::XMLDocument* inputs_trending_layer_document = inputs_trending_layer_pointer->to_XML();

        const tinyxml2::XMLElement* inputs_trending_layer_element = inputs_trending_layer_document->FirstChildElement("InputsTrendingLayer");

        tinyxml2::XMLNode* node = inputs_trending_layer_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete inputs_trending_layer_document;
    }

    // Scaling layer

    if(scaling_layer_pointer)
    {
        const tinyxml2::XMLDocument* scaling_layer_document = scaling_layer_pointer->to_XML();

        const tinyxml2::XMLElement* scaling_layer_element = scaling_layer_document->FirstChildElement("ScalingLayer");

        tinyxml2::XMLNode* node = scaling_layer_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete scaling_layer_document;
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        const tinyxml2::XMLDocument* multilayer_perceptron_document = multilayer_perceptron_pointer->to_XML();

        const tinyxml2::XMLElement* multilayer_perceptron_element = multilayer_perceptron_document->FirstChildElement("MultilayerPerceptron");

        tinyxml2::XMLNode* node = multilayer_perceptron_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete multilayer_perceptron_document;
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        const tinyxml2::XMLDocument* unscaling_layer_document = unscaling_layer_pointer->to_XML();

        const tinyxml2::XMLElement* unscaling_layer_element = unscaling_layer_document->FirstChildElement("UnscalingLayer");

        tinyxml2::XMLNode* node = unscaling_layer_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete unscaling_layer_document;
    }

    // Outputs trending layer

    if(outputs_trending_layer_pointer)
    {
        const tinyxml2::XMLDocument* outputs_trending_layer_document = outputs_trending_layer_pointer->to_XML();

        const tinyxml2::XMLElement* outputs_trending_layer_element = outputs_trending_layer_document->FirstChildElement("OutputsTrendingLayer");

        tinyxml2::XMLNode* node = outputs_trending_layer_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete outputs_trending_layer_document;
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        const tinyxml2::XMLDocument* probabilistic_layer_document = probabilistic_layer_pointer->to_XML();

        const tinyxml2::XMLElement* probabilistic_layer_element = probabilistic_layer_document->FirstChildElement("ProbabilisticLayer");

        tinyxml2::XMLNode* node = probabilistic_layer_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete probabilistic_layer_document;
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        const tinyxml2::XMLDocument* bounding_layer_document = bounding_layer_pointer->to_XML();

        const tinyxml2::XMLElement* bounding_layer_element = bounding_layer_document->FirstChildElement("BoundingLayer");

        tinyxml2::XMLNode* node = bounding_layer_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete bounding_layer_document;
    }

    // Outputs

    if(outputs_pointer)
    {
        const tinyxml2::XMLDocument* outputs_document = outputs_pointer->to_XML();

        const tinyxml2::XMLElement* outputs_element = outputs_document->FirstChildElement("Outputs");

        tinyxml2::XMLNode* node = outputs_element->DeepClone(document);

        neural_network_element->InsertEndChild(node);

        delete outputs_document;
    }

    //   // Display warnings
    //   {
    //      tinyxml2::XMLElement* display_element = document->NewElement("Display");
    //      neural_network_element->LinkEndChild(display_element);

    //      buffer.str("");
    //      buffer << display;

    //      tinyxml2::XMLText* display_text = document->NewText(buffer.str().c_str());
    //      display_element->LinkEndChild(display_text);
    //   }

    return(document);
}


/// Serializes the neural network object into a XML document of the TinyXML library without keep the DOM tree in memory.
/// See the OpenNN manual for more information about the format of this document.

void NeuralNetwork::write_XML(tinyxml2::XMLPrinter& file_stream) const
{
    file_stream.OpenElement("NeuralNetwork");

    // Inputs

    if(inputs_pointer)
    {
        inputs_pointer->write_XML(file_stream);
    }

    // Inputs trending layer

    if(inputs_trending_layer_pointer)
    {
        inputs_trending_layer_pointer->write_XML(file_stream);
    }

    // Scaling layer

    if(scaling_layer_pointer)
    {
        scaling_layer_pointer->write_XML(file_stream);
    }

    // Principal components layer

    if(principal_components_layer_pointer)
    {
        principal_components_layer_pointer->write_XML(file_stream);
    }

    // Multilayer perceptron

    if(multilayer_perceptron_pointer)
    {
        multilayer_perceptron_pointer->write_XML(file_stream);
    }

    // Unscaling layer

    if(unscaling_layer_pointer)
    {
        unscaling_layer_pointer->write_XML(file_stream);
    }

    // Outputs trending layer

    if(outputs_trending_layer_pointer)
    {
        outputs_trending_layer_pointer->write_XML(file_stream);
    }

    // Probabilistic layer

    if(probabilistic_layer_pointer)
    {
        probabilistic_layer_pointer->write_XML(file_stream);
    }

    // Bounding layer

    if(bounding_layer_pointer)
    {
        bounding_layer_pointer->write_XML(file_stream);
    }

    // Outputs

    if(outputs_pointer)
    {
        outputs_pointer->write_XML(file_stream);
    }

    file_stream.CloseElement();
}


/// Deserializes a TinyXML document into this neural network object.
/// @param document XML document containing the member data.

void NeuralNetwork::from_XML(const tinyxml2::XMLDocument& document)
{
    ostringstream buffer;

    const tinyxml2::XMLElement* root_element = document.FirstChildElement("NeuralNetwork");

    if(!root_element)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void from_XML(const tinyxml2::XMLDocument&) method.\n"
               << "Neural network element is nullptr.\n";

        throw logic_error(buffer.str());
    }

    // Inputs

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Inputs");

        if(element)
        {
            if(!inputs_pointer)
            {
                inputs_pointer = new Inputs();
            }

            tinyxml2::XMLDocument inputs_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&inputs_document);

            inputs_document.InsertFirstChild(element_clone);

            inputs_pointer->from_XML(inputs_document);
        }
    }

    // Inputs trending layer
    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("InputsTrendingLayer");

        if(element)
        {
            if(!inputs_trending_layer_pointer)
            {
                inputs_trending_layer_pointer = new InputsTrendingLayer();
            }

            tinyxml2::XMLDocument inputs_trending_layer_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&inputs_trending_layer_document);

            inputs_trending_layer_document.InsertFirstChild(element_clone);

            inputs_trending_layer_pointer->from_XML(inputs_trending_layer_document);
        }
    }

    // Scaling layer

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ScalingLayer");

        if(element)
        {
            if(!scaling_layer_pointer)
            {
                scaling_layer_pointer = new ScalingLayer();
            }

            tinyxml2::XMLDocument scaling_layer_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&scaling_layer_document);

            scaling_layer_document.InsertFirstChild(element_clone);

            scaling_layer_pointer->from_XML(scaling_layer_document);
        }
    }

    // Principal components layer

    {

        const tinyxml2::XMLElement* element = root_element->FirstChildElement("PrincipalComponentsLayer");

        if(element)
        {
            if(!principal_components_layer_pointer)
            {
                principal_components_layer_pointer = new PrincipalComponentsLayer();
            }

            tinyxml2::XMLDocument principal_components_layer_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&principal_components_layer_document);

            principal_components_layer_document.InsertFirstChild(element_clone);

            principal_components_layer_pointer->from_XML(principal_components_layer_document);
        }
    }


    // Multilayer perceptron

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("MultilayerPerceptron");


        if(element)
        {
            if(!multilayer_perceptron_pointer)
            {
                multilayer_perceptron_pointer = new MultilayerPerceptron();
            }

            tinyxml2::XMLDocument multilayer_perceptron_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&multilayer_perceptron_document);

            multilayer_perceptron_document.InsertFirstChild(element_clone);

            multilayer_perceptron_pointer->from_XML(multilayer_perceptron_document);
        }
    }

    // Unscaling layer

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("UnscalingLayer");

        if(element)
        {

            if(!unscaling_layer_pointer)
            {
                unscaling_layer_pointer = new UnscalingLayer();
            }

            tinyxml2::XMLDocument unscaling_layer_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&unscaling_layer_document);

            unscaling_layer_document.InsertFirstChild(element_clone);

            unscaling_layer_pointer->from_XML(unscaling_layer_document);
        }
    }


    // Outputs trending layer

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("OutputsTrendingLayer");

        if(element)
        {
            if(!outputs_trending_layer_pointer)
            {
                outputs_trending_layer_pointer = new OutputsTrendingLayer();
            }

            tinyxml2::XMLDocument outputs_trending_layer_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&outputs_trending_layer_document);

            outputs_trending_layer_document.InsertFirstChild(element_clone);

            outputs_trending_layer_pointer->from_XML(outputs_trending_layer_document);
        }
    }

    // Probabilistic layer

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("ProbabilisticLayer");

        if(element)
        {

            if(!probabilistic_layer_pointer)
            {
                probabilistic_layer_pointer = new ProbabilisticLayer();
            }

            tinyxml2::XMLDocument probabilistic_layer_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&probabilistic_layer_document);

            probabilistic_layer_document.InsertFirstChild(element_clone);

            probabilistic_layer_pointer->from_XML(probabilistic_layer_document);
        }
    }

    // Bounding layer

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("BoundingLayer");

        if(element)
        {
            if(!bounding_layer_pointer)
            {
                bounding_layer_pointer = new BoundingLayer();
            }

            tinyxml2::XMLDocument bounding_layer_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&bounding_layer_document);

            bounding_layer_document.InsertFirstChild(element_clone);

            bounding_layer_pointer->from_XML(bounding_layer_document);
        }
    }

    // Outputs

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Outputs");

        if(element)
        {

            if(!outputs_pointer)
            {
                outputs_pointer = new Outputs();
            }

            tinyxml2::XMLDocument outputs_document;
            tinyxml2::XMLNode* element_clone;

            element_clone = element->DeepClone(&outputs_document);

            outputs_document.InsertFirstChild(element_clone);

            outputs_pointer->from_XML(outputs_document);
        }
    }

    // Display

    {
        const tinyxml2::XMLElement* element = root_element->FirstChildElement("Display");

        if(element)
        {
            const string new_display_string = element->GetText();

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


/// Prints to the screen the members of a neural network object in a XML-type format.

void NeuralNetwork::print() const
{
    if(display)
    {
        cout << object_to_string();
    }
}


/// Saves to a XML file the members of a neural network object.
/// @param file_name Name of neural network XML file.

void NeuralNetwork::save(const string& file_name) const
{
    tinyxml2::XMLDocument* document = to_XML();

    document->SaveFile(file_name.c_str());

    delete document;
}


/// Saves to a data file the parameters of a neural network object.
/// @param file_name Name of parameters data file.

void NeuralNetwork::save_parameters(const string& file_name) const
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_parameters(const string&) const method.\n"
               << "Cannot open parameters data file.\n";

        throw logic_error(buffer.str());
    }

    const Vector<double> parameters = get_parameters();

    file << parameters << endl;

    // Close file

    file.close();
}


/// Loads from a XML file the members for this neural network object.
/// Please mind about the file format, which is specified in the User's Guide. 
/// @param file_name Name of neural network XML file.

void NeuralNetwork::load(const string& file_name)
{
    set_default();

    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void load(const string&) method.\n"
               << "Cannot load XML file " << file_name << ".\n";

        throw logic_error(buffer.str());
    }

    from_XML(document);
}


/// Loads the multilayer perceptron parameters from a data file.
/// The format of this file is just a sequence of numbers. 
/// @param file_name Name of parameters data file. 

void NeuralNetwork::load_parameters(const string& file_name)
{
    ifstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void load_parameters(const string&) method.\n"
               << "Cannot open parameters data file.\n";

        throw logic_error(buffer.str());
    }

    const size_t parameters_number = get_parameters_number();

    Vector<double> new_parameters(parameters_number);

    new_parameters.load(file_name);

    set_parameters(new_parameters);

    file.close();
}


/// Returns a string with the expression of the function represented by the neural network.

string NeuralNetwork::write_expression() const
{
    ostringstream buffer;

#ifdef __OPENNN_DEBUG__

    if(!inputs_pointer)
    {
        buffer.str("");

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string write_expression() const method.\n"
               << "Pointer to inputs is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(!multilayer_perceptron_pointer)
    {
        buffer.str("");

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string write_expression() const method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(!outputs_pointer)
    {
        buffer.str("");

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string write_expression() const method.\n"
               << "Pointer to outputs is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    Vector<string> inputs_name = inputs_pointer->get_names();
    Vector<string> outputs_name = outputs_pointer->get_names();

    size_t position = 0;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(inputs_name[i].begin(), inputs_name[i].end(), ' ');
        inputs_name[i].erase(end_pos, inputs_name[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ":";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }
    }

    for(size_t i = 0; i < outputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(outputs_name[i].begin(), outputs_name[i].end(), ' ');
        outputs_name[i].erase(end_pos, outputs_name[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ":";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }
    }

    // Scaled inputs

    Vector<string> scaled_inputs_name(/*inputs_number*/inputs_name.size());

    for(size_t i = 0; i < inputs_name.size()/*inputs_number*/; i++)
    {
        buffer.str("");

        buffer << "scaled_" << inputs_name[i];

        scaled_inputs_name[i] = buffer.str();
    }

    // Principal components

    Vector<string> principal_components_name(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer.str("");

        buffer << "principal_component_" <<(i+1);

        principal_components_name[i] = buffer.str();
    }

    // Scaled outputs

    Vector<string> scaled_outputs_name(/*outputs_number*/outputs_name.size());

    for(size_t i = 0; i < outputs_name.size()/*outputs_number*/; i++)
    {
        buffer.str("");

        buffer << "scaled_" << outputs_name[i];

        scaled_outputs_name[i] = buffer.str();
    }

    // Non probabilistic outputs

    Vector<string> non_probabilistic_outputs_name(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "non_probabilistic_" << outputs_name[i];

        non_probabilistic_outputs_name[i] = buffer.str();
    }

    buffer.str("");

    // Inputs trending layer

    if(has_inputs_trending_layer())
    {
        buffer << inputs_trending_layer_pointer->write_expression(inputs_name, inputs_name);
    }

    // Scaling layer

    if(has_scaling_layer())
    {
        buffer << scaling_layer_pointer->write_expression(inputs_name, scaled_inputs_name);
    }

    // Principal components layer

    if(has_principal_components_layer())
    {
        buffer << principal_components_layer_pointer->write_expression(scaled_inputs_name, principal_components_name);
    }

    // Multilayer perceptron

    if(has_multilayer_perceptron())
    {
        if(scaling_layer_pointer && unscaling_layer_pointer)
        {
            if(has_principal_components_layer() && principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
            {
                buffer << multilayer_perceptron_pointer->write_expression(principal_components_name, scaled_outputs_name);
            }
            else
            {
                buffer << multilayer_perceptron_pointer->write_expression(scaled_inputs_name, scaled_outputs_name);
            }
        }
        else if(scaling_layer_pointer && probabilistic_layer_pointer)
        {
            if(has_principal_components_layer() && principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
            {
                buffer << multilayer_perceptron_pointer->write_expression(principal_components_name, scaled_outputs_name);
            }
            else
            {
                buffer << multilayer_perceptron_pointer->write_expression(scaled_inputs_name, non_probabilistic_outputs_name);
            }
        }
        else
        {
            buffer << multilayer_perceptron_pointer->write_expression(inputs_name, outputs_name);
        }
    }

    // Outputs unscaling

    if(has_unscaling_layer())
    {
        buffer << unscaling_layer_pointer->write_expression(scaled_outputs_name, outputs_name);
    }

    // Outputs trending layer

    if(has_outputs_trending_layer())
    {
        buffer << outputs_trending_layer_pointer->write_expression(outputs_name, outputs_name);
    }

    // Probabilistic layer

    if(has_probabilistic_layer())
    {
        buffer << probabilistic_layer_pointer->write_expression(non_probabilistic_outputs_name, outputs_name);
    }

    // Bounding layer

    if(has_bounding_layer())
    {
        buffer << bounding_layer_pointer->write_expression(outputs_name, outputs_name);
    }

    string expression = buffer.str();

    position = 0;

    search = "--";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "+-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n+";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\"";
    replace = "";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    return(expression);
}


/// Returns a string with the expression of the function represented by the neural network.

string NeuralNetwork::write_mathematical_expression_php() const
{
    ostringstream buffer;

#ifdef __OPENNN_DEBUG__

    if(!inputs_pointer)
    {
        buffer.str("");

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string write_expression() const method.\n"
               << "Pointer to inputs is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(!multilayer_perceptron_pointer)
    {
        buffer.str("");

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string write_expression() const method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

    if(!outputs_pointer)
    {
        buffer.str("");

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string write_expression() const method.\n"
               << "Pointer to outputs is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    Vector<string> inputs_name = inputs_pointer->get_names();
    Vector<string> outputs_name = outputs_pointer->get_names();

    size_t position = 0;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(inputs_name[i].begin(), inputs_name[i].end(), ' ');
        inputs_name[i].erase(end_pos, inputs_name[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = inputs_name[i].find(search, position)) != string::npos)
        {
            inputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }
    }

    for(size_t i = 0; i < outputs_number; i++)
    {
        position = 0;

        search = "(";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        string::iterator end_pos = remove(outputs_name[i].begin(), outputs_name[i].end(), ' ');
        outputs_name[i].erase(end_pos, outputs_name[i].end());

        position = 0;

        search = "-";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = "/";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }

        position = 0;

        search = ")";
        replace = "_";

        while((position = outputs_name[i].find(search, position)) != string::npos)
        {
            outputs_name[i].replace(position, search.length(), replace);
            position += replace.length();
        }
    }

    // Scaled inputs

    Vector<string> scaled_inputs_name(/*inputs_number*/inputs_name.size());

    for(size_t i = 0; i < inputs_name.size()/*inputs_number*/; i++)
    {
        buffer.str("");

        buffer << "$scaled_" << inputs_name[i];

        scaled_inputs_name[i] = buffer.str();
    }

    // Principal components

    Vector<string> principal_components_name(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer.str("");

        buffer << "$principal_component_" <<(i+1);

        principal_components_name[i] = buffer.str();
    }

    // Scaled outputs

    Vector<string> scaled_outputs_name(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "$scaled_" << outputs_name[i];

        scaled_outputs_name[i] = buffer.str();
    }

    // Non probabilistic outputs

    Vector<string> non_probabilistic_outputs_name(outputs_number);

    for(size_t i = 0; i < outputs_number; i++)
    {
        buffer.str("");

        buffer << "$non_probabilistic_" << outputs_name[i];

        non_probabilistic_outputs_name[i] = buffer.str();
    }

    buffer.str("");

    for (size_t i = 0; i < inputs_name.size(); i++)
    {
        inputs_name[i] = "$"+inputs_name[i];
    }

    for (size_t i = 0; i < outputs_name.size(); i++)
    {
        outputs_name[i] = "$"+outputs_name[i];
    }

    // Inputs trending layer

    if(has_inputs_trending_layer())
    {
        buffer << inputs_trending_layer_pointer->write_expression(inputs_name, outputs_name);
    }

    // Scaling layer

    if(has_scaling_layer())
    {
        buffer << scaling_layer_pointer->write_expression(inputs_name, scaled_inputs_name);
    }

    // Principal components layer

    if(has_principal_components_layer())
    {
        buffer << principal_components_layer_pointer->write_expression(scaled_inputs_name, principal_components_name);
    }

    // Multilayer perceptron

    if(has_multilayer_perceptron())
    {
        if(scaling_layer_pointer && unscaling_layer_pointer)
        {
            if(has_principal_components_layer() && principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
            {
                buffer << multilayer_perceptron_pointer->write_expression_php(principal_components_name, scaled_outputs_name);
            }
            else
            {
                buffer << multilayer_perceptron_pointer->write_expression_php(scaled_inputs_name, scaled_outputs_name);
            }
        }
        else if(scaling_layer_pointer && probabilistic_layer_pointer)
        {
            if(has_principal_components_layer() && principal_components_layer_pointer->write_principal_components_method() != "NoPrincipalComponents")
            {
                buffer << multilayer_perceptron_pointer->write_expression_php(principal_components_name, scaled_outputs_name);
            }
            else
            {
                buffer << multilayer_perceptron_pointer->write_expression_php(scaled_inputs_name, non_probabilistic_outputs_name);
            }
        }
        else
        {
            buffer << multilayer_perceptron_pointer->write_expression_php(inputs_name, outputs_name);
        }
    }

    // Outputs unscaling

    if(has_unscaling_layer())
    {
        buffer << unscaling_layer_pointer->write_expression(scaled_outputs_name, outputs_name);
    }

    // Outputs trending layer

    if(has_outputs_trending_layer())
    {
        buffer << outputs_trending_layer_pointer->write_expression(outputs_name, outputs_name);
    }

    // Probabilistic layer

    if(has_probabilistic_layer())
    {
        buffer << probabilistic_layer_pointer->write_expression(non_probabilistic_outputs_name, outputs_name);
    }

    // Bounding layer

    if(has_bounding_layer())
    {
        buffer << bounding_layer_pointer->write_expression_php(outputs_name, outputs_name);
    }

    string expression = buffer.str();

    position = 0;

    search = "--";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "+-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n-";
    replace = "-";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\n+";
    replace = "+";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    position = 0;

    search = "\"";
    replace = "";

    while((position = expression.find(search, position)) != string::npos)
    {
        expression.replace(position, search.length(), replace);
        position += replace.length();
    }

    return(expression);
}


/// Returns a string with the python function of the expression represented by the neural network.

string NeuralNetwork::write_expression_python() const
{
    ostringstream buffer;

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    Vector<string> inputs_name = inputs_pointer->get_names();
    Vector<string> outputs_name = outputs_pointer->get_names();

    size_t pos;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
    {
        string::iterator end_pos = remove(inputs_name[i].begin(), inputs_name[i].end(), ' ');
        inputs_name[i].erase(end_pos, inputs_name[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    for(size_t i = 0; i < outputs_number; i++)
    {
        string::iterator end_pos = remove(outputs_name[i].begin(), outputs_name[i].end(), ' ');
        outputs_name[i].erase(end_pos, outputs_name[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    Vector<PerceptronLayer::ActivationFunction> activations;

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
        activations.push_back(multilayer_perceptron_pointer->get_layer(i).get_activation_function());

    buffer.str("");

    buffer << "#!/usr/bin/python\n\n";

    if(activations.contains(PerceptronLayer::Threshold))
    {
        buffer << "def Threshold(x) : \n"
                  "   if x < 0 : \n"
                  "       return 0\n"
                  "   else : \n"
                  "       return 1\n\n";
    }

    if(activations.contains(PerceptronLayer::SymmetricThreshold))
    {
        buffer << "def SymmetricThreshold(x) : \n"
                  "   if x < 0 : \n"
                  "       return -1\n"
                  "   else : \n"
                  "       return 1\n\n";
    }

    if(activations.contains(PerceptronLayer::Logistic))
    {
        buffer << "from math import exp\n"
                  "def Logistic(x) : \n"
                  "   return(1/(1+exp(-x))) \n\n";
    }

    if(activations.contains(PerceptronLayer::HyperbolicTangent))
    {
        buffer << "from math import tanh\n\n";
    }

    if(has_probabilistic_layer())
    {
        double decision_threshold = probabilistic_layer_pointer->get_decision_threshold();

        switch(probabilistic_layer_pointer->get_probabilistic_method())
        {
        case ProbabilisticLayer::Binary :

            buffer << "def Binary(x) : \n"
                      "   if x < " << decision_threshold << " : \n"
                                                             "       return 0\n"
                                                             "   else : \n"
                                                             "       return 1\n\n";

            break;
        case ProbabilisticLayer::Probability :

            buffer << "def Probability(x) : \n"
                      "   if x < 0 :\n"
                      "       return 0\n"
                      "   elif x > 1 :\n"
                      "       return 1\n"
                      "   else : \n"
                      "       return x\n\n";

            break;
        case ProbabilisticLayer::Competitive :

            buffer << "def Competitive(";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << ") :\n";

            buffer << "   inputs = [";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << "]\n";
            buffer << "   competitive = [0 for i in range(" << outputs_number << ")]\n"
                                                                                  "   maximal_index = inputs.index(max(inputs))\n"
                                                                                  "   competitive[maximal_index] = 1\n"
                                                                                  "   return competitive\n\n";

            break;
        case ProbabilisticLayer::Softmax :

            buffer << "from math import exp\n"
                      "def Softmax(";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << ") :\n";

            buffer << "   inputs = [";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << "]\n";
            buffer << "   softmax = [0 for i in range(" << outputs_number << ")]\n"
                                                                              "   sum = 0\n"
                                                                              "   for i in range(" << outputs_number << ") :\n"
                                                                                                                         "       sum += exp(inputs[i])\n"
                                                                                                                         "   for i in range(" << outputs_number << ") :\n"
                                                                                                                                                                    "       softmax[i] = exp(inputs[i])/sum\n";
            buffer << "   return softmax\n\n";

            break;
        case ProbabilisticLayer::NoProbabilistic :
            break;
        }
    }

    buffer << "def expression(inputs) : \n\n    ";

    buffer << "if type(inputs) != list:\n    "
           << "   print('Argument must be a list')\n    "
           << "   return\n    ";

    buffer << "if len(inputs) != " << inputs_number << ":\n    "
           << "   print('Incorrect number of inputs')\n    "
           << "   return\n    ";

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer << inputs_name[i] << "=inputs[" << i << "]\n    ";
    }

    string neural_network_expression =  write_expression();

    pos = 0;

    search = "\n";
    replace = "\n    ";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    buffer << neural_network_expression;

    buffer << "\n    return ";

    for(size_t i = 0; i < outputs_number; i++)
    {
        buffer << outputs_name[i];

        if(i != outputs_number - 1)
            buffer << ", ";
    }

    buffer << " \n";
    string expression = buffer.str();

    pos = 0;

    search = "\"";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    pos = 0;

    search = ";";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    return(expression);

}


/// Returns a string with the php function of the expression represented by the neural network.

string NeuralNetwork::write_expression_php() const
{
    ostringstream buffer;

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    Vector<string> inputs_name = inputs_pointer->get_names();
    Vector<string> outputs_name = outputs_pointer->get_names();

    size_t pos;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
    {
        string::iterator end_pos = remove(inputs_name[i].begin(), inputs_name[i].end(), ' ');
        inputs_name[i].erase(end_pos, inputs_name[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    for(size_t i = 0; i < outputs_number; i++)
    {
        string::iterator end_pos = remove(outputs_name[i].begin(), outputs_name[i].end(), ' ');
        outputs_name[i].erase(end_pos, outputs_name[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    Vector<PerceptronLayer::ActivationFunction> activations;

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
        activations.push_back(multilayer_perceptron_pointer->get_layer(i).get_activation_function());

    buffer.str("");

    if(activations.contains(PerceptronLayer::Threshold))
    {
        buffer << "function Threshold($x)\n"
                  "{\n"
                  "   if ($x < 0)\n"
                  "   {\n"
                  "       return 0;\n"
                  "   }\n"
                  "   else\n"
                  "   {\n"
                  "       return 1;\n"
                  "   }\n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::SymmetricThreshold))
    {
        buffer << "function SymmetricThreshold(&x)\n"
                  "{\n"
                  "   if ($x < 0)\n"
                  "   {\n"
                  "       return -1;\n"
                  "   }\n"
                  "   else\n"
                  "   {\n"
                  "       return 1;\n"
                  "   }\n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::Logistic))
    {
        buffer << "function Logistic($x)\n"
                  "{\n"
                  "   return(1/(1+exp(-$x)));"
                  "}\n\n";
    }

    if(has_probabilistic_layer())
    {
        double decision_threshold = probabilistic_layer_pointer->get_decision_threshold();

        switch(probabilistic_layer_pointer->get_probabilistic_method())
        {
        case ProbabilisticLayer::Binary :

            buffer << "function Binary($x)\n"
                      "{\n"
                      "   if ($x<" << decision_threshold << ")\n"
                      "   {\n"
                      "       return 0;\n"
                      "   }\n"
                      "   else\n"
                      "   {\n"
                      "       return 1;\n"
                      "   }\n"
                      "}\n\n";
            break;
        case ProbabilisticLayer::Probability :

            buffer << "function Probability($x)\n"
                      "   if ($x<0)\n"
                      "   {\n"
                      "       return 0;\n"
                      "   }\n"
                      "   elif ($x>1)\n"
                      "   {\n"
                      "       return 1;\n"
                      "   }\n"
                      "   else\n"
                      "   {\n"
                      "       return $x;\n"
                      "   }\n"
                      "}\n\n";
            break;
//        case ProbabilisticLayer::Competitive :

//            buffer << "def Competitive(";
//            for(size_t i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << ") :\n";

//            buffer << "   inputs = [";
//            for(size_t i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << "]\n";
//            buffer << "   competitive = [0 for i in range(" << outputs_number << ")]\n"
//                                                                                  "   maximal_index = inputs.index(max(inputs))\n"
//                                                                                  "   competitive[maximal_index] = 1\n"
//                                                                                  "   return competitive\n\n";

//            break;
//        case ProbabilisticLayer::Softmax :

//            buffer << "from math import exp\n"
//                      "def Softmax(";
//            for(size_t i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << ") :\n";

//            buffer << "   inputs = [";
//            for(size_t i = 0; i < outputs_number; i++)
//            {
//                buffer << "x" << i;

//                if(i != outputs_number - 1)
//                    buffer << ", ";
//            }
//            buffer << "]\n";
//            buffer << "   softmax = [0 for i in range(" << outputs_number << ")]\n"
//                                                                              "   sum = 0\n"
//                                                                              "   for i in range(" << outputs_number << ") :\n"
//                                                                                                                         "       sum += exp(inputs[i])\n"
//                                                                                                                         "   for i in range(" << outputs_number << ") :\n"
//                                                                                                                                                                    "       softmax[i] = exp(inputs[i])/sum\n";
//            buffer << "   return softmax\n\n";

//            break;
//        case ProbabilisticLayer::NoProbabilistic :
//            break;
        default:

            buffer.str("");

            buffer << "OpenNN Exception: NeuralNetwork class.\n"
                   << "string write_expression_python() const method.\n"
                   << "Unknown probabilistic method.\n";

            throw logic_error(buffer.str());
        }
    }

    buffer << "function expression($inputs)\n"
              "{\n";

    buffer << "   if (!is_array($inputs))\n"
              "   {\n"
              "       throw new \\InvalidArgumentException('Argument must be a list.', 1);\n"
              "   }\n";

    buffer << "   if (count($inputs) != " << inputs_number << ")\n"
              "   {\n"
              "       throw new \\InvalidArgumentException('Incorrect number of inputs.', 2);\n"
              "   }\n\n";

    for (size_t i = 0; i < inputs_name.size(); i++)
    {
        inputs_name[i] = "$"+inputs_name[i];
    }

    for (size_t i = 0; i < outputs_name.size(); i++)
    {
        outputs_name[i] = "$"+outputs_name[i];
    }

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer << "   " << inputs_name[i] << "=$inputs[" << i << "];\n";
    }

    string neural_network_expression =  write_mathematical_expression_php();

    pos = 0;

    search = "\n";
    replace = "\n    ";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    buffer << neural_network_expression;

    buffer << "\n    return ";

    for(size_t i = 0; i < outputs_number; i++)
    {
        buffer << outputs_name[i];

        if(i != outputs_number - 1)
            buffer << ", ";
        else
            buffer << ";";
    }

    buffer << " \n";
    buffer << "}";

    string expression = buffer.str();

    pos = 0;

    search = "\"";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

//    pos = 0;

//    search = ";";
//    replace = "";

//    while((pos = expression.find(search, pos)) != string::npos)
//    {
//        expression.replace(pos, search.length(), replace);
//        pos += replace.length();
//    }

    return(expression);

}


/// Returns a string with the R function of the expression represented by the neural network.

string NeuralNetwork::write_expression_R() const
{
    ostringstream buffer;

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();
    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    Vector<string> inputs_name = inputs_pointer->get_names();
    Vector<string> outputs_name = outputs_pointer->get_names();

    size_t pos;

    string search;
    string replace;

    for(size_t i = 0; i < inputs_number; i++)
    {
        string::iterator end_pos = remove(inputs_name[i].begin(), inputs_name[i].end(), ' ');
        inputs_name[i].erase(end_pos, inputs_name[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = inputs_name[i].find(search, pos)) != string::npos)
        {
            inputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    for(size_t i = 0; i < outputs_number; i++)
    {
        string::iterator end_pos = remove(outputs_name[i].begin(), outputs_name[i].end(), ' ');
        outputs_name[i].erase(end_pos, outputs_name[i].end());

        pos = 0;

        search = "-";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "(";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ")";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "+";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "*";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = "/";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }

        pos = 0;

        search = ":";
        replace = "_";

        while((pos = outputs_name[i].find(search, pos)) != string::npos)
        {
            outputs_name[i].replace(pos, search.length(), replace);
            pos += replace.length();
        }
    }

    Vector<PerceptronLayer::ActivationFunction> activations;

    const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

    for(size_t i = 0; i < layers_number; i++)
        activations.push_back(multilayer_perceptron_pointer->get_layer(i).get_activation_function());

    buffer.str("");

    if(activations.contains(PerceptronLayer::Threshold))
    {
        buffer << "Threshold <- function(x) { \n"
                  "   if(x < 0)  0 \n"
                  "   else 1 \n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::SymmetricThreshold))
    {
        buffer << "SymmetricThreshold <- function(x) { \n"
                  "   if(x < 0)  -1 \n"
                  "   else 1 \n"
                  "}\n\n";
    }

    if(activations.contains(PerceptronLayer::Logistic))
    {
        buffer << "Logistic <- function(x) { \n"
                  "   1/(1+exp(-x))\n"
                  "}\n\n";
    }

    if(has_probabilistic_layer())
    {
        double decision_threshold = probabilistic_layer_pointer->get_decision_threshold();

        switch(probabilistic_layer_pointer->get_probabilistic_method())
        {
        case ProbabilisticLayer::Binary :

            buffer << "Binary <- function(x) { \n"
                      "   if(x < " << decision_threshold << ") 0 \n"
                                                             "   else 1 \n"
                                                             "}\n\n";

            break;
        case ProbabilisticLayer::Probability :

            buffer << "Probability <- function(x) { \n"
                      "   if(x < 0)  0\n"
                      "   else if(x > 1)  1\n"
                      "   else  x\n"
                      "}\n\n";

            break;
        case ProbabilisticLayer::Competitive :

            buffer << "Competitive <- function(";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << ") {\n";

            buffer << "   inputs <- c(";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << ")\n";
            buffer << "   competitive <- array(0, " << outputs_number << ")\n"
                                                                          "   maximal_index <- which.max(inputs)\n"
                                                                          "   competitive[maximal_index] <- 1\n"
                                                                          "   competitive\n"
                                                                          "}\n\n";

            break;
        case ProbabilisticLayer::Softmax :

            buffer << "Softmax <- function(";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << ") {\n";

            buffer << "   inputs <- c(";
            for(size_t i = 0; i < outputs_number; i++)
            {
                buffer << "x" << i;

                if(i != outputs_number - 1)
                    buffer << ", ";
            }
            buffer << ")\n";
            buffer << "   softmax <- array(0, " << outputs_number << ")\n"
                                                                      "   sum <- 0\n"
                                                                      "   for(i in 1:" << outputs_number << ") \n"
                                                                                                              "       sum <- sum + exp(inputs[i])\n"
                                                                                                              "   for(i in 1:" << outputs_number << ")\n"
                                                                                                                                                      "       softmax[i] <- exp(inputs[i])/sum\n"
                                                                                                                                                      "   softmax\n"
                                                                                                                                                      "}\n\n";

            break;
        case ProbabilisticLayer::NoProbabilistic :
            break;
        }
    }

    buffer << "expression <- function(inputs) {\n\n    ";

    buffer << "if(length(inputs) != " << inputs_number << ") {\n    "
           << "   print('Incorrect number of inputs')\n    "
           << "   return()\n    "
              "}\n    ";

    for(size_t i = 0; i < inputs_number; i++)
    {
        buffer << inputs_name[i] << "=inputs[" << i+1 << "]\n    ";
    }

    string neural_network_expression =  write_expression();

    pos = 0;

    search = "\n";
    replace = "\n    ";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    pos = 0;

    search = ";";
    replace = "";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    pos = 0;

    search = "=";
    replace = "<-";

    while((pos = neural_network_expression.find(search, pos)) != string::npos)
    {
        neural_network_expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    if(has_bounding_layer())
    {
        Vector< Vector<double> > boundings = bounding_layer_pointer->get_bounds();

        for(size_t i = 0; i < outputs_number; i++)
        {
            pos = 0;

            ostringstream bound;
            bound.precision(10);
            bound << boundings[0][i];

            search = outputs_name[i] + " <- max(" + bound.str() + ", " + outputs_name[i] + ")";
            replace = "outputs[" + to_string(i+1) + "] <- max(" + bound.str() + ", outputs[" + to_string(i+1) + "])";

            while((pos = neural_network_expression.find(search, pos)) != string::npos)
            {
                neural_network_expression.replace(pos, search.length(), replace);
                pos += replace.length();
            }

            pos = 0;

            bound.str("");
            bound.clear();
            bound << boundings[1][i];

            search = outputs_name[i] + " <- min(" + bound.str() + ", " + outputs_name[i] + ")";
            replace = "outputs[" + to_string(i+1) + "] <- min(" + bound.str() + ", outputs[" + to_string(i+1) + "])";

            while((pos = neural_network_expression.find(search, pos)) != string::npos)
            {
                neural_network_expression.replace(pos, search.length(), replace);
                pos += replace.length();
            }
        }
    }

    ostringstream outputs;

    if(has_probabilistic_layer())
    {
        pos = 0;

        switch(probabilistic_layer_pointer->get_probabilistic_method())
        {
        case ProbabilisticLayer::Binary :

            outputs << "(" << outputs_name.vector_to_string(',') << ") <- Binary(";

            search = outputs.str();
            replace = "outputs <- Binary(";

            break;
        case ProbabilisticLayer::Probability :

            outputs << "(" << outputs_name.vector_to_string(',') << ") <- Probability(";

            search = outputs.str();
            replace = "outputs <- Probability(";

            break;
        case ProbabilisticLayer::Competitive :

            outputs << "(" << outputs_name.vector_to_string(',') << ") <- Competitive(";

            search = outputs.str();
            replace = "outputs <- Competitive(";

            break;
        case ProbabilisticLayer::Softmax :

            outputs << "(" << outputs_name.vector_to_string(',') << ") <- Softmax(";

            search = outputs.str();
            replace = "outputs <- Softmax(";

            break;
        case ProbabilisticLayer::NoProbabilistic :

            outputs << "(" << outputs_name.vector_to_string(',') << ") <- (";

            search = outputs.str();
            replace = "outputs <- c(";

            break;
        }

        while((pos = neural_network_expression.find(search, pos)) != string::npos)
        {
            neural_network_expression.replace(pos, search.length(), replace);
            pos += replace.length();
        }

        buffer << neural_network_expression;
    }
    else if(has_unscaling_layer())
    {
        outputs << "(" << outputs_name.vector_to_string(',') << ") <- (";

        pos = 0;

        search = outputs.str();
        replace = "outputs <- c(";

        while((pos = neural_network_expression.find(search, pos)) != string::npos)
        {
            neural_network_expression.replace(pos, search.length(), replace);
            pos += replace.length();
        }

        buffer << neural_network_expression;
    }
    else
    {
        outputs << "outputs <- c(";

        for(size_t i = 0; i < outputs_number; i++)
        {
            outputs << outputs_name[i];

            if(i != outputs_number - 1)
                outputs << ", ";
        }

        outputs << ")\n    ";

        buffer << neural_network_expression;

        buffer << outputs.str();
    }

    buffer << "outputs \n} \n";

    string expression = buffer.str();

    pos = 0;

    search = "\"";
    replace = "";

    while((pos = expression.find(search, pos)) != string::npos)
    {
        expression.replace(pos, search.length(), replace);
        pos += replace.length();
    }

    return(expression);
}


/// Saves the mathematical expression represented by the neural network to a text file.
/// @param file_name Name of expression text file. 

void NeuralNetwork::save_expression(const string& file_name)
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw logic_error(buffer.str());
    }

    file << write_expression();

    file.close();
}


/// Saves the python function of the expression represented by the neural network to a text file.
/// @param file_name Name of expression text file.

void NeuralNetwork::save_expression_python(const string& file_name)
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_python(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw logic_error(buffer.str());
    }

    file << write_expression_python();

    file.close();
}


/// Saves the R function of the expression represented by the neural network to a text file.
/// @param file_name Name of expression text file.

void NeuralNetwork::save_expression_R(const string& file_name)
{
    ofstream file(file_name.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void  save_expression_R(const string&) method.\n"
               << "Cannot open expression text file.\n";

        throw logic_error(buffer.str());
    }

    file << write_expression_R();

    file.close();
}


/// Saves a set of input-output values from the neural network to a data file.
/// @param file_name Name of data file. 

void NeuralNetwork::save_data(const string& file_name) const
{
#ifdef __OPENNN_DEBUG__

    ostringstream buffer;

    if(!multilayer_perceptron_pointer)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_data(const string&) const method.\n"
               << "Pointer to multilayer perceptron is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t inputs_number = multilayer_perceptron_pointer->get_inputs_number();

#ifdef __OPENNN_DEBUG__

    if(inputs_number != 1)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_data(const string&) const method.\n"
               << "Number of inputs is not 1.\n";

        throw logic_error(buffer.str());
    }

#endif

#ifdef __OPENNN_DEBUG__

    if(!scaling_layer_pointer)
    {
        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "void save_data(const string&) const method.\n"
               << "Pointer to scaling layer is nullptr.\n";

        throw logic_error(buffer.str());
    }

#endif

    const size_t outputs_number = multilayer_perceptron_pointer->get_outputs_number();

    const size_t variables_number = inputs_number + outputs_number;

    const Vector< Statistics<double> > scaling_layer_statistics = scaling_layer_pointer->get_statistics();
    //   const Vector< Statistics<double> > unscaling_layer_statistics = unscaling_layer_pointer->get_statistics();

    //   const Vector< Statistics<double> > inputs_minimums = scaling_layer_pointer->get_minimums();
    //   const Vector<double> inputs_maximums = scaling_layer_pointer->get_maximums();

    const size_t points_number = 101;

    Matrix<double> data(points_number, variables_number);

    Vector<double> inputs(inputs_number);
    Vector<double> outputs(outputs_number);
    Vector<double> row(variables_number);

    Vector<double> increments(inputs_number);

    for(size_t i = 0; i < inputs_number; i++)
    {
        inputs[i] = scaling_layer_statistics[i].minimum;
        increments[i] = (scaling_layer_statistics[i].maximum - scaling_layer_statistics[i].minimum)/static_cast<double>(points_number-1.0);
    }

    for(size_t i = 0; i < points_number; i++)
    {
        outputs = calculate_outputs(inputs.to_column_matrix());

        row = inputs.assemble(outputs);

        data.set_row(i, row);

        inputs += increments;
    }

    data.save(file_name);
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
