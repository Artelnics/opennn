//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/opennn.h"
#include <filesystem>
#include <iostream>

using namespace opennn;

namespace
{

void extract_child_document(const filesystem::path& model_path,
                            const char* child_name,
                            XMLDocument& child_document)
{
    XMLDocument model_document;
    if(model_document.LoadFile(model_path.c_str()) != XML_SUCCESS)
        throw runtime_error("Cannot load model file: " + model_path.string());

    const XMLElement* root = model_document.FirstChildElement("NeuralDesignerModel");
    if(!root)
        throw runtime_error("Missing NeuralDesignerModel root.");

    const XMLElement* child = root->FirstChildElement(child_name);
    if(!child)
        throw runtime_error(string("Missing child element: ") + child_name);

    XMLPrinter printer;
    child->Accept(&printer);

    if(child_document.Parse(printer.CStr()) != XML_SUCCESS)
        throw runtime_error(string("Cannot parse child element: ") + child_name);
}


ImageDataset load_project_dataset(const filesystem::path& model_path,
                                  const filesystem::path& binary_data_path)
{
    XMLDocument data_document;
    extract_child_document(model_path, "ImageDataset", data_document);

    ImageDataset dataset;
    dataset.from_XML(data_document);
    dataset.set_data_path(binary_data_path);
    dataset.load_data_binary();

    return dataset;
}


unique_ptr<NeuralNetwork> make_manual_network(const Shape& input_shape, const Shape& output_shape)
{
    unique_ptr<NeuralNetwork> neural_network = make_unique<NeuralNetwork>();
    neural_network->reference_all_layers();

    unique_ptr<Scaling<4>> scaling_layer = make_unique<Scaling<4>>(input_shape);
    scaling_layer->set_scalers("ImageMinMax");
    neural_network->add_layer(move(scaling_layer));

    const Shape filters = {32, 64, 128};
    for(Index i = 0; i < filters.size(); i++)
    {
        neural_network->add_layer(make_unique<Convolutional>(neural_network->get_output_shape(),
                                                             Shape{3, 3, neural_network->get_output_shape()[2], filters[i]},
                                                             "RectifiedLinear",
                                                             Shape{1, 1},
                                                             "Same",
                                                             false,
                                                             "convolutional_layer_" + to_string(i + 1)));

        neural_network->add_layer(make_unique<Pooling>(neural_network->get_output_shape(),
                                                       Shape{2, 2},
                                                       Shape{2, 2},
                                                       Shape{0, 0},
                                                       "MaxPooling",
                                                       "pooling_layer_" + to_string(i + 1)));
    }

    neural_network->add_layer(make_unique<Flatten<4>>(neural_network->get_output_shape()));

    neural_network->add_layer(make_unique<opennn::Dense<2>>(neural_network->get_output_shape(),
                                                            Shape{128},
                                                            "HyperbolicTangent",
                                                            false,
                                                            "dense2d_layer_1"));

    neural_network->add_layer(make_unique<opennn::Dense<2>>(neural_network->get_output_shape(),
                                                            output_shape,
                                                            "Sigmoid",
                                                            false,
                                                            "classification_layer"));

    neural_network->compile();
    neural_network->set_parameters_glorot();

    return neural_network;
}


unique_ptr<NeuralNetwork> load_xml_network(const filesystem::path& model_path)
{
    XMLDocument network_document;
    extract_child_document(model_path, "NeuralNetwork", network_document);

    unique_ptr<NeuralNetwork> neural_network = make_unique<NeuralNetwork>();
    neural_network->from_XML(network_document);
    neural_network->set_parameters_glorot();

    return neural_network;
}


void train_case(const string& label, NeuralNetwork& neural_network, ImageDataset& dataset)
{
    cout << "\n[CASE] " << label << endl;
    cout << "[DATASET] train=" << dataset.get_samples_number("Training")
         << " val=" << dataset.get_samples_number("Validation")
         << " test=" << dataset.get_samples_number("Testing")
         << " input=" << dataset.get_shape("Input")
         << " target=" << dataset.get_shape("Target") << endl;
    cout << "[NETWORK] params=" << neural_network.get_parameters_number()
         << " output=" << neural_network.get_output_shape() << endl;

    const vector<vector<Index>>& input_indices = neural_network.get_layer_input_indices();
    for(Index i = 0; i < neural_network.get_layers_number(); i++)
    {
        const Layer* layer = neural_network.get_layer(i).get();
        cout << "  layer " << i
             << " name=" << layer->get_name()
             << " label=" << layer->get_label()
             << " in=" << layer->get_input_shape()
             << " out=" << layer->get_output_shape()
             << " inputs=[";
        for(Index j = 0; j < static_cast<Index>(input_indices[i].size()); j++)
            cout << (j == 0 ? "" : ",") << input_indices[i][j];
        cout << "]" << endl;
    }

    const VectorR& parameters = neural_network.get_parameters();
    cout << "[PARAMS] min=" << parameters.minCoeff()
         << " max=" << parameters.maxCoeff()
         << " mean=" << parameters.mean()
         << " l2=" << parameters.norm() << endl;

    TrainingStrategy training_strategy(&neural_network, &dataset);
    training_strategy.set_loss("CrossEntropyError2d");
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
    training_strategy.get_loss()->set_regularization_method("None");

    AdaptiveMomentEstimation* adam =
        dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
    adam->set_batch_size(32);
    adam->set_learning_rate(type(0.0001));
    adam->set_maximum_epochs(20);
    adam->set_display_period(2);

    training_strategy.train_cuda();
}

}


int main()
{
    try
    {
        cout << "OpenNN. Neural Designer image debug." << endl;

#ifdef OPENNN_CUDA

        const filesystem::path project_dir =
            "/home/artelnics/Documents/NeuralDesignerProjects/histoa/histoa_v1.0.0";
        const filesystem::path model_path = project_dir / "histoa_v1.0.0.ndm";
        const filesystem::path binary_data_path =
            project_dir / "histoa_train_1000_filter_small_bmp_Data.bin";

        ImageDataset manual_dataset = load_project_dataset(model_path, binary_data_path);
        unique_ptr<NeuralNetwork> manual_network =
            make_manual_network(manual_dataset.get_shape("Input"), Shape{1});
        const VectorR shared_initial_parameters = manual_network->get_parameters();
        train_case("manual network + project binary dataset", *manual_network, manual_dataset);

        ImageDataset xml_dataset = load_project_dataset(model_path, binary_data_path);
        unique_ptr<NeuralNetwork> xml_network = load_xml_network(model_path);
        xml_network->set_parameters(shared_initial_parameters);
        train_case("XML network + manual initial parameters + project binary dataset", *xml_network, xml_dataset);

#endif

        cout << "Bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
