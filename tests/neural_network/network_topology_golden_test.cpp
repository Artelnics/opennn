//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E T W O R K   T O P O L O G Y   G O L D E N   T E S T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The standard-network builders are long straight-line functions that wire
// layers together by index, and nothing pins the graph they produce: a test
// that trains one only notices a mistake that changes the loss. This dumps the
// full topology -- every layer's label, type, shapes and inputs -- for a set of
// configurations into OPENNN_TOPOLOGY_DUMP_DIR, so a builder refactor can be
// verified by diffing the graph before and after.
//
//     set OPENNN_TOPOLOGY_DUMP_DIR=C:\some\dir  &&  opennn_tests.exe
//         --gtest_filter=NetworkTopologyGolden.*
//
// The test skips when the variable is unset, so it costs nothing in CI.

#include "tests/pch.h"

#include "opennn/neural_network/neural_network.h"
#include "opennn/models/models.h"
#include "opennn/neural_network/layers/layer.h"
#include "opennn/registry.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>

using namespace opennn;

namespace
{

filesystem::path dump_directory()
{
    const char* const directory = getenv("OPENNN_TOPOLOGY_DUMP_DIR");
    return directory ? filesystem::path(directory) : filesystem::path();
}


string shape_text(const Shape& shape)
{
    string text = "(";

    for (size_t i = 0; i < shape.get_rank(); ++i)
        text += (i ? ", " : "") + to_string(shape[Index(i)]);

    return text + ")";
}


void dump_topology(const NeuralNetwork& neural_network, const string& case_name)
{
    ofstream file(dump_directory() / (case_name + ".topology"));

    const vector<unique_ptr<Layer>>& layers = neural_network.get_layers();

    file << case_name << "\n"
         << "layers: " << layers.size() << "\n"
         << "parameters: " << neural_network.get_parameters_number() << "\n"
         << "parameter buffer: " << neural_network.get_parameters_buffer_size() << "\n\n";

    for (size_t i = 0; i < layers.size(); ++i)
    {
        file << i << "  " << layers[i]->get_label()
             << "  type=" << layer_type_to_string(layers[i]->get_type())
             << "  in=" << shape_text(layers[i]->get_input_shape())
             << "  out=" << shape_text(layers[i]->get_output_shape())
             << "  params=" << layers[i]->get_parameters_number()
             << "  sources=[";

        const vector<Index>& inputs = neural_network.get_source_layers()[i];

        for (size_t j = 0; j < inputs.size(); ++j)
            file << (j ? "," : "") << inputs[j];

        file << "]\n";
    }
}

}


class NetworkTopologyGolden : public ::testing::Test
{
protected:

    void SetUp() override
    {
        if (dump_directory().empty())
            GTEST_SKIP() << "OPENNN_TOPOLOGY_DUMP_DIR is not set.";

        filesystem::create_directories(dump_directory());
    }
};


TEST_F(NetworkTopologyGolden, YoloEveryBackboneAndHead)
{
    using B  = YoloNetwork::Backbone;
    using CA = YoloNetwork::ClassActivation;
    using HS = YoloNetwork::HeadStyle;
    using BA = YoloNetwork::BodyActivation;
    using MS = YoloNetwork::ModelSize;

    struct Case
    {
        const char* name;
        B backbone;
        HS head;
        Index anchors_number;
        bool use_sppf;
        Index reg_max;
        MS model_size;
    };

    // One per branch the constructor can take, including the sizes that scale
    // the v11 backbone and the SPPF neck that only Darknet53 inserts.
    const Case cases[] = {
        {"yolo_vgg_single",        B::Vgg,             HS::Single, 3, false,  1, MS::l},
        {"yolo_tiny_single",       B::DarknetTiny,     HS::Single, 3, false,  1, MS::l},
        {"yolo_tiny_fpn",          B::DarknetTiny,     HS::FPN,    9, false,  1, MS::l},
        {"yolo_tinyv3_fpn",        B::DarknetTinyV3,   HS::FPN,    6, false,  1, MS::l},
        {"yolo_darknet53_fpn",     B::Darknet53,       HS::FPN,    9, false,  1, MS::l},
        {"yolo_darknet53_fpn_sppf",B::Darknet53,       HS::FPN,    9, true,   1, MS::l},
        {"yolo_darknet53_panet",   B::Darknet53,       HS::PANet,  9, false,  1, MS::l},
        {"yolo_csp53_panet",       B::CSPDarknet53,    HS::PANet,  9, false,  1, MS::l},
        {"yolo_darknet53_fpnv8",   B::Darknet53,       HS::FPNv8,  9, false, 16, MS::l},
        {"yolo_v11_n",             B::CSPDarknet53v11, HS::FPNv8,  9, false, 16, MS::n},
        {"yolo_v11_s",             B::CSPDarknet53v11, HS::FPNv8,  9, false, 16, MS::s},
        {"yolo_v11_m",             B::CSPDarknet53v11, HS::FPNv8,  9, false, 16, MS::m},
        {"yolo_v11_l",             B::CSPDarknet53v11, HS::FPNv8,  9, false, 16, MS::l},
        {"yolo_v11_x",             B::CSPDarknet53v11, HS::FPNv8,  9, false, 16, MS::x}
    };

    const Shape input_shape{320, 320, 3};
    const Index classes = 4;
    const Index grid_size = 10;

    for (const Case& test_case : cases)
    {
        SCOPED_TRACE(test_case.name);

        const vector<std::array<float, 2>> anchors(size_t(test_case.anchors_number), {0.1f, 0.1f});

        YoloNetwork network(input_shape, classes, anchors, grid_size,
                            test_case.backbone, CA::Sigmoid, test_case.head,
                            BA::LeakyReLU, test_case.use_sppf,
                            test_case.reg_max, test_case.model_size);

        dump_topology(network, test_case.name);
    }
}


TEST_F(NetworkTopologyGolden, TabularAndSequenceNetworks)
{
    ApproximationNetwork approximation(Shape{3}, Shape{4}, Shape{2});
    dump_topology(approximation, "approximation");

    ClassificationNetwork classification(Shape{4}, Shape{5}, Shape{3});
    dump_topology(classification, "classification");

    ForecastingNetwork forecasting(Shape{2, 3}, Shape{4}, Shape{1});
    dump_topology(forecasting, "forecasting");
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
