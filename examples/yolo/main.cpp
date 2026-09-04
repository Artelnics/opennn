//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y O L O   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

//   Trains on data_directory/{images,labels}. Usage: yolo <data_directory>. Requires CUDA.

#include <filesystem>
#include <iostream>

#include "opennn/core/random_utilities.h"
#include "opennn/dataset/image_processing.h"
#include "opennn/dataset/yolo_dataset.h"
#include "opennn/models/models.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        cout << "OpenNN. YOLO example." << endl;

        throw_if(argc < 2, "Usage: yolo <data_directory> (containing images/ and labels/).");

        set_seed(0);

        Configuration::instance().set(Device::CUDA, Type::FP32);
        (void)Configuration::instance().resolve();

        const filesystem::path data_directory = argv[1];
        const filesystem::path images_directory = data_directory / "images";
        const filesystem::path labels_directory = data_directory / "labels";

        throw_if(!filesystem::is_directory(images_directory)
                 || !filesystem::is_directory(labels_directory),
                 "YOLO data not found in '{}'. Expected images/ and labels/ subdirectories.",
                 data_directory.string());

        const Shape image_shape{416, 416, 3};

        YoloDataset dataset(images_directory,
                            labels_directory,
                            image_shape);

        YoloNetwork yolo_network(image_shape,
                                 dataset.get_classes_number(),
                                 dataset.get_anchors(),
                                 dataset.get_grid_size(),
                                 YoloNetwork::Backbone::DarknetTinyV3,
                                 YoloNetwork::ClassActivation::Sigmoid,
                                 YoloNetwork::HeadStyle::Single,
                                 YoloNetwork::BodyActivation::LeakyReLU);

        yolo_network.load_pretrained_backbone(data_directory);

        TrainingStrategy training_strategy(&yolo_network, &dataset);
        training_strategy.set_loss("Yolo");
        training_strategy.get_optimization_algorithm()->set_maximum_epochs(10);
        training_strategy.get_optimization_algorithm()->set_maximum_time(600.0f);

        training_strategy.train();

        const Index sample_index = dataset.get_sample_indices("Training").front();
        Tensor4 image(1, image_shape[0], image_shape[1], image_shape[2]);
        dataset.fill_inputs({sample_index}, {}, image.data(), FillMode::Inference);

        const MatrixR outputs = yolo_network.calculate_outputs(image);
        const Tensor3 original_image = load_image(dataset.get_image_path(sample_index));
        const vector<YoloDetection> detections = decode_yolo_detections(
            span<const float>(outputs.data(), size_t(outputs.size())),
            original_image.dimension(0), original_image.dimension(1),
            image_shape[0], image_shape[1]);

        cout << "Detections in " << dataset.get_image_path(sample_index) << ':' << endl;

        for (const YoloDetection& detection : detections)
            cout << dataset.get_class_names()[size_t(detection.class_id)]
                 << " (" << detection.score << "), center: ("
                 << detection.center_x << ", " << detection.center_y
                 << "), size: (" << detection.width << ", " << detection.height
                 << ')' << endl;

        cout << "Good bye!" << endl;
        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
