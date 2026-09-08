// Reproducible reference models. Generated artifacts belong outside the checkout.
#include "opennn/core/configuration.h"
#include "opennn/core/device_backend.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/models/models.h"
#include "opennn/network/model_expression.h"
#include "opennn/training/adam.h"
#include "opennn/training/training.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv)
{
    using namespace opennn;
    try
    {
        if (argc != 4 || (string(argv[1]) != "iris" && string(argv[1]) != "concrete"))
            throw runtime_error("Usage: opennn_reproduce_models iris|concrete DATA.csv EMPTY_OUTPUT_DIRECTORY");
        const bool classification = string(argv[1]) == "iris";
        const filesystem::path output = filesystem::absolute(argv[3]);
        if (filesystem::exists(output) && !filesystem::is_empty(output))
            throw runtime_error("Output directory must be empty; preserve previous generation records.");
        filesystem::create_directories(output);
        Configuration::instance().set(Device::CPU, Type::FP32);
        Configuration::instance().set_blas(Blas::Eigen);
        set_threads_number(1);
        set_seed(1729);
        srand(1729);

        TabularDataset dataset(argv[2], classification ? ";" : ",", true, false);
        dataset.set_display(false);
        std::ofstream split(output / "split.csv");
        split << "row_zero_based,role\n";
        for (Index i = 0; i < dataset.get_samples_number(); ++i)
        {
            const char* role = i % 10 < 6 ? "Training" : i % 10 < 8 ? "Validation" : "Testing";
            dataset.set_sample_role(i, role);
            split << i << ',' << role << '\n';
        }
        unique_ptr<Network> network;
        if (classification)
            network = make_unique<ClassificationNetwork>(dataset.get_input_shape(), Shape{16}, dataset.get_target_shape());
        else
            network = make_unique<ApproximationNetwork>(dataset.get_input_shape(), Shape{32,16}, dataset.get_target_shape());

        Training training(network.get(), &dataset);
        training.set_optimization_algorithm("Adam");
        auto* optimizer = dynamic_cast<Adam*>(training.get_optimization_algorithm());
        optimizer->set_learning_rate(0.01f);
        optimizer->set_beta_1(0.9f);
        optimizer->set_beta_2(0.999f);
        optimizer->set_bf16_first_moment(false);
        optimizer->set_workers_number(1);
        optimizer->set_shuffle(false);
        optimizer->set_batch_size(128);
        optimizer->set_maximum_epochs(1000);
        optimizer->set_maximum_validation_failures(1001);
        optimizer->set_maximum_time(3600);
        optimizer->set_restore_best(true);
        optimizer->set_display(false);
        training.save(output / "training.json");
        const TrainingResult result = training.train();

        const MatrixR inputs = dataset.get_data("Testing", "Input");
        const MatrixR targets = dataset.get_data("Testing", "Target");
        const MatrixR predictions = network->calculate_outputs(inputs);
        if (!predictions.allFinite()) throw runtime_error("Nonfinite model predictions.");
        float quality = 0;
        if (classification)
        {
            for (Index i = 0; i < predictions.rows(); ++i)
            {
                Index actual, expected;
                predictions.row(i).maxCoeff(&actual);
                targets.row(i).maxCoeff(&expected);
                quality += actual == expected ? 1.0f : 0.0f;
            }
            quality /= static_cast<float>(predictions.rows());
        }
        else
        {
            const float baseline = (targets.array() - targets.mean()).square().sum();
            quality = 1.0f - (predictions - targets).squaredNorm() / baseline;
        }
        if (!std::isfinite(quality) || quality < (classification ? 0.85f : 0.70f))
            throw runtime_error("Reference model does not meet the documented held-out quality threshold: " + to_string(quality));

        network->save(output / "model.json"); // Writes the paired model.bin as well.
        Network restored;
        restored.load(output / "model.json");
        const MatrixR reloaded = restored.calculate_outputs(inputs);
        const float error = (predictions - reloaded).cwiseAbs().maxCoeff();
        if (!reloaded.allFinite() || error > 1.0e-6f * max(1.0f, predictions.cwiseAbs().maxCoeff()))
            throw runtime_error("Saved model prediction round trip failed.");
        ModelExpression(network.get()).save(output / "model.py", ModelExpression::ProgrammingLanguage::Python);

        std::ofstream reference(output / "reference.csv");
        reference << std::setprecision(9);
        for (Index i = 0; i < inputs.rows(); ++i)
        {
            for (Index j = 0; j < inputs.cols(); ++j) reference << inputs(i,j) << ',';
            for (Index j = 0; j < predictions.cols(); ++j)
                reference << predictions(i,j) << (j+1 == predictions.cols() ? '\n' : ',');
        }
        std::ofstream metadata(output / "generation.json");
        metadata << std::setprecision(9)
                 << "{\n  \"recipe\": " << std::quoted(argv[1])
                 << ",\n  \"seed\": 1729,\n  \"device\": \"CPU\",\n  \"precision\": \"FP32\",\n  \"blas\": \"Eigen\",\n  \"threads\": 1,"
                 << "\n  \"epochs_recorded\": " << result.get_epochs_number()
                 << ",\n  \"stopping_condition\": " << std::quoted(result.write_stopping_condition())
                 << ",\n  \"test_rows\": " << inputs.rows()
                 << ",\n  \"input_columns\": " << inputs.cols()
                 << ",\n  \"metric\": " << std::quoted(classification ? "accuracy" : "R2")
                 << ",\n  \"value\": " << quality
                 << ",\n  \"reload_max_absolute_error\": " << error << "\n}\n";
        if (!reference || !metadata || !split) throw runtime_error("Failed to write generation record.");
        std::cout << argv[1] << ": held-out " << (classification ? "accuracy=" : "R2=") << quality
                  << ", reload error=" << error << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
