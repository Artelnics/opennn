//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   R E S U L T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "training_result.h"
#include "memory_debug.h"

namespace opennn
{

TrainingResult::TrainingResult(const Index epochs_number)
{
    training_error_history = VectorR::Constant(epochs_number, -1.0f);
    validation_error_history = VectorR::Constant(epochs_number, -1.0f);
}

string TrainingResult::write_stopping_condition() const
{
    if (!stopping_condition) return "None";

    using enum StoppingCondition;
    switch (*stopping_condition)
    {
    case MinimumLossDecrease:
        return "Minimum loss decrease";

    case LossGoal:
        return "Loss goal";

    case MaximumValidationErrorIncreases:
        return "Maximum validation error increases";

    case MaximumEpochsNumber:
        return "Maximum epochs number";

    case MaximumTime:
        return "Maximum training time";

    default:
        return {};
    }
}

float TrainingResult::get_training_error() const
{
    return training_error_history(training_error_history.size() - 1);
}

float TrainingResult::get_validation_error() const
{
    if (validation_error_history.size() == 0) return 0.0f;

    return validation_error_history(validation_error_history.size() - 1);
}

Index TrainingResult::get_epochs_number() const
{
    return training_error_history.size() - 1;
}

void TrainingResult::resize_training_error_history(const Index new_size)
{
    training_error_history.conservativeResize(new_size);
}

void TrainingResult::resize_validation_error_history(const Index new_size)
{
    validation_error_history.conservativeResize(new_size);
}

void TrainingResult::save(const filesystem::path& file_name) const
{
    const Tensor<string, 2> override_results = write_override_results();

    ofstream file(file_name);

    throw_if(!file, format("TrainingResult::save: cannot open {}", file_name.string()));

    for (Index i = 0; i < override_results.dimension(0); ++i)
        file << override_results(i,0) << "; " << override_results(i,1) << "\n";
}

void TrainingResult::print(const string &message) const
{
    const Index epochs_number = training_error_history.size();
    const Index final_epoch = epochs_number - 1;

    Index best_epoch = final_epoch;
    if (validation_error_history.size() > 0)
    {
        Eigen::Index best_validation_index;
        validation_error_history.minCoeff(&best_validation_index);
        best_epoch = Index(best_validation_index);
    }

    const bool restored_best_epoch = restored_best_parameters
        && restored_epoch >= 0
        && restored_epoch < epochs_number;

    const Index reported_epoch = restored_best_epoch ? restored_epoch : final_epoch;

    cout << message << "\n"
         << "Training results" << "\n"
         << "Epochs number: " << final_epoch << "\n"
         << "Training error: " << training_error_history(reported_epoch) << "\n";
    if (validation_error_history.size() > 0)
        cout << "Validation error: " << validation_error_history(reported_epoch) << "\n";
    if (validation_error_history.size() > 0 && best_epoch != final_epoch)
    {
        if (restored_best_epoch)
            cout << "Best epoch: " << restored_epoch
                 << " (restored parameters and states correspond to this epoch)\n";
        else
            cout << "Best validation epoch: " << best_epoch
                 << " (final parameters correspond to epoch " << final_epoch << ")\n";
    }
    cout << "Stopping condition: " << write_stopping_condition() << "\n";
}

Tensor<string, 2> TrainingResult::write_override_results(const Index precision) const
{
    Tensor<string, 2> override_results(5, 2);

    override_results(0, 0) = "Epochs number";
    override_results(1, 0) = "Elapsed time";
    override_results(2, 0) = "Stopping criterion";
    override_results(3, 0) = "Training error";
    override_results(4, 0) = "Validation error";

    const Index size = training_error_history.size();

    if (size == 0)
    {
        for (Index i = 0; i < 5; ++i)
            override_results(i, 1) = "NA";

        return override_results;
    }

    override_results(0, 1) = to_string(size - 1);
    override_results(1, 1) = elapsed_time;
    override_results(2, 1) = write_stopping_condition();
    override_results(3, 1) = to_string(training_error_history(size - 1));


    override_results(4, 1) = validation_error_history.size() == 0
        ? "NAN"
        : format("{:.{}g}", validation_error_history(size - 1), precision);

    return override_results;
}

void OptimizerData::print() const
{
    cout << "Potential parameters:" << "\n"
         << potential_parameters << "\n"
         << "Training direction:" << "\n"
         << training_direction << "\n"
         << "Initial learning rate:" << "\n"
         << initial_learning_rate << "\n";
}

void OptimizerData::set(const vector<Shape>& slot_shapes, Device device)
{
    const Index total_bytes = get_aligned_bytes(slot_shapes, Type::FP32);

    data.resize_bytes(total_bytes, device);
    memory_debug::record("optimizer", "OptimizerData::data", total_bytes,
                         format("slots={}", slot_shapes.size()));

    if (total_bytes > 0)
    {
        if (device == Device::CUDA)
            opennn::device::set_zero_async(data.data, total_bytes, Backend::get_compute_stream());
        else
            data.setZero();
    }

    views.clear();
    views.reserve(slot_shapes.size());

    uint8_t* cursor = data.as<uint8_t>();

    for (const Shape& shape : slot_shapes)
    {
        if (shape.size() > 0)
        {
            views.emplace_back(cursor, shape, Type::FP32, data.device_type);
            cursor += get_aligned_bytes(shape.size(), Type::FP32);
        }
        else
        {
            views.emplace_back();
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
