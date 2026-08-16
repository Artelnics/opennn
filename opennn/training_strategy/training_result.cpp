//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   R E S U L T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/training_result.h"
#include "opennn/core/memory_debug.h"
#include "opennn/core/statistics.h"

namespace opennn
{

TrainingResult::TrainingResult(const Index epochs_number)
{
    // NaN, not -1: with validation_period > 1 some epochs are never evaluated, and the
    // marker for those is read back by minimum searches. -1 sorts below every real error,
    // so an unevaluated epoch always won; NaN is skipped instead of preferred.
    training_error_history = VectorR::Constant(epochs_number, QUIET_NAN);
    validation_error_history = VectorR::Constant(epochs_number, QUIET_NAN);
}

string TrainingResult::write_stopping_condition() const
{
    if (!stopping_condition) return "None";

    static constexpr const char* names[] = {"Minimum loss decrease",
                                            "Loss goal",
                                            "Maximum validation error increases",
                                            "Maximum epochs number",
                                            "Maximum training time"};

    const size_t index = size_t(*stopping_condition);
    return index < size(names) ? names[index] : "";
}

float TrainingResult::get_training_error() const
{
    return training_error_history(training_error_history.size() - 1);
}

float TrainingResult::get_validation_error() const
{
    // The final epoch is not necessarily a validation epoch, so walk back to the last one
    // that produced a number rather than handing back the not-evaluated marker.
    for (Index i = validation_error_history.size() - 1; i >= 0; --i)
        if (isfinite(validation_error_history(i)))
            return validation_error_history(i);

    return 0.0f;
}

Index TrainingResult::get_epochs_number() const
{
    return training_error_history.size();
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

    throw_if(!file, "TrainingResult::save: cannot open {}", file_name.string());

    for (Index i = 0; i < override_results.dimension(0); ++i)
        file << override_results(i,0) << "; " << override_results(i,1) << "\n";
}

void TrainingResult::print(const string &message) const
{
    const Index epochs_number = training_error_history.size();
    const Index final_epoch = epochs_number - 1;

    const Index best_epoch = validation_error_history.size() > 0
        ? minimal_index(validation_error_history)
        : final_epoch;

    const bool restored_best_epoch = restored_epoch
        && *restored_epoch >= 0
        && *restored_epoch < epochs_number;

    const Index reported_epoch = restored_best_epoch ? *restored_epoch : final_epoch;

    cout << message << "\n"
         << "Training results" << "\n"
         << "Epochs number: " << epochs_number << "\n"
         << "Training error: " << training_error_history(reported_epoch) << "\n";
    if (validation_error_history.size() > 0)
    {
        cout << "Validation error: " << validation_error_history(reported_epoch) << "\n";

        if (best_epoch != final_epoch)
        {
            if (restored_best_epoch)
                cout << "Best epoch: " << *restored_epoch
                     << " (restored parameters and states correspond to this epoch)\n";
            else
                cout << "Best validation epoch: " << best_epoch
                     << " (final parameters correspond to epoch " << final_epoch << ")\n";
        }
    }
    cout << "Stopping condition: " << write_stopping_condition() << "\n";
}

Tensor<string, 2> TrainingResult::write_override_results(const Index precision) const
{
    Tensor<string, 2> override_results(5, 2);

    static constexpr const char* labels[] = {"Epochs number", "Elapsed time", "Stopping criterion",
                                             "Training error", "Validation error"};
    for (Index i = 0; i < 5; ++i)
        override_results(i, 0) = labels[i];

    const Index size = training_error_history.size();

    if (size == 0)
    {
        for (Index i = 0; i < 5; ++i)
            override_results(i, 1) = "NA";

        return override_results;
    }

    override_results(0, 1) = to_string(size);
    override_results(1, 1) = elapsed_time;
    override_results(2, 1) = write_stopping_condition();

    override_results(3, 1) = format("{:.{}g}", training_error_history(size - 1), precision);

    override_results(4, 1) = validation_error_history.size() == 0
        ? "QUIET_NAN"
        : format("{:.{}g}", validation_error_history(size - 1), precision);

    return override_results;
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
            opennn::device::set_zero_async(data.data(), total_bytes, device::get_compute_stream());
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
            views.emplace_back(cursor, shape, Type::FP32, data.get_device());
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
