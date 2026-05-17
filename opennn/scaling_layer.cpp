//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "scaling_layer.h"
#include "math_utilities.h"
#include "forward_propagation.h"
#include "string_utilities.h"
#include "cuda_dispatch.h"
#include "json.h"

namespace opennn
{

namespace
{

VectorR descriptives_field(const vector<Descriptives>& descriptives,
                           float Descriptives::* member)
{
    VectorR values(ssize(descriptives));
    for (Index i = 0; i < values.size(); ++i)
        values(i) = descriptives[size_t(i)].*member;
    return values;
}

}

Scaling::Scaling(const Shape& new_input_shape)
    : Layer("Scaling", LayerType::Scaling, false)
{
    operators = {&scale_op};
    set(new_input_shape);
}

VectorR Scaling::get_minimums()            const { return descriptives_field(descriptives, &Descriptives::minimum); }
VectorR Scaling::get_maximums()            const { return descriptives_field(descriptives, &Descriptives::maximum); }
VectorR Scaling::get_means()               const { return descriptives_field(descriptives, &Descriptives::mean); }
VectorR Scaling::get_standard_deviations() const { return descriptives_field(descriptives, &Descriptives::standard_deviation); }

void Scaling::set(const Shape& new_input_shape)
{
    input_shape = new_input_shape;

    set_label("scaling_layer");

    scale_op.input_slots  = {Input};
    scale_op.output_slots = {Output};

    const Index features = input_shape.size();
    descriptives.assign(size_t(features), Descriptives(-1.0f, 1.0f, 0.0f, 1.0f));
    scalers.assign(size_t(features), ScalerMethod::MeanStandardDeviation);
    min_range = -1.0f;
    max_range = 1.0f;
    op_storage_dirty = true;

    if (input_shape.empty()) return;

    check_rank(input_shape, {1, 2, 3}, "Scaling", "input");
}

void Scaling::set_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape);
}

void Scaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    if (ssize(new_descriptives) != ssize(descriptives))
        throw runtime_error("Scaling::set_descriptives: size mismatch (expected "
                            + to_string(descriptives.size()) + ", got "
                            + to_string(new_descriptives.size()) + ").");
    descriptives = new_descriptives;
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Scaling::set_min_max_range(float new_min, float new_max)
{
    min_range = new_min;
    max_range = new_max;
    scale_op.min_range = new_min;
    scale_op.max_range = new_max;
}

void Scaling::set_scalers(const vector<string>& scalers_str)
{
    if (ssize(scalers_str) != ssize(scalers))
        throw runtime_error("Scaling::set_scalers: size mismatch (expected "
                            + to_string(scalers.size()) + ", got "
                            + to_string(scalers_str.size()) + ").");
    ranges::transform(scalers_str, scalers.begin(), string_to_scaler_method);
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Scaling::set_scalers(const string& scaler)
{
    const ScalerMethod method = string_to_scaler_method(scaler);
    ranges::fill(scalers, method);
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

float* Scaling::link_states(float* pointer)
{
    refresh_op_storage(current_device());
    return pointer;
}

void Scaling::refresh_op_storage(Device device)
{
    const Index features = ssize(descriptives);
    const Index bytes    = 5 * features * Index(sizeof(float));

    const bool needs = op_storage_dirty
                    || op_storage.bytes       != bytes
                    || op_storage.device_type != device;
    if (!needs) return;

    op_storage.resize_bytes(bytes, device);
    scale_op.min_range = min_range;
    scale_op.max_range = max_range;

    if (features == 0)
    {
        scale_op.minimums = scale_op.maximums = scale_op.means =
            scale_op.standard_deviations = scale_op.scalers = TensorView();
        op_storage_dirty = false;
        return;
    }

    vector<float> staging(size_t(5 * features));
    for (Index i = 0; i < features; ++i)
    {
        staging[size_t(0 * features + i)] = descriptives[size_t(i)].minimum;
        staging[size_t(1 * features + i)] = descriptives[size_t(i)].maximum;
        staging[size_t(2 * features + i)] = descriptives[size_t(i)].mean;
        staging[size_t(3 * features + i)] = descriptives[size_t(i)].standard_deviation;
        staging[size_t(4 * features + i)] = float(int(scalers[size_t(i)]));
    }

#ifdef OPENNN_HAS_CUDA
    if (device == Device::CUDA)
    {
        CHECK_CUDA(cudaMemcpy(op_storage.data, staging.data(),
                              size_t(bytes), cudaMemcpyHostToDevice));
    }
    else
#endif
    {
        memcpy(op_storage.data, staging.data(), size_t(bytes));
    }

    float* const base = op_storage.as<float>();
    const Shape shape{features};
    scale_op.minimums            = TensorView(base + 0 * features, shape, Type::FP32);
    scale_op.maximums            = TensorView(base + 1 * features, shape, Type::FP32);
    scale_op.means               = TensorView(base + 2 * features, shape, Type::FP32);
    scale_op.standard_deviations = TensorView(base + 3 * features, shape, Type::FP32);
    scale_op.scalers             = TensorView(base + 4 * features, shape, Type::FP32);

    op_storage_dirty = false;
}

void Scaling::read_JSON_body(const Json* scaling_layer_element)
{
    if (!scaling_layer_element) return;

    auto parse_field = [&](const string& field, float Descriptives::* member)
    {
        if (!scaling_layer_element->has(field)) return;
        VectorR values;
        string_to_vector(read_json_string(scaling_layer_element, field), values);
        if (values.size() != ssize(descriptives))
            throw runtime_error("Scaling::read_JSON_body: field \"" + field
                                + "\" has size " + to_string(values.size())
                                + ", expected " + to_string(descriptives.size()) + ".");
        for (Index i = 0; i < values.size(); ++i)
            descriptives[size_t(i)].*member = values(i);
    };

    parse_field("Minimums",           &Descriptives::minimum);
    parse_field("Maximums",           &Descriptives::maximum);
    parse_field("Means",              &Descriptives::mean);
    parse_field("StandardDeviations", &Descriptives::standard_deviation);

    if (scaling_layer_element->has("Scalers"))
    {
        const vector<string> tokens = get_tokens(
            read_json_string(scaling_layer_element, "Scalers"), " ");
        if (ssize(tokens) != ssize(scalers))
            throw runtime_error("Scaling::read_JSON_body: \"Scalers\" has "
                                + to_string(tokens.size()) + " entries, expected "
                                + to_string(scalers.size()) + ".");
        ranges::transform(tokens, scalers.begin(), string_to_scaler_method);
    }

    if (scaling_layer_element->has("MinRange"))
        min_range = float(stof(read_json_string(scaling_layer_element, "MinRange")));
    if (scaling_layer_element->has("MaxRange"))
        max_range = float(stof(read_json_string(scaling_layer_element, "MaxRange")));

    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Scaling::write_JSON_body(JsonWriter& printer) const
{
    const Index features = ssize(descriptives);

    VectorR mins(features), maxs(features), mns(features), stds(features);
    for (Index i = 0; i < features; ++i)
    {
        mins(i) = descriptives[size_t(i)].minimum;
        maxs(i) = descriptives[size_t(i)].maximum;
        mns(i)  = descriptives[size_t(i)].mean;
        stds(i) = descriptives[size_t(i)].standard_deviation;
    }

    vector<string> scaler_names(scalers.size());
    ranges::transform(scalers, scaler_names.begin(), scaler_method_to_string);

    write_json(printer, {
        {"Means",              vector_to_string(mns)},
        {"StandardDeviations", vector_to_string(stds)},
        {"Minimums",           vector_to_string(mins)},
        {"Maximums",           vector_to_string(maxs)},
        {"Scalers",            vector_to_string(scaler_names)},
        {"MinRange",           to_string(min_range)},
        {"MaxRange",           to_string(max_range)}
    });
}

string Scaling::write_expression(const vector<string>& input_names,
                                 const vector<string>& /*output_names*/) const
{
    const Index outputs_number = get_outputs_number();
    if (outputs_number == 0 || ssize(scalers) != outputs_number)
        throw runtime_error("Scaling::write_expression: layer not configured.");

    ostringstream buffer;
    buffer.precision(10);

    for (Index i = 0; i < outputs_number; ++i)
    {
        const Descriptives& d = descriptives[size_t(i)];
        switch (scalers[size_t(i)])
        {
        case ScalerMethod::None:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << ";\n";
            break;
        case ScalerMethod::MinimumMaximum:
            buffer << "scaled_" << input_names[i]
                   << " = " << input_names[i] << "*(" << max_range << "-" << min_range << ")/("
                   << d.maximum << "-(" << d.minimum << "))-" << d.minimum << "*("
                   << max_range << "-" << min_range << ")/("
                   << d.maximum << "-" << d.minimum << ")+" << min_range << ";\n";
            break;
        case ScalerMethod::MeanStandardDeviation:
            buffer << "scaled_" << input_names[i] << " = (" << input_names[i] << "-"
                   << d.mean << ")/" << d.standard_deviation << ";\n";
            break;
        case ScalerMethod::StandardDeviation:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i]
                   << "/(" << d.standard_deviation << ");\n";
            break;
        case ScalerMethod::Logarithm:
            buffer << "scaled_" << input_names[i] << " = log(" << input_names[i] << ");\n";
            break;
        case ScalerMethod::ImageMinMax:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << " / 255.0;\n";
            break;
        default:
            throw runtime_error("Unknown inputs scaling method.\n");
        }
    }

    string expression = buffer.str();
    expression = regex_replace(expression, regex("\\+-"), "-");
    expression = regex_replace(expression, regex("--"), "+");

    return expression;
}

REGISTER(Layer, Scaling, "Scaling")

}
