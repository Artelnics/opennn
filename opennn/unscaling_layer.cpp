//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R    C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "string_utilities.h"
#include "tensor_utilities.h"
#include "math_utilities.h"
#include "neural_network.h"
#include "unscaling_layer.h"
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

VectorR Unscaling::get_minimums()            const { return descriptives_field(descriptives, &Descriptives::minimum); }
VectorR Unscaling::get_maximums()            const { return descriptives_field(descriptives, &Descriptives::maximum); }
VectorR Unscaling::get_means()               const { return descriptives_field(descriptives, &Descriptives::mean); }
VectorR Unscaling::get_standard_deviations() const { return descriptives_field(descriptives, &Descriptives::standard_deviation); }

Unscaling::Unscaling(const Shape& new_input_shape, const string& new_label)
    : Layer("Unscaling", LayerType::Unscaling, false)
{
    operators = {&unscale_op};
    set(new_input_shape.dim_or_zero(0), new_label);
}

void Unscaling::set(Index new_neurons_number, const string& new_label)
{
    set_label(new_label);

    descriptives.assign(size_t(new_neurons_number), Descriptives(-1.0f, 1.0f, 0.0f, 1.0f));
    scalers.assign(size_t(new_neurons_number), ScalerMethod::MinimumMaximum);
    min_range = -1.0f;
    max_range = 1.0f;
    op_storage_dirty = true;
}

void Unscaling::set_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape.dim_or_zero(0));
}

void Unscaling::set_output_shape(const Shape& /*new_output_shape*/)
{
}

void Unscaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    if (ssize(new_descriptives) != ssize(descriptives))
        throw runtime_error("Unscaling::set_descriptives: size mismatch (expected "
                            + to_string(descriptives.size()) + ", got "
                            + to_string(new_descriptives.size()) + ").");
    descriptives = new_descriptives;
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Unscaling::set_min_max_range(float new_min, float new_max)
{
    min_range = new_min;
    max_range = new_max;
    unscale_op.min_range = new_min;
    unscale_op.max_range = new_max;
}

void Unscaling::set_scalers(const vector<string>& scalers_str)
{
    if (ssize(scalers_str) != ssize(scalers))
        throw runtime_error("Unscaling::set_scalers: size mismatch (expected "
                            + to_string(scalers.size()) + ", got "
                            + to_string(scalers_str.size()) + ").");
    ranges::transform(scalers_str, scalers.begin(), string_to_scaler_method);
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Unscaling::set_scalers(const string& scaler)
{
    const ScalerMethod method = string_to_scaler_method(scaler);
    ranges::fill(scalers, method);
    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

float* Unscaling::link_states(float* pointer)
{
    refresh_op_storage(current_device());
    return pointer;
}

void Unscaling::refresh_op_storage(Device device)
{
    const Index features = ssize(descriptives);
    const Index bytes    = 5 * features * Index(sizeof(float));

    const bool needs = op_storage_dirty
                    || op_storage.bytes       != bytes
                    || op_storage.device_type != device;
    if (!needs) return;

    op_storage.resize_bytes(bytes, device);
    unscale_op.min_range = min_range;
    unscale_op.max_range = max_range;

    if (features == 0)
    {
        unscale_op.minimums = unscale_op.maximums = unscale_op.means =
            unscale_op.standard_deviations = unscale_op.scalers = TensorView();
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
    unscale_op.minimums            = TensorView(base + 0 * features, shape, Type::FP32);
    unscale_op.maximums            = TensorView(base + 1 * features, shape, Type::FP32);
    unscale_op.means               = TensorView(base + 2 * features, shape, Type::FP32);
    unscale_op.standard_deviations = TensorView(base + 3 * features, shape, Type::FP32);
    unscale_op.scalers             = TensorView(base + 4 * features, shape, Type::FP32);

    op_storage_dirty = false;
}

void Unscaling::read_JSON_body(const Json* root_element)
{
    if (!root_element) return;

    if (root_element->has("MinRange"))
        min_range = float(stof(read_json_string(root_element, "MinRange")));
    if (root_element->has("MaxRange"))
        max_range = float(stof(read_json_string(root_element, "MaxRange")));

    const Json* neurons_array = root_element->find("Neurons");
    if (!neurons_array || !neurons_array->is_array()) return;

    if (ssize(neurons_array->array_value) != ssize(scalers))
        throw runtime_error("Unscaling::read_JSON_body: \"Neurons\" has "
                            + to_string(neurons_array->array_value.size())
                            + " entries, expected " + to_string(scalers.size()) + ".");

    for (size_t i = 0; i < neurons_array->array_value.size(); ++i)
    {
        const Json* neuron = &neurons_array->array_value[i];

        scalers[i] = string_to_scaler_method(read_json_string(neuron, "Scaler"));

        const string descriptives_text = read_json_string(neuron, "Descriptives");
        const vector<string> tokens = get_tokens(descriptives_text, " ");
        if (tokens.size() < 4)
            throw runtime_error("Unscaling::read_JSON_body: neuron " + to_string(i)
                                + " \"Descriptives\" has " + to_string(tokens.size())
                                + " tokens, expected 4.");
        descriptives[i].minimum            = float(stof(tokens[0]));
        descriptives[i].maximum            = float(stof(tokens[1]));
        descriptives[i].mean               = float(stof(tokens[2]));
        descriptives[i].standard_deviation = float(stof(tokens[3]));
    }

    op_storage_dirty = true;
    refresh_op_storage(current_device());
}

void Unscaling::write_JSON_body(JsonWriter& printer) const
{
    add_json_field(printer, "MinRange", to_string(min_range));
    add_json_field(printer, "MaxRange", to_string(max_range));

    const Index features = ssize(descriptives);

    printer.begin_array("Neurons");
    for (Index i = 0; i < features; ++i)
    {
        printer.begin_array_object();

        ostringstream descriptives_stream;
        descriptives_stream.precision(10);
        descriptives_stream << descriptives[size_t(i)].minimum << ' '
                            << descriptives[size_t(i)].maximum << ' '
                            << descriptives[size_t(i)].mean    << ' '
                            << descriptives[size_t(i)].standard_deviation;

        add_json_field(printer, "Descriptives", descriptives_stream.str());
        add_json_field(printer, "Scaler", scaler_method_to_string(scalers[size_t(i)]));

        printer.end_array_object();
    }
    printer.end_array();
}

string Unscaling::write_expression(const vector<string>& input_names,
                                   const vector<string>& output_names) const
{
    const Index outputs_number = get_outputs_number();
    if (outputs_number == 0 || ssize(scalers) != outputs_number)
        throw runtime_error("Unscaling::write_expression: layer not configured.");

    ostringstream buffer;
    buffer.precision(10);

    for (Index i = 0; i < outputs_number; ++i)
    {
        const Descriptives& d = descriptives[size_t(i)];
        using enum ScalerMethod;
        switch (scalers[size_t(i)])
        {
        case None:
            buffer << output_names[i] << " = " << input_names[i] << ";\n";
            break;
        case MinimumMaximum:
            if (abs(d.minimum - d.maximum) < EPSILON)
                buffer << output_names[i] << "=" << d.minimum << ";\n";
            else
                buffer << output_names[i] << "=" << input_names[i] << "*"
                       << "(" << (d.maximum - d.minimum) / (max_range - min_range)
                       << ")+" << (d.minimum - min_range * (d.maximum - d.minimum) / (max_range - min_range)) << ";\n";
            break;
        case MeanStandardDeviation:
            buffer << output_names[i] << "=" << input_names[i] << "*" << d.standard_deviation << "+" << d.mean << ";\n";
            break;
        case StandardDeviation:
            buffer << output_names[i] << "=" << input_names[i] << "*" << d.standard_deviation << ";\n";
            break;
        case Logarithm:
            buffer << output_names[i] << "=" << "exp(" << input_names[i] << ");\n";
            break;
        case ImageMinMax:
            buffer << output_names[i] << "=" << input_names[i] << " * 255.0;\n";
            break;
        default:
            throw runtime_error("Unknown inputs scaling method.\n");
        }
    }

    string expression = buffer.str();
    replace(expression, "+-", "-");
    replace(expression, "--", "+");

    return expression;
}

REGISTER(Layer, Unscaling, "Unscaling")

}
