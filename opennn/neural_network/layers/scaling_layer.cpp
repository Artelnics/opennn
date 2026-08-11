//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/device_backend.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/core/scaling.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/json.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "opennn/neural_network/layers/kernel_scaling.cuh"
#endif

namespace opennn
{

// Defined below: against the CUDA kernels, or as a throwing stub. Both directions
// share one entry point so the forward and inverse paths cannot drift apart.
static void scale_gpu(const TensorView&, const TensorView&, const TensorView&,
                      const TensorView&, const TensorView&, const TensorView&,
                      float, float, TensorView&, bool);

template<typename Column>
static void scale_column_cpu(Column& column, ScalerMethod method,
                             const Descriptives& descriptives,
                             float min_range, float max_range)
{
    using enum ScalerMethod;

    switch (method)
    {
    case MinimumMaximum:
        if (descriptives.maximum - descriptives.minimum < EPSILON)
            column.setZero();
        else
            column = scale_minimum_maximum_formula(column, descriptives, min_range, max_range);
        break;
    case MeanStandardDeviation:
        if (descriptives.standard_deviation > EPSILON)
            column = scale_mean_standard_deviation_formula(column, descriptives);
        else
            column.setZero();
        break;
    case StandardDeviation:
        column *= descriptives.standard_deviation > EPSILON
                ? 1.0f / descriptives.standard_deviation
                : 0.0f;
        break;
    case Logarithm:
        column = column.max(EPSILON).log();
        break;
    case ImageMinMax:
        column /= 255.0f;
        break;
    case None:
    default:
        break;
    }
}

template<typename Column>
static void unscale_column_cpu(Column& column, ScalerMethod method,
                               const Descriptives& descriptives,
                               float min_range, float max_range)
{
    using enum ScalerMethod;

    switch (method)
    {
    case MinimumMaximum:
        throw_if(max_range - min_range < EPSILON, "The range values are not valid.");
        // Constant feature: the forward scaling produced zeros, so invert to the constant.
        if (descriptives.maximum - descriptives.minimum < EPSILON)
            column.setConstant(descriptives.minimum);
        else
            column = unscale_minimum_maximum_formula(column, descriptives, min_range, max_range);
        break;
    case MeanStandardDeviation:
        column = unscale_mean_standard_deviation_formula(column, descriptives);
        break;
    case StandardDeviation:
        if (descriptives.standard_deviation > EPSILON)
            column *= descriptives.standard_deviation;
        else
            column.setConstant(descriptives.mean);
        break;
    case Logarithm:
        column = column.exp();
        break;
    case ImageMinMax:
        column *= 255.0f;
        break;
    case None:
    default:
        break;
    }
}

static void scale_cpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output, bool inverse)
{
    const Index features = scalers.size();
    if (features == 0) { output.as_matrix().noalias() = input.as_matrix(); return; }

    const MatrixMap input_matrix = input.as_flat_matrix();
    const VectorMap minimums_vector = minimums.as_vector();
    const VectorMap maximums_vector = maximums.as_vector();
    const VectorMap means_vector  = means.as_vector();
    const VectorMap standard_deviations_vector  = standard_deviations.as_vector();
    const VectorMap scalers_vector   = scalers.as_vector();

    MatrixMap output_matrix = output.as_flat_matrix();

    output_matrix.noalias() = input_matrix;

    const Index cols = output_matrix.cols();
    for (Index col = 0; col < cols; ++col)
    {
        const Index feature_index = col % features;
        const auto method = static_cast<ScalerMethod>(static_cast<int>(scalers_vector(feature_index)));
        auto column = output_matrix.col(col).array();

        const Descriptives descriptives(minimums_vector(feature_index),
                                        maximums_vector(feature_index),
                                        means_vector(feature_index),
                                        standard_deviations_vector(feature_index));

        if (inverse)
            unscale_column_cpu(column, method, descriptives, min_range, max_range);
        else
            scale_column_cpu(column, method, descriptives, min_range, max_range);
    }
}

void scale(const TensorView& input,
           const TensorView& minimums, const TensorView& maximums,
           const TensorView& means, const TensorView& standard_deviations,
           const TensorView& scalers,
           float min_range, float max_range,
           TensorView& output)
{
    if (input.is_cuda())
    {
        scale_gpu(input, minimums, maximums, means, standard_deviations, scalers,
                  min_range, max_range, output, false);
        return;
    }
    scale_cpu(input, minimums, maximums, means, standard_deviations, scalers,
              min_range, max_range, output, false);
}

void unscale(const TensorView& input,
             const TensorView& minimums, const TensorView& maximums,
             const TensorView& means, const TensorView& standard_deviations,
             const TensorView& scalers,
             float min_range, float max_range,
             TensorView& output)
{
    if (input.is_cuda())
    {
        scale_gpu(input, minimums, maximums, means, standard_deviations, scalers,
                  min_range, max_range, output, true);
        return;
    }

    scale_cpu(input, minimums, maximums, means, standard_deviations, scalers,
              min_range, max_range, output, true);
}

#ifdef OPENNN_HAS_CUDA

static void scale_gpu(const TensorView& input,
               const TensorView& minimums, const TensorView& maximums,
               const TensorView& means, const TensorView& standard_deviations,
               const TensorView& scalers,
               float min_range, float max_range,
               TensorView& output, bool inverse)
{
    const Index features = scalers.size();

    visit_type_pair<Type::FP32, Type::BF16>(input.type, output.type, [&]<typename TIn, typename TOut>() {
        if (inverse)
        {
            unscale_cuda<TIn, TOut>(output.size(), to_int(features),
                                    input.as<TIn>(),
                                    minimums.as_float(),
                                    maximums.as_float(),
                                    means.as_float(),
                                    standard_deviations.as_float(),
                                    scalers.as_float(),
                                    min_range, max_range,
                                    output.as<TOut>());
            return;
        }
        scale_cuda<TIn, TOut>(output.size(), to_int(features),
                              input.as<TIn>(),
                              minimums.as_float(),
                              maximums.as_float(),
                              means.as_float(),
                              standard_deviations.as_float(),
                              scalers.as_float(),
                              min_range, max_range,
                              output.as<TOut>());
    });
}

#else

static void scale_gpu(const TensorView&, const TensorView&, const TensorView&,
                      const TensorView&, const TensorView&, const TensorView&,
                      float, float, TensorView&, bool)
{
    throw runtime_error("scale_gpu: CUDA support not compiled in.");
}

#endif

void ScaleOperator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input = get_input(forward_propagation, layer);
    TensorView& output      = get_output(forward_propagation, layer);

    if (!minimums.data)
    {
        copy(input, output);
        return;
    }

    if (invert)
        unscale(input, minimums, maximums, means, standard_deviations, scalers,
                min_range, max_range, output);
    else
        scale(input, minimums, maximums, means, standard_deviations, scalers,
              min_range, max_range, output);
}

Scaling::Scaling(const Shape& new_input_shape)
    : Scaling(LayerType::Scaling)
{
    set(new_input_shape);
}

Scaling::Scaling(LayerType layer_type)
    : Layer(layer_type, false)
{
    operators = {&scale_op};
}

VectorR Scaling::get_minimums()            const { return descriptives_field(descriptives, &Descriptives::minimum); }
VectorR Scaling::get_maximums()            const { return descriptives_field(descriptives, &Descriptives::maximum); }
VectorR Scaling::get_means()               const { return descriptives_field(descriptives, &Descriptives::mean); }
VectorR Scaling::get_standard_deviations() const { return descriptives_field(descriptives, &Descriptives::standard_deviation); }

void Scaling::set(const Shape& new_input_shape)
{
    input_shape = new_input_shape;

    set_label("scaling_layer");

    const Index features = input_shape.empty() ? 0 : input_shape.back();
    descriptives.assign(size_t(features), Descriptives(-1.0f, 1.0f, 0.0f, 1.0f));
    scalers.assign(size_t(features), ScalerMethod::MeanStandardDeviation);
    min_range = -1.0f;
    max_range = 1.0f;
    op_storage_dirty = true;

    check_rank(input_shape, {1, 2, 3}, "Scaling", "input");
}

void Scaling::apply_input_shape(const Shape& new_input_shape)
{
    set(new_input_shape);
}

void Scaling::set_descriptives(const vector<Descriptives>& new_descriptives)
{
    throw_if(ssize(new_descriptives) != ssize(descriptives),
             "{}::set_descriptives: size mismatch (expected {}, got {}).",
                    get_name(), descriptives.size(), new_descriptives.size());
    descriptives = new_descriptives;
    op_storage_dirty = true;
    refresh_op_storage(op_storage.device_type);
}

void Scaling::set_scalers(const vector<string>& scalers_str)
{
    throw_if(ssize(scalers_str) != ssize(scalers),
             "{}::set_scalers: size mismatch (expected {}, got {}).",
                    get_name(), scalers.size(), scalers_str.size());
    ranges::transform(scalers_str, scalers.begin(), string_to_scaler_method);
    op_storage_dirty = true;
    refresh_op_storage(op_storage.device_type);
}

void Scaling::set_scalers(const string& scaler)
{
    const ScalerMethod method = string_to_scaler_method(scaler);
    ranges::fill(scalers, method);
    op_storage_dirty = true;
    refresh_op_storage(op_storage.device_type);
}

bool Scaling::is_passthrough() const
{
    return ranges::all_of(scalers, [](ScalerMethod m) { return m == ScalerMethod::None; });
}

vector<TensorSpec> Scaling::get_forward_specs(Index batch_size) const
{
    if (is_passthrough())
        return {};
    return Layer::get_forward_specs(batch_size);
}

void Scaling::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool is_training)
{
    if (is_passthrough())
        return;
    Layer::forward_propagate(forward_propagation, layer, is_training);
}

float* Scaling::link_states(float* pointer, Device device)
{
    refresh_op_storage(device);
    return pointer;
}

void Scaling::refresh_op_storage(Device device)
{
    const Index features = ssize(descriptives);

    if (!refresh_feature_storage(op_storage, op_storage_dirty, device, features, 5,
            [&](float* staging)
            {
                for (Index i = 0; i < features; ++i)
                {
                    staging[size_t(0 * features + i)] = descriptives[size_t(i)].minimum;
                    staging[size_t(1 * features + i)] = descriptives[size_t(i)].maximum;
                    staging[size_t(2 * features + i)] = descriptives[size_t(i)].mean;
                    staging[size_t(3 * features + i)] = descriptives[size_t(i)].standard_deviation;
                    staging[size_t(4 * features + i)] = float(int(scalers[size_t(i)]));
                }
            }))
        return;

    scale_op.min_range = min_range;
    scale_op.max_range = max_range;

    if (features == 0)
    {
        scale_op.minimums = scale_op.maximums = scale_op.means =
            scale_op.standard_deviations = scale_op.scalers = TensorView();
        return;
    }

    float* const base = op_storage.as<float>();
    const Shape shape{features};
    scale_op.minimums            = TensorView(base, shape, Type::FP32, device);
    scale_op.maximums            = TensorView(base + 1 * features, shape, Type::FP32, device);
    scale_op.means               = TensorView(base + 2 * features, shape, Type::FP32, device);
    scale_op.standard_deviations = TensorView(base + 3 * features, shape, Type::FP32, device);
    scale_op.scalers             = TensorView(base + 4 * features, shape, Type::FP32, device);
}

void Scaling::read_JSON_body(const Json* scaling_layer_element)
{
    if (!scaling_layer_element) return;

    const auto parse_field = [&](const string& field, float Descriptives::* member)
    {
        if (!scaling_layer_element->has(field)) return;
        VectorR values;
        string_to_vector(read_json_string(scaling_layer_element, field), values);
        throw_if(values.size() != ssize(descriptives),
                 "Scaling::read_JSON_body: field \"{}\" has size {}, expected {}.",
                        field, values.size(), descriptives.size());
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
        throw_if(ssize(tokens) != ssize(scalers),
                 "Scaling::read_JSON_body: \"Scalers\" has {} entries, expected {}.",
                        tokens.size(), scalers.size());
        ranges::transform(tokens, scalers.begin(), string_to_scaler_method);
    }

    if (scaling_layer_element->has("MinRange"))
        min_range = parse_float(read_json_string(scaling_layer_element, "MinRange"), "Scaling: MinRange");
    if (scaling_layer_element->has("MaxRange"))
        max_range = parse_float(read_json_string(scaling_layer_element, "MaxRange"), "Scaling: MaxRange");

    op_storage_dirty = true;
    refresh_op_storage(op_storage.device_type);
}

void Scaling::write_JSON_body(JsonWriter& printer) const
{
    vector<string> scaler_names(scalers.size());
    ranges::transform(scalers, scaler_names.begin(), scaler_method_to_string);

    write_json(printer, {
        {"Means",              vector_to_string(get_means())},
        {"StandardDeviations", vector_to_string(get_standard_deviations())},
        {"Minimums",           vector_to_string(get_minimums())},
        {"Maximums",           vector_to_string(get_maximums())},
        {"Scalers",            vector_to_string(scaler_names)},
        {"MinRange",           min_range},
        {"MaxRange",           max_range}
    });
}

namespace
{

// Emitted numbers always carry a decimal point, and constants are folded before
// they are written rather than left as arithmetic over literals.
//
// The expression body is shared by every target language, so the same text has
// to mean the same thing in all of them. It did not: this layer used to emit
// the min-max offset as "-2*(1+1)/(6+2)", which Python and JavaScript evaluate
// to 0.5 but C evaluates as integer division to 0, silently shifting every
// exported C model whose inputs were min-max scaled.
string expression_literal(float value)
{
    ostringstream stream;
    stream.precision(10);
    stream << value;

    string text = stream.str();

    if (text.find_first_of(".eE") == string::npos)
        text += ".0";

    return text;
}

// Folded through scaling_affine, the same map the numeric paths use, so the
// exported model cannot drift away from what the layer computes.
string affine_line(const string& input_name, ScalerMethod scaler,
                   const Descriptives& descriptives, float min_range, float max_range)
{
    const auto [scale, offset] = scaling_affine(scaler, descriptives, min_range, max_range);

    return "scaled_" + input_name + " = " + input_name
         + "*" + expression_literal(scale)
         + "+" + expression_literal(offset) + ";\n";
}

}

string Scaling::write_expression(const vector<string>& input_names,
                                 const vector<string>&) const
{
    const Index outputs_number = get_outputs_number();
    throw_if(outputs_number == 0 || ssize(scalers) == 0
             || outputs_number % ssize(scalers) != 0,
             "Scaling::write_expression: layer not configured.");

    ostringstream buffer;
    buffer.precision(10);

    for (Index i = 0; i < outputs_number; ++i)
    {
        const size_t feature = size_t(i % ssize(scalers));
        const Descriptives& d = descriptives[feature];
        using enum ScalerMethod;
        switch (scalers[feature])
        {
        case None:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << ";\n";
            break;

        case MinimumMaximum:
            if (d.maximum - d.minimum < EPSILON)
                buffer << "scaled_" << input_names[i] << " = 0;\n";
            else
                buffer << affine_line(input_names[i], MinimumMaximum, d, min_range, max_range);
            break;
        case MeanStandardDeviation:
            if (d.standard_deviation > EPSILON)
                buffer << affine_line(input_names[i], MeanStandardDeviation, d, min_range, max_range);
            else
                buffer << "scaled_" << input_names[i] << " = 0;\n";
            break;
        case StandardDeviation:
            if (d.standard_deviation > EPSILON)
                buffer << affine_line(input_names[i], StandardDeviation, d, min_range, max_range);
            else
                buffer << "scaled_" << input_names[i] << " = 0;\n";
            break;
        case Logarithm:
            buffer << "scaled_" << input_names[i] << " = log(" << input_names[i] << ");\n";
            break;
        case ImageMinMax:
            buffer << "scaled_" << input_names[i] << " = " << input_names[i] << " / 255.0;\n";
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

}
