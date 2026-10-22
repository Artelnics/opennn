// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/response_optimization/feasibility_repair_system.h"

#include <Eigen/QR>
#include <unsupported/Eigen/LevenbergMarquardt>

#include "opennn/network/network.h"
#include "opennn/network/layers/clamping_layer.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/layer_registry.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/core/tensor_operations.h"

namespace opennn
{

namespace
{

// The network reads points in float, whatever the precision of the repair.
template<typename Scalar>
decltype(auto) as_float(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& values)
{
    if constexpr (is_same_v<Scalar, float>) return (values);
    else return VectorR(values.template cast<float>());
}


// The network's float outputs, as the rows of a repair in Scalar read them.
template<typename Scalar>
decltype(auto) as_scalar(const VectorR& values)
{
    if constexpr (is_same_v<Scalar, float>) return (values);
    else return Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(values.template cast<Scalar>());
}


float evaluate_row(const CompiledExpression& expression, const VectorR& point, const VectorR& output)
{
    return expression.evaluate(point, output);
}


double evaluate_row(const CompiledExpression& expression, const Eigen::VectorXd& point, const Eigen::VectorXd& output)
{
    return expression.evaluate_double(point, output);
}

}


FeasibilityRepairSystem::FeasibilityRepairSystem(Network* new_network, span<const Constraint> new_constraints,
                                               pair<VectorR, VectorR> domain, const Index rounds,
                                               const Index evaluations, const float tolerance, const float margin,
                                               const float new_constraint_tolerance)
    : network(new_network), constraints(new_constraints), feasibility_rounds(rounds),
      feasibility_evaluations(evaluations), numeric_tolerance(tolerance), feasibility_margin_factor(margin),
      constraint_tolerance(new_constraint_tolerance)
{
    throw_if(!network, "Feasibility repair requires a network.");
    throw_if(rounds < 0 || evaluations < 1 || !isfinite(tolerance) || tolerance <= 0.0f
             || !isfinite(margin) || margin < 0.0f
             || !isfinite(constraint_tolerance) || constraint_tolerance <= 0.0f, "Invalid feasibility repair settings.");
    for (const Constraint& constraint : constraints)
    {
        constraint.validate();
        const auto [lower, upper] = constraint.get_bounds(numeric_tolerance, constraint_tolerance);
        throw_if(!(lower <= upper) || lower == numeric_limits<float>::infinity()
                 || upper == -numeric_limits<float>::infinity(), "Invalid feasibility constraint bounds.");
        throw_if(constraint.condition == Condition::Cardinality
                 && constraint.equation.symmetric_order != Index(constraint.values[0]) + 1,
                 "Cardinality repair requires an equation compiled for its budget.");
    }
    categorical_blocks = get_categorical_blocks(network->get_input_variables());
    constraints_read_output = ranges::any_of(constraints, [](const Constraint& constraint)
        { return is_output_coupled(constraint.equation); });
    set_domain(move(domain));
}


void FeasibilityRepairSystem::set_domain(pair<VectorR, VectorR> domain)
{
    throw_if(domain.first.size() == 0 || domain.first.size() != network->get_inputs_number()
             || domain.second.size() != domain.first.size()
             || !domain.first.allFinite() || !domain.second.allFinite(),
             "Feasibility repair requires finite input bounds matching the network.");
    throw_if(!(domain.second - domain.first).allFinite() || (domain.first.array() > domain.second.array()).any(),
             "Feasibility repair domain must have finite, ordered bounds.");
    input_bounds = move(domain);
    double_bounds = {input_bounds.first.cast<double>(), input_bounds.second.cast<double>()};
}


void FeasibilityRepairSystem::set_solver(const Solver new_solver)
{
    throw_if(!is_one_of(new_solver, Solver::LevenbergMarquardt, Solver::GaussNewton),
             "Invalid feasibility repair solver.");
    solver = new_solver;
}


template<typename Scalar>
FeasibilityRepairSystem::Workspace<Scalar>& FeasibilityRepairSystem::get_workspace()
{
    if constexpr (is_same_v<Scalar, float>) return float_workspace;
    else return double_workspace;
}


template<typename Scalar>
const FeasibilityRepairSystem::Workspace<Scalar>& FeasibilityRepairSystem::get_workspace() const
{
    if constexpr (is_same_v<Scalar, float>) return float_workspace;
    else return double_workspace;
}


template<typename Scalar>
const pair<FeasibilityRepairSystem::Vector<Scalar>, FeasibilityRepairSystem::Vector<Scalar>>&
FeasibilityRepairSystem::get_bounds() const
{
    if constexpr (is_same_v<Scalar, float>) return input_bounds;
    else return double_bounds;
}


float FeasibilityRepairSystem::get_bound_tolerance(const float bound) const
{
    return max(EPSILON, abs(bound)*numeric_tolerance);
}


template<typename Scalar>
Scalar FeasibilityRepairSystem::calculate_constraint_value(const Index i) const
{
    const Constraint& constraint = constraints[size_t(i)];
    const Scalar value = get_workspace<Scalar>().row_values(i);
    if (constraint.condition == Condition::Integer) return value - round(value);
    if (constraint.condition == Condition::AllowedSet)
        return value - Scalar(*ranges::min_element(constraint.values, {},
            [&](const float allowed) { return abs(double(value) - allowed); }));
    return value;
}


template<typename Scalar>
pair<FeasibilityRepairSystem::Vector<Scalar>, VectorR> FeasibilityRepairSystem::repair(Vector<Scalar> point)
{
    const auto& [lower_bounds, upper_bounds] = get_bounds<Scalar>();
    Workspace<Scalar>& work = get_workspace<Scalar>();

    throw_if(point.size() != lower_bounds.size(), "Feasibility repair point has the wrong size.");
    work.evaluated_point.resize(0);
    work.evaluated_output.resize(0);

    Vector<Scalar> previous;
    Vector<Scalar> before_previous;

    for (Index i = 0; ; i++)
    {
        if (!point.allFinite()) return {};

        if (!round_discrete<Scalar>(point) || !point.allFinite()) return {};

        evaluate_constraints<Scalar>(point);

        if (!work.row_values.allFinite() || !work.row_residuals.allFinite()) return {};

        if ((work.row_residuals.array() == Scalar(0)).all())
        {
            if (!evaluate_outputs<Scalar>(point).allFinite()) return {};
            return {move(point), move(work.evaluated_output)};
        }

        update_closest_failure<Scalar>();

        if (i == feasibility_rounds) return {};

        if (previous.size() == point.size()
         && ((point - previous).cwiseAbs().array()
             <= Scalar(numeric_tolerance)*(upper_bounds - lower_bounds).cwiseMax(Scalar(EPSILON)).array()).all())
            return {};

        if (before_previous.size() == point.size() && (before_previous.array() == point.array()).all())
            return {};

        before_previous.swap(previous);
        previous = point;

        if (i == 0) work.box_scales = (upper_bounds - lower_bounds).cwiseMax(Scalar(EPSILON))/Scalar(box_weight);
        Matrix<Scalar> initial_jacobian = calculate_jacobian<Scalar>(point, work.row_values, work.evaluated_output);
        work.row_scales = initial_jacobian.rowwise().stableNorm();

        work.row_scales = (work.row_scales.array().isFinite() && (work.row_scales.array() > Scalar(EPSILON)))
                          .select(work.row_scales.array(), Scalar(1));

        if (solver == Solver::GaussNewton) solve_gauss_newton<Scalar>(point, move(initial_jacobian));
        else solve_levenberg_marquardt<Scalar>(point, move(initial_jacobian));
    }
}


template<typename Scalar>
bool FeasibilityRepairSystem::round_discrete(Vector<Scalar>& point) const
{
    const auto& bounds = get_bounds<Scalar>();
    const auto& [lower_bounds, upper_bounds] = bounds;

    const auto always_on = [&bounds](const Index column)
        { return bounds.first(column) > Scalar(0) || bounds.second(column) < Scalar(0); };


    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition != Condition::Cardinality) continue;

        vector<pair<Index, double>> members = constraint.equation.symmetric_terms;

        ranges::stable_sort(members, {}, [&](const pair<Index, double>& member)
            { return pair(!always_on(member.first), -abs(double(point(member.first)))*member.second); });

        const size_t budget = size_t(constraint.values[0]);

        if (budget < members.size() && always_on(members[budget].first)) return false;

        for (size_t j = budget; j < members.size(); j++)
            point(members[j].first) = Scalar(0);
    }

    point = point.cwiseMax(lower_bounds).cwiseMin(upper_bounds);

    for (const auto& [first_column, categories_number] : categorical_blocks)
    {
        Index category = 0;

        const bool none_open =
            (upper_bounds.segment(first_column, categories_number).array() > Scalar(0))
            .select(point.segment(first_column, categories_number).array(), Scalar(-MAX)).maxCoeff(&category)
            == Scalar(-MAX);

        if (none_open) return false;

        point.segment(first_column, categories_number).setZero();

        point(first_column + category) = Scalar(1);
    }

    for (const Constraint& constraint : constraints)
        if (const Index column = constraint.get_discrete_column(); column >= 0)
        {
            const Scalar value = point(column);

            const Scalar lower = lower_bounds(column);
            const Scalar upper = upper_bounds(column);

            if (constraint.condition == Condition::Integer && ceil(lower) > floor(upper)) return false;

            point(column) = constraint.condition == Condition::Integer
                          ? clamp(round(value), ceil(lower), floor(upper))
                          : Scalar(*ranges::min_element(constraint.values, {}, [&](const float allowed)
                                { return (allowed < lower || allowed > upper)
                                       ? numeric_limits<double>::infinity() : abs(double(allowed) - value); }));
        }

    if ((point.array() < lower_bounds.array()).any() || (point.array() > upper_bounds.array()).any()) return false;

    for (const auto& [first, size] : categorical_blocks)
    {
        const auto block = point.segment(first, size).array();
        if (!(block == Scalar(0) || block == Scalar(1)).all() || block.sum() != Scalar(1)) return false;
    }

    return true;
}


template<typename Scalar>
const VectorR& FeasibilityRepairSystem::evaluate_outputs(const Vector<Scalar>& point)
{
    Workspace<Scalar>& work = get_workspace<Scalar>();

    if (work.evaluated_output.size() == 0 || work.evaluated_point.size() != point.size()
     || (work.evaluated_point.array() != point.array()).any())
    {
        work.evaluated_output = network->calculate_outputs(as_float(point).transpose()).row(0).transpose();
        work.evaluated_point = point;
    }

    return work.evaluated_output;
}


template<typename Scalar>
void FeasibilityRepairSystem::evaluate_constraints(const Vector<Scalar>& point)
{
    Workspace<Scalar>& work = get_workspace<Scalar>();

    if (constraints_read_output) evaluate_outputs<Scalar>(point);

    const Index rows_number = Index(constraints.size());

    work.row_values.resize(rows_number);
    work.row_residuals.resize(rows_number);

    const auto& output = as_scalar<Scalar>(work.evaluated_output);

    for (Index i = 0; i < rows_number; i++)
    {
        const Constraint& constraint = constraints[size_t(i)];

        work.row_values(i) = evaluate_row(constraint.equation, point, output);

        work.row_residuals(i) = calculate_constraint_residual<Scalar>(i).first;
    }
}


template<typename Scalar>
void FeasibilityRepairSystem::update_closest_failure()
{
    const Workspace<Scalar>& work = get_workspace<Scalar>();

    Index worst = -1;
    double worst_ratio = 0.0;
    double worst_gap = 0.0;
    double worst_tolerance = 0.0;
    float worst_bound = 0.0f;

    for (Index i = 0; i < work.row_values.size(); i++)
    {
        if (work.row_residuals(i) == Scalar(0)) continue;
        const Constraint& constraint = constraints[size_t(i)];
        const auto [lower, upper] = constraint.get_bounds(numeric_tolerance, constraint_tolerance);
        const float bound = work.row_residuals(i) < Scalar(0) ? lower : upper;
        const bool discrete = is_one_of(constraint.condition, Condition::Integer, Condition::AllowedSet);
        const double gap = discrete ? abs(double(calculate_constraint_value<Scalar>(i)))
                                    : abs(double(work.row_values(i)) - bound);
        const double tolerance = get_bound_tolerance(bound) + (discrete ? double(constraint_tolerance) : 0.0);
        const double ratio = gap/tolerance;
        if (ratio <= worst_ratio) continue;
        worst = i;
        worst_ratio = ratio;
        worst_gap = gap;
        worst_tolerance = tolerance;
        worst_bound = bound;
    }
    if (worst < 0 || worst_ratio >= closest_failure.first) return;

    const Constraint& constraint = constraints[size_t(worst)];
    static constexpr const char* names[] = {"Equal", "Between", "GreaterEqual", "LessEqual", "Greater", "Less",
                                           "AllowedSet", "Integer", "Cardinality"};
    const Scalar value = work.row_values(worst);
    ostringstream message;
    message << setprecision(numeric_limits<Scalar>::max_digits10)
            << "Best checked rounded candidate with a constraint violation (smallest maximum miss/tolerance): constraint '"
            << constraint.equation.text << "' (" << names[size_t(constraint.condition)] << "): "
            << (constraint.condition == Condition::Cardinality ? "cardinality measure " : "value ") << value;
    if (constraint.condition == Condition::Integer) message << ", nearest integer " << round(value);
    else if (constraint.condition == Condition::AllowedSet)
        message << ", nearest allowed value " << *ranges::min_element(constraint.values, {},
            [&](const float allowed) { return abs(double(value) - allowed); });
    else message << ", required " << (constraint.condition == Condition::Equal ? "value " : "bound ") << worst_bound;
    message << ", absolute miss " << worst_gap << " against tolerance " << worst_tolerance
            << " (" << worst_ratio << " times tolerance). ";
    if (constraint.condition == Condition::Equal)
        message << "If exact equality is not required, consider Between. ";
    closest_failure = {worst_ratio, message.str()};
}


static bool calculate_activation_derivatives(const ActivationFunction activation, VectorR& values, VectorR& derivatives)
{
    if (activation == ActivationFunction::Softmax) return false;

    derivatives.resize(values.size());
    for (Index i = 0; i < values.size(); i++)
    {
        const float input = values(i);
        if (input == 0.0f && is_one_of(activation, ActivationFunction::ReLU, ActivationFunction::LeakyReLU))
            return false;
        values(i) = activation_forward_value(activation, input);
        if (activation == ActivationFunction::GELU) derivatives(i) = gelu_derivative(input);
        else if (activation == ActivationFunction::GELUTanh) derivatives(i) = gelu_tanh_derivative(input);
        else if (activation == ActivationFunction::SiLU) derivatives(i) = silu_derivative(input);
        else derivatives(i) = activation_derivative_from_output_value(activation, values(i));
    }
    return true;
}


static const CombinationOperator* get_dense_combination(const Dense& dense)
{
    if (dense.get_gated() || dense.get_batch_normalization() || dense.get_tied_weight().source
     || dense.get_operators().empty()) return nullptr;
    const auto* combination = dynamic_cast<const CombinationOperator*>(dense.get_operators().front());
    if (!combination || combination->transposed_inference_active || combination->tied_transposed) return nullptr;
    const TensorView& weights = combination->weights;
    if (!weights.get_data() || !weights.is_fp32() || weights.get_device() != Device::CPU
     || weights.get_shape() != Shape{dense.get_inputs_number(), dense.get_outputs_number()}) return nullptr;
    const TensorView& bias = combination->bias;
    if (combination->use_bias && (!bias.get_data() || !bias.is_fp32() || bias.get_device() != Device::CPU
                              || bias.size() != dense.get_outputs_number())) return nullptr;
    return combination;
}


static bool calculate_scaling_derivatives(const Scaling& layer, VectorR& values, VectorR& derivatives)
{
    const auto& scalers = layer.get_scalers();
    const auto& descriptives = layer.get_descriptives();
    if (Index(scalers.size()) != values.size() || scalers.size() != descriptives.size()
     || layer.get_operators().empty()) return false;
    const auto* operation = dynamic_cast<const ScaleOperator*>(layer.get_operators().front());
    if (!operation) return false;
    derivatives.resize(values.size());
    for (Index i = 0; i < values.size(); i++)
    {
        const ScalerMethod scaler = scalers[size_t(i)];
        if (scaler == ScalerMethod::Logarithm) return false;
        const Descriptives& descriptive = descriptives[size_t(i)];
        derivatives(i) = (layer.is_inverse() ? unscaling_affine : scaling_affine)
            (scaler, descriptive, layer.get_min_range(), layer.get_max_range()).slope;
        if (layer.is_inverse() && scaler == ScalerMethod::StandardDeviation
         && descriptive.standard_deviation <= EPSILON) derivatives(i) = 0.0f;
    }
    TensorView buffer(values.data(), {1, values.size()}, Type::FP32, Device::CPU);
    (layer.is_inverse() ? unscale : scale)(buffer, operation->minimums, operation->maximums,
        operation->means, operation->standard_deviations, operation->scalers,
        operation->min_range, operation->max_range, buffer);
    return true;
}


static bool calculate_clamping_derivatives(const Clamping& layer, VectorR& values, VectorR& derivatives)
{
    derivatives.setOnes(values.size());
    if (layer.get_clamping_method() == Clamping::ClampingMethod::NoClamping) return true;
    const VectorR lower = layer.get_lower_bounds();
    const VectorR upper = layer.get_upper_bounds();
    if (lower.size() != values.size() || upper.size() != values.size() || !lower.allFinite() || !upper.allFinite()
     || (lower.array() > upper.array()).any())
        return false;
    for (Index i = 0; i < values.size(); i++)
    {
        if (lower(i) != upper(i) && (values(i) == lower(i) || values(i) == upper(i))) return false;
        derivatives(i) = values(i) > lower(i) && values(i) < upper(i) ? 1.0f : 0.0f;
    }
    values = values.cwiseMax(lower).cwiseMin(upper);
    return true;
}


static bool is_sequential_cpu_network(const Network& network)
{
    if (network.get_device() != Device::CPU || network.get_activation_type() != Type::FP32) return false;
    const auto& layers = network.get_layers();
    const auto& sources = network.get_source_layers();
    if (layers.empty() || sources.size() != layers.size()) return false;
    Index inputs = network.get_inputs_number();
    for (size_t i = 0; i < layers.size(); i++)
    {
        const Layer& layer = *layers[i];
        if (sources[i].size() != 1 || sources[i][0] != Index(i) - 1
         || layer.get_input_shape().get_rank() != 1 || layer.get_output_shape().get_rank() != 1
         || layer.get_inputs_number() != inputs
         || layer.get_compute_device() != Device::CPU || layer.get_compute_dtype() != Type::FP32) return false;
        inputs = layer.get_outputs_number();
    }
    return true;
}


MatrixR FeasibilityRepairSystem::calculate_network_jacobian(const VectorR& point, const VectorR& output) const
{
    if (!is_sequential_cpu_network(*network)) return {};
    const auto& layers = network->get_layers();
    vector<VectorR> derivatives(layers.size());
    vector<const CombinationOperator*> combinations(layers.size(), nullptr);
    VectorR values = point;

    for (size_t i = 0; i < layers.size(); i++)
    {
        const Layer& layer = *layers[i];
        const LayerType type = layer.get_type();
        if (is_one_of(type, LayerType::Dense, LayerType::Activation))
        {
            if (type == LayerType::Dense)
            {
                combinations[i] = get_dense_combination(static_cast<const Dense&>(layer));
                if (!combinations[i]) return {};
                values = (combinations[i]->weights.as_matrix().transpose()*values).eval();
                if (combinations[i]->use_bias) values += combinations[i]->bias.as_vector();
            }
            if (!calculate_activation_derivatives(layer.get_output_activation(), values, derivatives[i])) return {};
        }
        else if (is_one_of(type, LayerType::Scaling, LayerType::Unscaling))
        {
            if (!calculate_scaling_derivatives(static_cast<const Scaling&>(layer), values, derivatives[i])) return {};
        }
        else if (type == LayerType::Clamping)
        {
            if (!calculate_clamping_derivatives(static_cast<const Clamping&>(layer), values, derivatives[i])) return {};
        }
        else return {};
        if (!values.allFinite() || !derivatives[i].allFinite()) return {};
    }
    if (values.size() != output.size() || !output.allFinite()
     || ((values - output).cwiseAbs().array() > 32.0f*numeric_limits<float>::epsilon()
           *output.cwiseAbs().cwiseMax(1.0f).array()).any()) return {};

    MatrixR jacobian = MatrixR::Identity(output.size(), output.size());
    for (size_t i = layers.size(); i-- > 0;)
    {
        jacobian.array().rowwise() *= derivatives[i].transpose().array();
        if (combinations[i]) jacobian = (jacobian*combinations[i]->weights.as_matrix().transpose()).eval();
    }
    return jacobian.allFinite() ? jacobian : MatrixR{};
}


template<typename Scalar>
FeasibilityRepairSystem::Matrix<Scalar> FeasibilityRepairSystem::calculate_jacobian(const Vector<Scalar>& point,
                                                                                   const Vector<Scalar>& values,
                                                                                   const VectorR& output,
                                                                                   const Vector<Scalar>& residuals) const
{
    const auto& [lower_bounds, upper_bounds] = get_bounds<Scalar>();

    const bool every_row = residuals.size() != values.size();

    Vector<Scalar> steps = (upper_bounds - lower_bounds).cwiseMax(Scalar(0));

    for (const auto& [first_column, categories_number] : categorical_blocks)
        steps.segment(first_column, categories_number).setZero();

    Matrix<Scalar> jacobian = Matrix<Scalar>::Zero(values.size(), point.size());

    vector<Index> probed_rows;

    const auto& scalar_output = as_scalar<Scalar>(output);
    Vector<Scalar> gradient(point.size());
    Vector<Scalar> output_gradient;
    MatrixR network_jacobian;
    bool network_jacobian_evaluated = false;

    for (Index i = 0; i < values.size(); i++)
    {
        if (!every_row && residuals(i) == Scalar(0)) continue;

        const CompiledExpression& expression = constraints[size_t(i)].equation;

        const bool reads_output = is_output_coupled(expression);
        if (reads_output && !network_jacobian_evaluated)
        {
            network_jacobian = calculate_network_jacobian(as_float(point), output);
            network_jacobian_evaluated = true;
        }
        if (!reads_output || network_jacobian.size() != 0)
        {
            evaluate_input_gradient(expression, point, scalar_output, gradient);
            if (reads_output)
            {
                evaluate_output_gradient(expression, point, scalar_output, output_gradient);
                gradient += network_jacobian.transpose().template cast<Scalar>()*output_gradient;
            }

            if (gradient.allFinite())
            {
                jacobian.row(i) = gradient.transpose();

                continue;
            }
        }

        probed_rows.push_back(i);
    }

    for (Index j = 0; j < point.size(); j++)
        if (steps(j) == Scalar(0))
            jacobian.col(j).setZero();

    if (probed_rows.empty()) return jacobian;

    for (Index j = 0; j < steps.size(); j++)
        if (steps(j) > Scalar(0))
            steps(j) = max(Scalar(difference_step)*max(steps(j), abs(point(j))), Scalar(EPSILON));

    for (const Constraint& constraint : constraints)
        if (const Index column = constraint.get_discrete_column(); column >= 0)
            steps(column) = min(steps(column), Scalar(discrete_difference_step));

    calculate_numerical_jacobian<Scalar>(point, values, steps, probed_rows, jacobian);

    return jacobian;
}


template<typename Scalar>
void FeasibilityRepairSystem::calculate_numerical_jacobian(const Vector<Scalar>& point,
                                                          const Vector<Scalar>& values,
                                                          const Vector<Scalar>& steps,
                                                          const vector<Index>& rows,
                                                          Matrix<Scalar>& jacobian) const
{
    const auto& [lower_bounds, upper_bounds] = get_bounds<Scalar>();

    const bool reads_output = ranges::any_of(rows, [&](const Index i)
        { return is_output_coupled(constraints[size_t(i)].equation); });

    Vector<Scalar> probe = point;
    VectorR probe_output;

    for (Index j = 0; j < point.size(); j++)
    {
        if (steps(j) == Scalar(0)) continue;

        const Scalar first_direction = point(j) + steps(j) > upper_bounds(j) ? Scalar(-1) : Scalar(1);

        for (const Index i : rows) jacobian(i, j) = numeric_limits<Scalar>::quiet_NaN();

        for (const Scalar direction : {first_direction, -first_direction})
        {
            const Scalar bound = direction > Scalar(0) ? upper_bounds(j) : lower_bounds(j);

            probe(j) = clamp(point(j) + direction*steps(j), lower_bounds(j), upper_bounds(j));
            if (probe(j) == point(j)) probe(j) = nextafter(point(j), bound);

            const double step = double(probe(j)) - point(j);
            if (step == 0.0) continue;

            if (reads_output)
                probe_output = network->calculate_outputs(as_float(probe).transpose()).row(0).transpose();

            const auto& scalar_probe_output = as_scalar<Scalar>(probe_output);

            bool complete = true;

            for (const Index i : rows)
            {
                if (isfinite(jacobian(i, j))) continue;

                const Scalar value = evaluate_row(constraints[size_t(i)].equation, probe, scalar_probe_output);
                const Scalar derivative = Scalar((double(value) - values(i))/step);
                if (isfinite(derivative)) jacobian(i, j) = derivative;
                else complete = false;
            }

            probe(j) = point(j);

            if (complete) break;
        }

        for (const Index i : rows)
            if (!isfinite(jacobian(i, j))) jacobian(i, j) = Scalar(0);
    }
}


template<typename Scalar>
pair<Scalar, Scalar> FeasibilityRepairSystem::calculate_constraint_residual(const Index i) const
{
    const auto [lower, upper] = constraints[size_t(i)].get_bounds(numeric_tolerance, constraint_tolerance);
    const Scalar value = calculate_constraint_value<Scalar>(i);

    Scalar crossed_bound = Scalar(0);

    if (value < Scalar(lower) - Scalar(get_bound_tolerance(lower)))
        crossed_bound = Scalar(lower);
    else if (value > Scalar(upper) + Scalar(get_bound_tolerance(upper)))
        crossed_bound = Scalar(upper);
    else
        return {};

    const Scalar residual = value - crossed_bound;
    const Scalar margin = Scalar(feasibility_margin_factor);

    const Scalar inset = min(margin*max(abs(residual), margin*abs(crossed_bound)),
                             Scalar(0.5f)*(Scalar(upper) - Scalar(lower)));

    const Scalar result = residual + ((residual > Scalar(0)) ? inset : -inset);
    const double bound = result < Scalar(0) ? lower : upper;
    const double violation = abs(double(value) - bound);
    const Scalar slope = violation >= double(margin)*abs(bound)
                      && double(margin)*violation < 0.5*(double(upper) - lower)
                       ? Scalar(1) + margin : Scalar(1);
    return {result, slope};
}


template<typename Scalar>
struct FeasibilityRepairSystem::LevenbergMarquardtFunctor : Eigen::DenseFunctor<Scalar>
{
    using Base = Eigen::DenseFunctor<Scalar>;

    LevenbergMarquardtFunctor(FeasibilityRepairSystem& system, Matrix<Scalar> jacobian)
        : Base(int(system.input_bounds.first.size()),
               int(system.constraints.size() + system.input_bounds.first.size())),
          system(system), initial_jacobian(move(jacobian)) {}

    int operator()(const Vector<Scalar>& point, Vector<Scalar>& violations)
    {
        const Workspace<Scalar>& work = system.get_workspace<Scalar>();
        system.evaluate_constraints<Scalar>(point);
        violations.resize(work.row_values.size() + point.size());
        violations.head(work.row_values.size()) = work.row_values.array().isFinite()
            .select(work.row_residuals.array()/work.row_scales.array(), Scalar(1)/Scalar(system.numeric_tolerance));
        const auto& [lower, upper] = system.get_bounds<Scalar>();
        violations.tail(point.size()) =
            ((point - upper).cwiseMax(Scalar(0)) + (point - lower).cwiseMin(Scalar(0))).cwiseQuotient(work.box_scales);
        return 0;
    }

    int df(const Vector<Scalar>& point, typename Base::JacobianType& jacobian)
    {
        const Workspace<Scalar>& work = system.get_workspace<Scalar>();
        system.evaluate_constraints<Scalar>(point);
        const Matrix<Scalar> value_jacobian = initial_jacobian.size() != 0 ? exchange(initial_jacobian, Matrix<Scalar>{})
            : system.calculate_jacobian<Scalar>(point, work.row_values, work.evaluated_output, work.row_residuals);
        jacobian.setZero(work.row_values.size() + point.size(), point.size());
        for (Index i = 0; i < work.row_values.size(); i++)
            if (work.row_residuals(i) != Scalar(0) && isfinite(work.row_residuals(i)))
                jacobian.row(i) = value_jacobian.row(i)
                                  *(system.calculate_constraint_residual<Scalar>(i).second/work.row_scales(i));
        const auto& [lower, upper] = system.get_bounds<Scalar>();
        for (Index j = 0; j < point.size(); j++)
            if (point(j) > upper(j) || point(j) < lower(j))
                jacobian(work.row_values.size() + j, j) = Scalar(1)/work.box_scales(j);
        return 0;
    }

    FeasibilityRepairSystem& system;
    Matrix<Scalar> initial_jacobian;
};


template<typename Scalar>
void FeasibilityRepairSystem::solve_levenberg_marquardt(Vector<Scalar>& point, Matrix<Scalar> initial_jacobian)
{
    // The float repair stops at the library's tolerance; the double repair goes on to a finer one, which
    // is what its precision is for.
    const Scalar tolerance = is_same_v<Scalar, float> ? Scalar(numeric_tolerance) : Scalar(double_solver_tolerance);
    LevenbergMarquardtFunctor<Scalar> functor(*this, move(initial_jacobian));
    Eigen::LevenbergMarquardt<LevenbergMarquardtFunctor<Scalar>> levenberg_marquardt(functor);
    levenberg_marquardt.setMaxfev(feasibility_evaluations);
    levenberg_marquardt.setFtol(tolerance);
    levenberg_marquardt.setXtol(tolerance);
    levenberg_marquardt.setGtol(Scalar(0));
    levenberg_marquardt.minimize(point);
}


template<typename Scalar>
void FeasibilityRepairSystem::solve_gauss_newton(Vector<Scalar>& point, Matrix<Scalar> jacobian)
{
    const auto& [lower, upper] = get_bounds<Scalar>();
    const Workspace<Scalar>& work = get_workspace<Scalar>();
    const Vector<Scalar> spans = upper - lower;
    Vector<Scalar> residuals(work.row_residuals.size());
    const auto calculate_merit = [&]
        { return work.row_residuals.template cast<double>().cwiseQuotient(work.row_scales.template cast<double>())
                     .squaredNorm(); };

    for (Index i = 0; i < feasibility_evaluations;)
    {
        if (!work.row_values.allFinite() || !work.row_residuals.allFinite())
        {
            point.setConstant(numeric_limits<Scalar>::quiet_NaN());
            return;
        }
        if ((work.row_residuals.array() == Scalar(0)).all()) return;

        if (i > 0) jacobian = calculate_jacobian<Scalar>(point, work.row_values, work.evaluated_output, work.row_residuals);
        Index rows = 0;
        for (Index j = 0; j < work.row_residuals.size(); j++)
        {
            if (work.row_residuals(j) == Scalar(0)) continue;
            jacobian.row(rows) = jacobian.row(j)*(calculate_constraint_residual<Scalar>(j).second/work.row_scales(j));
            residuals(rows++) = work.row_residuals(j)/work.row_scales(j);
        }
        auto active_jacobian = jacobian.topRows(rows);
        active_jacobian.array().rowwise() *= spans.transpose().array();
        if (!active_jacobian.allFinite() || !residuals.head(rows).allFinite())
        {
            point.setConstant(numeric_limits<Scalar>::quiet_NaN());
            return;
        }

        const Scalar scale = max(Scalar(1), active_jacobian.cwiseAbs().maxCoeff());
        if (scale > Scalar(1))
        {
            active_jacobian /= scale;
            residuals.head(rows) /= scale;
        }
        const Vector<Scalar> step = active_jacobian.completeOrthogonalDecomposition().solve(-residuals.head(rows));
        if (!step.allFinite())
        {
            point.setConstant(numeric_limits<Scalar>::quiet_NaN());
            return;
        }
        const double merit = calculate_merit();
        Scalar fraction = Scalar(1);
        bool accepted = false;
        for (Index j = 0; j < 8 && i < feasibility_evaluations; j++, fraction *= Scalar(0.5f))
        {
            i++;
            Vector<Scalar> candidate = point + (fraction*spans).cwiseProduct(step);
            if (!candidate.allFinite()) continue;
            candidate = candidate.cwiseMax(lower).cwiseMin(upper);
            if ((candidate.array() == point.array()).all()) return;
            evaluate_constraints<Scalar>(candidate);
            if (work.row_values.allFinite() && work.row_residuals.allFinite() && calculate_merit() < merit)
            {
                point = move(candidate);
                accepted = true;
                break;
            }
        }
        if (!accepted) return;
    }
}


pair<VectorR, VectorR> FeasibilityRepairSystem::solve(VectorR point)
{
    return repair<float>(move(point));
}


pair<Eigen::VectorXd, VectorR> FeasibilityRepairSystem::solve_double(Eigen::VectorXd point)
{
    return repair<double>(move(point));
}

}
