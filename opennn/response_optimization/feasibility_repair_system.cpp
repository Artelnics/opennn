// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

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
}


void FeasibilityRepairSystem::set_solver(const Solver new_solver)
{
    throw_if(!is_one_of(new_solver, Solver::LevenbergMarquardt, Solver::GaussNewton),
             "Invalid feasibility repair solver.");
    solver = new_solver;
}


float FeasibilityRepairSystem::get_bound_tolerance(const float bound) const
{
    return max(EPSILON, abs(bound)*numeric_tolerance);
}


float FeasibilityRepairSystem::calculate_constraint_value(const Index i) const
{
    const Constraint& constraint = constraints[size_t(i)];
    const float value = row_values(i);
    if (constraint.condition == Condition::Integer) return value - round(value);
    if (constraint.condition == Condition::AllowedSet)
        return value - *ranges::min_element(constraint.values, {},
            [&](const float allowed) { return abs(double(value) - allowed); });
    return value;
}


pair<VectorR, VectorR> FeasibilityRepairSystem::solve(VectorR point)
{
    const auto& [lower_bounds, upper_bounds] = input_bounds;

    throw_if(point.size() != lower_bounds.size(), "Feasibility repair point has the wrong size.");
    evaluated_point.resize(0);
    evaluated_output.resize(0);

    VectorR previous;
    VectorR before_previous;

    for (Index i = 0; ; i++)
    {
        if (!point.allFinite()) return {};

        if (!round_discrete(point) || !point.allFinite()) return {};

        evaluate_constraints(point);

        if (!row_values.allFinite() || !row_residuals.allFinite()) return {};

        if ((row_residuals.array() == 0.0f).all())
        {
            if (!evaluate_outputs(point).allFinite()) return {};
            return {move(point), move(evaluated_output)};
        }

        update_closest_failure();

        if (i == feasibility_rounds) return {};

        if (previous.size() == point.size()
         && ((point - previous).cwiseAbs().array()
             <= numeric_tolerance*(upper_bounds - lower_bounds).cwiseMax(EPSILON).array()).all())
            return {};

        if (before_previous.size() == point.size() && (before_previous.array() == point.array()).all())
            return {};

        before_previous.swap(previous);
        previous = point;

        if (i == 0) box_scales = (upper_bounds - lower_bounds).cwiseMax(EPSILON)/box_weight;
        MatrixR initial_jacobian = calculate_jacobian(point, row_values, evaluated_output);
        row_scales = initial_jacobian.rowwise().stableNorm();

        row_scales = (row_scales.array().isFinite() && (row_scales.array() > EPSILON))
                     .select(row_scales.array(), 1.0f);

        if (solver == Solver::GaussNewton) solve_gauss_newton(point, move(initial_jacobian));
        else solve_levenberg_marquardt(point, move(initial_jacobian));
    }
}


bool FeasibilityRepairSystem::round_discrete(VectorR& point) const
{
    const auto& [lower_bounds, upper_bounds] = input_bounds;

    const auto always_on = [&bounds = input_bounds](const Index column)
        { return bounds.first(column) > 0.0f || bounds.second(column) < 0.0f; };


    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition != Condition::Cardinality) continue;

        vector<pair<Index, double>> members = constraint.equation.symmetric_terms;

        ranges::stable_sort(members, {}, [&](const pair<Index, double>& member)
            { return pair(!always_on(member.first), -abs(double(point(member.first)))*member.second); });

        const size_t budget = size_t(constraint.values[0]);

        if (budget < members.size() && always_on(members[budget].first)) return false;

        for (size_t j = budget; j < members.size(); j++)
            point(members[j].first) = 0.0f;
    }

    point = point.cwiseMax(lower_bounds).cwiseMin(upper_bounds);

    for (const auto& [first_column, categories_number] : categorical_blocks)
    {
        Index category = 0;

        const bool none_open =
            (upper_bounds.segment(first_column, categories_number).array() > 0.0f)
            .select(point.segment(first_column, categories_number).array(), -MAX).maxCoeff(&category) == -MAX;

        if (none_open) return false;

        point.segment(first_column, categories_number).setZero();

        point(first_column + category) = 1.0f;
    }

    for (const Constraint& constraint : constraints)
        if (const Index column = constraint.get_discrete_column(); column >= 0)
        {
            const float value = point(column);

            const float lower = lower_bounds(column);
            const float upper = upper_bounds(column);

            if (constraint.condition == Condition::Integer && ceil(lower) > floor(upper)) return false;

            point(column) = constraint.condition == Condition::Integer
                          ? clamp(round(value), ceil(lower), floor(upper))
                          : *ranges::min_element(constraint.values, {}, [&](const float allowed)
                                { return (allowed < lower || allowed > upper)
                                       ? numeric_limits<double>::infinity() : abs(double(allowed) - value); });
        }

    if ((point.array() < lower_bounds.array()).any() || (point.array() > upper_bounds.array()).any()) return false;

    for (const auto& [first, size] : categorical_blocks)
    {
        const auto block = point.segment(first, size).array();
        if (!(block == 0.0f || block == 1.0f).all() || block.sum() != 1.0f) return false;
    }

    return true;
}


const VectorR& FeasibilityRepairSystem::evaluate_outputs(const VectorR& point)
{
    if (evaluated_output.size() == 0 || evaluated_point.size() != point.size()
     || (evaluated_point.array() != point.array()).any())
    {
        evaluated_output = network->calculate_outputs(point.transpose()).row(0).transpose();
        evaluated_point = point;
    }

    return evaluated_output;
}


void FeasibilityRepairSystem::evaluate_constraints(const VectorR& point)
{
    if (constraints_read_output) evaluate_outputs(point);

    const Index rows_number = Index(constraints.size());

    row_values.resize(rows_number);
    row_residuals.resize(rows_number);

    for (Index i = 0; i < rows_number; i++)
    {
        const Constraint& constraint = constraints[size_t(i)];

        row_values(i) = constraint.equation.evaluate(point, evaluated_output);

        row_residuals(i) = calculate_constraint_residual(i).first;
    }
}


void FeasibilityRepairSystem::update_closest_failure()
{
    Index worst = -1;
    double worst_ratio = 0.0;
    double worst_gap = 0.0;
    double worst_tolerance = 0.0;
    float worst_bound = 0.0f;

    for (Index i = 0; i < row_values.size(); i++)
    {
        if (row_residuals(i) == 0.0f) continue;
        const Constraint& constraint = constraints[size_t(i)];
        const auto [lower, upper] = constraint.get_bounds(numeric_tolerance, constraint_tolerance);
        const float bound = row_residuals(i) < 0.0f ? lower : upper;
        const bool discrete = is_one_of(constraint.condition, Condition::Integer, Condition::AllowedSet);
        const double gap = discrete ? abs(double(calculate_constraint_value(i))) : abs(double(row_values(i)) - bound);
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
    ostringstream message;
    message << setprecision(numeric_limits<float>::max_digits10)
            << "Best checked rounded candidate with a constraint violation (smallest maximum miss/tolerance): constraint '"
            << constraint.equation.text << "' (" << names[size_t(constraint.condition)] << "): "
            << (constraint.condition == Condition::Cardinality ? "cardinality measure " : "value ") << row_values(worst);
    if (constraint.condition == Condition::Integer) message << ", nearest integer " << round(row_values(worst));
    else if (constraint.condition == Condition::AllowedSet)
        message << ", nearest allowed value " << *ranges::min_element(constraint.values, {},
            [&](const float allowed) { return abs(double(row_values(worst)) - allowed); });
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


MatrixR FeasibilityRepairSystem::calculate_jacobian(const VectorR& point,
                                                   const VectorR& values,
                                                   const VectorR& output,
                                                   const VectorR& residuals) const
{
    const auto& [lower_bounds, upper_bounds] = input_bounds;

    const bool every_row = residuals.size() != values.size();

    VectorR steps = (upper_bounds - lower_bounds).cwiseMax(0.0f);

    for (const auto& [first_column, categories_number] : categorical_blocks)
        steps.segment(first_column, categories_number).setZero();

    MatrixR jacobian = MatrixR::Zero(values.size(), point.size());

    vector<Index> probed_rows;

    VectorR gradient(point.size());
    VectorR output_gradient;
    MatrixR network_jacobian;
    bool network_jacobian_evaluated = false;

    for (Index i = 0; i < values.size(); i++)
    {
        if (!every_row && residuals(i) == 0.0f) continue;

        const CompiledExpression& expression = constraints[size_t(i)].equation;

        const bool reads_output = is_output_coupled(expression);
        if (reads_output && !network_jacobian_evaluated)
        {
            network_jacobian = calculate_network_jacobian(point, output);
            network_jacobian_evaluated = true;
        }
        if (!reads_output || network_jacobian.size() != 0)
        {
            evaluate_input_gradient(expression, point, output, gradient);
            if (reads_output)
            {
                evaluate_output_gradient(expression, point, output, output_gradient);
                gradient += network_jacobian.transpose()*output_gradient;
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
        if (steps(j) == 0.0f)
            jacobian.col(j).setZero();

    if (probed_rows.empty()) return jacobian;

    for (Index j = 0; j < steps.size(); j++)
        if (steps(j) > 0.0f)
            steps(j) = max(difference_step*max(steps(j), abs(point(j))), EPSILON);

    for (const Constraint& constraint : constraints)
        if (const Index column = constraint.get_discrete_column(); column >= 0)
            steps(column) = min(steps(column), discrete_difference_step);

    calculate_numerical_jacobian(point, values, steps, probed_rows, jacobian);

    return jacobian;
}


void FeasibilityRepairSystem::calculate_numerical_jacobian(const VectorR& point,
                                                          const VectorR& values,
                                                          const VectorR& steps,
                                                          const vector<Index>& rows,
                                                          MatrixR& jacobian) const
{
    const auto& [lower_bounds, upper_bounds] = input_bounds;

    const bool reads_output = ranges::any_of(rows, [&](const Index i)
        { return is_output_coupled(constraints[size_t(i)].equation); });

    VectorR probe = point;
    VectorR probe_output;

    for (Index j = 0; j < point.size(); j++)
    {
        if (steps(j) == 0.0f) continue;

        const float first_direction = point(j) + steps(j) > upper_bounds(j) ? -1.0f : 1.0f;

        for (const Index i : rows) jacobian(i, j) = QUIET_NAN;

        for (const float direction : {first_direction, -first_direction})
        {
            const float bound = direction > 0.0f ? upper_bounds(j) : lower_bounds(j);

            probe(j) = clamp(point(j) + direction*steps(j), lower_bounds(j), upper_bounds(j));
            if (probe(j) == point(j)) probe(j) = nextafter(point(j), bound);

            const double step = double(probe(j)) - point(j);
            if (step == 0.0) continue;

            if (reads_output)
                probe_output = network->calculate_outputs(probe.transpose()).row(0).transpose();

            bool complete = true;

            for (const Index i : rows)
            {
                if (isfinite(jacobian(i, j))) continue;

                const float value = constraints[size_t(i)].equation.evaluate(probe, probe_output);
                const float derivative = float((double(value) - values(i))/step);
                if (isfinite(derivative)) jacobian(i, j) = derivative;
                else complete = false;
            }

            probe(j) = point(j);

            if (complete) break;
        }

        for (const Index i : rows)
            if (!isfinite(jacobian(i, j))) jacobian(i, j) = 0.0f;
    }
}


pair<float, float> FeasibilityRepairSystem::calculate_constraint_residual(const Index i) const
{
    const auto [lower, upper] = constraints[size_t(i)].get_bounds(numeric_tolerance, constraint_tolerance);
    const float value = calculate_constraint_value(i);

    float crossed_bound = 0.0f;

    if (value < lower - get_bound_tolerance(lower))
        crossed_bound = lower;
    else if (value > upper + get_bound_tolerance(upper))
        crossed_bound = upper;
    else
        return {};

    const float residual = value - crossed_bound;
    const float margin = feasibility_margin_factor;

    const float inset = min(margin*max(abs(residual), margin*abs(crossed_bound)),
                            0.5f*(upper - lower));

    const float result = residual + ((residual > 0.0f) ? inset : -inset);
    const double bound = result < 0.0f ? lower : upper;
    const double violation = abs(double(value) - bound);
    const float slope = violation >= double(margin)*abs(bound)
                     && double(margin)*violation < 0.5*(double(upper) - lower)
                      ? 1.0f + margin : 1.0f;
    return {result, slope};
}


struct FeasibilityRepairSystem::LevenbergMarquardtFunctor : Eigen::DenseFunctor<float>
{
    LevenbergMarquardtFunctor(FeasibilityRepairSystem& system, MatrixR jacobian)
        : Eigen::DenseFunctor<float>(int(system.input_bounds.first.size()),
              int(system.constraints.size() + system.input_bounds.first.size())),
          system(system), initial_jacobian(move(jacobian)) {}

    int operator()(const VectorR& point, VectorR& violations)
    {
        system.evaluate_constraints(point);
        violations.resize(system.row_values.size() + point.size());
        violations.head(system.row_values.size()) = system.row_values.array().isFinite()
            .select(system.row_residuals.array()/system.row_scales.array(), 1.0f/system.numeric_tolerance);
        const auto& [lower, upper] = system.input_bounds;
        violations.tail(point.size()) =
            ((point - upper).cwiseMax(0.0f) + (point - lower).cwiseMin(0.0f)).cwiseQuotient(system.box_scales);
        return 0;
    }

    int df(const VectorR& point, JacobianType& jacobian)
    {
        system.evaluate_constraints(point);
        const MatrixR value_jacobian = initial_jacobian.size() != 0 ? exchange(initial_jacobian, MatrixR{})
            : system.calculate_jacobian(point, system.row_values, system.evaluated_output, system.row_residuals);
        jacobian.setZero(system.row_values.size() + point.size(), point.size());
        for (Index i = 0; i < system.row_values.size(); i++)
            if (system.row_residuals(i) != 0.0f && isfinite(system.row_residuals(i)))
                jacobian.row(i) = value_jacobian.row(i)*(system.calculate_constraint_residual(i).second/system.row_scales(i));
        const auto& [lower, upper] = system.input_bounds;
        for (Index j = 0; j < point.size(); j++)
            if (point(j) > upper(j) || point(j) < lower(j))
                jacobian(system.row_values.size() + j, j) = 1.0f/system.box_scales(j);
        return 0;
    }

    FeasibilityRepairSystem& system;
    MatrixR initial_jacobian;
};


void FeasibilityRepairSystem::solve_levenberg_marquardt(VectorR& point, MatrixR initial_jacobian)
{
    LevenbergMarquardtFunctor functor(*this, move(initial_jacobian));
    Eigen::LevenbergMarquardt<LevenbergMarquardtFunctor> levenberg_marquardt(functor);
    levenberg_marquardt.setMaxfev(feasibility_evaluations);
    levenberg_marquardt.setFtol(numeric_tolerance);
    levenberg_marquardt.setXtol(numeric_tolerance);
    levenberg_marquardt.setGtol(0.0f);
    levenberg_marquardt.minimize(point);
}


void FeasibilityRepairSystem::solve_gauss_newton(VectorR& point, MatrixR jacobian)
{
    const auto& [lower, upper] = input_bounds;
    const VectorR spans = upper - lower;
    VectorR residuals(row_residuals.size());
    const auto calculate_merit = [&]
        { return row_residuals.cast<double>().cwiseQuotient(row_scales.cast<double>()).squaredNorm(); };

    for (Index i = 0; i < feasibility_evaluations;)
    {
        if (!row_values.allFinite() || !row_residuals.allFinite())
        {
            point.setConstant(QUIET_NAN);
            return;
        }
        if ((row_residuals.array() == 0.0f).all()) return;

        if (i > 0) jacobian = calculate_jacobian(point, row_values, evaluated_output, row_residuals);
        Index rows = 0;
        for (Index j = 0; j < row_residuals.size(); j++)
        {
            if (row_residuals(j) == 0.0f) continue;
            jacobian.row(rows) = jacobian.row(j)*(calculate_constraint_residual(j).second/row_scales(j));
            residuals(rows++) = row_residuals(j)/row_scales(j);
        }
        auto active_jacobian = jacobian.topRows(rows);
        active_jacobian.array().rowwise() *= spans.transpose().array();
        if (!active_jacobian.allFinite() || !residuals.head(rows).allFinite())
        {
            point.setConstant(QUIET_NAN);
            return;
        }

        const float scale = max(1.0f, active_jacobian.cwiseAbs().maxCoeff());
        if (scale > 1.0f)
        {
            active_jacobian /= scale;
            residuals.head(rows) /= scale;
        }
        const VectorR step = active_jacobian.completeOrthogonalDecomposition().solve(-residuals.head(rows));
        if (!step.allFinite())
        {
            point.setConstant(QUIET_NAN);
            return;
        }
        const double merit = calculate_merit();
        float fraction = 1.0f;
        bool accepted = false;
        for (Index j = 0; j < 8 && i < feasibility_evaluations; j++, fraction *= 0.5f)
        {
            i++;
            VectorR candidate = point + (fraction*spans).cwiseProduct(step);
            if (!candidate.allFinite()) continue;
            candidate = candidate.cwiseMax(lower).cwiseMin(upper);
            if ((candidate.array() == point.array()).all()) return;
            evaluate_constraints(candidate);
            if (row_values.allFinite() && row_residuals.allFinite() && calculate_merit() < merit)
            {
                point = move(candidate);
                accepted = true;
                break;
            }
        }
        if (!accepted) return;
    }
}

}
