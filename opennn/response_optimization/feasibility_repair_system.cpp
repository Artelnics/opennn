// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/response_optimization/feasibility_repair_system.h"

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
    bounds = {domain.first.cast<double>(), domain.second.cast<double>()};
}


void FeasibilityRepairSystem::set_solver(const Solver new_solver)
{
    throw_if(!is_one_of(new_solver, Solver::LevenbergMarquardt, Solver::GaussNewton, Solver::Broyden),
             "Invalid feasibility repair solver.");
    solver = new_solver;
}


pair<VectorR, VectorR> FeasibilityRepairSystem::solve(VectorR point)
{
    auto [repaired, output] = repair(point.cast<double>(), true);
    return {repaired.cast<float>(), move(output)};
}


pair<Eigen::VectorXd, VectorR> FeasibilityRepairSystem::solve_double(Eigen::VectorXd point)
{
    return repair(move(point), false);
}


// Each round projects the point and evaluates it; a met point is returned, otherwise the solver takes over from
// the Jacobian at the projected point. The rounds stop when the point stops moving or comes back.
pair<FeasibilityRepairSystem::Vector, VectorR> FeasibilityRepairSystem::repair(Vector point, const bool for_float)
{
    const auto& [lower_bounds, upper_bounds] = bounds;

    throw_if(point.size() != lower_bounds.size(), "Feasibility repair point has the wrong size.");
    float_points = for_float;
    work.evaluated_point.resize(0);
    work.evaluated_output.resize(0);
    work.row_margins.resize(0);

    Vector previous;
    Vector before_previous;

    for (Index i = 0; ; i++)
    {
        if (!point.allFinite()) return {};

        if (!round_discrete(point) || !point.allFinite()) return {};

        evaluate_constraints(point);

        if (!work.row_values.allFinite() || !work.row_residuals.allFinite()) return {};

        if ((work.row_residuals.array() == 0.0).all())
        {
            if (!evaluate_outputs(point).allFinite()) return {};
            return {move(point), move(work.evaluated_output)};
        }

        update_closest_failure();

        if (i == feasibility_rounds) return {};

        if (previous.size() == point.size()
         && ((point - previous).cwiseAbs().array()
             <= double(numeric_tolerance)*(upper_bounds - lower_bounds).cwiseMax(double(EPSILON)).array()).all())
            return {};

        if (before_previous.size() == point.size() && (before_previous.array() == point.array()).all())
            return {};

        before_previous.swap(previous);
        previous = point;

        if (i == 0) work.box_scales = (upper_bounds - lower_bounds).cwiseMax(double(EPSILON))/double(box_weight);
        Matrix initial_jacobian = calculate_jacobian(point, work.row_values, work.evaluated_output);
        work.row_scales = initial_jacobian.rowwise().stableNorm();

        work.row_scales = (work.row_scales.array().isFinite() && (work.row_scales.array() > double(EPSILON)))
                          .select(work.row_scales.array(), 1.0);

        if (float_points) set_float_margins(point, initial_jacobian);

        if (solver == Solver::LevenbergMarquardt) solve_levenberg_marquardt(point, move(initial_jacobian));
        else solve_gauss_newton(point, move(initial_jacobian), solver == Solver::Broyden);
    }
}


// Switches off the members a cardinality budget leaves out, clips the point to the box, picks one category per
// categorical input and places the discrete inputs; false when the conditions leave no valid point.
bool FeasibilityRepairSystem::round_discrete(Vector& point) const
{
    const auto& [lower_bounds, upper_bounds] = bounds;

    const auto always_on = [&](const Index column)
        { return lower_bounds(column) > 0.0 || upper_bounds(column) < 0.0; };

    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition != Condition::Cardinality) continue;

        vector<pair<Index, double>> members = constraint.equation.symmetric_terms;

        ranges::stable_sort(members, {}, [&](const pair<Index, double>& member)
            { return pair(!always_on(member.first), -abs(point(member.first))*member.second); });

        const size_t budget = size_t(constraint.values[0]);

        if (budget < members.size() && always_on(members[budget].first)) return false;

        for (size_t j = budget; j < members.size(); j++)
            point(members[j].first) = 0.0;
    }

    point = point.cwiseMax(lower_bounds).cwiseMin(upper_bounds);

    for (const auto& [first_column, categories_number] : categorical_blocks)
    {
        Index category = 0;

        const bool none_open =
            (upper_bounds.segment(first_column, categories_number).array() > 0.0)
            .select(point.segment(first_column, categories_number).array(), double(-MAX)).maxCoeff(&category)
            == double(-MAX);

        if (none_open) return false;

        point.segment(first_column, categories_number).setZero();

        point(first_column + category) = 1.0;
    }

    for (const Constraint& constraint : constraints)
        if (const Index column = constraint.get_discrete_column(); column >= 0)
        {
            const double value = point(column);

            const double lower = lower_bounds(column);
            const double upper = upper_bounds(column);

            if (constraint.condition == Condition::Integer && ceil(lower) > floor(upper)) return false;

            point(column) = constraint.condition == Condition::Integer
                          ? clamp(round(value), ceil(lower), floor(upper))
                          : double(*ranges::min_element(constraint.values, {}, [&](const float allowed)
                                { return (allowed < lower || allowed > upper)
                                       ? numeric_limits<double>::infinity() : abs(double(allowed) - value); }));
        }

    if ((point.array() < lower_bounds.array()).any() || (point.array() > upper_bounds.array()).any()) return false;

    for (const auto& [first, size] : categorical_blocks)
    {
        const auto block = point.segment(first, size).array();
        if (!(block == 0.0 || block == 1.0).all() || block.sum() != 1.0) return false;
    }

    return true;
}


// A float step in an input moves a constraint by about its derivative times that step. For a caller that keeps
// float points, a violated bound is aimed that far inside, so that rounding to float keeps the repaired point in.
void FeasibilityRepairSystem::set_float_margins(const Vector& point, const Matrix& jacobian)
{
    work.row_margins = Vector::Zero(work.row_values.size());

    for (Index i = 0; i < jacobian.rows(); i++)
        for (Index j = 0; j < point.size(); j++)
        {
            const float coordinate = float(point(j));
            const double float_step = double(nextafter(coordinate, numeric_limits<float>::infinity())) - coordinate;
            if (isfinite(jacobian(i, j))) work.row_margins(i) += abs(jacobian(i, j))*float_step;
        }

    for (Index i = 0; i < work.row_values.size(); i++)
        work.row_residuals(i) = calculate_constraint_residual(i).first;
}


const VectorR& FeasibilityRepairSystem::evaluate_outputs(const Vector& point)
{
    if (work.evaluated_output.size() == 0 || work.evaluated_point.size() != point.size()
     || (work.evaluated_point.array() != point.array()).any())
    {
        work.evaluated_output = network->calculate_outputs(VectorR(point.cast<float>()).transpose()).row(0).transpose();
        work.evaluated_point = point;
    }

    return work.evaluated_output;
}


// The network reads float points. For a caller that keeps float points, the constraints are also evaluated at
// the float point the repair will return, so its feasibility is the caller's.
void FeasibilityRepairSystem::evaluate_constraints(const Vector& point)
{
    if (constraints_read_output) evaluate_outputs(point);

    const Index rows_number = Index(constraints.size());

    work.row_values.resize(rows_number);
    work.row_residuals.resize(rows_number);

    const Vector output = work.evaluated_output.cast<double>();
    const Vector float_point = float_points ? Vector(point.cast<float>().cast<double>()) : Vector();
    const Vector& evaluated_point = float_points ? float_point : point;

    for (Index i = 0; i < rows_number; i++)
    {
        work.row_values(i) = constraints[size_t(i)].equation.evaluate_double(evaluated_point, output);

        work.row_residuals(i) = calculate_constraint_residual(i).first;
    }
}


// The quantity a constraint's bounds apply to: its value or, for Integer and AllowedSet, the offset of its value
// from the nearest whole number or allowed value.
double FeasibilityRepairSystem::calculate_constraint_value(const Index i) const
{
    const Constraint& constraint = constraints[size_t(i)];
    const double value = work.row_values(i);
    if (constraint.condition == Condition::Integer) return value - round(value);
    if (constraint.condition == Condition::AllowedSet)
        return value - double(*ranges::min_element(constraint.values, {},
            [&](const float allowed) { return abs(value - allowed); }));
    return value;
}


// The residual of a constraint is zero within its tolerance of the bounds. Outside, it is the distance past the
// crossed bound plus an inset that aims a little inside: the margin factor times the distance, or times the margin
// factor and the bound once the distance is small. The slope is the derivative of the residual by the value.
pair<double, double> FeasibilityRepairSystem::calculate_constraint_residual(const Index i) const
{
    const auto [lower, upper] = constraints[size_t(i)].get_bounds(numeric_tolerance, constraint_tolerance);
    const double value = calculate_constraint_value(i);

    double crossed_bound = 0.0;

    if (value < double(lower) - double(get_bound_tolerance(lower)))
        crossed_bound = double(lower);
    else if (value > double(upper) + double(get_bound_tolerance(upper)))
        crossed_bound = double(upper);
    else
        return {};

    if (work.row_margins.size() != 0)
    {
        const double shift = min(work.row_margins(i), 0.5*(double(upper) - double(lower)));
        crossed_bound += crossed_bound == double(lower) ? shift : -shift;
    }

    const double residual = value - crossed_bound;
    const double margin = double(feasibility_margin_factor);

    const double inset = min(margin*max(abs(residual), margin*abs(crossed_bound)), 0.5*(double(upper) - double(lower)));

    const double result = residual + ((residual > 0.0) ? inset : -inset);
    const double bound = result < 0.0 ? lower : upper;
    const double violation = abs(value - bound);
    const double slope = violation >= margin*abs(bound) && margin*violation < 0.5*(double(upper) - lower)
                       ? 1.0 + margin : 1.0;
    return {result, slope};
}


float FeasibilityRepairSystem::get_bound_tolerance(const float bound) const
{
    return max(EPSILON, abs(bound)*numeric_tolerance);
}


void FeasibilityRepairSystem::update_closest_failure()
{
    Index worst = -1;
    double worst_ratio = 0.0;
    double worst_gap = 0.0;
    double worst_tolerance = 0.0;
    float worst_bound = 0.0f;

    for (Index i = 0; i < work.row_values.size(); i++)
    {
        if (work.row_residuals(i) == 0.0) continue;
        const Constraint& constraint = constraints[size_t(i)];
        const auto [lower, upper] = constraint.get_bounds(numeric_tolerance, constraint_tolerance);
        const float bound = work.row_residuals(i) < 0.0 ? lower : upper;
        const bool discrete = is_one_of(constraint.condition, Condition::Integer, Condition::AllowedSet);
        const double gap = discrete ? abs(calculate_constraint_value(i)) : abs(work.row_values(i) - bound);
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
    const double value = work.row_values(worst);
    ostringstream message;
    message << setprecision(numeric_limits<double>::max_digits10)
            << "Best checked rounded candidate with a constraint violation (smallest maximum miss/tolerance): constraint '"
            << constraint.equation.text << "' (" << names[size_t(constraint.condition)] << "): "
            << (constraint.condition == Condition::Cardinality ? "cardinality measure " : "value ") << value;
    if (constraint.condition == Condition::Integer) message << ", nearest integer " << round(value);
    else if (constraint.condition == Condition::AllowedSet)
        message << ", nearest allowed value " << *ranges::min_element(constraint.values, {},
            [&](const float allowed) { return abs(value - allowed); });
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


// The derivatives of the constraint values by the inputs, of every row or of the rows with a nonzero entry in
// rows: analytical where the expression and the network allow it, by finite differences otherwise. Fixed and
// categorical columns have none.
FeasibilityRepairSystem::Matrix FeasibilityRepairSystem::calculate_jacobian(const Vector& point, const Vector& values,
                                                                            const VectorR& output,
                                                                            const Vector& rows) const
{
    const auto& [lower_bounds, upper_bounds] = bounds;

    const bool every_row = rows.size() != values.size();

    Vector steps = (upper_bounds - lower_bounds).cwiseMax(0.0);

    for (const auto& [first_column, categories_number] : categorical_blocks)
        steps.segment(first_column, categories_number).setZero();

    Matrix jacobian = Matrix::Zero(values.size(), point.size());

    vector<Index> probed_rows;

    const Vector double_output = output.cast<double>();
    Vector gradient(point.size());
    Vector output_gradient;
    MatrixR network_jacobian;
    bool network_jacobian_evaluated = false;

    for (Index i = 0; i < values.size(); i++)
    {
        if (!every_row && rows(i) == 0.0) continue;

        const CompiledExpression& expression = constraints[size_t(i)].equation;

        const bool reads_output = is_output_coupled(expression);
        if (reads_output && !network_jacobian_evaluated)
        {
            network_jacobian = calculate_network_jacobian(VectorR(point.cast<float>()), output);
            network_jacobian_evaluated = true;
        }
        if (!reads_output || network_jacobian.size() != 0)
        {
            evaluate_input_gradient(expression, point, double_output, gradient);
            if (reads_output)
            {
                evaluate_output_gradient(expression, point, double_output, output_gradient);
                gradient += network_jacobian.transpose().cast<double>()*output_gradient;
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
        if (steps(j) == 0.0)
            jacobian.col(j).setZero();

    if (probed_rows.empty()) return jacobian;

    for (Index j = 0; j < steps.size(); j++)
        if (steps(j) > 0.0)
            steps(j) = max(double(difference_step)*max(steps(j), abs(point(j))), double(EPSILON));

    for (const Constraint& constraint : constraints)
        if (const Index column = constraint.get_discrete_column(); column >= 0)
            steps(column) = min(steps(column), double(discrete_difference_step));

    calculate_numerical_jacobian(point, values, steps, probed_rows, jacobian);

    return jacobian;
}


// One-sided differences, probing the other side when a row is not finite on the first; probes are evaluated
// where the rows are, at float points for a caller that keeps them.
void FeasibilityRepairSystem::calculate_numerical_jacobian(const Vector& point, const Vector& values,
                                                          const Vector& steps, const vector<Index>& rows,
                                                          Matrix& jacobian) const
{
    const auto& [lower_bounds, upper_bounds] = bounds;

    const bool reads_output = ranges::any_of(rows, [&](const Index i)
        { return is_output_coupled(constraints[size_t(i)].equation); });

    Vector probe = point;
    VectorR probe_output;

    for (Index j = 0; j < point.size(); j++)
    {
        if (steps(j) == 0.0) continue;

        const double first_direction = point(j) + steps(j) > upper_bounds(j) ? -1.0 : 1.0;

        for (const Index i : rows) jacobian(i, j) = numeric_limits<double>::quiet_NaN();

        for (const double direction : {first_direction, -first_direction})
        {
            const double bound = direction > 0.0 ? upper_bounds(j) : lower_bounds(j);

            probe(j) = clamp(point(j) + direction*steps(j), lower_bounds(j), upper_bounds(j));
            if (probe(j) == point(j)) probe(j) = nextafter(point(j), bound);

            const double step = probe(j) - point(j);
            if (step == 0.0) continue;

            if (reads_output)
                probe_output = network->calculate_outputs(VectorR(probe.cast<float>()).transpose()).row(0).transpose();

            const Vector double_probe_output = probe_output.cast<double>();
            const Vector float_probe = float_points ? Vector(probe.cast<float>().cast<double>()) : Vector();
            const Vector& evaluated_probe = float_points ? float_probe : probe;

            bool complete = true;

            for (const Index i : rows)
            {
                if (isfinite(jacobian(i, j))) continue;

                const double value = constraints[size_t(i)].equation.evaluate_double(evaluated_probe, double_probe_output);
                const double derivative = (value - values(i))/step;
                if (isfinite(derivative)) jacobian(i, j) = derivative;
                else complete = false;
            }

            probe(j) = point(j);

            if (complete) break;
        }

        for (const Index i : rows)
            if (!isfinite(jacobian(i, j))) jacobian(i, j) = 0.0;
    }
}

}
