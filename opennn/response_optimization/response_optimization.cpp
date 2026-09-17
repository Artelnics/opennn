// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/response_optimization/response_optimization.h"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "opennn/registry.h"
#include "opennn/network/network.h"
#include "opennn/network/layers/scaling_layer.h"
#include "opennn/response_optimization/expression_evaluator.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/statistics.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_operations.h"

namespace opennn
{

namespace
{

using Constraint = ResponseOptimization::Constraint;
using Condition = Constraint::Condition;
using FeasibilitySystem = ResponseOptimization::FeasibilitySystem;


float membership_scale(const vector<float>& values)
{
    const auto [smallest, largest] = ranges::minmax(values);

    const float span = max(largest - smallest, EPSILON);

    float scale = numeric_limits<float>::infinity();

    for (const float root : values)
    {
        float slope = 1.0f;

        for (const float other : values)
            if (other != root)
                slope *= (root - other)/span;

        scale = min(scale, abs(slope));
    }

    return scale;
}


vector<Index> get_group_members(const string& expression, const Network* network)
{
    vector<Index> members;

    for (const string_view entry : get_token_views(expression, ';'))
    {
        const string name(trim_view(entry));

        throw_if(name.empty(), "Constraint on '" + expression + "' leaves an empty entry in its list of variables.");

        const CompiledExpression member = compile_expression(name, network, "Constraint");

        throw_if(is_output_coupled(member) || !is_bare_variable(member),
                 "Constraint on '" + expression + "' lists '" + name + "', which is not a single input variable. "
                 "The Cardinality condition takes a list of input variables, as in 'x1; x2; x3'.");

        const Index column = member.linear_input_terms.front().first;

        throw_if(ranges::find(members, column) != members.end(),
                 "Constraint on '" + expression + "' lists '" + name + "' twice.");

        members.push_back(column);
    }

    throw_if(members.size() < 2,
             "Constraint on '" + expression + "' counts how many variables are in play. "
             "The Cardinality condition applies to a list of at least two input variables, as in 'x1; x2; x3'.");

    return members;
}


Index get_discrete_column(const Constraint& constraint)
{
    return ((constraint.condition == Condition::Integer || constraint.condition == Condition::AllowedSet)
            && constraint.equation.input_indices.size() == 1)
         ? constraint.equation.input_indices.front()
         : -1;
}

}


ResponseOptimization::ResponseOptimization(Network* new_network)
{
    feasibility_system.problem = this;

    set(new_network);
}


ResponseOptimization::~ResponseOptimization() = default;


float ResponseOptimization::get_bound_tolerance(const float bound) const
{
    return max(EPSILON, abs(bound)*numeric_tolerance);
}


float ResponseOptimization::calculate_band_residual(const pair<float, float>& band, const float value) const
{
    const auto [lower, upper] = band;

    float residual = 0.0f;
    float crossed_bound = 0.0f;

    if (value < lower - get_bound_tolerance(lower))
    {
        residual = value - lower;
        crossed_bound = lower;
    }
    else if (value > upper + get_bound_tolerance(upper))
    {
        residual = value - upper;
        crossed_bound = upper;
    }
    else
        return 0.0f;

    const float inset = min(feasibility_margin_factor*max(abs(residual), feasibility_margin_factor*abs(crossed_bound)),
                            0.5f*(upper - lower));

    return residual + ((residual > 0.0f) ? inset : -inset);
}


pair<float, float> ResponseOptimization::get_band(const Constraint& constraint) const
{
    const vector<float>& values = constraint.values;

    const float unbounded = numeric_limits<float>::infinity();

    switch (constraint.condition)
    {
    case Condition::Between:      return {values[0], values[1]};

    case Condition::Equal:        return {values[0], values[0]};

    case Condition::GreaterEqual: return {values[0], unbounded};

    case Condition::Greater:      return {values[0] + 2.0f*get_bound_tolerance(values[0]), unbounded};

    case Condition::LessEqual:    return {-unbounded, values[0]};

    case Condition::Less:         return {-unbounded, values[0] - 2.0f*get_bound_tolerance(values[0])};

    case Condition::Integer:      return {-constraint_tolerance, constraint_tolerance};

    case Condition::AllowedSet:
    {
        const float width = constraint_tolerance*membership_scale(values);

        return {-width, width};
    }

    case Condition::Cardinality:  return {-1.0f, 1.0f};
    }

    return {-unbounded, unbounded};
}


pair<VectorR, VectorR> ResponseOptimization::narrow_domain(pair<VectorR, VectorR> domain) const
{
    for (const Constraint& constraint : constraints)
    {
        const CompiledExpression& expression = constraint.equation;

        Index column = get_discrete_column(constraint);

        float lower = 0.0f;
        float upper = 0.0f;

        if (constraint.condition == Condition::AllowedSet && column >= 0)
        {
            lower = constraint.values.front();
            upper = constraint.values.back();
        }
        else if (!is_output_coupled(expression)
              && expression.linearity == ExpressionLinearity::Linear
              && expression.linear_input_terms.size() == 1
              && abs(expression.linear_input_terms.front().second) > EPSILON)
        {
            const float coefficient = expression.linear_input_terms.front().second;

            const auto [band_lower, band_upper] = get_band(constraint);

            const float at_lower = (band_lower - expression.linear_constant)/coefficient;
            const float at_upper = (band_upper - expression.linear_constant)/coefficient;

            column = expression.linear_input_terms.front().first;
            lower = min(at_lower, at_upper);
            upper = max(at_lower, at_upper);
        }
        else
            continue;

        domain.first(column) = max(domain.first(column), lower);
        domain.second(column) = min(domain.second(column), upper);

        throw_if(domain.first(column) > domain.second(column) + get_bound_tolerance(domain.second(column)),
                 "The constraints leave input column " + to_string(column) + " with an empty range ["
                 + to_string(domain.first(column)) + ", " + to_string(domain.second(column)) + "].");
    }

    return domain;
}


void ResponseOptimization::check_domain(const pair<VectorR, VectorR>& domain) const
{
    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition == Condition::Cardinality)
            for (const Index counted : constraint.equation.input_indices)
                throw_if(domain.first(counted) > get_bound_tolerance(domain.first(counted))
                      || domain.second(counted) < -get_bound_tolerance(domain.second(counted)),
                         "Constraint on '" + constraint.equation.text + "' counts input column "
                         + to_string(counted) + ", whose range [" + to_string(domain.first(counted)) + ", "
                         + to_string(domain.second(counted)) + "] excludes zero, so it can never be switched off.");

        const Index column = get_discrete_column(constraint);

        if (column < 0) continue;

        const float lower = domain.first(column) - get_bound_tolerance(domain.first(column));
        const float upper = domain.second(column) + get_bound_tolerance(domain.second(column));

        if (constraint.condition == Condition::Integer)
            throw_if(ceil(lower) > floor(upper),
                     "Constraint on '" + constraint.equation.text + "' asks for a whole number in ["
                     + to_string(lower) + ", " + to_string(upper) + "], which holds none.");
        else
            throw_if(ranges::none_of(constraint.values,
                                     [&](const float allowed)
                                     { return allowed >= lower && allowed <= upper; }),
                     "Constraint on '" + constraint.equation.text + "' has no allowed value inside ["
                     + to_string(lower) + ", " + to_string(upper) + "].");
    }
}


void ResponseOptimization::FeasibilitySystem::initialize()
{
    borders = problem->narrow_domain(problem->get_unconstrained_domain());

    problem->check_domain(borders);
}


VectorR ResponseOptimization::FeasibilitySystem::round_to_grid(const VectorR& point) const
{
    VectorR forced = point;

    for (const Constraint& constraint : problem->constraints)
    {
        if (constraint.condition != Condition::Cardinality) continue;

        vector<Index> counted = constraint.equation.input_indices;

        ranges::stable_sort(counted, {}, [&](const Index column) { return -abs(forced(column)); });

        for (size_t j = size_t(constraint.values[0]); j < counted.size(); j++)
            forced(counted[j]) = 0.0f;
    }

    forced = forced.cwiseMax(borders.first).cwiseMin(borders.second);

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(problem->network->get_input_variables()))
    {
        Index category = 0;

        const bool none_open =
            (borders.second.segment(first_column, categories_number).array() > 0.0f)
            .select(forced.segment(first_column, categories_number).array(), -MAX).maxCoeff(&category) == -MAX;

        if (none_open) continue;

        forced.segment(first_column, categories_number).setZero();

        forced(first_column + category) = 1.0f;
    }

    for (const Constraint& constraint : problem->constraints)
        if (const Index column = get_discrete_column(constraint); column >= 0)
        {
            const float value = forced(column);

            const float lower = borders.first(column) - problem->get_bound_tolerance(borders.first(column));
            const float upper = borders.second(column) + problem->get_bound_tolerance(borders.second(column));

            forced(column) = (constraint.condition == Condition::Integer)
                           ? clamp(round(value), ceil(lower), floor(upper))
                           : *ranges::min_element(constraint.values, {}, [&](const float allowed)
                                 { return (allowed < lower || allowed > upper) ? MAX : abs(allowed - value); });
        }

    return forced;
}


VectorR ResponseOptimization::FeasibilitySystem::evaluate(const VectorR& point,
                                                          VectorR& values,
                                                          VectorR& residuals,
                                                          const VectorR& output) const
{
    const VectorR response =
        output.size() > 0
        ? output
        : VectorR(problem->network->calculate_outputs(point.transpose()).row(0).transpose());

    const Index rows_number = Index(problem->constraints.size());

    values.resize(rows_number);
    residuals.resize(rows_number);

    for (Index i = 0; i < rows_number; i++)
    {
        const Constraint& constraint = problem->constraints[size_t(i)];

        values(i) = constraint.equation.evaluate(point, response);

        residuals(i) = problem->calculate_band_residual(problem->get_band(constraint), values(i));
    }

    return response;
}


MatrixR ResponseOptimization::FeasibilitySystem::calculate_jacobian(const VectorR& point,
                                                                    const VectorR& values,
                                                                    const VectorR& output,
                                                                    const VectorR& residuals) const
{
    const bool every_row = residuals.size() != values.size();

    const float difference_step = problem->difference_step;

    VectorR steps = difference_step*(borders.second - borders.first).cwiseMax(0.0f);

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(problem->network->get_input_variables()))
        steps.segment(first_column, categories_number).setZero();

    for (Index j = 0; j < steps.size(); j++)
        steps(j) = (steps(j) > EPSILON) ? max(steps(j), difference_step*abs(point(j))) : 0.0f;

    for (const Constraint& constraint : problem->constraints)
        if (const Index column = get_discrete_column(constraint); column >= 0)
            steps(column) = min(steps(column), problem->discrete_difference_step);

    MatrixR jacobian = MatrixR::Zero(values.size(), point.size());

    vector<Index> probed_rows;

    bool probe_reads_output = false;

    VectorR gradient(point.size());

    for (Index i = 0; i < values.size(); i++)
    {
        if (!every_row && residuals(i) == 0.0f) continue;

        const CompiledExpression& expression = problem->constraints[size_t(i)].equation;

        if (!is_output_coupled(expression))
        {
            evaluate_input_gradient(expression, point, output, gradient);

            if (gradient.allFinite())
            {
                jacobian.row(i) = gradient.transpose();

                continue;
            }
        }

        probed_rows.push_back(i);

        probe_reads_output = probe_reads_output || is_output_coupled(expression);
    }

    for (Index j = 0; j < point.size(); j++)
        if (steps(j) <= EPSILON)
            jacobian.col(j).setZero();

    if (probed_rows.empty()) return jacobian;

    VectorR probe = point;

    VectorR probe_values;
    VectorR probe_residuals;

    for (Index j = 0; j < point.size(); j++)
    {
        if (steps(j) <= EPSILON) continue;

        float step = steps(j);

        if (point(j) + step > borders.second(j)) step = -step;

        if (point(j) + step < borders.first(j)) continue;

        probe(j) = point(j) + step;

        if (probe_reads_output)
            evaluate(probe, probe_values, probe_residuals);
        else
            evaluate(probe, probe_values, probe_residuals, output);

        probe(j) = point(j);

        if (!probe_values.allFinite()) continue;

        for (const Index i : probed_rows)
            jacobian(i, j) = (probe_values(i) - values(i))/step;
    }

    return jacobian;
}


namespace
{

struct FeasibilityFunctor : Eigen::DenseFunctor<float>
{
    FeasibilityFunctor(const FeasibilitySystem& new_feasibility_system,
                       const VectorR& point,
                       const VectorR& values,
                       const VectorR& output,
                       const float box_weight,
                       const float new_unmeasurable_residual)
        : Eigen::DenseFunctor<float>(int(point.size()), int(values.size() + point.size())),
          feasibility_system(new_feasibility_system),
          unmeasurable_residual(new_unmeasurable_residual),
          box_scales((new_feasibility_system.borders.second - new_feasibility_system.borders.first).cwiseMax(EPSILON)/box_weight),
          row_scales(new_feasibility_system.calculate_jacobian(point, values, output).rowwise().norm())
    {
        for (Index i = 0; i < row_scales.size(); i++)
            if (!isfinite(row_scales(i)) || row_scales(i) <= EPSILON)
                row_scales(i) = 1.0f;
    }


    int operator()(const VectorR& point, VectorR& violations) const
    {
        feasibility_system.evaluate(point, row_values, row_residuals);

        violations.resize(row_values.size() + point.size());

        for (Index i = 0; i < row_values.size(); i++)
            violations(i) = isfinite(row_values(i)) ? row_residuals(i)/row_scales(i) : unmeasurable_residual;

        const auto& [lower, upper] = feasibility_system.borders;

        violations.tail(point.size()) =
            ((point - upper).cwiseMax(0.0f) + (point - lower).cwiseMin(0.0f)).cwiseQuotient(box_scales);

        return 0;
    }


    int df(const VectorR& point, JacobianType& jacobian) const
    {
        const VectorR output = feasibility_system.evaluate(point, row_values, row_residuals);

        const MatrixR value_jacobian =
            feasibility_system.calculate_jacobian(point, row_values, output, row_residuals);

        jacobian.setZero(row_values.size() + point.size(), point.size());

        for (Index i = 0; i < row_values.size(); i++)
            if (row_residuals(i) != 0.0f && isfinite(row_residuals(i)))
                jacobian.row(i) = value_jacobian.row(i)/row_scales(i);

        const auto& [lower, upper] = feasibility_system.borders;

        for (Index j = 0; j < point.size(); j++)
            if (point(j) > upper(j) || point(j) < lower(j))
                jacobian(row_values.size() + j, j) = 1.0f/box_scales(j);

        return 0;
    }

    const FeasibilitySystem& feasibility_system;

    float unmeasurable_residual = 0.0f;

    VectorR box_scales;

    VectorR row_scales;

    mutable VectorR row_values;
    mutable VectorR row_residuals;
};

}


pair<VectorR, VectorR> ResponseOptimization::FeasibilitySystem::solve(VectorR point) const
{
    VectorR values;
    VectorR residuals;
    VectorR previous;

    for (Index i = 0; ; i++)
    {
        if (!point.allFinite()) return {};

        point = round_to_grid(point);

        const VectorR output = evaluate(point, values, residuals);

        if (!values.allFinite()) return {};

        if ((residuals.array() == 0.0f).all())
            return {point, output};

        if (i == problem->feasibility_rounds) return {};

        if (previous.size() == point.size()
         && ((point - previous).cwiseAbs().array()
             <= problem->numeric_tolerance*(borders.second - borders.first).cwiseMax(EPSILON).array()).all())
            return {};

        previous = point;

        FeasibilityFunctor functor(*this, point, values, output, problem->box_weight, 1.0f/problem->numeric_tolerance);

        Eigen::LevenbergMarquardt<FeasibilityFunctor> levenberg_marquardt(functor);

        levenberg_marquardt.setMaxfev(problem->feasibility_evaluations);
        levenberg_marquardt.setFtol(problem->numeric_tolerance);
        levenberg_marquardt.setXtol(problem->numeric_tolerance);
        levenberg_marquardt.setGtol(0.0f);

        levenberg_marquardt.minimize(point);
    }
}


void ResponseOptimization::set(Network* new_network)
{
    network = new_network;
}


void ResponseOptimization::set_iterations_number(const Index new_iterations_number)
{
    iterations_number = max<Index>(new_iterations_number, 1);
}


void ResponseOptimization::set_points_number(const Index new_points_number)
{
    points_number = max<Index>(new_points_number, 1);
}


void ResponseOptimization::set_feasibility_margin_factor(const float new_feasibility_margin_factor)
{
    feasibility_margin_factor = max(new_feasibility_margin_factor, 0.0f);
}


pair<VectorR, VectorR> ResponseOptimization::get_unconstrained_domain() const
{
    throw_if(!network, "The neural network has not been set.");

    const Scaling* scaling_layer = static_cast<const Scaling*>(network->get_first(LayerType::Scaling));

    throw_if(!scaling_layer, "The neural network has no scaling layer to take the input domain from.");

    return {scaling_layer->get_minimums(), scaling_layer->get_maximums()};
}


void ResponseOptimization::add_objective(const string& expression, const Objective::Sense sense, const float value)
{
    objectives.push_back(Objective{compile_expression(expression, network, "Objective"), sense, value});
}


void ResponseOptimization::add_constraint(const string& expression,
                                          const Constraint::Condition condition,
                                          const vector<float>& values)
{
    Constraint constraint{condition, values, {}};

    throw_if(ranges::any_of(constraint.values, [](const float value) { return !isfinite(value); }),
             "Constraint on '" + expression + "' has a value that is not a finite number.");

    if (condition == Condition::Cardinality)
    {
        const vector<Index> counted = get_group_members(expression, network);

        const Index counted_number = Index(counted.size());

        throw_if(constraint.values.empty() || constraint.values[0] < 0.0f
              || constraint.values[0] > float(counted_number) || constraint.values[0] != round(constraint.values[0]),
                 "Constraint on '" + expression + "' needs one whole number between 0 and the "
                 + to_string(counted_number) + " variables it counts.");

        if (constraint.values[0] == float(counted_number))
        {
            logging::warning() << "Warning: constraint on '" << expression << "' allows all "
                               << counted_number
                               << " of the variables it counts, so it restricts nothing.\n";
            return;
        }

        const auto [minimums, maximums] = get_unconstrained_domain();

        vector<float> counted_spans;

        for (const Index column : counted)
            counted_spans.push_back(maximums(column) - minimums(column));

        constraint.equation = compile_elementary_symmetric(counted, counted_spans, Index(constraint.values[0]) + 1,
                                                           constraint_tolerance);
        constraint.equation.text = expression;

        constraints.push_back(move(constraint));

        return;
    }

    constraint.equation = compile_expression(expression, network, "Constraint");

    if (condition == Condition::Integer || condition == Condition::AllowedSet)
        throw_if(constraint.equation.input_indices.size() == 1
                 && !(constraint.equation.linear_input_terms.size() == 1 && is_bare_variable(constraint.equation)),
                 "Constraint on '" + expression + "' reads a single input through an expression. "
                 "The Integer and AllowedSet conditions take that input alone, as in 'x1', "
                 "or an expression of several variables.");

    if (condition == Condition::Integer)
    {
        constraint.equation = compile_integrality(expression, network);
    }
    else if (condition == Condition::AllowedSet)
    {
        throw_if(constraint.values.empty(),
                 "Constraint on '" + expression + "' needs at least one allowed value.");

        ranges::sort(constraint.values);

        if (ranges::adjacent_find(constraint.values) != constraint.values.end())
            logging::warning() << "Warning: constraint on '" << expression << "' repeats allowed values.\n";

        constraint.values.erase(ranges::unique(constraint.values).begin(), constraint.values.end());

        if (constraint.values.size() > discrete_values_warning)
            logging::warning() << "Warning: constraint on '" << expression << "' lists "
                               << constraint.values.size()
                               << " allowed values. The repair drives a polynomial of that degree, "
                                  "which loses precision as the degree grows.\n";

        constraint.equation = compile_membership(expression, network, constraint.values);
    }
    else
    {
        const size_t values_number = (condition == Condition::Between) ? 2 : 1;

        throw_if(constraint.values.size() < values_number,
                 "Constraint on '" + expression + "' needs "
                 + to_string(values_number) + " value(s).");

        if (constraint.values.size() > values_number)
            logging::warning() << "Warning: constraint on '" << expression << "' only uses "
                               << values_number << " of the " << constraint.values.size()
                               << " values given.\n";

        if (condition == Condition::Between)
        {
            throw_if(constraint.values[0] > constraint.values[1],
                     "Constraint on '" + expression + "' is between " + to_string(constraint.values[0])
                     + " and " + to_string(constraint.values[1]) + ", an empty interval.");

            if (constraint.values[0] == constraint.values[1])
                logging::warning() << "Warning: constraint on '" << expression
                                   << "' is between two equal values. "
                                      "Use the Equal condition instead.\n";
        }
    }

    constraint.equation.text = expression;

    constraints.push_back(move(constraint));
}


MatrixR ResponseOptimization::perform_response_optimization()
{
    throw_if(objectives.empty(), "No objective has been set.");

    return objectives.size() > 1 ? multi_optimization() : single_optimization();
}


pair<VectorR, VectorR> ResponseOptimization::calculate_domain()
{
    feasibility_system.initialize();

    return feasibility_system.borders;
}


VectorR ResponseOptimization::calculate_random_input(const pair<VectorR, VectorR>& domain) const
{
    VectorR input(domain.first.size());

    for (Index i = 0; i < input.size(); i++)
        input(i) = random_uniform(domain.first(i), domain.second(i));

    assign_random_categories(input);

    return input;
}


void ResponseOptimization::assign_random_categories(VectorR& point, const float probability) const
{
    vector<char> closed_categories;
    vector<float> block;

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(network->get_input_variables()))
    {
        if (probability < 1.0f && random_uniform(0.0f, 1.0f) >= probability) continue;

        closed_categories.resize(size_t(categories_number));

        for (Index j = 0; j < categories_number; j++)
            closed_categories[size_t(j)] = (feasibility_system.borders.second(first_column + j) <= 0.0f) ? 1 : 0;

        if (!draw_k_hot(categories_number, 1, {}, closed_categories, block)) continue;

        for (Index j = 0; j < categories_number; j++)
            point(first_column + j) = block[size_t(j)];
    }
}


MatrixR ResponseOptimization::evaluate_objectives(const MatrixR& inputs, const MatrixR& outputs) const
{
    MatrixR objective_values(inputs.rows(), Index(objectives.size()));

    for (Index i = 0; i < inputs.rows(); i++)
    {
        const VectorR input = inputs.row(i).transpose();
        const VectorR output = outputs.row(i).transpose();

        for (Index j = 0; j < Index(objectives.size()); j++)
        {
            const Objective& objective = objectives[j];

            const float value = objective.expression.evaluate(input, output);

            objective_values(i, j) = (objective.sense == Objective::Sense::Maximize) ?  value
                                   : (objective.sense == Objective::Sense::Minimize) ? -value
                                                                                     : -abs(value - objective.value);
        }
    }

    return objective_values;
}


vector<Index> ResponseOptimization::calculate_pareto_front(const MatrixR& objective_values) const
{
    if (objective_values.rows() == 0) return {};

    VectorR tolerance(objective_values.cols());

    for (Index j = 0; j < objective_values.cols(); j++)
        tolerance(j) = get_bound_tolerance(objective_values.col(j).maxCoeff() - objective_values.col(j).minCoeff());


    const auto is_as_good_as = [&](const Index point, const Index other)
    {
        for (Index j = 0; j < objective_values.cols(); j++)
            if (objective_values(point, j) < objective_values(other, j) - tolerance(j))
                return false;

        return true;
    };


    vector<Index> pareto_front;

    pareto_front.reserve(objective_values.rows());

    for (Index i = 0; i < objective_values.rows(); i++)
    {
        if (ranges::any_of(pareto_front, [&](const Index j) { return is_as_good_as(j, i); }))
            continue;

        erase_if(pareto_front, [&](const Index j) { return is_as_good_as(i, j); });

        pareto_front.push_back(i);
    }

    return pareto_front;
}


vector<Index> ResponseOptimization::clean_front(const MatrixR& inputs, const MatrixR& outputs) const
{
    const MatrixR objective_values = evaluate_objectives(inputs, outputs);

    const vector<Index> pareto_front = calculate_pareto_front(objective_values);

    if (Index(pareto_front.size()) <= requested_front_size) return pareto_front;

    const MatrixR point_values = minmax_score(slice_rows(objective_values, pareto_front));

    const vector<Index> extremes = extreme_indices(point_values);

    const Index cluster_size = max(Index(1),
                                   Index(diversity_factor*float(requested_front_size))/Index(extremes.size()));

    vector<char> chosen(pareto_front.size(), 0);

    vector<Index> selection;

    selection.reserve(size_t(requested_front_size));

    for (const Index extreme : extremes)
    {
        const VectorI cluster = get_nearest_points(point_values,
                                                   point_values.row(extreme).transpose(),
                                                   cluster_size);

        for (Index i = 0; i < cluster.size() && Index(selection.size()) < requested_front_size; i++)
        {
            if (chosen[size_t(cluster(i))]) continue;

            chosen[size_t(cluster(i))] = 1;

            selection.push_back(cluster(i));
        }
    }

    farthest_point_fill(calculate_distances(point_values), selection, requested_front_size);

    for (Index& point : selection)
        point = pareto_front[size_t(point)];

    return selection;
}


}
