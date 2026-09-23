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


vector<Index> get_group_members(const string& expression, const Network* network)
{
    vector<Index> members;

    for (const string& entry : split_expression_list(expression))
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
            && is_bare_variable(constraint.equation) && !is_output_coupled(constraint.equation))
         ? constraint.equation.input_indices.front()
         : -1;
}

}


ResponseOptimization::ResponseOptimization(Network* new_network)
{
    set(new_network);
}


ResponseOptimization::~ResponseOptimization() = default;


float ResponseOptimization::get_bound_tolerance(const float bound) const
{
    return max(EPSILON, abs(bound)*numeric_tolerance);
}


string ResponseOptimization::get_input_column_name(const Index column) const
{
    if (network)
        for (const auto& [name, index] : get_variable_columns(network->get_input_variables()))
            if (index == column)
                return name;

    return "input column " + to_string(column);
}


pair<float, float> ResponseOptimization::get_constraint_bounds(const Constraint& constraint) const
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

    case Condition::Integer:
    case Condition::AllowedSet:   return {-constraint_tolerance, constraint_tolerance};

    case Condition::Cardinality:  return {-1.0f, 1.0f};
    }

    return {-unbounded, unbounded};
}


pair<VectorR, VectorR> ResponseOptimization::calculate_domain()
{
    pair<VectorR, VectorR> domain = get_unconstrained_domain();

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
        else if (constraint.condition != Condition::Integer && constraint.condition != Condition::AllowedSet
              && !is_output_coupled(expression)
              && expression.linearity == ExpressionLinearity::Linear
              && expression.linear_input_terms.size() == 1
              && abs(expression.linear_input_terms.front().second) > EPSILON)
        {
            const float coefficient = expression.linear_input_terms.front().second;

            const auto [constraint_lower, constraint_upper] = get_constraint_bounds(constraint);

            const float at_lower = (constraint_lower - expression.linear_constant)/coefficient;
            const float at_upper = (constraint_upper - expression.linear_constant)/coefficient;

            column = expression.linear_input_terms.front().first;
            lower = min(at_lower, at_upper);
            upper = max(at_lower, at_upper);
        }
        else
            continue;

        domain.first(column) = max(domain.first(column), lower);
        domain.second(column) = min(domain.second(column), upper);

        throw_if(domain.first(column) > domain.second(column),
                 "The constraints leave '" + get_input_column_name(column) + "' with an empty range ["
                 + to_string(domain.first(column)) + ", " + to_string(domain.second(column)) + "].");
    }

    for (const auto& [first, size] : get_categorical_blocks(network->get_input_variables()))
    {
        auto lower = domain.first.segment(first, size);
        auto upper = domain.second.segment(first, size);
        lower = lower.array().max(0.0f).ceil();
        upper = upper.array().min(1.0f).floor();
        throw_if((lower.array() > upper.array()).any() || lower.sum() > 1.0f || upper.sum() < 1.0f,
                 "The constraints leave categorical input '" + get_input_column_name(first) + "' with no valid category.");
        if (lower.sum() == 1.0f) upper = lower;
    }

    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition == Condition::Cardinality)
            for (const Index counted : constraint.equation.input_indices)
                throw_if(domain.first(counted) > get_bound_tolerance(domain.first(counted))
                      || domain.second(counted) < -get_bound_tolerance(domain.second(counted)),
                         "Constraint on '" + constraint.equation.text + "' counts '"
                         + get_input_column_name(counted) + "', whose range ["
                         + to_string(domain.first(counted)) + ", "
                         + to_string(domain.second(counted)) + "] excludes zero, so it can never be switched off.");

        const Index column = get_discrete_column(constraint);

        if (column < 0) continue;

        const float lower = domain.first(column);
        const float upper = domain.second(column);

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

    input_bounds = move(domain);

    return input_bounds;
}


struct ResponseOptimization::FeasibilityRepairSystem : Eigen::DenseFunctor<float>
{
    FeasibilityRepairSystem(const ResponseOptimization& new_problem, const Index inputs_number)
        : Eigen::DenseFunctor<float>(int(inputs_number), int(new_problem.constraints.size()) + int(inputs_number)),
          problem(new_problem),
          categorical_blocks(get_categorical_blocks(problem.network->get_input_variables()))
    {
        constraint_bounds.reserve(problem.constraints.size());

        for (const Constraint& constraint : problem.constraints)
        {
            constraint_bounds.push_back(problem.get_constraint_bounds(constraint));
            constraints_read_output = constraints_read_output || is_output_coupled(constraint.equation);
        }
    }


    pair<VectorR, VectorR> solve(VectorR point)
    {
        const auto& [lower_bounds, upper_bounds] = problem.input_bounds;

        VectorR previous;
        VectorR before_previous;

        for (Index i = 0; ; i++)
        {
            if (!point.allFinite()) return {};

            point = round_discrete(point);

            if (point.size() == 0 || !point.allFinite()) return {};

            const VectorR output = evaluate_constraints(point, row_values, row_residuals);

            if (!row_values.allFinite() || !row_residuals.allFinite()) return {};

            if ((row_residuals.array() == 0.0f).all())
            {
                const VectorR& response = evaluate_outputs(point);
                if (!response.allFinite()) return {};
                for (Index j = 0; j < Index(problem.objectives.size()); j++)
                {
                    const Objective& objective = problem.objectives[size_t(j)];
                    const float value = objective.expression.evaluate(point, response);
                    if (!isfinite(value) || (objective.sense == Objective::Sense::Fixed
                                         && !isfinite(value - objective.value)))
                    {
                        problem.invalid_objective = j;
                        return {};
                    }
                }
                return {point, response};
            }

            if (i == problem.feasibility_rounds) return {};

            if (previous.size() == point.size()
             && ((point - previous).cwiseAbs().array()
                 <= problem.numeric_tolerance*(upper_bounds - lower_bounds).cwiseMax(EPSILON).array()).all())
                return {};

            if (before_previous.size() == point.size() && (before_previous.array() == point.array()).all())
                return {};

            before_previous.swap(previous);
            previous = point;

            if (i == 0) box_scales = (upper_bounds - lower_bounds).cwiseMax(EPSILON)/box_weight;
            row_scales = calculate_jacobian(point, row_values, output).rowwise().norm();

            row_scales = (row_scales.array().isFinite() && (row_scales.array() > EPSILON))
                         .select(row_scales.array(), 1.0f);

            Eigen::LevenbergMarquardt<FeasibilityRepairSystem> levenberg_marquardt(*this);

            levenberg_marquardt.setMaxfev(problem.feasibility_evaluations);
            levenberg_marquardt.setFtol(problem.numeric_tolerance);
            levenberg_marquardt.setXtol(problem.numeric_tolerance);
            levenberg_marquardt.setGtol(0.0f);

            levenberg_marquardt.minimize(point);
        }
    }


    VectorR round_discrete(const VectorR& point) const
    {
        const auto& [lower_bounds, upper_bounds] = problem.input_bounds;

        VectorR forced = point;

        for (const Constraint& constraint : problem.constraints)
        {
            if (constraint.condition != Condition::Cardinality) continue;

            vector<Index> counted = constraint.equation.input_indices;

            ranges::stable_sort(counted, {}, [&](const Index column) { return -abs(forced(column)); });

            for (size_t j = size_t(constraint.values[0]); j < counted.size(); j++)
                forced(counted[j]) = 0.0f;
        }

        forced = forced.cwiseMax(lower_bounds).cwiseMin(upper_bounds);

        for (const auto& [first_column, categories_number] : categorical_blocks)
        {
            Index category = 0;

            const bool none_open =
                (upper_bounds.segment(first_column, categories_number).array() > 0.0f)
                .select(forced.segment(first_column, categories_number).array(), -MAX).maxCoeff(&category) == -MAX;

            if (none_open) return {};

            forced.segment(first_column, categories_number).setZero();

            forced(first_column + category) = 1.0f;
        }

        for (const Constraint& constraint : problem.constraints)
            if (const Index column = get_discrete_column(constraint); column >= 0)
            {
                const float value = forced(column);

                const float lower = lower_bounds(column);
                const float upper = upper_bounds(column);

                if (constraint.condition == Condition::Integer && ceil(lower) > floor(upper)) return {};

                forced(column) = (constraint.condition == Condition::Integer)
                               ? clamp(round(value), ceil(lower), floor(upper))
                               : *ranges::min_element(constraint.values, {}, [&](const float allowed)
                                     { return (allowed < lower || allowed > upper)
                                            ? numeric_limits<double>::infinity() : abs(double(allowed) - value); });
            }

        if ((forced.array() < lower_bounds.array()).any() || (forced.array() > upper_bounds.array()).any()) return {};

        for (const auto& [first, size] : categorical_blocks)
        {
            const auto block = forced.segment(first, size).array();
            if (!(block == 0.0f || block == 1.0f).all() || block.sum() != 1.0f) return {};
        }

        return forced;
    }


    const VectorR& evaluate_outputs(const VectorR& point) const
    {
        if (evaluated_output.size() == 0 || evaluated_point.size() != point.size()
         || (evaluated_point.array() != point.array()).any())
        {
            evaluated_output = problem.network->calculate_outputs(point.transpose()).row(0).transpose();
            evaluated_point = point;
        }

        return evaluated_output;
    }


    VectorR evaluate_constraints(const VectorR& point,
                                 VectorR& values,
                                 VectorR& residuals,
                                 const VectorR& output = {}) const
    {
        const VectorR response = output.size() == 0 && constraints_read_output ? evaluate_outputs(point) : output;

        const Index rows_number = Index(problem.constraints.size());

        values.resize(rows_number);
        residuals.resize(rows_number);

        for (Index i = 0; i < rows_number; i++)
        {
            const Constraint& constraint = problem.constraints[size_t(i)];

            values(i) = constraint.equation.evaluate(point, response);

            float value = values(i);
            if (constraint.condition == Condition::Integer)
                value -= round(value);
            else if (constraint.condition == Condition::AllowedSet)
                value -= *ranges::min_element(constraint.values, {},
                            [&](const float allowed) { return abs(double(value) - allowed); });

            residuals(i) = calculate_constraint_residual(constraint_bounds[size_t(i)], value);
        }

        return response;
    }


    MatrixR calculate_jacobian(const VectorR& point,
                              const VectorR& values,
                              const VectorR& output,
                              const VectorR& residuals = {}) const
    {
        const auto& [lower_bounds, upper_bounds] = problem.input_bounds;

        const bool every_row = residuals.size() != values.size();

        VectorR steps = (upper_bounds - lower_bounds).cwiseMax(0.0f);

        for (const auto& [first_column, categories_number] : categorical_blocks)
            steps.segment(first_column, categories_number).setZero();

        for (Index j = 0; j < steps.size(); j++)
            if (steps(j) > 0.0f)
                steps(j) = max(difference_step*max(steps(j), abs(point(j))), EPSILON);

        for (const Constraint& constraint : problem.constraints)
            if (const Index column = get_discrete_column(constraint); column >= 0)
                steps(column) = min(steps(column), discrete_difference_step);

        MatrixR jacobian = MatrixR::Zero(values.size(), point.size());

        vector<Index> probed_rows;

        VectorR gradient(point.size());

        for (Index i = 0; i < values.size(); i++)
        {
            if (!every_row && residuals(i) == 0.0f) continue;

            const CompiledExpression& expression = problem.constraints[size_t(i)].equation;

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
        }

        for (Index j = 0; j < point.size(); j++)
            if (steps(j) == 0.0f)
                jacobian.col(j).setZero();

        calculate_numerical_jacobian(point, values, steps, probed_rows, jacobian);

        return jacobian;
    }


    void calculate_numerical_jacobian(const VectorR& point,
                                      const VectorR& values,
                                      const VectorR& steps,
                                      const vector<Index>& rows,
                                      MatrixR& jacobian) const
    {
        if (rows.empty()) return;

        const auto& [lower_bounds, upper_bounds] = problem.input_bounds;

        const bool reads_output = ranges::any_of(rows, [&](const Index i)
            { return is_output_coupled(problem.constraints[size_t(i)].equation); });

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
                    probe_output = problem.network->calculate_outputs(probe.transpose()).row(0).transpose();

                bool complete = true;

                for (const Index i : rows)
                {
                    if (isfinite(jacobian(i, j))) continue;

                    const float value = problem.constraints[size_t(i)].equation.evaluate(probe, probe_output);
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


    float calculate_constraint_residual(const pair<float, float>& bounds,
                                        const float value) const
    {
        const auto [lower, upper] = bounds;

        float crossed_bound = 0.0f;

        if (value < lower - problem.get_bound_tolerance(lower))
            crossed_bound = lower;
        else if (value > upper + problem.get_bound_tolerance(upper))
            crossed_bound = upper;
        else
            return 0.0f;

        const float residual = value - crossed_bound;
        const float margin = problem.feasibility_margin_factor;

        const float inset = min(margin*max(abs(residual), margin*abs(crossed_bound)),
                                0.5f*(upper - lower));

        return residual + ((residual > 0.0f) ? inset : -inset);
    }


    int operator()(const VectorR& point, VectorR& violations) const
    {
        evaluate_constraints(point, row_values, row_residuals);

        violations.resize(row_values.size() + point.size());

        violations.head(row_values.size()) = row_values.array().isFinite()
            .select(row_residuals.array()/row_scales.array(), 1.0f/problem.numeric_tolerance);

        const auto& [lower, upper] = problem.input_bounds;

        violations.tail(point.size()) =
            ((point - upper).cwiseMax(0.0f) + (point - lower).cwiseMin(0.0f)).cwiseQuotient(box_scales);

        return 0;
    }


    int df(const VectorR& point, JacobianType& jacobian) const
    {
        const VectorR output = evaluate_constraints(point, row_values, row_residuals);

        const MatrixR value_jacobian =
            calculate_jacobian(point, row_values, output, row_residuals);

        jacobian.setZero(row_values.size() + point.size(), point.size());

        for (Index i = 0; i < row_values.size(); i++)
            if (row_residuals(i) != 0.0f && isfinite(row_residuals(i)))
                jacobian.row(i) = value_jacobian.row(i)/row_scales(i);

        const auto& [lower, upper] = problem.input_bounds;

        for (Index j = 0; j < point.size(); j++)
            if (point(j) > upper(j) || point(j) < lower(j))
                jacobian(row_values.size() + j, j) = 1.0f/box_scales(j);

        return 0;
    }

    const ResponseOptimization& problem;

    vector<pair<float, float>> constraint_bounds;
    vector<pair<Index, Index>> categorical_blocks;
    bool constraints_read_output = false;

    static constexpr float difference_step = 1e-3f;
    static constexpr float discrete_difference_step = 0.25f;
    static constexpr float box_weight = 100.0f;

    VectorR box_scales;
    VectorR row_scales;

    mutable VectorR row_values;
    mutable VectorR row_residuals;
    mutable VectorR evaluated_point;
    mutable VectorR evaluated_output;
};


pair<VectorR, VectorR> ResponseOptimization::solve(VectorR point) const
{
    FeasibilityRepairSystem feasibility_system(*this, point.size());

    return feasibility_system.solve(move(point));
}


void ResponseOptimization::set(Network* new_network)
{
    network = new_network;
    objectives.clear();
    constraints.clear();
    input_bounds = {};
    invalid_objective = -1;
}


string ResponseOptimization::get_sampling_failure() const
{
    return invalid_objective < 0 ? "No feasible point with finite outputs and objectives was found. "
         : "Objective '" + objectives[size_t(invalid_objective)].expression.text
           + "' has no finite value at some sampled points. ";
}


void ResponseOptimization::set_iterations_number(const Index new_iterations_number)
{
    iterations_number = max<Index>(new_iterations_number, 1);
}


void ResponseOptimization::set_points_number(const Index new_points_number)
{
    points_number = max<Index>(new_points_number, 1);
}


void ResponseOptimization::set_sampling_budget_multiplier(const Index new_sampling_budget_multiplier)
{
    sampling_budget_multiplier = max<Index>(new_sampling_budget_multiplier, 0);
}


void ResponseOptimization::set_maximum_consecutive_failures(const Index new_maximum_consecutive_failures)
{
    maximum_consecutive_failures = max<Index>(new_maximum_consecutive_failures, 0);
}


void ResponseOptimization::set_feasibility_rounds(const Index new_feasibility_rounds)
{
    feasibility_rounds = max<Index>(new_feasibility_rounds, 1);
}


void ResponseOptimization::set_feasibility_evaluations(const Index new_feasibility_evaluations)
{
    feasibility_evaluations = max<Index>(new_feasibility_evaluations, 1);
}


void ResponseOptimization::set_feasibility_margin_factor(const float new_feasibility_margin_factor)
{
    throw_if(!isfinite(new_feasibility_margin_factor), "The feasibility margin factor must be finite.");
    feasibility_margin_factor = max(new_feasibility_margin_factor, 0.0f);
}


pair<VectorR, VectorR> ResponseOptimization::get_unconstrained_domain() const
{
    throw_if(!network, "The neural network has not been set.");

    const Scaling* scaling_layer = static_cast<const Scaling*>(network->get_first(LayerType::Scaling));

    throw_if(!scaling_layer, "The neural network has no scaling layer to take the input domain from.");

    pair<VectorR, VectorR> domain{scaling_layer->get_minimums(), scaling_layer->get_maximums()};
    throw_if(domain.first.size() != network->get_inputs_number() || domain.first.size() == 0
          || domain.second.size() != domain.first.size(), "The input domain has incompatible dimensions.");
    throw_if(!domain.first.allFinite() || !domain.second.allFinite()
          || !(domain.second - domain.first).allFinite()
          || (domain.first.array() > domain.second.array()).any(), "The input domain must have finite, ordered bounds.");
    return domain;
}


void ResponseOptimization::add_objective(const string& expression, const Objective::Sense sense, const float value)
{
    throw_if(!isfinite(value), "The objective target must be finite.");
    objectives.push_back(Objective{compile_expression(expression, network, "Objective"), sense, value});
}


void ResponseOptimization::add_constraint(const string& expression,
                                          const Constraint::Condition condition,
                                          const vector<float>& values)
{
    Constraint constraint{{}, condition, values};

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

    if (condition == Condition::AllowedSet)
    {
        throw_if(constraint.values.empty(),
                 "Constraint on '" + expression + "' needs at least one allowed value.");

        ranges::sort(constraint.values);

        const auto duplicates = ranges::unique(constraint.values);

        if (!duplicates.empty())
            logging::warning() << "Warning: constraint on '" << expression << "' repeats allowed values.\n";

        constraint.values.erase(duplicates.begin(), duplicates.end());
    }
    else if (condition != Condition::Integer)
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
    invalid_objective = -1;
    throw_if(!network, "The neural network has not been set.");
    throw_if(objectives.empty(), "No objective has been set.");
    const Index multiplier = sampling_budget_multiplier > 0 ? sampling_budget_multiplier : iterations_number;
    throw_if(multiplier > numeric_limits<Index>::max()/points_number, "The sampling budget is too large.");

    return objectives.size() > 1 ? multi_optimization() : single_optimization();
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
            closed_categories[size_t(j)] = (input_bounds.second(first_column + j) <= 0.0f) ? 1 : 0;

        if (!draw_k_hot(categories_number, 1, {}, closed_categories, block)) continue;

        for (Index j = 0; j < categories_number; j++)
            point(first_column + j) = block[size_t(j)];
    }
}


MatrixR ResponseOptimization::evaluate_objectives(const MatrixR& inputs, const MatrixR& outputs) const
{
    MatrixR objective_values(inputs.rows(), Index(objectives.size()));

    VectorR input;
    VectorR output;

    for (Index i = 0; i < inputs.rows(); i++)
    {
        input = inputs.row(i).transpose();
        output = outputs.row(i).transpose();

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
