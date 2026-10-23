// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#include "opennn/response_optimization/response_optimization.h"

#include "opennn/response_optimization/feasibility_repair_system.h"

#include "opennn/network/layers/layer_registry.h"
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


float ResponseOptimization::Objective::evaluate_score(const VectorR& input, const VectorR& output) const
{
    const float result = expression.evaluate(input, output);
    return sense == Sense::Maximize ? result : sense == Sense::Minimize ? -result : -abs(result - value);
}


void ResponseOptimization::Constraint::validate() const
{
    throw_if(condition < Condition::Equal || condition > Condition::Cardinality, "Invalid constraint condition.");
    throw_if(ranges::any_of(values, [](const float value) { return !isfinite(value); }),
             "Constraint on '" + equation.text + "' has a value that is not a finite number.");
    const size_t required = condition == Condition::Integer ? 0 : condition == Condition::Between ? 2 : 1;
    throw_if(values.size() < required,
             "Constraint on '" + equation.text + "' needs " + to_string(required) + " value(s).");
    if (condition == Condition::Integer || condition == Condition::AllowedSet)
        throw_if(equation.input_indices.size() == 1
                 && !(equation.linear_input_terms.size() == 1 && is_bare_variable(equation)),
                 "Constraint on '" + equation.text + "' reads a single input through an expression. "
                 "The Integer and AllowedSet conditions take that input alone, as in 'x1', "
                 "or an expression of several variables.");
    if (condition == Condition::Between)
        throw_if(values[0] > values[1], "Constraint on '" + equation.text + "' has an empty interval.");
    if (condition == Condition::Cardinality)
        throw_if(equation.input_indices.size() < 2 || values[0] < 0.0f
                 || double(values[0]) > double(equation.input_indices.size()) || values[0] != round(values[0]),
                 "Constraint on '" + equation.text + "' needs a whole budget within its group of inputs.");
}


Index ResponseOptimization::Constraint::get_discrete_column() const
{
    return ((condition == Condition::Integer || condition == Condition::AllowedSet)
            && is_bare_variable(equation) && !is_output_coupled(equation))
         ? equation.input_indices.front() : -1;
}


pair<float, float> ResponseOptimization::Constraint::get_bounds(const float numeric_tolerance,
                                                               const float constraint_tolerance) const
{
    const float strict_offset = condition == Condition::Greater || condition == Condition::Less
                              ? 2.0f*max(EPSILON, abs(values[0])*numeric_tolerance) : 0.0f;

    const float unbounded = numeric_limits<float>::infinity();

    switch (condition)
    {
    case Condition::Between:      return {values[0], values[1]};

    case Condition::Equal:        return {values[0], values[0]};

    case Condition::GreaterEqual: return {values[0], unbounded};

    case Condition::Greater:      return {values[0] + strict_offset, unbounded};

    case Condition::LessEqual:    return {-unbounded, values[0]};

    case Condition::Less:         return {-unbounded, values[0] - strict_offset};

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

        Index column = constraint.get_discrete_column();

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

            const auto [constraint_lower, constraint_upper] = constraint.get_bounds(numeric_tolerance, constraint_tolerance);

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

        const Index column = constraint.get_discrete_column();

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


FeasibilityRepairSystem ResponseOptimization::make_feasibility_system(const pair<VectorR, VectorR>& domain) const
{
    FeasibilityRepairSystem feasibility_system(network, constraints, domain,
                                               feasibility_rounds, feasibility_evaluations,
                                               numeric_tolerance, feasibility_margin_factor, constraint_tolerance);
    feasibility_system.set_solver(feasibility_solver == FeasibilitySolver::GaussNewton
        ? FeasibilityRepairSystem::Solver::GaussNewton : FeasibilityRepairSystem::Solver::LevenbergMarquardt);
    return feasibility_system;
}


pair<VectorR, VectorR> ResponseOptimization::get_feasible_points(VectorR point) const
{
    auto feasibility_system = make_feasibility_system(input_bounds);
    return get_feasible_points(move(point), feasibility_system);
}


pair<VectorR, VectorR> ResponseOptimization::get_feasible_points(VectorR point,
                                                               FeasibilityRepairSystem& feasibility_system) const
{
    // The repair runs in double precision. The searches keep the point in float, so the rounded point is
    // accepted only if it is still feasible in double: a repair from it must return it unchanged.
    const VectorR repaired = feasibility_system.solve_double(point.cast<double>()).first.cast<float>();
    const auto checked = repaired.size() == 0 ? pair<Eigen::VectorXd, VectorR>{}
                                              : feasibility_system.solve_double(repaired.cast<double>());
    pair<VectorR, VectorR> result{repaired, checked.second};
    if (checked.first.size() == 0 || checked.first != repaired.cast<double>())
    {
        const auto& failure = feasibility_system.get_closest_failure();
        if (failure.first < closest_failure.first) closest_failure = failure;
        return {};
    }
    for (Index i = 0; i < Index(objectives.size()); i++)
    {
        if (!isfinite(objectives[size_t(i)].evaluate_score(result.first, result.second)))
        {
            invalid_objective = i;
            return {};
        }
    }
    return result;
}


void ResponseOptimization::set(Network* new_network)
{
    network = new_network;
    objectives.clear();
    constraints.clear();
    input_bounds = {};
    invalid_objective = -1;
    closest_failure = {numeric_limits<double>::infinity(), {}};
}


string ResponseOptimization::get_sampling_failure() const
{
    return (invalid_objective < 0 ? "No feasible point with finite outputs and objectives was found. "
         : "Objective '" + objectives[size_t(invalid_objective)].expression.text
           + "' has no finite value at some sampled points. ") + closest_failure.second;
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


void ResponseOptimization::set_feasibility_solver(const FeasibilitySolver new_solver)
{
    throw_if(!is_one_of(new_solver, FeasibilitySolver::LevenbergMarquardt, FeasibilitySolver::GaussNewton),
             "Invalid feasibility solver.");
    feasibility_solver = new_solver;
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

    if (condition == Condition::Cardinality)
    {
        const vector<Index> counted = get_group_members(expression, network);
        const Index counted_number = Index(counted.size());
        constraint.equation.text = expression;
        constraint.equation.input_indices = counted;
        constraint.validate();

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
    constraint.validate();

    if (condition == Condition::AllowedSet)
    {
        ranges::sort(constraint.values);

        const auto duplicates = ranges::unique(constraint.values);

        if (!duplicates.empty())
            logging::warning() << "Warning: constraint on '" << expression << "' repeats allowed values.\n";

        constraint.values.erase(duplicates.begin(), duplicates.end());
    }
    else if (condition != Condition::Integer)
    {
        const size_t values_number = (condition == Condition::Between) ? 2 : 1;

        if (constraint.values.size() > values_number)
            logging::warning() << "Warning: constraint on '" << expression << "' only uses "
                               << values_number << " of the " << constraint.values.size()
                               << " values given.\n";

        if (condition == Condition::Between && constraint.values[0] == constraint.values[1])
            logging::warning() << "Warning: constraint on '" << expression
                               << "' is between two equal values. Use the Equal condition instead.\n";
    }

    constraints.push_back(move(constraint));
}


MatrixR ResponseOptimization::perform_response_optimization()
{
    invalid_objective = -1;
    closest_failure = {numeric_limits<double>::infinity(), {}};
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

    assign_random_categories(input, domain);

    return input;
}


void ResponseOptimization::assign_random_categories(VectorR& point, const float probability) const
{
    assign_random_categories(point, input_bounds, probability);
}


void ResponseOptimization::assign_random_categories(VectorR& point, const pair<VectorR, VectorR>& domain,
                                                    const float probability) const
{
    vector<char> closed_categories;
    vector<float> block;

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(network->get_input_variables()))
    {
        if (probability < 1.0f && random_uniform(0.0f, 1.0f) >= probability) continue;

        closed_categories.resize(size_t(categories_number));

        for (Index j = 0; j < categories_number; j++)
            closed_categories[size_t(j)] = (domain.second(first_column + j) <= 0.0f) ? 1 : 0;

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
            objective_values(i, j) = objectives[size_t(j)].evaluate_score(input, output);
    }

    return objective_values;
}


vector<Index> ResponseOptimization::calculate_pareto_front(const MatrixR& objective_values) const
{
    if (objective_values.rows() == 0) return {};

    const auto is_as_good_as = [&](const Index point, const Index other)
    {
        for (Index j = 0; j < objective_values.cols(); j++)
            if (!(objective_values(point, j) >= objective_values(other, j)))
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


pair<Index, VectorR> ResponseOptimization::get_advised_point(const MatrixR& results) const
{
    if (results.rows() == 0 || objectives.empty())
        return {-1, VectorR{}};

    const Index inputs_number = network->get_inputs_number();
    const MatrixR objective_values = evaluate_objectives(results.leftCols(inputs_number),
                                                        results.rightCols(results.cols() - inputs_number));
    const VectorR utopian = objective_values.colwise().maxCoeff().transpose();

    Eigen::VectorXd scale(objective_values.cols());
    for (Index j = 0; j < scale.size(); j++)
    {
        const double range = double(utopian(j)) - double(objective_values.col(j).minCoeff());
        scale(j) = range > 0.0 ? 1.0 / range : 1.0;
    }

    Index best_index = 0;
    double best_distance = numeric_limits<double>::infinity();
    for (Index i = 0; i < objective_values.rows(); i++)
    {
        double distance = 0.0;
        for (Index j = 0; j < objective_values.cols(); j++)
        {
            const double difference = (double(objective_values(i, j)) - double(utopian(j))) * scale(j);
            distance += difference * difference;
        }
        if (distance < best_distance) { best_distance = distance; best_index = i; }
    }
    return {best_index, results.row(best_index).transpose()};
}


}
