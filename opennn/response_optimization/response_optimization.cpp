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


constexpr float relative_precision = 1e-6f;

constexpr float unmeasurable_residual = 1.0f/relative_precision;

constexpr float difference_step = 1e-3f;
constexpr float discrete_difference_step = 0.25f;

constexpr float box_weight = 100.0f;

constexpr size_t discrete_values_warning = 8;


float bound_tolerance(const float bound)
{
    return max(EPSILON, abs(bound)*relative_precision);
}


float band_residual(const pair<float, float>& band, const float value, const float margin_factor)
{
    const auto [lower, upper] = band;

    float residual = 0.0f;
    float crossed_bound = 0.0f;

    if (value < lower - bound_tolerance(lower))
    {
        residual = value - lower;
        crossed_bound = lower;
    }
    else if (value > upper + bound_tolerance(upper))
    {
        residual = value - upper;
        crossed_bound = upper;
    }
    else
        return 0.0f;

    const float inset = min(margin_factor*max(abs(residual), margin_factor*abs(crossed_bound)),
                            0.5f*(upper - lower));

    return residual + ((residual > 0.0f) ? inset : -inset);
}


pair<float, float> interval_band(const Condition condition, const vector<float>& values)
{
    const float unbounded = numeric_limits<float>::infinity();

    const float strict_offset = 2.0f*bound_tolerance(values[0]);

    switch (condition)
    {
    case Condition::Between:      return {values[0], values[1]};

    case Condition::Equal:        return {values[0], values[0]};

    case Condition::GreaterEqual: return {values[0], unbounded};

    case Condition::Greater:      return {values[0] + strict_offset, unbounded};

    case Condition::LessEqual:    return {-unbounded, values[0]};

    case Condition::Less:         return {-unbounded, values[0] - strict_offset};

    case Condition::AllowedSet:
    case Condition::Integer:
    case Condition::Cardinality:  break;
    }

    return {-unbounded, unbounded};
}


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
    if (constraint.condition != Condition::Integer && constraint.condition != Condition::AllowedSet)
        return -1;

    const CompiledExpression& expression = constraint.equations.front();

    if (expression.linear_input_terms.size() != 1 || !is_bare_variable(expression))
        return -1;

    return expression.linear_input_terms.front().first;
}


pair<VectorR, VectorR> narrow_domain(pair<VectorR, VectorR> domain,
                                     const vector<pair<const Constraint*, Index>>& rows)
{
    for (const auto& [constraint, row] : rows)
    {
        const CompiledExpression& expression = constraint->equations[size_t(row)];

        if (is_output_coupled(expression)
         || expression.linearity != ExpressionLinearity::Linear
         || expression.linear_input_terms.size() != 1)
            continue;

        const auto [column, coefficient] = expression.linear_input_terms.front();

        if (abs(coefficient) <= EPSILON) continue;

        const auto [lower, upper] = constraint->equation_limits[size_t(row)];

        const float at_lower = (lower - expression.linear_constant)/coefficient;
        const float at_upper = (upper - expression.linear_constant)/coefficient;

        domain.first(column) = max(domain.first(column), min(at_lower, at_upper));
        domain.second(column) = min(domain.second(column), max(at_lower, at_upper));

        throw_if(domain.first(column) > domain.second(column) + bound_tolerance(domain.second(column)),
                 "The constraints leave input column " + to_string(column) + " with an empty range ["
                 + to_string(domain.first(column)) + ", " + to_string(domain.second(column)) + "].");
    }

    return domain;
}


void check_domain(const pair<VectorR, VectorR>& domain, const vector<Constraint>& constraints)
{
    for (const Constraint& constraint : constraints)
    {
        for (const auto& [counted, switch_column] : constraint.involved_variables)
            throw_if(domain.first(counted) > bound_tolerance(domain.first(counted))
                  || domain.second(counted) < -bound_tolerance(domain.second(counted)),
                     "Constraint on '" + constraint.string_expression + "' counts input column "
                     + to_string(counted) + ", whose range [" + to_string(domain.first(counted)) + ", "
                     + to_string(domain.second(counted)) + "] excludes zero, so it can never be switched off.");

        const Index column = get_discrete_column(constraint);

        if (column < 0) continue;

        const float lower = domain.first(column) - bound_tolerance(domain.first(column));
        const float upper = domain.second(column) + bound_tolerance(domain.second(column));

        if (constraint.condition == Condition::Integer)
            throw_if(ceil(lower) > floor(upper),
                     "Constraint on '" + constraint.string_expression + "' asks for a whole number in ["
                     + to_string(lower) + ", " + to_string(upper) + "], which holds none.");
        else
            throw_if(ranges::none_of(constraint.values,
                                     [&](const float allowed)
                                     { return allowed >= lower && allowed <= upper; }),
                     "Constraint on '" + constraint.string_expression + "' has no allowed value inside ["
                     + to_string(lower) + ", " + to_string(upper) + "].");
    }
}

}


ResponseOptimization::ResponseOptimization(Network* new_network)
{
    feasibility_system.problem = this;

    set(new_network);
}


ResponseOptimization::~ResponseOptimization() = default;


void ResponseOptimization::Constraint::compile_equations(const Network* network,
                                                         const VectorR& spans,
                                                         const Index first_switch,
                                                         const float tolerance)
{
    throw_if(ranges::any_of(values, [](const float value) { return !isfinite(value); }),
             "Constraint on '" + string_expression + "' has a value that is not a finite number.");

    if (condition == Condition::Cardinality)
    {
        const vector<Index> members = get_group_members(string_expression, network);

        throw_if(values.empty() || values[0] < 0.0f || values[0] > float(members.size())
              || values[0] != round(values[0]),
                 "Constraint on '" + string_expression + "' needs one whole number between 0 and the "
                 + to_string(members.size()) + " variables it counts.");

        if (values[0] == float(members.size()))
            logging::warning() << "Warning: constraint on '" << string_expression << "' allows all "
                               << members.size()
                               << " of the variables it counts, so it restricts nothing.\n";

        vector<Index> switch_columns;

        for (const Index member : members)
        {
            const Index switch_column = first_switch + Index(involved_variables.size());

            involved_variables.emplace_back(member, switch_column);
            switch_columns.push_back(switch_column);

            equations.push_back(compile_coupling(member, switch_column, spans(member)));
            equation_limits.emplace_back(-tolerance, tolerance);

            equations.push_back(compile_binarity(switch_column));
            equation_limits.emplace_back(-0.5f*tolerance, 0.5f*tolerance);
        }

        equations.push_back(compile_sum(switch_columns));
        equation_limits.emplace_back(values[0], values[0]);

        return;
    }

    equations.push_back(compile_expression(string_expression, network, "Constraint"));

    if (condition == Condition::Integer)
    {
        throw_if(is_output_coupled(equations.front()) || !is_bare_variable(equations.front()),
                 "Constraint on '" + string_expression + "' asks for integer values of an expression. "
                 "The Integer condition applies to a single input variable.");

        const float unbounded = numeric_limits<float>::infinity();

        equation_limits.emplace_back(-unbounded, unbounded);

        equations.push_back(compile_integrality(string_expression, network));
        equation_limits.emplace_back(-tolerance, tolerance);
    }
    else if (condition == Condition::AllowedSet)
    {
        throw_if(values.empty(),
                 "Constraint on '" + string_expression + "' needs at least one allowed value.");

        ranges::sort(values);

        if (ranges::adjacent_find(values) != values.end())
            logging::warning() << "Warning: constraint on '" << string_expression << "' repeats allowed values.\n";

        values.erase(ranges::unique(values).begin(), values.end());

        if (values.size() > discrete_values_warning)
            logging::warning() << "Warning: constraint on '" << string_expression << "' lists "
                               << values.size()
                               << " allowed values. The repair drives a polynomial of that degree, "
                                  "which loses precision as the degree grows.\n";

        equation_limits.emplace_back(values.front(), values.back());

        const float band = tolerance*membership_scale(values);

        equations.push_back(compile_membership(string_expression, network, values));
        equation_limits.emplace_back(-band, band);
    }
    else
    {
        const size_t values_number = (condition == Condition::Between) ? 2 : 1;

        throw_if(values.size() < values_number,
                 "Constraint on '" + string_expression + "' needs "
                 + to_string(values_number) + " value(s).");

        if (values.size() > values_number)
            logging::warning() << "Warning: constraint on '" << string_expression << "' only uses "
                               << values_number << " of the " << values.size()
                               << " values given.\n";

        if (condition == Condition::Between)
        {
            throw_if(values[0] > values[1],
                     "Constraint on '" + string_expression + "' is between " + to_string(values[0])
                     + " and " + to_string(values[1]) + ", an empty interval.");

            if (values[0] == values[1])
                logging::warning() << "Warning: constraint on '" << string_expression
                                   << "' is between two equal values. "
                                      "Use the Equal condition instead.\n";
        }

        equation_limits.push_back(interval_band(condition, values));
    }
}


void ResponseOptimization::FeasibilitySystem::initialize()
{
    rows.clear();

    for (const Constraint& constraint : problem->constraints)
        for (Index row = 0; row < Index(constraint.equations.size()); row++)
            rows.emplace_back(&constraint, row);

    const pair<VectorR, VectorR> domain = narrow_domain(problem->get_unconstrained_domain(), rows);

    check_domain(domain, problem->constraints);

    reshape_borders(domain);
}


void ResponseOptimization::FeasibilitySystem::reshape_borders(const pair<VectorR, VectorR>& domain)
{
    const Index inputs_number = domain.first.size();

    Index columns_number = inputs_number;

    for (const Constraint& constraint : problem->constraints)
        columns_number += Index(constraint.involved_variables.size());

    borders = {VectorR::Zero(columns_number), VectorR::Ones(columns_number)};

    borders.first.head(inputs_number) = domain.first;
    borders.second.head(inputs_number) = domain.second;
}


VectorR ResponseOptimization::FeasibilitySystem::force_into_borders(const VectorR& point) const
{
    VectorR forced = point;

    if (forced.size() < borders.first.size())
    {
        forced = VectorR::Zero(borders.first.size());

        forced.head(point.size()) = point;

        for (const Constraint& budget : problem->constraints)
        {
            const auto& counted = budget.involved_variables;

            if (counted.empty()) continue;

            vector<Index> positions(counted.size());

            iota(positions.begin(), positions.end(), Index(0));

            ranges::sort(positions, {},
                         [&](const Index position) { return -abs(point(counted[size_t(position)].first)); });

            for (Index j = 0; j < Index(budget.values[0]); j++)
                forced(counted[size_t(positions[size_t(j)])].second) = 1.0f;
        }
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

    return forced;
}


VectorR ResponseOptimization::FeasibilitySystem::evaluate(const VectorR& point,
                                                          VectorR& values,
                                                          VectorR& residuals,
                                                          const VectorR& output) const
{
    Network* network = problem->network;

    const VectorR response =
        output.size() > 0
        ? output
        : VectorR(network->calculate_outputs(point.head(network->get_inputs_number()).transpose())
                         .row(0).transpose());

    values.resize(Index(rows.size()));
    residuals.resize(Index(rows.size()));

    for (Index i = 0; i < Index(rows.size()); i++)
    {
        const auto& [constraint, row] = rows[size_t(i)];

        values(i) = constraint->equations[size_t(row)].evaluate(point, response);

        residuals(i) = band_residual(constraint->equation_limits[size_t(row)],
                                     values(i),
                                     problem->feasibility_margin_factor);
    }

    return response;
}


MatrixR ResponseOptimization::FeasibilitySystem::calculate_jacobian(const VectorR& point,
                                                                    const VectorR& values,
                                                                    const VectorR& output,
                                                                    const VectorR& residuals) const
{
    const Network& network = *problem->network;

    const Index inputs_number = network.get_inputs_number();

    const bool every_row = residuals.size() != values.size();

    VectorR steps = difference_step*(borders.second - borders.first).cwiseMax(0.0f);

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(network.get_input_variables()))
        steps.segment(first_column, categories_number).setZero();

    for (Index j = 0; j < steps.size(); j++)
        steps(j) = (steps(j) > EPSILON) ? max(steps(j), difference_step*abs(point(j))) : 0.0f;

    for (const Constraint& constraint : problem->constraints)
        if (const Index column = get_discrete_column(constraint); column >= 0)
            steps(column) = min(steps(column), discrete_difference_step);

    MatrixR jacobian = MatrixR::Zero(values.size(), point.size());

    vector<Index> probed_rows;

    bool probe_reads_output = false;

    VectorR gradient(point.size());

    for (Index i = 0; i < values.size(); i++)
    {
        if (!every_row && residuals(i) == 0.0f) continue;

        const auto& [constraint, row] = rows[size_t(i)];

        const CompiledExpression& expression = constraint->equations[size_t(row)];

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

        if (probe_reads_output && j < inputs_number)
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

pair<VectorR, VectorR> settle_discrete_variables(const FeasibilitySystem& feasibility_system,
                                                 const vector<Constraint>& constraints,
                                                 const VectorR& point,
                                                 const VectorR& output,
                                                 const Index inputs_number)
{
    VectorR placed = point;

    for (const Constraint& constraint : constraints)
    {
        const Index column = get_discrete_column(constraint);

        if (column < 0) continue;

        const float value = placed(column);

        placed(column) = (constraint.condition == Condition::Integer)
                       ? round(value)
                       : *ranges::min_element(constraint.values, {},
                                              [value](const float allowed)
                                              { return abs(allowed - value); });
    }

    if ((placed - point).isZero(0.0f))
        return {point.head(inputs_number), output};

    VectorR placed_values;
    VectorR placed_residuals;

    const VectorR placed_output = feasibility_system.evaluate(placed, placed_values, placed_residuals);

    if (placed_values.allFinite() && (placed_residuals.array() == 0.0f).all())
        return {placed.head(inputs_number), placed_output};

    FeasibilitySystem pinned = feasibility_system;

    bool discrete_columns_already_pinned = true;

    for (const Constraint& constraint : constraints)
        if (const Index column = get_discrete_column(constraint); column >= 0)
        {
            discrete_columns_already_pinned = discrete_columns_already_pinned
                && feasibility_system.borders.first(column) == feasibility_system.borders.second(column);

            pinned.borders.first(column) = pinned.borders.second(column) = placed(column);
        }

    if (discrete_columns_already_pinned) return {};

    return pinned.solve(placed);
}


struct FeasibilityFunctor : Eigen::DenseFunctor<float>
{
    FeasibilityFunctor(const FeasibilitySystem& new_feasibility_system,
                       const VectorR& point,
                       const VectorR& values,
                       const VectorR& output)
        : Eigen::DenseFunctor<float>(int(point.size()), int(values.size() + point.size())),
          feasibility_system(new_feasibility_system),
          row_scales(new_feasibility_system.calculate_jacobian(point, values, output).rowwise().norm())
    {
        for (Index i = 0; i < row_scales.size(); i++)
            if (!isfinite(row_scales(i)) || row_scales(i) <= EPSILON)
                row_scales(i) = 1.0f;
    }


    VectorR calculate_box_scales() const
    {
        return (feasibility_system.borders.second - feasibility_system.borders.first).cwiseMax(EPSILON)/box_weight;
    }


    VectorR calculate_box_violations(const VectorR& point) const
    {
        return ((point - feasibility_system.borders.second).cwiseMax(0.0f)
              + (point - feasibility_system.borders.first).cwiseMin(0.0f)).cwiseQuotient(calculate_box_scales());
    }


    int operator()(const VectorR& point, VectorR& violations) const
    {
        feasibility_system.evaluate(point, row_values, row_residuals);

        violations.resize(row_values.size() + point.size());

        for (Index i = 0; i < row_values.size(); i++)
            violations(i) = isfinite(row_values(i)) ? row_residuals(i)/row_scales(i) : unmeasurable_residual;

        violations.tail(point.size()) = calculate_box_violations(point);

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

        const VectorR box_scales = calculate_box_scales();
        const VectorR box_violations = calculate_box_violations(point);

        for (Index j = 0; j < point.size(); j++)
            if (box_violations(j) != 0.0f)
                jacobian(row_values.size() + j, j) = 1.0f/box_scales(j);

        return 0;
    }

    const FeasibilitySystem& feasibility_system;

    VectorR row_scales;

    mutable VectorR row_values;
    mutable VectorR row_residuals;
};

}


pair<VectorR, VectorR> ResponseOptimization::FeasibilitySystem::solve(VectorR point) const
{
    const Index inputs_number = problem->network->get_inputs_number();

    point = force_into_borders(point);

    VectorR values;
    VectorR residuals;

    VectorR output = evaluate(point, values, residuals);

    if (!values.allFinite()) return {};

    if ((residuals.array() == 0.0f).all())
        return settle_discrete_variables(*this, problem->constraints, point, output, inputs_number);

    FeasibilityFunctor functor(*this, point, values, output);

    Eigen::LevenbergMarquardt<FeasibilityFunctor> levenberg_marquardt(functor);

    levenberg_marquardt.setMaxfev(problem->feasibility_evaluations);
    levenberg_marquardt.setFtol(relative_precision);
    levenberg_marquardt.setXtol(relative_precision);
    levenberg_marquardt.setGtol(0.0f);

    levenberg_marquardt.minimize(point);

    point = force_into_borders(point);

    output = evaluate(point, values, residuals);

    return values.allFinite() && (residuals.array() == 0.0f).all()
         ? settle_discrete_variables(*this, problem->constraints, point, output, inputs_number)
         : pair<VectorR, VectorR>();
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
    Constraint constraint{expression, condition, values};

    VectorR spans;

    Index first_switch = 0;

    if (condition == Condition::Cardinality)
    {
        const auto [minimums, maximums] = get_unconstrained_domain();

        spans = maximums - minimums;

        first_switch = spans.size();

        for (const Constraint& other : constraints)
            first_switch += Index(other.involved_variables.size());
    }

    constraint.compile_equations(network, spans, first_switch, constraint_tolerance);

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

    const Index inputs_number = network->get_inputs_number();

    return {feasibility_system.borders.first.head(inputs_number),
            feasibility_system.borders.second.head(inputs_number)};
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
        tolerance(j) = bound_tolerance(objective_values.col(j).maxCoeff() - objective_values.col(j).minCoeff());


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
