//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/response_optimization/response_optimization.h"

#include <Eigen/Cholesky>

#include "opennn/registry.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/response_optimization/expression_evaluator.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/statistics.h"
#include "opennn/core/tensor_operations.h"

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network)
{
    set(new_neural_network);
}


ResponseOptimization::~ResponseOptimization() = default;


pair<float, float> ResponseOptimization::Constraint::calculate_bounds() const
{
    const float unbounded = numeric_limits<float>::infinity();

    if (values.empty())
        return {-unbounded, unbounded};

    switch (condition)
    {
    case Condition::AllowedSet:
    {
        const auto [smallest, largest] = ranges::minmax(values);

        return {smallest, largest};
    }

    case Condition::Between:      return {values[0], values[1]};

    case Condition::Equal:        return {values[0], values[0]};

    case Condition::GreaterEqual:
    case Condition::Greater:      return {values[0], unbounded};

    case Condition::LessEqual:
    case Condition::Less:         return {-unbounded, values[0]};

    case Condition::Integer:
    case Condition::Cardinality:  break;
    }

    return {-unbounded, unbounded};
}


float ResponseOptimization::Constraint::calculate_residual(const VectorR& input, const VectorR& output) const
{
    if (values.empty())
        return NAN;

    const float value = expression.evaluate(input, output);

    if (condition == Condition::AllowedSet)
    {
        const float nearest = *ranges::min_element(values, {},
                                                   [value](const float allowed) { return abs(allowed - value); });

        return abs(value - nearest) <= bound_tolerance(nearest) ? NAN : value - nearest;
    }

    const auto [lower_bound, upper_bound] = calculate_bounds();

    if (value < lower_bound - bound_tolerance(lower_bound))
        return value - lower_bound;

    if (value > upper_bound + bound_tolerance(upper_bound))
        return value - upper_bound;

    return NAN;
}


float ResponseOptimization::Constraint::calculate_inset(const float residual, const float margin) const
{
    if (condition == Condition::Equal || condition == Condition::AllowedSet)
        return 0.0f;

    const auto [lower_bound, upper_bound] = calculate_bounds();

    const float inset = min(margin*abs(residual), 0.5f*(upper_bound - lower_bound));

    return (residual > 0.0f) ? inset : -inset;
}


MatrixR ResponseOptimization::estimate_jacobian(const VectorR& input,
                                                const VectorR& values,
                                                const pair<VectorR, VectorR>& domain) const
{
    MatrixR jacobian = MatrixR::Zero(values.size(), input.size());

    vector<char> categorical_columns(size_t(input.size()), 0);

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(neural_network->get_input_variables()))
        fill_n(categorical_columns.begin() + first_column, categories_number, 1);

    VectorR probe = input;

    for (Index j = 0; j < input.size(); j++)
    {
        if (categorical_columns[size_t(j)]) continue;

        const float span = domain.second(j) - domain.first(j);

        if (span <= 0.0f) continue;

        float step = difference_step*max(span, abs(input(j)));

        if (step <= EPSILON) continue;

        if (input(j) + step > domain.second(j)) step = -step;

        if (input(j) + step < domain.first(j)) continue;

        probe(j) = input(j) + step;

        const VectorR probe_output = neural_network->calculate_outputs(probe.transpose()).row(0).transpose();

        VectorR probe_values(values.size());

        for (Index i = 0; i < values.size(); i++)
            probe_values(i) = constraints[size_t(i)].expression.evaluate(probe, probe_output);

        probe(j) = input(j);

        if (!probe_values.allFinite()) continue;

        jacobian.col(j) = (probe_values - values)/step;
    }

    return jacobian;
}


pair<VectorR, VectorR> ResponseOptimization::get_feasible_point(VectorR input,
                                                               const pair<VectorR, VectorR>& domain) const
{
    const Index constraints_number = Index(constraints.size());

    const auto evaluate = [&](const VectorR& point, VectorR& point_values, VectorR& point_residuals)
    {
        const VectorR point_output = neural_network->calculate_outputs(point.transpose()).row(0).transpose();

        for (Index i = 0; i < constraints_number; i++)
        {
            const Constraint& constraint = constraints[size_t(i)];

            point_values(i) = constraint.expression.evaluate(point, point_output);

            const float residual = constraint.calculate_residual(point, point_output);

            point_residuals(i) = isfinite(residual)
                               ? residual + constraint.calculate_inset(residual, feasibility_margin)
                               : 0.0f;
        }

        return point_output;
    };

    input = assign_categories(input.cwiseMax(domain.first).cwiseMin(domain.second));

    VectorR values(constraints_number);
    VectorR residuals(constraints_number);

    VectorR output = evaluate(input, values, residuals);

    if ((residuals.array() == 0.0f).all())
        return {input, output};

    MatrixR jacobian = estimate_jacobian(input, values, domain);

    VectorR trial_values(constraints_number);
    VectorR trial_residuals(constraints_number);

    for (Index pass = 0; pass < repair_passes; pass++)
    {
        MatrixR system = MatrixR::Zero(constraints_number, input.size());
        VectorR scaled_residuals = VectorR::Zero(constraints_number);

        for (Index i = 0; i < constraints_number; i++)
        {
            const auto [lower_bound, upper_bound] = constraints[size_t(i)].calculate_bounds();

            if (!isfinite(lower_bound) && !isfinite(upper_bound)) continue;

            if (!jacobian.row(i).allFinite()) continue;

            const float norm = jacobian.row(i).norm();

            if (norm <= 0.0f) continue;

            system.row(i) = jacobian.row(i)/norm;
            scaled_residuals(i) = residuals(i)/norm;
        }

        MatrixR gram = system*system.transpose();

        gram.diagonal().array() += EPSILON;

        const VectorR direction = -(system.transpose()*gram.ldlt().solve(scaled_residuals));

        if (direction.squaredNorm() <= EPSILON) break;

        VectorR trial, trial_output;

        float length = 1.0f;

        bool improved = false;

        for (Index attempt = 0; attempt < repair_passes; attempt++)
        {
            trial = assign_categories((input + length*direction).cwiseMax(domain.first).cwiseMin(domain.second));

            trial_output = evaluate(trial, trial_values, trial_residuals);

            if (trial_residuals.cwiseAbs().maxCoeff() < residuals.cwiseAbs().maxCoeff())
            {
                improved = true;
                break;
            }

            length *= 0.5f;
        }

        if (!improved) break;

        const VectorR step = trial - input;

        const float squared_length = step.squaredNorm();

        if (squared_length <= EPSILON) break;

        if (trial_values.allFinite() && values.allFinite())
            jacobian += (trial_values - values - jacobian*step)*step.transpose()/squared_length;

        input = trial;
        output = trial_output;
        values = trial_values;
        residuals = trial_residuals;

        if ((residuals.array() == 0.0f).all())
            return {input, output};
    }

    return {};
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void ResponseOptimization::add_objective(const string& expression, const Objective::Sense sense, const float value)
{
    objectives.push_back(Objective{compile_expression(expression, neural_network, "Objective"), sense, value});
}


void ResponseOptimization::add_constraint(const string& expression,
                                          const Constraint::Condition condition,
                                          const vector<float>& values)
{
    using Condition = Constraint::Condition;

    throw_if(condition == Condition::Cardinality,
             "Constraint on '" + expression + "' uses the Cardinality condition, which is not implemented.");

    throw_if(ranges::any_of(values, [](const float value) { return !isfinite(value); }),
             "Constraint on '" + expression + "' has a value that is not a finite number.");

    Constraint constraint{compile_expression(expression, neural_network, "Constraint"), condition, values};

    if (condition == Condition::AllowedSet)
    {
        throw_if(values.empty(), "Constraint on '" + expression + "' needs at least one allowed value.");

        vector<float> sorted_values = values;

        ranges::sort(sorted_values);

        if (ranges::adjacent_find(sorted_values) != sorted_values.end())
            cerr << "Warning: constraint on '" << expression << "' repeats allowed values.\n";
    }
    else if (condition == Condition::Integer)
    {
        throw_if(is_output_coupled(constraint.expression) || !is_bare_variable(constraint.expression),
                 "Constraint on '" + expression + "' asks for integer values of an expression. "
                 "The Integer condition applies to a single input variable.");

        cerr << "Warning: the integer condition on '" << expression
             << "' is recorded but not enforced by the current samplers.\n";
    }
    else
    {
        const size_t values_number = (condition == Condition::Between) ? 2 : 1;

        throw_if(values.size() < values_number,
                 "Constraint on '" + expression + "' needs " + to_string(values_number) + " value(s).");

        if (values.size() > values_number)
            cerr << "Warning: constraint on '" << expression << "' only uses "
                 << values_number << " of the " << values.size() << " values given.\n";

        if (condition == Condition::Between)
        {
            throw_if(values[0] > values[1],
                     "Constraint on '" + expression + "' is between " + to_string(values[0])
                     + " and " + to_string(values[1]) + ", an empty interval.");

            if (values[0] == values[1])
                cerr << "Warning: constraint on '" << expression << "' is between two equal values. "
                     << "Use the Equal condition instead.\n";
        }
    }

    constraints.push_back(move(constraint));
}


MatrixR ResponseOptimization::perform_response_optimization()
{
    throw_if(objectives.empty(), "No objective has been set.");

    return objectives.size() > 1 ? multi_optimization() : single_optimization();
}

pair<VectorR, VectorR> ResponseOptimization::calculate_domain() const
{
    throw_if(!neural_network, "The neural network has not been set.");

    const Scaling* scaling_layer = static_cast<const Scaling*>(neural_network->get_first(LayerType::Scaling));

    throw_if(!scaling_layer, "The neural network has no scaling layer to take the input domain from.");

    pair<VectorR, VectorR> domain = {scaling_layer->get_minimums(), scaling_layer->get_maximums()};

    for (const Constraint& constraint : constraints)
    {
        const CompiledExpression& expression = constraint.expression;

        if (is_output_coupled(expression)
         || expression.linearity != ExpressionLinearity::Linear
         || expression.linear_input_terms.size() != 1)
            continue;

        const auto [column, coefficient] = expression.linear_input_terms.front();

        if (abs(coefficient) <= EPSILON) continue;

        const auto [lower, upper] = constraint.calculate_bounds();

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


VectorR ResponseOptimization::calculate_random_input(const pair<VectorR, VectorR>& domain) const
{
    VectorR input(domain.first.size());

    for (Index i = 0; i < input.size(); i++)
        input(i) = random_uniform(domain.first(i), domain.second(i));

    vector<char> closed_categories;
    vector<float> block;

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(neural_network->get_input_variables()))
    {
        closed_categories.resize(size_t(categories_number));

        for (Index j = 0; j < categories_number; j++)
            closed_categories[size_t(j)] = (domain.second(first_column + j) <= 0.0f) ? 1 : 0;

        if (!draw_k_hot(categories_number, 1, {}, closed_categories, block)) continue;

        for (Index j = 0; j < categories_number; j++)
            input(first_column + j) = block[size_t(j)];
    }

    return input;
}


VectorR ResponseOptimization::assign_categories(const VectorR& input) const
{
    VectorR point = input;

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(neural_network->get_input_variables()))
    {
        Index category = 0;

        point.segment(first_column, categories_number).maxCoeff(&category);

        point.segment(first_column, categories_number).setZero();

        point(first_column + category) = 1.0f;
    }

    return point;
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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
