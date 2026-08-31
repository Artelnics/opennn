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

MatrixR append_rows(const MatrixR& starting_matrix, const MatrixR& block)
{
    if (starting_matrix.size() == 0)
        return block;
    if (block.size() == 0)
        return starting_matrix;

    throw_if(starting_matrix.cols() != block.cols(),
             "append_rows: Column mismatch ({} vs {})",
             starting_matrix.cols(), block.cols());

    MatrixR final_matrix(starting_matrix.rows() + block.rows(), starting_matrix.cols());

    final_matrix.topRows(starting_matrix.rows()) = starting_matrix;
    final_matrix.bottomRows(block.rows()) = block;

    return final_matrix;
}


MatrixR append_columns(const MatrixR& first_matrix, const MatrixR& second_matrix)
{
    MatrixR result(first_matrix.rows(), first_matrix.cols() + second_matrix.cols());
    result.leftCols(first_matrix.cols()) = first_matrix;
    result.rightCols(second_matrix.cols()) = second_matrix;
    return result;
}


VectorR slice_rows(const VectorR& values, const vector<Index>& indices)
{
    VectorR result(ssize(indices));

    for (Index i = 0; i < ssize(indices); ++i)
        result(i) = values(indices[i]);

    return result;
}


MatrixR slice_rows(const MatrixR& matrix, const vector<Index>& indices)
{
    MatrixR result(ssize(indices), matrix.cols());

    for (Index i = 0; i < ssize(indices); ++i)
        result.row(i) = matrix.row(indices[i]);

    return result;
}


pair<MatrixR, MatrixR> slice_rows(const pair<MatrixR, MatrixR>& matrices, const vector<Index>& indices)
{
    return {slice_rows(matrices.first, indices), slice_rows(matrices.second, indices)};
}


pair<MatrixR, MatrixR> append_rows(const pair<MatrixR, MatrixR>& matrices, const pair<MatrixR, MatrixR>& blocks)
{
    return {append_rows(matrices.first, blocks.first), append_rows(matrices.second, blocks.second)};
}


MatrixR append_columns(const pair<MatrixR, MatrixR>& matrices)
{
    return append_columns(matrices.first, matrices.second);
}


MatrixR calculate_distances(const MatrixR& points)
{
    const VectorR squared_norms = points.rowwise().squaredNorm();

    MatrixR squared_distances = -2.0f * points * points.transpose();
    squared_distances.colwise() += squared_norms;
    squared_distances.rowwise() += squared_norms.transpose();

    return squared_distances.cwiseMax(0.0f).cwiseSqrt();
}


VectorI get_nearest_points(const MatrixR& matrix, const VectorR& point, Index neighbors_number)
{
    const Index rows = matrix.rows();

    const VectorR distances = (matrix.rowwise() - point.transpose()).rowwise().norm();

    vector<pair<float, Index>> pairs(rows);

    for (Index i = 0; i < rows; ++i)
        pairs[i] = {distances(i), i};

    if (neighbors_number > rows)
        neighbors_number = rows;

    partial_sort(pairs.begin(), pairs.begin() + neighbors_number, pairs.end());

    VectorI result(neighbors_number);
    transform(pairs.begin(), pairs.begin() + neighbors_number, result.data(),
              [](const auto& p) { return p.second; });
    return result;
}


vector<VectorI> nearest_neighbors(const MatrixR& distances, Index neighbors_number)
{
    const Index points_number = distances.rows();

    neighbors_number = min(neighbors_number, points_number - 1);

    vector<VectorI> neighbors(points_number);

    for (Index i = 0; i < points_number; i++)
    {
        VectorR point_distances = distances.row(i).transpose();

        point_distances(i) = MAX;

        neighbors[i] = maximal_indices(-point_distances, neighbors_number);
    }

    return neighbors;
}


VectorR neighbor_distances(const MatrixR& points, const Index neighbors_number)
{
    const Index points_number = points.rows();

    if (points_number < 2 || neighbors_number < 1)
        return VectorR::Constant(points_number, MAX);

    const MatrixR distances = calculate_distances(points);

    const vector<VectorI> neighbors = nearest_neighbors(distances, neighbors_number);

    VectorR farthest_neighbor_distances(points_number);

    for (Index i = 0; i < points_number; i++)
        farthest_neighbor_distances(i) = distances(i, neighbors[i](neighbors[i].size() - 1));

    return farthest_neighbor_distances;
}


VectorR local_outlier_factor(const MatrixR& points, Index neighbors_number)
{
    const Index points_number = points.rows();

    if (points_number <= 1 || neighbors_number <= 0)
        return VectorR::Ones(points_number);

    neighbors_number = min(neighbors_number, points_number - 1);

    const MatrixR distances = calculate_distances(points);

    vector<vector<Index>> neighbors(points_number);
    VectorR neighbor_distance(points_number);

    for (Index i = 0; i < points_number; i++)
    {
        VectorR row = distances.row(i).transpose();
        row(i) = MAX;
        const VectorI nearest = maximal_indices(-row, neighbors_number);
        neighbor_distance(i) = row(nearest(neighbors_number - 1));

        const float tie_tolerance = EPSILON * max(1.0f, abs(neighbor_distance(i)));
        neighbors[i].reserve(static_cast<size_t>(neighbors_number));
        for (Index j = 0; j < points_number; ++j)
            if (j != i && distances(i, j) <= neighbor_distance(i) + tie_tolerance)
                neighbors[i].push_back(j);
    }

    VectorR reachability_density(points_number);

    for (Index i = 0; i < points_number; i++)
    {
        float reachability_sum = 0.0f;
        for (const Index neighbor : neighbors[i])
            reachability_sum += max(neighbor_distance(neighbor), distances(i, neighbor));
        reachability_density(i) = reachability_sum > EPSILON
            ? float(neighbors[i].size()) / reachability_sum
            : MAX;
    }

    VectorR outlier_factor(points_number);

    for (Index i = 0; i < points_number; i++)
    {
        float density_sum = 0.0f;
        for (const Index neighbor : neighbors[i])
            density_sum += reachability_density(neighbor);
        outlier_factor(i) = reachability_density(i) > EPSILON
            ? density_sum / (float(neighbors[i].size()) * reachability_density(i))
            : 1.0f;
    }

    return outlier_factor;
}


void farthest_point_fill(const MatrixR& distances, vector<Index>& selection, const Index quota)
{
    const Index points_number = distances.rows();

    vector<char> chosen(points_number, 0);

    for (const Index point : selection)
        chosen[point] = 1;

    VectorR minimum_distance = VectorR::Constant(points_number, MAX);

    for (const Index point : selection)
        minimum_distance = minimum_distance.cwiseMin(distances.col(point));

    for (Index i = 0; i < points_number; i++)
        if (chosen[i]) minimum_distance(i) = -MAX;

    while (ssize(selection) < quota)
    {
        Index farthest = 0;
        minimum_distance.maxCoeff(&farthest);

        if (minimum_distance(farthest) < 0.0f) break;

        selection.push_back(farthest);
        minimum_distance = minimum_distance.cwiseMin(distances.col(farthest));
        minimum_distance(farthest) = -MAX;
    }
}


vector<Index> ranked_indices(const VectorR& data)
{
    vector<Index> indices(data.size());
    iota(indices.begin(), indices.end(), 0);

    sort(indices.begin(), indices.end(),
         [&data](Index i, Index j) {
             if (data(i) == data(j)) return i < j;
             return data(i) > data(j);
         });

    return indices;
}


VectorR minmax_score(const VectorR& values, const bool invert)
{
    const float smallest = values.minCoeff();
    const float range = values.maxCoeff() - smallest;

    if (range < EPSILON)
        return VectorR::Constant(values.size(), invert ? 1.0f : 0.0f);

    const VectorR normalized = (values.array() - smallest) / range;

    return VectorR(invert ? (1.0f - normalized.array()).matrix() : normalized);
}


MatrixR minmax_score(const MatrixR& values)
{
    MatrixR scores(values.rows(), values.cols());

    for (Index j = 0; j < values.cols(); j++)
        scores.col(j) = minmax_score(VectorR(values.col(j)));

    return scores;
}


vector<Index> extreme_indices(const MatrixR& values)
{
    vector<Index> extremes;

    extremes.reserve(size_t(2*values.cols()));

    for (Index j = 0; j < values.cols(); j++)
    {
        extremes.push_back(maximal_index(values.col(j)));
        extremes.push_back(minimal_index(values.col(j)));
    }

    return extremes;
}


bool row_dominates(const MatrixR& values, const Index a, const Index b)
{
    bool strictly_better = false;

    for (Index j = 0; j < values.cols(); ++j)
    {
        const float difference = values(a, j) - values(b, j);

        if (difference < 0.0f) return false;
        if (difference > 0.0f) strictly_better = true;
    }

    return strictly_better;
}


bool gauss_newton_step(const MatrixR& jacobian,
                       const VectorR& residuals,
                       const VectorR& inferior,
                       const VectorR& superior,
                       VectorR& point)
{
    vector<Index> projectable;
    projectable.reserve(jacobian.rows());

    for (Index i = 0; i < jacobian.rows(); ++i)
        if (jacobian.row(i).cwiseAbs().maxCoeff() > 0.0f)
            projectable.push_back(i);

    if (projectable.empty())
        return false;

    const MatrixR reduced_jacobian = slice_rows(jacobian, projectable);
    const VectorR reduced_residuals = slice_rows(residuals, projectable);

    MatrixR gram = reduced_jacobian * reduced_jacobian.transpose();
    gram.diagonal().array() += EPSILON;

    point -= reduced_jacobian.transpose() * gram.ldlt().solve(reduced_residuals);
    point = point.cwiseMax(inferior).cwiseMin(superior);

    return true;
}


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


VectorR ResponseOptimization::get_feasible_input(VectorR input, const pair<VectorR, VectorR>& domain) const
{
    input = assign_categories(input);

    const VectorR no_output;

    vector<const Constraint*> violated_constraints;
    vector<float> residuals;

    violated_constraints.reserve(constraints.size());
    residuals.reserve(constraints.size());

    for (Index pass = 0; pass < maximum_adjustment_passes; pass++)
    {
        violated_constraints.clear();
        residuals.clear();

        for (const Constraint& constraint : constraints)
        {
            // This pass repairs the input point on its own, with no network evaluation
            // behind it, so a constraint that reads an output has nothing to be measured
            // against here. Those are enforced by filtering the sampled points instead.
            if (is_output_coupled(constraint.expression))
                continue;

            bool holds_when_met = false;

            switch (constraint.condition)
            {
            case Constraint::Condition::Equal:
            case Constraint::Condition::AllowedSet:

                holds_when_met = true;
                break;

            case Constraint::Condition::Between:
            case Constraint::Condition::GreaterEqual:
            case Constraint::Condition::LessEqual:
            case Constraint::Condition::Greater:
            case Constraint::Condition::Less:

                break;

            // Not solved here

            case Constraint::Condition::Integer:

            // Not solved here

            case Constraint::Condition::Cardinality:

                continue;
            }

            const float residual = constraint.calculate_residual(input, no_output);

            if (!isfinite(residual) && !holds_when_met)
                continue;

            violated_constraints.push_back(&constraint);
            residuals.push_back(isfinite(residual) ? residual : 0.0f);
        }

        if (violated_constraints.empty())
            break;

        if (VectorR::Map(residuals.data(), Index(residuals.size())).cwiseAbs().maxCoeff() <= EPSILON)
            break;

        MatrixR constraints_jacobian(Index(violated_constraints.size()), input.size());
        VectorR scaled_residuals(Index(violated_constraints.size()));

        for (Index i = 0; i < Index(violated_constraints.size()); i++)
        {
            VectorR gradient = evaluate_input_gradient(violated_constraints[size_t(i)]->expression, input, no_output);

            if (!gradient.allFinite())
                gradient = VectorR::Zero(input.size());

            const float norm = gradient.norm();

            constraints_jacobian.row(i) = (norm > 0.0f ? VectorR(gradient/norm) : gradient).transpose();
            scaled_residuals(i) = (norm > 0.0f) ? residuals[size_t(i)]/norm : 0.0f;
        }

        const VectorR previous_input = input;

        if (!gauss_newton_step(constraints_jacobian, scaled_residuals, domain.first, domain.second, input))
            break;

        if ((input - previous_input).cwiseAbs().maxCoeff() <= EPSILON)
            break;
    }

    return input;
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

    // A point is as good as another when no objective of its own falls short of it.

    const auto is_as_good_as = [&](const Index point, const Index other)
    {
        for (Index j = 0; j < objective_values.cols(); j++)
            if (objective_values(point, j) < objective_values(other, j) - tolerance(j))
                return false;

        return true;
    };

    // The front is grown one point at a time, so each point is only ever weighed
    // against the front so far rather than against every other point.

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

    const MatrixR distances = calculate_distances(point_values);

    vector<char> chosen(pareto_front.size(), 0);

    vector<Index> selection;

    selection.reserve(size_t(requested_front_size));

    const auto select_point = [&](const Index point)
    {
        if (Index(selection.size()) >= requested_front_size || chosen[size_t(point)]) 
            return;

        chosen[size_t(point)] = 1;

        selection.push_back(point);
    };

    const auto select_ranked = [&](const VectorI& ranking, const Index ranked_quota)
    {
        for (Index i = 0, taken = 0; i < ranking.size() && taken < ranked_quota; i++)
            if (!chosen[size_t(ranking(i))])
            {
                select_point(ranking(i));

                taken++;
            }
    };

    const Index diversity_quota = Index(diversity_factor*float(requested_front_size));

    const vector<Index> extremes = extreme_indices(point_values);

    const Index extreme_cluster_size = max(Index(1), diversity_quota/Index(extremes.size()));

    for (const Index extreme : extremes)
    {
        const VectorI cluster = get_nearest_points(point_values,
                                                   point_values.row(extreme).transpose(),
                                                   extreme_cluster_size);

        for (Index i = 0; i < cluster.size(); i++)
            select_point(cluster(i));
    }

    select_ranked(maximal_indices(local_outlier_factor(point_values, density_neighbors_number),
                                  Index(pareto_front.size())),
                  diversity_quota);

    MatrixR gap_distances = distances;

    gap_distances.diagonal().setConstant(MAX);

    select_ranked(maximal_indices(gap_distances.rowwise().minCoeff(), Index(pareto_front.size())),
                  diversity_quota);

    farthest_point_fill(distances, selection, requested_front_size);

    for (Index& point : selection)
        point = pareto_front[size_t(point)];

    return selection;
}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
