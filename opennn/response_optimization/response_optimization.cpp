//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/response_optimization/response_optimization.h"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "opennn/registry.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/layers/scaling_layer.h"
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


float nearest_discrete(const Constraint& constraint, const float value)
{
    if (constraint.condition == Condition::Integer)
        return round(value);

    return *ranges::min_element(constraint.values, {},
                                [value](const float allowed) { return abs(allowed - value); });
}


float discrete_measure(const Constraint& constraint, const float value)
{
    if (constraint.condition == Condition::Integer)
        return sin(numbers::pi_v<float>*value)/numbers::pi_v<float>;

    if (constraint.condition != Condition::AllowedSet)
        return value;

    const auto [smallest, largest] = ranges::minmax(constraint.values);

    const float span = max(largest - smallest, EPSILON);

    float measure = span;

    for (const float allowed : constraint.values)
        measure *= (value - allowed)/span;

    return measure;
}


Index count_activations(const vector<Constraint>& constraints)
{
    Index activations_number = 0;

    for (const Constraint& constraint : constraints)
        if (constraint.condition == Condition::Cardinality)
            activations_number += Index(constraint.expression.linear_input_terms.size());

    return activations_number;
}


// The variables a cardinality condition counts, written as a list: "x1; x2; x3".

CompiledExpression compile_group(const string& text, const NeuralNetwork* neural_network)
{
    vector<Index> members;

    for (const string_view entry : get_token_views(text, ';'))
    {
        const string name(trim_view(entry));

        throw_if(name.empty(), "Constraint on '" + text + "' leaves an empty entry in its list of variables.");

        const CompiledExpression member = compile_expression(name, neural_network, "Constraint");

        throw_if(is_output_coupled(member) || !is_bare_variable(member),
                 "Constraint on '" + text + "' lists '" + name + "', which is not a single input variable. "
                 "The Cardinality condition takes a list of input variables, as in 'x1; x2; x3'.");

        const Index column = member.linear_input_terms.front().first;

        throw_if(ranges::find(members, column) != members.end(),
                 "Constraint on '" + text + "' lists '" + name + "' twice.");

        members.push_back(column);
    }

    throw_if(members.size() < 2,
             "Constraint on '" + text + "' counts how many variables are in play. "
             "The Cardinality condition applies to a list of at least two input variables, as in 'x1; x2; x3'.");

    CompiledExpression group = compile_sum(members);

    group.text = text;

    return group;
}


void convert_cardinality(vector<Constraint>& constraints,
                         const Constraint& cardinality,
                         const VectorR& spans,
                         const float tolerance)
{
    const string& text = cardinality.expression.text;

    const Index first_activation = spans.size() + count_activations(constraints);

    const float activation_tolerance = 0.5f*tolerance;

    vector<Index> activations;

    for (const auto& [member, coefficient] : cardinality.expression.linear_input_terms)
    {
        const Index activation = first_activation + Index(activations.size());

        activations.push_back(activation);

        CompiledExpression coupling = compile_coupling(member, activation, spans(member));

        coupling.text = "coupling of " + text;

        constraints.push_back(Constraint{move(coupling), Condition::Between, {-tolerance, tolerance}});

        CompiledExpression binarity = compile_binarity(activation);

        binarity.text = "activation of " + text;

        constraints.push_back(Constraint{move(binarity), Condition::Between,
                                         {-activation_tolerance, activation_tolerance}});
    }

    CompiledExpression total = compile_sum(activations);

    total.text = "activations of " + text;

    constraints.push_back(Constraint{move(total), Condition::Equal, {cardinality.values[0]}});
}


Index get_discrete_column(const Constraint& constraint)
{
    if (constraint.condition != Condition::Integer && constraint.condition != Condition::AllowedSet)
        return -1;

    if (constraint.expression.linear_input_terms.size() != 1 || !is_bare_variable(constraint.expression))
        return -1;

    return constraint.expression.linear_input_terms.front().first;
}


pair<VectorR, VectorR> narrow_domain(pair<VectorR, VectorR> domain, const vector<const Constraint*>& rows)
{
    for (const Constraint* row : rows)
    {
        const CompiledExpression& expression = row->expression;

        if (is_output_coupled(expression)
         || expression.linearity != ExpressionLinearity::Linear
         || expression.linear_input_terms.size() != 1)
            continue;

        const auto [column, coefficient] = expression.linear_input_terms.front();

        if (abs(coefficient) <= EPSILON) continue;

        const auto [lower, upper] = row->calculate_bounds();

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


void check_domain(const pair<VectorR, VectorR>& domain,
                  const vector<const Constraint*>& rows,
                  const vector<Constraint>& constraints)
{
    for (const Constraint& constraint : constraints)
        if (constraint.condition == Condition::Cardinality)
            for (const auto& [column, coefficient] : constraint.expression.linear_input_terms)
                throw_if(domain.first(column) > bound_tolerance(domain.first(column))
                      || domain.second(column) < -bound_tolerance(domain.second(column)),
                         "Constraint on '" + constraint.expression.text + "' counts input column "
                         + to_string(column) + ", whose range [" + to_string(domain.first(column)) + ", "
                         + to_string(domain.second(column)) + "] excludes zero, so it can never be switched off.");

    for (const Constraint* row : rows)
    {
        const Index column = get_discrete_column(*row);

        if (column < 0) continue;

        const float lower = domain.first(column) - bound_tolerance(domain.first(column));
        const float upper = domain.second(column) + bound_tolerance(domain.second(column));

        if (row->condition == Condition::Integer)
            throw_if(ceil(lower) > floor(upper),
                     "Constraint on '" + row->expression.text + "' asks for a whole number in ["
                     + to_string(lower) + ", " + to_string(upper) + "], which holds none.");
        else
            throw_if(ranges::none_of(row->values,
                                     [&](const float allowed)
                                     { return allowed >= lower && allowed <= upper; }),
                     "Constraint on '" + row->expression.text + "' has no allowed value inside ["
                     + to_string(lower) + ", " + to_string(upper) + "].");
    }
}

}


ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network)
{
    system.problem = this;

    set(new_neural_network);
}


ResponseOptimization::~ResponseOptimization() = default;


pair<float, float> ResponseOptimization::Constraint::calculate_bounds() const
{
    const float unbounded = numeric_limits<float>::infinity();

    if (values.empty())
        return {-unbounded, unbounded};

    const float strict_offset = 2.0f*bound_tolerance(values[0]);

    switch (condition)
    {
    case Condition::AllowedSet:
    {
        const auto [smallest, largest] = ranges::minmax(values);

        return {smallest, largest};
    }

    case Condition::Between:      return {values[0], values[1]};

    case Condition::Equal:        return {values[0], values[0]};

    case Condition::GreaterEqual: return {values[0], unbounded};

    case Condition::Greater:      return {values[0] + strict_offset, unbounded};

    case Condition::LessEqual:    return {-unbounded, values[0]};

    case Condition::Less:         return {-unbounded, values[0] - strict_offset};

    case Condition::Integer:
    case Condition::Cardinality:  break;
    }

    return {-unbounded, unbounded};
}


float ResponseOptimization::Constraint::calculate_residual(const float value,
                                                           const float tolerance,
                                                           const float margin_factor) const
{
    if (condition == Condition::Integer || condition == Condition::AllowedSet)
        return abs(value - nearest_discrete(*this, value)) <= tolerance
             ? NAN
             : discrete_measure(*this, value);

    const auto [lower_bound, upper_bound] = calculate_bounds();

    float residual = 0.0f;
    float crossed_bound = 0.0f;

    if (value < lower_bound - bound_tolerance(lower_bound))
    {
        residual = value - lower_bound;
        crossed_bound = lower_bound;
    }
    else if (value > upper_bound + bound_tolerance(upper_bound))
    {
        residual = value - upper_bound;
        crossed_bound = upper_bound;
    }
    else
        return NAN;

    const float inset = min(margin_factor*max(abs(residual), margin_factor*abs(crossed_bound)),
                            0.5f*(upper_bound - lower_bound));

    return residual + ((residual > 0.0f) ? inset : -inset);
}


void ResponseOptimization::FeasibilitySystem::initialize()
{
    rows.clear();

    for (const Constraint& constraint : problem->constraints)
        if (constraint.condition == Condition::Integer
         || (constraint.condition != Condition::Cardinality && !constraint.values.empty()))
            rows.push_back(&constraint);

    const pair<VectorR, VectorR> domain = narrow_domain(problem->get_unconstrained_domain(), rows);

    check_domain(domain, rows, problem->constraints);

    reshape_borders(domain);
}


void ResponseOptimization::FeasibilitySystem::reshape_borders(const pair<VectorR, VectorR>& domain)
{
    const Index inputs_number = domain.first.size();

    const Index columns_number = inputs_number + count_activations(problem->constraints);

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

        Index first_activation = point.size();

        for (const Constraint& budget : problem->constraints)
        {
            if (budget.condition != Condition::Cardinality) continue;

            const auto& counted = budget.expression.linear_input_terms;

            vector<Index> positions(counted.size());

            iota(positions.begin(), positions.end(), Index(0));

            ranges::sort(positions, {},
                         [&](const Index position) { return -abs(point(counted[size_t(position)].first)); });

            for (Index j = 0; j < Index(budget.values[0]); j++)
                forced(first_activation + positions[size_t(j)]) = 1.0f;

            first_activation += Index(counted.size());
        }
    }

    forced = forced.cwiseMax(borders.first).cwiseMin(borders.second);

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(problem->neural_network->get_input_variables()))
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
    NeuralNetwork* network = problem->neural_network;

    const VectorR response =
        output.size() > 0
        ? output
        : VectorR(network->calculate_outputs(point.head(network->get_inputs_number()).transpose())
                         .row(0).transpose());

    values.resize(Index(rows.size()));
    residuals.resize(Index(rows.size()));

    for (Index i = 0; i < Index(rows.size()); i++)
    {
        const Constraint& row = *rows[size_t(i)];

        const float value = row.expression.evaluate(point, response);

        const float residual = row.calculate_residual(value,
                                                      problem->constraint_tolerance,
                                                      problem->feasibility_margin_factor);

        values(i) = discrete_measure(row, value);

        residuals(i) = isfinite(residual) ? residual : 0.0f;
    }

    return response;
}


MatrixR ResponseOptimization::FeasibilitySystem::calculate_jacobian(const VectorR& point,
                                                                    const VectorR& values,
                                                                    const VectorR& output,
                                                                    const VectorR& residuals) const
{
    const NeuralNetwork& network = *problem->neural_network;

    const Index inputs_number = network.get_inputs_number();

    const bool every_row = residuals.size() != values.size();

    VectorR steps = difference_step*(borders.second - borders.first).cwiseMax(0.0f);

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(network.get_input_variables()))
        steps.segment(first_column, categories_number).setZero();

    for (Index j = 0; j < steps.size(); j++)
        steps(j) = (steps(j) > EPSILON) ? max(steps(j), difference_step*abs(point(j))) : 0.0f;

    for (const Constraint* row : rows)
        if (const Index column = get_discrete_column(*row); column >= 0)
            steps(column) = min(steps(column), discrete_difference_step);

    MatrixR jacobian = MatrixR::Zero(values.size(), point.size());

    vector<Index> probed_rows;

    bool probe_reads_output = false;

    VectorR gradient(point.size());

    for (Index i = 0; i < values.size(); i++)
    {
        if (!every_row && residuals(i) == 0.0f) continue;

        const Constraint& row = *rows[size_t(i)];

        if (!is_output_coupled(row.expression)
         && row.condition != Condition::Integer
         && row.condition != Condition::AllowedSet)
        {
            evaluate_input_gradient(row.expression, point, output, gradient);

            if (gradient.allFinite())
            {
                jacobian.row(i) = gradient.transpose();

                continue;
            }
        }

        probed_rows.push_back(i);

        probe_reads_output = probe_reads_output || is_output_coupled(row.expression);
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

pair<VectorR, VectorR> place_discrete_variables(const FeasibilitySystem& system,
                                                const VectorR& point,
                                                const VectorR& output,
                                                const Index inputs_number)
{
    VectorR placed = point;

    for (const Constraint* row : system.rows)
        if (const Index column = get_discrete_column(*row); column >= 0)
            placed(column) = nearest_discrete(*row, placed(column));

    // Exactly, because the question here is whether a coordinate is on its lattice and not
    // whether an iteration converged. The default precision of isZero is 1e-5 for float, which
    // would return the point this function declined to place.
    if ((placed - point).isZero(0.0f))
        return {point.head(inputs_number), output};

    VectorR placed_values;
    VectorR placed_residuals;

    const VectorR placed_output = system.evaluate(placed, placed_values, placed_residuals);

    if (placed_values.allFinite() && (placed_residuals.array() == 0.0f).all())
        return {placed.head(inputs_number), placed_output};

    // Placing one variable on its lattice can move another row off its bound. Rather than lose
    // a point the search has already paid for, hold the discrete columns where they were placed
    // and let the repair work on what is left.
    FeasibilitySystem pinned = system;

    bool discrete_columns_already_pinned = true;

    for (const Constraint* row : system.rows)
        if (const Index column = get_discrete_column(*row); column >= 0)
        {
            discrete_columns_already_pinned = discrete_columns_already_pinned
                && system.borders.first(column) == system.borders.second(column);

            pinned.borders.first(column) = pinned.borders.second(column) = placed(column);
        }

    // Reached when this call is itself the re-solve: the columns are pinned already, so there
    // is nothing left to hold fixed and a second attempt would repeat the first. Stating it as
    // a property of the system rather than as a depth counter keeps it true however it is
    // reached, including when a caller pins a discrete variable by its own constraints.
    if (discrete_columns_already_pinned) return {};

    return pinned.solve(placed);
}


struct FeasibilityFunctor : Eigen::DenseFunctor<float>
{
    FeasibilityFunctor(const FeasibilitySystem& new_system,
                       const VectorR& point,
                       const VectorR& values,
                       const VectorR& output)
        : Eigen::DenseFunctor<float>(int(point.size()), int(values.size() + point.size())),
          system(new_system),
          row_scales(new_system.calculate_jacobian(point, values, output).rowwise().norm())
    {
        for (Index i = 0; i < row_scales.size(); i++)
            if (!isfinite(row_scales(i)) || row_scales(i) <= EPSILON)
                row_scales(i) = 1.0f;
    }


    VectorR calculate_box_scales() const
    {
        return (system.borders.second - system.borders.first).cwiseMax(EPSILON)/box_weight;
    }


    VectorR calculate_box_violations(const VectorR& point) const
    {
        return ((point - system.borders.second).cwiseMax(0.0f)
              + (point - system.borders.first).cwiseMin(0.0f)).cwiseQuotient(calculate_box_scales());
    }


    // calculate_violations: Eigen names it operator().

    int operator()(const VectorR& point, VectorR& violations) const
    {
        system.evaluate(point, row_values, row_residuals);

        violations.resize(row_values.size() + point.size());

        for (Index i = 0; i < row_values.size(); i++)
            violations(i) = isfinite(row_values(i)) ? row_residuals(i)/row_scales(i) : unmeasurable_residual;

        violations.tail(point.size()) = calculate_box_violations(point);

        return 0;
    }


    // calculate_violations_jacobian: Eigen names it df.

    int df(const VectorR& point, JacobianType& jacobian) const
    {
        const VectorR output = system.evaluate(point, row_values, row_residuals);

        const MatrixR value_jacobian = system.calculate_jacobian(point, row_values, output, row_residuals);

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

    const FeasibilitySystem& system;

    VectorR row_scales;

    // The system writes one value and one residual per row on every call. Kept here so that
    // the solver reuses the two vectors instead of sizing them again at each step.

    mutable VectorR row_values;
    mutable VectorR row_residuals;
};

}


pair<VectorR, VectorR> ResponseOptimization::FeasibilitySystem::solve(VectorR point) const
{
    const Index inputs_number = problem->neural_network->get_inputs_number();

    point = force_into_borders(point);

    VectorR values;
    VectorR residuals;

    VectorR output = evaluate(point, values, residuals);

    if (!values.allFinite()) return {};

    if ((residuals.array() == 0.0f).all())
        return place_discrete_variables(*this, point, output, inputs_number);

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
         ? place_discrete_variables(*this, point, output, inputs_number)
         : pair<VectorR, VectorR>();
}


void ResponseOptimization::set(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
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
    throw_if(!neural_network, "The neural network has not been set.");

    const Scaling* scaling_layer = static_cast<const Scaling*>(neural_network->get_first(LayerType::Scaling));

    throw_if(!scaling_layer, "The neural network has no scaling layer to take the input domain from.");

    return {scaling_layer->get_minimums(), scaling_layer->get_maximums()};
}


void ResponseOptimization::add_objective(const string& expression, const Objective::Sense sense, const float value)
{
    objectives.push_back(Objective{compile_expression(expression, neural_network, "Objective"), sense, value});
}


void ResponseOptimization::add_constraint(const string& expression,
                                          const Constraint::Condition condition,
                                          const vector<float>& values)
{
    throw_if(ranges::any_of(values, [](const float value) { return !isfinite(value); }),
             "Constraint on '" + expression + "' has a value that is not a finite number.");

    Constraint constraint{condition == Condition::Cardinality
                          ? compile_group(expression, neural_network)
                          : compile_expression(expression, neural_network, "Constraint"),
                          condition,
                          values};

    if (condition == Condition::AllowedSet)
    {
        throw_if(values.empty(), "Constraint on '" + expression + "' needs at least one allowed value.");

        vector<float> sorted_values = values;

        ranges::sort(sorted_values);

        if (ranges::adjacent_find(sorted_values) != sorted_values.end())
            cerr << "Warning: constraint on '" << expression << "' repeats allowed values.\n";

        if (values.size() > discrete_values_warning)
            cerr << "Warning: constraint on '" << expression << "' lists " << values.size()
                 << " allowed values. The repair drives a polynomial of that degree, which loses "
                    "precision as the degree grows.\n";
    }
    else if (condition == Condition::Integer)
    {
        throw_if(is_output_coupled(constraint.expression) || !is_bare_variable(constraint.expression),
                 "Constraint on '" + expression + "' asks for integer values of an expression. "
                 "The Integer condition applies to a single input variable.");
    }
    else if (condition == Condition::Cardinality)
    {
        const size_t members_number = constraint.expression.linear_input_terms.size();

        throw_if(values.empty() || values[0] < 0.0f || values[0] > float(members_number)
              || values[0] != round(values[0]),
                 "Constraint on '" + expression + "' needs one whole number between 0 and the "
                 + to_string(members_number) + " variables it counts.");

        if (values[0] == float(members_number))
            cerr << "Warning: constraint on '" << expression << "' allows all "
                 << members_number << " of the variables it counts, so it restricts nothing.\n";

        const auto [minimums, maximums] = get_unconstrained_domain();

        convert_cardinality(constraints, constraint, maximums - minimums, constraint_tolerance);
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


pair<VectorR, VectorR> ResponseOptimization::calculate_domain()
{
    system.initialize();

    const Index inputs_number = neural_network->get_inputs_number();

    return {system.borders.first.head(inputs_number), system.borders.second.head(inputs_number)};
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
         get_categorical_blocks(neural_network->get_input_variables()))
    {
        if (probability < 1.0f && random_uniform(0.0f, 1.0f) >= probability) continue;

        closed_categories.resize(size_t(categories_number));

        for (Index j = 0; j < categories_number; j++)
            closed_categories[size_t(j)] = (system.borders.second(first_column + j) <= 0.0f) ? 1 : 0;

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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
