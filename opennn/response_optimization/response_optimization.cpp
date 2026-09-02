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

namespace
{

// The trust region the repair steps inside, as a damping on the solve. The rows of the system
// below are normalized, so its gram matrix has a unit diagonal and the damping is read against
// that scale rather than against the units the constraints happen to be written in.

constexpr float initial_damping = 1e-3f;
constexpr float maximum_damping = 1e6f;


// The violation the inset assumes when the real one has closed to nothing, as a share of the
// bound. Small on purpose: it only has to keep the aim off the surface, and every unit of it is
// asked of constraints that may not have it to give.

constexpr float minimum_violation = 0.1f;


float integer_measure(const float value)
{
    return sin(numbers::pi_v<float>*value)/numbers::pi_v<float>;
}


float allowed_set_measure(const float value, const vector<float>& values)
{
    const auto [smallest, largest] = ranges::minmax(values);

    const float span = max(largest - smallest, EPSILON);

    float measure = span;

    for (const float allowed : values)
        measure *= (value - allowed)/span;

    return measure;
}


// Levenberg-Marquardt step towards zero residual: each row of the jacobian is normalized so that
// no constraint dominates the others by unit alone, and the damped minimum-norm solution of the
// resulting system is taken through its gram matrix. Damping is what makes this a trust region:
// raising it shortens the step and swings it from Gauss-Newton towards steepest descent, which
// is how a step the model got wrong is retried without a line search along the same direction.

VectorR calculate_repair_direction(const vector<ResponseOptimization::Constraint>& constraints,
                                   const MatrixR& jacobian,
                                   const VectorR& residuals,
                                   const float damping)
{
    const Index constraints_number = Index(constraints.size());

    MatrixR system = MatrixR::Zero(constraints_number, jacobian.cols());
    VectorR scaled_residuals = VectorR::Zero(constraints_number);

    for (Index i = 0; i < constraints_number; i++)
    {
        if (!constraints[size_t(i)].is_enforced()) continue;

        if (!jacobian.row(i).allFinite()) continue;

        const float norm = jacobian.row(i).norm();

        if (norm <= 0.0f) continue;

        system.row(i) = jacobian.row(i)/norm;
        scaled_residuals(i) = residuals(i)/norm;
    }

    MatrixR gram = system*system.transpose();

    gram.diagonal().array() += damping;

    return -(system.transpose()*gram.ldlt().solve(scaled_residuals));
}

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


bool ResponseOptimization::Constraint::is_enforced() const
{
    if (condition == Condition::Cardinality) return false;

    return condition == Condition::Integer || !values.empty();
}


float ResponseOptimization::Constraint::calculate_measure(const VectorR& input, const VectorR& output) const
{
    if (!is_enforced()) return 0.0f;

    const float value = expression.evaluate(input, output);

    if (condition == Condition::Integer)    return integer_measure(value);

    if (condition == Condition::AllowedSet) return allowed_set_measure(value, values);

    return value;
}


float ResponseOptimization::Constraint::calculate_residual(const VectorR& input,
                                                           const VectorR& output,
                                                           const float margin) const
{
    if (!is_enforced())
        return NAN;

    const float value = expression.evaluate(input, output);

    if (condition == Condition::Integer)
        return abs(value - round(value)) <= lattice_tolerance ? NAN : integer_measure(value);

    if (condition == Condition::AllowedSet)
    {
        const float nearest = *ranges::min_element(values, {},
                                                   [value](const float allowed) { return abs(allowed - value); });

        return abs(value - nearest) <= lattice_tolerance ? NAN : allowed_set_measure(value, values);
    }

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

    // Aim past the bound rather than at it, so the repair lands inside the region and the solvers
    // get a set to search rather than a surface. The floor on the violation is what makes that
    // hold for a repair that converges: a share of the violation alone would recede onto the
    // surface along with it, leaving the repaired points sitting exactly on the bound.

    const float inset = min(margin*max(abs(residual), minimum_violation*abs(crossed_bound)),
                            0.5f*(upper_bound - lower_bound));

    return residual + ((residual > 0.0f) ? inset : -inset);
}


MatrixR ResponseOptimization::estimate_jacobian(const VectorR& input,
                                                const VectorR& values,
                                                const VectorR& output,
                                                const pair<VectorR, VectorR>& domain) const
{
    const Index inputs_number = input.size() - activation_variables;

    MatrixR jacobian = MatrixR::Zero(values.size(), input.size());

    vector<char> categorical_columns(size_t(input.size()), 0);
    vector<char> integer_columns(size_t(input.size()), 0);

    for (const auto& [first_column, categories_number] :
         get_categorical_blocks(neural_network->get_input_variables()))
        fill_n(categorical_columns.begin() + first_column, categories_number, 1);

    for (const Constraint& constraint : constraints)
        if (constraint.condition == Constraint::Condition::Integer)
            integer_columns[size_t(constraint.expression.linear_input_terms.front().first)] = 1;

    VectorR probe = input;

    for (Index j = 0; j < input.size(); j++)
    {
        if (categorical_columns[size_t(j)]) continue;

        const float span = domain.second(j) - domain.first(j);

        if (span <= 0.0f) continue;

        float step = difference_step*max(span, abs(input(j)));

        if (integer_columns[size_t(j)]) step = min(step, integer_difference_step);

        if (step <= EPSILON) continue;

        if (input(j) + step > domain.second(j)) step = -step;

        if (input(j) + step < domain.first(j)) continue;

        probe(j) = input(j) + step;

        const VectorR network_input = probe.head(inputs_number);

        const VectorR probe_output =
            (j < inputs_number)
            ? neural_network->calculate_outputs(network_input.transpose()).row(0).transpose()
            : output;

        VectorR probe_values(values.size());

        for (Index i = 0; i < values.size(); i++)
            probe_values(i) = constraints[size_t(i)].calculate_measure(probe, probe_output);

        probe(j) = input(j);

        if (!probe_values.allFinite()) continue;

        jacobian.col(j) = (probe_values - values)/step;
    }

    return jacobian;
}


VectorR ResponseOptimization::round_lattice(const VectorR& input) const
{
    VectorR point = input;

    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition != Constraint::Condition::Integer
         && constraint.condition != Constraint::Condition::AllowedSet) continue;

        if (constraint.expression.linear_input_terms.size() != 1
         || !is_bare_variable(constraint.expression)) continue;

        const Index column = constraint.expression.linear_input_terms.front().first;

        point(column) = (constraint.condition == Constraint::Condition::Integer)
                      ? round(point(column))
                      : *ranges::min_element(constraint.values, {},
                                             [value = point(column)](const float allowed)
                                             { return abs(allowed - value); });
    }

    return point;
}


VectorR ResponseOptimization::evaluate_constraints(const VectorR& point,
                                                   VectorR& values,
                                                   VectorR& residuals) const
{
    const VectorR network_input = point.head(point.size() - activation_variables);

    const VectorR output = neural_network->calculate_outputs(network_input.transpose()).row(0).transpose();

    for (Index i = 0; i < Index(constraints.size()); i++)
    {
        const Constraint& constraint = constraints[size_t(i)];

        values(i) = constraint.calculate_measure(point, output);

        const float residual = constraint.calculate_residual(point, output, feasibility_margin);

        residuals(i) = isfinite(residual) ? residual : 0.0f;
    }

    return output;
}


pair<VectorR, VectorR> ResponseOptimization::get_feasible_point(VectorR input,
                                                               const pair<VectorR, VectorR>& domain) const
{
    const Index constraints_number = Index(constraints.size());

    const Index inputs_number = input.size();

    const pair<VectorR, VectorR> search_domain = augment_domain(domain);

    input = augment_point(input);

    const auto clamp_to_domain = [&](const VectorR& point)
    {
        return assign_categories(point.cwiseMax(search_domain.first).cwiseMin(search_domain.second));
    };

    const auto finish = [&](const VectorR& point, const VectorR& point_output) -> pair<VectorR, VectorR>
    {
        const VectorR rounded = round_lattice(point);

        if ((rounded - point).isZero())
            return {point.head(inputs_number), point_output};

        VectorR rounded_values(constraints_number);
        VectorR rounded_residuals(constraints_number);

        const VectorR rounded_output = evaluate_constraints(rounded, rounded_values, rounded_residuals);

        if (!rounded_values.allFinite() || !(rounded_residuals.array() == 0.0f).all())
            return {};

        return {rounded.head(inputs_number), rounded_output};
    };

    input = clamp_to_domain(input);

    VectorR values(constraints_number);
    VectorR residuals(constraints_number);

    VectorR output = evaluate_constraints(input, values, residuals);

    if (!values.allFinite())
        return {};

    if ((residuals.array() == 0.0f).all())
        return finish(input, output);

    MatrixR jacobian = estimate_jacobian(input, values, output, search_domain);

    VectorR trial_values(constraints_number);
    VectorR trial_residuals(constraints_number);

    float damping = initial_damping;
    float rejection_growth = 2.0f;

    bool jacobian_is_secant = false;

    for (Index pass = 0; pass < repair_passes && damping < maximum_damping; pass++)
    {
        const VectorR trial =
            clamp_to_domain(input + calculate_repair_direction(constraints, jacobian, residuals, damping));

        const VectorR step = trial - input;

        const float squared_length = step.squaredNorm();

        if (squared_length <= EPSILON) break;

        const VectorR trial_output = evaluate_constraints(trial, trial_values, trial_residuals);

        // The step the domain allowed, not the one the solve asked for, is what the linear model
        // has to be judged on. A step cut short by a bound would otherwise read as the model
        // failing and damp the search for nothing.

        const float squared_residual = residuals.squaredNorm();

        const float reduction = squared_residual - trial_residuals.squaredNorm();

        const float predicted = squared_residual - (residuals + jacobian*step).squaredNorm();

        if (!trial_values.allFinite() || reduction <= 0.0f || predicted <= 0.0f)
        {
            damping *= rejection_growth;
            rejection_growth *= 2.0f;

            // A rejected step blames the model first, but a fresh jacobian costs one network pass
            // per variable, so it is only worth buying once the secant updates have had a chance
            // to drift away from it.

            if (jacobian_is_secant)
            {
                jacobian = estimate_jacobian(input, values, output, search_domain);

                jacobian_is_secant = false;
            }

            continue;
        }

        const float gain_excess = 2.0f*(reduction/predicted) - 1.0f;

        damping *= max(1.0f/3.0f, 1.0f - gain_excess*gain_excess*gain_excess);
        rejection_growth = 2.0f;

        if (trial_values.allFinite() && values.allFinite())
        {
            jacobian += (trial_values - values - jacobian*step)*step.transpose()/squared_length;

            jacobian_is_secant = true;
        }

        input = trial;
        output = trial_output;
        values = trial_values;
        residuals = trial_residuals;

        if ((residuals.array() == 0.0f).all())
            return finish(input, output);
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


void ResponseOptimization::expand_cardinality(const Constraint& cardinality)
{
    using Kind = ExpressionOp::Kind;

    const Scaling* scaling_layer = static_cast<const Scaling*>(neural_network->get_first(LayerType::Scaling));

    throw_if(!scaling_layer, "The neural network has no scaling layer to take the input domain from.");

    const VectorR spans = scaling_layer->get_maximums() - scaling_layer->get_minimums();

    const Index members_number = Index(cardinality.expression.linear_input_terms.size());

    const Index first_activation = spans.size() + activation_variables;

    CompiledExpression sum;

    sum.text = "activations of " + cardinality.expression.text;
    sum.linearity = ExpressionLinearity::Linear;
    sum.involvement = ExpressionInvolvement::InputsOnly;
    sum.complexity = ExpressionComplexity::Multivariate;

    for (Index j = 0; j < members_number; j++)
    {
        const Index member = cardinality.expression.linear_input_terms[size_t(j)].first;
        const Index activation = first_activation + j;

        const float inverse_span = 1.0f/max(spans(member), EPSILON);

        CompiledExpression coupling;

        coupling.text = "coupling of " + cardinality.expression.text;
        coupling.involvement = ExpressionInvolvement::InputsOnly;
        coupling.input_indices = {member, activation};
        coupling.operations = {{Kind::PushInput, member}, {Kind::PushInput, member},
                               {Kind::PushInput, activation}, {Kind::Mul}, {Kind::Sub},
                               {Kind::PushConst, 0, inverse_span}, {Kind::Mul}};

        constraints.push_back(Constraint{move(coupling), Constraint::Condition::Between,
                                         {-lattice_tolerance, lattice_tolerance}});

        CompiledExpression binarity;

        binarity.text = "activation of " + cardinality.expression.text;
        binarity.involvement = ExpressionInvolvement::InputsOnly;
        binarity.input_indices = {activation};
        binarity.operations = {{Kind::PushInput, activation}, {Kind::PushInput, activation}, {Kind::Mul},
                               {Kind::PushInput, activation}, {Kind::Sub}};

        constraints.push_back(Constraint{move(binarity), Constraint::Condition::Between,
                                         {-activation_tolerance, activation_tolerance}});

        sum.input_indices.push_back(activation);
        sum.linear_input_terms.emplace_back(activation, 1.0f);
    }

    constraints.push_back(Constraint{move(sum), Constraint::Condition::Equal, {cardinality.values[0]}});

    activation_variables += members_number;
}


void ResponseOptimization::add_constraint(const string& expression,
                                          const Constraint::Condition condition,
                                          const vector<float>& values)
{
    using Condition = Constraint::Condition;

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

        if (values.size() > lattice_values_warning)
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

        throw_if(is_output_coupled(constraint.expression)
              || constraint.expression.linearity != ExpressionLinearity::Linear
              || members_number < 2,
                 "Constraint on '" + expression + "' counts how many variables are in play. "
                 "The Cardinality condition applies to a sum of at least two input variables.");

        throw_if(values.empty() || values[0] < 0.0f || values[0] > float(members_number)
              || values[0] != round(values[0]),
                 "Constraint on '" + expression + "' needs one whole number between 0 and the "
                 + to_string(members_number) + " variables it counts.");

        if (values[0] == float(members_number))
            cerr << "Warning: constraint on '" << expression << "' allows all "
                 << members_number << " of the variables it counts, so it restricts nothing.\n";
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

    if (condition == Condition::Cardinality)
        expand_cardinality(constraint);

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

    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition != Constraint::Condition::Cardinality) continue;

        for (const auto& [column, coefficient] : constraint.expression.linear_input_terms)
            throw_if(domain.first(column) > bound_tolerance(domain.first(column))
                  || domain.second(column) < -bound_tolerance(domain.second(column)),
                     "Constraint on '" + constraint.expression.text + "' counts input column "
                     + to_string(column) + ", whose range [" + to_string(domain.first(column)) + ", "
                     + to_string(domain.second(column)) + "] excludes zero, so it can never be switched off.");
    }

    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition != Constraint::Condition::Integer
         && constraint.condition != Constraint::Condition::AllowedSet) continue;

        if (constraint.expression.linear_input_terms.size() != 1
         || !is_bare_variable(constraint.expression)) continue;

        const Index column = constraint.expression.linear_input_terms.front().first;

        const float lower = domain.first(column) - bound_tolerance(domain.first(column));
        const float upper = domain.second(column) + bound_tolerance(domain.second(column));

        if (constraint.condition == Constraint::Condition::Integer)
            throw_if(ceil(lower) > floor(upper),
                     "Constraint on '" + constraint.expression.text + "' asks for a whole number in ["
                     + to_string(lower) + ", " + to_string(upper) + "], which holds none.");
        else
            throw_if(ranges::none_of(constraint.values,
                                     [&](const float allowed)
                                     { return allowed >= lower && allowed <= upper; }),
                     "Constraint on '" + constraint.expression.text + "' has no allowed value inside ["
                     + to_string(lower) + ", " + to_string(upper) + "].");
    }

    return domain;
}


pair<VectorR, VectorR> ResponseOptimization::augment_domain(const pair<VectorR, VectorR>& domain) const
{
    if (activation_variables == 0) return domain;

    const Index inputs_number = domain.first.size();

    pair<VectorR, VectorR> augmented = {VectorR(inputs_number + activation_variables),
                                        VectorR(inputs_number + activation_variables)};

    augmented.first << domain.first, VectorR::Zero(activation_variables);
    augmented.second << domain.second, VectorR::Ones(activation_variables);

    return augmented;
}


VectorR ResponseOptimization::augment_point(const VectorR& input) const
{
    if (activation_variables == 0) return input;

    const Index inputs_number = input.size();

    VectorR point(inputs_number + activation_variables);

    point << input, VectorR::Zero(activation_variables);

    Index activation = inputs_number;

    for (const Constraint& constraint : constraints)
    {
        if (constraint.condition != Constraint::Condition::Cardinality) continue;

        const auto& members = constraint.expression.linear_input_terms;

        vector<Index> positions;

        for (Index j = 0; j < Index(members.size()); j++)
            positions.push_back(j);

        ranges::sort(positions, {},
                     [&](const Index position) { return -abs(input(members[size_t(position)].first)); });

        for (Index j = 0; j < Index(constraint.values[0]); j++)
            point(activation + positions[size_t(j)]) = 1.0f;

        activation += Index(members.size());
    }

    return point;
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
