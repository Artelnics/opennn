//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_optimization.h"
#include "tensor_operations.h"
#include "statistics.h"
#include "neural_network.h"
#include "variable.h"
#include "scaling_layer.h"
#include "unscaling_layer.h"

namespace opennn
{

ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network)
{
    set(new_neural_network);
}


ResponseOptimization::~ResponseOptimization() = default;


void ResponseOptimization::set(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
    variables_descriptives.clear();
    network_jacobian = {};
}


void ResponseOptimization::set_constraint(const string& name, const ComparisonOperator comparison, float low, float up)
{
    constraint_set.univariate[name] = UnivariateConstraint(comparison, low, up);
}


void ResponseOptimization::set_constraint(const string& name, const vector<float>& allowed_values)
{
    throw_if(allowed_values.empty(),
             "ResponseOptimization: AllowedSet constraint for '" + name + "' needs at least one value");

    UnivariateConstraint constraint(ComparisonOperator::AllowedSet);
    constraint.allowed_values = allowed_values;
    constraint_set.univariate[name] = move(constraint);
}


void ResponseOptimization::set_cardinality_constraint(const vector<string>& variable_names, const Index k)
{
    throw_if(variable_names.empty(),
             "ResponseOptimization: cardinality constraint needs at least one indicator variable");

    throw_if(k < 0 || k > static_cast<Index>(variable_names.size()),
             "ResponseOptimization: cardinality target k=" + to_string(k)
             + " is out of range for " + to_string(variable_names.size()) + " indicator(s)");

    constraint_set.cardinality.push_back({ variable_names, k });
}


void ResponseOptimization::clear_cardinality_constraints()
{
    constraint_set.cardinality.clear();
}


void ResponseOptimization::set_objective(const string& name, const Sense sense)
{
    objectives[name] = sense;
}


void ResponseOptimization::set_time_role(const string& name, const TimeType role)
{
    time_roles[name] = role;
    variables_descriptives.clear();
}


vector<NamedColumn> ResponseOptimization::build_input_columns(const vector<Variable>& variables) const
{
    vector<NamedColumn> columns;
    columns.reserve(variables.size());

    Index column = 0;

    for (const Variable& variable : variables)
    {
        const Index dimension = variable.get_feature_count();

        if (variable.get_role() == "Input" && !is_history(variable.name) && dimension == 1)
            columns.push_back({variable.name, column});

        column += dimension;
    }

    return columns;
}


vector<NamedColumn> ResponseOptimization::build_output_columns(const vector<Variable>& variables) const
{
    vector<NamedColumn> columns;
    columns.reserve(variables.size());

    Index column = 0;

    for (const Variable& variable : variables)
    {
        if (variable.get_feature_count() == 1)
            columns.push_back({variable.name, column});

        column += variable.get_feature_count();
    }

    return columns;
}


void ResponseOptimization::set_formula_constraint(const string& expression,
                                                  const ComparisonOperator comparison,
                                                  const float low,
                                                  const float up)
{
    throw_if(!neural_network,
             "ResponseOptimization: set_formula_constraint requires a neural network to be set first");

    const vector<NamedColumn> input_columns = build_input_columns(neural_network->get_input_variables());
    const vector<NamedColumn> output_columns = build_output_columns(neural_network->get_output_variables());

    // Non-smooth (min/max/abs) constraints expand to a disjunction of smooth branches; a smooth
    // constraint or a top-level AND case stays a single branch added directly. A degenerate
    // expansion (e.g. a constant-valued piece) falls back to the original constraint, which is
    // still enforced by rejection at the feasibility filter.
    vector<vector<MultivariateConstraint>> branches;
    try
    {
        branches = expand_constraint(expression, comparison, low, up, input_columns, output_columns);
    }
    catch (const exception&)
    {
        MultivariateConstraint constraint;
        constraint.expression = expression;
        constraint.comparison_operator = comparison;
        constraint.low_bound = low;
        constraint.up_bound = up;
        constraint.compiled = compile_formula(expression, input_columns, output_columns);
        constraint.kind = classify(constraint);
        branches = { { move(constraint) } };
    }

    if (branches.size() == 1)
        for (MultivariateConstraint& constraint : branches[0])
            constraint_set.multivariate.push_back(move(constraint));
    else
        constraint_set.disjunctive.push_back(move(branches));

    network_jacobian.ready = false;
}


void ResponseOptimization::set_formula_constraint(function<float(const VectorR&, const VectorR&)> callback,
                                                  const ComparisonOperator comparison,
                                                  const float low,
                                                  const float up)
{
    MultivariateConstraint formula_constraint;
    formula_constraint.callback = move(callback);
    formula_constraint.uses_callback = true;
    formula_constraint.comparison_operator = comparison;
    formula_constraint.low_bound = low;
    formula_constraint.up_bound = up;

    formula_constraint.compiled.shape = FormulaShape::Nonlinear;
    formula_constraint.compiled.scope = FormulaScope::Mixed;
    formula_constraint.kind = classify(formula_constraint);

    constraint_set.multivariate.push_back(move(formula_constraint));

    network_jacobian.ready = false;
}


void ResponseOptimization::set_formula_constraint(const string& expression, const vector<float>& allowed_values)
{
    throw_if(!neural_network,
             "ResponseOptimization: set_formula_constraint requires a neural network to be set first");

    throw_if(allowed_values.empty(),
             "ResponseOptimization: AllowedSet formula constraint needs at least one value");

    MultivariateConstraint formula_constraint;
    formula_constraint.expression = expression;
    formula_constraint.comparison_operator = ComparisonOperator::AllowedSet;
    formula_constraint.allowed_values = allowed_values;
    formula_constraint.uses_callback = false;

    const vector<NamedColumn> input_columns = build_input_columns(neural_network->get_input_variables());
    const vector<NamedColumn> output_columns = build_output_columns(neural_network->get_output_variables());

    formula_constraint.compiled = compile_formula(expression, input_columns, output_columns);
    formula_constraint.kind = classify(formula_constraint);

    constraint_set.multivariate.push_back(move(formula_constraint));

    network_jacobian.ready = false;
}


void ResponseOptimization::clear_formula_constraints()
{
    constraint_set.multivariate.clear();
    constraint_set.disjunctive.clear();
    network_jacobian.ready = false;
}


void ResponseOptimization::set_min_feasible_ratio(float new_ratio)
{
    min_feasible_ratio = new_ratio;
}


void ResponseOptimization::set_max_oversample_factor(Index new_factor)
{
    max_oversample_factor = new_factor;
}


void ResponseOptimization::set_exploration_ratio(float new_ratio)
{
    exploration_ratio = new_ratio;
}


void ResponseOptimization::clear_constraints()
{
    constraint_set.univariate.clear();
}


void ResponseOptimization::clear_constraints(const string& name)
{
    constraint_set.univariate.erase(name);
}


void ResponseOptimization::clear_objectives()
{
    objectives.clear();
}


void ResponseOptimization::clear_objectives(const string& name)
{
    objectives.erase(name);
}


void ResponseOptimization::clear_time_roles()
{
    time_roles.clear();
    variables_descriptives.clear();
}


void ResponseOptimization::clear_time_roles(const string& name)
{
    time_roles.erase(name);
    variables_descriptives.clear();
}


UnivariateConstraint ResponseOptimization::get_constraint(const string& name) const
{
    const auto it = constraint_set.univariate.find(name);
    return it != constraint_set.univariate.end() ? it->second : UnivariateConstraint(ComparisonOperator::None);
}


bool ResponseOptimization::is_objective(const string& name) const
{
    return objectives.find(name) != objectives.end();
}


ResponseOptimization::Sense ResponseOptimization::get_sense(const string& name) const
{
    return objectives.at(name);
}


bool ResponseOptimization::is_past(const TimeType role)
{
    return role == TimeType::PastContinuous || role == TimeType::PastBatch;
}


bool ResponseOptimization::is_history(const string& name) const
{
    const auto it = time_roles.find(name);
    return it != time_roles.end() && is_past(it->second);
}


void ResponseOptimization::set_fixed_history(const Tensor3& history)
{
    fixed_history = history;
    network_jacobian.ready = false;
}


void ResponseOptimization::clear_fixed_history()
{
    fixed_history = Tensor3();
    network_jacobian.ready = false;
}


void ResponseOptimization::set_evaluations_number(const int new_evaluations_number)
{
    evaluations_number = new_evaluations_number;
}


void ResponseOptimization::set_iterations(const int new_max_iterations)
{
    max_iterations = new_max_iterations;
}


void ResponseOptimization::set_zoom_factor(float new_zoom_factor)
{
    zoom_factor = new_zoom_factor;
}


void ResponseOptimization::set_relative_tolerance(float new_relative_tolerance)
{
    relative_tolerance = new_relative_tolerance;
}


void ResponseOptimization::set_max_pareto_number(const Index new_max_pareto_number)
{
    max_pareto_number = new_max_pareto_number;
}


void ResponseOptimization::set_max_total_evaluations(const Index new_max_total_evaluations)
{
    max_total_evaluations = new_max_total_evaluations;
}


void ResponseOptimization::set_initial_sampling_factor(const Index new_initial_sampling_factor)
{
    initial_sampling_factor = max(Index(1), new_initial_sampling_factor);
}


void ResponseOptimization::set_branch_mode(const BranchMode new_branch_mode)
{
    branch_mode = new_branch_mode;
}


Index ResponseOptimization::get_evaluations_used() const
{
    return evaluations_used;
}


void ResponseOptimization::set_deformation_domain_factor(float new_deformation_domain_factor)
{
    deformation_domain_factor = new_deformation_domain_factor;
}


float ResponseOptimization::get_deformation_domain_factor()
{
    return deformation_domain_factor;
}

Index ResponseOptimization::get_objectives_number() const
{
    Index objectives_number = 0;

    for (const Variable& variable : get_variables_and_descriptives("Input").first)
        if (is_objective(variable.name))
            objectives_number++;

    for (const Variable& variable : get_variables_and_descriptives("Target").first)
        if (is_objective(variable.name))
            objectives_number++;

    return objectives_number;
}

vector<Descriptives> ResponseOptimization::get_descriptives(const string& role) const
{
    if (role == "Input")
    {
        if (neural_network->has("Scaling"))
            return static_cast<Scaling*>(neural_network->get_first("Scaling"))->get_descriptives();
    }
    else if (role == "Target")
        if (neural_network->has("Unscaling"))
            return static_cast<Unscaling*>(neural_network->get_first("Unscaling"))->get_descriptives();

    throw runtime_error("ResponseOptimization: Required Scaling/Unscaling layer for role '" + role + "' not found.");
}

const pair<vector<Variable>, vector<Descriptives>>& ResponseOptimization::get_variables_and_descriptives(const string& role) const
{
    // The filtered variables/descriptives depend only on the network and the time roles, both
    // stable during a solve; memoize to avoid re-reading the Scaling/Unscaling layers and
    // re-filtering on every sampling call. Invalidated by set()/set_time_role()/clear_time_roles().
    const auto cache_iterator = variables_descriptives.find(role);
    if (cache_iterator != variables_descriptives.end())
        return cache_iterator->second;

    const bool is_input_request = (role == "Input");

    const vector<Variable>& variables_unchecked = is_input_request ? neural_network->get_input_variables()
                                                        : neural_network->get_output_variables();

    const vector<Descriptives> descriptives_unchecked = get_descriptives(is_input_request ? "Input" : "Target");

    const vector<Index> feature_dimensions = get_feature_dimensions(variables_unchecked);
    const Index total_features = accumulate(feature_dimensions.begin(), feature_dimensions.end(), Index(0));

    const bool feature_level = (Index(descriptives_unchecked.size()) == total_features)
                            && (total_features != Index(variables_unchecked.size()));

    throw_if(!feature_level && variables_unchecked.size() != descriptives_unchecked.size(),
             "ResponseOptimization: Variable count and Descriptives count mismatch.");

    vector<Variable> filtered_variables;
    vector<Descriptives> filtered_descriptives;
    filtered_variables.reserve(variables_unchecked.size());
    filtered_descriptives.reserve(variables_unchecked.size());

    Index feature_cursor = 0;

    for (size_t i = 0; i < variables_unchecked.size(); ++i)
    {
        const Descriptives variable_descriptive = feature_level
            ? descriptives_unchecked[size_t(feature_cursor)]
            : descriptives_unchecked[i];

        feature_cursor += feature_dimensions[i];

        const string& variable_role = variables_unchecked[i].get_role();

        if (is_history(variables_unchecked[i].name))
            continue;

        const bool keep = is_input_request ? (variable_role == "Input")
                                           : (variable_role == "Target" || variable_role == "InputTarget");
        if (keep)
        {
            filtered_variables.push_back(variables_unchecked[i]);
            filtered_descriptives.push_back(variable_descriptive);
        }
    }

    const auto inserted = variables_descriptives.emplace(role, make_pair(move(filtered_variables), move(filtered_descriptives)));
    return inserted.first->second;
}

void ResponseOptimization::Domain::set(const vector<Variable>& variables, const vector<Descriptives>& descriptives, const float deformation_domain_factor)
{
    const Index variables_number = static_cast<Index>(variables.size());

    const vector<Index> feature_dimensions = get_feature_dimensions(variables);

    const Index total_feature_dimensions = accumulate(feature_dimensions.begin(), feature_dimensions.end(), Index(0));

    inferior_frontier.resize(total_feature_dimensions);
    superior_frontier.resize(total_feature_dimensions);


    Index feature_index = 0;

    for(Index variable = 0; variable < variables_number; ++variable)
    {
        const Index feature_dimension = feature_dimensions[variable];

        if (feature_dimension > 1)
        {
            inferior_frontier.segment(feature_index, feature_dimension).setConstant(0.0f);
            superior_frontier.segment(feature_index, feature_dimension).setConstant(1.0f);
        }
        else
        {
            const float original_minimum = static_cast<float>(descriptives[variable].minimum);
            const float original_maximum = static_cast<float>(descriptives[variable].maximum);

            const float center = (original_maximum + original_minimum) * 0.5f;
            const float half_range = (original_maximum - original_minimum) * 0.5f * deformation_domain_factor;

            inferior_frontier(feature_index) = center - half_range;
            superior_frontier(feature_index) = center + half_range;
        }

        feature_index += feature_dimension;
    }
}


ResponseOptimization::Domain ResponseOptimization::get_original_domain(const string role) const
{
    const auto& [variables, descriptives] = get_variables_and_descriptives(role);

    const size_t variables_number = variables.size();

    throw_if(descriptives.size() != variables_number,
             "ResponseOptimization: Descriptives count (" + to_string(descriptives.size()) +
             ") does not match variables count (" + to_string(variables_number) + ") for " + role);

    const vector<Index> feature_dimensions = get_feature_dimensions(variables);


    vector<UnivariateConstraint> applicable_constraints;
    applicable_constraints.reserve(variables_number);

    for (const Variable& variable : variables)
        applicable_constraints.push_back(get_constraint(variable.name));

    Domain original_domain(variables, descriptives, deformation_domain_factor);

    original_domain.bound(variables, applicable_constraints);

    return original_domain;
}


ResponseOptimization::Objectives::Objectives(const ResponseOptimization& response_optimization)
{
    const Index objectives_number = response_optimization.get_objectives_number();

    throw_if(objectives_number == 0,
             "No objectives found, make sure to call set_objective(name, Sense::Minimize/Maximize) on a variable");

    source_and_column.resize(2, objectives_number);

    scale_and_offset.resize(2, objectives_number);

    utopian_and_sense.resize(2, objectives_number);

    Index current_objective_index = 0;

    auto process_role = [&](const string& role)
    {
        const bool is_input = (role == "Input");

        const vector<Variable>& variables = response_optimization.get_variables_and_descriptives(role).first;

        const vector<Index> feature_dimensions_by_role = get_feature_dimensions(variables);

        const Domain domain = response_optimization.get_original_domain(role);

        Index feature_pointer = 0;

        for (Index i = 0; i < static_cast<Index>(variables.size()); ++i)
        {
            const string& variable_name = variables[i].name;

            if (response_optimization.is_objective(variable_name))
            {
                source_and_column(0, current_objective_index) = is_input ? 1.0f : 0.0f;

                source_and_column(1, current_objective_index) = static_cast<float>(feature_pointer);

                const float inferior_frontier = domain.inferior_frontier(feature_pointer);
                const float superior_frontier = domain.superior_frontier(feature_pointer);
                const float range = superior_frontier - inferior_frontier;
                const float safe_range = (range < EPSILON) ? EPSILON : range;

                scale_and_offset(0, current_objective_index) = 1.0 / safe_range;

                scale_and_offset(1, current_objective_index) = -inferior_frontier / safe_range;

                if (response_optimization.get_sense(variable_name) == Sense::Maximize)
                {
                    utopian_and_sense(0, current_objective_index) = superior_frontier;
                    utopian_and_sense(1, current_objective_index) = 1.0;
                }
                else
                {
                    utopian_and_sense(0, current_objective_index) = inferior_frontier;
                    utopian_and_sense(1, current_objective_index) = -1.0;
                }

                current_objective_index++;
            }

            feature_pointer += feature_dimensions_by_role[i];
        }
    };

    process_role("Input");
    process_role("Target");
}


void ResponseOptimization::Domain::bound(const vector<Variable>& variables, const vector<UnivariateConstraint>& constraints)
{
    const vector<Index> feature_dimensions = get_feature_dimensions(variables);
    Index feature_index = 0;

    for(size_t variable_index = 0; variable_index < variables.size(); ++variable_index)
    {
        const Index feature_dimension = feature_dimensions[variable_index];

        const UnivariateConstraint& constraint = constraints[variable_index];

        if(feature_dimension == 1)
        {
            float& inferior = inferior_frontier(feature_index);
            float& superior = superior_frontier(feature_index);

            switch(constraint.comparison)
            {
            case ComparisonOperator::EqualTo:
                inferior = max(inferior, constraint.low_bound);
                superior = min(superior, constraint.low_bound);
                break;
            case ComparisonOperator::Between:
                inferior = max(inferior, constraint.low_bound);
                superior = min(superior, constraint.up_bound);
                break;
            case ComparisonOperator::GreaterEqualTo:
                inferior = max(inferior, constraint.low_bound);
                break;
            case ComparisonOperator::LessEqualTo:
                superior = min(superior, constraint.up_bound);
                break;
            case ComparisonOperator::LessThan:
                superior = min(superior, constraint.up_bound);
                break;
            case ComparisonOperator::GreaterThan:
                inferior = max(inferior, constraint.low_bound);
                break;
            case ComparisonOperator::AllowedSet:
                if (!constraint.allowed_values.empty())
                {
                    float lowest = constraint.allowed_values.front();
                    float highest = constraint.allowed_values.front();
                    for (const float value : constraint.allowed_values)
                    {
                        lowest = min(lowest, value);
                        highest = max(highest, value);
                    }
                    inferior = max(inferior, lowest);
                    superior = min(superior, highest);
                }
                break;
            default:
                break;
            }
        }
        else if(constraint.comparison == ComparisonOperator::EqualTo)
        {
            const Index category_index = static_cast<Index>(llround(constraint.low_bound));

            for(Index j = 0; j < feature_dimension; ++j)
            {
                inferior_frontier(feature_index + j) = (j == category_index) ? 1.0 : 0.0;
                superior_frontier(feature_index + j) = (j == category_index) ? 1.0 : 0.0;
            }
        }
        else if(constraint.comparison == ComparisonOperator::AllowedSet)
        {
            for(Index j = 0; j < feature_dimension; ++j)
            {
                bool allowed = false;
                for (const float value : constraint.allowed_values)
                    if (static_cast<Index>(llround(value)) == j) { allowed = true; break; }

                inferior_frontier(feature_index + j) = 0.0;
                superior_frontier(feature_index + j) = allowed ? 1.0 : 0.0;
            }
        }

        feature_index += feature_dimension;
    }
}


static void round_discrete_inputs(MatrixR& inputs,
                                  const vector<Variable>& variables,
                                  const VectorR& inferior_frontier,
                                  const VectorR& superior_frontier)
{
    const vector<Index> feature_dimensions = get_feature_dimensions(variables);

    Index feature_index = 0;

    for(size_t i = 0; i < variables.size(); ++i)
    {
        const VariableType type = variables[i].type;

        if(type == VariableType::Binary)
            snap_to_lattice(inputs, feature_index, 0.0f, 1.0f);
        else if(type == VariableType::Integer)
            snap_to_lattice(inputs, feature_index,
                            ceil(inferior_frontier(feature_index)), floor(superior_frontier(feature_index)));

        feature_index += feature_dimensions[i];
    }
}


Lattice ResponseOptimization::build_input_lattice(const vector<Variable>& variables,
                                                  const vector<Index>& feature_dimensions,
                                                  const Domain& input_domain,
                                                  map<string, Index>& scalar_column_of) const
{
    Lattice lattice;
    Index feature = 0;

    for (size_t i = 0; i < variables.size(); ++i)
    {
        if (feature_dimensions[i] == 1)
        {
            scalar_column_of[variables[i].name] = feature;

            if (variables[i].type == VariableType::Binary || variables[i].type == VariableType::Integer)
            {
                lattice.columns.push_back(feature);
                lattice.min.push_back(ceil(input_domain.inferior_frontier(feature)));
                lattice.max.push_back(floor(input_domain.superior_frontier(feature)));
            }
        }

        feature += feature_dimensions[i];
    }

    return lattice;
}


vector<vector<Index>> ResponseOptimization::resolve_cardinality_columns(const Domain& input_domain,
                                                                        const map<string, Index>& scalar_column_of,
                                                                        const vector<char>& fixed_mask,
                                                                        const float discrete_explore,
                                                                        MatrixR& random_inputs) const
{
    const Index effective_evaluations = random_inputs.rows();
    const Index inputs_features_number = random_inputs.cols();

    vector<vector<Index>> cardinality_columns;

    for (const CardinalityConstraint& group : constraint_set.cardinality)
    {
        vector<Index> columns;
        columns.reserve(group.variable_names.size());

        for (const string& name : group.variable_names)
        {
            const auto found = scalar_column_of.find(name);
            throw_if(found == scalar_column_of.end(),
                     "ResponseOptimization: cardinality variable '" + name + "' is not a scalar input");
            throw_if(!fixed_mask[found->second],
                     "ResponseOptimization: cardinality variable '" + name + "' must be binary or integer");
            columns.push_back(found->second);
        }

        const Index count = static_cast<Index>(columns.size());
        vector<char> force_on(count, 0), force_off(count, 0);
        for (Index c = 0; c < count; ++c)
        {
            if (input_domain.superior_frontier(columns[c]) < 0.5f) force_off[c] = 1;
            if (input_domain.inferior_frontier(columns[c]) > 0.5f) force_on[c]  = 1;
        }

        const Index exploration_count = llround(discrete_explore * effective_evaluations);
        const bool have_preferred = (static_cast<Index>(sampling_memory.cardinality_preferred.size()) == inputs_features_number);

        vector<float> draw;
        for (Index r = 0; r < effective_evaluations; ++r)
        {
            bool drawn = false;

            if (have_preferred && r >= exploration_count)
            {
                vector<char> exploit_force_off = force_off;
                bool any_preferred_free = false;
                for (Index c = 0; c < count; ++c)
                {
                    if (force_on[c]) continue;
                    if (sampling_memory.cardinality_preferred[columns[c]]) any_preferred_free = true;
                    else exploit_force_off[c] = 1;
                }
                if (any_preferred_free)
                    drawn = draw_k_hot(count, group.k, force_on, exploit_force_off, draw);
            }

            if (!drawn)
                throw_if(!draw_k_hot(count, group.k, force_on, force_off, draw),
                         "ResponseOptimization: cardinality constraint (k=" + to_string(group.k)
                         + ") is infeasible under the current box pins.");

            for (Index c = 0; c < count; ++c)
                random_inputs(r, columns[c]) = draw[c];
        }

        cardinality_columns.push_back(move(columns));
    }

    return cardinality_columns;
}


MatrixR ResponseOptimization::calculate_random_inputs(const Domain& input_domain, const Index evaluations_count) const
{
    const vector<Variable>& variables = get_variables_and_descriptives("Input").first;

    const vector<Index> input_feature_dimensions = get_feature_dimensions(variables);

    const Index inputs_features_number = get_features_number(variables);

    const Index effective_evaluations = (evaluations_count > 0) ? evaluations_count : evaluations_number;

    MatrixR random_inputs(effective_evaluations, inputs_features_number);
    set_random_uniform(random_inputs, 0, 1);

    const Domain original_domain = get_original_domain("Input");

    const float continuous_explore = clamp(max(exploration_ratio, 1.0f - sampling_memory.last_feasibility_rate), 0.0f, 1.0f);
    const float discrete_explore   = min(1.0f, 1.5f * continuous_explore);

    const auto sample_scalar = [&](const Index col, const Index first_row, const Index row_count,
                                   const float inferior, const float superior, const VariableType type)
    {
        if (row_count <= 0) return;

        if (type == VariableType::Binary)
        {
            const float lo = ceil(inferior), hi = floor(superior);
            random_inputs.block(first_row, col, row_count, 1).array() = random_inputs.block(first_row, col, row_count, 1).array().round().max(lo).min(hi);
        }
        else if (type == VariableType::Integer)
        {
            const float lo = ceil(inferior), hi = floor(superior);
            random_inputs.block(first_row, col, row_count, 1).array()
                = (random_inputs.block(first_row, col, row_count, 1).array() * (hi - lo + 1.0f) + (lo - 0.5f)).round().max(lo).min(hi);
        }
        else
            random_inputs.block(first_row, col, row_count, 1).array() = random_inputs.block(first_row, col, row_count, 1).array() * (superior - inferior) + inferior;
    };

    const auto sample_allowed_set = [&](const Index col, const vector<float>& allowed_values)
    {
        const float inferior = input_domain.inferior_frontier(col);
        const float superior = input_domain.superior_frontier(col);

        vector<float> candidates;
        candidates.reserve(allowed_values.size());
        for (const float value : allowed_values)
            if (value >= inferior - EPSILON && value <= superior + EPSILON)
                candidates.push_back(value);

        if (candidates.empty())
        {
            const float center = 0.5f * (inferior + superior);
            float nearest = allowed_values.front();
            for (const float value : allowed_values)
                if (abs(value - center) < abs(nearest - center)) nearest = value;
            candidates.push_back(nearest);
        }

        // One locked bulk draw of every pick instead of locking the RNG once per row.
        MatrixR picks(effective_evaluations, 1);
        set_random_integer(picks, 0, static_cast<Index>(candidates.size()) - 1);

        for (Index i = 0; i < effective_evaluations; ++i)
            random_inputs(i, col) = candidates[static_cast<size_t>(picks(i, 0))];
    };

    const auto sample_categorical = [&](const Index first_col, const Index categories_number, const string& name)
    {
        random_inputs.block(0, first_col, effective_evaluations, categories_number).setZero();

        vector<Index> present_categories, original_categories;
        for(Index i = 0; i < categories_number; ++i)
        {
            if(input_domain.superior_frontier(first_col + i) > 0.5)    present_categories.push_back(i);
            if(original_domain.superior_frontier(first_col + i) > 0.5) original_categories.push_back(i);
        }

        throw_if(original_categories.empty(),
                 "ResponseOptimization: variable '" + name + "' has every category constrained out — cannot generate inputs.");

        if (present_categories.empty())
            present_categories = original_categories;

        vector<Index>& frequencies = sampling_memory.category_frequencies[name];
        if(Index(frequencies.size()) != categories_number)
            frequencies.assign(categories_number, 0);

        const Index explore_count = llround(discrete_explore * effective_evaluations);

        const auto least_sampled_category = [&frequencies](const vector<Index>& categories)
        {
            Index best = categories[0];
            for(const Index category : categories)
                if(frequencies[category] < frequencies[best]) best = category;
            return best;
        };

        // The exploit rows (i >= explore_count) draw uniformly over the same category range, so a
        // single bulk draw replaces one locked RNG call per exploit row. The explore rows are RNG-free.
        const Index exploit_count = max(Index(0), effective_evaluations - explore_count);
        MatrixR picks;
        if (exploit_count > 0)
        {
            picks.resize(exploit_count, 1);
            set_random_integer(picks, 0, static_cast<Index>(present_categories.size()) - 1);
        }

        for(Index i = 0; i < effective_evaluations; ++i)
        {
            const Index chosen = (i < explore_count)
                ? least_sampled_category(original_categories)
                : present_categories[static_cast<size_t>(picks(i - explore_count, 0))];

            random_inputs(i, first_col + chosen) = 1.0;
            frequencies[chosen]++;
        }
    };

    Index current_feature_index = 0;

    for(size_t input_variable = 0; input_variable < variables.size(); ++input_variable)
    {
        const Index categories_number = input_feature_dimensions[input_variable];
        const string& name = variables[input_variable].name;

        if(categories_number != 1)
        {
            sample_categorical(current_feature_index, categories_number, name);
            current_feature_index += categories_number;
            continue;
        }

        const UnivariateConstraint constraint = get_constraint(name);
        const VariableType type = variables[input_variable].type;

        if (constraint.comparison == ComparisonOperator::AllowedSet && !constraint.allowed_values.empty())
            sample_allowed_set(current_feature_index, constraint.allowed_values);
        else
        {
            if (type == VariableType::Integer)
                throw_if(floor(original_domain.superior_frontier(current_feature_index)) < ceil(original_domain.inferior_frontier(current_feature_index)),
                         "ResponseOptimization: integer variable '" + name + "' has no integer value within its range.");

            const float explore_fraction = (type == VariableType::Binary || type == VariableType::Integer)
                                          ? discrete_explore : continuous_explore;
            const Index explore_count = llround(explore_fraction * effective_evaluations);

            sample_scalar(current_feature_index, 0, explore_count,
                          original_domain.inferior_frontier(current_feature_index),
                          original_domain.superior_frontier(current_feature_index), type);
            sample_scalar(current_feature_index, explore_count, effective_evaluations - explore_count,
                          input_domain.inferior_frontier(current_feature_index),
                          input_domain.superior_frontier(current_feature_index), type);
        }

        current_feature_index++;
    }

    const vector<char> fixed_mask = discrete_column_mask(variables);

    map<string, Index> scalar_column_of;
    const Lattice lattice = build_input_lattice(variables, input_feature_dimensions, input_domain, scalar_column_of);

    const vector<vector<Index>> cardinality_columns =
        resolve_cardinality_columns(input_domain, scalar_column_of, fixed_mask, discrete_explore, random_inputs);

    std::set<Index> grouped_columns;
    for (const vector<Index>& group : cardinality_columns)
        grouped_columns.insert(group.begin(), group.end());

    Lattice free_lattice;
    for (size_t c = 0; c < lattice.columns.size(); ++c)
        if (!grouped_columns.count(lattice.columns[c]))
        {
            free_lattice.columns.push_back(lattice.columns[c]);
            free_lattice.min.push_back(lattice.min[c]);
            free_lattice.max.push_back(lattice.max[c]);
        }

    bool discrete_is_coupled = false;
    for (const MultivariateConstraint& constraint : constraint_set.multivariate)
    {
        if (constraint.kind != ConstraintKind::AffineInput && constraint.kind != ConstraintKind::NonlinearInput)
            continue;
        for (const Index column : constraint.compiled.input_indices)
            if (column >= 0 && column < inputs_features_number && fixed_mask[column])
                discrete_is_coupled = true;
    }

    if (!constraint_set.cardinality.empty() || discrete_is_coupled)
        repair_mixed_integer_inputs(random_inputs,
                                    input_domain.inferior_frontier,
                                    input_domain.superior_frontier,
                                    constraint_set.multivariate,
                                    fixed_mask,
                                    lattice,
                                    cardinality_columns,
                                    free_lattice,
                                    8, discrete_explore);
    else
    {
        repair_inputs(random_inputs,
                      input_domain.inferior_frontier,
                      input_domain.superior_frontier,
                      constraint_set.multivariate);

        round_discrete_inputs(random_inputs, variables,
                              input_domain.inferior_frontier,
                              input_domain.superior_frontier);
    }

    return random_inputs;
}


Tensor3 ResponseOptimization::combine_input(const MatrixR& input_control) const
{
    const vector<Variable>& input_variables = neural_network->get_input_variables();
    const Index batch_size = input_control.rows();
    const Shape input_shape = neural_network->get_input_shape();
    const Index total_lags = input_shape[0];
    const Index total_features = input_shape[1];

    Tensor3 input_combined(batch_size, total_lags, total_features);

    input_combined.device(get_device()) = fixed_history.broadcast(array<Index, 3>{batch_size, 1, 1});

    Index feature_cursor = 0;
    Index candidate_cursor = 0;

    for (const Variable& variable : input_variables)
    {
        const Index feature_count = variable.get_feature_count();

        if (variable.get_role() == "Input" && !is_history(variable.name))
        {
            const MatrixR block_data = input_control.block(0, candidate_cursor, batch_size, feature_count);

            TensorMap<const Tensor<float, 3, Layout>> block_tensor(block_data.data(), batch_size, 1, feature_count);

            input_combined.slice(array<Index, 3>{0, total_lags - 1, feature_cursor}, array<Index, 3>{batch_size, 1, feature_count}).device(get_device()) = block_tensor;

            candidate_cursor += feature_count;
        }

        feature_cursor += feature_count;
    }

    return input_combined;
}


MatrixR ResponseOptimization::calculate_outputs(const MatrixR& input) const
{
    if (is_forecasting())
    {
        throw_if(fixed_history.size() == 0,
                 "ResponseOptimization: Model is forecasting but fixed_history is empty. Call set_fixed_history() first.");

        const Tensor3 formatted_input = combine_input(input);

        return neural_network->calculate_outputs(formatted_input);
    }

    return neural_network->calculate_outputs(input);
}


void ResponseOptimization::Domain::reshape(const float zoom_factor,
                                           const VectorR& center,
                                           const MatrixR& points_inputs,
                                           const vector<Variable>& variables)
{
    const vector<Index> feature_dimensions = get_feature_dimensions(variables);

    VectorR categories_to_save = points_inputs.colwise().maxCoeff();

    for(Index i = 0; i < categories_to_save.size(); ++i)
        if(center(i) > categories_to_save(i))
            categories_to_save(i) = center(i);

    Index current_feature_index = 0;

    for(size_t input_variable = 0; input_variable < variables.size(); ++input_variable)
    {
        const Index categories_number = feature_dimensions[input_variable];

        if(categories_number == 1 && variables[input_variable].type != VariableType::Binary)
        {
            const float half_span = (superior_frontier(current_feature_index) - inferior_frontier(current_feature_index)) * zoom_factor / 2;
            inferior_frontier(current_feature_index) = max(center(current_feature_index) - half_span, inferior_frontier(current_feature_index));
            superior_frontier(current_feature_index) = min(center(current_feature_index) + half_span, superior_frontier(current_feature_index));

        }
        else
        {
            for(Index category_index = 0; category_index < categories_number; ++category_index)
            {
                const Index current_category = current_feature_index + category_index;
                inferior_frontier(current_category) = max(categories_to_save(current_category), inferior_frontier(current_category));
                superior_frontier(current_category) = min(categories_to_save(current_category), superior_frontier(current_category));
            }
        }

        current_feature_index += categories_number;
    }
}


bool ResponseOptimization::row_satisfies_formula_constraints(const VectorR& input_row,
                                                             const VectorR& output_row) const
{
    for (const MultivariateConstraint& formula_constraint : constraint_set.multivariate)
        if (!constraint_is_satisfied(formula_constraint, input_row, output_row))
            return false;

    return true;
}


pair<MatrixR, MatrixR> ResponseOptimization::filter_feasible_points(const MatrixR& inputs,
                                                                    const MatrixR& outputs,
                                                                    const Domain& output_domain) const
{
    const vector<Variable>& all_target_variables = neural_network->get_output_variables();
    const Index rows_number = outputs.rows();

    vector<Index> feasible_indices(rows_number);
    iota(feasible_indices.begin(), feasible_indices.end(), 0);

    Index domain_index = 0;

    for (Index column_index = 0; column_index < static_cast<Index>(all_target_variables.size()); ++column_index)
    {
        const string& variable_name = all_target_variables[column_index].name;

        if (is_history(variable_name))
            continue;

        if (get_constraint(variable_name).comparison != ComparisonOperator::None)
        {
            feasible_indices = filter_selected_indices_by_column(outputs,
                                                                 feasible_indices,
                                                                 column_index,
                                                                 output_domain.inferior_frontier(domain_index),
                                                                 output_domain.superior_frontier(domain_index));
        }

        ++domain_index;

        if (feasible_indices.empty())
            break;
    }

    if (!constraint_set.multivariate.empty() && !feasible_indices.empty())
    {
        vector<Index> formula_feasible_indices;
        formula_feasible_indices.reserve(feasible_indices.size());

        if (all_formula_constraints_are_linear(constraint_set.multivariate))
        {
            const Index feasible_count = static_cast<Index>(feasible_indices.size());
            const Index inputs_number = inputs.cols();
            const Index outputs_number = outputs.cols();

            const MatrixR feasible_inputs = slice_rows(inputs, feasible_indices);
            const MatrixR feasible_outputs = slice_rows(outputs, feasible_indices);

            const LinearConstraintSet linear_set = build_linear_constraint_set(constraint_set.multivariate, inputs_number, outputs_number);

            const MatrixR values = feasible_inputs * linear_set.A.leftCols(inputs_number).transpose()
                                 + feasible_outputs * linear_set.A.rightCols(outputs_number).transpose();

            for (Index i = 0; i < feasible_count; ++i)
            {
                const bool feasible = ((values.row(i).transpose().array() >= linear_set.lower.array()) &&
                                       (values.row(i).transpose().array() <= linear_set.upper.array())).all();

                if (feasible)
                    formula_feasible_indices.push_back(feasible_indices[i]);
            }
        }
        else
        {
            for (const Index row_index : feasible_indices)
            {
                const VectorR input_row = inputs.row(row_index).transpose();
                const VectorR output_row = outputs.row(row_index).transpose();

                if (row_satisfies_formula_constraints(input_row, output_row))
                    formula_feasible_indices.push_back(row_index);
            }
        }

        feasible_indices = move(formula_feasible_indices);
    }

    if (feasible_indices.empty())
        return {MatrixR(), MatrixR()};

    return {slice_rows(inputs, feasible_indices), slice_rows(outputs, feasible_indices)};
}


pair<MatrixR, MatrixR> ResponseOptimization::sample_feasible_points(const Domain& input_domain,
                                                                    const Domain& output_domain,
                                                                    const Index evaluations_multiplier) const
{
    const Index multiplier = max(Index(1), evaluations_multiplier);

    if (constraint_set.multivariate.empty())
        return generate_feasible_points(input_domain, output_domain, evaluations_number * multiplier);

    const Index base_evaluations = evaluations_number * multiplier;
    const Index evaluations_cap = base_evaluations * max_oversample_factor;
    const float low_ratio_threshold = min_feasible_ratio * float(0.25);

    Index current_evaluations = base_evaluations;
    Index consecutive_low_ratio_attempts = 0;

    pair<MatrixR, MatrixR> feasible_result;

    while (true)
    {
        feasible_result = generate_feasible_points(input_domain, output_domain, current_evaluations);

        const Index feasible_count = feasible_result.first.rows();
        const float feasible_ratio = (current_evaluations > 0)
            ? float(feasible_count) / float(current_evaluations)
            : float(0);

        if (feasible_count > 0 && feasible_ratio >= min_feasible_ratio)
            return feasible_result;

        if (feasible_ratio < low_ratio_threshold)
            ++consecutive_low_ratio_attempts;
        else
            consecutive_low_ratio_attempts = 0;

        if (current_evaluations >= evaluations_cap)
            break;
        if (consecutive_low_ratio_attempts >= 2)
            break;

        const Index next_evaluations = min(current_evaluations * Index(2), evaluations_cap);

        cout << "> Adaptive oversampling: feasible ratio " << feasible_ratio
             << " below threshold " << min_feasible_ratio
             << ", increasing evaluations from " << current_evaluations
             << " to " << next_evaluations << endl;

        current_evaluations = next_evaluations;
    }

    const bool early_stop_triggered = (consecutive_low_ratio_attempts >= 2);
    const Index final_feasible_count = feasible_result.first.rows();

    throw_if(final_feasible_count == 0,
             "ResponseOptimization: formula constraints appear infeasible — "
             "no feasible points found after adaptive oversampling up to "
             + to_string(current_evaluations) + " evaluations.");

    throw_if(early_stop_triggered,
             "ResponseOptimization: formula constraints are too tight — "
             "feasibility ratio stayed below "
             + to_string(low_ratio_threshold)
             + " over consecutive oversampling attempts (up to "
             + to_string(current_evaluations) + " evaluations, "
             + to_string(final_feasible_count) + " feasible points found).");

    return feasible_result;
}


pair<MatrixR, MatrixR> ResponseOptimization::generate_feasible_points(const Domain& input_domain,
                                                         const Domain& output_domain,
                                                         const Index evaluations_count) const
{
    MatrixR random_inputs = calculate_random_inputs(input_domain, evaluations_count);

    const vector<Variable>& input_variables = get_variables_and_descriptives("Input").first;
    const vector<char> fixed_columns = discrete_column_mask(input_variables);

    if (network_jacobian.differential)
    {
        repair_output_constraints(random_inputs,
                                  input_domain.inferior_frontier,
                                  input_domain.superior_frontier,
                                  constraint_set.multivariate,
                                  [this](const VectorR& x) { return network_jacobian.differential->forward(x); },
                                  [this](const VectorR& x, const VectorR& cotangent) { return network_jacobian.differential->vjp(x, cotangent); },
                                  64, fixed_columns);
    }
    else
    {
        const SurrogateBatchForward batch_forward = [this](const MatrixR& x) -> MatrixR
        {
            return calculate_outputs(x);
        };

        repair_output_constraints(random_inputs,
                                  input_domain.inferior_frontier,
                                  input_domain.superior_frontier,
                                  constraint_set.multivariate,
                                  batch_forward,
                                  64, fixed_columns);
    }

    round_discrete_inputs(random_inputs,
                          input_variables,
                          input_domain.inferior_frontier,
                          input_domain.superior_frontier);

    const MatrixR outputs = calculate_outputs(random_inputs);
    evaluations_used += evaluations_count;

    pair<MatrixR, MatrixR> feasible = filter_feasible_points(random_inputs, outputs, output_domain);

    sampling_memory.last_feasibility_rate = clamp(float(feasible.first.rows()) / float(max(Index(1), random_inputs.rows())), 0.0f, 1.0f);

    return feasible;
}


MatrixR ResponseOptimization::Objectives::extract(const MatrixR& inputs, const MatrixR& outputs) const
{
    const Index objectives_number = source_and_column.cols();

    MatrixR objective_matrix(inputs.rows(), objectives_number);

    for (Index j = 0; j < objectives_number; ++j)
        objective_matrix.col(j) = (source_and_column(0, j) > 0.5)
              ? inputs.col(static_cast<Index>(source_and_column(1, j)))
              : outputs.col(static_cast<Index>(source_and_column(1, j)));

    return objective_matrix;
}


void ResponseOptimization::Objectives::normalize(MatrixR& objective_matrix) const
{
    const auto combined_scale = scale_and_offset.row(0).array() * utopian_and_sense.row(1).array();
    const auto combined_offset = scale_and_offset.row(1).array() * utopian_and_sense.row(1).array();

    objective_matrix.array().rowwise() *= combined_scale;
    objective_matrix.array().rowwise() += combined_offset;
}


bool ResponseOptimization::Objectives::update_utopian_from_points(const MatrixR& unnormalized_objective_values)
{
    if (unnormalized_objective_values.rows() == 0)
        return false;

    const Index objectives_number = utopian_and_sense.cols();

    if (unnormalized_objective_values.cols() != objectives_number)
        return false;

    bool any_updated = false;

    for (Index j = 0; j < objectives_number; ++j)
    {
        const float sense = utopian_and_sense(1, j);
        const float current_utopian = utopian_and_sense(0, j);

        const float best = (sense > 0)
            ? unnormalized_objective_values.col(j).maxCoeff()
            : unnormalized_objective_values.col(j).minCoeff();

        if (sense * (best - current_utopian) <= float(0))
            continue;

        const float scale = scale_and_offset(0, j);
        const float offset = scale_and_offset(1, j);

        if (abs(scale) < EPSILON)
            continue;

        const float old_inferior = -offset / scale;
        const float old_superior = old_inferior + float(1) / scale;

        const float new_inferior = (sense > 0) ? old_inferior : best;
        const float new_superior = (sense > 0) ? best : old_superior;
        const float new_range = new_superior - new_inferior;

        if (new_range < EPSILON)
            continue;

        utopian_and_sense(0, j) = best;
        scale_and_offset(0, j) = float(1) / new_range;
        scale_and_offset(1, j) = -new_inferior / new_range;

        any_updated = true;
    }

    return any_updated;
}


pair<MatrixR, MatrixR> ResponseOptimization::calculate_optimal_points(const MatrixR& feasible_inputs,
                                                                      const MatrixR& feasible_outputs,
                                                                      const Objectives& objectives) const
{
    MatrixR objective_matrix = objectives.extract(feasible_inputs, feasible_outputs);
    objectives.normalize(objective_matrix);

    return calculate_optimal_points(feasible_inputs, feasible_outputs, objectives, objective_matrix);
}


pair<MatrixR, MatrixR> ResponseOptimization::calculate_optimal_points(const MatrixR& feasible_inputs,
                                                                      const MatrixR& feasible_outputs,
                                                                      const Objectives& objectives,
                                                                      const MatrixR& normalized_objective_matrix) const
{
    const Index subset_dimension = clamp<Index>(llround(zoom_factor * evaluations_number), 1, feasible_outputs.rows());

    const VectorR normalized_utopian_point = (objectives.utopian_and_sense.row(1).array() + (float)1.0) / (float)2.0;

    const VectorI nearest_rows = get_nearest_points(normalized_objective_matrix, normalized_utopian_point , (int)subset_dimension);

    MatrixR nearest_inputs(subset_dimension, feasible_inputs.cols());
    MatrixR nearest_outputs(subset_dimension, feasible_outputs.cols());

    for(Index i = 0; i < subset_dimension; ++i)
    {
        nearest_inputs.row(i) = feasible_inputs.row(nearest_rows(i));
        nearest_outputs.row(i) = feasible_outputs.row(nearest_rows(i));
    }

    return {nearest_inputs, nearest_outputs};
}


void ResponseOptimization::promote_single_variable_constraints()
{
    if (constraint_set.multivariate.empty() || !neural_network)
        return;

    const vector<Variable>& input_variables = neural_network->get_input_variables();
    const vector<NamedColumn> input_columns = build_input_columns(input_variables);

    map<Index, string> name_of_column;
    for (const NamedColumn& column : input_columns)
        name_of_column[column.column_index] = column.name;

    auto interval_of = [](const UnivariateConstraint& constraint, float& lo, float& hi) -> bool
    {
        lo = -numeric_limits<float>::infinity();
        hi =  numeric_limits<float>::infinity();
        switch (constraint.comparison)
        {
        case ComparisonOperator::None:           return true;
        case ComparisonOperator::EqualTo:        lo = hi = constraint.low_bound; return true;
        case ComparisonOperator::Between:        lo = constraint.low_bound; hi = constraint.up_bound; return true;
        case ComparisonOperator::GreaterEqualTo:
        case ComparisonOperator::GreaterThan:    lo = constraint.low_bound; return true;
        case ComparisonOperator::LessEqualTo:
        case ComparisonOperator::LessThan:       hi = constraint.up_bound; return true;
        case ComparisonOperator::AllowedSet:
        default:                                 return false;
        }
    };

    vector<MultivariateConstraint> kept;
    kept.reserve(constraint_set.multivariate.size());

    for (MultivariateConstraint& formula_constraint : constraint_set.multivariate)
    {
        const CompiledFormula& compiled = formula_constraint.compiled;

        const bool promotable = !formula_constraint.uses_callback
            && formula_constraint.comparison_operator != ComparisonOperator::None
            && formula_constraint.comparison_operator != ComparisonOperator::AllowedSet
            && compiled.shape == FormulaShape::Affine
            && compiled.scope == FormulaScope::InputsOnly
            && compiled.affine_input_terms.size() == 1
            && compiled.affine_output_terms.empty();

        if (!promotable) { kept.push_back(move(formula_constraint)); continue; }

        const Index column = compiled.affine_input_terms.front().first;
        const float coefficient = compiled.affine_input_terms.front().second;
        const auto found = name_of_column.find(column);

        if (found == name_of_column.end() || coefficient == 0.0f)
        { kept.push_back(move(formula_constraint)); continue; }

        const float constant = compiled.affine_constant;
        const float low = formula_constraint.low_bound;
        const float up  = formula_constraint.up_bound;
        const auto solve = [&](const float bound) { return (bound - constant) / coefficient; };

        float implied_lo = -numeric_limits<float>::infinity();
        float implied_hi =  numeric_limits<float>::infinity();

        switch (formula_constraint.comparison_operator)
        {
        case ComparisonOperator::EqualTo:
            implied_lo = implied_hi = solve(low); break;
        case ComparisonOperator::Between:
            implied_lo = min(solve(low), solve(up));
            implied_hi = max(solve(low), solve(up)); break;
        case ComparisonOperator::GreaterEqualTo:
        case ComparisonOperator::GreaterThan:
            (coefficient > 0.0f ? implied_lo : implied_hi) = solve(low); break;
        case ComparisonOperator::LessEqualTo:
        case ComparisonOperator::LessThan:
            (coefficient > 0.0f ? implied_hi : implied_lo) = solve(up); break;
        default: break;
        }

        const string& name = found->second;

        float existing_lo = -numeric_limits<float>::infinity();
        float existing_hi =  numeric_limits<float>::infinity();
        const auto existing = constraint_set.univariate.find(name);
        if (existing != constraint_set.univariate.end() && !interval_of(existing->second, existing_lo, existing_hi))
        { kept.push_back(move(formula_constraint)); continue; }

        const float new_lo = max(implied_lo, existing_lo);
        const float new_hi = min(implied_hi, existing_hi);

        throw_if(new_lo > new_hi + bound_tolerance(new_hi),
                 "ResponseOptimization: constraint '" + formula_constraint.expression
                 + "' leaves variable '" + name + "' with an empty box.");

        const bool lo_finite = isfinite(new_lo);
        const bool hi_finite = isfinite(new_hi);

        if (lo_finite && hi_finite && new_lo == new_hi)
            constraint_set.univariate[name] = UnivariateConstraint(ComparisonOperator::EqualTo, new_lo, new_lo);
        else if (lo_finite && hi_finite)
            constraint_set.univariate[name] = UnivariateConstraint(ComparisonOperator::Between, new_lo, new_hi);
        else if (lo_finite)
            constraint_set.univariate[name] = UnivariateConstraint(ComparisonOperator::GreaterEqualTo, new_lo, 0.0f);
        else if (hi_finite)
            constraint_set.univariate[name] = UnivariateConstraint(ComparisonOperator::LessEqualTo, 0.0f, new_hi);

        cout << "> Promoted single-variable constraint '" << formula_constraint.expression
             << "' to a box on '" << name << "'." << endl;
    }

    constraint_set.multivariate = move(kept);
}


vector<char> ResponseOptimization::discrete_column_mask(const vector<Variable>& variables) const
{
    const vector<Index> dimensions = get_feature_dimensions(variables);

    vector<char> mask(get_features_number(variables), 0);

    Index feature = 0;
    for (size_t i = 0; i < variables.size(); ++i)
    {
        const VariableType type = variables[i].type;
        if (type == VariableType::Binary || type == VariableType::Integer
            || type == VariableType::Categorical || dimensions[i] > 1)
            for (Index j = 0; j < dimensions[i]; ++j)
                mask[feature + j] = 1;
        feature += dimensions[i];
    }

    return mask;
}


void ResponseOptimization::restore_cardinality_columns(Domain& domain, const Domain& original) const
{
    if (constraint_set.cardinality.empty())
        return;

    if (sampling_memory.cardinality_indicator_columns.empty())
    {
        const vector<Variable>& variables = get_variables_and_descriptives("Input").first;
        const vector<Index> dimensions = get_feature_dimensions(variables);

        map<string, Index> column_of;
        Index feature = 0;
        for (size_t i = 0; i < variables.size(); ++i)
        {
            if (dimensions[i] == 1)
                column_of[variables[i].name] = feature;
            feature += dimensions[i];
        }

        for (const CardinalityConstraint& group : constraint_set.cardinality)
            for (const string& name : group.variable_names)
            {
                const auto found = column_of.find(name);
                if (found != column_of.end())
                    sampling_memory.cardinality_indicator_columns[name] = found->second;
            }
    }

    if (static_cast<Index>(sampling_memory.cardinality_preferred.size()) != domain.superior_frontier.size())
        sampling_memory.cardinality_preferred.assign(domain.superior_frontier.size(), 0);

    for (const auto& [name, column] : sampling_memory.cardinality_indicator_columns)
    {
        sampling_memory.cardinality_preferred[column] = (domain.superior_frontier(column) >= 0.5f) ? 1 : 0;

        domain.inferior_frontier(column) = original.inferior_frontier(column);
        domain.superior_frontier(column) = original.superior_frontier(column);
    }
}


MatrixR ResponseOptimization::perform_single_objective_optimization() const
{
    const Objectives objectives(*this);

    const vector<Variable>& input_variables = get_variables_and_descriptives("Input").first;

    const Domain original_input_domain = get_original_domain("Input");
    const Domain original_output_domain = get_original_domain("Target");
    Domain input_domain = original_input_domain;

    pair<MatrixR, MatrixR> optimal_set;

    float optimal_point;

    float previous_optimal_point = 0;

    cout << "> Optimization loop starting with zoom factor: " << zoom_factor << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        if (max_total_evaluations > 0 && evaluations_used >= max_total_evaluations)
        {
            cout << "> [Budget cap] Reached " << evaluations_used
                 << " surrogate evaluations (budget=" << max_total_evaluations
                 << "). Stopping at iteration " << i + 1 << "." << endl;
            break;
        }

        auto [feasible_inputs, feasible_outputs] = sample_feasible_points(input_domain, original_output_domain);

        if (feasible_outputs.size() > 0 && !feasible_outputs.allFinite())
        {
            cout << "Model produced NaN — aborting optimization loop." << endl;
            break;
        }

        if (feasible_inputs.rows() == 0)
        {
            cout << "!!! [Critical] Zero feasible points found. "
                 << "Check if your constraints are too strict. "
                 << "Aborting optimization loop." << endl;
            break;
        }

        cout << "\n> [Iteration " << i + 1 << " / " << max_iterations << "]" << endl;
        cout << "  - Feasible points: " << feasible_inputs.rows() << endl;

        optimal_set = calculate_optimal_points(feasible_inputs, feasible_outputs, objectives);

        if (optimal_set.first.rows() == 0 || optimal_set.second.rows() == 0)
        {
            cout << "!!! [Critical] calculate_optimal_points returned empty. "
                 << "Aborting optimization loop." << endl;
            break;
        }

        optimal_point = (objectives.source_and_column(0, 0) > 0.5f
            ? optimal_set.first
            : optimal_set.second)(0, static_cast<Index>(objectives.source_and_column(1, 0)));

        const float relative_error = abs((optimal_point - previous_optimal_point) / (objectives.utopian_and_sense(0,0) + 1e-6f));

        cout << "  - Relative error: " << relative_error << endl;

        if (relative_error < relative_tolerance && i > min_iterations)
        {
            cout << "> Optimization loop stopped for reaching the relative tolerance desired: " << relative_tolerance << endl;
            break;
        }

        previous_optimal_point = optimal_point;

        input_domain.reshape(zoom_factor, optimal_set.first.row(0), optimal_set.first, input_variables);
        restore_cardinality_columns(input_domain, original_input_domain);
    }

    return optimal_set.first.rows() == 0
        ? MatrixR()
        : append_columns(optimal_set.first, optimal_set.second);
}


// Vertically stack row-blocks in a single allocation. Replaces repeated append_rows, which copies
// the whole growing matrix every call (quadratic). Empty blocks contribute no rows.
static MatrixR stack_rows(const vector<MatrixR>& blocks)
{
    if (blocks.empty())
        return MatrixR();

    Index total_rows = 0;
    for (const MatrixR& block : blocks)
        total_rows += block.rows();

    MatrixR result(total_rows, blocks.front().cols());

    Index row = 0;
    for (const MatrixR& block : blocks)
    {
        if (block.rows() == 0)
            continue;
        result.middleRows(row, block.rows()) = block;
        row += block.rows();
    }

    return result;
}


// True if row `a` Pareto-dominates row `b` (maximization): no worse in every objective and
// strictly better in at least one. Short-circuits on the first objective where `a` falls behind.
static bool pareto_dominates(const MatrixR& objective_matrix, const Index a, const Index b)
{
    bool strictly_better = false;

    for (Index k = 0; k < objective_matrix.cols(); ++k)
    {
        const float difference = objective_matrix(a, k) - objective_matrix(b, k);
        if (difference < 0.0f) return false;
        if (difference > 0.0f) strictly_better = true;
    }

    return strictly_better;
}


// Row indices of the Pareto-optimal (non-dominated) points of an objective matrix, maximization
// sense. Rows with non-finite entries are skipped (they neither belong to the front nor enter it
// as dominators). Incremental cull: each point is tested only against the running front and, when
// kept, evicts the front members it dominates — far cheaper than all-pairs once the front is small.
static vector<Index> pareto_front_indices(const MatrixR& objective_matrix)
{
    const Index rows_number = objective_matrix.rows();

    vector<Index> front;
    front.reserve(rows_number);

    for (Index i = 0; i < rows_number; ++i)
    {
        if (!objective_matrix.row(i).allFinite())
            continue;

        bool dominated = false;
        for (const Index j : front)
            if (pareto_dominates(objective_matrix, j, i)) { dominated = true; break; }

        if (dominated)
            continue;

        front.erase(remove_if(front.begin(), front.end(),
                              [&](const Index j) { return pareto_dominates(objective_matrix, i, j); }),
                    front.end());
        front.push_back(i);
    }

    return front;
}


pair<MatrixR, MatrixR> ResponseOptimization::calculate_pareto(const MatrixR& inputs,
                                                              const MatrixR& outputs,
                                                              const MatrixR& objective_matrix) const
{
    if (inputs.rows() == 0)
        return {};

    const vector<Index> front = pareto_front_indices(objective_matrix);

    return {slice_rows(inputs, front), slice_rows(outputs, front)};
}


pair<float, float> ResponseOptimization::calculate_quality_metrics(const MatrixR& inputs,
                                                                 const MatrixR& outputs,
                                                                 const Objectives& objectives) const
{
    const Index points_number = inputs.rows();

    if (points_number == 0)
        return {static_cast<float>(1e6), static_cast<float>(1e6)};

    MatrixR objective_matrix = objectives.extract(inputs, outputs);

    objectives.normalize(objective_matrix);

    const Index objectives_number = objective_matrix.cols();

    const float hypercube_diagonal = sqrt(static_cast<float>(objectives_number));

    float maximum_internal_gap = 0.0;

    for (Index i = 0; i < points_number; ++i)
    {
        const auto current_point = objective_matrix.row(i);

        VectorR  distances = (objective_matrix.rowwise() - current_point).rowwise().squaredNorm();

        distances(i) = MAX;

        const float minimum_neighbor_distance = sqrt(distances.minCoeff());

        maximum_internal_gap = max(maximum_internal_gap, minimum_neighbor_distance);
    }

    if (points_number == 1)
        maximum_internal_gap = 1.0;

    maximum_internal_gap /= hypercube_diagonal;

    const VectorR max_objectives = objective_matrix.colwise().maxCoeff();

    const float sum_boundary_gaps = (1.0 - max_objectives.array()).abs().sum();

    const float average_boundary_gap = sum_boundary_gaps / static_cast<float>(objectives_number);
    const float normalized_boundary_gap = average_boundary_gap / hypercube_diagonal;

    return {maximum_internal_gap, normalized_boundary_gap};
}


static vector<Index> reselect_pareto_front(const MatrixR& objective_matrix, const Index maximum_number)
{
    const Index points_number = objective_matrix.rows();

    vector<Index> selection(points_number);
    iota(selection.begin(), selection.end(), 0);

    if (points_number <= maximum_number)
        return selection;

    const Index objectives_number = objective_matrix.cols();
    const Index cluster_size = 3;
    const Index outlier_number = Index(0.25f * maximum_number);
    const Index hole_number = Index(0.15f * maximum_number);

    const MatrixR distances = calculate_distances(objective_matrix);

    vector<char> chosen(points_number, 0);
    selection.clear();
    selection.reserve(maximum_number);

    const auto take = [&](const Index point)
    {
        if (Index(selection.size()) < maximum_number && !chosen[point])
        {
            chosen[point] = 1;
            selection.push_back(point);
        }
    };

    const auto take_ranked = [&](const VectorI& ranking, const Index quota)
    {
        for (Index i = 0, taken = 0; i < ranking.size() && taken < quota; i++)
            if (!chosen[ranking(i)]) { take(ranking(i)); taken++; }
    };

    for (Index j = 0; j < objectives_number; j++)
        for (const Index extreme : {maximal_index(objective_matrix.col(j)), minimal_index(objective_matrix.col(j))})
        {
            const VectorI cluster = get_nearest_points(objective_matrix, objective_matrix.row(extreme).transpose(), cluster_size);
            for (Index i = 0; i < cluster.size(); i++)
                take(cluster(i));
        }

    take_ranked(maximal_indices(local_outlier_factor(objective_matrix, 20), points_number), outlier_number);

    MatrixR gap_distances = distances;
    gap_distances.diagonal().setConstant(MAX);
    take_ranked(maximal_indices(gap_distances.rowwise().minCoeff(), points_number), hole_number);

    VectorR minimum_distance = VectorR::Constant(points_number, MAX);
    for (const Index point : selection)
        minimum_distance = minimum_distance.cwiseMin(distances.col(point));
    for (Index i = 0; i < points_number; i++)
        if (chosen[i]) minimum_distance(i) = -MAX;

    while (Index(selection.size()) < maximum_number)
    {
        const Index farthest = maximal_index(minimum_distance);
        if (minimum_distance(farthest) < 0.0f) break;
        take(farthest);
        minimum_distance = minimum_distance.cwiseMin(distances.col(farthest));
        minimum_distance(farthest) = -MAX;
    }

    return selection;
}


MatrixR ResponseOptimization::perform_multiobjective_optimization() const
{
    Objectives objectives(*this);

    const vector<Variable>& input_variables = get_variables_and_descriptives("Input").first;

    const Domain original_input_domain = get_original_domain("Input");
    const Domain original_output_domain = get_original_domain("Target");

    auto [first_feasible_inputs, first_feasible_outputs] = sample_feasible_points(original_input_domain, original_output_domain, initial_sampling_factor);

    if (first_feasible_inputs.rows() == 0)
    {
        cout << "!!! [Critical] Zero feasible points found. "
             << "Check if your constraints are too strict." << endl;
        return MatrixR();
    }

    MatrixR first_objective_matrix  = objectives.extract(first_feasible_inputs, first_feasible_outputs);
    objectives.normalize(first_objective_matrix);

    auto [global_pareto_inputs, global_pareto_outputs] = calculate_pareto(first_feasible_inputs, first_feasible_outputs, first_objective_matrix);

    if (global_pareto_inputs.rows() > 0)
    {
        const MatrixR initial_pareto_unnormalized = objectives.extract(global_pareto_inputs, global_pareto_outputs);
        if (objectives.update_utopian_from_points(initial_pareto_unnormalized))
            cout << "> Utopian promoted from initial Pareto front." << endl;
    }

    cout << "> Initial Pareto front size: " << global_pareto_inputs.rows() << " points." << endl;

    vector<Domain> input_domains(static_cast<size_t>(global_pareto_inputs.rows()), original_input_domain);

    float current_zoom = zoom_factor;

    float previous_holes_magnitude = 0.0;
    float previous_area_covered = 0.0;

    cout << "> Optimization loop starting with zoom factor: " << current_zoom << endl;

    for (Index i = 0; i < max_iterations; i++)
    {
        cout << "\n> [Iteration " << i + 1 << " / " << max_iterations << "]" << endl;

        vector<MatrixR> candidate_input_blocks { global_pareto_inputs };
        vector<MatrixR> candidate_output_blocks { global_pareto_outputs };

        for (Index j = 0; j < global_pareto_inputs.rows(); j++)
        {
            if (max_total_evaluations > 0 && evaluations_used >= max_total_evaluations)
                break;

            auto [local_feasible_inputs, local_feasible_outputs] = sample_feasible_points(input_domains[j], original_output_domain);

            MatrixR local_objective_matrix = objectives.extract(local_feasible_inputs, local_feasible_outputs);
            objectives.normalize(local_objective_matrix);

            auto [local_pareto_input, local_pareto_output] = calculate_pareto(local_feasible_inputs, local_feasible_outputs, local_objective_matrix );

            candidate_input_blocks.push_back(move(local_pareto_input));
            candidate_output_blocks.push_back(move(local_pareto_output));
        }

        // Single allocation instead of append_rows copying the growing matrix every iteration.
        const MatrixR candidate_inputs = stack_rows(candidate_input_blocks);
        const MatrixR candidate_outputs = stack_rows(candidate_output_blocks);

        cout << "  - Aggregated local Pareto candidates: " << candidate_inputs.rows() << endl;

        if (candidate_inputs.rows() == 0)
            break;

        // Normalized objectives of the candidate set, shared by optimal-point selection and Pareto.
        MatrixR objective_matrix = objectives.extract(candidate_inputs, candidate_outputs);
        objectives.normalize(objective_matrix);

        pair<MatrixR, MatrixR> optimal_set = calculate_optimal_points(candidate_inputs, candidate_outputs, objectives, objective_matrix);

        const auto pareto_pair = calculate_pareto(candidate_inputs, candidate_outputs, objective_matrix);

        global_pareto_inputs = pareto_pair.first;
        global_pareto_outputs = pareto_pair.second;

        cout << "  - New Pareto front size: " << global_pareto_inputs.rows()  << endl;

        if (max_pareto_number > 0 && global_pareto_inputs.rows() > max_pareto_number)
        {
            MatrixR pareto_objectives = objectives.extract(global_pareto_inputs, global_pareto_outputs);
            objectives.normalize(pareto_objectives);

            const vector<Index> selection = reselect_pareto_front(pareto_objectives, max_pareto_number);

            global_pareto_inputs = slice_rows(global_pareto_inputs, selection);
            global_pareto_outputs = slice_rows(global_pareto_outputs, selection);

            cout << "  - Pareto front reselected to " << global_pareto_inputs.rows() << " representatives." << endl;
        }

        if (max_total_evaluations > 0 && evaluations_used >= max_total_evaluations)
        {
            cout << "> [Budget cap] Reached " << evaluations_used
                 << " surrogate evaluations (budget=" << max_total_evaluations
                 << "). Stopping at iteration " << i + 1 << "." << endl;
            break;
        }

        if (global_pareto_inputs.rows() > 0)
        {
            const MatrixR pareto_objective_unnormalized = objectives.extract(global_pareto_inputs, global_pareto_outputs);
            if (objectives.update_utopian_from_points(pareto_objective_unnormalized))
            {
                cout << "  - Utopian promoted to better Pareto coordinate." << endl;
                previous_holes_magnitude = 0.0;
                previous_area_covered = 0.0;
            }
        }

        const pair<float, float> quality = calculate_quality_metrics(global_pareto_inputs, global_pareto_outputs, objectives);

        const float current_hole = quality.first;
        const float current_boundary = quality.second;

        cout << "  - Internal Hole: " << current_hole << " | Boundary Gap: " << current_boundary << endl;

        const float delta_hole = abs(current_hole - previous_holes_magnitude);
        const float delta_boundary = abs(current_boundary - previous_area_covered);

        if (i > min_iterations && delta_hole < relative_tolerance && delta_boundary < relative_tolerance)
        {
            cout << "> [Convergence] Quality metrics stabilized. Stopping at iteration " << i + 1 << endl;
            break;
        }

        previous_holes_magnitude = current_hole;
        previous_area_covered = current_boundary;

        input_domains.reserve(static_cast<size_t>(global_pareto_inputs.rows()));
        input_domains.assign(static_cast<size_t>(global_pareto_inputs.rows()), original_input_domain);

        const MatrixR best_and_pareto = append_rows(optimal_set.first, global_pareto_inputs);

        for (Index j = 0; j < global_pareto_inputs.rows(); j++)
        {
            input_domains[j].reshape(current_zoom, global_pareto_inputs.row(j), best_and_pareto , input_variables);
            restore_cardinality_columns(input_domains[j], original_input_domain);
        }

        current_zoom *= zoom_factor;
    }
    cout << "\n> [Optimization Complete] Assembling final results..." << endl;
    cout << "> Total surrogate evaluations used: " << evaluations_used << endl;

    return append_columns(global_pareto_inputs, global_pareto_outputs);
}


vector<float> ResponseOptimization::get_utopian_point() const
{
    const Objectives objectives(*this);

    const Index objectives_number = objectives.utopian_and_sense.cols();

    vector<float> utopian_point(static_cast<size_t>(objectives_number));

    for (Index j = 0; j < objectives_number; ++j)
        utopian_point[static_cast<size_t>(j)] = objectives.utopian_and_sense(0, j);

    return utopian_point;
}


pair<Index, VectorR> ResponseOptimization::get_advised_point(const MatrixR& pareto_front,
                                                             const VectorR& importance_scale) const
{
    if (pareto_front.rows() == 0)
        return {-1, VectorR()};

    if (pareto_front.rows() == 1)
        return {0, pareto_front.row(0).transpose()};

    const Index objectives_number = get_objectives_number();

    VectorR scale = (importance_scale.size() == 0)
        ? VectorR::Ones(objectives_number)
        : importance_scale;

    throw_if(scale.size() != objectives_number, "Importance scale size must match objectives number.\n");

    throw_if(scale.minCoeff() < float(0), "Importance scale must be non-negative.\n");

    throw_if(scale.maxCoeff() == float(0), "Importance scale must contain at least one non-zero entry.\n");

    const Index inputs_number = neural_network->get_inputs_number();

    throw_if(pareto_front.cols() < inputs_number,
             "Pareto front has fewer columns than the number of input features.\n");

    const MatrixR pareto_inputs  = pareto_front.leftCols(inputs_number);
    const MatrixR pareto_outputs = pareto_front.rightCols(pareto_front.cols() - inputs_number);

    const Objectives objectives(*this);

    MatrixR objective_matrix = objectives.extract(pareto_inputs, pareto_outputs);

    VectorR normalized_utopian(objectives_number);

    for (Index j = 0; j < objectives_number; ++j)
    {
        const float col_min = objective_matrix.col(j).minCoeff();
        const float col_max = objective_matrix.col(j).maxCoeff();
        const float col_range = col_max - col_min;

        if (col_range > EPSILON)
            objective_matrix.col(j).array() = (objective_matrix.col(j).array() - col_min) / col_range;
        else
            objective_matrix.col(j).setZero();

        const float sense = objectives.utopian_and_sense(1, j);

        normalized_utopian(j) = (sense > float(0)) ? float(1) : float(0);
    }

    for (Index j = 0; j < objectives_number; ++j)
    {
        objective_matrix.col(j) *= scale(j);
        normalized_utopian(j)   *= scale(j);
    }

    const VectorI nearest = get_nearest_points(objective_matrix, normalized_utopian, 1);

    const Index advised_row_index = nearest(0);

    return {advised_row_index, pareto_front.row(advised_row_index).transpose()};
}


void ResponseOptimization::initialize_network_differential() const
{
    // The analytic Jacobian models the network, not the constraints, so it is built and validated
    // once and reused across every branch; rebuilding it per solve_once would repeat the same work.
    // Invalidated when the network or the formula constraints change.
    if (network_jacobian.ready)
        return;

    network_jacobian.ready = true;
    network_jacobian.differential.reset();

    if (!neural_network || is_forecasting())
        return;

    // Only build if some part of the problem (base set or any disjunctive branch) carries a
    // non-callback output constraint that the analytic Jacobian could repair.
    const auto has_output_constraint = [](const vector<MultivariateConstraint>& list)
    {
        for (const MultivariateConstraint& constraint : list)
            if (!constraint.uses_callback
                && constraint.comparison_operator != ComparisonOperator::None
                && constraint.compiled.scope != FormulaScope::InputsOnly)
                return true;
        return false;
    };

    bool has_output = has_output_constraint(constraint_set.multivariate);
    for (const vector<vector<MultivariateConstraint>>& disjunction : constraint_set.disjunctive)
        for (const vector<MultivariateConstraint>& branch : disjunction)
            has_output = has_output || has_output_constraint(branch);

    if (!has_output)
        return;

    auto candidate = make_unique<NetworkDifferential>();

    try { candidate->build(*neural_network); }
    catch (const exception& error)
    {
        cout << "!!! [Warning] Analytic surrogate Jacobian unavailable (" << error.what()
             << "); falling back to the finite-difference VJP." << endl;
        return;
    }

    const Index inputs_number = neural_network->get_inputs_number();

    VectorR lower = VectorR::Constant(inputs_number, -1.0f);
    VectorR upper = VectorR::Constant(inputs_number,  1.0f);
    if (!candidate->layers.empty() && candidate->layers.front().kind == NetworkDifferential::Kind::Scale)
    {
        lower = candidate->layers.front().minimum;
        upper = candidate->layers.front().maximum;
    }

    const Index forward_probes = 16;
    const Index vjp_probes = 4;

    MatrixR probe(forward_probes, inputs_number);
    set_random_uniform(probe, 0, 1);
    for (Index j = 0; j < inputs_number; ++j)
        probe.col(j) = (lower(j) + probe.col(j).array() * (upper(j) - lower(j))).matrix();

    const MatrixR reference = calculate_outputs(probe);
    const Index outputs_number = reference.cols();

    VectorR finite_difference_step(inputs_number);
    for (Index j = 0; j < inputs_number; ++j)
        finite_difference_step(j) = max(1e-4f, 1e-3f * (upper(j) - lower(j)));

    const VectorR cotangent = VectorR::Ones(outputs_number);

    float worst_forward_error = 0.0f;
    float worst_vjp_error = 0.0f;

    for (Index i = 0; i < forward_probes; ++i)
    {
        const VectorR x = probe.row(i).transpose();

        const VectorR analytic_output = candidate->forward(x);
        const VectorR truth = reference.row(i).transpose();
        worst_forward_error = max(worst_forward_error,
            (analytic_output - truth).cwiseAbs().maxCoeff() / (1.0f + truth.cwiseAbs().maxCoeff()));

        if (i >= vjp_probes) continue;

        const VectorR analytic_gradient = candidate->vjp(x, cotangent);
        VectorR finite_difference_gradient = VectorR::Zero(inputs_number);
        for (Index k = 0; k < inputs_number; ++k)
        {
            MatrixR plus(1, inputs_number), minus(1, inputs_number);
            plus.row(0) = x.transpose();   minus.row(0) = x.transpose();
            plus(0, k) += finite_difference_step(k);      minus(0, k) -= finite_difference_step(k);
            const VectorR forward_plus  = calculate_outputs(plus).row(0).transpose();
            const VectorR forward_minus = calculate_outputs(minus).row(0).transpose();
            finite_difference_gradient(k) = cotangent.dot(forward_plus - forward_minus) / (2.0f * finite_difference_step(k));
        }
        worst_vjp_error = max(worst_vjp_error,
            (analytic_gradient - finite_difference_gradient).cwiseAbs().maxCoeff()
            / (1.0f + analytic_gradient.cwiseAbs().maxCoeff()));
    }

    if (worst_forward_error < 1e-3f && worst_vjp_error < 2e-2f)
    {
        network_jacobian.differential = move(candidate);
        cout << "> Analytic surrogate Jacobian active (forward error " << worst_forward_error
             << ", VJP-vs-finite-difference error " << worst_vjp_error << ")." << endl;
    }
    else
        cout << "!!! [Warning] Analytic surrogate Jacobian failed validation (forward error "
             << worst_forward_error << ", VJP error " << worst_vjp_error
             << "); falling back to the finite-difference VJP." << endl;
}


static bool branch_is_dominated(const MatrixR& branch_objective,
                                const MatrixR& incumbent_objective,
                                const Index objectives_number,
                                const float margin)
{
    if (incumbent_objective.rows() == 0)
        return false;

    for (Index i = 0; i < branch_objective.rows(); ++i)
    {
        bool point_dominated = false;

        for (Index q = 0; q < incumbent_objective.rows() && !point_dominated; ++q)
        {
            bool all_greater_equal = true;
            bool some_greater = false;

            for (Index k = 0; k < objectives_number; ++k)
            {
                const float difference = incumbent_objective(q, k) - branch_objective(i, k);
                if (difference < -margin) { all_greater_equal = false; break; }
                if (difference >  margin) some_greater = true;
            }

            point_dominated = all_greater_equal && some_greater;
        }

        if (!point_dominated)
            return false;
    }

    return true;
}


MatrixR ResponseOptimization::solve_once() const
{
    const Index objectives_number = get_objectives_number();

    throw_if(objectives_number == 0, "No objectives found\n");

    evaluations_used = 0;
    sampling_memory = SamplingMemory();

    initialize_network_differential();

    return (objectives_number == 1)
        ? perform_single_objective_optimization()
        : perform_multiobjective_optimization();
}


MatrixR ResponseOptimization::perform_response_optimization()
{
    promote_single_variable_constraints();


    struct BranchAxis
    {
        enum class Type { Variable, Formula, Disjunction };
        Type type = Type::Variable;
        string variable_name;
        Index index = 0;        // constraint_set.multivariate index (Formula) or constraint_set.disjunctive index (Disjunction)
        vector<float> values;   // membership values (Variable/Formula) or branch indices (Disjunction)
    };

    const vector<Variable>& input_variables = neural_network->get_input_variables();
    const vector<Variable>& output_variables = neural_network->get_output_variables();

    const vector<NamedColumn> input_columns = build_input_columns(input_variables);

    bool any_callback_formula = false;
    for (const MultivariateConstraint& formula_constraint : constraint_set.multivariate)
        if (formula_constraint.uses_callback)
            any_callback_formula = true;

    auto input_column_of = [&](const string& name) -> Index
    {
        for (const NamedColumn& column : input_columns)
            if (column.name == name) return column.column_index;
        return -1;
    };

    auto input_referenced_by_formula = [&](const Index column) -> bool
    {
        if (any_callback_formula) return true;
        for (const MultivariateConstraint& formula_constraint : constraint_set.multivariate)
            for (const Index referenced : formula_constraint.compiled.input_indices)
                if (referenced == column) return true;
        return false;
    };

    auto feature_count_of = [&](const string& name) -> Index
    {
        for (const Variable& variable : input_variables)  if (variable.name == name) return variable.get_feature_count();
        for (const Variable& variable : output_variables) if (variable.name == name) return variable.get_feature_count();
        return 1;
    };

    auto is_input_name = [&](const string& name) -> bool
    {
        for (const Variable& variable : input_variables) if (variable.name == name) return true;
        return false;
    };

    vector<BranchAxis> axes;

    for (const auto& [name, constraint] : constraint_set.univariate)
    {
        if (constraint.comparison != ComparisonOperator::AllowedSet || constraint.allowed_values.empty())
            continue;

        const bool scalar = (feature_count_of(name) == 1);

        if (!scalar)
            continue;

        if (is_input_name(name) && !input_referenced_by_formula(input_column_of(name)))
            continue;

        axes.push_back({BranchAxis::Type::Variable, name, 0, constraint.allowed_values});
    }

    for (Index j = 0; j < static_cast<Index>(constraint_set.multivariate.size()); ++j)
        if (constraint_set.multivariate[j].comparison_operator == ComparisonOperator::AllowedSet
            && !constraint_set.multivariate[j].allowed_values.empty())
            axes.push_back({BranchAxis::Type::Formula, string(), j, constraint_set.multivariate[j].allowed_values});

    for (Index d = 0; d < static_cast<Index>(constraint_set.disjunctive.size()); ++d)
    {
        vector<float> branch_indices(constraint_set.disjunctive[d].size());
        iota(branch_indices.begin(), branch_indices.end(), 0.0f);
        axes.push_back({BranchAxis::Type::Disjunction, string(), d, branch_indices});
    }

    if (axes.empty())
        return solve_once();

    Index branches_number = 1;
    for (const BranchAxis& axis : axes)
        branches_number *= static_cast<Index>(axis.values.size());

    cout << "> Branching: " << axes.size() << " axis(es) -> " << branches_number
         << (branch_mode == BranchMode::Budgeted ? " branch(es) (budgeted: successive-halving + dominated-drop)."
                                                  : " branch(es) (exhaustive).") << endl;

    vector<vector<float>> branch_values(branches_number, vector<float>(axes.size()));
    {
        vector<Index> radix(axes.size(), 0);
        for (Index branch = 0; branch < branches_number; ++branch)
        {
            for (size_t a = 0; a < axes.size(); ++a)
                branch_values[branch][a] = axes[a].values[radix[a]];

            for (size_t a = 0; a < axes.size(); ++a)
            {
                if (++radix[a] < static_cast<Index>(axes[a].values.size())) break;
                radix[a] = 0;
            }
        }
    }

    const map<string, UnivariateConstraint> saved_constraints = constraint_set.univariate;
    const vector<MultivariateConstraint> saved_formula_constraints = constraint_set.multivariate;
    const Index saved_budget = max_total_evaluations;

    const Index input_features = get_features_number(get_variables_and_descriptives("Input").first);

    const Objectives objectives(*this);
    const Index objectives_number = static_cast<Index>(objectives.utopian_and_sense.cols());

    Index spent = 0;

    auto solve_branch = [&](const vector<float>& values, const Index cap) -> MatrixR
    {
        for (size_t a = 0; a < axes.size(); ++a)
        {
            if (axes[a].type == BranchAxis::Type::Formula)
            {
                MultivariateConstraint& formula_constraint = constraint_set.multivariate[axes[a].index];
                formula_constraint.comparison_operator = ComparisonOperator::EqualTo;
                formula_constraint.low_bound = values[a];
                formula_constraint.up_bound = values[a];
                formula_constraint.kind = classify(formula_constraint);
            }
            else if (axes[a].type == BranchAxis::Type::Disjunction)
            {
                const vector<MultivariateConstraint>& branch = constraint_set.disjunctive[axes[a].index][static_cast<Index>(values[a])];
                constraint_set.multivariate.insert(constraint_set.multivariate.end(), branch.begin(), branch.end());
            }
            else
                constraint_set.univariate[axes[a].variable_name] = UnivariateConstraint(ComparisonOperator::EqualTo, values[a], values[a]);
        }

        max_total_evaluations = cap;

        MatrixR result;
        try { result = solve_once(); }
        catch (const exception& error) { cout << "    (branch infeasible: " << error.what() << ")" << endl; }

        spent += evaluations_used;

        constraint_set.univariate = saved_constraints;
        constraint_set.multivariate = saved_formula_constraints;
        max_total_evaluations = saved_budget;

        return result;
    };

    auto objective_of = [&](const MatrixR& result) -> MatrixR
    {
        MatrixR matrix = objectives.extract(result.leftCols(input_features),
                                            result.rightCols(result.cols() - input_features));
        objectives.normalize(matrix);
        return matrix;
    };

    MatrixR incumbent;
    MatrixR incumbent_objective;

    auto merge_into_incumbent = [&](const MatrixR& result)
    {
        if (result.rows() == 0)
            return;

        incumbent = (incumbent.rows() == 0) ? result : append_rows(incumbent, result);

        MatrixR matrix = objective_of(incumbent);
        const auto [front_inputs, front_outputs] = calculate_pareto(incumbent.leftCols(input_features),
                                                                    incumbent.rightCols(incumbent.cols() - input_features),
                                                                    matrix);
        incumbent = append_columns(front_inputs, front_outputs);
        incumbent_objective = objective_of(incumbent);
    };

    if (branch_mode == BranchMode::Exhaustive || branches_number == 1)
    {
        const Index per_branch_budget = (saved_budget > 0) ? max(Index(1), saved_budget / branches_number) : 0;

        for (Index branch = 0; branch < branches_number; ++branch)
        {
            cout << "\n=== AllowedSet branch " << branch + 1 << " / " << branches_number << " ===" << endl;
            merge_into_incumbent(solve_branch(branch_values[branch], per_branch_budget));
        }

        evaluations_used = spent;
        cout << "\n> AllowedSet: " << incumbent.rows() << " point(s) on the global front." << endl;
        return incumbent;
    }

    const Index reduction_factor = 3;

    Index rounds_number = 1;
    for (Index remaining = branches_number; remaining > 1; remaining = (remaining + reduction_factor - 1) / reduction_factor)
        ++rounds_number;

    const Index notional_total = branches_number * evaluations_number * max(Index(1), max_iterations);

    vector<Index> live(branches_number);
    iota(live.begin(), live.end(), 0);

    float drop_margin = 0.1f;

    for (Index round = 0; round < rounds_number && static_cast<Index>(live.size()) > 1; ++round)
    {
        const Index pool = (saved_budget > 0)
            ? max(Index(0), saved_budget - spent) / (rounds_number - round)
            : notional_total / rounds_number;

        if (saved_budget > 0 && pool == 0)
            break;

        const Index cap = max(evaluations_number, pool / static_cast<Index>(live.size()));

        cout << "\n=== AllowedSet round " << round + 1 << " / " << rounds_number
             << ": " << live.size() << " live branch(es), <= " << cap << " evals each ===" << endl;

        vector<float> reward(live.size(), -1e30f);
        vector<MatrixR> round_objective(live.size());
        vector<bool> feasible(live.size(), false);

        for (size_t i = 0; i < live.size(); ++i)
        {
            const MatrixR result = solve_branch(branch_values[live[i]], cap);
            merge_into_incumbent(result);

            if (result.rows() > 0)
            {
                feasible[i] = true;
                round_objective[i] = objective_of(result);
                reward[i] = round_objective[i].rowwise().sum().maxCoeff();
            }
        }

        vector<size_t> survivors;
        for (size_t i = 0; i < live.size(); ++i)
            if (feasible[i] && !branch_is_dominated(round_objective[i], incumbent_objective, objectives_number, drop_margin))
                survivors.push_back(i);

        sort(survivors.begin(), survivors.end(),
             [&](const size_t a, const size_t b) { return reward[a] > reward[b]; });

        const Index keep = max(Index(1), (static_cast<Index>(survivors.size()) + reduction_factor - 1) / reduction_factor);
        if (static_cast<Index>(survivors.size()) > keep)
            survivors.resize(static_cast<size_t>(keep));

        vector<Index> next_live;
        next_live.reserve(survivors.size());
        for (const size_t i : survivors)
            next_live.push_back(live[i]);

        live = move(next_live);
        drop_margin *= 0.5f;
    }

    const Index polish_cap = (saved_budget > 0) ? max(Index(0), saved_budget - spent) : 0;
    if (saved_budget == 0 || polish_cap > 0)
        for (const Index branch : live)
            merge_into_incumbent(solve_branch(branch_values[branch], polish_cap));

    evaluations_used = spent;

    cout << "\n> AllowedSet: spent ~" << spent << " evaluations -> "
         << incumbent.rows() << " point(s) on the global front." << endl;

    return incumbent;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
