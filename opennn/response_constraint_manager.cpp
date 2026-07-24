//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   C O N S T R A I N T   M A N A G E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "response_constraint_manager.h"
#include "response_optimization.h"
#include "neural_network.h"
#include "string_utilities.h"
#include "random_utilities.h"

#include <cctype>
#include <set>

namespace opennn
{



namespace
{

bool is_selector(const ExpressionNode& node)
{
    return node.kind == ExpressionNode::Kind::Func
        && (node.function_name == "min" || node.function_name == "max" || node.function_name == "abs");
}


void collect_selectors(const ExpressionNode& node, vector<const ExpressionNode*>& selectors)
{
    if (is_selector(node))
        selectors.push_back(&node);
    for (const ExpressionNodePtr& child : node.children)
        collect_selectors(*child, selectors);
}


vector<const ExpressionNode*> selectors_of(const ExpressionNode& node)
{
    vector<const ExpressionNode*> selectors;
    collect_selectors(node, selectors);
    return selectors;
}


bool has_selector(const ExpressionNode& node) { return !selectors_of(node).empty(); }


// Rebuild the AST with every min/max/abs replaced by the argument the mode selects, so the
// result is smooth in the region defined by those mode choices.
ExpressionNodePtr resolve_smooth(const ExpressionNode& node, const map<const ExpressionNode*, int>& modes)
{
    if (is_selector(node))
    {
        const int mode = modes.at(&node);
        if (node.function_name == "abs")
        {
            ExpressionNodePtr child = resolve_smooth(*node.children[0], modes);
            return (mode == 0) ? move(child) : make_neg(move(child));
        }
        return resolve_smooth(*node.children[mode], modes);
    }

    auto rebuilt = make_unique<ExpressionNode>();
    rebuilt->kind = node.kind;
    rebuilt->constant = node.constant;
    rebuilt->index = node.index;
    rebuilt->function_name = node.function_name;
    rebuilt->children.reserve(node.children.size());
    for (const ExpressionNodePtr& child : node.children)
        rebuilt->children.push_back(resolve_smooth(*child, modes));
    return rebuilt;
}


MultivariateConstraint make_smooth_constraint(ExpressionNodePtr ast, const Condition comparison, const float low, const float up)
{
    MultivariateConstraint constraint;
    constraint.condition = comparison;
    constraint.low_bound = low;
    constraint.up_bound = up;
    constraint.compiled = compile_ast(*ast);
    constraint.kind = classify(constraint);
    return constraint;
}


// Inequality that pins one min/max/abs node to the argument chosen by `mode`.
MultivariateConstraint region_constraint(const ExpressionNode& selector, const int mode, const map<const ExpressionNode*, int>& modes)
{
    if (selector.function_name == "abs")
        return make_smooth_constraint(resolve_smooth(*selector.children[0], modes),
                                      mode == 0 ? Condition::GreaterEqualTo : Condition::LessEqualTo, 0.0f, 0.0f);

    ExpressionNodePtr difference = make_sub(resolve_smooth(*selector.children[0], modes),
                                 resolve_smooth(*selector.children[1], modes));

    const bool less_equal = (selector.function_name == "min") ? (mode == 0) : (mode == 1);
    return make_smooth_constraint(move(difference),
                                  less_equal ? Condition::LessEqualTo : Condition::GreaterEqualTo, 0.0f, 0.0f);
}


// Full 2^k region enumeration: exact for any operator and any nesting of min/max/abs. Each
// branch is a conjunction (the substituted-smooth constraint plus the region inequalities); the
// union over branches reproduces the original feasible set.
vector<vector<MultivariateConstraint>> enumerate_regions(const ExpressionNode& root,
                                                         const Condition comparison, const float low, const float up,
                                                         const vector<const ExpressionNode*>& selectors)
{
    const int count = static_cast<int>(selectors.size());

    vector<vector<MultivariateConstraint>> branches;

    for (int combination = 0; combination < (1 << count); ++combination)
    {
        map<const ExpressionNode*, int> modes;
        for (int i = 0; i < count; ++i)
            modes[selectors[i]] = (combination >> i) & 1;

        vector<MultivariateConstraint> branch;
        branch.push_back(make_smooth_constraint(resolve_smooth(root, modes), comparison, low, up));
        for (int i = 0; i < count; ++i)
            branch.push_back(region_constraint(*selectors[i], modes.at(selectors[i]), modes));

        branches.push_back(move(branch));
    }

    return branches;
}


// Disjunctive-normal-form expansion of one (possibly non-smooth) constraint. A top-level AND
// case with smooth arguments stays a single branch (no disjunction); everything else falls back
// to the general region enumeration.
vector<vector<MultivariateConstraint>> expand_ast(const ExpressionNode& root,
                                                  const Condition comparison, const float low, const float up)
{
    const vector<const ExpressionNode*> selectors = selectors_of(root);

    if (selectors.empty())
        return { { make_smooth_constraint(clone(root), comparison, low, up) } };

    if (is_selector(root)
        && ranges::none_of(root.children, [](const ExpressionNodePtr& child) { return has_selector(*child); }))
    {
        const string& name = root.function_name;
        const bool ge = (comparison == Condition::GreaterEqualTo || comparison == Condition::GreaterThan);
        const bool le = (comparison == Condition::LessEqualTo || comparison == Condition::LessThan);

        if ((name == "min" && ge) || (name == "max" && le))
        {
            vector<MultivariateConstraint> branch;
            for (const ExpressionNodePtr& child : root.children)
                branch.push_back(make_smooth_constraint(clone(*child), comparison, low, up));
            return { move(branch) };
        }
        if (name == "abs" && le)
            return { { make_smooth_constraint(clone(*root.children[0]), Condition::Between, -up, up) } };
    }

    return enumerate_regions(root, comparison, low, up, selectors);
}

}


void ConstraintManager::build(const ResponseOptimization& new_problem, const ConstraintGeometry& new_geometry, const bool lower)
{
    problem = &new_problem;
    geometry = new_geometry;

    constraint_set = ConstraintSet{};
    absorbed_objectives.clear();

    build_columns();

    for (const auto& constraint : problem->get_constraints())
        if (constraint.condition == Condition::Cardinality)
            add_cardinality(constraint.expression, constraint.values);
        else
            add_constraint(constraint.expression, constraint.condition, constraint.values);

    if (lower)
    {
        expand_fixed_objectives();
        promote_single_variable_constraints();
    }
}


void ConstraintManager::build_columns()
{
    input_columns.clear();
    output_columns.clear();
    input_names.clear();

    NeuralNetwork* neural_network = geometry.neural_network;

    Index input_column = 0;
    for (const Variable& variable : neural_network->get_input_variables())
    {
        input_names.insert(variable.name);

        if (variable.get_feature_count() == 1 && variable.get_role() == "Input" && !geometry.is_history(variable.name))
            input_columns.emplace_back(variable.name, input_column);

        input_column += variable.get_feature_count();
    }

    Index output_column = 0;
    for (const Variable& variable : neural_network->get_output_variables())
    {
        if (variable.get_feature_count() == 1)
            output_columns.emplace_back(variable.name, output_column);

        output_column += variable.get_feature_count();
    }
}


Index ConstraintManager::input_column_of(const string& name) const
{
    for (const auto& column : input_columns)
        if (column.first == name)
            return column.second;
    return -1;
}


bool ConstraintManager::is_input_name(const string& name) const
{
    return input_names.count(name) > 0;
}


void ConstraintManager::add_constraint(const string& expression, const Condition condition, const vector<float>& values)
{
    if (condition == Condition::None || condition == Condition::Integer)
        return;

    const Condition comparison = condition;

    const float low = values.empty() ? 0.0f : values[0];
    const float up = values.size() > 1 ? values[1] : low;

    if (is_input_name(expression))
    {
        if (condition == Condition::AllowedSet)
        {
            UnivariateConstraint univariate(Condition::AllowedSet);
            univariate.allowed_values = values;
            constraint_set.univariate[expression] = move(univariate);
        }
        else
            constraint_set.univariate[expression] = UnivariateConstraint(comparison, low, up);

        return;
    }

    if (condition == Condition::AllowedSet)
    {
        MultivariateConstraint formula_constraint;
        formula_constraint.expression = expression;
        formula_constraint.condition = Condition::AllowedSet;
        formula_constraint.allowed_values = values;
        formula_constraint.compiled = compile_formula(expression, input_columns, output_columns);
        formula_constraint.kind = classify(formula_constraint);
        constraint_set.multivariate.push_back(move(formula_constraint));
        return;
    }

    add_formula(expression, comparison, low, up);
}


void ConstraintManager::add_cardinality(const string& expression, const vector<float>& values)
{
    vector<string> names;
    string current;
    for (const char character : expression)
    {
        if (character == ',') { names.push_back(current); current.clear(); }
        else current += character;
    }
    if (!current.empty())
        names.push_back(current);

    const Index k = values.empty() ? 0 : Index(llround(values[0]));
    const bool force_nonzero = values.size() > 1 ? values[1] > 0.5f : true;

    constraint_set.cardinality.push_back({ move(names), k, force_nonzero });
}


void ConstraintManager::add_formula(const string& expression, const Condition op, const float low, const float up)
{
    vector<vector<MultivariateConstraint>> branches;

    try
    {
        branches = expand_constraint(expression, op, low, up, input_columns, output_columns);
    }
    catch (const exception&)
    {
        MultivariateConstraint formula_constraint;
        formula_constraint.expression = expression;
        formula_constraint.condition = op;
        formula_constraint.low_bound = low;
        formula_constraint.up_bound = up;
        formula_constraint.compiled = compile_formula(expression, input_columns, output_columns);
        formula_constraint.kind = classify(formula_constraint);
        branches = { { move(formula_constraint) } };
    }

    if (branches.size() == 1)
        for (MultivariateConstraint& formula_constraint : branches[0])
            constraint_set.multivariate.push_back(move(formula_constraint));
    else
        constraint_set.disjunctive.push_back(move(branches));
}


void ConstraintManager::expand_fixed_objectives()
{
    // Fixed objectives keyed by name (sorted) — reproduces the original lowering order.
    map<string, float> fixed_targets;
    for (const ResponseOptimization::Objective& objective : problem->get_objectives())
        if (objective.sense == Sense::Fixed)
            fixed_targets[objective.expression] = objective.value;

    if (fixed_targets.empty())
        return;

    map<string, float> output_range;
    const bool any_output_fixed = ranges::any_of(fixed_targets,
        [&](const auto& entry){ return !is_input_name(entry.first); });

    if (any_output_fixed)
    {
        const vector<Variable>& target_variables = *geometry.target_variables;
        const vector<Descriptives>& target_descriptives = *geometry.target_descriptives;
        for (size_t i = 0; i < target_variables.size(); ++i)
            output_range[target_variables[i].name] =
                static_cast<float>(target_descriptives[i].maximum - target_descriptives[i].minimum);
    }

    for (const auto& [name, target] : fixed_targets)
    {
        if (is_input_name(name))
        {
            constraint_set.univariate[name] = UnivariateConstraint(Condition::EqualTo, target, target);
            absorbed_objectives.insert(name);
            continue;
        }

        const auto found = output_range.find(name);
        const float range = (found != output_range.end()) ? found->second : abs(target);
        const float half_width = max(bound_tolerance(target), relative_tolerance * max(EPSILON, range));

        add_formula(name, Condition::Between, target - half_width, target + half_width);

        cout << "> Fixed objective '" << name << "' -> output equality band ["
             << (target - half_width) << ", " << (target + half_width) << "].\n";
    }
}


void ConstraintManager::promote_single_variable_constraints()
{
    if (constraint_set.multivariate.empty())
        return;

    map<Index, string> name_of_column;
    for (const auto& column : input_columns)
        name_of_column[column.second] = column.first;

    auto interval_of = [](const UnivariateConstraint& constraint, float& lo, float& hi) -> bool
    {
        lo = -numeric_limits<float>::infinity();
        hi =  numeric_limits<float>::infinity();
        switch (constraint.condition)
        {
        case Condition::None:           return true;
        case Condition::EqualTo:        lo = hi = constraint.low_bound; return true;
        case Condition::Between:        lo = constraint.low_bound; hi = constraint.up_bound; return true;
        case Condition::GreaterEqualTo:
        case Condition::GreaterThan:    lo = constraint.low_bound; return true;
        case Condition::LessEqualTo:
        case Condition::LessThan:       hi = constraint.up_bound; return true;
        case Condition::AllowedSet:
        default:                                 return false;
        }
    };

    vector<MultivariateConstraint> kept;
    kept.reserve(constraint_set.multivariate.size());

    for (MultivariateConstraint& formula_constraint : constraint_set.multivariate)
    {
        const CompiledExpression& compiled = formula_constraint.compiled;

        const bool promotable = formula_constraint.condition != Condition::None
            && formula_constraint.condition != Condition::AllowedSet
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

        switch (formula_constraint.condition)
        {
        case Condition::EqualTo:
            implied_lo = implied_hi = solve(low); break;
        case Condition::Between:
            implied_lo = min(solve(low), solve(up));
            implied_hi = max(solve(low), solve(up)); break;
        case Condition::GreaterEqualTo:
        case Condition::GreaterThan:
            (coefficient > 0.0f ? implied_lo : implied_hi) = solve(low); break;
        case Condition::LessEqualTo:
        case Condition::LessThan:
            (coefficient > 0.0f ? implied_hi : implied_lo) = solve(up); break;
        case Condition::None:
        case Condition::AllowedSet:
            break;
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
                 "IDC: constraint '" + formula_constraint.expression
                 + "' leaves variable '" + name + "' with an empty box.");

        const bool lo_finite = isfinite(new_lo);
        const bool hi_finite = isfinite(new_hi);

        if (lo_finite && hi_finite && new_lo == new_hi)
            constraint_set.univariate[name] = UnivariateConstraint(Condition::EqualTo, new_lo, new_lo);
        else if (lo_finite && hi_finite)
            constraint_set.univariate[name] = UnivariateConstraint(Condition::Between, new_lo, new_hi);
        else if (lo_finite)
            constraint_set.univariate[name] = UnivariateConstraint(Condition::GreaterEqualTo, new_lo, 0.0f);
        else if (hi_finite)
            constraint_set.univariate[name] = UnivariateConstraint(Condition::LessEqualTo, 0.0f, new_hi);

        cout << "> Promoted single-variable constraint '" << formula_constraint.expression
             << "' to a box on '" << name << "'." << "\n";
    }

    constraint_set.multivariate = move(kept);
}


bool ConstraintManager::is_objective(const string& name) const
{
    return problem->is_objective(name) && absorbed_objectives.count(name) == 0;
}


Sense ConstraintManager::get_sense(const string& name) const
{
    return problem->get_sense(name);
}


float ConstraintManager::get_fixed_value(const string& name) const
{
    return problem->get_fixed_value(name);
}


UnivariateConstraint ConstraintManager::get_constraint(const string& name) const
{
    const auto it = constraint_set.univariate.find(name);
    return it != constraint_set.univariate.end() ? it->second : UnivariateConstraint(Condition::None);
}


Index ConstraintManager::get_optimizing_objectives_number() const
{
    const auto is_opt = [&](const Variable& v){ return is_objective(v.name) && get_sense(v.name) != Sense::Fixed; };
    return ranges::count_if((*geometry.input_variables), is_opt)
         + ranges::count_if((*geometry.target_variables), is_opt);
}


Index ConstraintManager::get_objectives_number() const
{
    const Index optimizing = get_optimizing_objectives_number();
    if (optimizing > 0)
        return optimizing;

    const auto is_fixed = [&](const Variable& v){ return is_objective(v.name) && get_sense(v.name) == Sense::Fixed; };
    return ranges::count_if((*geometry.input_variables), is_fixed)
         + ranges::count_if((*geometry.target_variables), is_fixed);
}


vector<vector<MultivariateConstraint>> expand_constraint(const string& expression,
                                                         const Condition comparison,
                                                         const float low, const float up,
                                                         const vector<pair<string, Index>>& inputs,
                                                         const vector<pair<string, Index>>& outputs)
{
    ExpressionNodePtr ast = parse_expression_tree(expression, inputs, outputs);

    vector<vector<MultivariateConstraint>> branches = expand_ast(*ast, comparison, low, up);

    if (branches.size() == 1 && branches[0].size() == 1)
        branches[0][0].expression = expression;

    return branches;
}


bool all_formula_constraints_are_linear(const vector<MultivariateConstraint>& formula_constraints)
{
    return !formula_constraints.empty()
        && ranges::all_of(formula_constraints, [](const MultivariateConstraint& formula_constraint)
           {
               return formula_constraint.compiled.shape == FormulaShape::Affine;
           });
}


LinearConstraintSet build_linear_constraint_set(const vector<MultivariateConstraint>& formula_constraints,
                                                const Index n_in,
                                                const Index n_out)
{
    const Index m = ssize(formula_constraints);

    LinearConstraintSet linear_set;
    linear_set.A     = MatrixR::Zero(m, n_in + n_out);
    linear_set.lower = VectorR::Constant(m, -numeric_limits<float>::infinity());
    linear_set.upper = VectorR::Constant(m,  numeric_limits<float>::infinity());

    for (Index i = 0; i < m; ++i)
    {
        const MultivariateConstraint& formula_constraint = formula_constraints[i];

        for (const auto& [column, coefficient] : formula_constraint.compiled.affine_input_terms)
            linear_set.A(i, column) += coefficient;

        for (const auto& [column, coefficient] : formula_constraint.compiled.affine_output_terms)
            linear_set.A(i, n_in + column) += coefficient;

        const float c = formula_constraint.compiled.affine_constant;
        const float low = formula_constraint.low_bound;
        const float up  = formula_constraint.up_bound;

        switch (formula_constraint.condition)
        {
            using enum Condition;
        case EqualTo:
            linear_set.lower(i) = low - c - bound_tolerance(low);
            linear_set.upper(i) = low - c + bound_tolerance(low);
            break;
        case Between:
            linear_set.lower(i) = low - c - bound_tolerance(low);
            linear_set.upper(i) = up  - c + bound_tolerance(up);
            break;
        case GreaterEqualTo:
            linear_set.lower(i) = low - c - bound_tolerance(low);
            break;
        case LessEqualTo:
            linear_set.upper(i) = up - c + bound_tolerance(up);
            break;
        case GreaterThan:
            linear_set.lower(i) = low - c + EPSILON;
            break;
        case LessThan:
            linear_set.upper(i) = up - c - EPSILON;
            break;
        case None:
        case AllowedSet:
        default:
            break;
        }
    }

    return linear_set;
}


bool constraint_is_satisfied(const MultivariateConstraint& constraint,
                             const VectorR& input_row,
                             const VectorR& output_row)
{
    const float value = constraint.compiled.evaluate(input_row, output_row);

    const float low = constraint.low_bound;
    const float up  = constraint.up_bound;

    switch (constraint.condition)
    {
        using enum Condition;
    case EqualTo:
        return abs(value - low) <= bound_tolerance(low);
    case Between:
        return value >= low - bound_tolerance(low) && value <= up + bound_tolerance(up);
    case GreaterEqualTo:
    case GreaterThan:
        return value >= low - bound_tolerance(low);
    case LessEqualTo:
    case LessThan:
        return value <= up + bound_tolerance(up);
    case None:
    case AllowedSet:
        return true;
    default:
        return true;
    }
}


RepairRegime classify(const MultivariateConstraint& constraint)
{
    if (constraint.condition == Condition::None)
        return RepairRegime::None;

    const CompiledExpression& formula = constraint.compiled;

    const bool affine = (formula.shape == FormulaShape::Affine);
    const bool nonlinear_ready = (formula.shape == FormulaShape::Nonlinear &&
                                  (!formula.input_gradient.empty() || !formula.output_gradient.empty()));

    if (formula.scope == FormulaScope::InputsOnly)
    {
        if (affine && !formula.affine_input_terms.empty())
            return RepairRegime::InputAffine;
        if (nonlinear_ready)
            return RepairRegime::InputNonlinear;
        return RepairRegime::None;
    }

    return (affine || nonlinear_ready) ? RepairRegime::OutputCoupled : RepairRegime::None;
}


void snap_to_lattice(MatrixR& inputs, const Index column, const float minimum, const float maximum)
{
    inputs.col(column).array() = inputs.col(column).array().round().max(minimum).min(maximum);
}


namespace
{

bool constraint_residual(const Condition comparison, const float low, const float up,
                         const float value, float& residual)
{
    residual = 0.0f;

    switch (comparison)
    {
        using enum Condition;
    case EqualTo:
        residual = value - low; return true;
    case Between:
        if (value < low) { residual = value - low; return true; }
        if (value > up)  { residual = value - up;  return true; }
        return false;
    case GreaterEqualTo:
    case GreaterThan:
        if (value < low) { residual = value - low; return true; }
        return false;
    case LessEqualTo:
    case LessThan:
        if (value > up) { residual = value - up; return true; }
        return false;
    case None:
    case AllowedSet:
    default:
        return false;
    }
}

bool gauss_newton_project_row(const MatrixR& jacobian, const VectorR& rhs,
                              const VectorR& inferior_frontier, const VectorR& superior_frontier,
                              VectorR& point)
{
    const Index active_number = jacobian.rows();

    vector<Index> projectable;
    projectable.reserve(active_number);
    for (Index i = 0; i < active_number; ++i)
        if (jacobian.row(i).cwiseAbs().maxCoeff() > 0.0f)
            projectable.push_back(i);

    if (projectable.empty())
        return false;

    const Index projectable_number = ssize(projectable);

    MatrixR reduced_jacobian(projectable_number, jacobian.cols());
    VectorR reduced_rhs(projectable_number);
    for (Index k = 0; k < projectable_number; ++k)
    {
        reduced_jacobian.row(k) = jacobian.row(projectable[k]);
        reduced_rhs(k) = rhs(projectable[k]);
    }

    MatrixR gram = reduced_jacobian * reduced_jacobian.transpose();
    gram.diagonal().array() += EPSILON;

    point -= reduced_jacobian.transpose() * gram.ldlt().solve(reduced_rhs);
    point = point.cwiseMax(inferior_frontier).cwiseMin(superior_frontier);
    return true;
}


vector<const MultivariateConstraint*> input_repairable_constraints(const vector<MultivariateConstraint>& formula_constraints)
{
    vector<const MultivariateConstraint*> constraints;

    for (const MultivariateConstraint& constraint : formula_constraints)
        if (constraint.kind == RepairRegime::InputAffine || constraint.kind == RepairRegime::InputNonlinear)
            constraints.push_back(&constraint);

    return constraints;
}


// Collects the constraints violated at `point` (with surrogate outputs `output`, empty for
// input-only constraints) together with their signed residuals. Returns false when none are
// violated or the worst residual is already within tolerance: nothing left to repair.
bool collect_violations(const vector<const MultivariateConstraint*>& constraints,
                        const VectorR& point, const VectorR& output,
                        vector<const MultivariateConstraint*>& active, vector<float>& residuals)
{
    active.clear();
    residuals.clear();
    active.reserve(constraints.size());
    residuals.reserve(constraints.size());

    for (const MultivariateConstraint* constraint : constraints)
    {
        const float value = constraint->compiled.evaluate(point, output);
        float residual;
        if (constraint_residual(constraint->condition, constraint->low_bound, constraint->up_bound, value, residual))
        {
            active.push_back(constraint);
            residuals.push_back(residual);
        }
    }

    if (active.empty())
        return false;

    return VectorR::Map(residuals.data(), Index(residuals.size())).cwiseAbs().maxCoeff() > EPSILON;
}


// Gradient of one constraint with respect to the inputs. For output-dependent constraints `vjp`
// back-propagates the output sensitivity through the surrogate; for input-only constraints it is
// null and only the direct input gradient is returned.
VectorR constraint_gradient(const MultivariateConstraint& constraint,
                            const VectorR& point, const VectorR& output, const SurrogateVjp& vjp)
{
    const CompiledExpression& compiled = constraint.compiled;

    VectorR gradient = VectorR::Zero(point.size());

    if (compiled.shape == FormulaShape::Affine)
        for (const auto& [column, coefficient] : compiled.affine_input_terms)
            gradient(column) = coefficient;
    else
        for (const auto& [column, program] : compiled.input_gradient)
            gradient(column) = evaluate_operations(program, point, output);

    if (!vjp)
        return gradient;

    VectorR cotangent = VectorR::Zero(output.size());

    if (compiled.shape == FormulaShape::Affine)
        for (const auto& [column, coefficient] : compiled.affine_output_terms)
            cotangent(column) = coefficient;
    else
        for (const auto& [column, program] : compiled.output_gradient)
            cotangent(column) = evaluate_operations(program, point, output);

    return gradient + vjp(point, cotangent);
}


// Iterated Gauss-Newton projection of a single row onto the feasible set of `constraints`.
// `forward` maps the row to its surrogate outputs (null for input-only constraints); `vjp`
// back-propagates an output cotangent (null for input-only). Columns flagged in `fixed_columns`
// are held constant.
void gauss_newton_repair_row(VectorR& point,
                             const vector<const MultivariateConstraint*>& constraints,
                             const SurrogateForward& forward, const SurrogateVjp& vjp,
                             const vector<char>& fixed_columns,
                             const VectorR& inferior_frontier, const VectorR& superior_frontier,
                             const Index passes)
{
    const Index inputs_number = point.size();
    const bool has_mask = (ssize(fixed_columns) == inputs_number);

    vector<const MultivariateConstraint*> active;
    vector<float> residuals;

    for (Index pass = 0; pass < passes; ++pass)
    {
        const VectorR output = forward ? forward(point) : VectorR();

        if (!collect_violations(constraints, point, output, active, residuals))
            break;

        const Index active_number = ssize(active);

        MatrixR jacobian(active_number, inputs_number);
        VectorR rhs(active_number);

        for (Index i = 0; i < active_number; ++i)
        {
            rhs(i) = residuals[i];
            jacobian.row(i) = constraint_gradient(*active[i], point, output, vjp).transpose();
        }

        if (has_mask)
            for (Index j = 0; j < inputs_number; ++j)
                if (fixed_columns[j])
                    jacobian.col(j).setZero();

        if (!gauss_newton_project_row(jacobian, rhs, inferior_frontier, superior_frontier, point))
            break;
    }
}

}


void repair_affine_inputs(MatrixR& random_inputs,
                          const VectorR& inferior_frontier,
                          const VectorR& superior_frontier,
                          const vector<MultivariateConstraint>& formula_constraints,
                          const Index max_correction_passes)
{
    vector<const MultivariateConstraint*> affine_constraints;

    for (const MultivariateConstraint& constraint : formula_constraints)
        if (constraint.kind == RepairRegime::InputAffine)
            affine_constraints.push_back(&constraint);

    if (affine_constraints.empty())
        return;

    const Index rows_number = random_inputs.rows();
    const Index inputs_number = random_inputs.cols();
    const Index constraints_number = ssize(affine_constraints);

    Index slacks_number = 0;
    for (const MultivariateConstraint* constraint : affine_constraints)
        if (constraint->condition != Condition::EqualTo)
            ++slacks_number;

    const Index augmented_number = inputs_number + slacks_number;

    MatrixR augmented_matrix = MatrixR::Zero(constraints_number, augmented_number);
    VectorR right_hand_side = VectorR::Zero(constraints_number);
    VectorR slack_inferior = VectorR::Zero(slacks_number);
    VectorR slack_superior = VectorR::Zero(slacks_number);

    Index slack_index = 0;

    for (Index i = 0; i < constraints_number; ++i)
    {
        const MultivariateConstraint& constraint = *affine_constraints[i];
        const float constant = constraint.compiled.affine_constant;
        const float low = constraint.low_bound;
        const float up = constraint.up_bound;

        float expression_minimum = constant;
        float expression_maximum = constant;

        for (const auto& [column, coefficient] : constraint.compiled.affine_input_terms)
        {
            augmented_matrix(i, column) += coefficient;
            expression_minimum += min(coefficient * inferior_frontier(column), coefficient * superior_frontier(column));
            expression_maximum += max(coefficient * inferior_frontier(column), coefficient * superior_frontier(column));
        }

        switch (constraint.condition)
        {
            using enum Condition;
        case EqualTo:
            right_hand_side(i) = low - constant;
            break;

        case Between:
            augmented_matrix(i, inputs_number + slack_index) = -1;
            right_hand_side(i) = low - constant;
            slack_inferior(slack_index) = max(0.0f, expression_minimum - low);
            slack_superior(slack_index) = max(slack_inferior(slack_index), min(up - low, expression_maximum - low));
            ++slack_index;
            break;

        case GreaterEqualTo:
        case GreaterThan:
            augmented_matrix(i, inputs_number + slack_index) = -1;
            right_hand_side(i) = low - constant;
            slack_inferior(slack_index) = max(0.0f, expression_minimum - low);
            slack_superior(slack_index) = max(slack_inferior(slack_index), expression_maximum - low);
            ++slack_index;
            break;

        case LessEqualTo:
        case LessThan:
            augmented_matrix(i, inputs_number + slack_index) = 1;
            right_hand_side(i) = up - constant;
            slack_inferior(slack_index) = max(0.0f, up - expression_maximum);
            slack_superior(slack_index) = max(slack_inferior(slack_index), up - expression_minimum);
            ++slack_index;
            break;

        case None:
        case AllowedSet:
        default:
            break;
        }
    }

    MatrixR augmented_points(rows_number, augmented_number);
    augmented_points.leftCols(inputs_number) = random_inputs;

    if (slacks_number > 0)
    {
        MatrixR slack_points(rows_number, slacks_number);
        set_random_uniform(slack_points, 0, 1);

        for (Index j = 0; j < slacks_number; ++j)
            slack_points.col(j) = (slack_inferior(j) + slack_points.col(j).array() * (slack_superior(j) - slack_inferior(j))).matrix();

        augmented_points.rightCols(slacks_number) = slack_points;
    }

    MatrixR gram_matrix = augmented_matrix * augmented_matrix.transpose();
    gram_matrix.diagonal().array() += EPSILON;
    const auto gram_solver = gram_matrix.ldlt();

    MatrixR affine_correction = MatrixR::Zero(rows_number, augmented_number);
    MatrixR box_correction = MatrixR::Zero(rows_number, augmented_number);

    const Index passes = max(Index(1), max_correction_passes);

    for (Index pass = 0; pass < passes; ++pass)
    {
        const MatrixR shifted_points = augmented_points + affine_correction;

        MatrixR residual = shifted_points * augmented_matrix.transpose();
        residual.rowwise() -= right_hand_side.transpose();

        const MatrixR projected_points = shifted_points - (gram_solver.solve(residual.transpose())).transpose() * augmented_matrix;
        affine_correction = shifted_points - projected_points;

        augmented_points = projected_points + box_correction;

        for (Index j = 0; j < inputs_number; ++j)
            augmented_points.col(j) = augmented_points.col(j).array().max(inferior_frontier(j)).min(superior_frontier(j)).matrix();

        for (Index j = 0; j < slacks_number; ++j)
            augmented_points.col(inputs_number + j) = augmented_points.col(inputs_number + j).array().max(slack_inferior(j)).min(slack_superior(j)).matrix();

        box_correction = projected_points + box_correction - augmented_points;

        MatrixR feasibility_residual = augmented_points * augmented_matrix.transpose();
        feasibility_residual.rowwise() -= right_hand_side.transpose();

        if (feasibility_residual.cwiseAbs().maxCoeff() <= EPSILON)
            break;
    }

    random_inputs = augmented_points.leftCols(inputs_number);
}


void repair_single_affine_input(MatrixR& random_inputs,
                                const VectorR& inferior_frontier,
                                const VectorR& superior_frontier,
                                const MultivariateConstraint& constraint)
{
    const vector<pair<Index, float>>& terms = constraint.compiled.affine_input_terms;

    if (terms.empty())
        return;

    const float constant = constraint.compiled.affine_constant;
    const float low = constraint.low_bound;
    const float up  = constraint.up_bound;
    const Index rows_number = random_inputs.rows();

    vector<pair<Index, float>> shuffled(terms.begin(), terms.end());

    const Index terms_number = ssize(shuffled);

    for (Index r = 0; r < rows_number; ++r)
    {
        float expression = constant;
        for (const auto& [column, coefficient] : shuffled)
            expression += coefficient * random_inputs(r, column);

        float residual;
        if (!constraint_residual(constraint.condition, low, up, expression, residual))
            continue;
        residual = -residual;

        partial_shuffle(shuffled, terms_number);

        for (const auto& [column, coefficient] : shuffled)
        {
            if (coefficient == 0.0f)
                continue;

            const float old_value = random_inputs(r, column);
            const float wanted = old_value + residual / coefficient;
            const float new_value = min(superior_frontier(column),
                                        max(inferior_frontier(column), wanted));

            residual -= coefficient * (new_value - old_value);
            random_inputs(r, column) = new_value;

            if (abs(residual) <= EPSILON)
                break;
        }
    }
}


void repair_single_affine_integer(MatrixR& random_inputs,
                                  const VectorR& inferior_frontier,
                                  const VectorR& superior_frontier,
                                  const MultivariateConstraint& constraint)
{
    const vector<pair<Index, float>>& terms = constraint.compiled.affine_input_terms;

    if (terms.empty())
        return;

    const float constant = constraint.compiled.affine_constant;
    const float low = constraint.low_bound;
    const float up  = constraint.up_bound;
    const Index rows_number = random_inputs.rows();

    vector<pair<Index, float>> shuffled(terms.begin(), terms.end());
    const Index terms_number = ssize(shuffled);

    for (Index r = 0; r < rows_number; ++r)
    {
        for (const auto& [column, coefficient] : shuffled)
        {
            const float lattice_min = ceil(inferior_frontier(column));
            const float lattice_max = floor(superior_frontier(column));
            random_inputs(r, column) = min(lattice_max, max(lattice_min, round(random_inputs(r, column))));
        }

        float expression = constant;
        for (const auto& [column, coefficient] : shuffled)
            expression += coefficient * random_inputs(r, column);

        float residual;
        if (!constraint_residual(constraint.condition, low, up, expression, residual))
            continue;
        residual = -residual;

        partial_shuffle(shuffled, terms_number);

        for (const auto& [column, coefficient] : shuffled)
        {
            if (coefficient == 0.0f)
                continue;

            const float lattice_min = ceil(inferior_frontier(column));
            const float lattice_max = floor(superior_frontier(column));

            const float real_delta = residual / coefficient;
            const float integer_delta = (real_delta > 0.0f) ? floor(real_delta) : ceil(real_delta);
            if (integer_delta == 0.0f)
                continue;

            const float old_value = random_inputs(r, column);
            const float new_value = min(lattice_max, max(lattice_min, old_value + integer_delta));

            residual -= coefficient * (new_value - old_value);
            random_inputs(r, column) = new_value;

            if (abs(residual) <= EPSILON)
                break;
        }
    }
}


void repair_nonlinear_inputs(MatrixR& random_inputs,
                             const VectorR& inferior_frontier,
                             const VectorR& superior_frontier,
                             const vector<MultivariateConstraint>& formula_constraints,
                             const Index max_correction_passes)
{
    const bool any_nonlinear = ranges::any_of(formula_constraints, [](const MultivariateConstraint& constraint)
        { return constraint.kind == RepairRegime::InputNonlinear; });

    if (!any_nonlinear)
        return;

    repair_affine_inputs_with_fixed(random_inputs, inferior_frontier, superior_frontier,
                                    formula_constraints, {}, max_correction_passes);
}


void repair_affine_inputs_with_fixed(MatrixR& random_inputs,
                                     const VectorR& inferior_frontier,
                                     const VectorR& superior_frontier,
                                     const vector<MultivariateConstraint>& formula_constraints,
                                     const vector<char>& fixed_columns,
                                     const Index max_correction_passes)
{
    const vector<const MultivariateConstraint*> constraints = input_repairable_constraints(formula_constraints);

    if (constraints.empty())
        return;

    const Index rows_number = random_inputs.rows();
    const Index passes      = max(Index(1), max_correction_passes);

    for (Index r = 0; r < rows_number; ++r)
    {
        VectorR point = random_inputs.row(r).transpose();

        gauss_newton_repair_row(point, constraints, {}, {}, fixed_columns,
                                inferior_frontier, superior_frontier, passes);

        random_inputs.row(r) = point.transpose();
    }
}


namespace
{

// Partition the input-repairable constraints (AffineInput / NonlinearInput) into blocks whose
// variable sets are pairwise disjoint -- the connected components of the variable-constraint
// bipartite graph. Two constraints fall in the same block iff they share at least one input column
// (directly or transitively). Each block can then be repaired independently with the projector
// matched to its OWN kind, so a single nonlinear constraint no longer drags unrelated affine blocks
// through Gauss-Newton. This is the standard independent-components decomposition (cf. SCIP
// cons_components; Pierra's product-space projection), here applied to surrogate input repair.
vector<vector<MultivariateConstraint>>
partition_input_constraints_by_variable(const vector<MultivariateConstraint>& formula_constraints)
{
    vector<const MultivariateConstraint*> repairable;

    for (const MultivariateConstraint& constraint : formula_constraints)
        if (constraint.kind == RepairRegime::InputAffine
            || constraint.kind == RepairRegime::InputNonlinear)
            repairable.push_back(&constraint);

    const Index constraint_number = ssize(repairable);

    vector<Index> parent(constraint_number);
    iota(parent.begin(), parent.end(), Index{0});

    function<Index(Index)> find = [&](Index x)
    {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };

    // Union any two constraints that reference a common input column.
    unordered_map<Index, Index> column_owner;

    for (Index i = 0; i < constraint_number; ++i)
        for (const Index column : repairable[i]->compiled.input_indices)
        {
            const auto found = column_owner.find(column);
            if (found == column_owner.end())
                column_owner[column] = i;
            else
                parent[find(i)] = find(found->second);
        }

    unordered_map<Index, vector<MultivariateConstraint>> blocks;
    for (Index i = 0; i < constraint_number; ++i)
        blocks[find(i)].push_back(*repairable[i]);

    vector<vector<MultivariateConstraint>> result;
    result.reserve(blocks.size());
    for (auto& [root, block] : blocks)
        result.push_back(move(block));

    return result;
}

}


void repair_inputs(MatrixR& random_inputs,
                   const VectorR& inferior_frontier,
                   const VectorR& superior_frontier,
                   const vector<MultivariateConstraint>& formula_constraints)
{
    // Each block holds constraints over a variable set disjoint from every other block, so repairing
    // them one block at a time on the shared matrix is exact: an inner repairer only writes the
    // columns its constraints reference, leaving the other blocks' columns untouched. The win is
    // routing purity -- the affine blocks get Dykstra and only the genuinely nonlinear block gets
    // Gauss-Newton, instead of the whole input vector being forced through GN by one nonlinear term.
    for (const vector<MultivariateConstraint>& block :
         partition_input_constraints_by_variable(formula_constraints))
    {
        const MultivariateConstraint* single_affine = nullptr;
        Index affine_number = 0;
        bool any_nonlinear = false;

        for (const MultivariateConstraint& constraint : block)
        {
            if (constraint.kind == RepairRegime::InputAffine)
            {
                ++affine_number;
                single_affine = &constraint;
            }
            else if (constraint.kind == RepairRegime::InputNonlinear)
                any_nonlinear = true;
        }

        if (any_nonlinear)
            repair_nonlinear_inputs(random_inputs, inferior_frontier, superior_frontier, block);
        else if (affine_number == 1)
            repair_single_affine_input(random_inputs, inferior_frontier, superior_frontier, *single_affine);
        else if (affine_number >= 2)
            repair_affine_inputs(random_inputs, inferior_frontier, superior_frontier, block);
    }
}


namespace
{

bool row_satisfies_input_affine(const VectorR& point,
                                const vector<const MultivariateConstraint*>& input_constraints)
{
    const VectorR empty_outputs;

    return ranges::all_of(input_constraints, [&](const MultivariateConstraint* c) {
        return constraint_is_satisfied(*c, point, empty_outputs);
    });
}


void cardinality_swap_row(VectorR& point,
                          const vector<Index>& columns,
                          const VectorR& inferior_frontier,
                          const VectorR& superior_frontier,
                          const vector<char>& is_discrete,
                          const Index swaps)
{
    // Turn a cardinality member "on" with a type-consistent nonzero value in its box.
    const auto sample_on = [&](const Index column) -> float
    {
        const float inferior = inferior_frontier(column);
        const float superior = superior_frontier(column);

        if (column < ssize(is_discrete) && is_discrete[column])
            return (floor(superior) >= 1.0f) ? 1.0f : ceil(inferior);   // nonzero integer/binary

        float value = random_uniform(inferior, superior);               // continuous
        if (abs(value) < EPSILON)
            value = (superior > EPSILON) ? superior : inferior;
        return value;
    };

    for (Index s = 0; s < swaps; ++s)
    {
        vector<Index> off_candidates, on_candidates;

        for (const Index column : columns)
        {
            const bool on = abs(point(column)) > EPSILON;

            const bool box_contains_zero = (inferior_frontier(column) <= EPSILON
                                            && superior_frontier(column) >= -EPSILON);
            const bool box_has_nonzero   = (superior_frontier(column) >  EPSILON
                                            || inferior_frontier(column) < -EPSILON);

            if (on && box_contains_zero)       off_candidates.push_back(column);
            else if (!on && box_has_nonzero)   on_candidates.push_back(column);
        }

        if (off_candidates.empty() || on_candidates.empty())
            break;

        const Index off_column = off_candidates[random_integer(0, ssize(off_candidates) - 1)];
        const Index on_column  = on_candidates [random_integer(0, ssize(on_candidates)  - 1)];

        point(off_column) = 0.0f;
        point(on_column)  = sample_on(on_column);
    }
}


void unlock_free_integers_row(VectorR& point, const Lattice& lattice, const float fraction)
{
    for (size_t c = 0; c < lattice.columns.size(); ++c)
        if (random_uniform(0.0f, 1.0f) < fraction)
        {
            const float span = max(0.0f, lattice.max[c] - lattice.min[c]);
            const float draw = random_uniform(0.0f, 1.0f) * (span + 1.0f) + (lattice.min[c] - 0.5f);
            point(lattice.columns[c]) = min(lattice.max[c], max(lattice.min[c], round(draw)));
        }
}

}


void repair_mixed_integer_inputs(MatrixR& inputs,
                                 const VectorR& inferior_frontier,
                                 const VectorR& superior_frontier,
                                 const vector<MultivariateConstraint>& formula_constraints,
                                 const vector<char>& fixed_mask,
                                 const Lattice& lattice,
                                 const vector<vector<Index>>& cardinality_columns,
                                 const Lattice& free_lattice,
                                 const Index outer_cap,
                                 const float exploration_ratio)
{
    const vector<const MultivariateConstraint*> input_constraints = input_repairable_constraints(formula_constraints);

    const Index rows = inputs.rows();
    const Index passes = max(Index(1), outer_cap);

    for (size_t c = 0; c < lattice.columns.size(); ++c)
        snap_to_lattice(inputs, lattice.columns[c], lattice.min[c], lattice.max[c]);

    std::set<Index> cardinality_set;
    for (const vector<Index>& group : cardinality_columns)
        cardinality_set.insert(group.begin(), group.end());

    for (const MultivariateConstraint& constraint : formula_constraints)
    {
        if (constraint.kind != RepairRegime::InputAffine)
            continue;

        bool all_free_discrete = true;
        for (const pair<Index, float>& term : constraint.compiled.affine_input_terms)
            if (term.first >= ssize(fixed_mask)
                || !fixed_mask[term.first] || cardinality_set.count(term.first))
            { all_free_discrete = false; break; }

        if (all_free_discrete)
            repair_single_affine_integer(inputs, inferior_frontier, superior_frontier, constraint);
    }

    for (Index outer = 0; outer < passes; ++outer)
    {
        repair_affine_inputs_with_fixed(inputs, inferior_frontier, superior_frontier,
                                        formula_constraints, fixed_mask);

        if (input_constraints.empty())
            return;

        bool all_feasible = true;
        const bool last_pass = (outer + 1 >= passes);
        const Index swaps = 1 + outer / 2;
        const float unlock_fraction = min(1.0f, exploration_ratio * float(outer + 1));

        for (Index r = 0; r < rows; ++r)
        {
            VectorR point = inputs.row(r).transpose();

            if (row_satisfies_input_affine(point, input_constraints))
                continue;

            all_feasible = false;
            if (last_pass)
                continue;

            for (const vector<Index>& columns : cardinality_columns)
                cardinality_swap_row(point, columns, inferior_frontier, superior_frontier, fixed_mask, swaps);

            unlock_free_integers_row(point, free_lattice, unlock_fraction);

            inputs.row(r) = point.transpose();
        }

        if (all_feasible)
            break;
    }
}


void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<MultivariateConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               const SurrogateVjp& vjp,
                               const Index max_correction_passes,
                               const vector<char>& fixed_columns)
{
    vector<const MultivariateConstraint*> constraints;

    for (const MultivariateConstraint& constraint : formula_constraints)
        if (constraint.kind == RepairRegime::OutputCoupled)
            constraints.push_back(&constraint);

    if (constraints.empty())
        return;

    const Index rows_number = inputs.rows();
    const Index passes      = max(Index(1), max_correction_passes);

    for (Index r = 0; r < rows_number; ++r)
    {
        VectorR point = inputs.row(r).transpose();

        gauss_newton_repair_row(point, constraints, forward, vjp, fixed_columns,
                                inferior_frontier, superior_frontier, passes);

        inputs.row(r) = point.transpose();
    }
}


void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<MultivariateConstraint>& formula_constraints,
                               const SurrogateBatchForward& batch_forward,
                               const Index max_correction_passes,
                               const vector<char>& fixed_columns)
{
    const Index inputs_number = inputs.cols();

    VectorR step(inputs_number);
    for (Index j = 0; j < inputs_number; ++j)
        step(j) = max(1e-4f, 1e-3f * (superior_frontier(j) - inferior_frontier(j)));

    // Single-row forward for the constraint evaluation inside the Gauss-Newton loop.
    const SurrogateForward forward = [&batch_forward](const VectorR& x) -> VectorR
    {
        MatrixR single(1, x.size());
        single.row(0) = x.transpose();
        return batch_forward(single).row(0).transpose();
    };

    // Central-difference VJP: stack all 2*inputs_number perturbations of the row and evaluate them
    // in one batched forward call (rows are independent), instead of two forwards per dimension.
    const SurrogateVjp finite_difference_vjp =
        [&batch_forward, inputs_number, step](const VectorR& x, const VectorR& cotangent)
    {
        MatrixR perturbed(2 * inputs_number, inputs_number);
        for (Index k = 0; k < inputs_number; ++k)
        {
            perturbed.row(2 * k)     = x.transpose();
            perturbed.row(2 * k + 1) = x.transpose();
            perturbed(2 * k,     k) += step(k);
            perturbed(2 * k + 1, k) -= step(k);
        }

        const MatrixR perturbed_outputs = batch_forward(perturbed);

        VectorR gradient(inputs_number);
        for (Index k = 0; k < inputs_number; ++k)
        {
            const VectorR derivative = (perturbed_outputs.row(2 * k) - perturbed_outputs.row(2 * k + 1)).transpose() / (2.0f * step(k));
            gradient(k) = derivative.dot(cotangent);
        }
        return gradient;
    };

    repair_output_constraints(inputs, inferior_frontier, superior_frontier,
                              formula_constraints, forward, finite_difference_vjp,
                              max_correction_passes, fixed_columns);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
