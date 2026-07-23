//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_optimization.h"
#include "response_idc.h"

namespace opennn
{

namespace
{

const ResponseOptimization::Objective* find_objective(const vector<ResponseOptimization::Objective>& objectives,
                                                      const string& name)
{
    const auto it = ranges::find_if(objectives, [&](const ResponseOptimization::Objective& objective){ return objective.expression == name; });
    return it != objectives.end() ? &*it : nullptr;
}

}


void ResponseOptimization::Constraint::check() const
{
    using enum Condition;

    switch (condition)
    {
    case AllowedSet:
        throw_if(values.empty(), "ResponseOptimization::Constraint: AllowedSet '" + expression + "' needs at least one value");
        break;
    case Cardinality:
        throw_if(expression.empty() || values.empty(), "ResponseOptimization::Constraint: Cardinality '" + expression + "' needs variables and a target k");
        break;
    case Between:
        throw_if(values.size() < 2 || values[0] > values[1], "ResponseOptimization::Constraint: Between '" + expression + "' needs [low, up] with low <= up");
        break;
    case Integer:
    case None:
        break;
    default:
        throw_if(values.empty(), "ResponseOptimization::Constraint: '" + expression + "' needs a bound value");
        break;
    }
}


ResponseOptimization::ResponseOptimization(NeuralNetwork* new_neural_network)
{
    set(new_neural_network);
}


ResponseOptimization::~ResponseOptimization() = default;


void ResponseOptimization::set(NeuralNetwork* new_neural_network)
{
    neural_network = new_neural_network;
}


void ResponseOptimization::set_optimization_algorithm(unique_ptr<ResponseAlgorithm> new_algorithm)
{
    optimization_algorithm = move(new_algorithm);
}


NeuralNetwork* ResponseOptimization::get_neural_network() const noexcept
{
    return neural_network;
}


ResponseAlgorithm* ResponseOptimization::get_optimization_algorithm() const noexcept
{
    return optimization_algorithm.get();
}


const vector<ResponseOptimization::Objective>& ResponseOptimization::get_objectives() const noexcept
{
    return objectives;
}


const vector<ResponseOptimization::Constraint>& ResponseOptimization::get_constraints() const noexcept
{
    return constraints;
}


Index ResponseOptimization::get_evaluations_used() const noexcept
{
    return evaluations_used;
}


void ResponseOptimization::add_objective(const string& name, const Sense sense, const float value)
{
    const auto it = ranges::find_if(objectives, [&](const Objective& o){ return o.expression == name; });

    if (it != objectives.end())
        *it = Objective{name, sense, value};
    else
        objectives.push_back(Objective{name, sense, value});
}


void ResponseOptimization::clear_objectives()
{
    objectives.clear();
}


void ResponseOptimization::clear_objectives(const string& name)
{
    erase_if(objectives, [&](const Objective& o){ return o.expression == name; });
}


bool ResponseOptimization::is_objective(const string& name) const
{
    return find_objective(objectives, name) != nullptr;
}


ResponseOptimization::Sense ResponseOptimization::get_sense(const string& name) const
{
    const Objective* objective = find_objective(objectives, name);
    throw_if(!objective, "ResponseOptimization: '" + name + "' is not an objective.");
    return objective->sense;
}


float ResponseOptimization::get_fixed_value(const string& name) const
{
    const Objective* objective = find_objective(objectives, name);
    throw_if(!objective, "ResponseOptimization: '" + name + "' is not an objective.");
    return objective->value;
}


void ResponseOptimization::add_constraint(const string& name, const Condition condition, const float low, const float up)
{
    Constraint constraint{name, condition, {low, up}};
    constraint.check();
    constraints.push_back(move(constraint));
}


void ResponseOptimization::add_constraint(const string& name, const vector<float>& allowed_values)
{
    Constraint constraint{name, Condition::AllowedSet, allowed_values};
    constraint.check();
    constraints.push_back(move(constraint));
}


void ResponseOptimization::add_cardinality_constraint(const vector<string>& variable_names, const Index k, const bool force_nonzero)
{
    throw_if(variable_names.empty(), "ResponseOptimization: cardinality constraint needs at least one variable");

    throw_if(k < 0 || k > ssize(variable_names),
             format("ResponseOptimization: cardinality target k={} is out of range for {} variable(s)", k, variable_names.size()));

    string expression;
    for (size_t i = 0; i < variable_names.size(); ++i)
        expression += (i ? "," : "") + variable_names[i];

    constraints.push_back(Constraint{expression, Condition::Cardinality, { float(k), force_nonzero ? 1.0f : 0.0f }});
}


void ResponseOptimization::clear_constraints()
{
    constraints.clear();
}


void ResponseOptimization::clear_constraints(const string& name)
{
    erase_if(constraints, [&](const Constraint& c){ return c.expression == name; });
}


MatrixR ResponseOptimization::perform_response_optimization()
{
    if (!optimization_algorithm)
        optimization_algorithm = make_unique<IDC>();

    MatrixR result = optimization_algorithm->optimize(*this);

    evaluations_used = optimization_algorithm->get_evaluations_used();

    return result;
}


vector<float> ResponseOptimization::get_utopian_point() const
{
    throw_if(!optimization_algorithm, "ResponseOptimization: no algorithm set; call perform_response_optimization() first.");
    return optimization_algorithm->get_utopian_point(*this);
}


pair<Index, VectorR> ResponseOptimization::get_advised_point(const MatrixR& front, const VectorR& importance_scale) const
{
    throw_if(!optimization_algorithm, "ResponseOptimization: no algorithm set; call perform_response_optimization() first.");
    return optimization_algorithm->get_advised_point(*this, front, importance_scale);
}


pair<Index, VectorR> ResponseOptimization::get_robust_point(const MatrixR& front, const float balance) const
{
    throw_if(!optimization_algorithm, "ResponseOptimization: no algorithm set; call perform_response_optimization() first.");
    return optimization_algorithm->get_robust_point(*this, front, balance);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
