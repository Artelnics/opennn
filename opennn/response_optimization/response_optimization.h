// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/core/pch.h"
#include "opennn/response_optimization/expression_evaluator.h"

namespace opennn
{

class Network;
class FeasibilityRepairSystem;

class ResponseOptimization
{
public:

    enum class FeasibilitySolver { LevenbergMarquardt, GaussNewton = 2 };

    void set(Network* = nullptr);

    struct Objective
    {
        enum class Sense { Minimize, Maximize, Fixed };

        CompiledExpression expression;

        Sense sense = Sense::Minimize;

        float value = 0.0f;

        float evaluate_score(const VectorR&, const VectorR&) const;
    };

    struct Constraint
    {
        enum class Condition
        {
            Equal, Between, GreaterEqual, LessEqual, Greater, Less, AllowedSet, Integer, Cardinality
        };

        CompiledExpression equation;

        Condition condition = Condition::Equal;

        vector<float> values;

        void validate() const;
        pair<float, float> get_bounds(float numeric_tolerance, float constraint_tolerance) const;
        Index get_discrete_column() const;
    };

    explicit ResponseOptimization(Network* = nullptr);

    virtual ~ResponseOptimization();

    void add_objective(const string&, Objective::Sense, float value = 0.0f);
    void add_constraint(const string&, Constraint::Condition, const vector<float>& values = {});

    pair<Index, VectorR> get_advised_point(const MatrixR& results) const;

    void set_iterations_number(Index);
    void set_points_number(Index);

    void set_sampling_budget_multiplier(Index);
    void set_maximum_consecutive_failures(Index);

    void set_feasibility_rounds(Index);
    void set_feasibility_evaluations(Index);
    void set_feasibility_margin_factor(float);
    void set_feasibility_solver(FeasibilitySolver);

    MatrixR perform_response_optimization();

protected:

    Network* network = nullptr;

    vector<Objective> objectives;
    vector<Constraint> constraints;

    pair<VectorR, VectorR> input_bounds;

    virtual MatrixR single_optimization() = 0;

    virtual MatrixR multi_optimization() = 0;

    pair<VectorR, VectorR> calculate_domain();

    pair<VectorR, VectorR> get_feasible_points(VectorR) const;
    pair<VectorR, VectorR> get_feasible_points(VectorR, FeasibilityRepairSystem&) const;

    FeasibilityRepairSystem make_feasibility_system(const pair<VectorR, VectorR>&) const;

    string get_sampling_failure() const;

    VectorR calculate_random_input(const pair<VectorR, VectorR>&) const;

    void assign_random_categories(VectorR&, float probability = 1.0f) const;
    void assign_random_categories(VectorR&, const pair<VectorR, VectorR>&, float probability = 1.0f) const;

    MatrixR evaluate_objectives(const MatrixR&, const MatrixR&) const;

    vector<Index> calculate_pareto_front(const MatrixR&) const;

    vector<Index> clean_front(const MatrixR&, const MatrixR&) const;

    Index iterations_number = 20;
    Index points_number = 1000;
    Index sampling_budget_multiplier = 0;
    Index maximum_consecutive_failures = 0;

    Index requested_front_size = 100;

private:

    pair<VectorR, VectorR> get_unconstrained_domain() const;

    string get_input_column_name(Index) const;

    float get_bound_tolerance(float) const;

    float numeric_tolerance = 1e-6f;

    float constraint_tolerance = 1e-3f;

    float feasibility_margin_factor = 0.1f;

    mutable Index invalid_objective = -1;

    mutable pair<double, string> closest_failure{numeric_limits<double>::infinity(), {}};

    Index feasibility_rounds = 1;

    Index feasibility_evaluations = 50;

    FeasibilitySolver feasibility_solver = FeasibilitySolver::LevenbergMarquardt;

    float diversity_factor = 0.2f;
};

}
