// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/pch.h"
#include "opennn/response_optimization/expression_evaluator.h"

namespace opennn
{

class Network;

class ResponseOptimization
{
public:

    // Rebinding clears objectives, constraints and cached bounds.
    void set(Network* = nullptr);

    struct Objective
    {
        enum class Sense { Minimize, Maximize, Fixed };

        CompiledExpression expression;

        Sense sense = Sense::Minimize;

        float value = 0.0f;
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
    };

    explicit ResponseOptimization(Network* = nullptr);

    virtual ~ResponseOptimization();

    void add_objective(const string&, Objective::Sense, float value = 0.0f);
    void add_constraint(const string&, Constraint::Condition, const vector<float>& values = {});

    void clear_objectives() { objectives.clear(); }
    Index get_objectives_number() const { return static_cast<Index>(objectives.size()); }
    bool is_objective(const string& name) const;

    struct Objectives
    {
        const ResponseOptimization& problem;
        VectorR utopian;

        explicit Objectives(const ResponseOptimization& p);
        MatrixR extract(const MatrixR& inputs, const MatrixR& outputs) const;
        float utopian_and_sense(Index row, Index col) const;
        void update_utopian_from_points(const MatrixR& values);
    };

    pair<Index, VectorR> get_advised_point(const MatrixR& results) const;

    void set_iterations_number(Index);
    void set_points_number(Index);

    // Zero inherits the optimization iteration count.
    void set_sampling_budget_multiplier(Index);
    // Zero disables the consecutive-failure limit.
    void set_maximum_consecutive_failures(Index);

    void set_feasibility_rounds(Index);
    // Eigen's per-round evaluation budget; excludes finite-difference probes.
    void set_feasibility_evaluations(Index);
    void set_feasibility_margin_factor(float);

    MatrixR perform_response_optimization();

protected:

    Network* network = nullptr;

    vector<Objective> objectives;
    vector<Constraint> constraints;

    pair<VectorR, VectorR> input_bounds;

    virtual MatrixR single_optimization() = 0;

    virtual MatrixR multi_optimization() = 0;

    pair<VectorR, VectorR> calculate_domain();

    pair<VectorR, VectorR> solve(VectorR) const;

    string get_sampling_failure() const;

    VectorR calculate_random_input(const pair<VectorR, VectorR>&) const;

    void assign_random_categories(VectorR&, float probability = 1.0f) const;

    MatrixR evaluate_objectives(const MatrixR&, const MatrixR&) const;

    vector<Index> calculate_pareto_front(const MatrixR&) const;

    vector<Index> clean_front(const MatrixR&, const MatrixR&) const;

    Index iterations_number = 20;
    Index points_number = 1000;
    Index sampling_budget_multiplier = 0;
    Index maximum_consecutive_failures = 0;

    Index requested_front_size = 100;

private:

    struct FeasibilityRepairSystem;

    pair<VectorR, VectorR> get_unconstrained_domain() const;

    string get_input_column_name(Index) const;

    float get_bound_tolerance(float) const;

    pair<float, float> get_constraint_bounds(const Constraint&) const;

    float numeric_tolerance = 1e-6f;

    float constraint_tolerance = 1e-3f;

    float feasibility_margin_factor = 0.1f;

    mutable Index invalid_objective = -1;

    // One round, not three. Rounding then re-solving is a feasibility pump, and at a matched
    // total budget the alternation buys nothing: one solve of 150 evaluations matches or beats
    // three of 50 on every case measured, and is the most precise on output-coupled equalities.
    // Three rounds cost twice the time on cardinality for an identical answer.
    Index feasibility_rounds = 1;

    // A ceiling, not a budget. The solver stops on ftol/xtol once the step stops paying, so the
    // average solve spends 8 to 16 evaluations and fewer than 0.5% ever reach 50. Raising this
    // changes nothing; lowering it below about 20 starts discarding repairable candidates.
    Index feasibility_evaluations = 50;

    float diversity_factor = 0.2f;
};

}
