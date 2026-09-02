//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/pch.h"
#include "opennn/response_optimization/expression_evaluator.h"

namespace opennn
{

class NeuralNetwork;

class ResponseOptimization
{
public:

    void set(NeuralNetwork* = nullptr);

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

        CompiledExpression expression;

        Condition condition = Condition::Equal;

        vector<float> values;

        float calculate_measure(const VectorR&, const VectorR&) const;

        float calculate_residual(const VectorR&, const VectorR&, float margin = 0.0f) const;

        pair<float, float> calculate_bounds() const;

        bool is_enforced() const;
    };

    explicit ResponseOptimization(NeuralNetwork* = nullptr);

    virtual ~ResponseOptimization();

    void add_objective(const string&, Objective::Sense, float value = 0.0f);
    void add_constraint(const string&, Constraint::Condition, const vector<float>& values = {});

    MatrixR perform_response_optimization();

protected:

    NeuralNetwork* neural_network = nullptr;

    vector<Objective> objectives;
    vector<Constraint> constraints;

    virtual MatrixR single_optimization() = 0;

    virtual MatrixR multi_optimization() = 0;

    pair<VectorR, VectorR> calculate_domain() const;

    VectorR calculate_random_input(const pair<VectorR, VectorR>&) const;

    pair<VectorR, VectorR> get_feasible_point(VectorR, const pair<VectorR, VectorR>&) const;

    MatrixR evaluate_objectives(const MatrixR&, const MatrixR&) const;

    vector<Index> calculate_pareto_front(const MatrixR&) const;

    vector<Index> clean_front(const MatrixR&, const MatrixR&) const;

    // The only two operations that touch the one-hot blocks of a point: drawing a category at
    // random, and folding an arbitrary point back onto one. Both leave out the categories the
    // domain has closed.

    VectorR set_random_categories(VectorR, const pair<VectorR, VectorR>&, float probability = 1.0f) const;

    VectorR assign_feasible_categories(VectorR, const pair<VectorR, VectorR>&) const;

    Index iterations_number = 20;
    Index points_number = 1000;

    Index requested_front_size = 100;

private:

    VectorR round_lattice(const VectorR&) const;

    VectorR evaluate_constraints(const VectorR&, VectorR& values, VectorR& residuals) const;

    MatrixR estimate_jacobian(const VectorR&, const VectorR&, const VectorR&,
                              const pair<VectorR, VectorR>&) const;

    void expand_cardinality(const Constraint&);

    pair<VectorR, VectorR> augment_domain(const pair<VectorR, VectorR>&) const;

    VectorR augment_point(const VectorR&) const;

    Index activation_variables = 0;

    Index repair_passes = 16;

    float feasibility_margin = 0.1f;

    float diversity_factor = 0.2f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
