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

bool gauss_newton_step(const MatrixR& jacobian,
                       const VectorR& residuals,
                       const VectorR& inferior,
                       const VectorR& superior,
                       VectorR& point);

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

        float calculate_residual(const VectorR&, const VectorR&) const;

        pair<float, float> calculate_bounds() const;
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

    VectorR get_feasible_input(VectorR, const pair<VectorR, VectorR>&) const;

    MatrixR evaluate_objectives(const MatrixR&, const MatrixR&) const;

    vector<Index> calculate_pareto_front(const MatrixR&) const;

    vector<Index> clean_front(const MatrixR&, const MatrixR&) const;

    Index iterations_number = 20;
    Index points_number = 1000;

    Index requested_front_size = 100;

private:

    VectorR assign_categories(const VectorR&) const;

    static float bound_tolerance(float bound) { return max(EPSILON, abs(bound) * bound_tolerance_factor); }

    static constexpr float bound_tolerance_factor = 1e-4f;

    Index maximum_adjustment_passes = 16;

    Index density_neighbors_number = 20;

    float diversity_factor = 0.2f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
