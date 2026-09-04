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

        pair<float, float> calculate_bounds() const;

        float calculate_residual(float value, float tolerance = 0.0f, float margin_factor = 0.0f) const;
    };

    struct FeasibilitySystem
    {
        void initialize();

        void reshape_borders(const pair<VectorR, VectorR>&);

        VectorR force_into_borders(const VectorR&) const;

        VectorR evaluate(const VectorR&, VectorR&, VectorR&, const VectorR& = {}) const;

        MatrixR calculate_jacobian(const VectorR&, const VectorR&, const VectorR&, const VectorR& = {}) const;

        pair<VectorR, VectorR> solve(VectorR) const;

        const ResponseOptimization* problem = nullptr;

        vector<const Constraint*> rows;

        pair<VectorR, VectorR> borders;
    };

    explicit ResponseOptimization(NeuralNetwork* = nullptr);

    virtual ~ResponseOptimization();

    void add_objective(const string&, Objective::Sense, float value = 0.0f);
    void add_constraint(const string&, Constraint::Condition, const vector<float>& values = {});

    void set_iterations_number(Index);
    void set_points_number(Index);

    void set_feasibility_margin_factor(float);

    MatrixR perform_response_optimization();

protected:

    NeuralNetwork* neural_network = nullptr;

    vector<Objective> objectives;
    vector<Constraint> constraints;

    FeasibilitySystem system;

    virtual MatrixR single_optimization() = 0;

    virtual MatrixR multi_optimization() = 0;

    pair<VectorR, VectorR> calculate_domain();

    VectorR calculate_random_input(const pair<VectorR, VectorR>&) const;

    void assign_random_categories(VectorR&, float probability = 1.0f) const;

    MatrixR evaluate_objectives(const MatrixR&, const MatrixR&) const;

    vector<Index> calculate_pareto_front(const MatrixR&) const;

    vector<Index> clean_front(const MatrixR&, const MatrixR&) const;

    Index iterations_number = 20;
    Index points_number = 1000;

    Index requested_front_size = 100;

private:

    pair<VectorR, VectorR> get_unconstrained_domain() const;

    float constraint_tolerance = 1e-3f;

    float feasibility_margin_factor = 0.1f;

    Index feasibility_evaluations = 50;

    float diversity_factor = 0.2f;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
