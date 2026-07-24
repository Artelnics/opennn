//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   O P T I M I Z A T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "response_constraint_manager.h"
#include "response_algorithm.h"

namespace opennn
{

class NeuralNetwork;

class ResponseOptimization
{
public:

    struct Objective
    {
        string expression;

        opennn::Sense sense = opennn::Sense::Minimize;

        float value = 0.0f;
    };

    struct Constraint
    {
        string expression;

        Condition condition = Condition::EqualTo;

        vector<float> values;

        void check() const;
    };

    using Sense = opennn::Sense;

    explicit ResponseOptimization(NeuralNetwork* = nullptr);

    ~ResponseOptimization();

    void set(NeuralNetwork* = nullptr);

    NeuralNetwork* get_neural_network() const noexcept;

    void set_optimization_algorithm(unique_ptr<ResponseAlgorithm>);
    ResponseAlgorithm* get_optimization_algorithm() const noexcept;

    void add_objective(const string&, Sense, float value = 0.0f);
    void clear_objectives();
    void clear_objectives(const string&);

    const vector<Objective>& get_objectives() const noexcept;
    bool is_objective(const string&) const;
    Sense get_sense(const string&) const;
    float get_fixed_value(const string&) const;

    void add_constraint(const string&, Condition, float low = 0.0f, float up = 0.0f);
    void add_constraint(const string&, const vector<float>& allowed_values);
    void add_cardinality_constraint(const vector<string>&, Index, bool force_nonzero = true);

    void clear_constraints();
    void clear_constraints(const string&);

    const vector<Constraint>& get_constraints() const noexcept;

    MatrixR perform_response_optimization();

    Index get_evaluations_used() const noexcept;

    vector<float> get_utopian_point() const;
    pair<Index, VectorR> get_advised_point(const MatrixR&, const VectorR& importance_scale = VectorR()) const;
    pair<Index, VectorR> get_robust_point(const MatrixR&, float balance = 0.5f) const;

private:

    NeuralNetwork* neural_network = nullptr;

    vector<Objective> objectives;
    vector<Constraint> constraints;

    unique_ptr<ResponseAlgorithm> optimization_algorithm;

    Index evaluations_used = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
