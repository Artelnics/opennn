//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I D C   R E S P O N S E   O P T I M I Z A T I O N   A L G O R I T H M   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"
#include "statistics.h"
#include "variable.h"
#include "response_algorithm.h"
#include "response_optimization.h"
#include "response_constraint_manager.h"
#include "network_differential.h"

#include <set>

namespace opennn
{

class IDC : public ResponseAlgorithm
{
public:

    using Sense = ResponseOptimization::Sense;
    using Constraint = ResponseOptimization::Constraint;

    enum class TimeType { PresentContinuous, PresentBatch, PastContinuous, PastBatch };

    enum class BranchMode { Budgeted, Exhaustive };

    IDC() = default;
    ~IDC() override;

    string get_name() const override;

    MatrixR optimize(const ResponseOptimization&) override;

    Index get_evaluations_used() const override;

    vector<float> get_utopian_point(const ResponseOptimization&) const override;
    pair<Index, VectorR> get_advised_point(const ResponseOptimization&, const MatrixR&, const VectorR& importance_scale = VectorR()) const override;
    pair<Index, VectorR> get_robust_point(const ResponseOptimization&, const MatrixR&, float balance = 0.5f) const override;

    void set_time_role(const string&, TimeType);
    void clear_time_roles();
    void clear_time_roles(const string&);

    void set_fixed_history(const Tensor3&);
    void clear_fixed_history();
    bool is_forecasting() const;

    void set_iterations(int);
    void set_evaluations_number(int);
    void set_zoom_factor(float);
    void set_relative_tolerance(float);
    void set_max_pareto_number(Index);
    void set_max_total_evaluations(Index);
    void set_initial_sampling_factor(Index);
    void set_min_feasible_ratio(float);
    void set_max_oversample_factor(Index);
    void set_exploration_ratio(float);
    void set_deformation_domain_factor(float);
    float get_deformation_domain_factor() const noexcept;
    void set_branch_mode(BranchMode);

private:

    struct SamplingMemory
    {
        map<string, vector<Index>> category_frequencies;
        vector<char> cardinality_preferred;
        map<string, Index> cardinality_indicator_columns;
        float last_feasibility_rate = 1.0f;
    };

    struct Domain
    {
        Domain() = default;

        Domain(const vector<Variable>& variables,
               const vector<Descriptives>& descriptives,
               const float deformation_domain_factor = 1.0f)
        {
            set(variables, descriptives, deformation_domain_factor);
        }

        void set(const vector<Variable>&, const vector<Descriptives>&, const float deformation_domain_factor = 1.0f);
        void bound(const vector<Variable>&, const vector<UnivariateConstraint>&);
        void reshape(const float, const VectorR&, const MatrixR&, const vector<Variable>&);

        VectorR inferior_frontier;
        VectorR superior_frontier;
    };

    struct ObjectiveNormalizer
    {
        ObjectiveNormalizer(const IDC&);

        MatrixR source_and_column;
        MatrixR utopian_and_sense;
        MatrixR scale_and_offset;

        vector<char> closeness_mask;
        VectorR closeness_target;
        VectorR closeness_scale;

        MatrixR extract(const MatrixR&, const MatrixR&) const;
        void normalize(MatrixR&) const;
        bool update_utopian_from_points(const MatrixR&);
    };

    enum class BranchAxisType { Variable, Formula, Disjunction };

    struct BranchAxis
    {
        BranchAxisType type = BranchAxisType::Variable;
        string variable_name;
        Index index = 0;
        vector<float> values;
    };

    static bool is_past(TimeType);
    bool is_history(const string&) const;

    vector<Descriptives> get_descriptives(const string&) const;
    const pair<vector<Variable>, vector<Descriptives>>& get_variables_and_descriptives(const string&) const;
    Tensor3 combine_input(const MatrixR&) const;
    MatrixR calculate_outputs(const MatrixR&) const;

    Domain get_original_domain(string_view role) const;

    ConstraintGeometry build_geometry() const;

    vector<char> discrete_column_mask(const vector<Variable>&) const;

    Lattice build_input_lattice(const vector<Variable>&, const Domain&, map<string, Index>&) const;

    vector<vector<Index>> resolve_cardinality_columns(const Domain&, const map<string, Index>&, float, MatrixR&) const;

    MatrixR calculate_random_inputs(const Domain&, Index evaluations_count = -1) const;

    pair<MatrixR, MatrixR> sample_feasible_points(const Domain&, const Domain&, Index evaluations_multiplier = 1) const;
    pair<MatrixR, MatrixR> generate_feasible_points(const Domain&, const Domain&, Index) const;

    void restore_cardinality_columns(Domain&, const Domain&) const;
    void initialize_network_differential() const;

    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR&, const MatrixR&, const ObjectiveNormalizer&) const;
    pair<MatrixR, MatrixR> calculate_optimal_points(const MatrixR&, const MatrixR&, const ObjectiveNormalizer&, const MatrixR&) const;
    pair<MatrixR, MatrixR> calculate_pareto(const MatrixR&, const MatrixR&, const MatrixR&) const;
    pair<float, float> calculate_quality_metrics(const MatrixR&, const MatrixR&, const ObjectiveNormalizer&) const;

    MatrixR perform_single_objective_optimization() const;
    MatrixR perform_multiobjective_optimization() const;
    MatrixR solve_once() const;

    vector<BranchAxis> collect_branch_axes() const;
    static vector<vector<float>> enumerate_branch_values(const vector<BranchAxis>&);
    MatrixR run_branch_search(const vector<BranchAxis>&, const vector<vector<float>>&);

    mutable const ResponseOptimization* problem = nullptr;
    mutable ConstraintManager handler;
    mutable SamplingMemory sampling_memory;
    mutable NetworkJacobian network_jacobian;
    mutable Index evaluations_used = 0;

    mutable map<string, pair<vector<Variable>, vector<Descriptives>>> variables_descriptives;

    map<string, TimeType> time_roles;
    Tensor3 fixed_history;

    Index evaluations_number = 2000;
    Index max_iterations = 20;
    Index min_iterations = 4;
    Index initial_sampling_factor = 1;
    Index max_pareto_number = 2000;
    Index max_total_evaluations = 0;
    Index max_oversample_factor = 8;
    float zoom_factor = 0.85f;
    float relative_tolerance = 1e-6f;
    float min_feasible_ratio = 0.01f;
    float exploration_ratio = 0.1f;
    float deformation_domain_factor = 1.0f;
    BranchMode branch_mode = BranchMode::Budgeted;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
