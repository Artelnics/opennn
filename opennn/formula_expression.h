//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R M U L A   E X P R E S S I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

enum class FormulaShape { Affine, Nonlinear };
enum class FormulaScope { InputsOnly, OutputsOnly, Mixed };

struct RpnOp
{
    enum class Kind : uint8_t
    {
        PushConst, PushInput, PushOutput,
        Add, Sub, Mul, Div, Pow, Neg,
        Sqrt, Exp, Log, Abs, Sin, Cos, Tan, Min, Max
    };

    Kind kind = Kind::PushConst;
    Index index = 0;
    float constant = 0.0f;
};

struct NamedColumn
{
    string name;
    Index column_index = 0;
};

struct CompiledFormula
{
    vector<RpnOp> bytecode;

    FormulaShape shape = FormulaShape::Nonlinear;
    FormulaScope scope = FormulaScope::InputsOnly;

    vector<Index> input_indices;
    vector<Index> output_indices;

    vector<pair<Index, float>> affine_input_terms;
    vector<pair<Index, float>> affine_output_terms;
    float affine_constant = 0.0f;

    float evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const;
};

CompiledFormula compile_formula(const string& expression,
                                              const vector<NamedColumn>& inputs,
                                              const vector<NamedColumn>& outputs);


enum class ComparisonOp : uint8_t
{
    None, EqualTo, Between, GreaterEqualTo, LessEqualTo, GreaterThan, LessThan
};


struct FormulaConstraint
{
    string expression;
    function<float(const VectorR&, const VectorR&)> callback;
    bool uses_callback = false;

    ComparisonOp op = ComparisonOp::None;
    float low_bound = 0.0f;
    float up_bound = 0.0f;

    CompiledFormula compiled;
};


struct LinearConstraintSet
{
    MatrixR A;       // (m_constraints, n_inputs + n_outputs); first n_inputs cols are input coefficients
    VectorR lower;   // (m_constraints), -infinity for one-sided upper-bound constraints
    VectorR upper;   // (m_constraints), +infinity for one-sided lower-bound constraints
};


bool all_formula_constraints_are_linear(const vector<FormulaConstraint>& formula_constraints);

LinearConstraintSet build_linear_constraint_set(const vector<FormulaConstraint>& formula_constraints,
                                                const Index n_in,
                                                const Index n_out);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
