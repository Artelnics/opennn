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

    [[nodiscard]] float evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const;
};

[[nodiscard]] CompiledFormula compile_formula(const string& expression,
                                              const vector<NamedColumn>& inputs,
                                              const vector<NamedColumn>& outputs);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
