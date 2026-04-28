//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R M U L A   E X P R E S S I O N   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef FORMULA_EXPRESSION_H
#define FORMULA_EXPRESSION_H

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
    type constant = 0;
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

    vector<pair<Index, type>> affine_input_terms;
    vector<pair<Index, type>> affine_output_terms;
    type affine_constant = 0;

    type evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const;
};

CompiledFormula compile_formula(const string& expression,
                                const vector<NamedColumn>& inputs,
                                const vector<NamedColumn>& outputs);

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
