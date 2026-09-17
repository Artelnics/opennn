// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/pch.h"

namespace opennn
{

class Network;

enum class ExpressionLinearity { Linear, Nonlinear };

struct ExpressionOp
{
    enum class Kind : unsigned char
    {
        PushConst, PushInput, PushOutput,
        Add, Sub, Mul, Div, Pow, Neg,
        Sqrt, Exp, Log, Abs, Sin, Cos, Tan, Min, Max
    };

    Kind kind = Kind::PushConst;
    int index = 0;
    float constant = 0.0f;

    bool operator==(const ExpressionOp&) const = default;
};


struct ExpressionProgram
{
    vector<ExpressionOp> operations;

    int stack_depth = 0;
};


struct CompiledExpression
{
    string text;

    ExpressionLinearity linearity = ExpressionLinearity::Nonlinear;

    vector<Index> input_indices;
    vector<Index> output_indices;

    vector<pair<Index, float>> linear_input_terms;
    vector<pair<Index, float>> linear_output_terms;

    float linear_constant = 0.0f;

    ExpressionProgram program;

    vector<pair<Index, ExpressionProgram>> input_gradient;

    Index symmetric_order = 0;

    vector<pair<Index, double>> symmetric_terms;

    double symmetric_scale = 1.0;

    float evaluate(const VectorR&, const VectorR&) const;
};


CompiledExpression compile_expression(const string&,
                                      const vector<pair<string, Index>>&,
                                      const vector<pair<string, Index>>&);

CompiledExpression compile_expression(const string&, const Network*, const string& role = "Expression");

CompiledExpression compile_elementary_symmetric(const vector<Index>& variables,
                                                const vector<float>& spans,
                                                Index order,
                                                float tolerance);

CompiledExpression compile_integrality(const string&, const Network*);

CompiledExpression compile_membership(const string&, const Network*, const vector<float>& allowed);

bool is_output_coupled(const CompiledExpression&);

bool is_univariate(const CompiledExpression&);

bool is_bare_variable(const CompiledExpression&);

void evaluate_input_gradient(const CompiledExpression&,
                             const VectorR& point,
                             const VectorR& output,
                             VectorR& gradient);

VectorR evaluate_input_gradient(const CompiledExpression&, const VectorR& point, const VectorR& output);

}
