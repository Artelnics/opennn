//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E X P R E S S I O N   E V A L U A T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

enum class FormulaShape { Affine, Nonlinear };
enum class FormulaScope { InputsOnly, OutputsOnly, Mixed };

struct ExpressionOp
{
    enum class Kind
    {
        PushConst, PushInput, PushOutput,
        Add, Sub, Mul, Div, Pow, Neg,
        Sqrt, Exp, Log, Abs, Sin, Cos, Tan, Min, Max
    };

    Kind kind = Kind::PushConst;
    Index index = 0;
    float constant = 0.0f;
};

struct CompiledExpression
{
    vector<ExpressionOp> operations;

    FormulaShape shape = FormulaShape::Nonlinear;
    FormulaScope scope = FormulaScope::InputsOnly;

    vector<Index> input_indices;
    vector<Index> output_indices;

    vector<pair<Index, float>> affine_input_terms;
    vector<pair<Index, float>> affine_output_terms;

    float affine_constant = 0.0f;

    vector<pair<Index, vector<ExpressionOp>>> input_gradient;
    vector<pair<Index, vector<ExpressionOp>>> output_gradient;

    float evaluate(const VectorR&, const VectorR&) const;
};

struct ExpressionNode;
using ExpressionNodePtr = unique_ptr<ExpressionNode>;

struct ExpressionNode
{
    enum class Kind { Const, Input, Output, UnaryNeg, Add, Sub, Mul, Div, Pow, Func };

    Kind kind = Kind::Const;
    float constant = 0.0f;
    Index index = 0;
    string function_name;
    vector<ExpressionNodePtr> children;
};

float evaluate_operations(const vector<ExpressionOp>&,
                   const VectorR&,
                   const VectorR&);

CompiledExpression compile_formula(const string&,
                                const vector<pair<string, Index>>&,
                            const vector<pair<string, Index>>&);

CompiledExpression compile_ast(const ExpressionNode&);

ExpressionNodePtr parse_expression_tree(const string&,
                                        const vector<pair<string, Index>>&,
                                        const vector<pair<string, Index>>&);

ExpressionNodePtr differentiate(const ExpressionNode&, bool wrt_is_output, Index wrt_index);

ExpressionNodePtr clone(const ExpressionNode&);

ExpressionNodePtr make_neg(ExpressionNodePtr);

ExpressionNodePtr make_sub(ExpressionNodePtr, ExpressionNodePtr);


class ExpressionEvaluator
{
public:
    explicit ExpressionEvaluator(const string&);
    float evaluate(const map<string, float>& = {}) const;
private:
    string source;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
