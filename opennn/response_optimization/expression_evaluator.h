//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E X P R E S S I O N   E V A L U A T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/pch.h"

namespace opennn
{

class NeuralNetwork;

enum class ExpressionLinearity   { Linear, Nonlinear };
enum class ExpressionInvolvement { InputsOnly, OutputsOnly, Mixed };
enum class ExpressionComplexity  { Univariate, Multivariate };
enum class ExpressionSmoothness  { Smooth, NonSmooth };

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
    string text;

    vector<ExpressionOp> operations;

    ExpressionLinearity   linearity   = ExpressionLinearity::Nonlinear;
    ExpressionInvolvement involvement = ExpressionInvolvement::InputsOnly;
    ExpressionComplexity  complexity  = ExpressionComplexity::Multivariate;
    ExpressionSmoothness  smoothness  = ExpressionSmoothness::Smooth;

    vector<Index> input_indices;
    vector<Index> output_indices;

    vector<pair<Index, float>> linear_input_terms;
    vector<pair<Index, float>> linear_output_terms;

    float linear_constant = 0.0f;

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

CompiledExpression compile_expression(const string&,
                                const vector<pair<string, Index>>&,
                            const vector<pair<string, Index>>&);

CompiledExpression compile_expression(const string&, const NeuralNetwork*, const string& role = "Expression");

CompiledExpression compile_ast(const ExpressionNode&);


inline bool is_bare_variable(const CompiledExpression& expression)
{
    if (expression.linearity != ExpressionLinearity::Linear
     || expression.complexity != ExpressionComplexity::Univariate
     || abs(expression.linear_constant) > EPSILON)
        return false;

    const auto& terms = expression.linear_input_terms.empty() ? expression.linear_output_terms
                                                              : expression.linear_input_terms;

    return abs(terms.front().second - 1.0f) <= EPSILON;
}


inline bool is_output_coupled(const CompiledExpression& expression)
{
    return expression.involvement != ExpressionInvolvement::InputsOnly;
}


bool same_expression(const CompiledExpression&, const CompiledExpression&);

VectorR evaluate_input_gradient(const CompiledExpression&, const VectorR& point, const VectorR& output);

VectorR evaluate_output_cotangent(const CompiledExpression&, const VectorR& point, const VectorR& output);

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
