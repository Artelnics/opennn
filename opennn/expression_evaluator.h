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

class ExpressionEvaluator
{
public:

    explicit ExpressionEvaluator(const string&);

    float evaluate(const map<string, float>& = {}) const;

private:

    struct Token
    {
        enum class Kind { Number, Identifier, Operator, LeftParen, RightParen, Comma, End };

        Kind kind = Kind::End;
        string text;
        float number = 0.0f;
        size_t position = 0;
    };

    string source;
    vector<Token> tokens;

    void tokenize();

    float parse_expression(size_t&, const map<string, float>&) const;
    float parse_term(size_t&, const map<string, float>&) const;
    float parse_factor(size_t&, const map<string, float>&) const;
    float parse_unary(size_t&, const map<string, float>&) const;
    float parse_primary(size_t&, const map<string, float>&) const;

    float evaluate_function(const string&, const vector<float>&) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
