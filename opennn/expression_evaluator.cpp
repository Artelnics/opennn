//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E X P R E S S I O N   E V A L U A T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "expression_evaluator.h"
#include "string_utilities.h"

#include <cctype>

namespace opennn
{

ExpressionEvaluator::ExpressionEvaluator(const string& expression)
    : source(expression)
{
    throw_if(expression.empty(), "ExpressionEvaluator: empty expression");

    tokenize();
}


void ExpressionEvaluator::tokenize()
{
    tokens.reserve(source.size() / 2 + 1);

    size_t position = 0;

    while (position < source.size())
    {
        const char character = source[position];

        if (isspace(static_cast<unsigned char>(character)))
        {
            ++position;
            continue;
        }

        Token token;
        token.position = position;

        if (isdigit(static_cast<unsigned char>(character))
        || (character == '.'
        && position + 1 < source.size()
        && isdigit(static_cast<unsigned char>(source[position + 1]))))
        {
            const size_t token_start = position;

            while (position < source.size()
                && (isdigit(static_cast<unsigned char>(source[position])) || source[position] == '.'))
                ++position;

            if (position < source.size() && (source[position] == 'e' || source[position] == 'E'))
            {
                ++position;
                if (position < source.size() && (source[position] == '+' || source[position] == '-'))
                    ++position;

                while (position < source.size() && isdigit(static_cast<unsigned char>(source[position])))
                    ++position;
            }

            token.kind = Token::Kind::Number;
            token.text = source.substr(token_start, position - token_start);
            token.number = parse_float(token.text, "ExpressionEvaluator: numeric literal");
            tokens.push_back(move(token));
            continue;
        }

        if (isalpha(static_cast<unsigned char>(character)) || character == '_')
        {
            const size_t token_start = position;

            while (position < source.size()
                && (isalnum(static_cast<unsigned char>(source[position])) || source[position] == '_'))
                ++position;

            token.kind = Token::Kind::Identifier;
            token.text = source.substr(token_start, position - token_start);
            tokens.push_back(move(token));
            continue;
        }

        ++position;

        switch (character)
        {
        case '(': token.kind = Token::Kind::LeftParen;  token.text = "("; break;
        case ')': token.kind = Token::Kind::RightParen; token.text = ")"; break;
        case ',': token.kind = Token::Kind::Comma;      token.text = ","; break;
        case '+': case '-': case '*': case '/': case '^':
            token.kind = Token::Kind::Operator;
            token.text = string(1, character);
            break;
        default:
            throw runtime_error(format("ExpressionEvaluator: unexpected character '{}' at position {}",
                                       character, position - 1));
        }

        tokens.push_back(move(token));
    }

    Token end_token;
    end_token.kind = Token::Kind::End;
    end_token.position = source.size();
    tokens.push_back(move(end_token));
}


float ExpressionEvaluator::evaluate(const map<string, float>& variables) const
{
    size_t cursor = 0;

    const float result = parse_expression(cursor, variables);

    throw_if(tokens[cursor].kind != Token::Kind::End,
             format("ExpressionEvaluator: trailing tokens after valid expression in '{}'", source));

    return result;
}


float ExpressionEvaluator::parse_expression(size_t& cursor, const map<string, float>& variables) const
{
    float left_value = parse_term(cursor, variables);

    while (tokens[cursor].kind == Token::Kind::Operator
        && (tokens[cursor].text == "+" || tokens[cursor].text == "-"))
    {
        const string operator_text = tokens[cursor++].text;
        const float right_value = parse_term(cursor, variables);

        left_value = (operator_text == "+") ? left_value + right_value : left_value - right_value;
    }

    return left_value;
}


float ExpressionEvaluator::parse_term(size_t& cursor, const map<string, float>& variables) const
{
    float left_value = parse_factor(cursor, variables);

    while (tokens[cursor].kind == Token::Kind::Operator
        && (tokens[cursor].text == "*" || tokens[cursor].text == "/"))
    {
        const string operator_text = tokens[cursor++].text;
        const float right_value = parse_factor(cursor, variables);

        left_value = (operator_text == "*") ? left_value * right_value : left_value / right_value;
    }

    return left_value;
}


float ExpressionEvaluator::parse_factor(size_t& cursor, const map<string, float>& variables) const
{
    const float left_value = parse_unary(cursor, variables);

    if (tokens[cursor].kind == Token::Kind::Operator && tokens[cursor].text == "^")
    {
        ++cursor;
        const float right_value = parse_factor(cursor, variables);

        return pow(left_value, right_value);
    }

    return left_value;
}


float ExpressionEvaluator::parse_unary(size_t& cursor, const map<string, float>& variables) const
{
    if (tokens[cursor].kind == Token::Kind::Operator && tokens[cursor].text == "-")
    {
        ++cursor;
        return -parse_unary(cursor, variables);
    }

    if (tokens[cursor].kind == Token::Kind::Operator && tokens[cursor].text == "+")
    {
        ++cursor;
        return parse_unary(cursor, variables);
    }

    return parse_primary(cursor, variables);
}


float ExpressionEvaluator::parse_primary(size_t& cursor, const map<string, float>& variables) const
{
    const Token token = tokens[cursor++];

    if (token.kind == Token::Kind::Number)
        return token.number;

    if (token.kind == Token::Kind::LeftParen)
    {
        const float inner_value = parse_expression(cursor, variables);

        throw_if(tokens[cursor].kind != Token::Kind::RightParen,
                 format("ExpressionEvaluator: expected ')' at position {}", tokens[cursor].position));
        ++cursor;

        return inner_value;
    }

    if (token.kind == Token::Kind::Identifier)
    {
        if (tokens[cursor].kind == Token::Kind::LeftParen)
        {
            ++cursor;

            vector<float> arguments;

            if (tokens[cursor].kind != Token::Kind::RightParen)
            {
                arguments.push_back(parse_expression(cursor, variables));

                while (tokens[cursor].kind == Token::Kind::Comma)
                {
                    ++cursor;
                    arguments.push_back(parse_expression(cursor, variables));
                }
            }

            throw_if(tokens[cursor].kind != Token::Kind::RightParen,
                     format("ExpressionEvaluator: expected ')' in call to '{}'", token.text));
            ++cursor;

            return evaluate_function(token.text, arguments);
        }

        const auto variable_iterator = variables.find(token.text);

        throw_if(variable_iterator == variables.end(),
                 format("ExpressionEvaluator: unknown identifier '{}' "
                        "(not a registered variable or supported function)",
                        token.text));

        return variable_iterator->second;
    }

    throw runtime_error(format("ExpressionEvaluator: unexpected token '{}' at position {}",
                               token.text, token.position));
}


float ExpressionEvaluator::evaluate_function(const string& function_name, const vector<float>& arguments) const
{
    static const unordered_map<string, size_t> unary_functions =
    { {"sqrt",1}, {"exp",1}, {"log",1}, {"abs",1}, {"sin",1}, {"cos",1}, {"tan",1} };
    static const unordered_map<string, size_t> binary_functions =
    { {"min",2}, {"max",2}, {"pow",2} };

    const auto unary_iterator = unary_functions.find(function_name);
    const auto binary_iterator = binary_functions.find(function_name);

    if (unary_iterator != unary_functions.end())
    {
        throw_if(arguments.size() != 1,
                 format("ExpressionEvaluator: function '{}' expects 1 argument, got {}",
                        function_name, arguments.size()));

        const float x = arguments[0];

        if (function_name == "sqrt") return sqrt(x);
        if (function_name == "exp")  return exp(x);
        if (function_name == "log")  return log(x);
        if (function_name == "abs")  return abs(x);
        if (function_name == "sin")  return sin(x);
        if (function_name == "cos")  return cos(x);
        return tan(x);
    }

    if (binary_iterator != binary_functions.end())
    {
        throw_if(arguments.size() != 2,
                 format("ExpressionEvaluator: function '{}' expects 2 arguments, got {}",
                        function_name, arguments.size()));

        if (function_name == "min") return min(arguments[0], arguments[1]);
        if (function_name == "max") return max(arguments[0], arguments[1]);
        return pow(arguments[0], arguments[1]);
    }

    throw runtime_error(format("ExpressionEvaluator: unknown function '{}'", function_name));
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
