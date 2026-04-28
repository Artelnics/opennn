//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R M U L A   E X P R E S S I O N   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "formula_expression.h"

#include <cctype>

namespace opennn
{

namespace
{

struct Token
{
    enum class Kind { Number, Identifier, Operator, LeftParen, RightParen, Comma, End };

    Kind kind = Kind::End;
    string text;
    type number = 0;
    size_t position = 0;
};


struct Lexer
{
    vector<Token> tokens;
    size_t cursor = 0;

    explicit Lexer(const string& source)
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
                token.number = stof(token.text);
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
                throw runtime_error("FormulaParser: unexpected character '" + string(1, character)
                                    + "' at position " + to_string(position - 1));
            }

            tokens.push_back(move(token));
        }

        Token end_token;
        end_token.kind = Token::Kind::End;
        end_token.position = source.size();
        tokens.push_back(move(end_token));
    }

    const Token& peek() const { return tokens[cursor]; }
    Token consume() { return tokens[cursor++]; }
};


struct Ast;
using AstPtr = unique_ptr<Ast>;

struct Ast
{
    enum class Kind { Const, Input, Output, UnaryNeg, Add, Sub, Mul, Div, Pow, Func };

    Kind kind = Kind::Const;
    type constant = 0;
    Index index = 0;
    string function_name;
    vector<AstPtr> children;
};


struct Parser
{
    Lexer& lexer;
    const vector<NamedColumn>& input_columns;
    const vector<NamedColumn>& output_columns;

    Parser(Lexer& new_lexer,
           const vector<NamedColumn>& new_input_columns,
           const vector<NamedColumn>& new_output_columns)
        : lexer(new_lexer),
          input_columns(new_input_columns),
          output_columns(new_output_columns)
    {
    }

    AstPtr parse_expression()
    {
        AstPtr left_node = parse_term();

        while (true)
        {
            const Token& next_token = lexer.peek();

            if (next_token.kind != Token::Kind::Operator) break;
            if (next_token.text != "+" && next_token.text != "-") break;

            const string operator_text = lexer.consume().text;

            AstPtr right_node = parse_term();

            auto combined_node = make_unique<Ast>();
            combined_node->kind = (operator_text == "+") ? Ast::Kind::Add : Ast::Kind::Sub;
            combined_node->children.reserve(2);
            combined_node->children.push_back(move(left_node));
            combined_node->children.push_back(move(right_node));
            left_node = move(combined_node);
        }

        return left_node;
    }

    AstPtr parse_term()
    {
        AstPtr left_node = parse_factor();

        while (true)
        {
            const Token& next_token = lexer.peek();

            if (next_token.kind != Token::Kind::Operator) break;
            if (next_token.text != "*" && next_token.text != "/") break;

            const string operator_text = lexer.consume().text;

            AstPtr right_node = parse_factor();

            auto combined_node = make_unique<Ast>();
            combined_node->kind = (operator_text == "*") ? Ast::Kind::Mul : Ast::Kind::Div;
            combined_node->children.reserve(2);
            combined_node->children.push_back(move(left_node));
            combined_node->children.push_back(move(right_node));
            left_node = move(combined_node);
        }

        return left_node;
    }

    AstPtr parse_factor()
    {
        AstPtr left_node = parse_unary();

        const Token& next_token = lexer.peek();

        if (next_token.kind == Token::Kind::Operator && next_token.text == "^")
        {
            lexer.consume();

            AstPtr right_node = parse_factor();

            auto combined_node = make_unique<Ast>();
            combined_node->kind = Ast::Kind::Pow;
            combined_node->children.reserve(2);
            combined_node->children.push_back(move(left_node));
            combined_node->children.push_back(move(right_node));
            return combined_node;
        }

        return left_node;
    }

    AstPtr parse_unary()
    {
        const Token& next_token = lexer.peek();

        if (next_token.kind == Token::Kind::Operator && next_token.text == "-")
        {
            lexer.consume();

            AstPtr child_node = parse_unary();

            auto negation_node = make_unique<Ast>();
            negation_node->kind = Ast::Kind::UnaryNeg;
            negation_node->children.push_back(move(child_node));
            return negation_node;
        }

        if (next_token.kind == Token::Kind::Operator && next_token.text == "+")
        {
            lexer.consume();
            return parse_unary();
        }

        return parse_primary();
    }

    AstPtr parse_primary()
    {
        Token token = lexer.consume();

        if (token.kind == Token::Kind::Number)
        {
            auto constant_node = make_unique<Ast>();
            constant_node->kind = Ast::Kind::Const;
            constant_node->constant = token.number;
            return constant_node;
        }

        if (token.kind == Token::Kind::LeftParen)
        {
            AstPtr inner_node = parse_expression();
            const Token closing_token = lexer.consume();

            if (closing_token.kind != Token::Kind::RightParen)
                throw runtime_error("FormulaParser: expected ')' at position "
                                    + to_string(closing_token.position));

            return inner_node;
        }

        if (token.kind == Token::Kind::Identifier)
        {
            if (lexer.peek().kind == Token::Kind::LeftParen)
            {
                lexer.consume();

                auto function_node = make_unique<Ast>();
                function_node->kind = Ast::Kind::Func;
                function_node->function_name = token.text;

                if (lexer.peek().kind != Token::Kind::RightParen)
                {
                    function_node->children.push_back(parse_expression());
                    while (lexer.peek().kind == Token::Kind::Comma)
                    {
                        lexer.consume();
                        function_node->children.push_back(parse_expression());
                    }
                }

                const Token closing_token = lexer.consume();
                if (closing_token.kind != Token::Kind::RightParen)
                    throw runtime_error("FormulaParser: expected ')' in call to '" + token.text + "'");

                return function_node;
            }

            for (const NamedColumn& named_column : input_columns)
                if (named_column.name == token.text)
                {
                    auto input_node = make_unique<Ast>();
                    input_node->kind = Ast::Kind::Input;
                    input_node->index = named_column.column_index;
                    return input_node;
                }

            for (const NamedColumn& named_column : output_columns)
                if (named_column.name == token.text)
                {
                    auto output_node = make_unique<Ast>();
                    output_node->kind = Ast::Kind::Output;
                    output_node->index = named_column.column_index;
                    return output_node;
                }

            throw runtime_error("FormulaParser: unknown identifier '" + token.text
                                + "' (not a registered input, output, or supported function)");
        }

        throw runtime_error("FormulaParser: unexpected token '" + token.text
                            + "' at position " + to_string(token.position));
    }
};


struct AffineForm
{
    bool is_affine = true;
    unordered_map<Index, type> input_terms;
    unordered_map<Index, type> output_terms;
    type constant = 0;

    bool is_constant() const { return input_terms.empty() && output_terms.empty(); }
};


void accumulate_into(unordered_map<Index, type>& destination,
                     const unordered_map<Index, type>& source,
                     const type scaling)
{
    for (const auto& [column, coefficient] : source)
    {
        const type contribution = scaling * coefficient;

        const auto existing = destination.find(column);

        if (existing == destination.end())
            destination.emplace(column, contribution);
        else
            existing->second += contribution;
    }
}


void scale_terms_in_place(unordered_map<Index, type>& terms, const type scaling)
{
    for (auto& [column, coefficient] : terms)
        coefficient *= scaling;
}


AffineForm analyze_affine(const Ast& node)
{
    AffineForm result;

    switch (node.kind)
    {
    case Ast::Kind::Const:
        result.constant = node.constant;
        return result;

    case Ast::Kind::Input:
        result.input_terms[node.index] = 1;
        return result;

    case Ast::Kind::Output:
        result.output_terms[node.index] = 1;
        return result;

    case Ast::Kind::UnaryNeg:
    {
        AffineForm child_form = analyze_affine(*node.children[0]);
        if (!child_form.is_affine) { result.is_affine = false; return result; }

        result.constant = -child_form.constant;
        scale_terms_in_place(child_form.input_terms, -1);
        scale_terms_in_place(child_form.output_terms, -1);
        result.input_terms = move(child_form.input_terms);
        result.output_terms = move(child_form.output_terms);
        return result;
    }

    case Ast::Kind::Add:
    case Ast::Kind::Sub:
    {
        AffineForm left_form = analyze_affine(*node.children[0]);
        AffineForm right_form = analyze_affine(*node.children[1]);
        if (!left_form.is_affine || !right_form.is_affine) { result.is_affine = false; return result; }

        const type sign = (node.kind == Ast::Kind::Add) ? type(1) : type(-1);
        result.constant = left_form.constant + sign * right_form.constant;
        result.input_terms = move(left_form.input_terms);
        result.output_terms = move(left_form.output_terms);
        accumulate_into(result.input_terms, right_form.input_terms, sign);
        accumulate_into(result.output_terms, right_form.output_terms, sign);
        return result;
    }

    case Ast::Kind::Mul:
    {
        AffineForm left_form = analyze_affine(*node.children[0]);
        AffineForm right_form = analyze_affine(*node.children[1]);
        if (!left_form.is_affine || !right_form.is_affine) { result.is_affine = false; return result; }

        if (left_form.is_constant())
        {
            result.constant = left_form.constant * right_form.constant;
            scale_terms_in_place(right_form.input_terms, left_form.constant);
            scale_terms_in_place(right_form.output_terms, left_form.constant);
            result.input_terms = move(right_form.input_terms);
            result.output_terms = move(right_form.output_terms);
            return result;
        }

        if (right_form.is_constant())
        {
            result.constant = left_form.constant * right_form.constant;
            scale_terms_in_place(left_form.input_terms, right_form.constant);
            scale_terms_in_place(left_form.output_terms, right_form.constant);
            result.input_terms = move(left_form.input_terms);
            result.output_terms = move(left_form.output_terms);
            return result;
        }

        result.is_affine = false;
        return result;
    }

    case Ast::Kind::Div:
    {
        AffineForm left_form = analyze_affine(*node.children[0]);
        AffineForm right_form = analyze_affine(*node.children[1]);
        if (!left_form.is_affine || !right_form.is_affine) { result.is_affine = false; return result; }

        if (!right_form.is_constant() || abs(right_form.constant) < EPSILON)
        {
            result.is_affine = false;
            return result;
        }

        const type inverse = type(1) / right_form.constant;
        result.constant = left_form.constant * inverse;
        scale_terms_in_place(left_form.input_terms, inverse);
        scale_terms_in_place(left_form.output_terms, inverse);
        result.input_terms = move(left_form.input_terms);
        result.output_terms = move(left_form.output_terms);
        return result;
    }

    case Ast::Kind::Pow:
    {
        AffineForm base_form = analyze_affine(*node.children[0]);
        AffineForm exponent_form = analyze_affine(*node.children[1]);
        if (!base_form.is_affine || !exponent_form.is_affine) { result.is_affine = false; return result; }

        if (base_form.is_constant() && exponent_form.is_constant())
        {
            result.constant = pow(base_form.constant, exponent_form.constant);
            return result;
        }

        if (exponent_form.is_constant() && abs(exponent_form.constant - 1) < EPSILON)
            return base_form;

        if (exponent_form.is_constant() && abs(exponent_form.constant) < EPSILON)
        {
            result.constant = 1;
            return result;
        }

        result.is_affine = false;
        return result;
    }

    case Ast::Kind::Func:
        result.is_affine = false;
        return result;
    }

    result.is_affine = false;
    return result;
}


void collect_variable_references(const Ast& node,
                                 set<Index>& input_references,
                                 set<Index>& output_references)
{
    if (node.kind == Ast::Kind::Input)  { input_references.insert(node.index);  return; }
    if (node.kind == Ast::Kind::Output) { output_references.insert(node.index); return; }

    for (const AstPtr& child : node.children)
        collect_variable_references(*child, input_references, output_references);
}


void validate_function_arities(const Ast& node)
{
    if (node.kind == Ast::Kind::Func)
    {
        const size_t arguments_count = node.children.size();
        const string& function_name = node.function_name;

        static const unordered_map<string, size_t> unary_functions =
        { {"sqrt",1}, {"exp",1}, {"log",1}, {"abs",1}, {"sin",1}, {"cos",1}, {"tan",1} };
        static const unordered_map<string, size_t> binary_functions =
        { {"min",2}, {"max",2}, {"pow",2} };

        const auto unary_iterator = unary_functions.find(function_name);
        const auto binary_iterator = binary_functions.find(function_name);

        if (unary_iterator != unary_functions.end())
        {
            if (arguments_count != unary_iterator->second)
                throw runtime_error("FormulaParser: function '" + function_name + "' expects "
                                    + to_string(unary_iterator->second)
                                    + " argument, got " + to_string(arguments_count));
        }
        else if (binary_iterator != binary_functions.end())
        {
            if (arguments_count != binary_iterator->second)
                throw runtime_error("FormulaParser: function '" + function_name + "' expects "
                                    + to_string(binary_iterator->second)
                                    + " arguments, got " + to_string(arguments_count));
        }
        else
        {
            throw runtime_error("FormulaParser: unknown function '" + function_name + "'");
        }
    }

    for (const AstPtr& child : node.children)
        validate_function_arities(*child);
}


void emit_bytecode(const Ast& node, vector<RpnOp>& bytecode)
{
    switch (node.kind)
    {
    case Ast::Kind::Const:
    {
        RpnOp operation;
        operation.kind = RpnOp::Kind::PushConst;
        operation.constant = node.constant;
        bytecode.push_back(operation);
        return;
    }

    case Ast::Kind::Input:
    {
        RpnOp operation;
        operation.kind = RpnOp::Kind::PushInput;
        operation.index = node.index;
        bytecode.push_back(operation);
        return;
    }

    case Ast::Kind::Output:
    {
        RpnOp operation;
        operation.kind = RpnOp::Kind::PushOutput;
        operation.index = node.index;
        bytecode.push_back(operation);
        return;
    }

    case Ast::Kind::UnaryNeg:
        emit_bytecode(*node.children[0], bytecode);
        bytecode.push_back({RpnOp::Kind::Neg, 0, 0});
        return;

    case Ast::Kind::Add:
    case Ast::Kind::Sub:
    case Ast::Kind::Mul:
    case Ast::Kind::Div:
    case Ast::Kind::Pow:
    {
        emit_bytecode(*node.children[0], bytecode);
        emit_bytecode(*node.children[1], bytecode);

        const RpnOp::Kind rpn_kind =
            node.kind == Ast::Kind::Add ? RpnOp::Kind::Add :
            node.kind == Ast::Kind::Sub ? RpnOp::Kind::Sub :
            node.kind == Ast::Kind::Mul ? RpnOp::Kind::Mul :
            node.kind == Ast::Kind::Div ? RpnOp::Kind::Div :
                                          RpnOp::Kind::Pow;

        bytecode.push_back({rpn_kind, 0, 0});
        return;
    }

    case Ast::Kind::Func:
    {
        for (const AstPtr& child : node.children)
            emit_bytecode(*child, bytecode);

        const string& function_name = node.function_name;

        RpnOp::Kind rpn_kind = RpnOp::Kind::Sqrt;
        if      (function_name == "sqrt") rpn_kind = RpnOp::Kind::Sqrt;
        else if (function_name == "exp")  rpn_kind = RpnOp::Kind::Exp;
        else if (function_name == "log")  rpn_kind = RpnOp::Kind::Log;
        else if (function_name == "abs")  rpn_kind = RpnOp::Kind::Abs;
        else if (function_name == "sin")  rpn_kind = RpnOp::Kind::Sin;
        else if (function_name == "cos")  rpn_kind = RpnOp::Kind::Cos;
        else if (function_name == "tan")  rpn_kind = RpnOp::Kind::Tan;
        else if (function_name == "min")  rpn_kind = RpnOp::Kind::Min;
        else if (function_name == "max")  rpn_kind = RpnOp::Kind::Max;
        else if (function_name == "pow")  rpn_kind = RpnOp::Kind::Pow;
        else throw runtime_error("FormulaParser: unknown function '" + function_name + "'");

        bytecode.push_back({rpn_kind, 0, 0});
        return;
    }
    }
}

} // namespace


type CompiledFormula::evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const
{
    if (shape == FormulaShape::Affine)
    {
        type result = affine_constant;

        for (const auto& [column, coefficient] : affine_input_terms)
            result += coefficient * inputs_row(column);

        for (const auto& [column, coefficient] : affine_output_terms)
            result += coefficient * outputs_row(column);

        return result;
    }

    vector<type> evaluation_stack;
    evaluation_stack.reserve(16);

    for (const RpnOp& operation : bytecode)
    {
        switch (operation.kind)
        {
        case RpnOp::Kind::PushConst:  evaluation_stack.push_back(operation.constant); break;
        case RpnOp::Kind::PushInput:  evaluation_stack.push_back(inputs_row(operation.index)); break;
        case RpnOp::Kind::PushOutput: evaluation_stack.push_back(outputs_row(operation.index)); break;
        case RpnOp::Kind::Neg: evaluation_stack.back() = -evaluation_stack.back(); break;

        case RpnOp::Kind::Add:
        { const type right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() += right_operand; break; }
        case RpnOp::Kind::Sub:
        { const type right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() -= right_operand; break; }
        case RpnOp::Kind::Mul:
        { const type right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() *= right_operand; break; }
        case RpnOp::Kind::Div:
        { const type right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() /= right_operand; break; }
        case RpnOp::Kind::Pow:
        { const type right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() = pow(evaluation_stack.back(), right_operand); break; }

        case RpnOp::Kind::Sqrt: evaluation_stack.back() = sqrt(evaluation_stack.back()); break;
        case RpnOp::Kind::Exp:  evaluation_stack.back() = exp(evaluation_stack.back()); break;
        case RpnOp::Kind::Log:  evaluation_stack.back() = log(evaluation_stack.back()); break;
        case RpnOp::Kind::Abs:  evaluation_stack.back() = abs(evaluation_stack.back()); break;
        case RpnOp::Kind::Sin:  evaluation_stack.back() = sin(evaluation_stack.back()); break;
        case RpnOp::Kind::Cos:  evaluation_stack.back() = cos(evaluation_stack.back()); break;
        case RpnOp::Kind::Tan:  evaluation_stack.back() = tan(evaluation_stack.back()); break;

        case RpnOp::Kind::Min:
        { const type right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() = min(evaluation_stack.back(), right_operand); break; }
        case RpnOp::Kind::Max:
        { const type right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() = max(evaluation_stack.back(), right_operand); break; }
        }
    }

    return evaluation_stack.back();
}


CompiledFormula compile_formula(const string& expression,
                                const vector<NamedColumn>& inputs,
                                const vector<NamedColumn>& outputs)
{
    if (expression.empty())
        throw runtime_error("FormulaParser: empty expression");

    Lexer lexer(expression);
    Parser parser(lexer, inputs, outputs);

    AstPtr ast = parser.parse_expression();

    if (lexer.peek().kind != Token::Kind::End)
        throw runtime_error("FormulaParser: trailing tokens after valid expression in '" + expression + "'");

    validate_function_arities(*ast);

    CompiledFormula result;

    set<Index> input_references;
    set<Index> output_references;
    collect_variable_references(*ast, input_references, output_references);

    if (input_references.empty() && output_references.empty())
        throw runtime_error("FormulaParser: expression '" + expression
                            + "' references no input or output variables");

    result.input_indices.assign(input_references.begin(), input_references.end());
    result.output_indices.assign(output_references.begin(), output_references.end());

    if (output_references.empty())       result.scope = FormulaScope::InputsOnly;
    else if (input_references.empty())   result.scope = FormulaScope::OutputsOnly;
    else                                 result.scope = FormulaScope::Mixed;

    const AffineForm affine_form = analyze_affine(*ast);

    if (affine_form.is_affine)
    {
        result.shape = FormulaShape::Affine;
        result.affine_constant = affine_form.constant;

        result.affine_input_terms.reserve(affine_form.input_terms.size());
        for (const auto& [column, coefficient] : affine_form.input_terms)
            if (abs(coefficient) > EPSILON)
                result.affine_input_terms.emplace_back(column, coefficient);

        result.affine_output_terms.reserve(affine_form.output_terms.size());
        for (const auto& [column, coefficient] : affine_form.output_terms)
            if (abs(coefficient) > EPSILON)
                result.affine_output_terms.emplace_back(column, coefficient);
    }
    else
    {
        result.shape = FormulaShape::Nonlinear;
    }

    emit_bytecode(*ast, result.bytecode);

    return result;
}

}

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
