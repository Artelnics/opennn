// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/pch.h"
#include "opennn/response_optimization/expression_evaluator.h"
#include "opennn/network/network.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/variable.h"

#include <cctype>

namespace opennn
{

namespace
{

struct ExpressionNode;
using ExpressionNodePtr = unique_ptr<ExpressionNode>;

// Parsing, differentiating, analyzing and even destroying an expression tree all
// recurse once per level, so an expression that nests or chains far enough
// overflows the stack before any of them can report a problem. Both limits sit
// well below the depth that actually crashes (about a thousand levels of nesting,
// two thousand chained operations) and far above any expression a model needs.

constexpr Index max_nesting_depth = 256;

constexpr Index max_operation_depth = 512;


// A rejected expression is quoted back to the user, and the ones these limits
// reject can be tens of thousands of characters long. Cut on a character
// boundary, not in the middle of a multi-byte name.

string abbreviate(const string& expression)
{
    constexpr size_t limit = 80;

    if (expression.size() <= limit) return expression;

    size_t cut = limit;

    while (cut > 0 && (static_cast<unsigned char>(expression[cut]) & 0xC0) == 0x80)
        --cut;

    return expression.substr(0, cut) + "...";
}


struct ExpressionNode
{
    enum class Kind { Const, Input, Output, UnaryNeg, Add, Sub, Mul, Div, Pow, Func };

    Kind kind = Kind::Const;
    float constant = 0.0f;
    Index index = 0;

    ExpressionOp::Kind function = ExpressionOp::Kind::Sqrt;

    vector<ExpressionNodePtr> children;

    // Levels below this node, kept as the tree is built so no walk is needed.
    Index depth = 1;
};


ExpressionNodePtr make_const(const float value)
{
    auto node = make_unique<ExpressionNode>();
    node->kind = ExpressionNode::Kind::Const;
    node->constant = value;

    return node;
}


optional<float> as_constant(const ExpressionNode& node)
{
    if (node.kind == ExpressionNode::Kind::Const)
        return node.constant;

    return nullopt;
}


ExpressionNodePtr make_variable(const ExpressionNode::Kind kind, const Index index)
{
    auto node = make_unique<ExpressionNode>();
    node->kind = kind;
    node->index = index;

    return node;
}


ExpressionNodePtr make_input(const Index index)
{
    return make_variable(ExpressionNode::Kind::Input, index);
}


ExpressionNodePtr make_output(const Index index)
{
    return make_variable(ExpressionNode::Kind::Output, index);
}


ExpressionNodePtr make_binary(const ExpressionNode::Kind kind, ExpressionNodePtr left, ExpressionNodePtr right)
{
    auto node = make_unique<ExpressionNode>();
    node->kind = kind;
    node->depth = 1 + max(left->depth, right->depth);
    node->children.reserve(2);
    node->children.push_back(move(left));
    node->children.push_back(move(right));

    return node;
}


ExpressionNodePtr make_call(const ExpressionOp::Kind function, ExpressionNodePtr argument)
{
    auto node = make_unique<ExpressionNode>();
    node->kind = ExpressionNode::Kind::Func;
    node->function = function;
    node->depth = 1 + argument->depth;
    node->children.push_back(move(argument));

    return node;
}


ExpressionNodePtr make_neg(ExpressionNodePtr operand)
{
    if (const optional<float> constant = as_constant(*operand))
        return make_const(-*constant);

    auto node = make_unique<ExpressionNode>();
    node->kind = ExpressionNode::Kind::UnaryNeg;
    node->depth = 1 + operand->depth;
    node->children.push_back(move(operand));

    return node;
}


ExpressionNodePtr make_add(ExpressionNodePtr left, ExpressionNodePtr right)
{
    const optional<float> left_constant = as_constant(*left);
    const optional<float> right_constant = as_constant(*right);

    if (left_constant && right_constant) return make_const(*left_constant + *right_constant);
    if (left_constant == 0.0f) return right;
    if (right_constant == 0.0f) return left;

    return make_binary(ExpressionNode::Kind::Add, move(left), move(right));
}


ExpressionNodePtr make_sub(ExpressionNodePtr left, ExpressionNodePtr right)
{
    const optional<float> left_constant = as_constant(*left);
    const optional<float> right_constant = as_constant(*right);

    if (left_constant && right_constant) return make_const(*left_constant - *right_constant);
    if (right_constant == 0.0f) return left;
    if (left_constant == 0.0f) return make_neg(move(right));

    return make_binary(ExpressionNode::Kind::Sub, move(left), move(right));
}


ExpressionNodePtr make_mul(ExpressionNodePtr left, ExpressionNodePtr right)
{
    const optional<float> left_constant = as_constant(*left);
    const optional<float> right_constant = as_constant(*right);

    if (left_constant == 0.0f || right_constant == 0.0f) return make_const(0.0f);
    if (left_constant && right_constant) return make_const(*left_constant * *right_constant);
    if (left_constant == 1.0f) return right;
    if (right_constant == 1.0f) return left;

    return make_binary(ExpressionNode::Kind::Mul, move(left), move(right));
}


ExpressionNodePtr make_div(ExpressionNodePtr left, ExpressionNodePtr right)
{
    const optional<float> left_constant = as_constant(*left);
    const optional<float> right_constant = as_constant(*right);

    if (left_constant == 0.0f) return make_const(0.0f);
    if (right_constant == 1.0f) return left;
    if (left_constant && right_constant && *right_constant != 0.0f)
        return make_const(*left_constant / *right_constant);

    return make_binary(ExpressionNode::Kind::Div, move(left), move(right));
}


ExpressionNodePtr make_pow(ExpressionNodePtr base, ExpressionNodePtr exponent)
{
    const optional<float> base_constant = as_constant(*base);
    const optional<float> exponent_constant = as_constant(*exponent);

    if (exponent_constant == 0.0f) return make_const(1.0f);
    if (exponent_constant == 1.0f) return base;
    if (base_constant && exponent_constant) return make_const(pow(*base_constant, *exponent_constant));

    return make_binary(ExpressionNode::Kind::Pow, move(base), move(exponent));
}


ExpressionNodePtr clone(const ExpressionNode& node)
{
    auto copy = make_unique<ExpressionNode>();
    copy->kind = node.kind;
    copy->constant = node.constant;
    copy->index = node.index;
    copy->function = node.function;
    copy->depth = node.depth;

    copy->children.reserve(node.children.size());

    for (const ExpressionNodePtr& child : node.children)
        copy->children.push_back(clone(*child));

    return copy;
}


struct FunctionEntry
{
    string_view name;
    size_t arity;
    ExpressionOp::Kind operation;
};


constexpr FunctionEntry FUNCTIONS[] =
{
    { "sqrt", 1, ExpressionOp::Kind::Sqrt },
    { "exp",  1, ExpressionOp::Kind::Exp  },
    { "log",  1, ExpressionOp::Kind::Log  },
    { "abs",  1, ExpressionOp::Kind::Abs  },
    { "sin",  1, ExpressionOp::Kind::Sin  },
    { "cos",  1, ExpressionOp::Kind::Cos  },
    { "tan",  1, ExpressionOp::Kind::Tan  },
    { "min",  2, ExpressionOp::Kind::Min  },
    { "max",  2, ExpressionOp::Kind::Max  },
    { "pow",  2, ExpressionOp::Kind::Pow  }
};


const FunctionEntry* find_function(const string& name)
{
    for (const FunctionEntry& entry : FUNCTIONS)
        if (entry.name == name) return &entry;

    return nullptr;
}


struct Token
{
    enum class Kind { Number, Identifier, QuotedIdentifier, Operator, LeftParen, RightParen, Comma, Semicolon, End };

    Kind kind = Kind::End;
    string text;
    float number = 0.0f;
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

            if (character == '`')
            {
                token.kind = Token::Kind::QuotedIdentifier;
                ++position;
                bool closed = false;
                while (position < source.size())
                {
                    const char current = source[position++];
                    if (current != '`')
                        token.text += current;
                    else if (position < source.size() && source[position] == '`')
                    {
                        token.text += '`';
                        ++position;
                    }
                    else
                    {
                        closed = true;
                        break;
                    }
                }
                throw_if(!closed, format("ExpressionParser: unclosed quoted variable at position {}", token.position));
                throw_if(token.text.empty(), "ExpressionParser: quoted variable names cannot be empty.");
                tokens.push_back(move(token));
                continue;
            }

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
                token.number = parse_float(token.text, "ExpressionParser: numeric literal");
                tokens.push_back(move(token));
                continue;
            }

            if (isalpha(static_cast<unsigned char>(character)) || character == '_')
            {
                const size_t token_start = position;

                while (position < source.size()
                    && (isalnum(static_cast<unsigned char>(source[position]))
                     || source[position] == '_'
                     || source[position] == '.'))
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
            case ';': token.kind = Token::Kind::Semicolon;  token.text = ";"; break;
            case '+': case '-': case '*': case '/': case '^':
                token.kind = Token::Kind::Operator;
                token.text = string(1, character);
                break;
            default:
                throw runtime_error(format("ExpressionParser: unexpected character '{}' at position {}",
                                           character, position - 1));
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


struct Parser
{
    Lexer& lexer;
    const vector<pair<string, Index>>& input_columns;
    const vector<pair<string, Index>>& output_columns;

    Index nesting = 0;

    // Counts one level per nested parenthesis, call or right-hand power, and
    // refuses the expression before the recursion can exhaust the stack.
    struct NestingGuard
    {
        explicit NestingGuard(Index& counter) : level(counter)
        {
            throw_if(level >= max_nesting_depth,
                     format("ExpressionParser: the expression nests more than {} levels deep",
                            max_nesting_depth));

            ++level;
        }

        ~NestingGuard() { --level; }

        Index& level;
    };

    Parser(Lexer& new_lexer,
           const vector<pair<string, Index>>& new_input_columns,
           const vector<pair<string, Index>>& new_output_columns)
        : lexer(new_lexer),
          input_columns(new_input_columns),
          output_columns(new_output_columns)
    {
    }

    ExpressionNodePtr parse_binary(ExpressionNodePtr (Parser::*parse_operand)(),
                                   string_view operators,
                                   ExpressionNode::Kind first_kind,
                                   ExpressionNode::Kind second_kind)
    {
        ExpressionNodePtr left_node = (this->*parse_operand)();

        while (true)
        {
            const Token& next_token = lexer.peek();

            if (next_token.kind != Token::Kind::Operator) break;
            const size_t operation = operators.find(next_token.text);
            if (operation == string_view::npos) break;

            lexer.consume();
            ExpressionNodePtr right_node = (this->*parse_operand)();

            left_node = make_binary((operation == 0) ? first_kind : second_kind,
                                    move(left_node),
                                    move(right_node));
        }

        return left_node;
    }

    ExpressionNodePtr parse_expression()
    {
        const NestingGuard guard(nesting);

        return parse_binary(&Parser::parse_term, "+-",
                            ExpressionNode::Kind::Add, ExpressionNode::Kind::Sub);
    }

    ExpressionNodePtr parse_term()
    {
        return parse_binary(&Parser::parse_factor, "*/",
                            ExpressionNode::Kind::Mul, ExpressionNode::Kind::Div);
    }

    ExpressionNodePtr parse_factor()
    {
        ExpressionNodePtr left_node = parse_unary();

        const Token& next_token = lexer.peek();

        if (next_token.kind == Token::Kind::Operator && next_token.text == "^")
        {
            lexer.consume();

            const NestingGuard guard(nesting);

            return make_binary(ExpressionNode::Kind::Pow, move(left_node), parse_factor());
        }

        return left_node;
    }

    ExpressionNodePtr parse_unary()
    {
        const Token& next_token = lexer.peek();

        if (next_token.kind == Token::Kind::Operator && next_token.text == "-")
        {
            lexer.consume();

            const NestingGuard guard(nesting);

            return make_neg(parse_unary());
        }

        if (next_token.kind == Token::Kind::Operator && next_token.text == "+")
        {
            lexer.consume();

            const NestingGuard guard(nesting);

            return parse_unary();
        }

        return parse_primary();
    }

    ExpressionNodePtr parse_primary()
    {
        Token token = lexer.consume();

        if (token.kind == Token::Kind::Number)
            return make_const(token.number);

        if (token.kind == Token::Kind::LeftParen)
        {
            ExpressionNodePtr inner_node = parse_expression();

            const Token closing_token = lexer.consume();

            throw_if(closing_token.kind != Token::Kind::RightParen,
                     format("ExpressionParser: expected ')' at position {}", closing_token.position));

            return inner_node;
        }

        if (token.kind == Token::Kind::Identifier || token.kind == Token::Kind::QuotedIdentifier)
        {
            if (token.kind == Token::Kind::Identifier && lexer.peek().kind == Token::Kind::LeftParen)
                return parse_call(token.text);

            for (const auto& named_column : input_columns)
                if (named_column.first == token.text)
                    return make_input(named_column.second);

            for (const auto& named_column : output_columns)
                if (named_column.first == token.text)
                    return make_output(named_column.second);

            throw runtime_error(format("ExpressionParser: unknown identifier '{}' "
                                       "(not a registered input, output, or supported function)",
                                       token.text));
        }

        throw runtime_error(format("ExpressionParser: unexpected token '{}' at position {}",
                                   token.text, token.position));
    }

    ExpressionNodePtr parse_call(const string& name)
    {
        lexer.consume();

        vector<ExpressionNodePtr> arguments;

        if (lexer.peek().kind != Token::Kind::RightParen)
        {
            arguments.push_back(parse_expression());

            while (lexer.peek().kind == Token::Kind::Comma)
            {
                lexer.consume();
                arguments.push_back(parse_expression());
            }
        }

        const Token closing_token = lexer.consume();

        throw_if(closing_token.kind != Token::Kind::RightParen,
                 format("ExpressionParser: expected ')' in call to '{}'", name));

        const FunctionEntry* entry = find_function(name);

        throw_if(!entry, format("ExpressionParser: unknown function '{}'", name));

        throw_if(arguments.size() != entry->arity,
                 format("ExpressionParser: function '{}' expects {} argument{}, got {}",
                        name, entry->arity, entry->arity == 1 ? "" : "s", arguments.size()));

        if (entry->operation == ExpressionOp::Kind::Pow)
            return make_pow(move(arguments[0]), move(arguments[1]));

        auto call = make_unique<ExpressionNode>();
        call->kind = ExpressionNode::Kind::Func;
        call->function = entry->operation;
        call->children = move(arguments);

        return call;
    }
};


struct LinearForm
{
    bool is_linear = true;
    unordered_map<Index, float> input_terms;
    unordered_map<Index, float> output_terms;
    float constant = 0.0f;

    bool is_constant() const { return input_terms.empty() && output_terms.empty(); }
};


void accumulate_into(unordered_map<Index, float>& destination,
                     const unordered_map<Index, float>& source,
                     const float scaling)
{
    for (const auto& [column, coefficient] : source)
    {
        const float contribution = scaling * coefficient;

        const auto existing = destination.find(column);

        if (existing == destination.end())
            destination.emplace(column, contribution);
        else
            existing->second += contribution;
    }
}


void scale_terms_in_place(unordered_map<Index, float>& terms, const float scaling)
{
    for (auto& [column, coefficient] : terms)
        coefficient *= scaling;
}

static LinearForm scaled_linear_form(LinearForm form, float scaling)
{
    form.constant *= scaling;
    scale_terms_in_place(form.input_terms, scaling);
    scale_terms_in_place(form.output_terms, scaling);
    return form;
}


LinearForm analyze_linear(const ExpressionNode& node)
{
    LinearForm result;

    switch (node.kind)
    {
        using enum ExpressionNode::Kind;
    case Const:
        result.constant = node.constant;
        return result;

    case Input:
        result.input_terms[node.index] = 1.0f;
        return result;

    case Output:
        result.output_terms[node.index] = 1.0f;
        return result;

    case UnaryNeg:
    {
        LinearForm child_form = analyze_linear(*node.children[0]);
        if (!child_form.is_linear) { result.is_linear = false; return result; }
        return scaled_linear_form(move(child_form), -1.0f);
    }

    case Add:
    case Sub:
    {
        LinearForm left_form = analyze_linear(*node.children[0]);
        LinearForm right_form = analyze_linear(*node.children[1]);
        if (!left_form.is_linear || !right_form.is_linear) { result.is_linear = false; return result; }

        const float sign = (node.kind == ExpressionNode::Kind::Add) ? 1.0f : -1.0f;
        result.constant = left_form.constant + sign * right_form.constant;
        result.input_terms = move(left_form.input_terms);
        result.output_terms = move(left_form.output_terms);
        accumulate_into(result.input_terms, right_form.input_terms, sign);
        accumulate_into(result.output_terms, right_form.output_terms, sign);
        return result;
    }

    case Mul:
    {
        LinearForm left_form = analyze_linear(*node.children[0]);
        LinearForm right_form = analyze_linear(*node.children[1]);
        if (!left_form.is_linear || !right_form.is_linear) { result.is_linear = false; return result; }

        if (left_form.is_constant())
            return scaled_linear_form(move(right_form), left_form.constant);

        if (right_form.is_constant())
            return scaled_linear_form(move(left_form), right_form.constant);

        result.is_linear = false;
        return result;
    }

    case Div:
    {
        LinearForm left_form = analyze_linear(*node.children[0]);
        LinearForm right_form = analyze_linear(*node.children[1]);
        if (!left_form.is_linear || !right_form.is_linear) { result.is_linear = false; return result; }

        if (!right_form.is_constant() || abs(right_form.constant) < EPSILON)
        {
            result.is_linear = false;
            return result;
        }

        return scaled_linear_form(move(left_form), 1.0f / right_form.constant);
    }

    case Pow:
    {
        LinearForm base_form = analyze_linear(*node.children[0]);
        LinearForm exponent_form = analyze_linear(*node.children[1]);
        if (!base_form.is_linear || !exponent_form.is_linear) { result.is_linear = false; return result; }

        if (base_form.is_constant() && exponent_form.is_constant())
        {
            result.constant = pow(base_form.constant, exponent_form.constant);
            return result;
        }

        if (exponent_form.is_constant() && abs(exponent_form.constant - 1.0f) < EPSILON)
            return base_form;

        if (exponent_form.is_constant() && abs(exponent_form.constant) < EPSILON)
        {
            result.constant = 1.0f;
            return result;
        }

        result.is_linear = false;
        return result;
    }

    case Func:
        result.is_linear = false;
        return result;
    }

    result.is_linear = false;
    return result;
}


void collect_variable_references(const ExpressionNode& node,
                                 set<Index>& input_references,
                                 set<Index>& output_references)
{
    if (node.kind == ExpressionNode::Kind::Input)  { input_references.insert(node.index);  return; }
    if (node.kind == ExpressionNode::Kind::Output) { output_references.insert(node.index); return; }

    for (const ExpressionNodePtr& child : node.children)
        collect_variable_references(*child, input_references, output_references);
}


ExpressionNodePtr differentiate_call(const ExpressionNode& node, const bool wrt_is_output, const Index wrt_index);


ExpressionNodePtr differentiate(const ExpressionNode& node, const bool wrt_is_output, const Index wrt_index)
{
    switch (node.kind)
    {
        using enum ExpressionNode::Kind;
    case Const:
        return make_const(0.0f);

    case Input:
        return make_const((!wrt_is_output && node.index == wrt_index) ? 1.0f : 0.0f);

    case Output:
        return make_const((wrt_is_output && node.index == wrt_index) ? 1.0f : 0.0f);

    case UnaryNeg:
        return make_neg(differentiate(*node.children[0], wrt_is_output, wrt_index));

    case Add:
        return make_add(differentiate(*node.children[0], wrt_is_output, wrt_index),
                        differentiate(*node.children[1], wrt_is_output, wrt_index));

    case Sub:
        return make_sub(differentiate(*node.children[0], wrt_is_output, wrt_index),
                        differentiate(*node.children[1], wrt_is_output, wrt_index));

    case Mul:
    {
        ExpressionNodePtr left_derivative = differentiate(*node.children[0], wrt_is_output, wrt_index);
        ExpressionNodePtr right_derivative = differentiate(*node.children[1], wrt_is_output, wrt_index);

        return make_add(make_mul(move(left_derivative), clone(*node.children[1])),
                        make_mul(clone(*node.children[0]), move(right_derivative)));
    }

    case Div:
    {
        ExpressionNodePtr left_derivative = differentiate(*node.children[0], wrt_is_output, wrt_index);
        ExpressionNodePtr right_derivative = differentiate(*node.children[1], wrt_is_output, wrt_index);

        ExpressionNodePtr numerator = make_sub(make_mul(move(left_derivative), clone(*node.children[1])),
                                               make_mul(clone(*node.children[0]), move(right_derivative)));

        ExpressionNodePtr denominator = make_mul(clone(*node.children[1]), clone(*node.children[1]));

        return make_div(move(numerator), move(denominator));
    }

    case Pow:
    {
        const ExpressionNode& base = *node.children[0];
        const ExpressionNode& exponent = *node.children[1];

        if (const optional<float> constant_exponent = as_constant(exponent))
        {
            ExpressionNodePtr base_derivative = differentiate(base, wrt_is_output, wrt_index);

            ExpressionNodePtr power = make_pow(clone(base), make_const(*constant_exponent - 1.0f));

            return make_mul(make_mul(make_const(*constant_exponent), move(power)), move(base_derivative));
        }

        if (const optional<float> constant_base = as_constant(base))
        {
            ExpressionNodePtr exponent_derivative = differentiate(exponent, wrt_is_output, wrt_index);

            ExpressionNodePtr value = make_pow(make_const(*constant_base), clone(exponent));

            return make_mul(make_mul(move(value), make_const(log(*constant_base))), move(exponent_derivative));
        }

        ExpressionNodePtr base_derivative = differentiate(base, wrt_is_output, wrt_index);
        ExpressionNodePtr exponent_derivative = differentiate(exponent, wrt_is_output, wrt_index);

        ExpressionNodePtr from_exponent = make_mul(move(exponent_derivative),
                                                   make_call(ExpressionOp::Kind::Log, clone(base)));

        ExpressionNodePtr from_base = make_div(make_mul(clone(exponent), move(base_derivative)), clone(base));

        return make_mul(make_pow(clone(base), clone(exponent)),
                        make_add(move(from_exponent), move(from_base)));
    }

    case Func:
        return differentiate_call(node, wrt_is_output, wrt_index);
    }

    return make_const(0.0f);
}


ExpressionNodePtr differentiate_call(const ExpressionNode& node, const bool wrt_is_output, const Index wrt_index)
{
    const ExpressionNode& argument = *node.children[0];

    ExpressionNodePtr argument_derivative = differentiate(argument, wrt_is_output, wrt_index);

    switch (node.function)
    {
        using enum ExpressionOp::Kind;

    case Min:
    case Max:
    {
        const ExpressionNode& second = *node.children[1];

        ExpressionNodePtr second_derivative = differentiate(second, wrt_is_output, wrt_index);

        ExpressionNodePtr gap = make_sub(clone(argument), clone(second));
        ExpressionNodePtr magnitude = make_call(Abs, clone(*gap));
        ExpressionNodePtr side = make_div(move(gap), move(magnitude));

        ExpressionNodePtr average = make_mul(make_const(0.5f),
                                             make_add(clone(*argument_derivative), clone(*second_derivative)));

        ExpressionNodePtr spread = make_mul(make_const(0.5f),
                                            make_mul(move(side),
                                                     make_sub(move(argument_derivative),
                                                              move(second_derivative))));

        return node.function == Min ? make_sub(move(average), move(spread))
                                    : make_add(move(average), move(spread));
    }

    case Sqrt:
        return make_div(move(argument_derivative),
                        make_mul(make_const(2.0f), make_call(Sqrt, clone(argument))));

    case Exp:
        return make_mul(make_call(Exp, clone(argument)), move(argument_derivative));

    case Log:
        return make_div(move(argument_derivative), clone(argument));

    case Abs:
        return make_mul(make_div(clone(argument), make_call(Abs, clone(argument))),
                        move(argument_derivative));

    case Sin:
        return make_mul(make_call(Cos, clone(argument)), move(argument_derivative));

    case Cos:
        return make_neg(make_mul(make_call(Sin, clone(argument)), move(argument_derivative)));

    case Tan:
        return make_div(move(argument_derivative),
                        make_pow(make_call(Cos, clone(argument)), make_const(2.0f)));

    case PushConst: case PushInput: case PushOutput:
    case Add: case Sub: case Mul: case Div: case Pow: case Neg:
        break;
    }

    throw runtime_error("ExpressionParser: no derivative rule for a supported function");
}


ExpressionOp::Kind binary_operation(const ExpressionNode::Kind kind)
{
    switch (kind)
    {
        using enum ExpressionNode::Kind;
    case Add: return ExpressionOp::Kind::Add;
    case Sub: return ExpressionOp::Kind::Sub;
    case Mul: return ExpressionOp::Kind::Mul;
    case Div: return ExpressionOp::Kind::Div;
    case Pow: return ExpressionOp::Kind::Pow;

    case Const: case Input: case Output: case UnaryNeg: case Func:
        break;
    }

    throw runtime_error("ExpressionParser: not a binary operation");
}


void emit_operations(const ExpressionNode& node, vector<ExpressionOp>& operations)
{
    switch (node.kind)
    {
        using enum ExpressionNode::Kind;
    case Const:
        operations.push_back({ExpressionOp::Kind::PushConst, 0, node.constant});
        return;

    case Input:
        operations.push_back({ExpressionOp::Kind::PushInput, int(node.index), 0.0f});
        return;

    case Output:
        operations.push_back({ExpressionOp::Kind::PushOutput, int(node.index), 0.0f});
        return;

    case UnaryNeg:
        emit_operations(*node.children[0], operations);
        operations.push_back({ExpressionOp::Kind::Neg, 0, 0.0f});
        return;

    case Add:
    case Sub:
    case Mul:
    case Div:
    case Pow:
        emit_operations(*node.children[0], operations);
        emit_operations(*node.children[1], operations);
        operations.push_back({binary_operation(node.kind), 0, 0.0f});
        return;

    case Func:
        for (const ExpressionNodePtr& child : node.children)
            emit_operations(*child, operations);

        operations.push_back({node.function, 0, 0.0f});
        return;
    }
}


int stack_effect(const ExpressionOp::Kind kind)
{
    switch (kind)
    {
        using enum ExpressionOp::Kind;
    case PushConst:
    case PushInput:
    case PushOutput:
        return 1;

    case Add:
    case Sub:
    case Mul:
    case Div:
    case Pow:
    case Min:
    case Max:
        return -1;

    case Neg:
    case Sqrt:
    case Exp:
    case Log:
    case Abs:
    case Sin:
    case Cos:
    case Tan:
        return 0;
    }

    return 0;
}


ExpressionProgram build_program(const ExpressionNode& node)
{
    ExpressionProgram program;

    emit_operations(node, program.operations);

    int depth = 0;

    for (const ExpressionOp& operation : program.operations)
    {
        depth += stack_effect(operation.kind);

        program.stack_depth = max(program.stack_depth, depth);
    }

    return program;
}


float evaluate_program(const ExpressionProgram& program,
                       const VectorR& inputs_row,
                       const VectorR& outputs_row)
{
    if (program.operations.empty()) return 0.0f;

    thread_local vector<float> buffer;

    if (int(buffer.size()) < program.stack_depth)
        buffer.resize(size_t(program.stack_depth));

    float* const stack = buffer.data();

    int top = -1;

    for (const ExpressionOp& operation : program.operations)
    {
        switch (operation.kind)
        {
            using enum ExpressionOp::Kind;
        case PushConst:  stack[++top] = operation.constant; break;
        case PushInput:  stack[++top] = inputs_row(operation.index); break;
        case PushOutput: stack[++top] = outputs_row(operation.index); break;

        case Neg:  stack[top] = -stack[top]; break;
        case Sqrt: stack[top] = sqrt(stack[top]); break;
        case Exp:  stack[top] = exp(stack[top]); break;
        case Log:  stack[top] = log(stack[top]); break;
        case Abs:  stack[top] = abs(stack[top]); break;
        case Sin:  stack[top] = sin(stack[top]); break;
        case Cos:  stack[top] = cos(stack[top]); break;
        case Tan:  stack[top] = tan(stack[top]); break;

        case Add: --top; stack[top] += stack[top + 1]; break;
        case Sub: --top; stack[top] -= stack[top + 1]; break;
        case Mul: --top; stack[top] *= stack[top + 1]; break;
        case Div: --top; stack[top] /= stack[top + 1]; break;
        case Pow: --top; stack[top] = pow(stack[top], stack[top + 1]); break;
        case Min: --top; stack[top] = min(stack[top], stack[top + 1]); break;
        case Max: --top; stack[top] = max(stack[top], stack[top + 1]); break;
        }
    }

    return stack[top];
}


void collect_significant_terms(const unordered_map<Index, float>& terms,
                               vector<pair<Index, float>>& kept_terms,
                               vector<Index>& columns)
{
    kept_terms.clear();
    kept_terms.reserve(terms.size());

    for (const auto& [column, coefficient] : terms)
        if (abs(coefficient) > EPSILON)
            kept_terms.emplace_back(column, coefficient);

    columns.clear();
    columns.reserve(kept_terms.size());

    for (const auto& [column, coefficient] : kept_terms)
        columns.push_back(column);
}


CompiledExpression compile_ast(const ExpressionNode& ast)
{
    CompiledExpression result;

    set<Index> input_references;
    set<Index> output_references;
    collect_variable_references(ast, input_references, output_references);

    throw_if(input_references.empty() && output_references.empty(),
             "ExpressionParser: expression references no input or output variables");

    result.input_indices.assign(input_references.begin(), input_references.end());
    result.output_indices.assign(output_references.begin(), output_references.end());

    const LinearForm linear_form = analyze_linear(ast);

    if (linear_form.is_linear)
    {
        result.linearity = ExpressionLinearity::Linear;
        result.linear_constant = linear_form.constant;

        collect_significant_terms(linear_form.input_terms, result.linear_input_terms, result.input_indices);
        collect_significant_terms(linear_form.output_terms, result.linear_output_terms, result.output_indices);

        throw_if(result.input_indices.empty() && result.output_indices.empty(),
                 "ExpressionParser: expression simplifies to the constant "
                 + to_string(result.linear_constant) + " and constrains no variable");

        return result;
    }

    result.linearity = ExpressionLinearity::Nonlinear;
    result.program = build_program(ast);

    result.input_gradient.reserve(result.input_indices.size());

    for (const Index input_column : result.input_indices)
        result.input_gradient.emplace_back(input_column,
                                           build_program(*differentiate(ast, false, input_column)));

    return result;
}


ExpressionNodePtr parse_expression_tree(const string& expression,
                                        const vector<pair<string, Index>>& inputs,
                                        const vector<pair<string, Index>>& outputs)
{
    throw_if(expression.empty(), "ExpressionParser: empty expression");

    Lexer lexer(expression);
    Parser parser(lexer, inputs, outputs);

    ExpressionNodePtr ast = parser.parse_expression();

    throw_if(lexer.peek().kind != Token::Kind::End,
             format("ExpressionParser: trailing tokens after valid expression in '{}'", abbreviate(expression)));

    // Chained operations ('a + a + a + ...') leave the parser shallow but the tree
    // deep, so the tree itself is checked too.
    throw_if(ast->depth > max_operation_depth,
             format("ExpressionParser: the expression chains more than {} operations",
                    max_operation_depth));

    return ast;
}

}


namespace
{


double evaluate_symmetric_polynomial(const CompiledExpression& expression, const VectorR& point)
{
    const size_t order = size_t(expression.symmetric_order);

    thread_local vector<double> table;

    table.assign(order + 1, 0.0);
    table[0] = 1.0;

    size_t reachable = 0;

    for (const auto& [column, inverse_span] : expression.symmetric_terms)
    {
        const double scaled = double(point(column))*inverse_span;

        const double term = scaled*scaled;

        reachable = min(reachable + 1, order);

        for (size_t m = reachable; m >= 1; m--)
            table[m] += term*table[m - 1];
    }

    return table[order];
}


void evaluate_symmetric_gradient(const CompiledExpression& expression, const VectorR& point, VectorR& gradient)
{
    const size_t order = size_t(expression.symmetric_order);
    const size_t width = order + 1;
    const size_t terms_number = expression.symmetric_terms.size();

    thread_local vector<double> scaled;
    thread_local vector<double> prefix;
    thread_local vector<double> suffix;

    scaled.resize(terms_number);

    for (size_t j = 0; j < terms_number; j++)
    {
        const auto& [column, inverse_span] = expression.symmetric_terms[j];

        scaled[j] = double(point(column))*inverse_span;
    }

    prefix.assign((terms_number + 1)*width, 0.0);
    suffix.assign((terms_number + 1)*width, 0.0);

    prefix[0] = 1.0;
    suffix[terms_number*width] = 1.0;

    for (size_t j = 0; j < terms_number; j++)
    {
        const double term = scaled[j]*scaled[j];

        const double* before = &prefix[j*width];
        double* after = &prefix[(j + 1)*width];

        after[0] = 1.0;

        for (size_t m = 1; m < width; m++)
            after[m] = before[m] + term*before[m - 1];
    }

    for (size_t j = terms_number; j-- > 0;)
    {
        const double term = scaled[j]*scaled[j];

        const double* later = &suffix[(j + 1)*width];
        double* here = &suffix[j*width];

        here[0] = 1.0;

        for (size_t m = 1; m < width; m++)
            here[m] = later[m] + term*later[m - 1];
    }

    const double value = prefix[terms_number*width + order];

    if (!(value > 0.0)) return;

    const double chain = expression.symmetric_scale/sqrt(value);

    for (size_t j = 0; j < terms_number; j++)
    {
        double leave_one_out = 0.0;

        for (size_t a = 0; a < order; a++)
            leave_one_out += prefix[j*width + a]*suffix[(j + 1)*width + order - 1 - a];

        const auto& [column, inverse_span] = expression.symmetric_terms[j];

        gradient(column) = float(chain*scaled[j]*inverse_span*leave_one_out);
    }
}

}


float CompiledExpression::evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const
{
    if (symmetric_order > 0)
        return float(symmetric_scale*sqrt(evaluate_symmetric_polynomial(*this, inputs_row)));

    if (linearity == ExpressionLinearity::Nonlinear)
        return evaluate_program(program, inputs_row, outputs_row);

    float result = linear_constant;

    for (const auto& [column, coefficient] : linear_input_terms)
        result += coefficient * inputs_row(column);

    for (const auto& [column, coefficient] : linear_output_terms)
        result += coefficient * outputs_row(column);

    return result;
}


CompiledExpression compile_elementary_symmetric(const vector<Index>& variables,
                                                const vector<float>& spans,
                                                const Index order,
                                                const float tolerance)
{
    throw_if(variables.empty() || variables.size() != spans.size(),
             "ExpressionParser: a symmetric polynomial needs one span per variable");

    throw_if(order < 1 || order > Index(variables.size()),
             "ExpressionParser: a symmetric polynomial of order " + to_string(order)
             + " over " + to_string(variables.size()) + " variables is not defined");

    throw_if(!(tolerance > 0.0f), "ExpressionParser: a symmetric polynomial needs a positive tolerance");

    const double variables_number = double(variables.size());

    const double log_combinations = lgamma(variables_number + 1.0) - lgamma(double(order) + 1.0)
                                  - lgamma(variables_number - double(order) + 1.0);

    CompiledExpression result;

    result.text = "sqrt(e_" + to_string(order) + "(u^2)/C(" + to_string(variables.size()) + ", "
                + to_string(order) + "))";

    result.linearity = ExpressionLinearity::Nonlinear;

    result.input_indices = variables;

    ranges::sort(result.input_indices);

    result.symmetric_order = order;

    result.symmetric_scale = exp(-0.5*log_combinations - log(double(tolerance)));

    for (size_t i = 0; i < variables.size(); i++)
        result.symmetric_terms.emplace_back(variables[i], 1.0/max(double(spans[i]), double(EPSILON)));

    return result;
}


namespace
{

ExpressionNodePtr parse_for_network(const string& expression, const Network* network)
{
    throw_if(!network, "The neural network has not been set.");

    return parse_expression_tree(expression,
                                 get_variable_columns(network->get_input_variables()),
                                 get_variable_columns(network->get_output_variables()));
}

}


CompiledExpression compile_integrality(const string& expression, const Network* network)
{
    const float pi = numbers::pi_v<float>;

    return compile_ast(*make_div(make_call(ExpressionOp::Kind::Sin,
                                           make_mul(make_const(pi),
                                                    parse_for_network(expression, network))),
                                 make_const(pi)));
}


CompiledExpression compile_membership(const string& expression,
                                      const Network* network,
                                      const vector<float>& allowed)
{
    const auto [smallest, largest] = ranges::minmax(allowed);

    const float span = max(largest - smallest, EPSILON);

    const ExpressionNodePtr value = parse_for_network(expression, network);

    ExpressionNodePtr product = make_const(span);

    for (const float allowed_value : allowed)
        product = make_mul(move(product),
                           make_div(make_sub(clone(*value), make_const(allowed_value)),
                                    make_const(span)));

    return compile_ast(*product);
}


vector<string> split_expression_list(const string& expression)
{
    const Lexer lexer(expression);
    vector<string> members;
    size_t start = 0;
    for (const Token& token : lexer.tokens)
        if (token.kind == Token::Kind::Semicolon || token.kind == Token::Kind::End)
        {
            members.push_back(expression.substr(start, token.position - start));
            start = token.position + 1;
        }
    return members;
}


CompiledExpression compile_expression(const string& expression,
                                      const vector<pair<string, Index>>& inputs,
                                      const vector<pair<string, Index>>& outputs)
{
    CompiledExpression compiled = compile_ast(*parse_expression_tree(expression, inputs, outputs));

    compiled.text = expression;

    return compiled;
}


CompiledExpression compile_expression(const string& expression,
                                      const Network* network,
                                      const string& role)
{
    throw_if(!network, "The neural network has not been set.");

    throw_if(expression.find_first_of("<>=") != string::npos,
             role + " '" + abbreviate(expression) + "' cannot contain comparison symbols. Use a condition instead.");

    try
    {
        return compile_expression(expression,
                                  get_variable_columns(network->get_input_variables()),
                                  get_variable_columns(network->get_output_variables()));
    }
    catch (const exception& e)
    {
        throw runtime_error(role + " '" + abbreviate(expression) + "' cannot be read. " + e.what());
    }
}


bool is_output_coupled(const CompiledExpression& expression)
{
    return !expression.output_indices.empty();
}


bool is_univariate(const CompiledExpression& expression)
{
    return expression.input_indices.size() + expression.output_indices.size() == 1;
}


bool is_bare_variable(const CompiledExpression& expression)
{
    if (expression.linearity != ExpressionLinearity::Linear
     || !is_univariate(expression)
     || abs(expression.linear_constant) > EPSILON)
        return false;

    const auto& terms = expression.linear_input_terms.empty() ? expression.linear_output_terms
                                                              : expression.linear_input_terms;

    return abs(terms.front().second - 1.0f) <= EPSILON;
}


void evaluate_input_gradient(const CompiledExpression& expression,
                             const VectorR& point,
                             const VectorR& output,
                             VectorR& gradient)
{
    if (gradient.size() != point.size())
        gradient.resize(point.size());

    gradient.setZero();

    if (expression.symmetric_order > 0)
        evaluate_symmetric_gradient(expression, point, gradient);
    else if (expression.linearity == ExpressionLinearity::Linear)
        for (const auto& [column, coefficient] : expression.linear_input_terms)
            gradient(column) = coefficient;
    else
        for (const auto& [column, program] : expression.input_gradient)
            gradient(column) = evaluate_program(program, point, output);
}


VectorR evaluate_input_gradient(const CompiledExpression& expression, const VectorR& point, const VectorR& output)
{
    VectorR gradient(point.size());

    evaluate_input_gradient(expression, point, output, gradient);

    return gradient;
}

}
