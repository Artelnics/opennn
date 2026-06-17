//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   C O N S T R A I N T S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "pch.h"
#include "response_constraints.h"
#include "string_utilities.h"
#include "random_utilities.h"

#include <cctype>
#include <Eigen/Dense>

namespace opennn
{

namespace
{

struct Token
{
    enum class Kind { Number, Identifier, Operator, LeftParen, RightParen, Comma, End };

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
                token.number = parse_float(token.text, "FormulaParser: numeric literal");
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
                throw runtime_error(format("FormulaParser: unexpected character '{}' at position {}",
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


struct Ast;
using AstPtr = unique_ptr<Ast>;

struct Ast
{
    enum class Kind { Const, Input, Output, UnaryNeg, Add, Sub, Mul, Div, Pow, Func };

    Kind kind = Kind::Const;
    float constant = 0.0f;
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

            throw_if(closing_token.kind != Token::Kind::RightParen,
                     format("FormulaParser: expected ')' at position {}",
                            closing_token.position));

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
                throw_if(closing_token.kind != Token::Kind::RightParen,
                         format("FormulaParser: expected ')' in call to '{}'", token.text));

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

            throw runtime_error(format("FormulaParser: unknown identifier '{}' "
                                       "(not a registered input, output, or supported function)",
                                       token.text));
        }

        throw runtime_error(format("FormulaParser: unexpected token '{}' at position {}",
                                   token.text, token.position));
    }
};


struct AffineForm
{
    bool is_affine = true;
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


AffineForm analyze_affine(const Ast& node)
{
    AffineForm result;

    switch (node.kind)
    {
    case Ast::Kind::Const:
        result.constant = node.constant;
        return result;

    case Ast::Kind::Input:
        result.input_terms[node.index] = 1.0f;
        return result;

    case Ast::Kind::Output:
        result.output_terms[node.index] = 1.0f;
        return result;

    case Ast::Kind::UnaryNeg:
    {
        AffineForm child_form = analyze_affine(*node.children[0]);
        if (!child_form.is_affine) { result.is_affine = false; return result; }

        result.constant = -child_form.constant;
        scale_terms_in_place(child_form.input_terms, -1.0f);
        scale_terms_in_place(child_form.output_terms, -1.0f);
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

        const float sign = (node.kind == Ast::Kind::Add) ? 1.0f : -1.0f;
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

        const float inverse = 1.0f / right_form.constant;
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

        if (exponent_form.is_constant() && abs(exponent_form.constant - 1.0f) < EPSILON)
            return base_form;

        if (exponent_form.is_constant() && abs(exponent_form.constant) < EPSILON)
        {
            result.constant = 1.0f;
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
            throw_if(arguments_count != unary_iterator->second,
                     format("FormulaParser: function '{}' expects {} argument, got {}",
                            function_name, unary_iterator->second, arguments_count));
        }
        else if (binary_iterator != binary_functions.end())
        {
            throw_if(arguments_count != binary_iterator->second,
                     format("FormulaParser: function '{}' expects {} arguments, got {}",
                            function_name, binary_iterator->second, arguments_count));
        }
        else
        {
            throw runtime_error(format("FormulaParser: unknown function '{}'", function_name));
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
        bytecode.push_back({RpnOp::Kind::PushConst, 0, node.constant});
        return;

    case Ast::Kind::Input:
        bytecode.push_back({RpnOp::Kind::PushInput, node.index, 0.0f});
        return;

    case Ast::Kind::Output:
        bytecode.push_back({RpnOp::Kind::PushOutput, node.index, 0.0f});
        return;

    case Ast::Kind::UnaryNeg:
        emit_bytecode(*node.children[0], bytecode);
        bytecode.push_back({RpnOp::Kind::Neg, 0, 0.0f});
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

        bytecode.push_back({rpn_kind, 0, 0.0f});
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
        else throw runtime_error(format("FormulaParser: unknown function '{}'", function_name));

        bytecode.push_back({rpn_kind, 0, 0.0f});
        return;
    }
    }
}


AstPtr clone(const Ast& node)
{
    auto copy = make_unique<Ast>();
    copy->kind = node.kind;
    copy->constant = node.constant;
    copy->index = node.index;
    copy->function_name = node.function_name;
    copy->children.reserve(node.children.size());
    for (const AstPtr& child : node.children)
        copy->children.push_back(clone(*child));
    return copy;
}


bool is_const(const Ast& node, float& value)
{
    if (node.kind == Ast::Kind::Const) { value = node.constant; return true; }
    return false;
}


AstPtr make_const(const float value)
{
    auto node = make_unique<Ast>();
    node->kind = Ast::Kind::Const;
    node->constant = value;
    return node;
}


AstPtr make_binary(const Ast::Kind kind, AstPtr left, AstPtr right)
{
    auto node = make_unique<Ast>();
    node->kind = kind;
    node->children.reserve(2);
    node->children.push_back(move(left));
    node->children.push_back(move(right));
    return node;
}


AstPtr make_neg(AstPtr a)
{
    float av;
    if (is_const(*a, av)) return make_const(-av);

    auto node = make_unique<Ast>();
    node->kind = Ast::Kind::UnaryNeg;
    node->children.push_back(move(a));
    return node;
}


AstPtr make_add(AstPtr a, AstPtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && b_c) return make_const(av + bv);
    if (a_c && av == 0.0f) return b;
    if (b_c && bv == 0.0f) return a;
    return make_binary(Ast::Kind::Add, move(a), move(b));
}


AstPtr make_sub(AstPtr a, AstPtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && b_c) return make_const(av - bv);
    if (b_c && bv == 0.0f) return a;
    if (a_c && av == 0.0f) return make_neg(move(b));
    return make_binary(Ast::Kind::Sub, move(a), move(b));
}


AstPtr make_mul(AstPtr a, AstPtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && av == 0.0f) return make_const(0.0f);
    if (b_c && bv == 0.0f) return make_const(0.0f);
    if (a_c && b_c) return make_const(av * bv);
    if (a_c && av == 1.0f) return b;
    if (b_c && bv == 1.0f) return a;
    return make_binary(Ast::Kind::Mul, move(a), move(b));
}


AstPtr make_div(AstPtr a, AstPtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && av == 0.0f) return make_const(0.0f);
    if (b_c && bv == 1.0f) return a;
    if (a_c && b_c && bv != 0.0f) return make_const(av / bv);
    return make_binary(Ast::Kind::Div, move(a), move(b));
}


AstPtr make_pow(AstPtr a, AstPtr b)
{
    float bv;
    if (is_const(*b, bv))
    {
        if (bv == 0.0f) return make_const(1.0f);
        if (bv == 1.0f) return a;
    }
    float av;
    if (is_const(*a, av) && is_const(*b, bv)) return make_const(pow(av, bv));
    return make_binary(Ast::Kind::Pow, move(a), move(b));
}


AstPtr make_func(const string& name, AstPtr argument)
{
    auto node = make_unique<Ast>();
    node->kind = Ast::Kind::Func;
    node->function_name = name;
    node->children.push_back(move(argument));
    return node;
}


bool is_smooth(const Ast& node)
{
    if (node.kind == Ast::Kind::Func
        && (node.function_name == "min" || node.function_name == "max"))
        return false;

    return ranges::all_of(node.children, [](const AstPtr& child) { return is_smooth(*child); });
}


AstPtr differentiate(const Ast& node, const bool wrt_is_output, const Index wrt_index)
{
    switch (node.kind)
    {
    case Ast::Kind::Const:
        return make_const(0.0f);

    case Ast::Kind::Input:
        return make_const((!wrt_is_output && node.index == wrt_index) ? 1.0f : 0.0f);

    case Ast::Kind::Output:
        return make_const((wrt_is_output && node.index == wrt_index) ? 1.0f : 0.0f);

    case Ast::Kind::UnaryNeg:
        return make_neg(differentiate(*node.children[0], wrt_is_output, wrt_index));

    case Ast::Kind::Add:
        return make_add(differentiate(*node.children[0], wrt_is_output, wrt_index),
                        differentiate(*node.children[1], wrt_is_output, wrt_index));

    case Ast::Kind::Sub:
        return make_sub(differentiate(*node.children[0], wrt_is_output, wrt_index),
                        differentiate(*node.children[1], wrt_is_output, wrt_index));

    case Ast::Kind::Mul:
    {
        AstPtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
        AstPtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
        return make_add(make_mul(move(da), clone(*node.children[1])),
                        make_mul(clone(*node.children[0]), move(db)));
    }

    case Ast::Kind::Div:
    {
        AstPtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
        AstPtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
        AstPtr numerator = make_sub(make_mul(move(da), clone(*node.children[1])),
                                    make_mul(clone(*node.children[0]), move(db)));
        AstPtr denominator = make_mul(clone(*node.children[1]), clone(*node.children[1]));
        return make_div(move(numerator), move(denominator));
    }

    case Ast::Kind::Pow:
    {
        float exponent;
        if (is_const(*node.children[1], exponent))
        {
            AstPtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
            AstPtr power = make_pow(clone(*node.children[0]), make_const(exponent - 1.0f));
            return make_mul(make_mul(make_const(exponent), move(power)), move(da));
        }

        float base;
        if (is_const(*node.children[0], base))
        {
            AstPtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
            AstPtr value = make_pow(make_const(base), clone(*node.children[1]));
            return make_mul(make_mul(move(value), make_const(log(base))), move(db));
        }

        AstPtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
        AstPtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
        AstPtr term1 = make_mul(move(db), make_func("log", clone(*node.children[0])));
        AstPtr term2 = make_div(make_mul(clone(*node.children[1]), move(da)),
                                clone(*node.children[0]));
        AstPtr value = make_pow(clone(*node.children[0]), clone(*node.children[1]));
        return make_mul(move(value), make_add(move(term1), move(term2)));
    }

    case Ast::Kind::Func:
    {
        const string& name = node.function_name;
        const Ast& u = *node.children[0];
        AstPtr du = differentiate(u, wrt_is_output, wrt_index);

        if (name == "sqrt")
            return make_div(move(du), make_mul(make_const(2.0f), make_func("sqrt", clone(u))));
        if (name == "exp")
            return make_mul(make_func("exp", clone(u)), move(du));
        if (name == "log")
            return make_div(move(du), clone(u));
        if (name == "abs")
            return make_mul(make_div(clone(u), make_func("abs", clone(u))), move(du));
        if (name == "sin")
            return make_mul(make_func("cos", clone(u)), move(du));
        if (name == "cos")
            return make_neg(make_mul(make_func("sin", clone(u)), move(du)));
        if (name == "tan")
            return make_div(move(du), make_pow(make_func("cos", clone(u)), make_const(2.0f)));

        return make_const(0.0f);
    }
    }

    return make_const(0.0f);
}

}


float evaluate_rpn(const vector<RpnOp>& bytecode,
                   const VectorR& inputs_row,
                   const VectorR& outputs_row)
{
    vector<float> evaluation_stack;
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
        { const float right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() += right_operand; break; }
        case RpnOp::Kind::Sub:
        { const float right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() -= right_operand; break; }
        case RpnOp::Kind::Mul:
        { const float right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() *= right_operand; break; }
        case RpnOp::Kind::Div:
        { const float right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() /= right_operand; break; }
        case RpnOp::Kind::Pow:
        { const float right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() = pow(evaluation_stack.back(), right_operand); break; }

        case RpnOp::Kind::Sqrt: evaluation_stack.back() = sqrt(evaluation_stack.back()); break;
        case RpnOp::Kind::Exp:  evaluation_stack.back() = exp(evaluation_stack.back()); break;
        case RpnOp::Kind::Log:  evaluation_stack.back() = log(evaluation_stack.back()); break;
        case RpnOp::Kind::Abs:  evaluation_stack.back() = abs(evaluation_stack.back()); break;
        case RpnOp::Kind::Sin:  evaluation_stack.back() = sin(evaluation_stack.back()); break;
        case RpnOp::Kind::Cos:  evaluation_stack.back() = cos(evaluation_stack.back()); break;
        case RpnOp::Kind::Tan:  evaluation_stack.back() = tan(evaluation_stack.back()); break;

        case RpnOp::Kind::Min:
        { const float right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() = min(evaluation_stack.back(), right_operand); break; }
        case RpnOp::Kind::Max:
        { const float right_operand = evaluation_stack.back(); evaluation_stack.pop_back();
          evaluation_stack.back() = max(evaluation_stack.back(), right_operand); break; }
        }
    }

    return evaluation_stack.back();
}


float CompiledFormula::evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const
{
    if (shape == FormulaShape::Affine)
    {
        float result = affine_constant;

        for (const auto& [column, coefficient] : affine_input_terms)
            result += coefficient * inputs_row(column);

        for (const auto& [column, coefficient] : affine_output_terms)
            result += coefficient * outputs_row(column);

        return result;
    }

    return evaluate_rpn(bytecode, inputs_row, outputs_row);
}


CompiledFormula compile_formula(const string& expression,
                                const vector<NamedColumn>& inputs,
                                const vector<NamedColumn>& outputs)
{
    throw_if(expression.empty(),
             "FormulaParser: empty expression");

    Lexer lexer(expression);
    Parser parser(lexer, inputs, outputs);

    AstPtr ast = parser.parse_expression();

    throw_if(lexer.peek().kind != Token::Kind::End,
             format("FormulaParser: trailing tokens after valid expression in '{}'", expression));

    validate_function_arities(*ast);

    CompiledFormula result;

    set<Index> input_references;
    set<Index> output_references;
    collect_variable_references(*ast, input_references, output_references);

    throw_if(input_references.empty() && output_references.empty(),
             format("FormulaParser: expression '{}' references no input or output variables",
                    expression));

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

        if (is_smooth(*ast))
        {
            result.gradient_available = true;

            result.input_gradient.reserve(result.input_indices.size());
            for (const Index input_column : result.input_indices)
            {
                const AstPtr partial = differentiate(*ast, false, input_column);
                vector<RpnOp> program;
                emit_bytecode(*partial, program);
                result.input_gradient.emplace_back(input_column, move(program));
            }

            result.output_gradient.reserve(result.output_indices.size());
            for (const Index output_column : result.output_indices)
            {
                const AstPtr partial = differentiate(*ast, true, output_column);
                vector<RpnOp> program;
                emit_bytecode(*partial, program);
                result.output_gradient.emplace_back(output_column, move(program));
            }
        }
    }

    emit_bytecode(*ast, result.bytecode);

    return result;
}


bool all_formula_constraints_are_linear(const vector<MultivariateConstraint>& formula_constraints)
{
    return !formula_constraints.empty()
        && ranges::all_of(formula_constraints, [](const MultivariateConstraint& formula_constraint)
           {
               return !formula_constraint.uses_callback
                   && formula_constraint.compiled.shape == FormulaShape::Affine;
           });
}


LinearConstraintSet build_linear_constraint_set(const vector<MultivariateConstraint>& formula_constraints,
                                                const Index n_in,
                                                const Index n_out)
{
    const Index m = static_cast<Index>(formula_constraints.size());

    LinearConstraintSet linear_set;
    linear_set.A     = MatrixR::Zero(m, n_in + n_out);
    linear_set.lower = VectorR::Constant(m, -numeric_limits<float>::infinity());
    linear_set.upper = VectorR::Constant(m,  numeric_limits<float>::infinity());

    for (Index i = 0; i < m; ++i)
    {
        const MultivariateConstraint& formula_constraint = formula_constraints[i];

        for (const auto& [column, coefficient] : formula_constraint.compiled.affine_input_terms)
            linear_set.A(i, column) += coefficient;

        for (const auto& [column, coefficient] : formula_constraint.compiled.affine_output_terms)
            linear_set.A(i, n_in + column) += coefficient;

        const float c = formula_constraint.compiled.affine_constant;
        const float low = formula_constraint.low_bound;
        const float up  = formula_constraint.up_bound;

        switch (formula_constraint.comparison_operator)
        {
        case ComparisonOperator::EqualTo:
            linear_set.lower(i) = low - c - bound_tolerance(low);
            linear_set.upper(i) = low - c + bound_tolerance(low);
            break;
        case ComparisonOperator::Between:
            linear_set.lower(i) = low - c - bound_tolerance(low);
            linear_set.upper(i) = up  - c + bound_tolerance(up);
            break;
        case ComparisonOperator::GreaterEqualTo:
            linear_set.lower(i) = low - c - bound_tolerance(low);
            break;
        case ComparisonOperator::LessEqualTo:
            linear_set.upper(i) = up - c + bound_tolerance(up);
            break;
        case ComparisonOperator::GreaterThan:
            linear_set.lower(i) = low - c + EPSILON;
            break;
        case ComparisonOperator::LessThan:
            linear_set.upper(i) = up - c - EPSILON;
            break;
        case ComparisonOperator::None:
        default:
            break;
        }
    }

    return linear_set;
}


bool constraint_is_satisfied(const MultivariateConstraint& constraint,
                             const VectorR& input_row,
                             const VectorR& output_row)
{
    const float value = constraint.uses_callback
        ? constraint.callback(input_row, output_row)
        : constraint.compiled.evaluate(input_row, output_row);

    const float low = constraint.low_bound;
    const float up  = constraint.up_bound;

    switch (constraint.comparison_operator)
    {
    case ComparisonOperator::EqualTo:
        return abs(value - low) <= bound_tolerance(low);
    case ComparisonOperator::Between:
        return value >= low - bound_tolerance(low) && value <= up + bound_tolerance(up);
    case ComparisonOperator::GreaterEqualTo:
    case ComparisonOperator::GreaterThan:
        return value >= low - bound_tolerance(low);
    case ComparisonOperator::LessEqualTo:
    case ComparisonOperator::LessThan:
        return value <= up + bound_tolerance(up);
    default:
        return true;
    }
}


ConstraintKind classify(const MultivariateConstraint& constraint)
{
    if (constraint.comparison_operator == ComparisonOperator::None)
        return ConstraintKind::Unrepairable;

    if (constraint.uses_callback)
        return ConstraintKind::Callback;

    const CompiledFormula& formula = constraint.compiled;

    const bool affine = (formula.shape == FormulaShape::Affine);
    const bool nonlinear_ready = (formula.shape == FormulaShape::Nonlinear && formula.gradient_available);

    if (formula.scope == FormulaScope::InputsOnly)
    {
        if (affine && !formula.affine_input_terms.empty())
            return ConstraintKind::AffineInput;
        if (nonlinear_ready)
            return ConstraintKind::NonlinearInput;
        return ConstraintKind::Unrepairable;
    }

    return (affine || nonlinear_ready) ? ConstraintKind::OutputDependent : ConstraintKind::Unrepairable;
}


void snap_to_lattice(MatrixR& inputs, const Index column, const float minimum, const float maximum)
{
    inputs.col(column).array() = inputs.col(column).array().round().max(minimum).min(maximum);
}


namespace
{

bool constraint_residual(const ComparisonOperator comparison, const float low, const float up,
                         const float value, float& residual)
{
    residual = 0.0f;

    switch (comparison)
    {
    case ComparisonOperator::EqualTo:
        residual = value - low; return true;
    case ComparisonOperator::Between:
        if (value < low) { residual = value - low; return true; }
        if (value > up)  { residual = value - up;  return true; }
        return false;
    case ComparisonOperator::GreaterEqualTo:
    case ComparisonOperator::GreaterThan:
        if (value < low) { residual = value - low; return true; }
        return false;
    case ComparisonOperator::LessEqualTo:
    case ComparisonOperator::LessThan:
        if (value > up) { residual = value - up; return true; }
        return false;
    case ComparisonOperator::None:
    default:
        return false;
    }
}

bool gauss_newton_project_row(const MatrixR& jacobian, const VectorR& rhs,
                              const VectorR& inferior_frontier, const VectorR& superior_frontier,
                              VectorR& point)
{
    const Index active_number = jacobian.rows();

    vector<Index> projectable;
    projectable.reserve(active_number);
    for (Index i = 0; i < active_number; ++i)
        if (jacobian.row(i).cwiseAbs().maxCoeff() > 0.0f)
            projectable.push_back(i);

    if (projectable.empty())
        return false;

    const Index projectable_number = static_cast<Index>(projectable.size());

    MatrixR reduced_jacobian(projectable_number, jacobian.cols());
    VectorR reduced_rhs(projectable_number);
    for (Index k = 0; k < projectable_number; ++k)
    {
        reduced_jacobian.row(k) = jacobian.row(projectable[k]);
        reduced_rhs(k) = rhs(projectable[k]);
    }

    MatrixR gram = reduced_jacobian * reduced_jacobian.transpose();
    gram.diagonal().array() += EPSILON;

    point -= reduced_jacobian.transpose() * gram.ldlt().solve(reduced_rhs);
    point = point.cwiseMax(inferior_frontier).cwiseMin(superior_frontier);
    return true;
}

}


void repair_affine_inputs(MatrixR& random_inputs,
                          const VectorR& inferior_frontier,
                          const VectorR& superior_frontier,
                          const vector<MultivariateConstraint>& formula_constraints,
                          const Index max_correction_passes)
{
    vector<const MultivariateConstraint*> affine_constraints;

    for (const MultivariateConstraint& constraint : formula_constraints)
        if (constraint.kind == ConstraintKind::AffineInput)
            affine_constraints.push_back(&constraint);

    if (affine_constraints.empty())
        return;

    const Index rows_number = random_inputs.rows();
    const Index inputs_number = random_inputs.cols();
    const Index constraints_number = static_cast<Index>(affine_constraints.size());

    Index slacks_number = 0;
    for (const MultivariateConstraint* constraint : affine_constraints)
        if (constraint->comparison_operator != ComparisonOperator::EqualTo)
            ++slacks_number;

    const Index augmented_number = inputs_number + slacks_number;

    MatrixR augmented_matrix = MatrixR::Zero(constraints_number, augmented_number);
    VectorR right_hand_side = VectorR::Zero(constraints_number);
    VectorR slack_inferior = VectorR::Zero(slacks_number);
    VectorR slack_superior = VectorR::Zero(slacks_number);

    Index slack_index = 0;

    for (Index i = 0; i < constraints_number; ++i)
    {
        const MultivariateConstraint& constraint = *affine_constraints[i];
        const float constant = constraint.compiled.affine_constant;
        const float low = constraint.low_bound;
        const float up = constraint.up_bound;

        float expression_minimum = constant;
        float expression_maximum = constant;

        for (const auto& [column, coefficient] : constraint.compiled.affine_input_terms)
        {
            augmented_matrix(i, column) += coefficient;
            expression_minimum += min(coefficient * inferior_frontier(column), coefficient * superior_frontier(column));
            expression_maximum += max(coefficient * inferior_frontier(column), coefficient * superior_frontier(column));
        }

        switch (constraint.comparison_operator)
        {
        case ComparisonOperator::EqualTo:
            right_hand_side(i) = low - constant;
            break;

        case ComparisonOperator::Between:
            augmented_matrix(i, inputs_number + slack_index) = -1;
            right_hand_side(i) = low - constant;
            slack_inferior(slack_index) = max(0.0f, expression_minimum - low);
            slack_superior(slack_index) = max(slack_inferior(slack_index), min(up - low, expression_maximum - low));
            ++slack_index;
            break;

        case ComparisonOperator::GreaterEqualTo:
        case ComparisonOperator::GreaterThan:
            augmented_matrix(i, inputs_number + slack_index) = -1;
            right_hand_side(i) = low - constant;
            slack_inferior(slack_index) = max(0.0f, expression_minimum - low);
            slack_superior(slack_index) = max(slack_inferior(slack_index), expression_maximum - low);
            ++slack_index;
            break;

        case ComparisonOperator::LessEqualTo:
        case ComparisonOperator::LessThan:
            augmented_matrix(i, inputs_number + slack_index) = 1;
            right_hand_side(i) = up - constant;
            slack_inferior(slack_index) = max(0.0f, up - expression_maximum);
            slack_superior(slack_index) = max(slack_inferior(slack_index), up - expression_minimum);
            ++slack_index;
            break;

        case ComparisonOperator::None:
        default:
            break;
        }
    }

    MatrixR augmented_points(rows_number, augmented_number);
    augmented_points.leftCols(inputs_number) = random_inputs;

    if (slacks_number > 0)
    {
        MatrixR slack_points(rows_number, slacks_number);
        set_random_uniform(slack_points, 0, 1);

        for (Index j = 0; j < slacks_number; ++j)
            slack_points.col(j) = (slack_inferior(j) + slack_points.col(j).array() * (slack_superior(j) - slack_inferior(j))).matrix();

        augmented_points.rightCols(slacks_number) = slack_points;
    }

    MatrixR gram_matrix = augmented_matrix * augmented_matrix.transpose();
    gram_matrix.diagonal().array() += EPSILON;
    const auto gram_solver = gram_matrix.ldlt();

    MatrixR affine_correction = MatrixR::Zero(rows_number, augmented_number);
    MatrixR box_correction = MatrixR::Zero(rows_number, augmented_number);

    const Index passes = max(Index(1), max_correction_passes);

    for (Index pass = 0; pass < passes; ++pass)
    {
        const MatrixR shifted_points = augmented_points + affine_correction;

        MatrixR residual = shifted_points * augmented_matrix.transpose();
        residual.rowwise() -= right_hand_side.transpose();

        const MatrixR projected_points = shifted_points - (gram_solver.solve(residual.transpose())).transpose() * augmented_matrix;
        affine_correction = shifted_points - projected_points;

        augmented_points = projected_points + box_correction;

        for (Index j = 0; j < inputs_number; ++j)
            augmented_points.col(j) = augmented_points.col(j).array().max(inferior_frontier(j)).min(superior_frontier(j)).matrix();

        for (Index j = 0; j < slacks_number; ++j)
            augmented_points.col(inputs_number + j) = augmented_points.col(inputs_number + j).array().max(slack_inferior(j)).min(slack_superior(j)).matrix();

        box_correction = projected_points + box_correction - augmented_points;

        MatrixR feasibility_residual = augmented_points * augmented_matrix.transpose();
        feasibility_residual.rowwise() -= right_hand_side.transpose();

        if (feasibility_residual.cwiseAbs().maxCoeff() <= EPSILON)
            break;
    }

    random_inputs = augmented_points.leftCols(inputs_number);
}


void repair_single_affine_input(MatrixR& random_inputs,
                                const VectorR& inferior_frontier,
                                const VectorR& superior_frontier,
                                const MultivariateConstraint& constraint)
{
    const vector<pair<Index, float>>& terms = constraint.compiled.affine_input_terms;

    if (terms.empty())
        return;

    const float constant = constraint.compiled.affine_constant;
    const float low = constraint.low_bound;
    const float up  = constraint.up_bound;
    const Index rows_number = random_inputs.rows();

    vector<pair<Index, float>> shuffled(terms.begin(), terms.end());

    const Index terms_number = static_cast<Index>(shuffled.size());

    for (Index r = 0; r < rows_number; ++r)
    {
        float expression = constant;
        for (const auto& [column, coefficient] : shuffled)
            expression += coefficient * random_inputs(r, column);

        float residual;
        if (!constraint_residual(constraint.comparison_operator, low, up, expression, residual))
            continue;
        residual = -residual;

        partial_shuffle(shuffled, terms_number);

        for (const auto& [column, coefficient] : shuffled)
        {
            if (coefficient == 0.0f)
                continue;

            const float old_value = random_inputs(r, column);
            const float wanted = old_value + residual / coefficient;
            const float new_value = min(superior_frontier(column),
                                        max(inferior_frontier(column), wanted));

            residual -= coefficient * (new_value - old_value);
            random_inputs(r, column) = new_value;

            if (abs(residual) <= EPSILON)
                break;
        }
    }
}


void repair_single_affine_integer(MatrixR& random_inputs,
                                  const VectorR& inferior_frontier,
                                  const VectorR& superior_frontier,
                                  const MultivariateConstraint& constraint)
{
    const vector<pair<Index, float>>& terms = constraint.compiled.affine_input_terms;

    if (terms.empty())
        return;

    const float constant = constraint.compiled.affine_constant;
    const float low = constraint.low_bound;
    const float up  = constraint.up_bound;
    const Index rows_number = random_inputs.rows();

    vector<pair<Index, float>> shuffled(terms.begin(), terms.end());
    const Index terms_number = static_cast<Index>(shuffled.size());

    for (Index r = 0; r < rows_number; ++r)
    {
        for (const auto& [column, coefficient] : shuffled)
        {
            const float lattice_min = ceil(inferior_frontier(column));
            const float lattice_max = floor(superior_frontier(column));
            random_inputs(r, column) = min(lattice_max, max(lattice_min, round(random_inputs(r, column))));
        }

        float expression = constant;
        for (const auto& [column, coefficient] : shuffled)
            expression += coefficient * random_inputs(r, column);

        float residual;
        if (!constraint_residual(constraint.comparison_operator, low, up, expression, residual))
            continue;
        residual = -residual;

        partial_shuffle(shuffled, terms_number);

        for (const auto& [column, coefficient] : shuffled)
        {
            if (coefficient == 0.0f)
                continue;

            const float lattice_min = ceil(inferior_frontier(column));
            const float lattice_max = floor(superior_frontier(column));

            const float real_delta = residual / coefficient;
            const float integer_delta = (real_delta > 0.0f) ? floor(real_delta) : ceil(real_delta);
            if (integer_delta == 0.0f)
                continue;

            const float old_value = random_inputs(r, column);
            const float new_value = min(lattice_max, max(lattice_min, old_value + integer_delta));

            residual -= coefficient * (new_value - old_value);
            random_inputs(r, column) = new_value;

            if (abs(residual) <= EPSILON)
                break;
        }
    }
}


void repair_nonlinear_inputs(MatrixR& random_inputs,
                             const VectorR& inferior_frontier,
                             const VectorR& superior_frontier,
                             const vector<MultivariateConstraint>& formula_constraints,
                             const Index max_correction_passes)
{
    vector<const MultivariateConstraint*> constraints;
    bool any_nonlinear = false;

    for (const MultivariateConstraint& constraint : formula_constraints)
    {
        const ConstraintKind kind = constraint.kind;

        if (kind == ConstraintKind::AffineInput)
            constraints.push_back(&constraint);
        else if (kind == ConstraintKind::NonlinearInput)
        {
            constraints.push_back(&constraint);
            any_nonlinear = true;
        }
    }

    if (!any_nonlinear || constraints.empty())
        return;

    const Index rows_number   = random_inputs.rows();
    const Index inputs_number = random_inputs.cols();
    const Index passes        = max(Index(1), max_correction_passes);

    const VectorR empty_outputs;

    for (Index r = 0; r < rows_number; ++r)
    {
        VectorR point = random_inputs.row(r).transpose();

        for (Index pass = 0; pass < passes; ++pass)
        {
            vector<const MultivariateConstraint*> active;
            vector<float> residuals;
            active.reserve(constraints.size());
            residuals.reserve(constraints.size());

            for (const MultivariateConstraint* constraint : constraints)
            {
                const float value = constraint->compiled.evaluate(point, empty_outputs);
                float residual;
                if (constraint_residual(constraint->comparison_operator, constraint->low_bound, constraint->up_bound, value, residual))
                {
                    active.push_back(constraint);
                    residuals.push_back(residual);
                }
            }

            if (active.empty())
                break;

            if (VectorR::Map(residuals.data(), Index(residuals.size())).cwiseAbs().maxCoeff() <= EPSILON)
                break;

            const Index active_number = static_cast<Index>(active.size());

            MatrixR jacobian = MatrixR::Zero(active_number, inputs_number);
            VectorR rhs(active_number);

            for (Index i = 0; i < active_number; ++i)
            {
                rhs(i) = residuals[i];
                const CompiledFormula& compiled = active[i]->compiled;

                if (compiled.shape == FormulaShape::Affine)
                    for (const auto& [column, coefficient] : compiled.affine_input_terms)
                        jacobian(i, column) = coefficient;
                else
                    for (const auto& [column, program] : compiled.input_gradient)
                        jacobian(i, column) = evaluate_rpn(program, point, empty_outputs);
            }

            if (!gauss_newton_project_row(jacobian, rhs, inferior_frontier, superior_frontier, point))
                break;
        }

        random_inputs.row(r) = point.transpose();
    }
}


void repair_affine_inputs_with_fixed(MatrixR& random_inputs,
                                     const VectorR& inferior_frontier,
                                     const VectorR& superior_frontier,
                                     const vector<MultivariateConstraint>& formula_constraints,
                                     const vector<char>& fixed_columns,
                                     const Index max_correction_passes)
{
    vector<const MultivariateConstraint*> constraints;

    for (const MultivariateConstraint& constraint : formula_constraints)
    {
        const ConstraintKind kind = constraint.kind;
        if (kind == ConstraintKind::AffineInput || kind == ConstraintKind::NonlinearInput)
            constraints.push_back(&constraint);
    }

    if (constraints.empty())
        return;

    const Index rows_number   = random_inputs.rows();
    const Index inputs_number = random_inputs.cols();
    const Index passes        = max(Index(1), max_correction_passes);

    const bool has_mask = (static_cast<Index>(fixed_columns.size()) == inputs_number);

    const VectorR empty_outputs;

    for (Index r = 0; r < rows_number; ++r)
    {
        VectorR point = random_inputs.row(r).transpose();

        for (Index pass = 0; pass < passes; ++pass)
        {
            vector<const MultivariateConstraint*> active;
            vector<float> residuals;
            active.reserve(constraints.size());
            residuals.reserve(constraints.size());

            for (const MultivariateConstraint* constraint : constraints)
            {
                const float value = constraint->compiled.evaluate(point, empty_outputs);
                float residual;
                if (constraint_residual(constraint->comparison_operator, constraint->low_bound, constraint->up_bound, value, residual))
                {
                    active.push_back(constraint);
                    residuals.push_back(residual);
                }
            }

            if (active.empty())
                break;

            if (VectorR::Map(residuals.data(), Index(residuals.size())).cwiseAbs().maxCoeff() <= EPSILON)
                break;

            const Index active_number = static_cast<Index>(active.size());

            MatrixR jacobian = MatrixR::Zero(active_number, inputs_number);
            VectorR rhs(active_number);

            for (Index i = 0; i < active_number; ++i)
            {
                rhs(i) = residuals[i];
                const CompiledFormula& compiled = active[i]->compiled;

                if (compiled.shape == FormulaShape::Affine)
                    for (const auto& [column, coefficient] : compiled.affine_input_terms)
                        jacobian(i, column) = coefficient;
                else
                    for (const auto& [column, program] : compiled.input_gradient)
                        jacobian(i, column) = evaluate_rpn(program, point, empty_outputs);
            }

            if (has_mask)
                for (Index j = 0; j < inputs_number; ++j)
                    if (fixed_columns[j])
                        jacobian.col(j).setZero();

            if (!gauss_newton_project_row(jacobian, rhs, inferior_frontier, superior_frontier, point))
                break;
        }

        random_inputs.row(r) = point.transpose();
    }
}


void repair_inputs(MatrixR& random_inputs,
                   const VectorR& inferior_frontier,
                   const VectorR& superior_frontier,
                   const vector<MultivariateConstraint>& formula_constraints)
{
    const MultivariateConstraint* single_affine = nullptr;
    Index affine_number = 0;
    bool any_nonlinear = false;

    for (const MultivariateConstraint& constraint : formula_constraints)
    {
        const ConstraintKind kind = constraint.kind;

        if (kind == ConstraintKind::AffineInput)
        {
            ++affine_number;
            single_affine = &constraint;
        }
        else if (kind == ConstraintKind::NonlinearInput)
            any_nonlinear = true;
    }

    if (any_nonlinear)
        repair_nonlinear_inputs(random_inputs, inferior_frontier, superior_frontier, formula_constraints);
    else if (affine_number == 1)
        repair_single_affine_input(random_inputs, inferior_frontier, superior_frontier, *single_affine);
    else if (affine_number >= 2)
        repair_affine_inputs(random_inputs, inferior_frontier, superior_frontier, formula_constraints);
}


namespace
{

bool row_satisfies_input_affine(const VectorR& point,
                                const vector<const MultivariateConstraint*>& input_constraints)
{
    const VectorR empty_outputs;

    for (const MultivariateConstraint* constraint : input_constraints)
        if (!constraint_is_satisfied(*constraint, point, empty_outputs))
            return false;

    return true;
}


void cardinality_swap_row(VectorR& point,
                          const vector<Index>& columns,
                          const VectorR& inferior_frontier,
                          const VectorR& superior_frontier,
                          const Index swaps)
{
    for (Index s = 0; s < swaps; ++s)
    {
        vector<Index> off_candidates, on_candidates;

        for (const Index column : columns)
        {
            const float value = point(column);
            if (value > 0.5f && (value - inferior_frontier(column)) >= 1.0f - EPSILON)
                off_candidates.push_back(column);
            else if (value < 0.5f && (superior_frontier(column) - value) >= 1.0f - EPSILON)
                on_candidates.push_back(column);
        }

        if (off_candidates.empty() || on_candidates.empty())
            break;

        const Index off_column = off_candidates[random_integer(0, static_cast<Index>(off_candidates.size()) - 1)];
        const Index on_column  = on_candidates [random_integer(0, static_cast<Index>(on_candidates.size())  - 1)];

        point(off_column) = round(point(off_column) - 1.0f);
        point(on_column)  = round(point(on_column)  + 1.0f);
    }
}


void unlock_free_integers_row(VectorR& point, const Lattice& lattice, const float fraction)
{
    for (size_t c = 0; c < lattice.columns.size(); ++c)
        if (random_uniform(0.0f, 1.0f) < fraction)
        {
            const float span = max(0.0f, lattice.max[c] - lattice.min[c]);
            const float draw = random_uniform(0.0f, 1.0f) * (span + 1.0f) + (lattice.min[c] - 0.5f);
            point(lattice.columns[c]) = min(lattice.max[c], max(lattice.min[c], round(draw)));
        }
}

}


void repair_mixed_integer_inputs(MatrixR& inputs,
                                 const VectorR& inferior_frontier,
                                 const VectorR& superior_frontier,
                                 const vector<MultivariateConstraint>& formula_constraints,
                                 const vector<char>& fixed_mask,
                                 const Lattice& lattice,
                                 const vector<vector<Index>>& cardinality_columns,
                                 const Lattice& free_lattice,
                                 const Index outer_cap,
                                 const float exploration_ratio)
{
    vector<const MultivariateConstraint*> input_constraints;
    for (const MultivariateConstraint& constraint : formula_constraints)
    {
        const ConstraintKind kind = constraint.kind;
        if (kind == ConstraintKind::AffineInput || kind == ConstraintKind::NonlinearInput)
            input_constraints.push_back(&constraint);
    }

    const Index rows = inputs.rows();
    const Index passes = max(Index(1), outer_cap);

    for (size_t c = 0; c < lattice.columns.size(); ++c)
        snap_to_lattice(inputs, lattice.columns[c], lattice.min[c], lattice.max[c]);

    std::set<Index> cardinality_set;
    for (const vector<Index>& group : cardinality_columns)
        cardinality_set.insert(group.begin(), group.end());

    for (const MultivariateConstraint& constraint : formula_constraints)
    {
        if (constraint.kind != ConstraintKind::AffineInput)
            continue;

        bool all_free_discrete = true;
        for (const pair<Index, float>& term : constraint.compiled.affine_input_terms)
            if (term.first >= static_cast<Index>(fixed_mask.size())
                || !fixed_mask[term.first] || cardinality_set.count(term.first))
            { all_free_discrete = false; break; }

        if (all_free_discrete)
            repair_single_affine_integer(inputs, inferior_frontier, superior_frontier, constraint);
    }

    for (Index outer = 0; outer < passes; ++outer)
    {
        repair_affine_inputs_with_fixed(inputs, inferior_frontier, superior_frontier,
                                        formula_constraints, fixed_mask);

        if (input_constraints.empty())
            return;

        bool all_feasible = true;
        const bool last_pass = (outer + 1 >= passes);
        const Index swaps = 1 + outer / 2;
        const float unlock_fraction = min(1.0f, exploration_ratio * float(outer + 1));

        for (Index r = 0; r < rows; ++r)
        {
            VectorR point = inputs.row(r).transpose();

            if (row_satisfies_input_affine(point, input_constraints))
                continue;

            all_feasible = false;
            if (last_pass)
                continue;

            for (const vector<Index>& columns : cardinality_columns)
                cardinality_swap_row(point, columns, inferior_frontier, superior_frontier, swaps);

            unlock_free_integers_row(point, free_lattice, unlock_fraction);

            inputs.row(r) = point.transpose();
        }

        if (all_feasible)
            break;
    }
}


void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<MultivariateConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               const SurrogateVjp& vjp,
                               const Index max_correction_passes,
                               const vector<char>& fixed_columns)
{
    vector<const MultivariateConstraint*> constraints;

    for (const MultivariateConstraint& constraint : formula_constraints)
        if (constraint.kind == ConstraintKind::OutputDependent)
            constraints.push_back(&constraint);

    if (constraints.empty())
        return;

    const Index rows_number   = inputs.rows();
    const Index inputs_number  = inputs.cols();
    const Index passes        = max(Index(1), max_correction_passes);

    const bool has_mask = (static_cast<Index>(fixed_columns.size()) == inputs_number);

    for (Index r = 0; r < rows_number; ++r)
    {
        VectorR point = inputs.row(r).transpose();

        for (Index pass = 0; pass < passes; ++pass)
        {
            const VectorR output = forward(point);
            const Index outputs_number = output.size();

            vector<const MultivariateConstraint*> active;
            vector<float> residuals;
            active.reserve(constraints.size());
            residuals.reserve(constraints.size());

            for (const MultivariateConstraint* constraint : constraints)
            {
                const float value = constraint->compiled.evaluate(point, output);
                float residual;
                if (constraint_residual(constraint->comparison_operator, constraint->low_bound, constraint->up_bound, value, residual))
                {
                    active.push_back(constraint);
                    residuals.push_back(residual);
                }
            }

            if (active.empty())
                break;

            if (VectorR::Map(residuals.data(), Index(residuals.size())).cwiseAbs().maxCoeff() <= EPSILON)
                break;

            const Index active_number = static_cast<Index>(active.size());

            MatrixR jacobian = MatrixR::Zero(active_number, inputs_number);
            VectorR rhs(active_number);

            for (Index i = 0; i < active_number; ++i)
            {
                rhs(i) = residuals[i];
                const CompiledFormula& compiled = active[i]->compiled;

                VectorR grad_x = VectorR::Zero(inputs_number);
                VectorR cotangent = VectorR::Zero(outputs_number);

                if (compiled.shape == FormulaShape::Affine)
                {
                    for (const auto& [column, coefficient] : compiled.affine_input_terms)
                        grad_x(column) = coefficient;
                    for (const auto& [column, coefficient] : compiled.affine_output_terms)
                        cotangent(column) = coefficient;
                }
                else
                {
                    for (const auto& [column, program] : compiled.input_gradient)
                        grad_x(column) = evaluate_rpn(program, point, output);
                    for (const auto& [column, program] : compiled.output_gradient)
                        cotangent(column) = evaluate_rpn(program, point, output);
                }

                jacobian.row(i) = (grad_x + vjp(point, cotangent)).transpose();
            }

            if (has_mask)
                for (Index j = 0; j < inputs_number; ++j)
                    if (fixed_columns[j])
                        jacobian.col(j).setZero();

            if (!gauss_newton_project_row(jacobian, rhs, inferior_frontier, superior_frontier, point))
                break;
        }

        inputs.row(r) = point.transpose();
    }
}


void repair_output_constraints(MatrixR& inputs,
                               const VectorR& inferior_frontier,
                               const VectorR& superior_frontier,
                               const vector<MultivariateConstraint>& formula_constraints,
                               const SurrogateForward& forward,
                               const Index max_correction_passes,
                               const vector<char>& fixed_columns)
{
    const Index inputs_number = inputs.cols();

    VectorR step(inputs_number);
    for (Index j = 0; j < inputs_number; ++j)
        step(j) = max(1e-4f, 1e-3f * (superior_frontier(j) - inferior_frontier(j)));

    const SurrogateVjp finite_difference_vjp =
        [&forward, inputs_number, step](const VectorR& x, const VectorR& cotangent)
    {
        VectorR gradient = VectorR::Zero(inputs_number);
        for (Index k = 0; k < inputs_number; ++k)
        {
            VectorR plus = x, minus = x;
            plus(k) += step(k);
            minus(k) -= step(k);
            const VectorR derivative = (forward(plus) - forward(minus)) / (2.0f * step(k));
            gradient(k) = derivative.dot(cotangent);
        }
        return gradient;
    };

    repair_output_constraints(inputs, inferior_frontier, superior_frontier,
                              formula_constraints, forward, finite_difference_vjp,
                              max_correction_passes, fixed_columns);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
