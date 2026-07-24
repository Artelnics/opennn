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
#include <set>

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

}


namespace
{

struct Parser
{
    Lexer& lexer;
    const vector<pair<string, Index>>& input_columns;
    const vector<pair<string, Index>>& output_columns;

    Parser(Lexer& new_lexer,
                     const vector<pair<string, Index>>& new_input_columns,
                     const vector<pair<string, Index>>& new_output_columns)
        : lexer(new_lexer),
                    input_columns(new_input_columns),
                    output_columns(new_output_columns)
    {
    }

    ExpressionNodePtr parse_expression()
    {
        ExpressionNodePtr left_node = parse_term();

        while (true)
        {
            const Token& next_token = lexer.peek();

            if (next_token.kind != Token::Kind::Operator) break;
            if (next_token.text != "+" && next_token.text != "-") break;

            const string operator_text = lexer.consume().text;

            ExpressionNodePtr right_node = parse_term();

            auto combined_node = make_unique<ExpressionNode>();
            combined_node->kind = (operator_text == "+") ? ExpressionNode::Kind::Add : ExpressionNode::Kind::Sub;
            combined_node->children.reserve(2);
            combined_node->children.push_back(move(left_node));
            combined_node->children.push_back(move(right_node));
            left_node = move(combined_node);
        }

        return left_node;
    }

    ExpressionNodePtr parse_term()
    {
        ExpressionNodePtr left_node = parse_factor();

        while (true)
        {
            const Token& next_token = lexer.peek();

            if (next_token.kind != Token::Kind::Operator) break;
            if (next_token.text != "*" && next_token.text != "/") break;

            const string operator_text = lexer.consume().text;

            ExpressionNodePtr right_node = parse_factor();

            auto combined_node = make_unique<ExpressionNode>();
            combined_node->kind = (operator_text == "*") ? ExpressionNode::Kind::Mul : ExpressionNode::Kind::Div;
            combined_node->children.reserve(2);
            combined_node->children.push_back(move(left_node));
            combined_node->children.push_back(move(right_node));
            left_node = move(combined_node);
        }

        return left_node;
    }

    ExpressionNodePtr parse_factor()
    {
        ExpressionNodePtr left_node = parse_unary();

        const Token& next_token = lexer.peek();

        if (next_token.kind == Token::Kind::Operator && next_token.text == "^")
        {
            lexer.consume();

            ExpressionNodePtr right_node = parse_factor();

            auto combined_node = make_unique<ExpressionNode>();
            combined_node->kind = ExpressionNode::Kind::Pow;
            combined_node->children.reserve(2);
            combined_node->children.push_back(move(left_node));
            combined_node->children.push_back(move(right_node));
            return combined_node;
        }

        return left_node;
    }

    ExpressionNodePtr parse_unary()
    {
        const Token& next_token = lexer.peek();

        if (next_token.kind == Token::Kind::Operator && next_token.text == "-")
        {
            lexer.consume();

            ExpressionNodePtr child_node = parse_unary();

            auto negation_node = make_unique<ExpressionNode>();
            negation_node->kind = ExpressionNode::Kind::UnaryNeg;
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

    ExpressionNodePtr parse_primary()
    {
        Token token = lexer.consume();

        if (token.kind == Token::Kind::Number)
        {
            auto constant_node = make_unique<ExpressionNode>();
            constant_node->kind = ExpressionNode::Kind::Const;
            constant_node->constant = token.number;
            return constant_node;
        }

        if (token.kind == Token::Kind::LeftParen)
        {
            ExpressionNodePtr inner_node = parse_expression();
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

                auto function_node = make_unique<ExpressionNode>();
                function_node->kind = ExpressionNode::Kind::Func;
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

            for (const auto& named_column : input_columns)
                if (named_column.first == token.text)
                {
                    auto input_node = make_unique<ExpressionNode>();
                    input_node->kind = ExpressionNode::Kind::Input;
                    input_node->index = named_column.second;
                    return input_node;
                }

            for (const auto& named_column : output_columns)
                if (named_column.first == token.text)
                {
                    auto output_node = make_unique<ExpressionNode>();
                    output_node->kind = ExpressionNode::Kind::Output;
                    output_node->index = named_column.second;
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

}


static void accumulate_into(unordered_map<Index, float>& destination,
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


static void scale_terms_in_place(unordered_map<Index, float>& terms, const float scaling)
{
    for (auto& [column, coefficient] : terms)
        coefficient *= scaling;
}


static AffineForm analyze_affine(const ExpressionNode& node)
{
    AffineForm result;

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
        AffineForm child_form = analyze_affine(*node.children[0]);
        if (!child_form.is_affine) { result.is_affine = false; return result; }

        result.constant = -child_form.constant;
        scale_terms_in_place(child_form.input_terms, -1.0f);
        scale_terms_in_place(child_form.output_terms, -1.0f);
        result.input_terms = move(child_form.input_terms);
        result.output_terms = move(child_form.output_terms);
        return result;
    }

    case Add:
    case Sub:
    {
        AffineForm left_form = analyze_affine(*node.children[0]);
        AffineForm right_form = analyze_affine(*node.children[1]);
        if (!left_form.is_affine || !right_form.is_affine) { result.is_affine = false; return result; }

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

    case Div:
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

    case Pow:
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

    case Func:
        result.is_affine = false;
        return result;
    }

    result.is_affine = false;
    return result;
}


static void collect_variable_references(const ExpressionNode& node,
                                 std::set<Index>& input_references,
                                 std::set<Index>& output_references)
{
    if (node.kind == ExpressionNode::Kind::Input)  { input_references.insert(node.index);  return; }
    if (node.kind == ExpressionNode::Kind::Output) { output_references.insert(node.index); return; }

    for (const ExpressionNodePtr& child : node.children)
        collect_variable_references(*child, input_references, output_references);
}


static void validate_function_arities(const ExpressionNode& node)
{
    if (node.kind == ExpressionNode::Kind::Func)
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

    for (const ExpressionNodePtr& child : node.children)
        validate_function_arities(*child);
}


static void emit_operations(const ExpressionNode& node, vector<ExpressionOp>& operations)
{
    switch (node.kind)
    {
        using enum ExpressionNode::Kind;
    case Const:
        operations.push_back({ExpressionOp::Kind::PushConst, 0, node.constant});
        return;

    case Input:
        operations.push_back({ExpressionOp::Kind::PushInput, node.index, 0.0f});
        return;

    case Output:
        operations.push_back({ExpressionOp::Kind::PushOutput, node.index, 0.0f});
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
    {
        emit_operations(*node.children[0], operations);
        emit_operations(*node.children[1], operations);

        const ExpressionOp::Kind rpn_kind =
            node.kind == ExpressionNode::Kind::Add ? ExpressionOp::Kind::Add :
            node.kind == ExpressionNode::Kind::Sub ? ExpressionOp::Kind::Sub :
            node.kind == ExpressionNode::Kind::Mul ? ExpressionOp::Kind::Mul :
            node.kind == ExpressionNode::Kind::Div ? ExpressionOp::Kind::Div :
                                          ExpressionOp::Kind::Pow;

        operations.push_back({rpn_kind, 0, 0.0f});
        return;
    }

    case Func:
    {
        for (const ExpressionNodePtr& child : node.children)
            emit_operations(*child, operations);

        const string& function_name = node.function_name;

        ExpressionOp::Kind rpn_kind = ExpressionOp::Kind::Sqrt;
        if      (function_name == "sqrt") rpn_kind = ExpressionOp::Kind::Sqrt;
        else if (function_name == "exp")  rpn_kind = ExpressionOp::Kind::Exp;
        else if (function_name == "log")  rpn_kind = ExpressionOp::Kind::Log;
        else if (function_name == "abs")  rpn_kind = ExpressionOp::Kind::Abs;
        else if (function_name == "sin")  rpn_kind = ExpressionOp::Kind::Sin;
        else if (function_name == "cos")  rpn_kind = ExpressionOp::Kind::Cos;
        else if (function_name == "tan")  rpn_kind = ExpressionOp::Kind::Tan;
        else if (function_name == "min")  rpn_kind = ExpressionOp::Kind::Min;
        else if (function_name == "max")  rpn_kind = ExpressionOp::Kind::Max;
        else if (function_name == "pow")  rpn_kind = ExpressionOp::Kind::Pow;
        else throw runtime_error(format("FormulaParser: unknown function '{}'", function_name));

        operations.push_back({rpn_kind, 0, 0.0f});
        return;
    }
    }
}


ExpressionNodePtr clone(const ExpressionNode& node)
{
    auto copy = make_unique<ExpressionNode>();
    copy->kind = node.kind;
    copy->constant = node.constant;
    copy->index = node.index;
    copy->function_name = node.function_name;
    copy->children.reserve(node.children.size());
    for (const ExpressionNodePtr& child : node.children)
        copy->children.push_back(clone(*child));
    return copy;
}


static bool is_const(const ExpressionNode& node, float& value)
{
    if (node.kind == ExpressionNode::Kind::Const) { value = node.constant; return true; }
    return false;
}


static ExpressionNodePtr make_const(const float value)
{
    auto node = make_unique<ExpressionNode>();
    node->kind = ExpressionNode::Kind::Const;
    node->constant = value;
    return node;
}


static ExpressionNodePtr make_binary(const ExpressionNode::Kind kind, ExpressionNodePtr left, ExpressionNodePtr right)
{
    auto node = make_unique<ExpressionNode>();
    node->kind = kind;
    node->children.reserve(2);
    node->children.push_back(move(left));
    node->children.push_back(move(right));
    return node;
}


ExpressionNodePtr make_neg(ExpressionNodePtr a)
{
    float av;
    if (is_const(*a, av)) return make_const(-av);

    auto node = make_unique<ExpressionNode>();
    node->kind = ExpressionNode::Kind::UnaryNeg;
    node->children.push_back(move(a));
    return node;
}


static ExpressionNodePtr make_add(ExpressionNodePtr a, ExpressionNodePtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && b_c) return make_const(av + bv);
    if (a_c && av == 0.0f) return b;
    if (b_c && bv == 0.0f) return a;
    return make_binary(ExpressionNode::Kind::Add, move(a), move(b));
}


ExpressionNodePtr make_sub(ExpressionNodePtr a, ExpressionNodePtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && b_c) return make_const(av - bv);
    if (b_c && bv == 0.0f) return a;
    if (a_c && av == 0.0f) return make_neg(move(b));
    return make_binary(ExpressionNode::Kind::Sub, move(a), move(b));
}


static ExpressionNodePtr make_mul(ExpressionNodePtr a, ExpressionNodePtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && av == 0.0f) return make_const(0.0f);
    if (b_c && bv == 0.0f) return make_const(0.0f);
    if (a_c && b_c) return make_const(av * bv);
    if (a_c && av == 1.0f) return b;
    if (b_c && bv == 1.0f) return a;
    return make_binary(ExpressionNode::Kind::Mul, move(a), move(b));
}


static ExpressionNodePtr make_div(ExpressionNodePtr a, ExpressionNodePtr b)
{
    float av, bv;
    const bool a_c = is_const(*a, av);
    const bool b_c = is_const(*b, bv);
    if (a_c && av == 0.0f) return make_const(0.0f);
    if (b_c && bv == 1.0f) return a;
    if (a_c && b_c && bv != 0.0f) return make_const(av / bv);
    return make_binary(ExpressionNode::Kind::Div, move(a), move(b));
}


static ExpressionNodePtr make_pow(ExpressionNodePtr a, ExpressionNodePtr b)
{
    float bv;
    if (is_const(*b, bv))
    {
        if (bv == 0.0f) return make_const(1.0f);
        if (bv == 1.0f) return a;
    }
    float av;
    if (is_const(*a, av) && is_const(*b, bv)) return make_const(pow(av, bv));
    return make_binary(ExpressionNode::Kind::Pow, move(a), move(b));
}


static ExpressionNodePtr make_func(const string& name, ExpressionNodePtr argument)
{
    auto node = make_unique<ExpressionNode>();
    node->kind = ExpressionNode::Kind::Func;
    node->function_name = name;
    node->children.push_back(move(argument));
    return node;
}


static bool is_smooth(const ExpressionNode& node)
{
    if (node.kind == ExpressionNode::Kind::Func
        && (node.function_name == "min" || node.function_name == "max"))
        return false;

    return ranges::all_of(node.children, [](const ExpressionNodePtr& child) { return is_smooth(*child); });
}


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
        ExpressionNodePtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
        ExpressionNodePtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
        return make_add(make_mul(move(da), clone(*node.children[1])),
                        make_mul(clone(*node.children[0]), move(db)));
    }

    case Div:
    {
        ExpressionNodePtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
        ExpressionNodePtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
        ExpressionNodePtr numerator = make_sub(make_mul(move(da), clone(*node.children[1])),
                                    make_mul(clone(*node.children[0]), move(db)));
        ExpressionNodePtr denominator = make_mul(clone(*node.children[1]), clone(*node.children[1]));
        return make_div(move(numerator), move(denominator));
    }

    case Pow:
    {
        float exponent;
        if (is_const(*node.children[1], exponent))
        {
            ExpressionNodePtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
            ExpressionNodePtr power = make_pow(clone(*node.children[0]), make_const(exponent - 1.0f));
            return make_mul(make_mul(make_const(exponent), move(power)), move(da));
        }

        float base;
        if (is_const(*node.children[0], base))
        {
            ExpressionNodePtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
            ExpressionNodePtr value = make_pow(make_const(base), clone(*node.children[1]));
            return make_mul(make_mul(move(value), make_const(log(base))), move(db));
        }

        ExpressionNodePtr da = differentiate(*node.children[0], wrt_is_output, wrt_index);
        ExpressionNodePtr db = differentiate(*node.children[1], wrt_is_output, wrt_index);
        ExpressionNodePtr term1 = make_mul(move(db), make_func("log", clone(*node.children[0])));
        ExpressionNodePtr term2 = make_div(make_mul(clone(*node.children[1]), move(da)),
                                clone(*node.children[0]));
        ExpressionNodePtr value = make_pow(clone(*node.children[0]), clone(*node.children[1]));
        return make_mul(move(value), make_add(move(term1), move(term2)));
    }

    case Func:
    {
        const string& name = node.function_name;
        const ExpressionNode& u = *node.children[0];
        ExpressionNodePtr du = differentiate(u, wrt_is_output, wrt_index);

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


float evaluate_operations(const vector<ExpressionOp>& operations,
                   const VectorR& inputs_row,
                   const VectorR& outputs_row)
{
    thread_local vector<float> evaluation_stack;
    evaluation_stack.clear();
    evaluation_stack.reserve(16);

    for (const ExpressionOp& operation : operations)
    {
        switch (operation.kind)
        {
            using enum ExpressionOp::Kind;
        case PushConst:  evaluation_stack.push_back(operation.constant); break;
        case PushInput:  evaluation_stack.push_back(inputs_row(operation.index)); break;
        case PushOutput: evaluation_stack.push_back(outputs_row(operation.index)); break;
        case Neg: evaluation_stack.back() = -evaluation_stack.back(); break;

                case Add:
                case Sub:
                case Mul:
                case Div:
                case Pow:
                {
                        const float right_operand = evaluation_stack.back();
                        evaluation_stack.pop_back();
                        float &left_operand = evaluation_stack.back();
                        switch (operation.kind)
                        {
                                case ExpressionOp::Kind::Add: left_operand += right_operand; break;
                                case ExpressionOp::Kind::Sub: left_operand -= right_operand; break;
                                case ExpressionOp::Kind::Mul: left_operand *= right_operand; break;
                                case ExpressionOp::Kind::Div: left_operand /= right_operand; break;
                                case ExpressionOp::Kind::Pow: left_operand = pow(left_operand, right_operand); break;
                                default: break;
                        }
                        break;
                }

        case Sqrt: evaluation_stack.back() = sqrt(evaluation_stack.back()); break;
        case Exp:  evaluation_stack.back() = exp(evaluation_stack.back()); break;
        case Log:  evaluation_stack.back() = log(evaluation_stack.back()); break;
        case Abs:  evaluation_stack.back() = abs(evaluation_stack.back()); break;
        case Sin:  evaluation_stack.back() = sin(evaluation_stack.back()); break;
        case Cos:  evaluation_stack.back() = cos(evaluation_stack.back()); break;
        case Tan:  evaluation_stack.back() = tan(evaluation_stack.back()); break;

                case Min:
                case Max:
                {
                        const float right_operand = evaluation_stack.back();
                        evaluation_stack.pop_back();
                        float &left_operand = evaluation_stack.back();
                        if (operation.kind == ExpressionOp::Kind::Min) left_operand = min(left_operand, right_operand);
                        else left_operand = max(left_operand, right_operand);
                        break;
                }
        }
    }

    return evaluation_stack.back();
}


float CompiledExpression::evaluate(const VectorR& inputs_row, const VectorR& outputs_row) const
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

    return evaluate_operations(operations, inputs_row, outputs_row);
}


CompiledExpression compile_ast(const ExpressionNode& ast)
{
    validate_function_arities(ast);

    CompiledExpression result;

    std::set<Index> input_references;
    std::set<Index> output_references;
    collect_variable_references(ast, input_references, output_references);

    throw_if(input_references.empty() && output_references.empty(),
             "FormulaParser: expression references no input or output variables");

    result.input_indices.assign(input_references.begin(), input_references.end());
    result.output_indices.assign(output_references.begin(), output_references.end());

    if (output_references.empty())       result.scope = FormulaScope::InputsOnly;
    else if (input_references.empty())   result.scope = FormulaScope::OutputsOnly;
    else                                 result.scope = FormulaScope::Mixed;

    const AffineForm affine_form = analyze_affine(ast);

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

        if (is_smooth(ast))
        {
            result.input_gradient.reserve(result.input_indices.size());
            for (const Index input_column : result.input_indices)
            {
                const ExpressionNodePtr partial = differentiate(ast, false, input_column);
                vector<ExpressionOp> program;
                emit_operations(*partial, program);
                result.input_gradient.emplace_back(input_column, move(program));
            }

            result.output_gradient.reserve(result.output_indices.size());
            for (const Index output_column : result.output_indices)
            {
                const ExpressionNodePtr partial = differentiate(ast, true, output_column);
                vector<ExpressionOp> program;
                emit_operations(*partial, program);
                result.output_gradient.emplace_back(output_column, move(program));
            }
        }
    }

    emit_operations(ast, result.operations);

    return result;
}


ExpressionNodePtr parse_expression_tree(const string& expression,
                    const vector<pair<string, Index>>& inputs,
                    const vector<pair<string, Index>>& outputs)
{
    throw_if(expression.empty(),
             "FormulaParser: empty expression");

    Lexer lexer(expression);
    Parser parser(lexer, inputs, outputs);

    ExpressionNodePtr ast = parser.parse_expression();

    throw_if(lexer.peek().kind != Token::Kind::End,
             format("FormulaParser: trailing tokens after valid expression in '{}'", expression));

    return ast;
}


CompiledExpression compile_formula(const string& expression,
                                const vector<pair<string, Index>>& inputs,
                                const vector<pair<string, Index>>& outputs)
{
    return compile_ast(*parse_expression_tree(expression, inputs, outputs));
}


ExpressionEvaluator::ExpressionEvaluator(const string& expression)
    : source(expression)
{
    throw_if(expression.empty(), "ExpressionEvaluator: empty expression");
}

float ExpressionEvaluator::evaluate(const map<string, float>& variables) const
{
    vector<pair<string, Index>> input_columns;
    input_columns.reserve(variables.size());
    VectorR values(Index(variables.size()));
    Index index = 0;
    for (const auto& [name, value] : variables)
    {
        input_columns.emplace_back(name, index);
        values(index) = value;
        ++index;
    }
    const CompiledExpression compiled = compile_formula(source, input_columns, {});
    return compiled.evaluate(values, VectorR());
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
