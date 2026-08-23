//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"

#include <charconv>
#include <cctype>
#include <initializer_list>
#include <limits>
#include <type_traits>
#ifdef __APPLE__
#include <cstdlib>
#include <cerrno>
#include <type_traits>
#endif

namespace opennn
{

#ifdef __APPLE__

    template <typename T>
    inline from_chars_result from_chars(const char* first, const char* last, T& value)
    {
        if constexpr (is_floating_point_v<T>)
        {
            const string buffer(first, last);
            char* parse_end = nullptr;
            errno = 0;

            if constexpr (is_same_v<T, float>)
                value = strtof(buffer.c_str(), &parse_end);
            else
                value = static_cast<T>(strtod(buffer.c_str(), &parse_end));

            if (parse_end == buffer.c_str())
                return {first, errc::invalid_argument};

            const size_t consumed = static_cast<size_t>(parse_end - buffer.c_str());

            if (errno == ERANGE)
                return {first + consumed, errc::result_out_of_range};

            return {first + consumed, errc{}};
        }
        else
        {
            // std:: is load-bearing. Unqualified lookup finds this overload
            // first - it is in the same namespace - and the call recursed until
            // the stack ran out for every integer parse on macOS.
            return std::from_chars(first, last, value);
        }
    }
#endif

    struct TransparentStringHash
    {
        using is_transparent = void;

        size_t operator()(string_view value) const noexcept
        {
            return hash<string_view>{}(value);
        }
    };

    template <typename Value>
    using StringMap = unordered_map<string, Value, TransparentStringHash, equal_to<>>;

    vector<string> get_tokens(string_view, string_view);

    vector<string_view> get_token_views(string_view, char);
    void split_views(string_view, char, vector<string_view>&);

    vector<string_view> get_token_views_maybe_quoted(string_view line, char separator,
                                                     bool file_has_quotes, string& scratch);

    void get_token_views_maybe_quoted(string_view line, char separator, bool file_has_quotes,
                                      string& scratch, vector<string_view>& out);

    string_view first_token_maybe_quoted(string_view line, char separator,
                                         bool file_has_quotes, string& scratch);

    string_view trim_view(string_view);
    void ascii_lowercase_in_place(string&) noexcept;
    string ascii_lowercase(string_view);
    void append_utf8(string&, uint32_t);

    vector<string> tokenize(const string&);

    vector<string_view> tokenize_views(string_view);

    vector<string> convert_string_vector(const vector<vector<string>>&, const string&);

    void replace_all_word_appearances(string&, const string&, const string&);

    string join_strings(span<const string>, string_view = " ");

    template <typename T>
    T parse_number(string_view text, string_view context, string_view value_kind = "numeric")
    {
        T value{};
        const char* const first = text.data();
        const char* const last = first + text.size();
        const auto [end, error] = from_chars(first, last, value);

        if (error != errc{} || end != last)
            throw runtime_error(format("{}: invalid {} value \"{}\".", context, value_kind, text));

        return value;
    }

    template <typename T>
    vector<T> parse_number_list(string_view text, string_view context, char separator = ' ')
    {
        vector<T> values;
        size_t position = 0;

        const auto is_separator = [separator](char character)
        {
            return separator == ' '
                ? isspace(static_cast<unsigned char>(character)) != 0
                : character == separator;
        };

        while (position < text.size())
        {
            while (position < text.size() && is_separator(text[position])) ++position;
            if (position == text.size()) break;

            size_t end = position;
            while (end < text.size() && !is_separator(text[end])) ++end;

            const string_view token = trim_view(text.substr(position, end - position));
            if (!token.empty()) values.push_back(parse_number<T>(token, context));
            position = end;
        }

        return values;
    }

    float parse_float(string_view, string_view);
    int   parse_int  (string_view, string_view);
    long  parse_long (string_view, string_view);

    void replace(string&, const string&, const string&);

    void display_progress_bar(Index, Index);

    float get_elapsed_time(const time_t& beginning_time);

    string get_time(float);

    string get_first_word(const string&);

    template <typename T>
    string vector_to_string(const vector<T>& values, const string& separator = " ")
    {
        ostringstream buffer;

        if constexpr (is_floating_point_v<T>)
            buffer.precision(numeric_limits<T>::max_digits10);

        for (size_t i = 0; i < values.size(); ++i)
        {
            buffer << values[i];
            if (i < values.size() - 1)
                buffer << separator;
        }

        return buffer.str();
    }

    inline string vector_to_string(const vector<string>& values, const string& separator = " ")
    {
        return join_strings(values, separator);
    }

    template <typename Derived>
    inline string vector_to_string(const Eigen::DenseBase<Derived>& values, const string& separator = " ")
    {
        ostringstream buffer;

        using Scalar = typename Derived::Scalar;
        if constexpr (is_floating_point_v<Scalar>)
            buffer.precision(numeric_limits<Scalar>::max_digits10);

        for (Index i = 0; i < values.size(); ++i) buffer << values(i) << separator;
        return buffer.str();
    }

    void string_to_vector(const string&, VectorR&);

    bool contains(const vector<string>&, string_view);
    bool contains(initializer_list<string_view>, string_view);
    bool starts_with_any(string_view, initializer_list<string_view>);
    bool env_flag_enabled(const char*) noexcept;
    // Flag with a default: unset/empty -> default_value; "0"/"false"/"off"/"no"
    // -> false; "1"/"true"/"on"/"yes" -> true; anything else -> default_value.
    bool env_flag_enabled(const char*, bool default_value) noexcept;
    // Integer with a default for unset/empty/unparsable.
    long long env_int_or(const char*, long long default_value) noexcept;

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
