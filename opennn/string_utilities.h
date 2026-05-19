//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

    /// @brief Splits a string on every occurrence of any character in the separator set.
    /// @param input Source string.
    /// @param separators Set of single-character separators.
    /// @return Tokens, with empty tokens dropped.
    [[nodiscard]] vector<string> get_tokens(const string&, const string&);

    /// @brief Splits a string view on the given separator, returning views into the original buffer.
    [[nodiscard]] vector<string_view> get_token_views(string_view, char);

    /// @brief Returns a view onto the input with leading and trailing whitespace removed.
    [[nodiscard]] string_view trim_view(string_view);

    /// @brief Splits the input on whitespace into individual tokens.
    [[nodiscard]] vector<string> tokenize(const string&);

    /// @brief Whitespace-tokenises a string view, returning views into the source buffer.
    [[nodiscard]] vector<string_view> tokenize_views(string_view);

    /// @brief Joins each inner vector with the separator, returning one flattened string per row.
    [[nodiscard]] vector<string> convert_string_vector(const vector<vector<string>>&, const string&);

    /// @brief Returns true if the string can be parsed as a numeric literal.
    [[nodiscard]] bool is_numeric_string(string_view);

    /// @brief Returns true if the string matches one of the supported date/time formats.
    [[nodiscard]] bool is_date_time_string(string_view);

    /// @brief Order of the day, month, and year fields in a date string (AUTO probes the input).
    enum DateFormat {AUTO, DMY, MDY, YMD};

    /// @brief Parses a date/time string into a Unix timestamp.
    /// @param date_string Date or datetime literal to parse.
    /// @param gmt_offset Number of hours to add when converting to UTC.
    /// @param format Day/month/year ordering, or AUTO to auto-detect.
    [[nodiscard]] time_t date_to_timestamp(const string&, Index = 0, const DateFormat& format = AUTO);

    /// @brief Replaces every occurrence of a substring with another, in place.
    void replace_all_appearances(string&, const string&, const string&);

    /// @brief Replaces every whole-word occurrence of a token with another, in place.
    void replace_all_word_appearances(string&, const string&, const string&);

    /// @brief Returns a copy of the string with leading and trailing whitespace removed.
    [[nodiscard]] string get_trimmed(const string&);

    /// @brief Returns true if any element of the vector parses as a number.
    [[nodiscard]] bool has_numbers(const vector<string>&);
    /// @brief Returns true if any element of the vector parses as a number.
    [[nodiscard]] bool has_numbers(const vector<string_view>&);

    /// @brief In-place replacement of every occurrence of a substring with another.
    void replace(string&, const string&, const string&);

    /// @brief Prints a textual progress bar to stdout for an in-progress operation.
    /// @param progress_index Current step.
    /// @param total Total number of steps.
    void display_progress_bar(int, int);

    /// @brief Formats a duration in seconds as a human-readable HH:MM:SS string.
    [[nodiscard]] string get_time(float);

    /// @brief Returns the first whitespace-delimited word of a string.
    [[nodiscard]] string get_first_word(const string&);

    // Vector/tensor string conversion

    /// @brief Serializes a vector to a string with the given element separator.
    template <typename T>
    [[nodiscard]] string vector_to_string(const vector<T>& values, const string& separator = " ")
    {
        ostringstream buffer;

        for (size_t i = 0; i < values.size(); ++i)
        {
            buffer << values[i];
            if (i < values.size() - 1)
                buffer << separator;
        }

        return buffer.str();
    }

    /// @brief Serializes an Eigen dense expression to a string with the given separator.
    template <typename Derived>
    [[nodiscard]] inline string vector_to_string(const Eigen::DenseBase<Derived>& values, const string& separator = " ")
    {
        ostringstream buffer;
        for (Index i = 0; i < values.size(); ++i) buffer << values(i) << separator;
        return buffer.str();
    }

    /// @brief Parses a whitespace-separated string of floats into a VectorR.
    void string_to_vector(const string& input, VectorR& values);

    /// @brief Serializes a tensor's flat data to a string with the given separator.
    template <typename T, size_t Rank>
    [[nodiscard]] string tensor_to_string(const TensorR<Rank>& values, const string& separator = " ")
    {
        ostringstream buffer;

        for (Index i = 0; i < values.size(); ++i)
            buffer << values(i) << separator;

        return buffer.str();
    }

    /// @brief Parses a whitespace-separated string into the flat storage of a tensor.
    template <typename T, size_t Rank>
    void string_to_tensor(const string& input, TensorR<Rank>& values)
    {
        istringstream stream(input);
        T value;
        Index i = 0;

        while (stream >> value)
            values(i++) = value;
    }

    /// @brief Returns true if the vector contains the given element.
    [[nodiscard]] bool contains(const vector<string>&, const string&);
    /// @brief Returns true if the vector contains the given element.
    [[nodiscard]] bool contains(const vector<string>&, string_view);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
