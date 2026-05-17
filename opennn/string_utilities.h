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

    [[nodiscard]] vector<string> get_tokens(const string&, const string&);

    [[nodiscard]] vector<string_view> get_token_views(string_view, char);

    [[nodiscard]] string_view trim_view(string_view);

    [[nodiscard]] vector<string> tokenize(const string&);

    [[nodiscard]] vector<string_view> tokenize_views(string_view);

    [[nodiscard]] vector<string> convert_string_vector(const vector<vector<string>>&, const string&);

    [[nodiscard]] bool is_numeric_string(string_view);
    [[nodiscard]] bool is_date_time_string(string_view);

    enum DateFormat {AUTO, DMY, MDY, YMD};

    [[nodiscard]] time_t date_to_timestamp(const string&, Index = 0, const DateFormat& format = AUTO);

    void replace_all_appearances(string&, const string&, const string&);
    void replace_all_word_appearances(string&, const string&, const string&);

    [[nodiscard]] string get_trimmed(const string&);

    [[nodiscard]] bool has_numbers(const vector<string>&);
    [[nodiscard]] bool has_numbers(const vector<string_view>&);

    void replace(string&, const string&, const string&);

    void display_progress_bar(int, int);

    [[nodiscard]] string get_time(float);

    [[nodiscard]] string get_first_word(const string&);

    // Vector/tensor string conversion

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

    template <typename Derived>
    [[nodiscard]] inline string vector_to_string(const Eigen::DenseBase<Derived>& values, const string& separator = " ")
    {
        ostringstream buffer;
        for (Index i = 0; i < values.size(); ++i) buffer << values(i) << separator;
        return buffer.str();
    }

    void string_to_vector(const string& input, VectorR& values);

    template <typename T, size_t Rank>
    [[nodiscard]] string tensor_to_string(const TensorR<Rank>& values, const string& separator = " ")
    {
        ostringstream buffer;

        for (Index i = 0; i < values.size(); ++i)
            buffer << values(i) << separator;

        return buffer.str();
    }

    template <typename T, size_t Rank>
    void string_to_tensor(const string& input, TensorR<Rank>& values)
    {
        istringstream stream(input);
        T value;
        Index i = 0;

        while (stream >> value)
            values(i++) = value;
    }

    [[nodiscard]] bool contains(const vector<string>&, const string&);
    [[nodiscard]] bool contains(const vector<string>&, string_view);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
