//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

#include <initializer_list>

namespace opennn
{

    vector<string> get_tokens(const string&, const string&);

    vector<string_view> get_token_views(string_view, char);

    // Quote-aware tokenizer: replica la semantica de
    // strip_quotes_and_quoted_separators por linea (borra comillas, elimina ','
    // y ';' dentro de comillas) sin copiar el fichero completo. Para lineas sin
    // '"' devuelve vistas zero-copy sobre `line`; para lineas con '"' escribe los
    // bytes limpios en `scratch` (reutilizable) y devuelve vistas sobre `scratch`.
    // Si file_has_quotes es false, va directo a get_token_views (sin coste extra).
    vector<string_view> get_token_views_maybe_quoted(string_view line, char separator,
                                                     bool file_has_quotes, string& scratch);

    // Igual que get_token_views_maybe_quoted pero rellenando `out` (out.clear() +
    // emplace). Permite reutilizar el mismo vector entre filas y evitar reservar
    // memoria por fila.
    void get_token_views_maybe_quoted(string_view line, char separator, bool file_has_quotes,
                                      string& scratch, vector<string_view>& out);

    // Devuelve SOLO el primer campo de `line`, con la misma semantica de comillas
    // que get_token_views_maybe_quoted(...)[0], sin trocear el resto de la linea.
    // Sin comillas: vista zero-copy sobre `line`; con comillas: escribe el primer
    // campo limpio en `scratch` y devuelve una vista sobre `scratch`.
    string_view first_token_maybe_quoted(string_view line, char separator,
                                         bool file_has_quotes, string& scratch);

    string_view trim_view(string_view);

    vector<string> tokenize(const string&);

    vector<string_view> tokenize_views(string_view);

    vector<string> convert_string_vector(const vector<vector<string>>&, const string&);

    void replace_all_appearances(string&, const string&, const string&);
    void replace_all_word_appearances(string&, const string&, const string&);

    string get_trimmed(const string&);

    float parse_float(const string&, const string&);
    int   parse_int  (const string&, const string&);
    long  parse_long (const string&, const string&);

    void replace(string&, const string&, const string&);

    void display_progress_bar(int, int);

    string get_time(float);

    string get_first_word(const string&);


    template <typename T>
    string vector_to_string(const vector<T>& values, const string& separator = " ")
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
    inline string vector_to_string(const Eigen::DenseBase<Derived>& values, const string& separator = " ")
    {
        ostringstream buffer;
        for (Index i = 0; i < values.size(); ++i) buffer << values(i) << separator;
        return buffer.str();
    }

    void string_to_vector(const string&, VectorR&);

    template <typename T, size_t Rank>
    string tensor_to_string(const TensorR<Rank>& values, const string& separator = " ")
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

    bool contains(const vector<string>&, string_view);
    bool contains(initializer_list<string_view>, string_view);
    bool starts_with_any(string_view, initializer_list<string_view>);
    bool env_flag_enabled(const char*) noexcept;

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
