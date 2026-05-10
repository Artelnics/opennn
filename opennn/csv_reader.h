//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C S V   R E A D E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

class CsvReader
{
public:

    struct Config
    {
        char separator = ',';
        function<void(string_view)> line_validator;
    };

    struct Result
    {
        string                      buffer;
        vector<vector<string_view>> rows;
    };

    explicit CsvReader(Config c) : config(std::move(c)) {}

    Result read(const filesystem::path& path) const;
    Result read_string(string_view csv_text) const;

private:

    Config config;

    void parse(Result& out) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
