//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F I E L D   P A R S I N G   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/io_utilities.h"

namespace opennn
{

// Reading delimited text files and deciding what their fields mean. The type
// detection below is what turns a raw CSV column into a Numeric, Binary or
// DateTime variable, so it lives with the datasets that make that decision
// rather than with the generic file machinery in core/io_utilities.

class CsvReader
{
public:

    struct Result
    {
        FileMapping         mapping;
        string              buffer;
        vector<string_view> lines;
        bool                has_quotes = false;
    };

    explicit CsvReader(char new_separator = ',',
                       function<void(string_view)> new_line_validator = {})
        : separator(new_separator),
          line_validator(move(new_line_validator))
    {
    }

    Result read(const filesystem::path&) const;

private:

    char separator;
    function<void(string_view)> line_validator;

    void parse(Result&, string_view) const;
};

bool is_numeric_string(string_view);
bool is_date_time_string(string_view);

extern const vector<string> positive_words;
extern const vector<string> negative_words;

enum DateFormat {Auto, Dmy, Mdy, Ymd};

DateFormat detect_date_format(string_view);
time_t date_to_timestamp(string_view, Index = 0, DateFormat format = Auto);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
