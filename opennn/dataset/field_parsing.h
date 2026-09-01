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

#include <utility>

namespace opennn
{

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

    explicit CsvReader(function<void(string_view)> new_line_validator = {})
        : line_validator(std::move(new_line_validator))
    {
    }

    Result read(const filesystem::path&) const;

private:

    function<void(string_view)> line_validator;

    void parse(Result&, string_view) const;
};

struct NumberFormat
{
    char decimal_separator = '.';
    char group_separator = '\0';

    bool is_default() const
    {
        return decimal_separator == '.' && group_separator == '\0';
    }
};

struct NumberFormatVotes
{
    Index point_decimal = 0;
    Index comma_decimal = 0;
    Index point_group = 0;
    Index comma_group = 0;
};

void vote_number_format(string_view, NumberFormatVotes&);
NumberFormat decide_number_format(const NumberFormatVotes&);

string number_format_name(char);
char number_format_separator(const string&, string_view);

bool parse_real(string_view, float&, const NumberFormat& = {});
bool parse_real(string_view, double&, const NumberFormat& = {});

bool is_numeric_string(string_view, const NumberFormat& = {});
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
