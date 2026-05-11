//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C S V   R E A D E R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "csv_reader.h"
#include "string_utilities.h"

namespace opennn
{

namespace
{

void strip_quotes_and_quoted_separators(string& buffer)
{
    if (buffer.find('"') == string::npos) return;

    char* write = buffer.data();
    const char* read = buffer.data();
    const char* end = buffer.data() + buffer.size();
    bool in_quote = false;

    while (read < end)
    {
        const char c = *read++;

        if (c == '"')
            in_quote = !in_quote;
        else if (in_quote && (c == ',' || c == ';'))
            continue;
        else
            *write++ = c;
    }

    buffer.resize(write - buffer.data());
}

}

void CsvReader::parse(Result& out) const
{
    string& buffer = out.buffer;

    if (buffer.size() >= 3
        && static_cast<unsigned char>(buffer[0]) == 0xEF
        && static_cast<unsigned char>(buffer[1]) == 0xBB
        && static_cast<unsigned char>(buffer[2]) == 0xBF)
        buffer.erase(0, 3);

    strip_quotes_and_quoted_separators(buffer);

    out.rows.reserve(count(buffer.begin(), buffer.end(), '\n') + 1);

    const string_view buffer_view(buffer);
    size_t line_start = 0;

    while (line_start < buffer_view.size())
    {
        size_t line_end = buffer_view.find('\n', line_start);
        if (line_end == string_view::npos) line_end = buffer_view.size();

        string_view line = buffer_view.substr(line_start, line_end - line_start);
        line_start = line_end + 1;

        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        line = trim_view(line);

        if (line.empty()) continue;

        if (config.line_validator) config.line_validator(line);

        out.rows.push_back(get_token_views(line, config.separator));
    }
}

CsvReader::Result CsvReader::read(const filesystem::path& path) const
{
    if (path.empty())
        throw runtime_error("Data path is empty.\n");

    ifstream file(path, ios::binary | ios::ate);

    if (!file.is_open())
        throw runtime_error("Cannot open file " + path.string() + "\n");

    const auto file_size = file.tellg();
    file.seekg(0);

    Result result;
    result.buffer.resize(static_cast<size_t>(file_size), '\0');
    if (file_size > 0)
        file.read(result.buffer.data(), file_size);

    parse(result);
    return result;
}

CsvReader::Result CsvReader::read_string(string_view csv_text) const
{
    Result result;
    result.buffer.assign(csv_text);
    parse(result);
    return result;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
