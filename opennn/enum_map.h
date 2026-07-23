//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E N U M   M A P
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"

namespace opennn
{

template <typename Enum>
struct EnumMap
{
    using Entry = pair<Enum, string>;

    const vector<Entry>& entries;

    const string& to_string(Enum value) const
    {
        for (const auto& [enum_value, name] : entries)
            if (enum_value == value)
                return name;
        throw runtime_error("Unknown enum value");
    }

    Enum from_string(const string& name) const
    {
        for (const auto& [enum_value, entry_name] : entries)
            if (entry_name == name)
                return enum_value;
        throw runtime_error(format("Unknown enum string: {}", name));
    }

    Enum from_string(const string& name, Enum fallback) const
    {
        for (const auto& [enum_value, entry_name] : entries)
            if (entry_name == name)
                return enum_value;
        return fallback;
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
