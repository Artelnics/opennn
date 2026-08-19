//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E N U M   M A P
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <initializer_list>

#include "opennn/core/opennn_types.h"

namespace opennn
{

template <typename Enum>
class EnumMap
{
public:
    using Entry = pair<Enum, string>;

    EnumMap(initializer_list<Entry> new_entries)
        : entries(new_entries)
    {
    }

    explicit EnumMap(vector<Entry> new_entries)
        : entries(std::move(new_entries))
    {
    }

    const vector<Entry>& get_entries() const noexcept { return entries; }

    const string& to_string(Enum value) const
    {
        const auto entry = ranges::find(entries, value, &Entry::first);
        throw_if(entry == entries.end(), "Unknown enum value");
        return entry->second;
    }

    Enum from_string(string_view name) const
    {
        const auto entry = ranges::find(entries, name, &Entry::second);
        throw_if(entry == entries.end(), "Unknown enum string: {}", name);
        return entry->first;
    }

    Enum from_string(string_view name, Enum fallback) const
    {
        const auto entry = ranges::find(entries, name, &Entry::second);
        return entry != entries.end() ? entry->first : fallback;
    }

private:
    vector<Entry> entries;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
