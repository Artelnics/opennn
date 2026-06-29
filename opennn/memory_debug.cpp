//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   D E B U G   U T I L I T I E S

#include "memory_debug.h"
#include "string_utilities.h"

namespace opennn::memory_debug
{

namespace
{

struct Entry
{
    string category;
    string name;
    string note;
    Index bytes = 0;
    Index count = 0;
};

map<string, Entry>& entries()
{
    static map<string, Entry> e;
    return e;
}

mutex& entries_mutex()
{
    static mutex m;
    return m;
}

string key_for(const string& category, const string& name, const string& note)
{
    return category + "\t" + name + "\t" + note;
}

}

bool enabled()
{
    static const bool on = env_flag_enabled("OPENNN_MEMORY_DEBUG");
    return on;
}

void reset()
{
    lock_guard lock(entries_mutex());
    entries().clear();
}

void record(const string& category,
            const string& name,
            Index bytes,
            const string& note)
{
    if (!enabled() || bytes <= 0) return;

    lock_guard lock(entries_mutex());
    const string key = key_for(category, name, note);
    Entry& entry = entries()[key];
    if (entry.count == 0)
    {
        entry.category = category;
        entry.name = name;
        entry.note = note;
    }

    entry.bytes += bytes;
    ++entry.count;
}

void print(ostream& os)
{
    if (!enabled()) return;

    vector<Entry> rows;
    {
        lock_guard lock(entries_mutex());
        for (const auto& [_, entry] : entries())
            rows.push_back(entry);
    }

    ranges::sort(rows, {}, &Entry::bytes);
    ranges::reverse(rows);

    Index total = 0;
    for (const Entry& row : rows) total += row.bytes;

    os << "[MEMORY_DEBUG] rows=" << rows.size()
       << " total_recorded_mib=" << fixed << setprecision(2)
       << double(total) / (1024.0 * 1024.0) << "\n";
    os << "[MEMORY_DEBUG] category,name,count,MiB,note\n";

    for (const Entry& row : rows)
    {
        os << "[MEMORY_DEBUG] "
           << row.category << ","
           << row.name << ","
           << row.count << ","
           << fixed << setprecision(2) << double(row.bytes) / (1024.0 * 1024.0) << ","
           << row.note << "\n";
    }
}

}

