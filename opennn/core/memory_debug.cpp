//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   D E B U G   U T I L I T I E S

#include "opennn/core/memory_debug.h"
#include "opennn/core/memory_pool.h"
#include "opennn/core/string_utilities.h"

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
    const string key = category + "\t" + name + "\t" + note;
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

void record_pool_lifetimes(const string& category_prefix,
                           const vector<MemoryPoolEntry>& lifetime_entries,
                           const string& timeline_note)
{
    if (!enabled()) return;

    record(category_prefix + ".lifetime_meta", "timeline", Index(1), timeline_note);
    for (size_t i = 0; i < lifetime_entries.size(); ++i)
        record(category_prefix + ".lifetime_entry", format("{}", i),
               lifetime_entries[i].bytes,
               format("first={},last={}",
                      lifetime_entries[i].first_step,
                      lifetime_entries[i].last_step));
}

void print(ostream& os)
{
    if (!enabled()) return;

    vector<Entry> rows;
    {
        lock_guard lock(entries_mutex());
        ranges::copy(entries() | views::values, back_inserter(rows));
    }

    ranges::sort(rows, greater<>{}, &Entry::bytes);

    const Index total = transform_reduce(rows.begin(), rows.end(), Index(0), plus<>{},
                                         [](const Entry& row) { return row.bytes; });

    os << "[MEMORY_DEBUG] rows=" << rows.size()
       << " total_recorded_mib=" << fixed << setprecision(2)
       << double(total) / (1024.0 * 1024.0) << "\n"
       << "[MEMORY_DEBUG] category,name,count,MiB,note\n";

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
