//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   D E B U G   U T I L I T I E S

#include "memory_debug.h"
#include "string_utilities.h"

#include <atomic>

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

const char* device_name(Device device)
{
    switch (device)
    {
    case Device::CPU:  return "CPU";
    case Device::CUDA: return "CUDA";
    case Device::Auto: return "Auto";
    }
    return "Unknown";
}

string short_location(const source_location& location)
{
    const filesystem::path path(location.file_name());
    return format("{}:{}", path.filename().string(), location.line());
}

struct BufferEntry
{
    uint64_t id = 0;
    const void* object = nullptr;
    const void* data = nullptr;
    string origin;
    string name;
    Device device = Device::CPU;
    Index current_bytes = 0;
    Index peak_bytes = 0;
    Index allocation_count = 0;
    Index change_count = 0;
    bool owns = true;
    bool live = true;
};

vector<BufferEntry>& buffer_entries()
{
    static vector<BufferEntry> rows;
    return rows;
}

unordered_map<const void*, size_t>& active_buffers()
{
    static unordered_map<const void*, size_t> active;
    return active;
}

mutex& buffer_mutex()
{
    static mutex m;
    return m;
}

atomic<uint64_t>& next_buffer_id()
{
    static atomic<uint64_t> id{1};
    return id;
}

struct AllocationEntry
{
    const void* pointer = nullptr;
    Device device = Device::CPU;
    Index bytes = 0;
    string kind;
    string phase;
    bool live = true;
};

struct AllocationSummary
{
    Index calls = 0;
    Index bytes = 0;
};

vector<AllocationEntry>& allocation_entries()
{
    static vector<AllocationEntry> rows;
    return rows;
}

unordered_map<const void*, size_t>& active_allocations()
{
    static unordered_map<const void*, size_t> active;
    return active;
}

map<string, AllocationSummary>& allocation_summaries()
{
    static map<string, AllocationSummary> summaries;
    return summaries;
}

mutex& allocation_mutex()
{
    static mutex m;
    return m;
}

string& current_phase()
{
    static thread_local string phase;
    return phase;
}

atomic_bool& allocation_forbidden()
{
    static atomic_bool forbidden{false};
    return forbidden;
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

    ranges::sort(rows, greater<>{}, &Entry::bytes);

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

void register_buffer(const void* buffer,
                     Device initial_device,
                     source_location location) noexcept
{
    if (!enabled() || !buffer) return;

    try
    {
        lock_guard lock(buffer_mutex());
        BufferEntry entry;
        entry.id = next_buffer_id().fetch_add(1, memory_order_relaxed);
        entry.object = buffer;
        entry.origin = short_location(location);
        entry.device = initial_device;
        buffer_entries().push_back(move(entry));
        active_buffers()[buffer] = buffer_entries().size() - 1;
    }
    catch (...)
    {
        // Debug accounting must never alter the behavior of the program.
    }
}

void update_buffer(const void* buffer,
                   const void* data,
                   Index bytes,
                   Device device,
                   bool owns) noexcept
{
    if (!enabled() || !buffer) return;

    try
    {
        lock_guard lock(buffer_mutex());
        const auto it = active_buffers().find(buffer);
        if (it == active_buffers().end()) return;

        BufferEntry& entry = buffer_entries()[it->second];
        if (entry.data != data || entry.current_bytes != bytes
            || entry.device != device || entry.owns != owns)
            ++entry.change_count;
        if (data && entry.data != data) ++entry.allocation_count;

        entry.data = data;
        entry.current_bytes = bytes;
        entry.peak_bytes = max(entry.peak_bytes, bytes);
        entry.device = device;
        entry.owns = owns;
    }
    catch (...)
    {
    }
}

void unregister_buffer(const void* buffer) noexcept
{
    if (!enabled() || !buffer) return;

    try
    {
        lock_guard lock(buffer_mutex());
        const auto it = active_buffers().find(buffer);
        if (it == active_buffers().end()) return;

        BufferEntry& entry = buffer_entries()[it->second];
        entry.data = nullptr;
        entry.current_bytes = 0;
        entry.live = false;
        active_buffers().erase(it);
    }
    catch (...)
    {
    }
}

void name_buffer(const void* buffer, string_view name) noexcept
{
    if (!enabled() || !buffer || name.empty()) return;

    try
    {
        lock_guard lock(buffer_mutex());
        const auto it = active_buffers().find(buffer);
        if (it == active_buffers().end()) return;
        BufferEntry& entry = buffer_entries()[it->second];
        if (entry.name != name) entry.name.assign(name);
    }
    catch (...)
    {
    }
}

void print_buffers(ostream& os)
{
    if (!enabled()) return;

    vector<BufferEntry> rows;
    {
        lock_guard lock(buffer_mutex());
        rows = buffer_entries();
    }

    ranges::sort(rows, [](const BufferEntry& a, const BufferEntry& b)
    {
        if (a.live != b.live) return a.live > b.live;
        if (a.current_bytes != b.current_bytes)
            return a.current_bytes > b.current_bytes;
        if (a.peak_bytes != b.peak_bytes) return a.peak_bytes > b.peak_bytes;
        return a.id < b.id;
    });

    Index live_bytes = 0;
    Index live_count = 0;
    for (const BufferEntry& row : rows)
        if (row.live)
        {
            ++live_count;
            if (row.owns) live_bytes += row.current_bytes;
        }

    os << "[MEMORY_BUFFERS] rows=" << rows.size()
       << " live=" << live_count
       << " live_owned_mib=" << fixed << setprecision(2)
       << double(live_bytes) / (1024.0 * 1024.0) << "\n";
    os << "[MEMORY_BUFFERS] id,name,origin,device,live,owns,current_MiB,peak_MiB,allocations,changes\n";

    for (const BufferEntry& row : rows)
        os << "[MEMORY_BUFFERS] "
           << row.id << ","
           << (row.name.empty() ? row.origin : row.name) << ","
           << row.origin << ","
           << device_name(row.device) << ","
           << (row.live ? 1 : 0) << ","
           << (row.owns ? 1 : 0) << ","
           << fixed << setprecision(4)
           << double(row.current_bytes) / (1024.0 * 1024.0) << ","
           << double(row.peak_bytes) / (1024.0 * 1024.0) << ","
           << row.allocation_count << ","
           << row.change_count << "\n";
}

void check_allocation_allowed(Device device, Index bytes)
{
    if (!enabled() || bytes <= 0) return;

    throw_if(allocation_forbidden().load(memory_order_relaxed),
             "Memory debug: {} allocation of {} bytes attempted during guarded inference phase '{}'.",
             device_name(device), bytes,
             current_phase().empty() ? "unlabelled" : current_phase());
}

void allocation_created(const void* pointer,
                        Device device,
                        Index bytes,
                        const string& kind)
{
    if (!enabled() || !pointer || bytes <= 0) return;

    lock_guard lock(allocation_mutex());
    AllocationEntry entry;
    entry.pointer = pointer;
    entry.device = device;
    entry.bytes = bytes;
    entry.kind = kind;
    entry.phase = current_phase().empty() ? "unlabelled" : current_phase();

    allocation_entries().push_back(entry);
    active_allocations()[pointer] = allocation_entries().size() - 1;

    const string key = entry.phase + "\t" + device_name(device) + "\t" + kind;
    AllocationSummary& summary = allocation_summaries()[key];
    ++summary.calls;
    summary.bytes += bytes;
}

void allocation_released(const void* pointer)
{
    if (!enabled() || !pointer) return;

    lock_guard lock(allocation_mutex());
    const auto it = active_allocations().find(pointer);
    if (it == active_allocations().end()) return;
    allocation_entries()[it->second].live = false;
    active_allocations().erase(it);
}

void print_allocations(ostream& os)
{
    if (!enabled()) return;

    vector<pair<string, AllocationSummary>> summaries;
    Index live_cpu = 0;
    Index live_cuda = 0;
    Index live_cpu_count = 0;
    Index live_cuda_count = 0;
    {
        lock_guard lock(allocation_mutex());
        summaries.assign(allocation_summaries().begin(),
                         allocation_summaries().end());
        for (const AllocationEntry& entry : allocation_entries())
            if (entry.live)
            {
                if (entry.device == Device::CUDA)
                {
                    live_cuda += entry.bytes;
                    ++live_cuda_count;
                }
                else
                {
                    live_cpu += entry.bytes;
                    ++live_cpu_count;
                }
            }
    }

    ranges::sort(summaries, [](const auto& a, const auto& b)
    {
        return a.second.bytes > b.second.bytes;
    });

    os << "[MEMORY_ALLOCATIONS] live_cpu_count=" << live_cpu_count
       << " live_cpu_mib=" << fixed << setprecision(2)
       << double(live_cpu) / (1024.0 * 1024.0)
       << " live_cuda_count=" << live_cuda_count
       << " live_cuda_mib="
       << double(live_cuda) / (1024.0 * 1024.0) << "\n";
    os << "[MEMORY_ALLOCATIONS] phase,device,kind,calls,allocated_MiB\n";

    for (const auto& [key, summary] : summaries)
    {
        string printable = key;
        ranges::replace(printable, '\t', ',');
        os << "[MEMORY_ALLOCATIONS] " << printable << ","
           << summary.calls << ","
           << fixed << setprecision(4)
           << double(summary.bytes) / (1024.0 * 1024.0) << "\n";
    }
}

ScopedPhase::ScopedPhase(string phase)
{
    if (!enabled()) return;

    active = true;
    previous = current_phase();
    current_phase() = previous.empty()
        ? move(phase)
        : previous + "/" + phase;
}

ScopedPhase::~ScopedPhase() noexcept
{
    if (active) current_phase() = move(previous);
}

AllocationGuard::AllocationGuard(bool enable)
    : active(enable && enabled())
{
    if (!active) return;
    previous = allocation_forbidden().exchange(true, memory_order_relaxed);
}

AllocationGuard::~AllocationGuard() noexcept
{
    if (active)
        allocation_forbidden().store(previous, memory_order_relaxed);
}

}
