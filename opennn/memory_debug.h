//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   D E B U G   U T I L I T I E S

#pragma once

#include "opennn_types.h"
#include "configuration.h"

namespace opennn::memory_debug
{

bool enabled();

void reset();

void record(const string&,
            const string&,
            Index,
            const string& note = {});

void print(ostream&);

// Buffer lifecycle tracking. These hooks are intentionally no-ops unless
// OPENNN_MEMORY_DEBUG=1, so normal execution does not retain debug metadata.
void register_buffer(const void* buffer,
                     Device initial_device,
                     source_location location = source_location::current()) noexcept;
void update_buffer(const void* buffer,
                   const void* data,
                   Index bytes,
                   Device device,
                   bool owns) noexcept;
void name_buffer(const void* buffer, string_view name) noexcept;
void unregister_buffer(const void* buffer) noexcept;
void print_buffers(ostream&);

// Backend allocation tracking includes Buffer storage and the few raw CUDA or
// pinned allocations used by inference backends.
void check_allocation_allowed(Device, Index);
void allocation_created(const void* pointer,
                        Device,
                        Index,
                        const string& kind = "backend");
void allocation_released(const void* pointer);
void print_allocations(ostream&);

class ScopedPhase
{
public:
    explicit ScopedPhase(string);
    ~ScopedPhase() noexcept;

    ScopedPhase(const ScopedPhase&) = delete;
    ScopedPhase& operator=(const ScopedPhase&) = delete;

private:
    string previous;
    bool active = false;
};

class AllocationGuard
{
public:
    explicit AllocationGuard(bool enable = true);
    ~AllocationGuard() noexcept;

    AllocationGuard(const AllocationGuard&) = delete;
    AllocationGuard& operator=(const AllocationGuard&) = delete;

private:
    bool active = false;
    bool previous = false;
};

}
