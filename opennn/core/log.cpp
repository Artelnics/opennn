// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/log.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>

namespace opennn::logging
{

namespace
{

Level level_from_environment() noexcept
{
    const char* const setting = std::getenv("OPENNN_LOG_LEVEL");
    if (!setting || !*setting) return Level::Info;
    if (!std::strcmp(setting, "silent"))  return Level::Silent;
    if (!std::strcmp(setting, "error"))   return Level::Error;
    if (!std::strcmp(setting, "warning")) return Level::Warning;
    if (!std::strcmp(setting, "info"))    return Level::Info;
    if (!std::strcmp(setting, "debug"))   return Level::Debug;
    return Level::Info;
}

struct State
{
    // Exit diagnostics can run after ordinary function-static destructors.
    std::ios_base::Init streams;
    std::atomic<Level> level{level_from_environment()};
    std::mutex sink_mutex;
    std::mutex stream_mutex;
    std::shared_ptr<const Sink> sink;
};

State& state()
{
    // Keep the small synchronization state alive, but release user captures
    // at shutdown. Later exit diagnostics use the default sink.
    static State* const value = []
    {
        auto* result = new State;
        std::atexit([] { set_sink({}); });
        return result;
    }();
    return *value;
}

void default_sink(const Level level, const std::string_view text)
{
    std::lock_guard lock(state().stream_mutex);
    std::ostream& stream = level <= Level::Warning ? std::cerr : std::cout;
    stream << text;
    stream.flush();
}

}

Level level() noexcept
{
    return state().level.load(std::memory_order_relaxed);
}

void set_level(const Level level) noexcept
{
    state().level.store(level, std::memory_order_relaxed);
}

bool enabled(const Level level) noexcept
{
    return level != Level::Silent && level <= state().level.load(std::memory_order_relaxed);
}

void set_sink(Sink sink)
{
    auto replacement = sink ? std::make_shared<const Sink>(std::move(sink)) : nullptr;
    {
        std::lock_guard lock(state().sink_mutex);
        state().sink.swap(replacement);
    }
    // Destroy the old callback outside the mutex too: its captures can own
    // objects whose destructors log or replace the sink.
}

void write(const Level level, const std::string_view text)
{
    if (!enabled(level) || text.empty()) return;

    static thread_local bool writing = false;
    if (writing) return;
    struct WriteGuard
    {
        bool& active;
        explicit WriteGuard(bool& value) : active(value) { active = true; }
        ~WriteGuard() { active = false; }
    } guard(writing);

    try
    {
        std::shared_ptr<const Sink> sink;
        {
            std::lock_guard lock(state().sink_mutex);
            sink = state().sink;
        }
        if (sink) (*sink)(level, text);
        else      default_sink(level, text);
    }
    catch (...)
    {
        // Logging is best-effort, including during stack unwinding. Do not
        // report a sink failure through the same sink or let it escape Line.
    }
}

// Called by the standalone FlashAttention launch-check shim before abort().
void cuda_launch_error(const char* file, int line, const char* message) noexcept
{
    try { error() << file << ':' << line << ": CUDA error: " << message << '\n'; }
    catch (...) { } // Preserve the caller's fatal-error path if formatting fails.
}

}
