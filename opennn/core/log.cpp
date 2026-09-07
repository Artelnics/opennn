//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O G

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

std::atomic<Level>& current_level() noexcept
{
    static std::atomic<Level> value{level_from_environment()};
    return value;
}

std::mutex& sink_mutex()
{
    static std::mutex mutex;
    return mutex;
}

std::shared_ptr<const Sink>& current_sink()
{
    static std::shared_ptr<const Sink> sink;
    return sink;
}

void default_sink(const Level level, const std::string_view text)
{
    static std::mutex stream_mutex;
    std::lock_guard lock(stream_mutex);
    std::ostream& stream = level <= Level::Warning ? std::cerr : std::cout;
    stream << text;
    if (level <= Level::Warning) stream.flush();
}

}

Level level() noexcept
{
    return current_level().load(std::memory_order_relaxed);
}

void set_level(const Level level) noexcept
{
    current_level().store(level, std::memory_order_relaxed);
}

bool enabled(const Level level) noexcept
{
    return level != Level::Silent && level <= current_level().load(std::memory_order_relaxed);
}

void set_sink(Sink sink)
{
    auto replacement = sink ? std::make_shared<const Sink>(std::move(sink)) : nullptr;
    {
        std::lock_guard lock(sink_mutex());
        current_sink().swap(replacement);
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
            std::lock_guard lock(sink_mutex());
            sink = current_sink();
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

}
