//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O G   H E A D E R

#pragma once

#include <functional>
#include <ostream>
#include <sstream>
#include <string_view>

namespace opennn::logging
{

// Everything the library says goes through here: training progress,
// warnings, backend diagnostics. By default it looks exactly as it always
// has -- information on standard output, warnings and errors on standard
// error -- and the two things a library must offer besides that are a way
// to silence it and a way to redirect it.
//
//   logging::set_level(logging::Level::Silent);           // a library that says nothing
//   logging::set_sink([](logging::Level, std::string_view text) { my_logger(text); });
//   OPENNN_LOG_LEVEL=warning                       // the environment sets the default
//
// Text is passed to the sink verbatim, newlines included, so redirecting it
// changes where it goes and not what it says.
enum class Level { Silent = 0, Error = 1, Warning = 2, Info = 3, Debug = 4 };

Level level() noexcept;
void set_level(Level) noexcept;
bool enabled(Level) noexcept;

using Sink = std::function<void(Level, std::string_view)>;
// Callbacks may run concurrently and must synchronize their own state. Replacing
// a sink affects subsequent writes; an in-flight callback keeps its sink alive.
// Recursive logging from a callback is discarded, and callback exceptions are
// contained so diagnostics cannot terminate the application.
// User captures are released at process shutdown; later exit diagnostics use
// the default sink, retaining the configured level. The default sink flushes
// each message, including interactive prompts and progress updates.
void set_sink(Sink);          // an empty sink restores the default streams
void write(Level, std::string_view);

// A line under construction: `logging::info() << "Epoch " << epoch << "\n";`.
// Whatever is streamed into it is handed to the sink when the temporary
// dies, at the end of the full expression, or dropped entirely when the level
// is off -- the formatting work is skipped too, since the stream is a no-op.
class Line
{
public:
    explicit Line(Level level) noexcept : level_(level), on_(enabled(level)) {}
    Line(const Line&) = delete;
    Line& operator=(const Line&) = delete;
    ~Line() noexcept { if (on_) write(level_, buffer_.view()); }

    template<typename T>
    Line& operator<<(const T& value)
    {
        if (on_) buffer_ << value;
        return *this;
    }

    // Manipulators: std::endl, std::fixed, std::setprecision(...) and the like.
    Line& operator<<(std::ostream& (*manipulator)(std::ostream&))
    {
        if (on_) buffer_ << manipulator;
        return *this;
    }

private:
    Level level_;
    bool on_;
    std::ostringstream buffer_;
};

inline Line error()   { return Line(Level::Error); }
inline Line warning() { return Line(Level::Warning); }
inline Line info()    { return Line(Level::Info); }
inline Line debug()   { return Line(Level::Debug); }

}
