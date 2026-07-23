//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "io_utilities.h"
#include "string_utilities.h"

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <sys/mman.h>
    #include <cerrno>
#endif

namespace opennn
{

string read_text_file(const filesystem::path& path)
{
    ifstream file(path, ios::binary | ios::ate);
    throw_if(!file.is_open(), format("Cannot open file {}", path.string()));

    const auto file_size = file.tellg();
    throw_if(file_size < 0, format("Cannot determine file size for {}", path.string()));

    file.seekg(0);
    string buffer(static_cast<size_t>(file_size), '\0');
    if (file_size > 0)
        file.read(buffer.data(), file_size);

    throw_if(!file, format("Cannot read file {}", path.string()));
    return buffer;
}

bool binary_cache_is_valid(const filesystem::path& cache_path,
                           const filesystem::path& data_path,
                           uintmax_t expected_size)
{
    return filesystem::exists(cache_path)
        && filesystem::file_size(cache_path) == expected_size
        && filesystem::last_write_time(cache_path) >= filesystem::last_write_time(data_path);
}

namespace
{

void atomic_rename(const filesystem::path& source_path, const filesystem::path& destination_path)
{
#if defined(_WIN32)
    if (!::MoveFileExW(source_path.wstring().c_str(),
                       destination_path.wstring().c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw runtime_error(format("atomic_rename: MoveFileExW failed for {} -> {} (GetLastError={}).",
                                   source_path.string(), destination_path.string(), ::GetLastError()));
#else
    throw_if(::rename(source_path.c_str(), destination_path.c_str()) != 0,
             format("atomic_rename: rename failed for {} -> {} (errno={}).",
                    source_path.string(), destination_path.string(), errno));
#endif
}

bool has_bom(string_view s)
{
    constexpr string_view bom = "\xEF\xBB\xBF";
    return s.starts_with(bom);
}

}

FileMapping::~FileMapping() { reset(); }

FileMapping::FileMapping(FileMapping&& other) noexcept { *this = move(other); }

FileMapping& FileMapping::operator=(FileMapping&& other) noexcept
{
    if (this != &other)
    {
        reset();
        data_ = other.data_;
        size_ = other.size_;
#if defined(_WIN32)
        file_handle_    = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_    = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

#if defined(_WIN32)

bool FileMapping::map(const filesystem::path& path)
{
    reset();

    file_handle_ = ::CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                                 nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) { file_handle_ = nullptr; return false; }

    LARGE_INTEGER file_size;
    if (!::GetFileSizeEx(file_handle_, &file_size) || file_size.QuadPart == 0)
    {
        reset();
        return false;
    }

    mapping_handle_ = ::CreateFileMappingW(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_handle_) { reset(); return false; }

    void* view = ::MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    if (!view) { reset(); return false; }

    data_ = static_cast<const char*>(view);
    size_ = static_cast<size_t>(file_size.QuadPart);
    return true;
}

void FileMapping::reset()
{
    if (data_) { ::UnmapViewOfFile(const_cast<char*>(data_)); data_ = nullptr; }
    if (mapping_handle_) { ::CloseHandle(mapping_handle_); mapping_handle_ = nullptr; }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_handle_); }
    file_handle_ = nullptr;
    size_ = 0;
}

#else

bool FileMapping::map(const filesystem::path& path)
{
    reset();

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { fd_ = -1; return false; }

    struct stat st;
    if (::fstat(fd_, &st) != 0 || st.st_size == 0) { reset(); return false; }

    void* addr = ::mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr == MAP_FAILED) { reset(); return false; }

    data_ = static_cast<const char*>(addr);
    size_ = size_t(st.st_size);
    return true;
}

void FileMapping::reset()
{
    if (data_) { ::munmap(const_cast<char*>(data_), size_); data_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    size_ = 0;
}

#endif

FileReader::~FileReader() { close(); }

void FileReader::close()
{
#if defined(_WIN32)
    if (handle_ && handle_ != INVALID_HANDLE_VALUE)
    {
        ::CloseHandle(handle_);
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

bool FileReader::is_open() const
{
#if defined(_WIN32)
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
#else
    return fd_ >= 0;
#endif
}

#if defined(_WIN32)

void FileReader::open(const filesystem::path& path)
{
    close();

    handle_ = ::CreateFileW(path.wstring().c_str(),
                            GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL,
                            nullptr);

    if (handle_ == INVALID_HANDLE_VALUE)
    {
        handle_ = nullptr;
        throw runtime_error(format("FileReader: cannot open {}", path.string()));
    }
}

void FileReader::read_at(void* buffer, size_t bytes, uint64_t offset) const
{
    throw_if(!is_open(), "FileReader::read_at: file not open.");

    size_t total = 0;
    auto* dst = static_cast<uint8_t*>(buffer);
    while (total < bytes)
    {
        const uint64_t current_offset = offset + total;
        const DWORD bytes_to_read = DWORD(min(bytes - total, size_t(0x7FFFFFFF)));

        OVERLAPPED overlapped{};
        overlapped.Offset     = DWORD(current_offset);
        overlapped.OffsetHigh = DWORD(current_offset >> 32);

        DWORD read_bytes = 0;
        const BOOL ok = ::ReadFile(handle_, dst + total, bytes_to_read, &read_bytes, &overlapped);
        throw_if(!ok || read_bytes == 0,
                 format("FileReader::read_at: ReadFile failed (offset={}, n={}).",
                        current_offset, bytes_to_read));
        total += read_bytes;
    }
}

#else

void FileReader::open(const filesystem::path& path)
{
    close();

    fd_ = ::open(path.c_str(), O_RDONLY);
    throw_if(fd_ < 0,
             format("FileReader: cannot open {} (errno={}).",
                    path.string(), errno));
}

void FileReader::read_at(void* buffer, size_t bytes, uint64_t offset) const
{
    throw_if(!is_open(), "FileReader::read_at: file not open.");

    size_t total = 0;
    auto* dst = static_cast<uint8_t*>(buffer);
    while (total < bytes)
    {
        const ssize_t n = ::pread(fd_, dst + total, bytes - total, off_t(offset + total));
        if (n < 0)
        {
            if (errno == EINTR) continue;
            throw runtime_error(format("FileReader::read_at: pread failed (errno={}, offset={}).",
                                       errno, offset + total));
        }
        throw_if(n == 0,
                 format("FileReader::read_at: unexpected EOF at offset {}.",
                        offset + total));
        total += size_t(n);
    }
}

#endif

uint64_t FileReader::file_size() const
{
    throw_if(!is_open(), "FileReader::file_size: file not open.");
#if defined(_WIN32)
    LARGE_INTEGER size{};
    throw_if(!::GetFileSizeEx(handle_, &size),
             "FileReader::file_size: GetFileSizeEx failed.");
    return uint64_t(size.QuadPart);
#else
    struct stat st{};
    throw_if(::fstat(fd_, &st) != 0,
             "FileReader::file_size: fstat failed.");
    return uint64_t(st.st_size);
#endif
}


FileWriter::~FileWriter()
{
    if (stream_.is_open()) stream_.close();
    if (!finalized_ && !tmp_path_.empty())
    {
        error_code ec;
        filesystem::remove(tmp_path_, ec);
    }
}

void FileWriter::open(const filesystem::path& tmp_path)
{
    tmp_path_ = tmp_path;
    finalized_ = false;

    filesystem::create_directories(tmp_path.parent_path());

    stream_.open(tmp_path, ios::binary | ios::trunc);
    throw_if(!stream_.is_open(),
             format("FileWriter: cannot open {}", tmp_path.string()));
}

void FileWriter::write(const void* buffer, size_t bytes)
{
    throw_if(!stream_.is_open(), "FileWriter::write: not open.");
    stream_.write(reinterpret_cast<const char*>(buffer), streamsize(bytes));
    throw_if(!stream_.good(), "FileWriter::write: stream error.");
}

void FileWriter::finish_with_rename(const filesystem::path& final_path)
{
    throw_if(!stream_.is_open(), "FileWriter::finish: not open.");
    stream_.flush();
    stream_.close();
    throw_if(stream_.fail(), "FileWriter::finish: flush/close failed.");

    atomic_rename(tmp_path_, final_path);
    finalized_ = true;
    tmp_path_.clear();
}

void CsvReader::parse(Result& out) const
{
    out.separator = configuration.separator;

    const string_view content = out.content;
    out.lines.reserve(ranges::count(content, '\n') + 1);

    size_t line_start = 0;

    while (line_start < content.size())
    {
        size_t line_end = content.find('\n', line_start);
        if (line_end == string_view::npos) line_end = content.size();

        string_view line = content.substr(line_start, line_end - line_start);
        line_start = line_end + 1;

        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        line = trim_view(line);

        if (line.empty()) continue;

        if (configuration.line_validator) configuration.line_validator(line);

        out.lines.push_back(line);
    }
}

CsvReader::Result CsvReader::read(const filesystem::path& path) const
{
    throw_if(path.empty(),
             "Data path is empty.\n");

    Result result;

    auto normalize_quotes = [this](string_view source)
    {
        string normalized;
        normalized.reserve(source.size());

        bool quoted = false;
        for (const char character : source)
        {
            if (character == '"')
            {
                quoted = !quoted;
                continue;
            }

            if (quoted && (character == ',' || character == ';' || character == '\t'))
                continue;

            normalized.push_back(character);
        }

        return normalized;
    };

    // Keep the common unquoted case zero-copy. Quoted input is normalized into
    // owned storage so Result::lines can still be string_views with stable lifetime.
    if (result.mapping.map(path))
    {
        string_view mapped(result.mapping.data(), result.mapping.size());

        if (has_bom(mapped)) mapped.remove_prefix(3);
        result.has_quotes = mapped.find('"') != string_view::npos;
        if (result.has_quotes)
        {
            result.buffer = normalize_quotes(mapped);
            result.mapping.reset();
            result.content = result.buffer;
        }
        else
            result.content = mapped;
        parse(result);
        return result;
    }

    ifstream input_file(path, ios::binary | ios::ate);

    throw_if(!input_file.is_open(),
             format("Cannot open file {}\n", path.string()));

    const streamoff file_byte_count = input_file.tellg();
    throw_if(file_byte_count < 0,
             format("Cannot determine size for file {}\n", path.string()));
    throw_if(file_byte_count > streamoff(numeric_limits<streamsize>::max()),
             format("File {} is too large to read into memory\n", path.string()));

    input_file.seekg(0);
    throw_if(!input_file.good(),
             format("Cannot seek file {}\n", path.string()));

    result.buffer.resize(static_cast<size_t>(file_byte_count), '\0');
    if (file_byte_count > 0)
    {
        input_file.read(result.buffer.data(), static_cast<streamsize>(file_byte_count));
        throw_if(!input_file,
                 format("Cannot read file {}\n", path.string()));
    }

    if (has_bom(result.buffer))
        result.buffer.erase(0, 3);

    result.has_quotes = result.buffer.find('"') != string::npos;
    if (result.has_quotes)
        result.buffer = normalize_quotes(result.buffer);

    result.content = result.buffer;
    parse(result);
    return result;
}

const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};

bool is_numeric_string(string_view text)
{
    if (text.empty()) return false;

    double value;
    const char* first = text.data();
    const char* last  = first + text.size();
    auto [ptr, ec] = from_chars(first, last, value);

    if (ec != errc{} || ptr == first) return false;

    const size_t consumed = static_cast<size_t>(ptr - first);

    return consumed == text.size()
        || (text.find('%') != string_view::npos && consumed + 1 == text.size());
}

static const regex re_ymd_hms_ms(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})\.(\d+))");
static const regex re_ymd_hms(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2}))");
static const regex re_ymd_hm(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}))");
static const regex re_ymd(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}))");
static const regex re_ym(R"((\d{4})[-/.](\d{1,2}))");
static const regex re_dmy_hms(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{1,2}):(\d{1,2})((?: ([AP]M))?)?)");
static const regex re_dmy_hm(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{1,2}))");
static const regex re_dmy(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}))");
static const regex re_hms(R"((\d{1,2}):(\d{1,2}):(\d{1,2}))");

bool is_date_time_string(string_view text)
{
    if (is_numeric_string(text))
        return false;

    const auto matches = [&](const regex& date_regex)
    {
        return regex_match(text.begin(), text.end(), date_regex);
    };

    return matches(re_ymd_hms_ms) || matches(re_ymd_hms) || matches(re_ymd_hm)
        || matches(re_ymd) || matches(re_ym)
        || matches(re_dmy_hms) || matches(re_dmy_hm) || matches(re_dmy)
        || matches(re_hms);
}

bool has_numbers(const vector<string>& string_list)
{
    return ranges::any_of(string_list, is_numeric_string);
}

bool has_numbers(const vector<string_view>& string_list)
{
    return ranges::any_of(string_list, is_numeric_string);
}

time_t date_to_timestamp(const string& date, Index gmt, const DateFormat& format)
{
    struct tm time_components = {};
    smatch match;

    const bool try_ymd = (format == Ymd || format == Auto);
    const bool try_dmy = (format == Dmy || format == Mdy || format == Auto);

    auto fill_dmy = [&](int part1, int part2)
    {
        const bool mdy = (format == Mdy) || (format == Auto && part1 <= 12 && part2 > 12);
        if (mdy) { time_components.tm_mon = part1 - 1; time_components.tm_mday = part2; }
        else    { time_components.tm_mday = part1;    time_components.tm_mon = part2 - 1; }
    };

    if (try_ymd && (regex_match(date, match, re_ymd_hms_ms) || regex_match(date, match, re_ymd_hms)))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = stoi(match[3]);
        time_components.tm_hour = stoi(match[4]) - gmt;
        time_components.tm_min  = stoi(match[5]);
        time_components.tm_sec  = stoi(match[6]);
        return mktime(&time_components);
    }
    if (try_ymd && regex_match(date, match, re_ymd_hm))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = stoi(match[3]);
        time_components.tm_hour = stoi(match[4]) - gmt;
        time_components.tm_min  = stoi(match[5]);
        return mktime(&time_components);
    }
    if (try_ymd && regex_match(date, match, re_ymd))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = stoi(match[3]);
        return mktime(&time_components);
    }
    if (try_ymd && regex_match(date, match, re_ym))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = 1;
        return mktime(&time_components);
    }
    if (try_dmy && regex_match(date, match, re_dmy_hms))
    {
        fill_dmy(stoi(match[1]), stoi(match[2]));
        time_components.tm_year = stoi(match[3]) - 1900;

        int hour = stoi(match[4]);
        if (match[8].matched)
        {
            const string ampm = match[8].str();
            if (ampm == "PM" && hour < 12) hour += 12;
            if (ampm == "AM" && hour == 12) hour = 0;
        }

        time_components.tm_hour = hour - gmt;
        time_components.tm_min  = stoi(match[5]);
        time_components.tm_sec  = stoi(match[6]);
        return mktime(&time_components);
    }
    if (try_dmy && regex_match(date, match, re_dmy_hm))
    {
        fill_dmy(stoi(match[1]), stoi(match[2]));
        time_components.tm_year = stoi(match[3]) - 1900;
        time_components.tm_hour = stoi(match[4]) - gmt;
        time_components.tm_min  = stoi(match[5]);
        return mktime(&time_components);
    }
    if (try_dmy && regex_match(date, match, re_dmy))
    {
        fill_dmy(stoi(match[1]), stoi(match[2]));
        time_components.tm_year = stoi(match[3]) - 1900;
        return mktime(&time_components);
    }
    if (format == Auto && regex_match(date, match, re_hms))
    {
        time_components.tm_hour = stoi(match[1]) - gmt;
        time_components.tm_min  = stoi(match[2]);
        time_components.tm_sec  = stoi(match[3]);
        return mktime(&time_components);
    }

    return -1;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
