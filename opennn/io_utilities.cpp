//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "io_utilities.h"
#include "string_utilities.h"

#include <cstdlib>
#include <iostream>

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
             "atomic_rename: rename failed for {} -> {} (errno={}).",
                    source_path.string(), destination_path.string(), errno);
#endif
}

bool has_bom(string_view s)
{
    constexpr string_view bom = "\xEF\xBB\xBF";
    return s.starts_with(bom);
}

}

void download_if_missing(const filesystem::path& path, const string& url)
{
    if (filesystem::exists(path)) return;

    if (path.has_parent_path())
        filesystem::create_directories(path.parent_path());

    cout << "Downloading " << url << " -> " << path.string() << " ..." << endl;

#if defined(_WIN32)
    const string curl = "curl.exe";
#else
    const string curl = "curl";
#endif

    const string command =
        curl + " -L --fail -o \"" + path.string() + "\" \"" + url + "\"";

    if (system(command.c_str()) != 0 || !filesystem::exists(path))
        throw runtime_error("Download failed. Get it manually from:\n  " + url);
}

void download_files_if_missing(const filesystem::path& directory,
                               const string_view base_url,
                               const vector<string_view>& filenames)
{
    for (const string_view filename : filenames)
        download_if_missing(directory / filename, string(base_url) + string(filename));
}

string read_text_file(const filesystem::path& path)
{
    ifstream file(path, ios::binary | ios::ate);

    throw_if(!file.is_open(), "Cannot open file {}", path.string());

    const streamoff byte_count = file.tellg();
    throw_if(byte_count < 0, "Cannot determine size for file {}", path.string());
    throw_if(byte_count > streamoff(numeric_limits<streamsize>::max()),
             "File {} is too large to read into memory", path.string());

    file.seekg(0);

    string contents(size_t(byte_count), '\0');
    if (byte_count > 0)
        file.read(contents.data(), streamsize(byte_count));

    throw_if(!file, "Cannot read file {}", path.string());
    return contents;
}

bool is_file_current(const filesystem::path& file,
                     const vector<filesystem::path>& sources,
                     const uintmax_t expected_size)
{
    error_code error;

    if (!filesystem::exists(file, error) || error) return false;

    if (expected_size > 0
     && (filesystem::file_size(file, error) != expected_size || error))
        return false;

    const auto file_time = filesystem::last_write_time(file, error);
    if (error) return false;

    for (const filesystem::path& source : sources)
    {
        const auto source_time = filesystem::last_write_time(source, error);
        if (error || file_time < source_time) return false;
    }

    return true;
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
                            FILE_SHARE_READ,
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
                 "FileReader::read_at: ReadFile failed (offset={}, n={}).",
                        current_offset, bytes_to_read);
        total += read_bytes;
    }
}

#else

void FileReader::open(const filesystem::path& path)
{
    close();

    fd_ = ::open(path.c_str(), O_RDONLY);
    throw_if(fd_ < 0,
             "FileReader: cannot open {} (errno={}).",
                    path.string(), errno);
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
                 "FileReader::read_at: unexpected EOF at offset {}.",
                        offset + total);
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

void read_int32_batch(const FileReader& reader,
                      const vector<Index>& sample_indices,
                      const Index samples_number,
                      const uint64_t record_values,
                      const Index source_offset,
                      const Index values_number,
                      float* output,
                      const Index output_stride,
                      const Index output_offset,
                      const string_view context)
{
    throw_if(record_values == 0,
             "{} record width must be greater than zero.", context);
    throw_if(source_offset < 0 || values_number < 0
          || (source_offset >= 0 && values_number >= 0
           && uint64_t(source_offset) + uint64_t(values_number) > record_values),
             "{} record range is invalid.", context);
    throw_if(output_offset < 0 || output_stride < 0
          || (output_offset >= 0 && values_number >= 0
           && uint64_t(output_stride) < uint64_t(output_offset) + uint64_t(values_number)),
             "{} output range is invalid.", context);
    throw_if(!sample_indices.empty() && !output,
             "{} output pointer is null.", context);

    string omp_error;

    #pragma omp parallel for
    for (Index i = 0; i < ssize(sample_indices); ++i)
    {
        try
        {
            const Index sample_index = sample_indices[size_t(i)];
            throw_if(sample_index < 0 || sample_index >= samples_number,
                     "{} sample index is out of range.", context);

            thread_local vector<int32_t> buffer;
            buffer.resize(size_t(values_number));

            reader.read_at(buffer.data(),
                           size_t(values_number) * sizeof(int32_t),
                           (uint64_t(sample_index) * record_values
                            + uint64_t(source_offset)) * sizeof(int32_t));

            for (Index j = 0; j < values_number; ++j)
                output[i * output_stride + output_offset + j] = float(buffer[size_t(j)]);
        }
        catch (const exception& exception)
        {
            #pragma omp critical
            {
                if (omp_error.empty()) omp_error = exception.what();
            }
        }
    }

    throw_if(!omp_error.empty(), omp_error);
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
             "FileWriter: cannot open {}", tmp_path.string());
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

    // Camino normal: mmap zero-copy. Las comillas ya NO fuerzan copiar el fichero
    // completo ni una pasada de strip: el tokenizer (get_token_views_maybe_quoted)
    // maneja las comillas por linea sobre el propio mmap.
    if (result.mapping.map(path))
    {
        string_view mapped(result.mapping.data(), result.mapping.size());

        if (has_bom(mapped)) mapped.remove_prefix(3);
        result.content = mapped;
        result.has_quotes = mapped.find('"') != string_view::npos;
        parse(result);
        return result;
    }

    // Fallback (el mmap fallo): leer a buffer, SIN strip. El tokenizer se encarga
    // de las comillas igual que en el camino mmap.
    ifstream input_file(path, ios::binary | ios::ate);

    throw_if(!input_file.is_open(),
             "Cannot open file {}\n", path.string());

    const streamoff file_byte_count = input_file.tellg();
    throw_if(file_byte_count < 0,
             "Cannot determine size for file {}\n", path.string());
    throw_if(file_byte_count > streamoff(numeric_limits<streamsize>::max()),
             "File {} is too large to read into memory\n", path.string());

    input_file.seekg(0);
    throw_if(!input_file.good(),
             "Cannot seek file {}\n", path.string());

    result.buffer.resize(static_cast<size_t>(file_byte_count), '\0');
    if (file_byte_count > 0)
    {
        input_file.read(result.buffer.data(), static_cast<streamsize>(file_byte_count));
        throw_if(!input_file,
                 "Cannot read file {}\n", path.string());
    }

    if (has_bom(result.buffer))
        result.buffer.erase(0, 3);

    result.has_quotes = result.buffer.find('"') != string::npos;

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

struct ParsedDateTime
{
    std::array<int, 4> date{};
    std::array<int, 4> time{};
    size_t date_count = 0;
    size_t time_count = 0;
    size_t first_date_digits = 0;
    size_t last_date_digits = 0;
    bool pm = false;
    bool am = false;
};

static bool parse_fields(string_view text,
                         string_view separators,
                         std::array<int, 4>& values,
                         size_t& count,
                         size_t* first_digits = nullptr,
                         size_t* last_digits = nullptr)
{
    count = 0;
    size_t position = 0;

    while (position < text.size())
    {
        if (count == values.size()) return false;

        const size_t start = position;
        while (position < text.size()
            && isdigit(static_cast<unsigned char>(text[position])))
            ++position;

        if (position == start) return false;

        if (count == 0 && first_digits) *first_digits = position - start;
        if (last_digits) *last_digits = position - start;

        const auto [end, error] =
            from_chars(text.data() + start, text.data() + position, values[count]);
        if (error != errc{} || end != text.data() + position) return false;

        ++count;
        if (position == text.size()) return true;
        if (separators.find(text[position]) == string_view::npos) return false;
        ++position;
    }

    return count > 0;
}

static bool parse_date_time(string_view text, ParsedDateTime& parsed)
{
    text = trim_view(text);
    if (text.empty()) return false;

    if (text.size() >= 3 && text[text.size() - 3] == ' ')
    {
        const string_view suffix = text.substr(text.size() - 2);
        parsed.am = suffix == "AM";
        parsed.pm = suffix == "PM";
        if (parsed.am || parsed.pm)
            text.remove_suffix(3);
    }

    const size_t space = text.find(' ');
    const bool time_only = space == string_view::npos
                        && text.find(':') != string_view::npos;

    if (time_only)
        return parse_fields(text, ":", parsed.time, parsed.time_count)
            && parsed.time_count == 3;

    const string_view date_text =
        space == string_view::npos ? text : text.substr(0, space);

    if (!parse_fields(date_text, "-/.", parsed.date, parsed.date_count,
                      &parsed.first_date_digits, &parsed.last_date_digits)
        || parsed.date_count < 2 || parsed.date_count > 3)
        return false;

    if (space == string_view::npos) return true;

    const string_view time_text = text.substr(space + 1);
    return parse_fields(time_text, ":.", parsed.time, parsed.time_count)
        && parsed.time_count >= 2 && parsed.time_count <= 4;
}

bool is_date_time_string(string_view text)
{
    if (is_numeric_string(text)) return false;
    ParsedDateTime parsed;
    return parse_date_time(text, parsed);
}

bool has_numbers(const vector<string>& string_list)
{
    return ranges::any_of(string_list, is_numeric_string);
}

bool has_numbers(const vector<string_view>& string_list)
{
    return ranges::any_of(string_list, is_numeric_string);
}

DateFormat detect_date_format(string_view text)
{
    ParsedDateTime parsed;
    if (!parse_date_time(text, parsed) || parsed.date_count != 3) return Auto;
    if (parsed.first_date_digits == 4) return Ymd;
    if (parsed.date[0] > 12) return Dmy;
    if (parsed.date[1] > 12) return Mdy;
    return Auto;
}

time_t date_to_timestamp(string_view text, Index gmt, DateFormat format)
{
    ParsedDateTime parsed;
    if (!parse_date_time(text, parsed)) return -1;

    tm time_components{};

    if (parsed.date_count == 0)
    {
        if (format != Auto || parsed.time_count != 3) return -1;
    }
    else if (parsed.first_date_digits == 4)
    {
        if (format != Auto && format != Ymd) return -1;
        time_components.tm_year = parsed.date[0] - 1900;
        time_components.tm_mon = parsed.date[1] - 1;
        time_components.tm_mday = parsed.date_count == 3 ? parsed.date[2] : 1;
    }
    else
    {
        if (parsed.date_count != 3 || format == Ymd || parsed.last_date_digits != 4)
            return -1;

        const bool month_first = format == Mdy
            || (format == Auto && parsed.date[0] <= 12 && parsed.date[1] > 12);
        time_components.tm_mday = month_first ? parsed.date[1] : parsed.date[0];
        time_components.tm_mon = (month_first ? parsed.date[0] : parsed.date[1]) - 1;
        time_components.tm_year = parsed.date[2] - 1900;
    }

    if (parsed.time_count > 0)
    {
        if (parsed.time_count < 2) return -1;

        int hour = parsed.time[0];
        if (parsed.pm && hour < 12) hour += 12;
        if (parsed.am && hour == 12) hour = 0;

        time_components.tm_hour = hour - int(gmt);
        time_components.tm_min = parsed.time[1];
        time_components.tm_sec = parsed.time_count >= 3 ? parsed.time[2] : 0;
    }

    return mktime(&time_components);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
