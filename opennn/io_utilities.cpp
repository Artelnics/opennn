//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I / O   U T I L I T I E S
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

FileMapping::~FileMapping() { reset(); }

FileMapping::FileMapping(FileMapping&& other) noexcept { *this = std::move(other); }

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

#if defined(_WIN32)

FileReader::~FileReader() { close(); }

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

void FileReader::close()
{
    if (handle_ && handle_ != INVALID_HANDLE_VALUE)
    {
        ::CloseHandle(handle_);
        handle_ = nullptr;
    }
}

bool FileReader::is_open() const
{
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
}

void FileReader::read_at(void* buffer, size_t bytes, uint64_t offset) const
{
    throw_if(!is_open(), "FileReader::read_at: file not open.");

    size_t total = 0;
    auto* dst = static_cast<uint8_t*>(buffer);
    while (total < bytes)
    {
        const size_t remaining = bytes - total;
        const DWORD chunk = (remaining > 0x7FFFFFFFULL) ? 0x7FFFFFFFu : DWORD(remaining);

        OVERLAPPED ov{};
        const uint64_t current = offset + total;
        ov.Offset     = DWORD(current & 0xFFFFFFFFULL);
        ov.OffsetHigh = DWORD((current >> 32) & 0xFFFFFFFFULL);

        DWORD read_bytes = 0;
        const BOOL ok = ::ReadFile(handle_, dst + total, chunk, &read_bytes, &ov);
        throw_if(!ok || read_bytes == 0,
                 format("FileReader::read_at: ReadFile failed (offset={}, n={}).",
                        current, chunk));
        total += read_bytes;
    }
}

uint64_t FileReader::file_size() const
{
    throw_if(!is_open(), "FileReader::file_size: file not open.");
    LARGE_INTEGER size{};
    throw_if(!::GetFileSizeEx(handle_, &size),
             "FileReader::file_size: GetFileSizeEx failed.");
    return uint64_t(size.QuadPart);
}

#else  // POSIX

FileReader::~FileReader() { close(); }

void FileReader::open(const filesystem::path& path)
{
    close();

    fd_ = ::open(path.c_str(), O_RDONLY);
    throw_if(fd_ < 0,
             format("FileReader: cannot open {} (errno={}).",
                    path.string(), errno));
}

void FileReader::close()
{
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }
}

bool FileReader::is_open() const { return fd_ >= 0; }

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

uint64_t FileReader::file_size() const
{
    throw_if(!is_open(), "FileReader::file_size: file not open.");
    struct stat st{};
    throw_if(::fstat(fd_, &st) != 0,
             "FileReader::file_size: fstat failed.");
    return uint64_t(st.st_size);
}

#endif


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

bool FileWriter::is_open() const { return stream_.is_open(); }

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
    throw_if(!stream_.good() && !stream_.eof() && stream_.fail(),
             "FileWriter::finish: flush/close failed.");

    atomic_rename(tmp_path_, final_path);
    finalized_ = true;
    tmp_path_.clear();
}


void atomic_rename(const filesystem::path& from, const filesystem::path& to)
{
#if defined(_WIN32)
    if (!::MoveFileExW(from.wstring().c_str(),
                       to.wstring().c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw runtime_error(format("atomic_rename: MoveFileExW failed for {} -> {}",
                                   from.string(), to.string()));
#else
    throw_if(::rename(from.c_str(), to.c_str()) != 0,
             format("atomic_rename: rename failed for {} -> {} (errno={}).",
                    from.string(), to.string(), errno));
#endif
}



namespace
{

void strip_quotes_and_quoted_separators(string& buffer)
{
    if (buffer.find('"') == string::npos) return;

    char* write = buffer.data();
    const char* read = buffer.data();
    const char* end = buffer.data() + buffer.size();
    bool in_quote = false;

    while (read < end)
    {
        const char c = *read++;

        if (c == '"')
            in_quote = !in_quote;
        else if (in_quote && (c == ',' || c == ';'))
            continue;
        else
            *write++ = c;
    }

    buffer.resize(write - buffer.data());
}

}

void CsvReader::parse(Result& out) const
{
    out.separator = config.separator;

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

        if (config.line_validator) config.line_validator(line);

        // Store the whole line; tokens are produced per-line on demand by the
        // consumer so we never hold every row's tokens at once.
        out.lines.push_back(line);
    }
}

static bool has_bom(string_view s)
{
    return s.size() >= 3
        && static_cast<unsigned char>(s[0]) == 0xEF
        && static_cast<unsigned char>(s[1]) == 0xBB
        && static_cast<unsigned char>(s[2]) == 0xBF;
}

CsvReader::Result CsvReader::read(const filesystem::path& path) const
{
    throw_if(path.empty(),
             "Data path is empty.\n");

    Result result;

    // Zero-copy fast path: memory-map the file and view it directly. Only valid
    // when no in-place edits are needed — i.e. the file has no quotes (quoted
    // separators are stripped in place). Quote-free covers all numeric data and
    // is where capacity matters most.
    if (result.mapping.map(path))
    {
        string_view mapped(result.mapping.data(), result.mapping.size());

        if (mapped.find('"') == string_view::npos)
        {
            if (has_bom(mapped)) mapped.remove_prefix(3);
            result.content = mapped;
            parse(result);
            return result;
        }

        // Quoted file: drop the map and fall through to the copy path, which can
        // mutate the buffer to strip quoted separators.
        result.mapping.reset();
    }

    // Fallback: copy the file into a heap buffer we can edit in place.
    ifstream file(path, ios::binary | ios::ate);

    throw_if(!file.is_open(),
             format("Cannot open file {}\n", path.string()));

    const auto file_size = file.tellg();
    file.seekg(0);

    result.buffer.resize(static_cast<size_t>(file_size), '\0');
    if (file_size > 0)
        file.read(result.buffer.data(), file_size);

    if (has_bom(result.buffer))
        result.buffer.erase(0, 3);

    strip_quotes_and_quoted_separators(result.buffer);

    result.content = result.buffer;
    parse(result);
    return result;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
