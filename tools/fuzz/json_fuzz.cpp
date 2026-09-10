#include "opennn/core/json.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size)
{
    try
    {
        const opennn::Json value = opennn::Json::parse(
            std::string_view(reinterpret_cast<const char*>(data), size));
        const std::string serialized = value.dump(0);
        (void)opennn::Json::parse(serialized);
    }
    catch (const std::exception&)
    {
    }
    return 0;
}
