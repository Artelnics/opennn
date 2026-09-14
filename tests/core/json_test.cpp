#include "tests/pch.h"

#include "opennn/core/json.h"

namespace opennn
{

TEST(JsonTest, AcceptsStrictNumbers)
{
    for (const std::string_view text : {"0", "-0", "12", "-12.5", "1e3", "1E-3"})
        EXPECT_NO_THROW(Json::parse(text));
}

TEST(JsonTest, RejectsNonstandardNumbers)
{
    for (const std::string_view text : {"01", "-01", "1.", "1e", "1e+", "+1", ".5"})
        EXPECT_THROW(Json::parse(text), std::runtime_error) << text;
}

TEST(JsonTest, RejectsUnescapedControlCharacters)
{
    EXPECT_THROW(Json::parse("\"line\nfeed\""), std::runtime_error);
    EXPECT_THROW(Json::parse(std::string("\"") + char(1) + "\""), std::runtime_error);
    EXPECT_NO_THROW(Json::parse("\"line\\nfeed\""));
}

TEST(JsonTest, RequiresValidUnicodeSurrogatePairs)
{
    EXPECT_NO_THROW(Json::parse("\"\\uD83D\\uDE00\""));
    EXPECT_THROW(Json::parse("\"\\uD83D\""), std::runtime_error);
    EXPECT_THROW(Json::parse("\"\\uDE00\""), std::runtime_error);
    EXPECT_THROW(Json::parse("\"\\uD83D\\u0041\""), std::runtime_error);
}

TEST(JsonTest, RequiresValidUtf8)
{
    EXPECT_NO_THROW(Json::parse("\"Espa\xC3\xB1" "a\""));
    EXPECT_THROW(Json::parse("\"\xC0\xAF\""), std::runtime_error);
    EXPECT_THROW(Json::parse("\"\xE2\x82\""), std::runtime_error);
    EXPECT_THROW(Json::parse("\"\xED\xA0\x80\""), std::runtime_error);
}

TEST(JsonTest, LimitsNestingDepth)
{
    std::string accepted(256, '[');
    accepted += '0';
    accepted.append(256, ']');
    EXPECT_NO_THROW(Json::parse(accepted));

    std::string rejected(257, '[');
    rejected += '0';
    rejected.append(257, ']');
    EXPECT_THROW(Json::parse(rejected), std::runtime_error);
}

}
