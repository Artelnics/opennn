#include "pch.h"

#include "../opennn/string_utilities.h"

using namespace opennn;

TEST(StringUtilitiesTest, GetTrimmed)
{
    EXPECT_EQ(get_trimmed("   hello   "), "hello");
    EXPECT_EQ(get_trimmed("\t\n hello world \r\n"), "hello world");
    EXPECT_EQ(get_trimmed("nopadding"), "nopadding");
    EXPECT_EQ(get_trimmed(""), "");
    EXPECT_EQ(get_trimmed("     "), "");
    EXPECT_EQ(get_trimmed("  a  "), "a");
    EXPECT_EQ(get_trimmed("a b"), "a b");
}

TEST(StringUtilitiesTest, TrimView)
{
    EXPECT_EQ(trim_view("   hello   "), "hello");
    EXPECT_EQ(trim_view("\t\n value \r"), "value");
    EXPECT_EQ(trim_view("plain"), "plain");
    EXPECT_EQ(trim_view(""), "");
    EXPECT_EQ(trim_view("    "), "");
    EXPECT_EQ(trim_view("x"), "x");
}

TEST(StringUtilitiesTest, GetTokens)
{
    const vector<string> tokens = get_tokens("a,b,c", ",");
    ASSERT_EQ(tokens.size(), size_t(3));
    EXPECT_EQ(tokens[0], "a");
    EXPECT_EQ(tokens[1], "b");
    EXPECT_EQ(tokens[2], "c");

    const vector<string> leading = get_tokens(",a,b", ",");
    ASSERT_EQ(leading.size(), size_t(3));
    EXPECT_EQ(leading[0], "");
    EXPECT_EQ(leading[1], "a");
    EXPECT_EQ(leading[2], "b");

    const vector<string> trailing = get_tokens("a,b,", ",");
    ASSERT_EQ(trailing.size(), size_t(3));
    EXPECT_EQ(trailing[0], "a");
    EXPECT_EQ(trailing[1], "b");
    EXPECT_EQ(trailing[2], "");

    const vector<string> multichar = get_tokens("a::b::c", "::");
    ASSERT_EQ(multichar.size(), size_t(3));
    EXPECT_EQ(multichar[0], "a");
    EXPECT_EQ(multichar[1], "b");
    EXPECT_EQ(multichar[2], "c");

    const vector<string> empty_sep = get_tokens("whole string", "");
    ASSERT_EQ(empty_sep.size(), size_t(1));
    EXPECT_EQ(empty_sep[0], "whole string");

    const vector<string> none = get_tokens("noseparator", ",");
    ASSERT_EQ(none.size(), size_t(1));
    EXPECT_EQ(none[0], "noseparator");
}

TEST(StringUtilitiesTest, GetTokenViews)
{
    const vector<string_view> tokens = get_token_views("a,b,c", ',');
    ASSERT_EQ(tokens.size(), size_t(3));
    EXPECT_EQ(tokens[0], "a");
    EXPECT_EQ(tokens[1], "b");
    EXPECT_EQ(tokens[2], "c");

    const vector<string_view> trailing = get_token_views("a,b,", ',');
    ASSERT_EQ(trailing.size(), size_t(3));
    EXPECT_EQ(trailing[2], "");

    const vector<string_view> single = get_token_views("noseparator", ',');
    ASSERT_EQ(single.size(), size_t(1));
    EXPECT_EQ(single[0], "noseparator");

    const vector<string_view> empty = get_token_views("", ',');
    ASSERT_EQ(empty.size(), size_t(1));
    EXPECT_EQ(empty[0], "");
}

TEST(StringUtilitiesTest, Tokenize)
{
    const vector<string> tokens = tokenize("Hello, World!");
    ASSERT_EQ(tokens.size(), size_t(4));
    EXPECT_EQ(tokens[0], "hello");
    EXPECT_EQ(tokens[1], ",");
    EXPECT_EQ(tokens[2], "world");
    EXPECT_EQ(tokens[3], "!");

    const vector<string> spaced = tokenize("  one   two  ");
    ASSERT_EQ(spaced.size(), size_t(2));
    EXPECT_EQ(spaced[0], "one");
    EXPECT_EQ(spaced[1], "two");

    const vector<string> alnum = tokenize("abc123");
    ASSERT_EQ(alnum.size(), size_t(1));
    EXPECT_EQ(alnum[0], "abc123");

    EXPECT_TRUE(tokenize("").empty());
    EXPECT_TRUE(tokenize("     ").empty());
}

TEST(StringUtilitiesTest, TokenizeViews)
{
    const vector<string_view> tokens = tokenize_views("Hi, there!");
    ASSERT_EQ(tokens.size(), size_t(4));
    EXPECT_EQ(tokens[0], "Hi");
    EXPECT_EQ(tokens[1], ",");
    EXPECT_EQ(tokens[2], "there");
    EXPECT_EQ(tokens[3], "!");

    const vector<string_view> alnum = tokenize_views("word42");
    ASSERT_EQ(alnum.size(), size_t(1));
    EXPECT_EQ(alnum[0], "word42");

    EXPECT_TRUE(tokenize_views("").empty());
    EXPECT_TRUE(tokenize_views("   ").empty());
}

TEST(StringUtilitiesTest, ConvertStringVector)
{
    const vector<vector<string>> input = {{"a", "b", "c"}, {"x", "y"}, {}};
    const vector<string> result = convert_string_vector(input, ",");

    ASSERT_EQ(result.size(), size_t(3));
    EXPECT_EQ(result[0], "a,b,c");
    EXPECT_EQ(result[1], "x,y");
    EXPECT_EQ(result[2], "");

    const vector<vector<string>> single = {{"only"}};
    const vector<string> single_result = convert_string_vector(single, "-");
    ASSERT_EQ(single_result.size(), size_t(1));
    EXPECT_EQ(single_result[0], "only");

    EXPECT_TRUE(convert_string_vector({}, ",").empty());
}

TEST(StringUtilitiesTest, ReplaceAllAppearances)
{
    string text = "ababab";
    replace_all_appearances(text, "a", "X");
    EXPECT_EQ(text, "XbXbXb");

    string overlap = "aaaa";
    replace_all_appearances(overlap, "aa", "b");
    EXPECT_EQ(overlap, "bb");

    string none = "hello";
    replace_all_appearances(none, "z", "Y");
    EXPECT_EQ(none, "hello");

    string empty_pattern = "hello";
    replace_all_appearances(empty_pattern, "", "Y");
    EXPECT_EQ(empty_pattern, "hello");

    string underscore_guard = "foo_bar bar";
    replace_all_appearances(underscore_guard, "bar", "X");
    EXPECT_EQ(underscore_guard, "foo_bar X");
}

TEST(StringUtilitiesTest, ReplaceAllWordAppearances)
{
    string text = "cat category cat";
    replace_all_word_appearances(text, "cat", "dog");
    EXPECT_EQ(text, "dog category dog");

    string boundary = "a-cat-a";
    replace_all_word_appearances(boundary, "cat", "dog");
    EXPECT_EQ(boundary, "a-dog-a");

    string underscore = "my_cat cat";
    replace_all_word_appearances(underscore, "cat", "dog");
    EXPECT_EQ(underscore, "my_cat dog");

    string empty_pattern = "cat";
    replace_all_word_appearances(empty_pattern, "", "dog");
    EXPECT_EQ(empty_pattern, "cat");

    string whole = "cat";
    replace_all_word_appearances(whole, "cat", "dog");
    EXPECT_EQ(whole, "dog");
}

TEST(StringUtilitiesTest, Replace)
{
    string text = "ababab";
    replace(text, "a", "XX");
    EXPECT_EQ(text, "XXbXXbXXb");

    string grow = "aa";
    replace(grow, "a", "aa");
    EXPECT_EQ(grow, "aaaa");

    string none = "hello";
    replace(none, "z", "Y");
    EXPECT_EQ(none, "hello");

    string empty_pattern = "hello";
    replace(empty_pattern, "", "Y");
    EXPECT_EQ(empty_pattern, "hello");

    string shrink = "a--b--c";
    replace(shrink, "--", "-");
    EXPECT_EQ(shrink, "a-b-c");
}

TEST(StringUtilitiesTest, ParseFloat)
{
    EXPECT_NEAR(parse_float("3.5", "ctx"), 3.5f, 1e-6f);
    EXPECT_NEAR(parse_float("-2.0", "ctx"), -2.0f, 1e-6f);
    EXPECT_NEAR(parse_float("0", "ctx"), 0.0f, 1e-6f);

    EXPECT_THROW(parse_float("abc", "ctx"), runtime_error);
    EXPECT_THROW(parse_float("", "ctx"), runtime_error);
}

TEST(StringUtilitiesTest, ParseInt)
{
    EXPECT_EQ(parse_int("42", "ctx"), 42);
    EXPECT_EQ(parse_int("-7", "ctx"), -7);
    EXPECT_EQ(parse_int("0", "ctx"), 0);

    EXPECT_THROW(parse_int("abc", "ctx"), runtime_error);
    EXPECT_THROW(parse_int("", "ctx"), runtime_error);
}

TEST(StringUtilitiesTest, ParseLong)
{
    EXPECT_EQ(parse_long("1000000", "ctx"), 1000000L);
    EXPECT_EQ(parse_long("-123", "ctx"), -123L);

    EXPECT_THROW(parse_long("xyz", "ctx"), runtime_error);
    EXPECT_THROW(parse_long("", "ctx"), runtime_error);
}

TEST(StringUtilitiesTest, GetFirstWord)
{
    EXPECT_EQ(get_first_word("hello world"), "hello");
    EXPECT_EQ(get_first_word("key=value"), "key");
    EXPECT_EQ(get_first_word("name = value"), "name");
    EXPECT_EQ(get_first_word("single"), "single");
    EXPECT_EQ(get_first_word(""), "");
}

TEST(StringUtilitiesTest, GetTime)
{
    EXPECT_EQ(get_time(0.0f), "00:00:00");
    EXPECT_EQ(get_time(59.0f), "00:00:59");
    EXPECT_EQ(get_time(60.0f), "00:01:00");
    EXPECT_EQ(get_time(3661.0f), "01:01:01");
    EXPECT_EQ(get_time(3600.0f), "01:00:00");
}

TEST(StringUtilitiesTest, VectorToString)
{
    const vector<int> values = {1, 2, 3};
    EXPECT_EQ(vector_to_string(values), "1 2 3");
    EXPECT_EQ(vector_to_string(values, ","), "1,2,3");

    const vector<string> words = {"a", "b"};
    EXPECT_EQ(vector_to_string(words, "-"), "a-b");

    const vector<int> single = {5};
    EXPECT_EQ(vector_to_string(single), "5");
}

TEST(StringUtilitiesTest, StringToVector)
{
    VectorR values;
    string_to_vector("1 2 3 4", values);

    ASSERT_EQ(values.size(), Index(4));
    EXPECT_NEAR(values(0), 1.0f, 1e-6f);
    EXPECT_NEAR(values(1), 2.0f, 1e-6f);
    EXPECT_NEAR(values(2), 3.0f, 1e-6f);
    EXPECT_NEAR(values(3), 4.0f, 1e-6f);

    VectorR empty_values;
    string_to_vector("", empty_values);
    EXPECT_EQ(empty_values.size(), Index(0));

    VectorR decimals;
    string_to_vector("0.5 -1.5", decimals);
    ASSERT_EQ(decimals.size(), Index(2));
    EXPECT_NEAR(decimals(0), 0.5f, 1e-6f);
    EXPECT_NEAR(decimals(1), -1.5f, 1e-6f);
}

TEST(StringUtilitiesTest, ContainsVector)
{
    const vector<string> data = {"alpha", "beta", "gamma"};
    EXPECT_TRUE(contains(data, "beta"));
    EXPECT_FALSE(contains(data, "delta"));
    EXPECT_FALSE(contains(vector<string>{}, "x"));
}

TEST(StringUtilitiesTest, ContainsInitializerList)
{
    EXPECT_TRUE(contains({"one", "two", "three"}, "two"));
    EXPECT_FALSE(contains({"one", "two"}, "four"));
}

TEST(StringUtilitiesTest, StartsWithAny)
{
    EXPECT_TRUE(starts_with_any("hello world", {"foo", "hello"}));
    EXPECT_TRUE(starts_with_any("abcdef", {"abc"}));
    EXPECT_FALSE(starts_with_any("hello", {"foo", "bar"}));
    EXPECT_FALSE(starts_with_any("hi", {}));
}

TEST(StringUtilitiesTest, EnvFlagEnabled)
{
    setenv("OPENNN_TEST_FLAG_ON", "1", 1);
    EXPECT_TRUE(env_flag_enabled("OPENNN_TEST_FLAG_ON"));

    setenv("OPENNN_TEST_FLAG_ON", "TRUE", 1);
    EXPECT_TRUE(env_flag_enabled("OPENNN_TEST_FLAG_ON"));

    setenv("OPENNN_TEST_FLAG_ON", "Yes", 1);
    EXPECT_TRUE(env_flag_enabled("OPENNN_TEST_FLAG_ON"));

    setenv("OPENNN_TEST_FLAG_OFF", "0", 1);
    EXPECT_FALSE(env_flag_enabled("OPENNN_TEST_FLAG_OFF"));

    setenv("OPENNN_TEST_FLAG_OFF", "no", 1);
    EXPECT_FALSE(env_flag_enabled("OPENNN_TEST_FLAG_OFF"));

    unsetenv("OPENNN_TEST_FLAG_MISSING");
    EXPECT_FALSE(env_flag_enabled("OPENNN_TEST_FLAG_MISSING"));
}
