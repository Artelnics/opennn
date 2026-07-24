#include "pch.h"

#include "opennn/json.h"

using namespace std;
using namespace opennn;

TEST(JsonTest, TypedFieldsRemainTyped)
{
    JsonWriter writer;
    write_json(writer, {
        {"Count", 7},
        {"Ratio", 0.5},
        {"Enabled", true},
        {"Names", json_array(vector<string>{"one", "two"})}
    });

    const Json root = Json::parse(writer.c_str(0));
    EXPECT_TRUE(root.at("Count").is_number());
    EXPECT_TRUE(root.at("Ratio").is_number());
    EXPECT_TRUE(root.at("Enabled").is_bool());
    EXPECT_TRUE(root.at("Names").is_array());
    EXPECT_EQ(read_json_strings(&root, "Names"),
              (vector<string>{"one", "two"}));
}

TEST(JsonTest, StringArrayReaderAcceptsLegacyEncoding)
{
    const Json root = Json::parse(R"({"Names":"one\ntwo"})");
    EXPECT_EQ(read_json_strings(&root, "Names"),
              (vector<string>{"one", "two"}));
}
