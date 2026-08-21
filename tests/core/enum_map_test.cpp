#include "tests/pch.h"

#include "opennn/core/enum_map.h"

using namespace opennn;

namespace
{

enum class TestValue { First, Second };

EnumMap<TestValue> make_map()
{
    vector<EnumMap<TestValue>::Entry> entries{
        {TestValue::First, "First"},
        {TestValue::Second, "Second"}
    };
    return EnumMap<TestValue>(std::move(entries));
}

}

TEST(EnumMapTest, OwnsEntries)
{
    const EnumMap<TestValue> map = make_map();

    EXPECT_EQ(map.get_entries().size(), 2);
    EXPECT_EQ(map.to_string(TestValue::First), "First");
    EXPECT_EQ(map.from_string("Second"), TestValue::Second);
    EXPECT_EQ(map.from_string("Unknown", TestValue::First), TestValue::First);
}
