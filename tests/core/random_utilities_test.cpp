#include "tests/pch.h"

#include "opennn/core/random_utilities.h"

using namespace opennn;

TEST(RandomUtilitiesTest, SeedReplaysSequence)
{
    set_seed(42);
    const float first_uniform = random_uniform();
    const Index first_integer = random_integer(1, 100);
    const bool first_boolean = random_bool();

    set_seed(42);
    EXPECT_EQ(random_uniform(), first_uniform);
    EXPECT_EQ(random_integer(1, 100), first_integer);
    EXPECT_EQ(random_bool(), first_boolean);
}
