#include "tests/pch.h"

#include "opennn/core/configuration.h"

using namespace opennn;

TEST(ConfigurationTest, EffectiveConfigDefaultsToCpuFp32)
{
    const EffectiveConfig config;

    EXPECT_EQ(config.device, Device::CPU);
    EXPECT_EQ(config.training_type, Type::FP32);
    EXPECT_EQ(config.generation, 0U);
}

TEST(ConfigurationTest, ResolveForCpuReturnsIndependentValue)
{
    Configuration& configuration = Configuration::instance();
    configuration.set(Device::Auto, Type::Auto);

    const EffectiveConfig config = configuration.resolve_for(Device::CPU);

    EXPECT_EQ(config.device, Device::CPU);
    EXPECT_EQ(config.training_type, Type::FP32);
    EXPECT_EQ(config.generation, configuration.get_generation());
}
