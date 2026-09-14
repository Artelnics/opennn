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

TEST(ConfigurationTest, PrecisionPlanSeparatesStorageAndComputeTypes)
{
    const PrecisionPlan bf16 = make_precision_plan(Type::BF16);
    EXPECT_EQ(bf16.activations, Type::BF16);
    EXPECT_EQ(bf16.weights, Type::BF16);
    EXPECT_EQ(bf16.master_weights, Type::FP32);
    EXPECT_EQ(bf16.gradients, Type::FP32);

    const PrecisionPlan int8 = make_precision_plan(Type::INT8);
    EXPECT_EQ(int8.activations, Type::BF16);
    EXPECT_EQ(int8.weights, Type::INT8);
    EXPECT_EQ(int8.gradients, Type::FP32);
    EXPECT_TRUE(int8.quantized_weights());
}
