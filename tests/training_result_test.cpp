#include "pch.h"

#include "opennn/training_result.h"

using namespace opennn;

TEST(TrainingResult, EpochsNumberMatchesRecordedHistory)
{
    const TrainingResult empty_results;
    EXPECT_EQ(empty_results.get_epochs_number(), 0);

    const TrainingResult results(3);
    EXPECT_EQ(results.get_epochs_number(), 3);
    EXPECT_EQ(results.write_override_results()(0, 1), "3");

    testing::internal::CaptureStdout();
    results.print();
    const string output = testing::internal::GetCapturedStdout();

    EXPECT_NE(output.find("Epochs number: 3"), string::npos);
}

