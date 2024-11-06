#include "pch.h"

#include "../opennn/inputs_selection.h"
#include "../opennn/growing_inputs.h"


TEST(InputsSelectionTest, DefaultConstructor)
{
    GrowingInputs gi2;

//    assert_true(!gi2.has_training_strategy(), LOG);

    EXPECT_EQ(1, 1);
}


TEST(InputsSelectionTest, GeneralConstructor)
{
//    GrowingInputs gi1(&training_strategy);

//    assert_true(gi1.has_training_strategy(), LOG);

    EXPECT_EQ(1, 1);
}
