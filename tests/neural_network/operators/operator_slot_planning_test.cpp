#include "tests/pch.h"

#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/operators/activation_operator.h"
#include "opennn/neural_network/operators/c2psa_operator.h"
#include "opennn/neural_network/operators/combination_operator.h"
#include "opennn/neural_network/operators/dropout_operator.h"

using namespace opennn;


TEST(OperatorSlotPlanningTest, OperatorOwnedSlotsStartUnplanned)
{
    ActivationOperator activation;
    DropoutOperator dropout;
    CombinationOperator combination;
    C2PSAOperator c2psa;

    EXPECT_FALSE(activation.saved_output_slot.has_value());
    EXPECT_FALSE(dropout.mask_slot.has_value());
    EXPECT_FALSE(combination.relu_mask_slot.has_value());
    EXPECT_FALSE(c2psa.forward_scratch_slot.has_value());
    EXPECT_FALSE(c2psa.backward_scratch_slot.has_value());
}


TEST(OperatorSlotPlanningTest, ActiveDropoutRequiresAPlannedMaskSlot)
{
    DropoutOperator dropout;
    dropout.set_rate(0.5f);

    ForwardPropagation forward_propagation;

    EXPECT_THROW(
        dropout.forward_propagate(forward_propagation, 0, ForwardPropagationMode::Training),
        runtime_error);
}
