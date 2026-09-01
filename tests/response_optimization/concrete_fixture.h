//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N C R E T E   F I X T U R E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// The trained UCI concrete network that the concrete example ships, and the handful of
// facts about it that the response tests are written in terms of. Shared so that the
// scenario tests and the feasibility study read the same network and name its columns
// the same way.

#pragma once

#include <filesystem>

#include "opennn/neural_network/neural_network.h"
#include "opennn/response_optimization/response_optimization.h"

using namespace opennn;

using Sense = ResponseOptimization::Objective::Sense;
using Condition = ResponseOptimization::Constraint::Condition;

// Result columns: the eight mix variables the network takes, then the response it gives.

enum Column { Cement, Slag, FlyAsh, Water, Sp, CoarseAgg, FineAgg, Age, Strength, ColumnsNumber };

inline const char* const column_names[ColumnsNumber] =
    {"cement", "slag", "fly_ash", "water", "sp", "coarse_agg", "fine_agg", "age", "strength"};

// The mass of one cubic metre of mix, used by every case that closes the batch.

inline constexpr float mix_mass = 2325.012558f;

// Constraints are met to a relative slack inside the optimizer, and a repaired point is
// placed a little inside the bound rather than on it. These checks only have to catch a
// point that is actually outside, so they read the bound with a wider tolerance.

inline float bound_slack(const float bound) { return max(1e-2f, abs(bound)*1e-3f); }


// One shared network for every test in the binary. Nothing writes to it.

inline NeuralNetwork& concrete_network()
{
    static NeuralNetwork network(std::filesystem::path(CONCRETE_NETWORK_DIR) / "nn" / "concrete_uci.json");

    return network;
}


inline const char* condition_name(const Condition condition)
{
    switch (condition)
    {
    case Condition::Equal:        return "equal";
    case Condition::Between:      return "between";
    case Condition::GreaterEqual: return "at least";
    case Condition::LessEqual:    return "at most";
    case Condition::Greater:      return "above";
    case Condition::Less:         return "below";
    case Condition::AllowedSet:   return "one of";
    case Condition::Integer:      return "integer";
    case Condition::Cardinality:  return "cardinality";
    }

    return "?";
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
