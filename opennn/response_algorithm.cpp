//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   A L G O R I T H M   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "response_algorithm.h"

namespace opennn
{

Index ResponseAlgorithm::get_evaluations_used() const
{
    return 0;
}


vector<float> ResponseAlgorithm::get_utopian_point(const ResponseOptimization&) const
{
    return {};
}


void ResponseAlgorithm::set_display(bool new_display)
{
    display = new_display;
}


bool ResponseAlgorithm::get_display() const noexcept
{
    return display;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
