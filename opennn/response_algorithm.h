//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E S P O N S E   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

class ResponseOptimization;

class ResponseAlgorithm
{
public:

    virtual ~ResponseAlgorithm() = default;

    virtual string get_name() const = 0;

    virtual MatrixR optimize(const ResponseOptimization&) = 0;

    virtual Index get_evaluations_used() const;

    virtual vector<float> get_utopian_point(const ResponseOptimization&) const;

    virtual pair<Index, VectorR> get_advised_point(const ResponseOptimization&,
                                                   const MatrixR&,
                                                   const VectorR& importance_scale = VectorR()) const = 0;

    virtual pair<Index, VectorR> get_robust_point(const ResponseOptimization&,
                                                  const MatrixR&,
                                                  float balance = 0.5f) const = 0;

    void set_display(bool);
    bool get_display() const noexcept;

protected:

    bool display = true;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
