#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <string>

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct BoundingBox
{
    /// Default constructor.

    explicit BoundingBox() {}

    explicit BoundingBox(const Index&, const Index&, const Index&);

    explicit BoundingBox(const Index&,
                         const Tensor<Index, 1>&,
                         const Index&,
                         const Index&);

    explicit BoundingBox(const Index&,
                         const Index&,
                         const Index&,
                         const Index&,
                         const Index&);

    /// Destructor.

    virtual ~BoundingBox() {}

    Index get_size() const;

    BoundingBox resize(const Index&, const Index&, const Index&) const;

    void print() const;

    Index channels_number;

    Tensor<type, 1> data;

    Index x_center;
    Index y_center;
    Index width;
    Index height;

    Index x_top_left;
    Index y_top_left;
    Index x_bottom_right;
    Index y_bottom_right;
};


}
#endif
