#ifndef BOXPLOT_H
#define BOXPLOT_H

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct BoxPlot {

    type minimum = type(NAN);

    type first_quartile = type(NAN);

    type median = type(NAN);

    type third_quartile = type(NAN);

    type maximum = type(NAN);

    explicit BoxPlot() {}

    explicit BoxPlot(const type&, const type&, const type&, const type&, const type&);

    void set(const type&, const type&, const type&, const type&, const type&);
};


}
#endif
