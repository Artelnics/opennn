#ifndef BOXPLOT_H
#define BOXPLOT_H

//#include <string>

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct BoxPlot {

    type minimum = type(0);

    type first_quartile = type(0);

    type median = type(0);

    type third_quartile = type(0);

    type maximum = type(0);

    // Default constructor.

    explicit BoxPlot() {}

    // Values constructor.

    explicit BoxPlot(const type&, const type&, const type&, const type&, const type&);

    void set(const type&, const type&, const type&, const type&, const type&);
};


}
#endif
