#ifndef WORDBAG_H
#define WORDBAG_H

#include <string>
//#include "config.h"
#include <iostream>

#include "../eigen/unsupported/Eigen/CXX11/Tensor"


using namespace std;
using namespace Eigen;

namespace opennn
{

struct WordBag
{
    /// Default constructor.

    explicit WordBag() {}

    /// Destructor.

    virtual ~WordBag() {}

    Tensor<string, 1> words;
    Tensor<Index, 1> frequencies;
    Tensor<double, 1> percentages;

    Index size() const
    {
        return words.size();
    }

    void print() const
    {
        const Index words_size = words.size();

        cout << "Word bag size: " << words_size << endl;

        for(Index i = 0; i < words_size; i++)
            cout << words(i) << ": frequency= " << frequencies(i) << ", percentage= " << percentages(i) << endl;
    }
};


}
#endif
