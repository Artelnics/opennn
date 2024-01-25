#ifndef WORDBAG_H
#define WORDBAG_H

#include <string>
#include "config.h"
#include <iostream>


using namespace std;
using namespace Eigen;

namespace opennn
{

///
/// This structure is a necessary tool in text analytics, the word bag is similar a notebook
/// where you store the words and statistical processing is done to obtain relevant information.
/// Return various list with words, repetition frequencies and percentages.

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
