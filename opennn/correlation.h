#ifndef CORRELATION_H
#define CORRELATION_H

#include <iostream>
#include <string>

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct Correlation
{
    enum class Method{Pearson, Spearman};

    enum class Form{Linear, Logistic, Logarithmic, Exponential, Power};

    explicit Correlation() {}

    string write_type() const
    {
        switch(form)
        {
        case Form::Linear: return "linear";
        case Form::Logistic: return "logistic";
        case Form::Logarithmic: return "logarithmic";
        case Form::Exponential: return "exponential";
        case Form::Power: return "power";
        default:
            return string();
        }
    }

    void print() const
    {
        cout << "Correlation" << endl;
        cout << "Type: " << write_type() << endl;
        cout << "a: " << a << endl;
        cout << "b: " << b << endl;
        cout << "r: " << r << endl;
        cout << "Lower confidence: " << lower_confidence << endl;
        cout << "Upper confidence: " << upper_confidence << endl;
    }

    type a = type(NAN);

    type b = type(NAN);

    type r = type(NAN);

    type lower_confidence = type(NAN);
    type upper_confidence = type(NAN);

    Method method = Method::Pearson;
    Form form = Form::Linear;
};

}
#endif
