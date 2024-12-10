#ifndef CORRELATION_H
#define CORRELATION_H

#include "pch.h"

namespace opennn
{

struct Correlation
{
    enum class Method{Pearson, Spearman};

    enum class Form{Linear, Logistic, Logarithmic, Exponential, Power};

    explicit Correlation() {}

    void set_perfect()
    {
        r = type(1);
        a = type(0);
        b = type(1);

        upper_confidence = type(1);
        lower_confidence = type(1);
        form = Correlation::Form::Linear;
    }

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
        cout << "Correlation" << endl
             << "Type: " << write_type() << endl
             << "a: " << a << endl
             << "b: " << b << endl
             << "r: " << r << endl
             << "Lower confidence: " << lower_confidence << endl
             << "Upper confidence: " << upper_confidence << endl;
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
