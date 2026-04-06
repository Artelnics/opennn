#include "neural_designer_utilities.h"

namespace opennn
{

void sort_string_vector(vector<string>& string_vector)
{
    auto compare_string_length = [](const string& a, const string& b)
    {
        return a.length() > b.length();
    };

    sort(string_vector.begin(), string_vector.end(), compare_string_length);
}


vector<string> concatenate_string_vectors(const vector<string>& string_vector_1,
                                          const vector<string>& string_vector_2)
{
    vector<string> string_vector;
    string_vector.reserve(string_vector_1.size() + string_vector_2.size());

    string_vector.insert(string_vector.end(), string_vector_2.begin(), string_vector_2.end());
    string_vector.insert(string_vector.end(), string_vector_1.begin(), string_vector_1.end());

    return string_vector;
}


string formatNumber(type value, int precision)
{
    ostringstream oss;
    oss << fixed << setprecision(precision) << value;

    string str = oss.str();

    auto pos = str.find('.');

    if (pos != string::npos)
    {
        str.erase(str.find_last_not_of('0') + 1);

        if (str.back() == '.')
            str.pop_back();
    }

    return str;
}


type round_to_precision(type x, const int& precision)
{
    const type factor = type(pow(10, precision));

    return round(factor*x)/factor;
}

}
