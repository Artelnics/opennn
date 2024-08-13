//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include <math.h>
#include <regex>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <cctype>
#include <iomanip>

#include "strings_utilities.h"
#include "word_bag.h"
#include "tensors.h"

namespace opennn
{

/// Returns the number of strings delimited by separator.
/// If separator does not match anywhere in the string, this method returns 0.
/// @param str String to be tokenized.

//Index count_tokens(string& text, const char& separator)
//{
//    trim(text);

//    Index tokens_count = 0;

    // Skip delimiters at beginning.

//    string::size_type last_position = text.find_first_not_of(separator, 0);

    // Find first "non-delimiter".

//    string::size_type position = text.find_first_of(separator, last_position);

//    while(string::npos != position || string::npos != last_position)
//    {
        // Found a token, add it to the vector

//        tokens_count++;

        // Skip delimiters.  Note the "not_of"

//        last_position = text.find_first_not_of(separator, position);

        // Find next "non-delimiter"

//        position = text.find_first_of(separator, last_position);
//    }

//    return tokens_count;
//}


// Index count_tokens(const string& text, const char& separator)
// {
//     Index tokens_number = count(text.begin(), text.end(), separator);

//     if(text[0] == separator)
//     {
//         tokens_number--;
//     }

//     if(text[text.size() - 1] == separator)
//     {
//         tokens_number--;
//     }

//     return tokens_number + 1;
// }


/// Splits the string into substrings(tokens) wherever separator occurs, and returns a vector with those strings.
/// If separator does not match anywhere in the string, this method returns a single-element list containing this string.
/// @param str String to be tokenized.

void fill_tokens(const string& text, const string& separator, Tensor<string, 1>& tokens)
{
    tokens.setConstant("");

    // Skip delimiters at beginning.

    string::size_type last_position = text.find_first_not_of(separator, 0);

    string::size_type position = text.find_first_of(separator, last_position);

    // Find first "non-delimiter"

    Index index = 0;

    Index old_position = last_position;

    while(string::npos != position || string::npos != last_position)
    {
        // Found a token, add it to the vector

        if(last_position - old_position != 1 && index != 0)
        {
            tokens[index] = "";
            index++;
            old_position++;
            continue;
        }
        else
        {
            // Found a token, add it to the vector

            tokens[index] = text.substr(last_position, position - last_position);
        }

        old_position = position;

        // Skip delimiters. Note the "not_of"

        last_position = text.find_first_not_of(separator, position);

        // Find next "non-delimiter"

        position = text.find_first_of(separator, last_position);

        index++;
    }
}


/// Returns the number of strings delimited by separator.
/// If separator does not match anywhere in the string, this method returns 0.
/// @param str String to be tokenized.

Index count_tokens(const string& text, const string& separator)
{
    Index tokens_number = 0;

    string::size_type position = 0;

    while(text.find(separator, position) != string::npos)
    {
        position = text.find(separator, position);
        tokens_number++;
        position += separator.length();
    }

    if(text.find(separator, 0) == 0)
    {
        tokens_number--;
    }

    if(position == text.length())
    {
        tokens_number--;
    }

    return tokens_number + 1;
}


/// Splits the string into substrings(tokens) wherever separator occurs, and returns a vector with those strings.
/// If separator does not match anywhere in the string, this method returns a single-element list containing this string.
/// @param str String to be tokenized.

Tensor<string, 1> get_tokens(const string& text, const string& separator)
{
    const Index tokens_number = count_tokens(text, separator);

    Tensor<string,1> tokens(tokens_number);

    // Skip delimiters at beginning.

    string::size_type last_position = text.find_first_not_of(separator, 0);

    // Find first "non-delimiter"

    Index index = 0;
    Index old_position = last_position;

    string::size_type position = text.find_first_of(separator, last_position);

    while(string::npos != position || string::npos != last_position)
    {
        if(last_position - old_position != 1 && index != 0)
        {
            index++;
            old_position++;
            continue;
        }

        // Found a token, add it to the vector

        tokens[index] = text.substr(last_position, position - last_position);

        old_position = position;

        // Skip delimiters. Note the "not_of"

        last_position = text.find_first_not_of(separator, position);

        // Find next "non-delimiter"

        position = text.find_first_of(separator, last_position);

        index++;
    }

/*
    //    string str = s;
//    size_t position = 0;
    size_t last_position = 0;
    Index index = 0;

    string::size_type position = 0;

    while(text.find(separator, position) != string::npos)
    {

        if(position == 0) // Skip first position
        {
            position += separator.length();
            last_position = position;
            continue;
        }

        tokens(index) = text.substr(last_position, position - last_position);

        position += separator.length();
        last_position = position;
        index++;
    }

    if(last_position != text.length())
    {
        tokens(index) = text.substr(last_position, text.length() - last_position);
    }
*/
    return tokens;
}


/// Returns a new vector with the elements of this string vector casted to type.

Tensor<type, 1> to_type_vector(const string& text, const string& separator)
{
    const Tensor<string, 1> tokens = get_tokens(text, separator);

    const Index tokens_size = tokens.dimension(0);

    Tensor<type, 1> type_vector(tokens_size);

    for(Index i = 0; i < tokens_size; i++)
    {
        try
        {
            stringstream buffer;

            buffer << tokens[i];

            type_vector(i) = type(stof(buffer.str()));
        }
        catch(const exception&)
        {
            type_vector(i) = type(nan(""));
        }
    }

    return type_vector;
}


/// Returns a new vector with the elements of this string vector casted to type.

Tensor<Index, 1> to_index_vector(const string& text, const string& separator)
{
    const Tensor<string, 1> tokens = get_tokens(text, separator);

    const Index tokens_size = tokens.dimension(0);

    Tensor<Index, 1> index_vector(tokens_size);

    for(Index i = 0; i < tokens_size; i++)
    {
        try
        {
            stringstream buffer;

            buffer << tokens[i];

            index_vector(i) = Index(stoi(buffer.str()));
        }
        catch(const exception&)
        {
            index_vector(i) = Index(-1);
        }
    }

    return index_vector;
}


Tensor<string, 1> get_unique_elements(const Tensor<string,1>& tokens)
{
    string result = " ";

    for(Index i = 0; i < tokens.size(); i++)
    {
        if( !contains_substring(result, " " + tokens(i) + " ") )
        {
            result += tokens(i) + " ";
        }
    }

    return get_tokens(result, " ");
}


Tensor<Index, 1> count_unique(const Tensor<string,1>& tokens)
{
    Tensor<string, 1> unique_elements = get_unique_elements(tokens);

    const Index unique_size = unique_elements.size();

    Tensor<Index, 1> unique_count(unique_size);

    for(Index i = 0; i < unique_size; i++)
    {
        unique_count(i) = Index(count(tokens.data(), tokens.data() + tokens.size(), unique_elements(i)));
    }

    return unique_count;
}



/// Returns true if the string passed as argument represents a number, and false otherwise.
/// @param str String to be checked.

bool is_numeric_string(const string& text)
{
    string::size_type index;

    istringstream iss(text.data());

    float dTestSink;

    iss >> dTestSink;

    if(!iss) return false;

    try
    {
        stod(text, &index);

        if(index == text.size()
        || (text.find("%") != string::npos && index+1 == text.size()))
        {
            return true;
        }
        else
        {
            return  false;
        }
    }
    catch(const exception&)
    {
        return false;
    }
}


/// Returns true if given string vector is constant and false otherwise.
/// @param str vector to be checked.
///
bool is_constant_string(const Tensor<string, 1>& string_list)
{
    const string str0 = string_list[0];

    string str1;

    for(int i = 1; i < string_list.size(); i++)
    {
        str1 = string_list[i];

        if(str1.compare(str0) != 0)
            return false;
    }

    return true;
}


/// Returns true if given numeric vector is constant and false otherwise.
/// @param str vector to be checked.

bool is_constant_numeric(const Tensor<type, 1>& str)
{
    const type a0 = str[0];

    for(int i = 1; i < str.size(); i++)
    {
        if(abs(str[i]-a0) > type(1e-3) || isnan(str[i]) || isnan(a0)) return false;
    }

    return true;
}


/// Returns true if given string is a date and false otherwise.
/// @param str String to be checked.

bool is_date_time_string(const string& text)
{
    if(is_numeric_string(text))return false;

    const string format_1 = "(201[0-9]|202[0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_2 = "(201[0-9]|202[0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])";
    const string format_3 = "(201[0-9]|202[0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])";
    const string format_4 = "(0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|\\s|/|.](0[1-9]|1[0-2])+[-|\\s|/|.](200[0-9]|201[0-9]|202[0-9]|19[0-9][0-9])+[,| ||-]([0-1][0-9]|2[0-3]|[0-9])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_5 = "(0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|\\s|/|.](0[1-9]|1[0-2])+[-|\\s|/|.](200[0-9]|201[0-9]|202[0-9]|19[0-9][0-9])+[,| ||-]([0-1][0-9]|2[0-3]|[0-9])+[:]([0-5][0-9])";
    const string format_6 = "(0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|\\s|/|.](0[1-9]|1[0-2])+[-|\\s|/|.](200[0-9]|201[0-9]|202[0-9]|19[0-9][0-9])";
    const string format_7 = "(201[0-9]|202[0-9]|19[0-9][0-9])+[-|/|.]([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_8 = "(201[0-9]|202[0-9]|19[0-9][0-9])+[-|/|.]([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])";
    const string format_9 = "(201[0-9]|202[0-9]|19[0-9][0-9])+[-|/|.]([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])";
    const string format_10 = "([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+ (0[1-9]|1[0-9]|2[0-9]|3[0-1])+[| ][,|.| ](201[0-9]|202[0-9]|19[0-9][0-9])";
    const string format_11 = "(20[0-9][0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])";
    const string format_12 = "^\\d{1,2}/\\d{1,2}/\\d{4}$";
    const string format_13 = "([0-2][0-9])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_14 = "([1-9]|0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|/|.](201[0-9]|202[0-9]|19[0-9][0-9])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])+[:]([0-5][0-9])+[,| ||-][AP]M";
//    const string format_15  = "(0[1-9]|[1-2][0-9]|3[0-1])[.|/|-](0[1-9]|1[0-2])[.|/|-](20[0-9]{2}|[2-9][0-9]{3})\\s([0-1][0-9]|2[0-3])[:]([0-5][0-9])[:]([0-5][0-9])[.][0-9]{6}";
    const string format_15 = "(\\d{4})[.|/|-](\\d{2})[.|/|-](\\d{2})\\s(\\d{2})[:](\\d{2}):(\\d{2})\\.\\d{6}";
    const string format_16 = "(\\d{2})[.|/|-](\\d{2})[.|/|-](\\d{4})\\s(\\d{2})[:](\\d{2}):(\\d{2})\\.\\d{6}";
    const string format_17 = "^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/(\\d{2}) ([01]?\\d|2[0-3]):([0-5]\\d)$";

    const regex regular_expression(format_1 + "|"
                                   + format_2 + "|"
                                   + format_3 + "|"
                                   + format_4 + "|"
                                   + format_5 + "|"
                                   + format_6 + "|"
                                   + format_7 + "|"
                                   + format_8 + "|"
                                   + format_9 + "|"
                                   + format_10 + "|"
                                   + format_11 +"|"
                                   + format_12  + "|"
                                   + format_13 + "|"
                                   + format_14 + "|"
                                   + format_15 + "|"
                                   + format_16 + "|"
                                   + format_17);

    if(regex_match(text, regular_expression))
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Return true if word is a email, and false otherwise.
/// @param word Word to check.

bool is_email(const string& word)
{
    // define a regular expression
    const regex pattern("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+");

    // try to match the string with the regular expression
    return regex_match(word, pattern);
}


/// Return true if word contains a number, and false otherwise.
/// @param word Word to check.

bool contains_number(const string& word)
{
    return(find_if(word.begin(), word.end(), ::isdigit) != word.end());
}


/// Returns true if a word starting with a given substring, and false otherwise.
/// @param word Word to check.
/// @param starting Substring to comparison given word.

bool starts_with(const string& word, const string& starting)
{
    if(starting.length() > word.length() || starting.length() == 0)
    {
        return false;
    }

    return(word.substr(0,starting.length()) == starting);
}


/// Returns true if a word ending with a given substring, and false otherwise.
/// @param word Word to check.
/// @param ending Substring to comparison given word.

//bool ends_with(const string& word, const string& ending)
//{
//    if(ending.length() > word.length())
//    {
//        return false;
//    }

//    return(word.substr(word.length() - ending.length()) == ending);
//}


/// Returns true if a word ending with a given substring Tensor, and false otherwise.
/// @param word Word to check.
/// @param ending Substring Tensor with possibles endings words.

bool ends_with(const string& word, const Tensor<string,1>& endings)
{
    const Index endings_size = endings.size();

    for(Index i = 0; i < endings_size; i++)
    {
        if(ends_with(word, endings[i]))
        {
            return true;
        }
    }

    return false;
}


/// Transforms human date into timestamp.
/// @param date Date in string fortmat to be converted.
/// @param gmt Greenwich Mean Time.

time_t date_to_timestamp(const string& date, const Index& gmt)
{
    struct tm time_structure = {};

    smatch month;

    const regex months("([Jj]an(?:uary)?)|([Ff]eb(?:ruary)?)|([Mm]ar(?:ch)?)|([Aa]pr(?:il)?)|([Mm]ay)|([Jj]un(?:e)?)|([Jj]ul(?:y)?)"
                       "|([Aa]ug(?:gust)?)|([Ss]ep(?:tember)?)|([Oo]ct(?:ober)?)|([Nn]ov(?:ember)?)|([Dd]ec(?:ember)?)");

    smatch matchs;

    const string format_1 = "(201[0-9]|202[0-9]|200[0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_2 = "(201[0-9]|202[0-9]|200[0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])";
    const string format_3 = "(201[0-9]|202[0-9]|200[0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])";
    const string format_4 = "(0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|\\s|/|.](0[1-9]|1[0-2])+[-|\\s|/|.](200[0-9]|201[0-9]|202[0-9]|19[0-9][0-9])+[,| ||-]([0-1][0-9]|2[0-3]|[0-9])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_5 = "(0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|\\s|/|.](0[1-9]|1[0-2])+[-|\\s|/|.](200[0-9]|201[0-9]|202[0-9]|19[0-9][0-9])+[,| ||-]([0-1][0-9]|2[0-3]|[0-9])+[:]([0-5][0-9])";
    const string format_6 = "(0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|\\s|/|.](0[1-9]|1[0-2])+[-|\\s|/|.](200[0-9]|201[0-9]|202[0-9]|19[0-9][0-9])";
    const string format_7 = "(201[0-9]|202[0-9]|200[0-9]|19[0-9][0-9])+[-|/|.]([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_8 = "(201[0-9]|202[0-9]|200[0-9]|19[0-9][0-9])+[-|/|.]([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])";
    const string format_9 = "(201[0-9]|202[0-9]|200[0-9]|19[0-9][0-9])+[-|/|.]([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])";
    const string format_10 = "([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:gust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)+ (0[1-9]|1[0-9]|2[0-9]|3[0-1])+[| ][,|.| ](201[0-9]|202[0-9]|19[0-9][0-9])";
    const string format_11 = "(20[0-9][0-9]|19[0-9][0-9])+[-|/|.](0[1-9]|1[0-2])";
    const string format_12 = "([0-2][0-9])+[:]([0-5][0-9])+[:]([0-5][0-9])";
    const string format_13 = "([1-9]|0[1-9]|1[0-2])+[-|/|.](0[1-9]|1[0-9]|2[0-9]|3[0-1])+[-|/|.](201[0-9]|202[0-9]|19[0-9][0-9])+[,| ||-]([0-1][0-9]|2[0-3])+[:]([0-5][0-9])+[:]([0-5][0-9])+[,| ||-][AP]M";
    const string format_14 = "(201[0-9]|202[0-9]|200[0-9]|19[0-9][0-9])";
    const string format_15 = "(\\d{4})[.|/|-](\\d{2})[.|/|-](\\d{2})\\s(\\d{2})[:](\\d{2}):(\\d{2})\\.\\d{6}";
    const string format_16 = "(\\d{2})[.|/|-](\\d{2})[.|/|-](\\d{4})\\s(\\d{2})[:](\\d{2}):(\\d{2})\\.\\d{6}";
    const string format_17 = "^(0?[1-9]|[12][0-9]|3[01])/(0?[1-9]|1[0-2])/(\\d{2}) ([01]?\\d|2[0-3]):([0-5]\\d)$";

    const regex regular_expression(format_1 + "|" + format_2 + "|" + format_3 + "|" + format_4 + "|" + format_5 + "|" + format_6 + "|" + format_7 + "|" + format_8
                                   + "|" + format_9 + "|" + format_10 + "|" + format_11 +"|" + format_12  + "|" + format_13 + "|" + format_14 + "|" + format_15
                                   + "|" + format_16 + "|" + format_17);

    regex_search(date, matchs, regular_expression);

    if(matchs[1] != "") // yyyy/mm/dd hh:mm:ss
    {

        if(stoi(matchs[1].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {

            time_structure.tm_year = stoi(matchs[1].str())-1900;
            time_structure.tm_mon = stoi(matchs[2].str())-1;
            time_structure.tm_mday = stoi(matchs[3].str());
            time_structure.tm_hour = stoi(matchs[4].str()) - int(gmt);
            time_structure.tm_min = stoi(matchs[5].str());
            time_structure.tm_sec = stoi(matchs[6].str());
        }
    }
    else if(matchs[7] != "") // yyyy/mm/dd hh:mm
    {
        if(stoi(matchs[7].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            time_structure.tm_year = stoi(matchs[7].str())-1900;
            time_structure.tm_mon = stoi(matchs[8].str())-1;
            time_structure.tm_mday = stoi(matchs[9].str());
            time_structure.tm_hour = stoi(matchs[10].str()) - int(gmt);
            time_structure.tm_min = stoi(matchs[11].str());
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[12] != "") // yyyy/mm/dd
    {
        if(stoi(matchs[12].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            time_structure.tm_year = stoi(matchs[12].str())-1900;
            time_structure.tm_mon = stoi(matchs[13].str())-1;
            time_structure.tm_mday = stoi(matchs[14].str());
            time_structure.tm_hour = 0;
            time_structure.tm_min = 0;
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[15] != "") // dd/mm/yyyy hh:mm:ss
    {
        if(stoi(matchs[17].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            time_structure.tm_year = stoi(matchs[17].str()) - 1900;
            time_structure.tm_mon = stoi(matchs[16].str()) - 1;
            time_structure.tm_mday = stoi(matchs[15].str());
            time_structure.tm_hour = stoi(matchs[18].str()) - int(gmt);
            time_structure.tm_min = stoi(matchs[19].str());
            time_structure.tm_sec = stoi(matchs[20].str());
        }
    }
    else if(matchs[21] != "") // dd/mm/yyyy hh:mm
    {
        if(stoi(matchs[23].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            time_structure.tm_year = stoi(matchs[23].str())-1900;
            time_structure.tm_mon = stoi(matchs[22].str())-1;
            time_structure.tm_mday = stoi(matchs[21].str());
            time_structure.tm_hour = stoi(matchs[24].str()) - int(gmt);
            time_structure.tm_min = stoi(matchs[25].str());
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[26] != "") // dd/mm/yyyy
    {
        if(stoi(matchs[28].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            time_structure.tm_year = stoi(matchs[28].str())-1900;
            time_structure.tm_mon = stoi(matchs[27].str())-1;
            time_structure.tm_mday = stoi(matchs[26].str());
            time_structure.tm_hour = 0;
            time_structure.tm_min = 0;
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[29] != "") // yyyy/mmm|mmmm/dd hh:mm:ss
    {
        if(stoi(matchs[29].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            regex_search(date, month, months);

            Index month_number = 0;

            if(!month.empty())
            {
                for(Index i = 1; i < 13; i++)
                {
                    if(month[size_t(i)] != "") month_number = i;
                }
            }

            time_structure.tm_year = stoi(matchs[29].str())-1900;
            time_structure.tm_mon = int(month_number) - 1;
            time_structure.tm_mday = stoi(matchs[31].str());
            time_structure.tm_hour = stoi(matchs[32].str()) - int(gmt);
            time_structure.tm_min = stoi(matchs[33].str());
            time_structure.tm_sec = stoi(matchs[34].str());
        }
    }
    else if(matchs[35] != "") // yyyy/mmm|mmmm/dd hh:mm
    {
        if(stoi(matchs[35].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            regex_search(date, month, months);

            Index month_number = 0;
            if(!month.empty())
            {
                for(Index i =1 ; i<13  ; i++)
                {
                    if(month[size_t(i)] != "") month_number = i;
                }
            }

            time_structure.tm_year = stoi(matchs[35].str()) - 1900;
            time_structure.tm_mon = int(month_number) - 1;
            time_structure.tm_mday = stoi(matchs[37].str());
            time_structure.tm_hour = stoi(matchs[38].str()) - int(gmt);
            time_structure.tm_min = stoi(matchs[39].str());
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[40] != "") // yyyy/mmm|mmmm/dd
    {
        if(stoi(matchs[40].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            regex_search(date, month, months);

            Index month_number = 0;
            if(!month.empty())
            {
                for(Index i =1 ; i<13  ; i++)
                {
                    if(month[size_t(i)] != "") month_number = i;
                }
            }

            time_structure.tm_year = stoi(matchs[40].str())-1900;
            time_structure.tm_mon = int(month_number)-1;
            time_structure.tm_mday = stoi(matchs[42].str())- int(gmt);
            time_structure.tm_hour = 0;
            time_structure.tm_min = 0;
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[43] != "") // mmm dd, yyyy
    {
        if(stoi(matchs[45].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            regex_search(date,month,months);

            Index month_number = 0;

            if(!month.empty())
            {
                for(Index i =1 ; i<13  ; i++)
                {
                    if(month[size_t(i)] != "") month_number = i;
                }
            }

            time_structure.tm_year = stoi(matchs[45].str())-1900;
            time_structure.tm_mon = int(month_number)-1;
            time_structure.tm_mday = stoi(matchs[44].str());
            time_structure.tm_hour = 0;
            time_structure.tm_min = 0;
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[46] != "") // yyyy/ mm
    {
        if(stoi(matchs[46].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            time_structure.tm_year = stoi(matchs[46].str())-1900;
            time_structure.tm_mon = stoi(matchs[47].str())-1;
            time_structure.tm_mday = 1;
            time_structure.tm_hour = 0;
            time_structure.tm_min = 0;
            time_structure.tm_sec = 0;
        }
    }
    else if(matchs[48] != "") // hh:mm:ss
    {
        time_structure.tm_year = 70;
        time_structure.tm_mon = 0;
        time_structure.tm_mday = 1;
        time_structure.tm_hour = stoi(matchs[48].str());
        time_structure.tm_min = stoi(matchs[49].str());
        time_structure.tm_sec = stoi(matchs[50].str());
    }
    else if(matchs[51] != "") // mm/dd/yyyy hh:mm:ss [AP]M
    {
        time_structure.tm_year = stoi(matchs[53].str())-1900;
        time_structure.tm_mon = stoi(matchs[51].str());
        time_structure.tm_mday = stoi(matchs[52].str());
        time_structure.tm_min = stoi(matchs[55].str());
        time_structure.tm_sec = stoi(matchs[56].str());
        if(matchs[57].str()=="PM"){
            time_structure.tm_hour = stoi(matchs[54].str())+12;
        }
        else{
            time_structure.tm_hour = stoi(matchs[54].str());
        }
    }
    else if(matchs[58] != "") // yyyy
    {
        time_structure.tm_year = stoi(matchs[57].str())-1900;
        time_structure.tm_mon = 0;
        time_structure.tm_mday = 1;
        time_structure.tm_hour = 0;
        time_structure.tm_min = 0;
        time_structure.tm_sec = 0;

        return mktime(&time_structure);
    }
    else if(matchs[59] != "") // yyyy/mm/dd hh:mm:ss.ssssss
    {
        if(stoi(matchs[60].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw runtime_error(buffer.str());
        }
        else
        {
            time_structure.tm_year = stoi(matchs[60].str())-1900;
            time_structure.tm_mon = stoi(matchs[59].str())-1;
            time_structure.tm_mday = stoi(matchs[58].str());
            time_structure.tm_hour = stoi(matchs[61].str()) - int(gmt);
            time_structure.tm_min = stoi(matchs[62].str());
            time_structure.tm_sec = stof(matchs[63].str());
        }
    }
    else if(matchs[70] != "") // %d/%m/%y %H:%M
    {
        time_structure.tm_year = stoi(matchs[72].str()) + 100;
        time_structure.tm_mon = stoi(matchs[71].str())-1;
        time_structure.tm_mday = stoi(matchs[70].str());
        time_structure.tm_hour = stoi(matchs[73].str());
        time_structure.tm_min = stoi(matchs[74].str());
        time_structure.tm_sec = 0;
    }
    else if(is_numeric_string(date)){
    }
    else
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: DataSet Class.\n"
               << "time_t date_to_timestamp(const string&) method.\n"
               << "Date format (" << date << ") is not implemented.\n";
        throw runtime_error(buffer.str());
    }

    if(is_numeric_string(date))
    {
        time_t time_t_date = stoi(date);
        return time_t_date;
    }
    else
    {
        return mktime(&time_structure);
    }
}


/// Returns true if the string contains the given substring and false otherwise.
/// @param str String.
/// @param sub_string Substring to search.

bool contains_substring(const string& text, const string& sub_string)
{
    if(text.find(sub_string) != string::npos)
    {
        return true;
    }

    return false;
}


///Replaces all apprearances of a substring with another string
///@param s
///@param to_replace
///@param replace_with

void replace_all_word_appearances(string& text, const string& to_replace, const string& replace_with)
{
    string buffer;
    size_t position = 0;
    size_t previous_position;
    const string underscore = "_";

    // Reserva una estimación aproximada del tamaño final de la cadena.
    buffer.reserve(text.size());

    while (true)
    {
        previous_position = position;
        position = text.find(to_replace, position);

        if(position == string::npos)
            break;

        // Verifica que no haya letras antes ni después de to_replace
        if((previous_position == 0 || !isalpha(text[previous_position - 1])) &&
            (position + to_replace.size() == text.size() || !isalpha(text[position + to_replace.size()])))
        {
            // Verifica que no haya guiones bajos antes ni después de to_replace
            if((previous_position == 0 || text[previous_position - 1] != '_') &&
                (position + to_replace.size() == text.size() || text[position + to_replace.size()] != '_'))
            {
                buffer.append(text, previous_position, position - previous_position);
                buffer += replace_with;
                position += to_replace.size();
            }
            else
            {
                buffer.append(text, previous_position, position - previous_position + to_replace.size());
                position += to_replace.size();
            }
        }
        else
        {
            buffer.append(text, previous_position, position - previous_position + to_replace.size());
            position += to_replace.size();
        }
    }

    buffer.append(text, previous_position, text.size() - previous_position);
    text.swap(buffer);
}



 ///Replaces all apprearances of a substring with another string
 ///@param s
 ///@param to_replace
 ///@param replace_with

void replace_all_appearances(string& text, string const& to_replace, string const& replace_with)
{
    string buffer;
    size_t position = 0;
    size_t previous_position;

    // Reserves rough estimate of final size of string.
    buffer.reserve(text.size());

    while(true)
    {
        previous_position = position;

        position = text.find(to_replace, position);

        if(position == string::npos) break;

        buffer.append(text, previous_position, position - previous_position);

        if(buffer.back() == '_')
        {
            buffer += to_replace;
            position += to_replace.size();

        }
        else
        {
            buffer += replace_with;
            position += to_replace.size();
        }
    }

    buffer.append(text, previous_position, text.size() - previous_position);

    text.swap(buffer);
}


/// Replaces all apprearances non allowed programming characters of a substring with allowed characters
/// \brief replace_non_allowed_programming_characters
/// \param s
/// \return

string replace_non_allowed_programming_expressions(string& s)
{
        string out = "";

        if(s[0] == '$')
            out = s;

        replace_all_appearances(s, "fn", "f_n");
        replace_all_appearances(s, "if", "i_f");
        replace_all_appearances(s, "do", "d_o");
        replace_all_appearances(s, "or", "o_r");
        replace_all_appearances(s, "is", "i_s");
        replace_all_appearances(s, "as", "a_s");
        replace_all_appearances(s, "or", "o_r");
        replace_all_appearances(s, "if", "i_f");
        replace_all_appearances(s, "in", "in_");
        replace_all_appearances(s, "del", "del");
        replace_all_appearances(s, "max","ma_x");
        replace_all_appearances(s, "min","mi_n");
        replace_all_appearances(s, "and", "an_d");
        replace_all_appearances(s, "for", "fo_r");
        replace_all_appearances(s, "die", "di_e");
        replace_all_appearances(s, "int", "in_t");
        replace_all_appearances(s, "new", "ne_w");
        replace_all_appearances(s, "use", "us_e");
        replace_all_appearances(s, "var", "va_r");
        replace_all_appearances(s, "try", "tr_y");
        replace_all_appearances(s, "xor", "xo_r");
        replace_all_appearances(s, "def", "de_f");
        replace_all_appearances(s, "for", "fo_r");
        replace_all_appearances(s, "not", "no_t_");
        replace_all_appearances(s, "rise","ri_se");
        replace_all_appearances(s, "byte", "byt_e");
        replace_all_appearances(s, "echo", "ech_o");
        replace_all_appearances(s, "eval", "eva_l");
        replace_all_appearances(s, "pass", "pa_ss");
        replace_all_appearances(s, "form", "for_m");
        replace_all_appearances(s, "else", "el_se");
        replace_all_appearances(s, "with", "w_ith");
        replace_all_appearances(s, "exit", "exi_t");
        replace_all_appearances(s, "auto", "aut_o");
        replace_all_appearances(s, "enum", "enu_m");
        replace_all_appearances(s, "case", "cas_e");
        replace_all_appearances(s, "char", "cha_r");
        replace_all_appearances(s, "void", "voi_d");
        replace_all_appearances(s, "goto", "got_o");
        replace_all_appearances(s, "long", "lon_g");
        replace_all_appearances(s, "else", "els_e");
        replace_all_appearances(s, "goto", "got_o");
        replace_all_appearances(s, "type", "ty_pe");
        replace_all_appearances(s, "self", "se_lf");
        replace_all_appearances(s, "list", "lis_t");
        replace_all_appearances(s, "None", "No_ne");
        replace_all_appearances(s, "elif", "el_if");
        replace_all_appearances(s, "True", "t_rue_");
        replace_all_appearances(s, "super","sup_er");
        replace_all_appearances(s, "endif", "endi_f");
        replace_all_appearances(s, "await", "awai_t");
        replace_all_appearances(s, "catch", "catc_h");
        replace_all_appearances(s, "class", "clas_s");
        replace_all_appearances(s, "clone", "clon_e");
        replace_all_appearances(s, "empty", "empt_y");
        replace_all_appearances(s, "final", "fina_l");
        replace_all_appearances(s, "break", "brea_k");
        replace_all_appearances(s, "while", "whil_e");
        replace_all_appearances(s, "float", "floa_t");
        replace_all_appearances(s, "union", "unio_n");
        replace_all_appearances(s, "short", "shor_t");
        replace_all_appearances(s, "const", "cons_t");
        replace_all_appearances(s, "match", "matc_h");
        replace_all_appearances(s, "isset", "isse_t");
        replace_all_appearances(s, "while", "whil_e");
        replace_all_appearances(s, "yield", "yiel_d");
        replace_all_appearances(s, "False", "Fa_lse");
        replace_all_appearances(s, "unset", "unse_t");
        replace_all_appearances(s, "print", "prin_t");
        replace_all_appearances(s, "trait", "trai_t");
        replace_all_appearances(s, "throw", "thro_w");
        replace_all_appearances(s, "raise", "rai_se");
        replace_all_appearances(s, "while", "wh_ile");
        replace_all_appearances(s, "yield", "yi_eld");
        replace_all_appearances(s, "break", "bre_ak");
        replace_all_appearances(s, "class", "c_lass");
        replace_all_appearances(s, "string","str_ing");
        replace_all_appearances(s, "except", "exc_ept");
        replace_all_appearances(s, "lambda", "lamb_da");
        replace_all_appearances(s, "assert", "asser_t");
        replace_all_appearances(s, "global", "glo_bal");
        replace_all_appearances(s, "elseif", "elsei_f");
        replace_all_appearances(s, "endfor", "endfo_r");
        replace_all_appearances(s, "static", "stati_c");
        replace_all_appearances(s, "switch", "switc_h");
        replace_all_appearances(s, "struct", "struc_t");
        replace_all_appearances(s, "double", "doubl_e");
        replace_all_appearances(s, "sizeof", "sizeo_f");
        replace_all_appearances(s, "extern", "exter_n");
        replace_all_appearances(s, "signed", "signe_d");
        replace_all_appearances(s, "return", "retur_n");
        replace_all_appearances(s, "global", "globa_l");
        replace_all_appearances(s, "public", "publi_c");
        replace_all_appearances(s, "return", "retur_n");
        replace_all_appearances(s, "static", "stati_c");
        replace_all_appearances(s, "switch", "switc_h");
        replace_all_appearances(s, "import", "imp_ort");
        replace_all_appearances(s, "return", "retu_rn");
        replace_all_appearances(s, "boolea", "boole_an");
        replace_all_appearances(s, "import", "includ_e");
        replace_all_appearances(s, "friend", "frie_end");
        replace_all_appearances(s, "foreach", "foreac_h");
        replace_all_appearances(s, "private", "privat_e");
        replace_all_appearances(s, "require", "requir_e");
        replace_all_appearances(s, "typedef", "typede_f");
        replace_all_appearances(s, "_Packed", "_P_acked");
        replace_all_appearances(s, "default", "defaul_t");
        replace_all_appearances(s, "extends", "extend_s");
        replace_all_appearances(s, "finally", "finall_y");
        replace_all_appearances(s, "finally", "final_ly");
        replace_all_appearances(s, "nonlocal", "nonlo_cal");
        replace_all_appearances(s, "continue", "con_tinue");
        replace_all_appearances(s, "continue", "continu_e");
        replace_all_appearances(s, "volatile", "volatil_e");
        replace_all_appearances(s, "unsigned", "unsigne_d");
        replace_all_appearances(s, "abstract", "abstrac_t");
        replace_all_appearances(s, "register", "registe_r");
        replace_all_appearances(s, "endwhile", "endwhil_e");
        replace_all_appearances(s, "function", "functio_n");
        replace_all_appearances(s, "readonly", "readonl_y");
        replace_all_appearances(s, "arguments", "argument_s");
        replace_all_appearances(s, "endswitch", "endswitc_h");
        replace_all_appearances(s, "protected", "protecte_d");
        replace_all_appearances(s, "insteadof", "insteado_f");
        replace_all_appearances(s, "interface", "interfac_e");
        replace_all_appearances(s, "namespace", "namespac_e");
        replace_all_appearances(s, "enddeclare", "enddeclar_e");
        replace_all_appearances(s, "endforeach", "endforeac_h");
        replace_all_appearances(s, "implements", "implement_s");
        replace_all_appearances(s, "instanceof", "instanceo_f");
        replace_all_appearances(s, "include_once", "include_on_ce_");
        replace_all_appearances(s, "require_once", "require_on_ce_");
        replace_all_appearances(s, "__halt_compiler", "__h_a_l_t_c_o_m_p_i_l_e_r_");

        for(char& c: s)
        {
            if(c=='1'){ out+="_one_";   continue;}
            if(c=='2'){ out+="_two_";   continue;}
            if(c=='3'){ out+="_three_"; continue;}
            if(c=='4'){ out+="_four_";  continue;}
            if(c=='5'){ out+="_five_";  continue;}
            if(c=='6'){ out+="_six_";   continue;}
            if(c=='7'){ out+="_seven_"; continue;}
            if(c=='8'){ out+="_eight_"; continue;}
            if(c=='9'){ out+="_nine_";  continue;}
            if(c=='0'){ out+="_zero_";  continue;}

            if(c=='.'){ out+="_dot_";   continue;}
            if(c=='/'){ out+="_div_";   continue;}
            if(c=='*'){ out+="_mul_";   continue;}
            if(c=='+'){ out+="_sum_";   continue;}
            if(c=='-'){ out+="_res_";   continue;}
            if(c=='='){ out+="_equ_";   continue;}
            if(c=='!'){ out+="_not_";   continue;}
            if(c==','){ out+="_colon_"; continue;}
            if(c==';'){ out+="_semic_"; continue;}
            if(c=='\\'){ out+="_slash_";continue;}

            if(c=='&'){ out+="_amprsn_"; continue;}
            if(c=='?'){ out+="_ntrgtn_"; continue;}
            if(c=='<'){ out+="_lower_" ; continue;}
            if(c=='>'){ out+="_higher_"; continue;}

            if(isalnum(c)!=0){ out += c; continue;}
            if(isalnum(c)==0){ out+='_'; continue;}
        }

        return out;
}


vector<string> get_words_in_a_string(string str)
{
    vector<string> output;
    string word = "";

    for(auto x : str)
    {
        if(isalnum(x))
        {
            word = word + x;
        }else if(x=='_')
        {
            word = word + x;
        }
        else
        //if(x == ' ')
        {
            output.push_back(word);
            word = "";
        }
    }

    output.push_back(word);
    return output;
}


///Returns the number of apprearances of a substring
///@brief WordOccurrence
///@param sentence
///@param word
///@return


//int WordOccurrence(char *sentence, char *word)
//{
//    int slen = strlen(sentence);
//    int wordlen = strlen(word);
//    int count = 0;
//    int i, j;

//    for(i = 0; i<slen; i++)
//    {
//        for(j = 0; j<wordlen; j++)
//        {
//            if(sentence[i+j]!=word[j])
//            break;
//        }
//        if(j==wordlen)
//        {
//            count++;
//        }
//    }
//    return count;
//}


/// Removes whitespaces from the start and the end of the string passed as argument.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

void trim(string& text)
{
    // Prefixing spaces

    text.erase(0, text.find_first_not_of(' '));
    text.erase(0, text.find_first_not_of('\t'));
    text.erase(0, text.find_first_not_of('\n'));
    text.erase(0, text.find_first_not_of('\r'));
    text.erase(0, text.find_first_not_of('\f'));
    text.erase(0, text.find_first_not_of('\v'));

    // Surfixing spaces

    text.erase(text.find_last_not_of(' ') + 1);
    text.erase(text.find_last_not_of('\t') + 1);
    text.erase(text.find_last_not_of('\n') + 1);
    text.erase(text.find_last_not_of('\r') + 1);
    text.erase(text.find_last_not_of('\f') + 1);
    text.erase(text.find_last_not_of('\v') + 1);
    text.erase(text.find_last_not_of('\b') + 1);

    // Special character and string modifications

    replace_first_and_last_char_with_missing_label(text, ';', "NA", "");
    replace_first_and_last_char_with_missing_label(text, ',', "NA", "");

    replace_double_char_with_label(text, ";", "NA");
    replace_double_char_with_label(text, ",", "NA");

    replac_substring_within_quotes(text, ",", "");
    replac_substring_within_quotes(text, ";", "");
}


void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &first_missing_label, const string &last_missing_label)
{
    if(!str.empty())
    {
        if(str[0] == target_char)
        {
            string new_string = first_missing_label + target_char;
            str.replace(0, 1, new_string);
        }

        if(str[str.length() - 1] == target_char)
        {
            string new_string = target_char + last_missing_label;
            str.replace(str.length() - 1, 1, new_string);
        }
    }
}


void replace_double_char_with_label(string &str, const string &target_char, const string &missing_label)
{
    string target_pattern = target_char + target_char;
    string new_pattern = target_char + missing_label + target_char;

    size_t position = 0;
    while((position = str.find(target_pattern, position)) != string::npos)
    {
        str.replace(position, target_pattern.length(), new_pattern);
        position += new_pattern.length();
    }
}


void replac_substring_within_quotes(string &str, const string &target, const string &replacement)
{
    regex r("\"([^\"]*)\"");
    smatch match;
    string result = "";
    string prefix = str;

    while(regex_search(prefix, match, r))
    {
        string match_str = match.str();
        string replaced_str = match_str;
        size_t position = 0;
        while((position = replaced_str.find(target, position)) != string::npos)
        {
            replaced_str.replace(position, target.length(), replacement);
            position += replacement.length();
        }
        result += match.prefix().str() + replaced_str;
        prefix = match.suffix().str();
    }
    result += prefix;
    str = result;
}


void erase(string& text, const char& character)
{
    text.erase(remove(text.begin(), text.end(), character), text.end());
}


/// Returns a string that has whitespace removed from the start and the end.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

string get_trimmed(const string& text)
{
    string output(text);

    //prefixing spaces

    output.erase(0, output.find_first_not_of(' '));
    output.erase(0, output.find_first_not_of('\t'));
    output.erase(0, output.find_first_not_of('\n'));
    output.erase(0, output.find_first_not_of('\r'));
    output.erase(0, output.find_first_not_of('\f'));
    output.erase(0, output.find_first_not_of('\v'));

    //surfixing spaces

    output.erase(output.find_last_not_of(' ') + 1);
    output.erase(output.find_last_not_of('\t') + 1);
    output.erase(output.find_last_not_of('\n') + 1);
    output.erase(output.find_last_not_of('\r') + 1);
    output.erase(output.find_last_not_of('\f') + 1);
    output.erase(output.find_last_not_of('\v') + 1);
    output.erase(output.find_last_not_of('\b') + 1);

    return output;
}


/// Prepends the string pre to the beginning of the string str and returns the whole string.
/// @param pre String to be prepended.
/// @param str original string.

string prepend(const string& pre, const string& text)
{
    ostringstream buffer;

    buffer << pre << text;

    return buffer.str();
}


/// Returns true if all the elements in a string list are numeric, and false otherwise.
/// @param v String list to be checked.

bool is_numeric_string_vector(const Tensor<string, 1>& string_list)
{
    for(Index i = 0; i < string_list.size(); i++)
    {
        if(!is_numeric_string(string_list[i])) return false;
    }

    return true;
}


bool has_numbers(const Tensor<string, 1>& string_list)
{
    for(Index i = 0; i < string_list.size(); i++)
    {
        if(is_numeric_string(string_list[i]))
        {
            cout << "The number is: " << string_list[i] << endl;
            return true;
        }
    }

    return false;
}


bool has_strings(const Tensor<string, 1>& string_list)
{
    for(Index i = 0; i < string_list.size(); i++)
    {
        if(!is_numeric_string(string_list[i])) return true;
    }

    return false;
}

/// Returns true if none element in a string list is numeric, and false otherwise.
/// @param v String list to be checked.

bool is_not_numeric(const Tensor<string, 1>& string_list)
{
    for(Index i = 0; i < string_list.size(); i++)
    {
        if(is_numeric_string(string_list[i])) return false;
    }

    return true;
}


/// Returns true if some the elements in a string list are numeric and some others are not numeric.
/// @param v String list to be checked.

bool is_mixed(const Tensor<string, 1>& string_list)
{
    unsigned count_numeric = 0;
    unsigned count_not_numeric = 0;

    for(Index i = 0; i < string_list.size(); i++)
    {
        if(is_numeric_string(string_list[i]))
        {
            count_numeric++;
        }
        else
        {
            count_not_numeric++;
        }
    }

    if(count_numeric > 0 && count_not_numeric > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/// Checks if a string is valid encoded in UTF-8 or not
/// @param string String to be checked.

void delete_non_printable_chars(string& text)
{
    typedef ctype<wchar_t> ctype;

    const ctype& ct = use_facet<ctype>(locale());

    text.erase(remove_if(text.begin(),
                         text.end(),
                         [&ct](wchar_t ch) {return !ct.is(ctype::print, ch);}),
                         text.end()) ;
}


/// Replaces a substring by another one in each element of this vector.
/// @param find_what String to be replaced.
/// @param replace_with String to be put instead.

void replace_substring(Tensor<string, 1>& vector, const string& find_what, const string& replace_with)
{
    const Index size = vector.dimension(0);

    for(Index i = 0; i < size; i++)
    {
        size_t position = 0;

        while((position = vector(i).find(find_what, position)) != string::npos)
        {
            vector(i).replace(position, find_what.length(), replace_with);

            position += replace_with.length();
        }
    }
}


void replace(string& source, const string& find_what, const string& replace_with)
{
    size_t position = 0;

    while((position = source.find(find_what, position)) != string::npos)
    {
        source.replace(position, find_what.length(), replace_with);

        position += replace_with.length();
    }
}


bool is_not_alnum (char &c)
{
    return (c < ' ' || c > '~');
}


void remove_not_alnum(string &str)
{
        str.erase(remove_if(str.begin(), str.end(), is_not_alnum), str.end());
}


bool find_string_in_tensor(Tensor<string, 1>& t, const string& val)
{
    for(Index i = 0; i < t.dimension(0); i++)
    {
        const string elem = t(i);

        if(elem == val) return true;
    }

    return false;
}


string get_word_from_token(string& token)
{
    string word = "";

    for(char& c : token)
    {
        if( c!=' ' && c!='=' )
        {
            word += c;
        }
        else
        {
            break;
        }
    }

    return word;
}


Tensor<string, 1> fix_write_expression_outputs(const string &str,
                                               const Tensor<string, 1> &outputs,
                                               const string &programming_languaje)
{
    Tensor<string,1> out;
    Tensor<string,1> tokens;
    Tensor<string,1> found_tokens;

    string token;
    string out_string;
    string new_variable;
    string old_variable;
    string expression = str;

    stringstream ss(expression);

    int option = 0;

    if(programming_languaje == "javascript") { option = 1; }
    else if(programming_languaje == "php")   { option = 2; }
    else if(programming_languaje == "python"){ option = 3; }
    else if(programming_languaje == "c")     { option = 4; }

    size_t dimension = outputs.dimension(0);

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{'){ break; }
        if(token.size() > 1 && token.back() != ';'){ token += ';'; }
        push_back_string(tokens, token);
    }

    for(Index i = 0; i < tokens.dimension(0); i++)
    {
        string s = tokens(i);
        string word = "";

        for(char& c : s)
        {
            if( c!=' ' && c!='=' ) word += c; 
            else break; 
        }

        if(word.size() > 1)
        {
            push_back_string(found_tokens, word);
        }
    }

    new_variable = found_tokens[found_tokens.size()-1];
    old_variable = outputs[dimension-1];

    if(new_variable != old_variable)
    {
        Index j = found_tokens.size();

        for(Index i = dimension; i --> 0;)
        {
            j -= 1;

            new_variable = found_tokens[j];
            old_variable = outputs[i];

            switch(option)
            {
                //JavaScript
                case 1:
                    out_string = "\tvar ";
                    out_string += old_variable;
                    out_string += " = ";
                    out_string += new_variable;
                    out_string += ";";
                    push_back_string(out, out_string);
                break;

                //Php
                case 2:
                    out_string = "$";
                    out_string += old_variable;
                    out_string += " = ";
                    out_string += "$";
                    out_string += new_variable;
                    out_string += ";";
                    push_back_string(out, out_string);
                break;

                //Python
                case 3:
                    out_string = old_variable;
                    out_string += " = ";
                    out_string += new_variable;
                    push_back_string(out, out_string);
                break;

                //C
                case 4:
                    out_string = "double ";
                    out_string += old_variable;
                    out_string += " = ";
                    out_string += new_variable;
                    out_string += ";";
                    push_back_string(out, out_string);
                break;

                default:
                break;
            }
        }
    }

    return out;
}

Tensor<Tensor<string,1>, 1> fix_input_output_variables(Tensor<string, 1>& inputs_names,
                                                       Tensor<string, 1>& outputs_names,
                                                       ostringstream& buffer_)
{
    //preparing output information

    Tensor<Tensor<string,1>, 1> output(3);

    ostringstream buffer;
    buffer << buffer_.str();

    Tensor<string, 1> outputs(outputs_names.dimension(0));
    Tensor<string, 1> inputs(inputs_names.dimension(0));
    Tensor<string,1> buffer_out;

    string output_name_aux;
    string input_name_aux;

    for(int i = 0; i < inputs_names.dimension(0); i++)
    {
        if(inputs_names[i].empty())
        {
            inputs(i) = "input_" + to_string(i);
            buffer << "\t" << to_string(i) + ") " << inputs_names(i) << endl;
        }
        else
        {
            input_name_aux = inputs_names[i];
            inputs(i) = replace_non_allowed_programming_expressions(input_name_aux);
            buffer << "\t" << to_string(i) + ") " << inputs(i) << endl;
        }
    }

    for(int i = 0; i < outputs_names.dimension(0); i++)
    {
        if(outputs_names[i].empty())
        {
            outputs(i) = "output_" + to_string(i);
        }
        else
        {
            output_name_aux = outputs_names[i];
            outputs(i) = replace_non_allowed_programming_expressions(output_name_aux);
        }
    }

    push_back_string(buffer_out, buffer.str());

    output(0) = inputs;
    output(1) = outputs;
    output(2) = buffer_out;

    return output;
}


string round_to_precision_string(type x, const int& precision)
{
    const type factor = type(pow(10, precision));

    const type rounded_value = (round(factor*x))/factor;

    stringstream ss;
    ss << fixed << setprecision(precision) << rounded_value;

    const string result = ss.str();

    return result;
}


Tensor<string,2> round_to_precision_string_matrix(Tensor<type,2> matrix, const int& precision)
{
    Tensor<string,2> matrix_rounded(matrix.dimension(0), matrix.dimension(1));

    const type factor = type(pow(10, precision));

    for(int i = 0; i< matrix_rounded.dimension(0); i++)
    {
        for(int j = 0; j < matrix_rounded.dimension(1); j++)
        {
            const type rounded_value = (round(factor*matrix(i,j)))/factor;

            stringstream ss;
            ss << fixed << setprecision(precision) << rounded_value;

            const string result = ss.str();

            matrix_rounded(i,j) = result;
        }
    }

    return matrix_rounded;
}


/// @todo clean this method Clang-tidy gives warnings.

Tensor<string,1> sort_string_tensor(Tensor<string, 1>& tensor)
{
    auto compare_string_length = [](const string& a, const string& b)
    {
        return a.length() > b.length();
    };

    vector<string> tensor_as_vector(tensor.data(), tensor.data() + tensor.size());
    
    sort(tensor_as_vector.begin(), tensor_as_vector.end(), compare_string_length);

    for(int i = 0; i < tensor.size(); i++)
    {
        tensor(i) = tensor_as_vector[i];
    }

    return tensor;
}


Tensor<string,1> concatenate_string_tensors(const Tensor<string, 1>& tensor_1, const Tensor<string, 1>& tensor_2)
{
    Tensor<string, 1> tensor = tensor_2;

    for(int i = 0; i < tensor_1.dimension(0); i++)
        push_back_string(tensor, tensor_1(i));

    return tensor;
}


/// changes the first apparition of all tokens found in the espression by adding the keyword before each of them.
/// @param input_string String whre changes will be done.
/// @param token_to_replace String to be put modyfied.
/// @param new_token String to be put instead.

void replace_substring_in_string (Tensor<string, 1>& tokens, string& espression, const string& keyword)
{
    string::size_type previous_pos = 0;

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        const string found_token = tokens(i);
        const string to_replace(found_token);
        const string newword = keyword + " " + found_token;

        string::size_type position = 0;

        while((position = espression.find(to_replace, position)) != string::npos)
        {
            if(position > previous_pos)
            {
                espression.replace(position, to_replace.length(), newword);
                position += newword.length();
                previous_pos = position;
                break;
            }
            else
            {
                position += newword.length();
            }
        }
    }
}


void display_progress_bar(const int& completed, const int& total)
{
    const int width = 100;
    const float progress = (float)completed / total;
    const int position = width * progress;

    cout << "[";

    for(int i = 0; i < width; i++)
    {
        if(i < position)
            cout << "=";

        else if(i == position)
            cout << ">";

        else cout << " ";
    }
    
    cout << "] " << int(progress * 100.0) << " %\r";

    cout.flush();
}


void create_alphabet()
{
/*
    string text_copy = text;

    sort(text_copy.begin(), text_copy.end());

    auto ip = unique(text_copy.begin(), text_copy.end());

    text_copy.resize(distance(text_copy.begin(), ip));

    alphabet.resize(text_copy.length());

    copy(execution::par,
        text_copy.begin(),
        text_copy.end(),
        alphabet.data());
*/
}


void encode_alphabet()
{
/*
    const Index rows_number = text.length();

    const Index raw_variables_number = alphabet.size();

    data_tensor.resize(rows_number, raw_variables_number);
    data_tensor.setZero();

    const Index length = text.length();

#pragma omp parallel for

    for(Index i = 0; i < length; i++)
    {
        const int word_index = get_alphabet_index(text[i]);

        data_tensor(i, word_index) = type(1);
    }
*/
}


Index get_alphabet_index(const char& ch) 
{
/*
    auto alphabet_begin = alphabet.data();
    auto alphabet_end = alphabet.data() + alphabet.size();

    const string str(1, ch);

    auto it = find(alphabet_begin, alphabet_end, str);

    if(it != alphabet_end)
    {
        Index index = it - alphabet_begin;
        return index;
    }
    else
    {
        return -1;
    }
*/
    return 0;
}


Tensor<type, 1> one_hot_encode(const string& ch) 
{
/*
    Tensor<type, 1> result(alphabet.size());

    result.setZero();

    const int word_index = get_alphabet_index(ch[0]);

    result(word_index) = type(1);

    return result;
*/
    return Tensor<type, 1>();
}


Tensor<type, 2> multiple_one_hot_encode(const string& phrase) 
{
/*
    const Index phrase_length = phrase.length();

    const Index alphabet_length = get_alphabet_length();

    Tensor<type, 2> result(phrase_length, alphabet_length);

    result.setZero();

    for(Index i = 0; i < phrase_length; i++)
    {
        const Index index = get_alphabet_index(phrase[i]);

        result(i, index) = type(1);
    }

    return result;
*/
    return Tensor<type, 2>();
}


string one_hot_decode(const Tensor<type, 1>& tensor) 
{
/*
    const Index length = alphabet.size();

    if(tensor.size() != length)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextGenerationAlphabet class.\n"
            << "string one_hot_decode(Tensor<type, 1>& tensor).\n"
            << "Tensor length must be equal to alphabet length.";

        throw runtime_error(buffer.str());
    }

    auto index = max_element(tensor.data(), tensor.data() + tensor.size()) - tensor.data();

    return alphabet(index);
*/
    return string();
}


string multiple_one_hot_decode(const Tensor<type, 2>& tensor) 
{
/*
    const Index length = alphabet.size();

    if(tensor.dimension(1) != length)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextGenerationAlphabet class.\n"
            << "string one_hot_decode(Tensor<type, 1>& tensor).\n"
            << "Tensor length must be equal to alphabet length.";

        throw runtime_error(buffer.str());
    }

    string result = "";

    for(Index i = 0; i < tensor.dimension(0); i++)
    {
        Tensor<type, 1> row = tensor.chip(i, 0);

        auto index = max_element(row.data(), row.data() + row.size()) - row.data();

        result += alphabet(index);
    }

    return result;
*/
    return string();
}


//Tensor<type, 2> str_to_input(const string& input_string)
//{
//    Tensor<type, 2> input_data = multiple_one_hot_encode(input_string);

//    Tensor<type, 2> flatten_input_data(1, input_data.size());

//    copy(input_data.data(),
//         input_data.data() + input_data.size(),
//         flatten_input_data.data());

//    return flatten_input_data;
//}


/// Calculate the total number of tokens in the documents.

Index count_tokens(const Tensor<Tensor<string, 1>, 1>& tokens)
{
    const Index documents_number = tokens.size();

    Index count = 0;

    for(Index i = 0; i < documents_number; i++)
    {
        count += tokens(i).size();
    }

    return count;
}


/// Returns a Tensor with all the words as elements keeping the order.

Tensor<string, 1> tokens_list(const Tensor<Tensor<string, 1>, 1>& documents_tokens)
{
    const Index documents_number = documents_tokens.size();

    const Index total_tokens_number = count_tokens(documents_tokens);

    Tensor<string, 1> total_tokens(total_tokens_number);

    Index position = 0;

    for(Index i = 0; i < documents_number; i++)
    {
        copy(documents_tokens(i).data(),
             documents_tokens(i).data() + documents_tokens(i).size(),
             total_tokens.data() + position);

        position += documents_tokens(i).size();
    }

    return total_tokens;
}


void to_lower(string& text)
{
    transform(text.begin(), text.end(), text.begin(), ::tolower);
}


/// Transforms all the letters of the documents into lower case.

void to_lower(Tensor<string, 1>& documents)
{
    const Index documents_number = documents.size();

    for(Index i = 0; i < documents_number; i++)
    {
        to_lower(documents(i));
    }
}


void to_lower(Tensor<Tensor<string, 1>, 1>& text)
{
    for(Index i = 0; i < text.size(); i++)
    {
        to_lower(text(i));
    }
}


Tensor<Tensor<string, 1>, 1> get_tokens(const Tensor<string, 1>& documents, const string& separator)
{
    const Index documents_number = documents.size();

    Tensor<Tensor<string, 1>, 1> tokens(documents_number);

    //#pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        tokens(i) = get_tokens(documents(i), separator);
    }

    return tokens;
}


void delete_blanks(Tensor<string, 1>& words)
{
    const Index words_number = words.size();

    const Index empty_number = count_empty(words);

    Tensor<string, 1> vector_copy(words);

    words.resize(words_number - empty_number);

    Index index = 0;

    for(Index i = 0; i < words_number; i++)
    {
        trim(vector_copy(i));

        if(!vector_copy(i).empty())
        {
            words(index) = vector_copy(i);
            index++;
        }
    }
}


void delete_blanks(Tensor<Tensor<string, 1>, 1>& documents_tokens)
{
    const Index documents_number = documents_tokens.size();

    #pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        const Index new_size = count_not_empty(documents_tokens(i));

        Tensor<string, 1> new_document_tokens(new_size);

        Index index = 0;

        for(Index j = 0; j < documents_tokens(i).size(); j++)
        {
            if(!documents_tokens(i)(j).empty())
            {
                new_document_tokens(index) = documents_tokens(i)(j);
                index++;
            }
        }

        documents_tokens(i) = new_document_tokens;
    }
}


Tensor<Tensor<string, 1>, 1> preprocess_language_documents(const Tensor<string, 1>& documents)
{
/*
    Tensor<string, 1> documents_copy(documents);

    to_lower(documents_copy);

    split_punctuation(documents_copy);

    delete_non_printable_chars(documents_copy);

    delete_extra_spaces(documents_copy);

    delete_non_alphanumeric(documents_copy);

    return get_tokens(documents_copy);
*/
    return Tensor<Tensor<string, 1>, 1>();
}


vector<pair<string, int>> count_words(const Tensor<string, 1>& total_tokens)
{
    unordered_map<string, int> count;

    for(Index i = 0; i < total_tokens.size(); i++)
        count[total_tokens(i)]++;

    vector<pair<string, int>> word_counts(count.begin(), count.end());

    sort(word_counts.begin(), word_counts.end(), [](const auto& a, const auto& b)
        {
            if(a.second != b.second)
                return a.second > b.second;
            else
                return a.first < b.first;
        }
    );

    return word_counts;
}


/// Deletes punctuation in documents.

void delete_punctuation(Tensor<string, 1>& documents) 
{
    replace_substring(documents, "�", " ");
    replace_substring(documents, "\"", " ");
    replace_substring(documents, ".", " ");
    replace_substring(documents, "!", " ");
    replace_substring(documents, "#", " ");
    replace_substring(documents, "$", " ");
    replace_substring(documents, "~", " ");
    replace_substring(documents, "%", " ");
    replace_substring(documents, "&", " ");
    replace_substring(documents, "/", " ");
    replace_substring(documents, "(", " ");
    replace_substring(documents, ")", " ");
    replace_substring(documents, "\\", " ");
    replace_substring(documents, "=", " ");
    replace_substring(documents, "?", " ");
    replace_substring(documents, "}", " ");
    replace_substring(documents, "^", " ");
    replace_substring(documents, "`", " ");
    replace_substring(documents, "[", " ");
    replace_substring(documents, "]", " ");
    replace_substring(documents, "*", " ");
    replace_substring(documents, "+", " ");
    replace_substring(documents, ",", " ");
    replace_substring(documents, ";", " ");
    replace_substring(documents, ":", " ");
    replace_substring(documents, "-", " ");
    replace_substring(documents, ">", " ");
    replace_substring(documents, "<", " ");
    replace_substring(documents, "|", " ");
    replace_substring(documents, "–", " ");
    replace_substring(documents, "Ø", " ");
    replace_substring(documents, "º", " ");
    replace_substring(documents, "°", " ");
    replace_substring(documents, "'", " ");
    replace_substring(documents, "ç", " ");
    replace_substring(documents, "✓", " ");
    replace_substring(documents, "|", " ");
    replace_substring(documents, "@", " ");
    replace_substring(documents, "#", " ");
    replace_substring(documents, "^", " ");
    replace_substring(documents, "*", " ");
    replace_substring(documents, "€", " ");
    replace_substring(documents, "¬", " ");
    replace_substring(documents, "•", " ");
    replace_substring(documents, "·", " ");
    replace_substring(documents, "”", " ");
    replace_substring(documents, "“", " ");
    replace_substring(documents, "´", " ");
    replace_substring(documents, "§", " ");
    replace_substring(documents, "_", " ");
    replace_substring(documents, ".", " ");

    delete_extra_spaces(documents);
}


/// Deletes consecutive extra spaces in documents.
/// @param documents Document to be proccesed.

void delete_extra_spaces(Tensor<string, 1>& documents) 
{
    Tensor<string, 1> new_documents(documents);

    for(Index i = 0; i < documents.size(); i++)
    {
        string::iterator new_end = unique(new_documents[i].begin(), new_documents[i].end(),
            [](char lhs, char rhs) { return(lhs == rhs) && (lhs == ' '); });

        new_documents[i].erase(new_end, new_documents[i].end());
    }

    documents = new_documents;
}


/// Deletes line breaks and tabulations
/// @param documents Document to be proccesed.

void delete_breaks_and_tabs(Tensor<string, 1>& documents) 
{
    for(Index i = 0; i < documents.size(); i++)
    {                
        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\n', ' ');
        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\t', ' ');
        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\f', ' ');
        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\r', ' ');
    }
}


/// Deletes unicode non printable characters

void delete_non_printable_chars(Tensor<string, 1>& documents) 
{
    for(Index i = 0; i < documents.size(); i++) 
        delete_non_printable_chars(documents(i));
}


/// Splits punctuation symbols in documents.

void split_punctuation(Tensor<string, 1>& documents) 
{
    replace_substring(documents, "�", " � ");
    replace_substring(documents, "\"", " \" ");
    replace_substring(documents, ".", " . ");
    replace_substring(documents, "!", " ! ");
    replace_substring(documents, "#", " # ");
    replace_substring(documents, "$", " $ ");
    replace_substring(documents, "~", " ~ ");
    replace_substring(documents, "%", " % ");
    replace_substring(documents, "&", " & ");
    replace_substring(documents, "/", " / ");
    replace_substring(documents, "(", " ( ");
    replace_substring(documents, ")", " ) ");
    replace_substring(documents, "\\", " \\ ");
    replace_substring(documents, "=", " = ");
    replace_substring(documents, "?", " ? ");
    replace_substring(documents, "}", " } ");
    replace_substring(documents, "^", " ^ ");
    replace_substring(documents, "`", " ` ");
    replace_substring(documents, "[", " [ ");
    replace_substring(documents, "]", " ] ");
    replace_substring(documents, "*", " * ");
    replace_substring(documents, "+", " + ");
    replace_substring(documents, ",", " , ");
    replace_substring(documents, ";", " ; ");
    replace_substring(documents, ":", " : ");
    replace_substring(documents, "-", " - ");
    replace_substring(documents, ">", " > ");
    replace_substring(documents, "<", " < ");
    replace_substring(documents, "|", " | ");
    replace_substring(documents, "–", " – ");
    replace_substring(documents, "Ø", " Ø ");
    replace_substring(documents, "º", " º ");
    replace_substring(documents, "°", " ° ");
    replace_substring(documents, "'", " ' ");
    replace_substring(documents, "ç", " ç ");
    replace_substring(documents, "✓", " ✓ ");
    replace_substring(documents, "|", " | ");
    replace_substring(documents, "@", " @ ");
    replace_substring(documents, "#", " # ");
    replace_substring(documents, "^", " ^ ");
    replace_substring(documents, "*", " * ");
    replace_substring(documents, "€", " € ");
    replace_substring(documents, "¬", " ¬ ");
    replace_substring(documents, "•", " • ");
    replace_substring(documents, "·", " · ");
    replace_substring(documents, "”", " ” ");
    replace_substring(documents, "“", " “ ");
    replace_substring(documents, "´", " ´ ");
    replace_substring(documents, "§", " § ");
    replace_substring(documents, "_", " _ ");
    replace_substring(documents, ".", " . ");

    delete_extra_spaces(documents);
}


void delete_non_alphanumeric(Tensor<string, 1>& documents)
{
    Tensor<string, 1> new_documents(documents);

    for(Index i = 0; i < documents.size(); i++)
    {
        new_documents[i].erase(remove_if(new_documents[i].begin(), new_documents[i].end(), is_not_alnum), new_documents[i].end());
    }

    documents = new_documents;
}


/// Joins a string tensor into a string
/// @param token String tensor we will join

string to_string(Tensor<string, 1> token) 
{
    string word;

    for(Index i = 0; i < token.size() - 1; i++)
        word += token(i) + " ";
    word += token(token.size() - 1);

    return word;
}


/// Join the words Tensors into strings documents
/// @param tokens Tensor of Tensor of words we want to join

Tensor<string, 1> detokenize(const Tensor<Tensor<string, 1>, 1>& tokens)
{
    const Index documents_number = tokens.size();

    Tensor<string, 1> new_documents(documents_number);

    for(Index i = 0; i < documents_number; i++)
    {
        new_documents[i] = to_string(tokens(i));
    }

    return new_documents;
}


void filter_not_equal_to(Tensor<string, 1>& document, const Tensor<string, 1>& delete_words) 
{
    for(Index i = 0; i < document.size(); i++)
    {
        const Index tokens_number = count_tokens(document(i), " ");
        const Tensor<string, 1> tokens = get_tokens(document(i), " ");

        string result;

        for(Index j = 0; j < tokens_number; j++)
        {
            if(!contains(delete_words, tokens(j)))
            {
                result += tokens(j) + " ";
            }
        }

        document(i) = result;
    }
}


/// Delete the words we want from the documents
/// @param delete_words Tensor of words we want to delete

void delete_words(Tensor<Tensor<string, 1>, 1>& documents_words, const Tensor<string, 1>& deletion_words)
{
    const Index documents_number = documents_words.size();

    const Index deletion_words_number = deletion_words.size();

    #pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        for(Index j = 0; j < documents_words(i).size(); j++)
        {
            const string word = documents_words(i)(j);

            for(Index k = 0; k < deletion_words_number; k++)
            {
                if(word == deletion_words(k))
                {
                    documents_words(i)(j).clear();

                    continue;
                }
            }
        }
    }
}


/// Delete short words from the documents
/// @param minimum_length Minimum length of the words that new documents must have(including herself)

void delete_short_long_words(Tensor<Tensor<string,1>,1>& documents_words,
                        const Index& minimum_length,
                        const Index& maximum_length)
{
    const Index documents_number = documents_words.size();

    #pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        for(Index j = 0; j < documents_words(i).size(); j++)
        {
            const Index length = documents_words(i)(j).length();

            if(length <= minimum_length || length >= maximum_length)
            {
                documents_words(i)(j).clear();
            }
        }
    }
}


/// Delete the numbers of the documents.

void delete_numbers(Tensor<Tensor<string,1>,1>& documents_words)
{
    const Index documents_number = documents_words.size();

    #pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        for(Index j = 0; j < documents_words(i).size(); j++)
        {
            if(is_numeric_string(documents_words(i)(j)))
            {
                documents_words(i)(j).clear();
            }
        }
    }
}


/// Remove emails from documents.

void delete_emails(Tensor<Tensor<string,1>,1>& documents)
{
    const Index documents_number = documents.size();

    #pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        Tensor<string, 1> document = documents(i);

        for(Index j = 0; j < document.size(); j++)
        {
            /*
            Tensor<string, 1> tokens = get_tokens(document(j));

            string result;

            for(Index k = 0; k < tokens.size(); k++)
            {
                if(!is_email(tokens(k)))
                {
                    result += tokens(k) + " ";
                }
            }

            document(j) = result;
*/
        }

        documents(i) = document;
    }
}


/// Remove the accents of the vowels in the documents.

void replace_accented_words(Tensor<Tensor<string,1>, 1>& documents)
{
    const Index documents_size = documents.size();

    for(Index i = 0; i < documents_size; i++)
    {
        const Index document_size = documents(i).size();

        for(Index j = 0; j < document_size; j++)
        {
            replace_accented_words(documents(i)(j));
        }
    }
}


/// Remove the accents of the vowels of a word.

void replace_accented_words(string& word)
{
    replace(word, "á", "a");
    replace(word, "é", "e");
    replace(word, "í", "i");
    replace(word, "ó", "o");
    replace(word, "ú", "u");

    replace(word, "Á", "A");
    replace(word, "É", "E");
    replace(word, "Í", "I");
    replace(word, "Ó", "O");
    replace(word, "Ú", "U");

    replace(word, "ä", "a");
    replace(word, "ë", "e");
    replace(word, "ï", "i");
    replace(word, "ö", "o");
    replace(word, "ü", "u");

    replace(word, "â", "a");
    replace(word, "ê", "e");
    replace(word, "î", "i");
    replace(word, "ô", "o");
    replace(word, "û", "u");

    replace(word, "à", "a");
    replace(word, "è", "e");
    replace(word, "ì", "i");
    replace(word, "ò", "o");
    replace(word, "ù", "u");

    replace(word, "ã", "a");
    replace(word, "õ", "o");
}


Tensor<string,1> get_r1_r2(const string& word, const Tensor<string,1>& vowels)
{
    const Index word_length = word.length();

    string r1 = "";

    for(Index i = 1; i < word_length; i++)
    {
        if(!contains(vowels, word.substr(i,1)) && contains(vowels, word.substr(i-1,1)))
        {
            r1 = word.substr(i+1);
            break;
        }
    }

    const Index r1_length = r1.length();

    string r2 = "";

    for(Index i = 1; i < r1_length; i++)
    {
        if(!contains(vowels, r1.substr(i,1)) && contains(vowels, r1.substr(i-1,1)))
        {
            r2 = r1.substr(i+1);
            break;
        }
    }

    Tensor<string,1> r1_r2(2);

    r1_r2[0] = r1;
    r1_r2[1] = r2;

    return r1_r2;
}


string get_rv(const string& word, const Tensor<string,1>& vowels)
{
    string rv = "";

    const Index word_lenght = word.length();

    if(word_lenght >= 2)
    {
        if(!contains(vowels, word.substr(1,1)))
        {
            for(Index i = 2; i < word_lenght; i++)
            {
                if(contains(vowels, word.substr(i,1)))
                {
                    rv = word.substr(i+1);
                    break;
                }
            }
        }
        else if(contains(vowels, word.substr(0,1)) && contains(vowels, word.substr(1,1)))
        {
            for(Index i = 2; i < word_lenght; i++)
            {
                if(!contains(vowels, word.substr(i,1)))
                {
                    rv = word.substr(i+1);
                    break;
                }
            }
        }
        else
        {
            rv = word.substr(3);
        }
    }

    return rv;
}


/// Create a word bag that contains all the unique words of the documents,
/// their frequencies and their percentages in descending order

WordBag calculate_word_bag(const Tensor<string,1>& words)
{
    WordBag word_bag;

    const Tensor<Index, 1> count = count_unique(words);

    const Tensor<Index, 1> descending_rank = calculate_rank_greater(count.cast<type>());

    word_bag.words = sort_by_rank(get_unique_elements(words), descending_rank);

    word_bag.frequencies = sort_by_rank(count, descending_rank);

    const Tensor<Index, 0> total_frequencies = word_bag.frequencies.sum();

    word_bag.percentages = 100*word_bag.frequencies.cast<double>()/double(total_frequencies(0));

    return word_bag;
}


/// Create a word bag that contains the unique words that appear a minimum number
/// of times in the documents, their frequencies and their percentages in descending order.
/// @param minimum_frequency Minimum frequency that words must have.

//WordBag calculate_word_bag_minimum_frequency(const Tensor<Tensor<string,1>,1>& tokens,
//                                             const Index& minimum_frequency)
// {
//     WordBag word_bag = calculate_word_bag(tokens);

//     Tensor<string,1> words = word_bag.words;
//     Tensor<Index,1> frequencies = word_bag.frequencies;
//     Tensor<double,1> percentages = word_bag.percentages;

//     const Tensor<Index,1> indices = get_indices_less_than(frequencies, minimum_frequency);

//     delete_indices(words, indices);
//     delete_indices(frequencies, indices);
//     delete_indices(percentages, indices);

//     word_bag.words = words;
//     word_bag.frequencies = frequencies;
//     word_bag.percentages = percentages;

//     return word_bag;
// }


/// Create a word bag that contains the unique words that appear a minimum percentage
/// in the documents, their frequencies and their percentages in descending order.
/// @param minimum_percentage Minimum percentage of occurrence that words must have.

// WordBag calculate_word_bag_minimum_percentage(const Tensor<Tensor<string,1>,1>& tokens,
//                                               const double& minimum_percentage)
// {
//     WordBag word_bag = calculate_word_bag(tokens);

//     Tensor<string,1> words = word_bag.words;
//     Tensor<Index,1> frequencies = word_bag.frequencies;
//     Tensor<double,1> percentages = word_bag.percentages;

//     const Tensor<Index,1> indices = get_indices_less_than(percentages, minimum_percentage);

//     delete_indices(words, indices);
//     delete_indices(frequencies, indices);
//     delete_indices(percentages, indices);

//     word_bag.words = words;
//     word_bag.frequencies = frequencies;
//     word_bag.percentages = percentages;

//     return word_bag;
// }


/// Create a word bag that contains the unique words that appear a minimum ratio
/// of frequency in the documents, their frequencies and their percentages in descending order.
/// @param minimum_ratio Minimum ratio of frequency that words must have.

// WordBag calculate_word_bag_minimum_ratio(const Tensor<Tensor<string,1>,1>& tokens,
//                                          const double& minimum_ratio)
// {
//     WordBag word_bag = calculate_word_bag(tokens);

//     Tensor<string,1> words = word_bag.words;
//     Tensor<Index,1> frequencies = word_bag.frequencies;
//     Tensor<double,1> percentages = word_bag.percentages;

//     const Tensor<Index,0> frequencies_sum = frequencies.sum();

//     const Tensor<double,1> ratios = frequencies.cast<double>()/double(frequencies_sum(0));

//     const Tensor<Index, 1> indices = get_indices_less_than(ratios, minimum_ratio);

//     delete_indices(words, indices);
//     delete_indices(frequencies, indices);
//     delete_indices(percentages, indices);

//     word_bag.words = words;
//     word_bag.frequencies = frequencies;
//     word_bag.percentages = percentages;

//     return word_bag;
// }


/// Create a word bag that contains the unique most frequent words whose sum
/// of frequencies is less than the specified number, their frequencies
/// and their percentages in descending order.
/// @param total_frequency Maximum cumulative frequency that words must have.

// WordBag calculate_word_bag_total_frequency(const Tensor<Tensor<string,1>,1>& tokens,
//                                            const Index& total_frequency)
// {
//     WordBag word_bag = calculate_word_bag(tokens);

//     const Tensor<string,1> words = word_bag.words;
//     const Tensor<Index, 1> frequencies = word_bag.frequencies;

//     Tensor<Index, 1> cumulative_frequencies = frequencies.cumsum(0);

//     Index i;

//     for( i = 0; i < frequencies.size(); i++)
//     {
//         if(cumulative_frequencies(i) >= total_frequency)
//             break;
//     }

//     word_bag.words = get_first(words, i);
//     word_bag.frequencies = get_first(frequencies, i);

//     return word_bag;
// }


/// Create a word bag that contains a maximum number of the unique most
/// frequent words, their frequencies and their percentages in descending order.
/// @param maximum_size Maximum size of words Tensor.

// WordBag calculate_word_bag_maximum_size(const Tensor<Tensor<string,1>,1>& tokens,
//                                         const Index& maximum_size)
// {
//     WordBag word_bag = calculate_word_bag(tokens);

//     const Tensor<string, 1> words = word_bag.words;
//     const Tensor<Index ,1> frequencies = word_bag.frequencies;

//     word_bag.words = get_first(words, maximum_size);
//     word_bag.frequencies = get_first(frequencies, maximum_size);

//     return word_bag;
// }


/// Returns weights.

Index calculate_weight(const Tensor<string, 1>& document_words, const WordBag& word_bag)
{
    Index weight = 0;

    const Tensor<string, 1> bag_words = word_bag.words;

    const Tensor<Index, 1> bag_frequencies = word_bag.frequencies;

    for(Index i = 0; i < document_words.size(); i++)
    {
        for(Index j = 0; j < word_bag.size(); j++)
        {
            if(document_words[i] == bag_words[j])
            {
                weight += bag_frequencies[j];
            }
        }
    }

    return weight;
}


/// Returns the documents easier to work with them

Tensor<Tensor<string,1>,1> preprocess(const Tensor<string,1>& documents)
{
/*
    Tensor<string,1> documents_copy(documents);

    to_lower(documents_copy);

    delete_punctuation(documents_copy);

    delete_non_printable_chars(documents_copy);

    delete_extra_spaces(documents_copy);

    delete_non_alphanumeric(documents_copy);

    Tensor<Tensor<string,1>,1> tokens = get_tokens(documents_copy);

    delete_stop_words(tokens);

    delete_short_long_words(tokens, short_words_length, long_words_length);

    replace_accented_words(tokens);

    delete_emails(tokens);

    //tokens = apply_stemmer(tokens); deleted recover from git

    delete_numbers(tokens);

    delete_blanks(tokens);

    return tokens;
*/
    return Tensor<Tensor<string,1>,1>();
}


Tensor<Tensor<string,1>,1> preprocess_language_model(const Tensor<string,1>& documents)
{
/*
    Tensor<string,1> documents_copy(documents);

    to_lower(documents_copy);

    split_punctuation(documents_copy);

    delete_non_printable_chars(documents_copy);

    delete_extra_spaces(documents_copy);

    delete_non_alphanumeric(documents_copy);

    Tensor<Tensor<string,1>,1> tokens = get_tokens(documents_copy);

    delete_emails(tokens);

    delete_blanks(tokens);

    return tokens;
*/
    return Tensor<Tensor<string,1>, 1>();
}


/// Returns a Tensor with the number of words that each document contains.

Tensor<Index, 1> get_words_number(const Tensor<Tensor<string,1>,1>& tokens)
{
    const Index documents_number = tokens.size();

    Tensor<Index, 1> words_number(documents_number);

    for(Index i = 0; i < documents_number; i++)
    {
        words_number(i) = tokens(i).size();
    }

    return words_number;
}


/// Returns a Tensor with the number of sentences that each document contains.

Tensor<Index, 1> get_sentences_number(const Tensor<string, 1>& documents)
{
    const Index documents_number = documents.size();

    Tensor<Index, 1> sentences_number(documents_number);

    for(Index i = 0; i < documents_number; i++)
    {
        sentences_number(i) = count_tokens(documents(i), ".");
    }

    return sentences_number;
}


/// Returns a Tensor with the percentage of presence in the documents with respect to all.
/// @param words_name Tensor of words from which you want to know the percentage of presence.

Tensor<double, 1> get_words_presence_percentage(const Tensor<Tensor<string, 1>, 1>& tokens,
                                                const Tensor<string, 1>& words_name)
{
    Tensor<double, 1> word_presence_percentage(words_name.size());

    for(Index i = 0; i < words_name.size(); i++)
    {
        Index sum = 0;

        for(Index j = 0; j < tokens.size(); j++)
        {
            if(contains(tokens(j),words_name(i) ))
            {
                sum = sum + 1;
            }
        }

        word_presence_percentage(i) = double(sum)*(double(100.0/tokens.size()));
    }

    return word_presence_percentage;
}


/// This function calculates the frequency of sets of consecutive words in all documents.
/// @param minimum_frequency Minimum frequency that a word must have to obtain its combinations.
/// @param combinations_length Words number of the combinations from 2.
/*
Tensor<string, 2> calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>& tokens,
                                                       const Index& minimum_frequency,
                                                       const Index& combinations_length)
{
    const Tensor<string, 1> words = tokens_list(tokens);

    const WordBag top_word_bag = calculate_word_bag_minimum_frequency(tokens, minimum_frequency);
    const Tensor<string, 1> words_name = top_word_bag.words;

    if(words_name.size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "Tensor<string, 2> calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>& tokens,"
                  "const Index& minimum_frequency,"
                  "const Index& combinations_length)  method."
               << "Words number must be greater than 1.\n";

        throw runtime_error(buffer.str());
    }

    if(combinations_length < 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "Tensor<string, 2> calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>& tokens,"
                  "const Index& minimum_frequency,"
                  "const Index& combinations_length) method."
               << "Length of combinations not valid, must be greater than 1";

        throw runtime_error(buffer.str());
    }

    Index combinated_words_size = 0;

    for(Index i = 0; i < words_name.size(); i++)
    {
        for(Index j = 0; j < words.size()-1; j++)
        {
            if(words_name[i] == words[j])
            {
                combinated_words_size++;
            }
        }
    }

    Tensor<string, 1> combinated_words(combinated_words_size);

    Index index = 0;

    for(Index i = 0; i < words_name.size(); i++)
    {
        for(Index j = 0; j < words.size()-1; j++)
        {
            if(words_name[i] == words[j])
            {
                string word = words[j];

                for(Index k = 1; k < combinations_length; k++)
                {
                    word += " " + words[j+k];
                }

                combinated_words[index] = word;

                index++;
            }
        }
    }

//    const Tensor<string, 1> combinated_words_frequency = to_string_tensor( ( count_unique( combinated_words ) ) );

//    Tensor<string, 2> combinated_words_frequency_matrix(combinated_words_frequency.size(),2);

    combinated_words_frequency_matrix.chip(0,1) = get_unique_elements(combinated_words),"Combinated words");
    combinated_words_frequency_matrix.chip(1,0) = combinated_words_frequency,"Frequency");

    combinated_words_frequency_matrix = combinated_words_frequency_matrix.sort_descending_strings(1);

//    return(combinated_words_frequency_matrix);

    return Tensor<string,2>();
}
*/

/// Returns the correlations of words that appear a minimum percentage of times
/// with the targets in descending order.
/// @param minimum_percentage Minimum percentage of frequency that the word must have.
/*
Tensor<string, 2> top_words_correlations(const Tensor<Tensor<string, 1>, 1>& tokens,
                                         const double& minimum_percentage,
                                         const Tensor<Index, 1>& targets)
{
    const WordBag top_word_bag = calculate_word_bag_minimum_percentage(tokens, minimum_percentage);
    const Tensor<string, 1> words_name = top_word_bag.words;

    if(words_name.size() == 0)
    {
        cout << "There are no words with such high percentage of appearance" << endl;
    }

    Tensor<string, 1> new_documents(tokens.size());

    for(size_t i = 0; i < tokens.size(); i++)
    {
      new_documents[i] = tokens[i].Tensor_to_string(';');
    }

    const Matrix<double> top_words_binary_matrix;// = new_documents.get_unique_binary_matrix(';', words_name).to_double_matrix();

    Tensor<double> correlations(top_words_binary_matrix.get_raw_variables_number());

    for(size_t i = 0; i < top_words_binary_matrix.get_raw_variables_number(); i++)
    {
        correlations[i] = linear_correlation(top_words_binary_matrix.get_column(i), targets.to_double_Tensor());
    }

    Matrix<string> top_words_correlations(correlations.size(),2);

    top_words_correlations.set_column(0,top_words_binary_matrix.get_header(),"Words");

    top_words_correlations.set_column(1,correlations.to_string_Tensor(),"Correlations");

    top_words_correlations = top_words_correlations.sort_descending_strings(1);

    return(top_words_correlations);


    return Tensor<string, 2>();
}
*/

/// Generates a text output based on the neural network and some input letters given by the user.
/// @param text_generation_alphabet TextGenerationAlphabet object used for the text generation model
/// @param input_string Input string given by the user
/// @param max_length Maximum length of the returned string
/// @param one_word Boolean, if true returns just one word, if false returns a phrase
/*
string calculate_text_outputs(TextGenerationAlphabet& text_generation_alphabet,
                              const string& input_string,
                              const Index& max_length,
                              const bool& one_word)
{
    string result = one_word
? generate_word(text_generation_alphabet, input_string, max_length)
: generate_phrase(text_generation_alphabet, input_string, max_length);

    return result;
}


/// @todo TEXT GENERATION

string generate_word(TextGenerationAlphabet& text_generation_alphabet, const string& first_letters, const Index& length)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: NeuralNetwork class.\n"
           << "string generate_word(TextGenerationAlphabet&, const string&, const Index&) method.\n"
           << "This method is not implemented yet.\n";

    throw runtime_error(buffer.str());

    return string();

    // Under development

    //    const Index alphabet_length = text_generation_alphabet.get_alphabet_length();

    //    if(first_letters.length()*alphabet_length != get_inputs_number())
    //    {
    //        ostringstream buffer;

    //        buffer << "OpenNN Exception: NeuralNetwork class.\n"
    //               << "string generate_word(TextGenerationAlphabet&, const string&, const Index&) method.\n"
    //               << "Input string length must be equal to " << int(get_inputs_number()/alphabet_length) << "\n";

    //        throw runtime_error(buffer.str());
    //    }


    //    string result = first_letters;

    //    // 1. Input letters to one hot encode

    //    Tensor<type, 2> input_data = text_generation_alphabet.multiple_one_hot_encode(first_letters);

    //    Tensor<Index, 1> input_dimensions = get_dimensions(input_data);

    //    Tensor<string, 1> punctuation_signs(6); // @todo change for multiple letters predicted

    //    punctuation_signs.setValues({" ",",",".","\n",":",";"});

    //    // 2. Loop for forecasting the following letter in function of the last letters

    //    do{
    //        Tensor<type, 2> output = calculate_outputs(input_data.data(), input_dimensions);

    //        string letter = text_generation_alphabet.multiple_one_hot_decode(output);

    //        if(!contains(punctuation_signs, letter))
    //        {
    //            result += letter;

    //            input_data = text_generation_alphabet.multiple_one_hot_encode(result.substr(result.length() - first_letters.length()));
    //        }

    //    }while(result.length() < length);

    //    return result;
}


/// @todo TEXT GENERATION

string generate_phrase(TextGenerationAlphabet& text_generation_alphabet, const string& first_letters, const Index& length)
{
    const Index alphabet_length = text_generation_alphabet.get_alphabet_length();

    if(first_letters.length()*alphabet_length != get_inputs_number())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: NeuralNetwork class.\n"
               << "string generate_word(TextGenerationAlphabet&, const string&, const Index&) method.\n"
               << "Input string length must be equal to " << int(get_inputs_number()/alphabet_length) << "\n";

        throw runtime_error(buffer.str());
    }

    string result = first_letters;

    Tensor<type, 2> input_data = text_generation_alphabet.multiple_one_hot_encode(first_letters);

    Tensor<Index, 1> input_dimensions = get_dimensions(input_data);

    do{
        Tensor<type, 2> input_data(get_inputs_number(), 1);
        input_data.setZero();
        Tensor<Index, 1> input_dimensions = get_dimensions(input_data);

        Tensor<type, 2> output = calculate_outputs(input_data.data(), input_dimensions);

        string letter = text_generation_alphabet.multiple_one_hot_decode(output);

        result += letter;

        input_data = text_generation_alphabet.multiple_one_hot_encode(result.substr(result.length() - first_letters.length()));

    }while(result.length() < length);

    return result;

    return string();
}


/// @todo TEXT GENERATION Explain.

TextGenerationAlphabet::TextGenerationAlphabet()
{
}


TextGenerationAlphabet::TextGenerationAlphabet(const string& new_text)
{
    text = new_text;

    set();
}

TextGenerationAlphabet::~TextGenerationAlphabet()
{
}


string TextGenerationAlphabet::get_text()
{
    return text;
}


Tensor<type, 2> TextGenerationAlphabet::get_data_tensor()
{
    return data_tensor;
}


Tensor<string, 1> TextGenerationAlphabet::get_alphabet()
{
    return alphabet;
}


Index TextGenerationAlphabet::get_alphabet_length()
{
    return alphabet.size();
}


void TextGenerationAlphabet::set()
{
    preprocess();

    create_alphabet();

    encode_alphabet();
}


void TextGenerationAlphabet::set_text(const string& new_text)
{
    text = new_text;
}


void TextGenerationAlphabet::set_data_tensor(const Tensor<type, 2>& new_data_tensor)
{
    data_tensor = new_data_tensor;
}


void TextGenerationAlphabet::set_alphabet(const Tensor<string, 1>& new_alphabet)
{
    alphabet = new_alphabet;
}


void TextGenerationAlphabet::print()
{
    const Index alphabet_length = get_alphabet_length();

    cout << "Alphabet characters number: " << alphabet_length << endl;

    cout << "Alphabet characters:\n" << alphabet << endl;

    if(alphabet_length <= 10 && data_tensor.dimension(1) <= 20)
    {
        cout << "Data tensor:\n" << data_tensor << endl;
    }
}


void TextGenerationAlphabet::create_alphabet()
{
    string text_copy = text;

    sort(text_copy.begin(), text_copy.end());

    auto ip = unique(text_copy.begin(), text_copy.end());

    text_copy.resize(distance(text_copy.begin(), ip));

    alphabet.resize(text_copy.length());

    copy(execution::par,
        text_copy.begin(),
        text_copy.end(),
        alphabet.data());
}


void TextGenerationAlphabet::encode_alphabet()
{
    const Index rows_number = text.length();

    const Index raw_variables_number = alphabet.size();

    data_tensor.resize(rows_number, raw_variables_number);
    data_tensor.setZero();

    const Index length = text.length();

#pragma omp parallel for

    for(Index i = 0; i < length; i++)
    {
        const int word_index = get_alphabet_index(text[i]);

        data_tensor(i, word_index) = type(1);
    }
}


void TextGenerationAlphabet::preprocess()
{
    TextAnalytics ta;

    ta.replace_accented_words(text);

    transform(text.begin(), text.end(), text.begin(), ::tolower); // To lower
}


Index TextGenerationAlphabet::get_alphabet_index(const char& ch)
{
    auto alphabet_begin = alphabet.data();
    auto alphabet_end = alphabet.data() + alphabet.size();

    const string str(1, ch);

    auto it = find(alphabet_begin, alphabet_end, str);

    if(it != alphabet_end)
    {
        Index index = it - alphabet_begin;
        return index;
    }
    else
    {
        return -1;
    }
}


Tensor<type, 1> TextGenerationAlphabet::one_hot_encode(const string &ch)
{
    Tensor<type, 1> result(alphabet.size());

    result.setZero();

    const int word_index = get_alphabet_index(ch[0]);

    result(word_index) = type(1);

    return result;
}


Tensor<type, 2> TextGenerationAlphabet::multiple_one_hot_encode(const string &phrase)
{
    const Index phrase_length = phrase.length();

    const Index alphabet_length = get_alphabet_length();

    Tensor<type, 2> result(phrase_length, alphabet_length);

    result.setZero();

    for(Index i = 0; i < phrase_length; i++)
    {
        const Index index = get_alphabet_index(phrase[i]);

        result(i, index) = type(1);
    }

    return result;
}


string TextGenerationAlphabet::one_hot_decode(const Tensor<type, 1>& tensor)
{
    const Index length = alphabet.size();

    if(tensor.size() != length)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextGenerationAlphabet class.\n"
               << "string one_hot_decode(Tensor<type, 1>& tensor).\n"
               << "Tensor length must be equal to alphabet length.";

        throw runtime_error(buffer.str());
    }

    auto index = max_element(tensor.data(), tensor.data() + tensor.size()) - tensor.data();

    return alphabet(index);
}


string TextGenerationAlphabet::multiple_one_hot_decode(const Tensor<type, 2>& tensor)
{
    const Index length = alphabet.size();

    if(tensor.dimension(1) != length)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextGenerationAlphabet class.\n"
               << "string one_hot_decode(Tensor<type, 1>& tensor).\n"
               << "Tensor length must be equal to alphabet length.";

        throw runtime_error(buffer.str());
    }

    string result = "";

    for(Index i = 0; i < tensor.dimension(0); i++)
    {
        Tensor<type, 1> row = tensor.chip(i, 0);

        auto index = max_element(row.data(), row.data() + row.size()) - row.data();

        result += alphabet(index);
    }

    return result;
}


Tensor<type, 2> TextGenerationAlphabet::str_to_input(const string &input_string)
{
    Tensor<type, 2> input_data = multiple_one_hot_encode(input_string);

    Tensor<type, 2> flatten_input_data(1, input_data.size());

    copy(execution::par,
        input_data.data(),
        input_data.data() + input_data.size(),
        flatten_input_data.data());

    return flatten_input_data;
}

}
*/

void print_tokens(const Tensor<Tensor<string,1>,1>& tokens)
{
    for(Index i = 0; i < tokens.size(); i++)
    {
        for(Index j = 0; j < tokens(i).size(); j++)
        {
            cout << tokens(i)(j) << " - ";
        }

        cout << endl;
    }
}

bool is_vowel(char ch)
{
    return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u';
}

bool ends_with(const string& word, const string& suffix)
{
    return word.length() >= suffix.length() && word.compare(word.length() - suffix.length(), suffix.length(), suffix) == 0;
}

int measure(const string& word)
{
    int count = 0;
    bool vowel_seen = false;
    for (char ch : word)
    {
        if(is_vowel(ch))
        {
            vowel_seen = true;
        } else if(vowel_seen)
        {
            count++;
            vowel_seen = false;
        }
    }

    return count;
}


bool contains_vowel(const string& word)
{
    for (char ch : word)
    {
        if(is_vowel(ch))
            return true;
    }

    return false;
}


bool is_double_consonant(const string& word)
{
    if(word.length() < 2) return false;

    char last = word[word.length() - 1];

    char second_last = word[word.length() - 2];

    return last == second_last && !is_vowel(last);
}


bool is_consonant_vowel_consonant(const string& word)
{
    if(word.length() < 3) return false;
    char last = word[word.length() - 1];
    char second_last = word[word.length() - 2];
    char third_last = word[word.length() - 3];
    return !is_vowel(last) && is_vowel(second_last) && !is_vowel(third_last) && last != 'w' && last != 'x' && last != 'y';
}


// Porter Stemmer algorithm

string stem(const string& word)
{
    string result = word;

    if(result.length() <= 2) return result;

    // Convert to lowercase
    transform(result.begin(), result.end(), result.begin(), ::tolower);

    // Step 1a
    if(ends_with(result, "sses"))
    {
        result = result.substr(0, result.length() - 2);
    }
    else if(ends_with(result, "ies"))
    {
        result = result.substr(0, result.length() - 2);
    }
    else if(ends_with(result, "ss"))
    {
        // Do nothing
    }
    else if(ends_with(result, "s"))
    {
        result = result.substr(0, result.length() - 1);
    }

    // Step 1b

    if(ends_with(result, "eed"))
    {
        if(measure(result.substr(0, result.length() - 3)) > 0)
        {
            result = result.substr(0, result.length() - 1);
        }
    }
    else if((ends_with(result, "ed") || ends_with(result, "ing"))
             && contains_vowel(result.substr(0, result.length() - 2)))
    {
        result = result.substr(0, result.length() - (ends_with(result, "ed") ? 2 : 3));
        if(ends_with(result, "at") || ends_with(result, "bl") || ends_with(result, "iz"))
        {
            result += "e";
        }
        else if(is_double_consonant(result))
        {
            result = result.substr(0, result.length() - 1);
        }
        else if(measure(result) == 1 && is_consonant_vowel_consonant(result))
        {
            result += "e";
        }
    }

    // Step 1c

    if(ends_with(result, "y") && contains_vowel(result.substr(0, result.length() - 1)))
    {
        result[result.length() - 1] = 'i';
    }

    // Additional steps can be added here following the Porter Stemmer algorithm

    return result;
}


void stem(Tensor<string, 1>& words)
{

    for(Index i = 0; i < words.size(); i++)
    {
        words(i) = stem(words(i));
    }
}


void stem(Tensor<Tensor<string, 1>, 1>& words)
{
    #pragma omp parallel for

    for(Index i = 0; i < words.size(); i++)
    {
        for(Index j = 0; j < words(i).size(); j++)
        {
            stem(words(i)(j));
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
