//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "strings.h"

namespace opennn
{

/// Returns the number of strings delimited by separator.
/// If separator does not match anywhere in the string, this method returns 0.
/// @param str String to be tokenized.

Index count_tokens(string& str, const char& separator)
{
    trim(str);

    Index tokens_count = 0;

    // Skip delimiters at beginning.

    string::size_type last_pos = str.find_first_not_of(separator, 0);

    // Find first "non-delimiter".

    string::size_type pos = str.find_first_of(separator, last_pos);

    while(string::npos != pos || string::npos != last_pos)
    {
        // Found a token, add it to the vector

        tokens_count++;

        // Skip delimiters.  Note the "not_of"

        last_pos = str.find_first_not_of(separator, pos);

        // Find next "non-delimiter"

        pos = str.find_first_of(separator, last_pos);
    }

    return tokens_count;
}


Index count_tokens(const string& s, const char& c)
{
    string str_copy = s;

    Index tokens_number = count(s.begin(), s.end(), c);

    if(s[0] == c)
    {
        tokens_number--;
    }
    if(s[s.size() - 1] == c)
    {
        tokens_number--;
    }

    return (tokens_number+1);

}


/// Splits the string into substrings(tokens) wherever separator occurs, and returns a vector with those strings.
/// If separator does not match anywhere in the string, this method returns a single-element list containing this string.
/// @param str String to be tokenized.

Tensor<string, 1> get_tokens(const string& str, const char& separator)
{
    const Index tokens_number = count_tokens(str, separator);

    Tensor<string, 1> tokens(tokens_number);

    // Skip delimiters at beginning.

    string::size_type lastPos = str.find_first_not_of(separator, 0);

    // Find first "non-delimiter"

    Index index = 0;
    Index old_pos = lastPos;

    string::size_type pos = str.find_first_of(separator, lastPos);

    while(string::npos != pos || string::npos != lastPos)
    {
        if((lastPos-old_pos != 1) && index!= 0)
        {
            tokens[index] = "";
            index++;
            old_pos = old_pos+1;
            continue;
        }
        else
        {
            // Found a token, add it to the vector

            tokens[index] = str.substr(lastPos, pos - lastPos);
        }

        old_pos = pos;

        // Skip delimiters. Note the "not_of"

        lastPos = str.find_first_not_of(separator, pos);

        // Find next "non-delimiter"

        pos = str.find_first_of(separator, lastPos);

        index++;
    }

    return tokens;
}


/// Splits the string into substrings(tokens) wherever separator occurs, and returns a vector with those strings.
/// If separator does not match anywhere in the string, this method returns a single-element list containing this string.
/// @param str String to be tokenized.

void fill_tokens(const string& str, const char& separator, Tensor<string, 1>& tokens)
{
    tokens.setConstant("");

    // Skip delimiters at beginning.

    string::size_type last_position = str.find_first_not_of(separator, 0);

    string::size_type position = str.find_first_of(separator, last_position);

    // Find first "non-delimiter"

    Index index = 0;

    Index old_pos = last_position;

    while(string::npos != position || string::npos != last_position)
    {
        // Found a token, add it to the vector

        if((last_position-old_pos != 1) && index!= 0)
        {
            tokens[index] = "";
            index++;
            old_pos = old_pos+1;
            continue;
        }
        else
        {
            // Found a token, add it to the vector

            tokens[index] = str.substr(last_position, position - last_position);
        }

        old_pos = position;

        // Skip delimiters. Note the "not_of"

        last_position = str.find_first_not_of(separator, position);

        // Find next "non-delimiter"

        position = str.find_first_of(separator, last_position);

        index++;
    }
}


/// Returns the number of strings delimited by separator.
/// If separator does not match anywhere in the string, this method returns 0.
/// @param str String to be tokenized.


Index count_tokens(const string& s, const string& sep)
{
    Index tokens_number = 0;

    string::size_type pos = 0;

    while( s.find(sep, pos) != string::npos )
    {
        pos = s.find(sep, pos);
       ++ tokens_number;
        pos += sep.length();
    }

    if(s.find(sep,0) == 0)
    {
        tokens_number--;
    }
    if(pos == s.length())
    {
        tokens_number--;
    }

    return (tokens_number+1);
}


/// Splits the string into substrings(tokens) wherever separator occurs, and returns a vector with those strings.
/// If separator does not match anywhere in the string, this method returns a single-element list containing this string.
/// @param str String to be tokenized.

Tensor<string, 1> get_tokens(const string& s, const string& sep)
{
    const Index tokens_number = count_tokens(s, sep);

    Tensor<string,1> tokens(tokens_number);

    string str = s;
    size_t pos = 0;
    size_t last_pos = 0;
    Index i = 0;

    while((pos = str.find(sep,pos)) != string::npos)
    {
        if(pos == 0) // Skip first position
        {
            pos += sep.length();
            last_pos = pos;
            continue;
        }

        tokens(i) = str.substr(last_pos, pos - last_pos);

        pos += sep.length();
        last_pos = pos;
        i++;
    }

    if(last_pos != s.length()) // Reading last element
    {
        tokens(i) = str.substr(last_pos, s.length() - last_pos);
    }

    return tokens;
}

/// Returns a new vector with the elements of this string vector casted to type.

Tensor<type, 1> to_type_vector(const string& str, const char& separator)
{
    const Tensor<string, 1> tokens = get_tokens(str, separator);

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


Tensor<Index, 1> to_index_vector(const string& str, const char& separator)
{
    const Tensor<string, 1> tokens = get_tokens(str, separator);

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

    return get_tokens(result,' ');
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

bool is_numeric_string(const string& str)
{
    string::size_type index;

    istringstream iss(str.data());

    float dTestSink;

    iss >> dTestSink;

    // was any input successfully consumed/converted?

    if(!iss) return false;

    // was all the input successfully consumed/converted?

    try
    {
        stod(str, &index);

        if(index == str.size() || (str.find("%") != string::npos && index+1 == str.size()))
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
bool is_constant_string(const Tensor<string, 1>& str)
{
    const string str0 = str[0];
    string str1;

    for(int i = 1; i < str.size(); i++)
    {
        str1 = str[i];
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

bool is_date_time_string(const string& str)
{
    if(is_numeric_string(str))return false;

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

    const regex regular_expression(format_1 + "|" + format_2 + "|" + format_3 + "|" + format_4 + "|" + format_5 + "|" + format_6 + "|" + format_7 + "|" + format_8
                                   + "|" + format_9 + "|" + format_10 + "|" + format_11 +"|" + format_12  + "|" + format_13 + "|" + format_14 + "|" + format_15
                                   + "|" + format_16 + "|" + format_17);

    if(regex_match(str, regular_expression))
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

bool ends_with(const string& word, const string& ending)
{
    if(ending.length() > word.length())
    {
        return false;
    }

    return(word.substr(word.length() - ending.length()) == ending);
}


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
    struct tm time_structure;

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
            time_structure.tm_hour = stoi(matchs[4].str()) - static_cast<int>(gmt);
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
            time_structure.tm_hour = stoi(matchs[10].str()) - static_cast<int>(gmt);
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
            time_structure.tm_hour = stoi(matchs[18].str()) - static_cast<int>(gmt);
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
            time_structure.tm_hour = stoi(matchs[24].str()) - static_cast<int>(gmt);
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
            time_structure.tm_mon = static_cast<int>(month_number) - 1;
            time_structure.tm_mday = stoi(matchs[31].str());
            time_structure.tm_hour = stoi(matchs[32].str()) - static_cast<int>(gmt);
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

            time_structure.tm_year = stoi(matchs[35].str())-1900;
            time_structure.tm_mon = static_cast<int>(month_number) - 1;
            time_structure.tm_mday = stoi(matchs[37].str());
            time_structure.tm_hour = stoi(matchs[38].str())- static_cast<int>(gmt);
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
            time_structure.tm_mon = static_cast<int>(month_number)-1;
            time_structure.tm_mday = stoi(matchs[42].str())- static_cast<int>(gmt);
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
            time_structure.tm_mon = static_cast<int>(month_number)-1;
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
            time_structure.tm_hour = stoi(matchs[61].str()) - static_cast<int>(gmt);
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
/// @param sub_str Substring to search.

bool contains_substring(const string& str, const string& sub_str)
{
    if(str.find(sub_str)  != string::npos)
    {
        return true;
    }
    return false;
}


///Replaces all apprearances of a substring with another string
///@param s
///@param toReplace
///@param replaceWith

void replace_all_word_appearances(string& s, string const& toReplace, string const& replaceWith) {

    string buf;
    size_t pos = 0;
    size_t prevPos;
    const string underscore = "_";

    // Reserva una estimación aproximada del tamaño final de la cadena.
    buf.reserve(s.size());

    while (true) {

        prevPos = pos;
        pos = s.find(toReplace, pos);

        if(pos == string::npos)
            break;

        // Verifica que no haya letras antes ni después de toReplace
        if((prevPos == 0 || !isalpha(s[prevPos - 1])) &&
            (pos + toReplace.size() == s.size() || !isalpha(s[pos + toReplace.size()])))
        {
            // Verifica que no haya guiones bajos antes ni después de toReplace
            if((prevPos == 0 || s[prevPos - 1] != '_') &&
                (pos + toReplace.size() == s.size() || s[pos + toReplace.size()] != '_'))
            {
                buf.append(s, prevPos, pos - prevPos);
                buf += replaceWith;
                pos += toReplace.size();
            }
            else
            {
                buf.append(s, prevPos, pos - prevPos + toReplace.size());
                pos += toReplace.size();
            }
        }
        else
        {
            buf.append(s, prevPos, pos - prevPos + toReplace.size());
            pos += toReplace.size();
        }
    }

    buf.append(s, prevPos, s.size() - prevPos);
    s.swap(buf);
}



 ///Replaces all apprearances of a substring with another string
 ///@param s
 ///@param toReplace
 ///@param replaceWith

void replace_all_appearances(string& s, string const& toReplace, string const& replaceWith) {

    string buf;
    size_t pos = 0;
    size_t prevPos;

    // Reserves rough estimate of final size of string.
    buf.reserve(s.size());

    while(true) {

        prevPos =    pos;
        pos = s.find(toReplace, pos);

        if(pos == string::npos)
            break;

        buf.append(s, prevPos, pos - prevPos);

        if(buf.back() == '_')
        {
            buf += toReplace;
            pos += toReplace.size();

        }else
        {
            buf += replaceWith;
            pos += toReplace.size();

        }
    }

    buf.append(s, prevPos, s.size() - prevPos);
    s.swap(buf);
}


/// Replaces all apprearances non allowed programming characters of a substring with allowed characters
/// \brief replace_non_allowed_programming_characters
/// \param s
/// \return

string replace_non_allowed_programming_expressions(string& s)
{
        string out = "";

        if(s[0] == '$')
            out=s;

        /*
        switch(l) {

        // C++ reserved words
        case 1:
            replace_all_appearances(s, "int", "in_t");
            replace_all_appearances(s, "sizeof", "sizeo_f");
            replace_all_appearances(s, "auto", "aut_o");
            replace_all_appearances(s, "extern", "exter_n");
            replace_all_appearances(s, "return", "retur_n");
            replace_all_appearances(s, "typedef", "typede_f");
            replace_all_appearances(s, "for", "fo_r");
            replace_all_appearances(s, "_Packed", "_P_acked");
            replace_all_appearances(s, "signed", "signe_d");
            replace_all_appearances(s, "enum", "enu_m");
            replace_all_appearances(s, "case", "cas_e");
            replace_all_appearances(s, "char", "cha_r");
            replace_all_appearances(s, "union", "unio_n");
            replace_all_appearances(s, "float", "floa_t");
            replace_all_appearances(s, "short", "shor_t");
            replace_all_appearances(s, "const", "cons_t");
            replace_all_appearances(s, "void", "voi_d");
            replace_all_appearances(s, "continue", "continu_e");
            replace_all_appearances(s, "goto", "got_o");
            replace_all_appearances(s, "volatile", "volatil_e");
            replace_all_appearances(s, "default", "defaul_t");
            replace_all_appearances(s, "if", "i_f");
            replace_all_appearances(s, "else", "els_e");
            replace_all_appearances(s, "long", "lon_g");
            replace_all_appearances(s, "static", "stati_c");
            replace_all_appearances(s, "while", "whil_e");
            replace_all_appearances(s, "do", "d_o");
            replace_all_appearances(s, "break", "brea_k");
            replace_all_appearances(s, "switch", "switc_h");
            replace_all_appearances(s, "struct", "struc_t");
            replace_all_appearances(s, "double", "doubl_e");
            replace_all_appearances(s, "unsigned", "unsigne_d");
            replace_all_appearances(s, "register", "registe_r");
            break;

        // JAVASCRIPT reserved words
        case 2:
            replace_all_appearances(s, "abstract", "abstrac_t");
            replace_all_appearances(s, "arguments", "argument_s");
            replace_all_appearances(s, "await", "awai_t");
            replace_all_appearances(s, "boolea", "boole_an");
            replace_all_appearances(s, "break", "brea_k");
            replace_all_appearances(s, "byte", "byt_e");
            replace_all_appearances(s, "case", "cas_e");
            replace_all_appearances(s, "catc", "cat_ch");
            replace_all_appearances(s, "char", "cha_r");
            replace_all_appearances(s, "class", "clas_s");
            replace_all_appearances(s, "const", "cons_t");
            replace_all_appearances(s, "continu", "contin_ue");
            replace_all_appearances(s, "debugger", "debugge_r");
            replace_all_appearances(s, "default", "defaul_t");
            replace_all_appearances(s, "delete", "delet_e");
            replace_all_appearances(s, "d", "d_o");
            replace_all_appearances(s, "double", "doubl_e");
            replace_all_appearances(s, "else", "els_e");
            replace_all_appearances(s, "enum", "enu_m");
            replace_all_appearances(s, "eva", "ev_al");
            replace_all_appearances(s, "export", "expor_t");
            replace_all_appearances(s, "extends", "extend_s");
            replace_all_appearances(s, "false", "fals_e");
            replace_all_appearances(s, "fina", "fin_al");
            replace_all_appearances(s, "finally", "finall_y");
            replace_all_appearances(s, "float", "floa_t");
            replace_all_appearances(s, "for", "fo_r");
            replace_all_appearances(s, "functio", "functi_on");
            replace_all_appearances(s, "goto", "got_o");
            replace_all_appearances(s, "if", "i_f");
            replace_all_appearances(s, "implements", "implement_s");
            replace_all_appearances(s, "impor", "impo_rt");
            replace_all_appearances(s, "in", "i_n");
            replace_all_appearances(s, "instanceof", "instanceo_f");
            replace_all_appearances(s, "int", "in_t");
            replace_all_appearances(s, "interfac", "interfa_ce");
            replace_all_appearances(s, "let", "le_t");
            replace_all_appearances(s, "long", "lon_g");
            replace_all_appearances(s, "native", "nativ_e");
            replace_all_appearances(s, "ne", "n_ew");
            replace_all_appearances(s, "null", "nul_l");
            replace_all_appearances(s, "package", "packag_e");
            replace_all_appearances(s, "private", "privat_e");
            replace_all_appearances(s, "protecte", "protect_ed");
            replace_all_appearances(s, "public", "publi_c");
            replace_all_appearances(s, "return", "retur_n");
            replace_all_appearances(s, "short", "shor_t");
            replace_all_appearances(s, "stati", "stat_ic");
            replace_all_appearances(s, "super", "supe_r");
            replace_all_appearances(s, "switch", "switc_h");
            replace_all_appearances(s, "synchronized", "synchronize_d");
            replace_all_appearances(s, "thi", "th_is");
            replace_all_appearances(s, "throw", "thro_w");
            replace_all_appearances(s, "throws", "throw_s");
            replace_all_appearances(s, "transient", "transien_t");
            replace_all_appearances(s, "tru", "tr_ue");
            replace_all_appearances(s, "try", "tr_y");
            replace_all_appearances(s, "typeof", "typeo_f");
            replace_all_appearances(s, "var", "va_r");
            replace_all_appearances(s, "voi", "vo_id");
            replace_all_appearances(s, "volatile", "volatil_e");
            replace_all_appearances(s, "while", "whil_e");
            replace_all_appearances(s, "with", "wit_h");
            replace_all_appearances(s, "yiel", "yie_ld");
            break;

        // PHP reserved words
        case 3:
            replace_all_appearances(s, "abstract", "abstrac_t");
            replace_all_appearances(s, "and", "an_d");
            replace_all_appearances(s, "array", "arra_y");
            replace_all_appearances(s, "as", "a_s");
            replace_all_appearances(s, "break", "brea_k");
            replace_all_appearances(s, "callable", "callabl_e");
            replace_all_appearances(s, "case", "cas_e");
            replace_all_appearances(s, "catch", "catc_h");
            replace_all_appearances(s, "class", "clas_s");
            replace_all_appearances(s, "clone", "clon_e");
            replace_all_appearances(s, "const", "cons_t");
            replace_all_appearances(s, "continue", "continu_e");
            replace_all_appearances(s, "declare", "declar_e");
            replace_all_appearances(s, "default", "defaul_t");
            replace_all_appearances(s, "die", "di_e");
            replace_all_appearances(s, "do", "d_o");
            replace_all_appearances(s, "echo", "ech_o");
            replace_all_appearances(s, "else", "els_e");
            replace_all_appearances(s, "elseif", "elsei_f");
            replace_all_appearances(s, "empty", "empt_y");
            replace_all_appearances(s, "enddeclare", "enddeclar_e");
            replace_all_appearances(s, "endfor", "endfo_r");
            replace_all_appearances(s, "endforeach", "endforeac_h");
            replace_all_appearances(s, "endif", "endi_f");
            replace_all_appearances(s, "endswitch", "endswitc_h");
            replace_all_appearances(s, "endwhile", "endwhil_e");
            replace_all_appearances(s, "eval", "eva_l");
            replace_all_appearances(s, "exit", "exi_t");
            replace_all_appearances(s, "extends", "extend_s");
            replace_all_appearances(s, "final", "fina_l");
            replace_all_appearances(s, "finally", "finall_y");
            replace_all_appearances(s, "fn", "f_n");
            replace_all_appearances(s, "for", "fo_r");
            replace_all_appearances(s, "foreach", "foreac_h");
            replace_all_appearances(s, "function", "functio_n");
            replace_all_appearances(s, "global", "globa_l");
            replace_all_appearances(s, "goto", "got_o");
            replace_all_appearances(s, "if", "i_f");
            replace_all_appearances(s, "implements", "implement_s");
            replace_all_appearances(s, "include", "includ_e");
            replace_all_appearances(s, "instanceof", "instanceo_f");
            replace_all_appearances(s, "insteadof", "insteado_f");
            replace_all_appearances(s, "interface", "interfac_e");
            replace_all_appearances(s, "isset", "isse_t");
            replace_all_appearances(s, "list", "lis_t");
            replace_all_appearances(s, "match", "matc_h");
            replace_all_appearances(s, "namespace", "namespac_e");
            replace_all_appearances(s, "new", "ne_w");
            replace_all_appearances(s, "or", "o_r");
            replace_all_appearances(s, "print", "prin_t");
            replace_all_appearances(s, "private", "privat_e");
            replace_all_appearances(s, "protected", "protecte_d");
            replace_all_appearances(s, "public", "publi_c");
            replace_all_appearances(s, "readonly", "readonl_y");
            replace_all_appearances(s, "require", "requir_e");
            replace_all_appearances(s, "return", "retur_n");
            replace_all_appearances(s, "static", "stati_c");
            replace_all_appearances(s, "switch", "switc_h");
            replace_all_appearances(s, "throw", "thro_w");
            replace_all_appearances(s, "trait", "trai_t");
            replace_all_appearances(s, "try", "tr_y");
            replace_all_appearances(s, "unset", "unse_t");
            replace_all_appearances(s, "use", "us_e");
            replace_all_appearances(s, "var", "va_r");
            replace_all_appearances(s, "while", "whil_e");
            replace_all_appearances(s, "xor", "xo_r");
            replace_all_appearances(s, "yield", "yiel_d");
            replace_all_appearances(s, "include_once", "include_on_ce_");
            replace_all_appearances(s, "require_once", "require_on_ce_");
            replace_all_appearances(s, "__halt_compiler", "__h_a_l_t_c_o_m_p_i_l_e_r_");
            break;

        // PYTHON reserved words
        case 4:
            replace_all_appearances(s, "False", "F_alse");
            replace_all_appearances(s, "def", "de_f");
            replace_all_appearances(s, "if", "i_f");
            replace_all_appearances(s, "raise", "ra_ise");
            replace_all_appearances(s, "None", "Non_e");
            replace_all_appearances(s, "del", "de_l");
            replace_all_appearances(s, "import", "imp_ort");
            replace_all_appearances(s, "return", "retu_rn");
            replace_all_appearances(s, "True", "Tr_ue");
            replace_all_appearances(s, "elif", "el_if");
            replace_all_appearances(s, "in", "i_n");
            replace_all_appearances(s, "try", "t_ry");
            replace_all_appearances(s, "and", "an_d");
            replace_all_appearances(s, "else", "el_se");
            replace_all_appearances(s, "is", "i_s");
            replace_all_appearances(s, "while", "wh_ile");
            replace_all_appearances(s, "as", "a_s");
            replace_all_appearances(s, "except", "ex_cept");
            replace_all_appearances(s, "lambda", "lam_bda");
            replace_all_appearances(s, "with", "wit_h");
            replace_all_appearances(s, "assert", "as_sert");
            replace_all_appearances(s, "finally", "fi_nally");
            replace_all_appearances(s, "nonlocal", "no_nlocal");
            replace_all_appearances(s, "yield", "yie_ld");
            replace_all_appearances(s, "break", "bre_ak");
            replace_all_appearances(s, "for", "fo_r");
            replace_all_appearances(s, "not", "no_t");
            replace_all_appearances(s, "class", "c_lass");
            replace_all_appearances(s, "form", "for_m");
            replace_all_appearances(s, "or", "o_r");
            replace_all_appearances(s, "continue", "continu_e");
            replace_all_appearances(s, "global", "glob_al");
            replace_all_appearances(s, "pass", "pa_00ss");
            break;

        default:
            return 0;
        }
        */

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


int WordOccurrence(char *sentence, char *word)
{
    int slen = strlen(sentence);
    int wordlen = strlen(word);
    int count = 0;
    int i, j;

    for(i = 0; i<slen; i++)
    {
        for(j = 0; j<wordlen; j++)
        {
            if(sentence[i+j]!=word[j])
            break;
        }
        if(j==wordlen)
        {
            count++;
        }
    }
    return count;
}


/// Removes whitespaces from the start and the end of the string passed as argument.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

void trim(string& str)
{
    // Prefixing spaces

    str.erase(0, str.find_first_not_of(' '));
    str.erase(0, str.find_first_not_of('\t'));
    str.erase(0, str.find_first_not_of('\n'));
    str.erase(0, str.find_first_not_of('\r'));
    str.erase(0, str.find_first_not_of('\f'));
    str.erase(0, str.find_first_not_of('\v'));

    // Surfixing spaces

    str.erase(str.find_last_not_of(' ') + 1);
    str.erase(str.find_last_not_of('\t') + 1);
    str.erase(str.find_last_not_of('\n') + 1);
    str.erase(str.find_last_not_of('\r') + 1);
    str.erase(str.find_last_not_of('\f') + 1);
    str.erase(str.find_last_not_of('\v') + 1);
    str.erase(str.find_last_not_of('\b') + 1);

    // Special character and string modifications

    replace_first_and_last_char_with_missing_label(str, ';', "NA", "");
    replace_first_and_last_char_with_missing_label(str, ',', "NA", "");

    replace_double_char_with_label(str, ";", "NA");
    replace_double_char_with_label(str, ",", "NA");

    replac_substring_within_quotes(str, ",", "");
    replac_substring_within_quotes(str, ";", "");
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

    size_t pos = 0;
    while((pos = str.find(target_pattern, pos)) != string::npos)
    {
        str.replace(pos, target_pattern.length(), new_pattern);
        pos += new_pattern.length();
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
        size_t pos = 0;
        while((pos = replaced_str.find(target, pos)) != string::npos)
        {
            replaced_str.replace(pos, target.length(), replacement);
            pos += replacement.length();
        }
        result += match.prefix().str() + replaced_str;
        prefix = match.suffix().str();
    }
    result += prefix;
    str = result;
}


void erase(string& s, const char& c)
{
    s.erase(remove(s.begin(), s.end(), c), s.end());
}


/// Returns a string that has whitespace removed from the start and the end.
/// This includes the ASCII characters "\t", "\n", "\v", "\f", "\r", and " ".
/// @param str String to be checked.

string get_trimmed(const string& str)
{
    string output(str);

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

string prepend(const string& pre, const string& str)
{
    ostringstream buffer;

    buffer << pre << str;

    return buffer.str();
}


/// Returns true if all the elements in a string list are numeric, and false otherwise.
/// @param v String list to be checked.

bool is_numeric_string_vector(const Tensor<string, 1>& v)
{
    for(Index i = 0; i < v.size(); i++)
    {
        if(!is_numeric_string(v[i])) return false;
    }

    return true;
}


bool has_numbers(const Tensor<string, 1>& v)
{
    for(Index i = 0; i < v.size(); i++)
    {
//        if(is_numeric_string(v[i])) return true;
        if(is_numeric_string(v[i]))
        {
            cout << "The number is: " << v[i] << endl;
            return true;
        }
    }

    return false;
}


bool has_strings(const Tensor<string, 1>& v)
{
    for(Index i = 0; i < v.size(); i++)
    {
        if(!is_numeric_string(v[i])) return true;
    }

    return false;
}

/// Returns true if none element in a string list is numeric, and false otherwise.
/// @param v String list to be checked.

bool is_not_numeric(const Tensor<string, 1>& v)
{
    for(Index i = 0; i < v.size(); i++)
    {
        if(is_numeric_string(v[i])) return false;
    }

    return true;
}


/// Returns true if some the elements in a string list are numeric and some others are not numeric.
/// @param v String list to be checked.

bool is_mixed(const Tensor<string, 1>& v)
{
    unsigned count_numeric = 0;
    unsigned count_not_numeric = 0;

    for(Index i = 0; i < v.size(); i++)
    {
        if(is_numeric_string(v[i]))
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

void remove_non_printable_chars( string& wstr)
{
    // get the ctype facet for wchar_t (Unicode code points in pactice)
    typedef ctype< wchar_t > ctype ;
    const ctype& ct = use_facet<ctype>( locale() ) ;

    // remove non printable Unicode characters
    wstr.erase( remove_if( wstr.begin(), wstr.end(),
                    [&ct]( wchar_t ch ) { return !ct.is( ctype::print, ch ) ; } ),
                wstr.end() ) ;
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


bool isNotAlnum (char &c)
{
    return (c < ' ' || c > '~');
}


void remove_not_alnum(string &str)
{
        str.erase(remove_if(str.begin(), str.end(), isNotAlnum), str.end());
}

bool find_string_in_tensor(Tensor<string, 1>& t, string val)
{
    for(Index i = 0; i < t.dimension(0);++i)
    {
        string elem = t(i);

        if(elem == val)
        {
            return true;
        }
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


Tensor<string, 1> fix_write_expression_outputs(const string &str, const Tensor<string, 1> &outputs, const string &programming_languaje)
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

    int dimension = outputs.dimension(0);

    while(getline(ss, token, '\n'))
    {
        if(token.size() > 1 && token.back() == '{'){ break; }
        if(token.size() > 1 && token.back() != ';'){ token += ';'; }
        push_back_string(tokens, token);
    }

    for(int i = 0; i < tokens.dimension(0); i++)
    {
        string s = tokens(i);
        string word = "";

        for(char& c : s)
        {
            if( c!=' ' && c!='=' ){ word += c; }
            else { break; }
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
        int j = found_tokens.size();

        for(int i = dimension; i --> 0;)
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

Tensor<Tensor<string,1>, 1> fix_input_output_variables(Tensor<string, 1>& inputs_names, Tensor<string, 1>& outputs_names, ostringstream& buffer_)
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
    type factor = type(pow(10, precision));

    type rounded_value = (round(factor*x))/factor;

    stringstream ss;
    ss << fixed << setprecision(precision) << rounded_value;
    string result = ss.str();
    return result;
}

Tensor<string,2> round_to_precision_string_matrix(Tensor<type,2> matrix, const int& precision)
{
    Tensor<string,2> matrix_rounded(matrix.dimension(0), matrix.dimension(1));

    type factor = type(pow(10, precision));

    for(int i = 0; i< matrix_rounded.dimension(0); i++)
    {
        for(int j = 0; j < matrix_rounded.dimension(1); j++)
        {
            type rounded_value = (round(factor*matrix(i,j)))/factor;
            stringstream ss;
            ss << fixed << setprecision(precision) << rounded_value;
            string result = ss.str();
            matrix_rounded(i,j) = result;
        }
    }
    return matrix_rounded;
}


/// @todo clean this method Clang-tidy gives warnings.

Tensor<string,1> sort_string_tensor(Tensor<string, 1> tensor)
{
    auto compareStringLength = [](const string& a, const string& b)
    {
        return a.length() > b.length();
    };

    vector<string> tensor_as_vector(tensor.data(), tensor.data() + tensor.size());
    
    sort(tensor_as_vector.begin(), tensor_as_vector.end(), compareStringLength);

    for(int i = 0; i < tensor.size(); i++)
    {
        tensor(i) = tensor_as_vector[i];
    }

    return tensor;
}


Tensor<string,1> concatenate_string_tensors(const Tensor<string, 1>& tensor_1, const Tensor<string, 1>& tensor_2)
{
    Tensor<string, 1> tensor = tensor_2;

    for(int i = 0; i < tensor_1.dimension(0); ++i) push_back_string(tensor, tensor_1(i));

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
        string found_token = tokens(i);
        string toReplace(found_token);
        string newword = keyword + " " + found_token;

        string::size_type pos = 0;

        while((pos = espression.find(toReplace, pos)) != string::npos)
        {
            if(pos > previous_pos)
            {
                espression.replace(pos, toReplace.length(), newword);
                pos += newword.length();
                previous_pos = pos;
                break;
            }
            else
            {
                pos += newword.length();
            }
        }
    }
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

    for (Index i = 0; i < length; i++)
    {
        const int word_index = get_alphabet_index(text[i]);

        data_tensor(i, word_index) = type(1);
    }
*/
}


void preprocess()
{
/*
    TextAnalytics ta;

    ta.replace_accented(text);

    transform(text.begin(), text.end(), text.begin(), ::tolower); // To lower
*/
}


Index get_alphabet_index(const char& ch) 
{
/*
    auto alphabet_begin = alphabet.data();
    auto alphabet_end = alphabet.data() + alphabet.size();

    const string str(1, ch);

    auto it = find(alphabet_begin, alphabet_end, str);

    if (it != alphabet_end)
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

    for (Index i = 0; i < phrase_length; i++)
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

    if (tensor.size() != length)
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

    if (tensor.dimension(1) != length)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextGenerationAlphabet class.\n"
            << "string one_hot_decode(Tensor<type, 1>& tensor).\n"
            << "Tensor length must be equal to alphabet length.";

        throw runtime_error(buffer.str());
    }

    string result = "";

    for (Index i = 0; i < tensor.dimension(0); i++)
    {
        Tensor<type, 1> row = tensor.chip(i, 0);

        auto index = max_element(row.data(), row.data() + row.size()) - row.data();

        result += alphabet(index);
    }

    return result;
*/
    return string();
}


Tensor<type, 2> str_to_input(const string& input_string) 
{
    Tensor<type, 2> input_data = multiple_one_hot_encode(input_string);

    Tensor<type, 2> flatten_input_data(1, input_data.size());

    copy(/*execution::par,*/
        input_data.data(),
        input_data.data() + input_data.size(),
        flatten_input_data.data());

    return flatten_input_data;
}


string output_to_str(const Tensor<type, 2>& flatten_output_data) 
{
/*
    const Index alphabet_length = get_alphabet_length();

    const Index tensor_size = Index(flatten_output_data.size() / alphabet_length);

    Tensor<type, 2> output_data(tensor_size, alphabet_length);

    copy(/*execution::par,
        flatten_output_data.data(),
        flatten_output_data.data() + tensor_size, output_data.data());

    return multiple_one_hot_decode(output_data);
*/
    return string();
}


/// Calculate the total number of tokens in the documents.

Index count(const Tensor<Tensor<string, 1>, 1>& documents)
{
    const Index documents_number = documents.dimension(0);

    Index total_size = 0;

    for (Index i = 0; i < documents_number; i++)
    {
        for (Index j = 0; j < documents(i).dimension(0); j++)
        {
            total_size += count_tokens(documents(i)(j));
        }
    }

    return total_size;
}


/// Returns a Tensor with all the words as elements keeping the order.

Tensor<string, 1> join(const Tensor<Tensor<string, 1>, 1>& documents)
{
    const type words_number = type(count(documents));

    Tensor<string, 1> words_list(words_number);

    Index current_tokens = 0;

    for (Index i = 0; i < documents.dimension(0); i++)
    {
        for (Index j = 0; j < documents(i).dimension(0); j++)
        {
            Tensor<string, 1> tokens = get_tokens(documents(i)(j));

            copy(tokens.data(), tokens.data() + tokens.size(), words_list.data() + current_tokens);

            current_tokens += tokens.size();
        }
    }

    return words_list;
}


/// Transforms all the letters of the documents into lower case.

void to_lower(Tensor<string, 1>& documents)
{
    const size_t documents_number = documents.size();

    for (size_t i = 0; i < documents_number; i++)
    {
        transform(documents[i].begin(), documents[i].end(), documents[i].begin(), ::tolower);
    }

}


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


void delete_non_printable_chars(Tensor<string, 1>& documents)
{
    for (Index i = 0; i < documents.size(); i++) remove_non_printable_chars(documents(i));
}


void delete_extra_spaces(Tensor<string, 1>& documents)
{
    Tensor<string, 1> new_documents(documents);

    for (Index i = 0; i < documents.size(); i++)
    {
        string::iterator new_end = unique(new_documents[i].begin(), new_documents[i].end(),
            [](char lhs, char rhs) { return(lhs == rhs) && (lhs == ' '); });

        new_documents[i].erase(new_end, new_documents[i].end());
    }

    documents = new_documents;
}


void aux_remove_non_printable_chars(Tensor<string, 1>& documents)
{
    Tensor<string, 1> new_documents(documents);

    for (Index i = 0; i < documents.size(); i++)
    {
        new_documents[i].erase(remove_if(new_documents[i].begin(), new_documents[i].end(), isNotAlnum), new_documents[i].end());
    }

    documents = new_documents;
}


Tensor<Tensor<string, 1>, 1> tokenize(const Tensor<string, 1>& documents)
{
    const Index documents_number = documents.size();

    Tensor<Tensor<string, 1>, 1> new_tokenized_documents(documents_number);

#pragma omp parallel for
    for (Index i = 0; i < documents_number; i++)
    {
        new_tokenized_documents(i) = get_tokens(documents(i));
    }

    return new_tokenized_documents;
}


void delete_emails(Tensor<Tensor<string, 1>, 1>& documents)
{
    const Index documents_number = documents.size();

#pragma omp parallel for
    for (Index i = 0; i < documents_number; i++)
    {
        Tensor<string, 1> document = documents(i);

        for (Index j = 0; j < document.size(); j++)
        {
            Tensor<string, 1> tokens = get_tokens(document(j));

            string result;

            for (Index k = 0; k < tokens.size(); k++)
            {
                if (!is_email(tokens(k)))
                {
                    result += tokens(k) + " ";
                }
            }

            document(j) = result;
        }

        documents(i) = document;
    }
}


void delete_blanks(Tensor<string, 1>& vector)
{
    const Index words_number = vector.size();

    const Index empty_number = count_empty(vector);

    Tensor<string, 1> vector_copy(vector);

    vector.resize(words_number - empty_number);

    Index index = 0;

    string empty_string;

    for (Index i = 0; i < words_number; i++)
    {
        trim(vector_copy(i));

        if (!vector_copy(i).empty())
        {
            vector(index) = vector_copy(i);
            index++;
        }
    }
}


void delete_blanks(Tensor<Tensor<string, 1>, 1>& tokens)
{
    const Index documents_size = tokens.size();

    for (Index i = 0; i < documents_size; i++)
    {
        delete_blanks(tokens(i));
    }
}


}

/*
//   OpenNN: Open Neural Networks Library
//   www.opennn.net

//   T E X T   A N A L Y S I S   C L A S S

//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

namespace opennn
{
// DEFAULT CONSTRUCTOR


/// Default constructor.

TextAnalytics::TextAnalytics()
{
    set_english_stop_words();
}



TextAnalytics::~TextAnalytics()
{
}

// Get methods


/// Returns a Tensor containing the documents.

Tensor<Tensor<string,1> ,1> TextAnalytics::get_documents() const
{
    return documents;
}


/// Returns a Tensor containing the targets.

Tensor<Tensor<string,1> ,1> TextAnalytics::get_targets() const
{
    return targets;
}


/// Returns the language selected.

TextAnalytics::Language TextAnalytics::get_language() const
{
    return lang;
}


/// Returns the language selected in string format.

string TextAnalytics::get_language_string() const
{
    if(lang == ENG)
    {
        return "ENG";
    }
    else if(lang == SPA)
    {
        return "SPA";
    }
    else
    {
        return string();
    }
}


Index TextAnalytics::get_short_words_length() const
{
    return short_words_length;
}


Index TextAnalytics::get_long_words_length() const
{
    return long_words_length;
}


/// Returns the stop words.

Tensor<string, 1> TextAnalytics::get_stop_words() const
{
    return stop_words;
}


Index TextAnalytics::get_document_sentences_number() const
{
    Index count = 0;

    for(Index i = 0; i < documents.dimension(0); i++)
    {
        count += documents(i).dimension(0);
    }

    return count;
}


// Set methods


/// Sets a language.

void TextAnalytics::set_language(const Language& new_language)
{
    lang = new_language;

    if(lang == ENG)
    {
        set_english_stop_words();
    }
    else if(lang == SPA)
    {
        set_spanish_stop_words();
    }
    else
    {
        //        clear_stop_words();
    }
}

/// Sets a language.

void TextAnalytics::set_language(const string& new_language_string)
{
    if(new_language_string == "ENG")
    {
        set_language(ENG);
    }
    else if(new_language_string == "SPA")
    {
        set_language(SPA);
    }
    else
    {
        //        clear_stop_words();
    }
}


/// Sets a stop words.
/// @param new_stop_words String Tensor with the new stop words.

void TextAnalytics::set_stop_words(const Tensor<string, 1>& new_stop_words)
{
    stop_words = new_stop_words;
}


void TextAnalytics::set_short_words_length(const Index& new_short_words_length)
{
    short_words_length = new_short_words_length;
}


void TextAnalytics::set_long_words_length(const Index& new_long_words_length)
{
    long_words_length = new_long_words_length;
}


void TextAnalytics::set_separator(const string& new_separator)
{
    if(new_separator == "Semicolon")
    {
        separator = ";";
    }
    else if(new_separator == "Tab")
    {
        separator = "\t";
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "void set_separator(const string&) method.\n"
               << "Unknown separator: " << new_separator << ".\n";

        throw runtime_error(buffer.str());
    }
}

// Preprocess methods

/// Deletes consecutive extra spaces in documents.
/// @param documents Document to be proccesed.

void TextAnalytics::delete_extra_spaces(Tensor<string, 1>& documents) const
{
    Tensor<string, 1> new_documents(documents);

    for(Index i = 0; i < documents.size(); i++)
    {
        string::iterator new_end = unique(new_documents[i].begin(), new_documents[i].end(),
                                          [](char lhs, char rhs){ return(lhs == rhs) &&(lhs == ' '); });

        new_documents[i].erase(new_end, new_documents[i].end());
    }

    documents = new_documents;
}


/// Deletes line breaks and tabulations
/// @param documents Document to be proccesed.

void TextAnalytics::delete_breaks_and_tabs(Tensor<string, 1>& documents) const
{
    for(Index i = 0; i < documents.size(); i++)
    {
        string line = documents(i);

        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\n' ,' ');
        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\t' ,' ');
        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\f' ,' ');
        replace(documents(i).begin(), documents(i).end() + documents(i).size(), '\r' ,' ');
    }
}


/// Deletes unicode non printable characters

void TextAnalytics::delete_non_printable_chars(Tensor<string, 1>& documents) const
{

    for(Index i = 0; i < documents.size(); i++) remove_non_printable_chars(documents(i));
}


/// Deletes punctuation in documents.

void TextAnalytics::delete_punctuation(Tensor<string, 1>& documents) const
{
    replace_substring(documents, "�"," ");
    replace_substring(documents, "\""," ");
    replace_substring(documents, "."," ");
    replace_substring(documents, "!"," ");
    replace_substring(documents, "#"," ");
    replace_substring(documents, "$"," ");
    replace_substring(documents, "~"," ");
    replace_substring(documents, "%"," ");
    replace_substring(documents, "&"," ");
    replace_substring(documents, "/"," ");
    replace_substring(documents, "("," ");
    replace_substring(documents, ")"," ");
    replace_substring(documents, "\\"," ");
    replace_substring(documents, "="," ");
    replace_substring(documents, "?"," ");
    replace_substring(documents, "}"," ");
    replace_substring(documents, "^"," ");
    replace_substring(documents, "`"," ");
    replace_substring(documents, "["," ");
    replace_substring(documents, "]"," ");
    replace_substring(documents, "*"," ");
    replace_substring(documents, "+"," ");
    replace_substring(documents, ","," ");
    replace_substring(documents, ";"," ");
    replace_substring(documents, ":"," ");
    replace_substring(documents, "-"," ");
    replace_substring(documents, ">"," ");
    replace_substring(documents, "<"," ");
    replace_substring(documents, "|"," ");
    replace_substring(documents, "–"," ");
    replace_substring(documents, "Ø"," ");
    replace_substring(documents, "º"," ");
    replace_substring(documents, "°"," ");
    replace_substring(documents, "'"," ");
    replace_substring(documents, "ç"," ");
    replace_substring(documents, "✓"," ");
    replace_substring(documents, "|"," ");
    replace_substring(documents, "@"," ");
    replace_substring(documents, "#"," ");
    replace_substring(documents, "^"," ");
    replace_substring(documents, "*"," ");
    replace_substring(documents, "€"," ");
    replace_substring(documents, "¬"," ");
    replace_substring(documents, "•"," ");
    replace_substring(documents, "·"," ");
    replace_substring(documents, "”"," ");
    replace_substring(documents, "“"," ");
    replace_substring(documents, "´"," ");
    replace_substring(documents, "§"," ");
    replace_substring(documents,"_", " ");
    replace_substring(documents,".", " ");

    delete_extra_spaces(documents);
}


/// Splits punctuation symbols in documents.

void TextAnalytics::split_punctuation(Tensor<string, 1>& documents) const
{
    replace_substring(documents, "�"," � ");
    replace_substring(documents, "\""," \" ");
    replace_substring(documents, "."," . ");
    replace_substring(documents, "!"," ! ");
    replace_substring(documents, "#"," # ");
    replace_substring(documents, "$"," $ ");
    replace_substring(documents, "~"," ~ ");
    replace_substring(documents, "%"," % ");
    replace_substring(documents, "&"," & ");
    replace_substring(documents, "/"," / ");
    replace_substring(documents, "("," ( ");
    replace_substring(documents, ")"," ) ");
    replace_substring(documents, "\\"," \\ ");
    replace_substring(documents, "="," = ");
    replace_substring(documents, "?"," ? ");
    replace_substring(documents, "}"," } ");
    replace_substring(documents, "^"," ^ ");
    replace_substring(documents, "`"," ` ");
    replace_substring(documents, "["," [ ");
    replace_substring(documents, "]"," ] ");
    replace_substring(documents, "*"," * ");
    replace_substring(documents, "+"," + ");
    replace_substring(documents, ","," , ");
    replace_substring(documents, ";"," ; ");
    replace_substring(documents, ":"," : ");
    replace_substring(documents, "-"," - ");
    replace_substring(documents, ">"," > ");
    replace_substring(documents, "<"," < ");
    replace_substring(documents, "|"," | ");
    replace_substring(documents, "–"," – ");
    replace_substring(documents, "Ø"," Ø ");
    replace_substring(documents, "º"," º ");
    replace_substring(documents, "°"," ° ");
    replace_substring(documents, "'"," ' ");
    replace_substring(documents, "ç"," ç ");
    replace_substring(documents, "✓"," ✓ ");
    replace_substring(documents, "|"," | ");
    replace_substring(documents, "@"," @ ");
    replace_substring(documents, "#"," # ");
    replace_substring(documents, "^"," ^ ");
    replace_substring(documents, "*"," * ");
    replace_substring(documents, "€"," € ");
    replace_substring(documents, "¬"," ¬ ");
    replace_substring(documents, "•"," • ");
    replace_substring(documents, "·"," · ");
    replace_substring(documents, "”"," ” ");
    replace_substring(documents, "“"," “ ");
    replace_substring(documents, "´"," ´ ");
    replace_substring(documents, "§"," § ");
    replace_substring(documents,"_", " _ ");
    replace_substring(documents,".", " . ");

    delete_extra_spaces(documents);
}


void TextAnalytics::aux_remove_non_printable_chars(Tensor<string, 1> &documents) const
{
    Tensor<string, 1> new_documents(documents);

    for(Index i = 0; i < documents.size(); i++)
    {
        new_documents[i].erase(remove_if(new_documents[i].begin(), new_documents[i].end(), isNotAlnum), new_documents[i].end());
    }

    documents = new_documents;
}


/// Split documents into words Tensors. Each word is equivalent to a token.
/// @param documents String tensor we will split

Tensor<Tensor<string,1>,1> TextAnalytics::tokenize(const Tensor<string,1>& documents) const
{
    const Index documents_number = documents.size();

    Tensor<Tensor<string,1>,1> new_tokenized_documents(documents_number);

#pragma omp parallel for
    for(Index i = 0; i < documents_number; i++)
    {
        new_tokenized_documents(i) = get_tokens(documents(i));
    }

    return new_tokenized_documents;
}


/// Joins a string tensor into a string
/// @param token String tensor we will join

string TextAnalytics::to_string(Tensor<string,1> token) const
{
    string word;

    for(Index i = 0; i < token.size() - 1; i++)
        word += token(i) + " ";
    word += token(token.size() - 1);

    return word;
}


/// Join the words Tensors into strings documents
/// @param tokens Tensor of Tensor of words we want to join

Tensor<string,1> TextAnalytics::detokenize(const Tensor<Tensor<string,1>,1>& tokens) const
{
    const Index documents_number = tokens.size();

    Tensor<string,1> new_documents(documents_number);

    for(Index i = 0; i < documents_number; i++)
    {
        new_documents[i] = to_string(tokens(i));
    }

    return new_documents;
}

void TextAnalytics::filter_not_equal_to(Tensor<string,1>& document, const Tensor<string,1>& delete_words) const
{

    for(Index i = 0; i < document.size(); i++)
    {
        const Index tokens_number = count_tokens(document(i),' ');
        const Tensor<string, 1> tokens = get_tokens(document(i), ' ');

        string result;

        for(Index j = 0; j < tokens_number; j++)
        {
            if( ! contains(delete_words, tokens(j)) )
            {
                result += tokens(j) + " ";
            }
        }

        document(i) = result;
    }
}


/// Delete the words we want from the documents
/// @param delete_words Tensor of words we want to delete

void TextAnalytics::delete_words(Tensor<Tensor<string,1>,1>& tokens, const Tensor<string,1>& delete_words) const
{
    const Index documents_number = tokens.size();

    for(Index i = 0; i < documents_number; i++)
    {
        filter_not_equal_to(tokens(i), delete_words);
    }
}



void TextAnalytics::delete_stop_words(Tensor<Tensor<string,1>,1>& tokens) const
{
    delete_words(tokens, stop_words);
}


/// Delete short words from the documents
/// @param minimum_length Minimum length of the words that new documents must have(including herself)

void TextAnalytics::delete_short_words(Tensor<Tensor<string,1>,1>& documents, const Index& minimum_length) const
{
    const Index documents_number = documents.size();

#pragma omp parallel for
    for(Index i = 0; i < documents_number; i++)
    {
        Tensor<string,1> document = documents(i);

        for(Index j = 0; j < document.size(); j++)
        {
            const Index tokens_number = count_tokens(document(j),' ');
            const Tensor<string, 1> tokens = get_tokens(document(j), ' ');

            string result;

            for(Index k = 0; k < tokens_number; k++)
            {
                if( Index(tokens(k).length()) >= minimum_length )
                {
                    result += tokens(k) + " ";
                }
            }

            document(j) = result;
        }

        documents(i) = document;
    }
}



/// Delete short words from the documents
/// @param maximum_length Maximum length of the words new documents must have(including herself)

void TextAnalytics::delete_long_words(Tensor<Tensor<string,1>,1>& documents, const Index& maximum_length) const
{
    const Index documents_number = documents.size();

#pragma omp parallel for
    for(Index i = 0; i < documents_number; i++)
    {
        Tensor<string,1> document = documents(i);

        for(Index j = 0; j < document.size(); j++)
        {
            const Index tokens_number = count_tokens(document(j),' ');
            const Tensor<string, 1> tokens = get_tokens(document(j), ' ');

            string result;

            for(Index k = 0; k < tokens_number; k++)
            {
                if( Index(tokens(k).length()) <= maximum_length )
                {
                    result += tokens(k) + " ";
                }
            }

            document(j) = result;
        }

        documents(i) = document;
    }
}


void TextAnalytics::delete_blanks(Tensor<string, 1>& vector) const
{
    const Index words_number = vector.size();

    const Index empty_number = count_empty(vector);

    Tensor<string, 1> vector_copy(vector);

    vector.resize(words_number - empty_number);

    Index index = 0;

    string empty_string;

    for(Index i = 0; i < words_number; i++)
    {
        trim(vector_copy(i));

        if(!vector_copy(i).empty())
        {
            vector(index) = vector_copy(i);
            index++;
        }
    }
}


void TextAnalytics::delete_blanks(Tensor<Tensor<string, 1>, 1>& tokens) const
{
    const Index documents_size = tokens.size();

    for(Index i = 0; i < documents_size; i++)
    {
        delete_blanks(tokens(i));
    }
}


/// Reduces inflected(or sometimes derived) words to their word stem, base or root form.

Tensor<Tensor<string,1>,1> TextAnalytics::apply_stemmer(const Tensor<Tensor<string,1>,1>& tokens) const
{
    if(lang == ENG)
    {
        return apply_english_stemmer(tokens);
    }
    else if(lang == SPA)
    {
        //        return apply_spanish_stemmer(tokens);
    }

    return tokens;
}


/// Reduces inflected (or sometimes derived) words to their word stem, base or root form (english language).
/// @param tokens

Tensor<Tensor<string,1>,1> TextAnalytics::apply_english_stemmer(const Tensor<Tensor<string,1>,1>& tokens) const
{
    const Index documents_number = tokens.size();

    Tensor<Tensor<string,1>,1> new_tokenized_documents(documents_number);

    // Set vowels and suffixes

    Tensor<string,1> vowels(6);

    vowels.setValues({"a","e","i","o","u","y"});

    Tensor<string,1> double_consonants(9);

    double_consonants.setValues({"bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"});

    Tensor<string,1> li_ending(10);

    li_ending.setValues({"c", "d", "e", "g", "h", "k", "m", "n", "r", "t"});

    const Index step0_suffixes_size = 3;

    Tensor<string,1> step0_suffixes(step0_suffixes_size);

    step0_suffixes.setValues({"'s'", "'s", "'"});

    const Index step1a_suffixes_size = 6;

    Tensor<string,1> step1a_suffixes(step1a_suffixes_size);

    step1a_suffixes.setValues({"sses", "ied", "ies", "us", "ss", "s"});

    const Index step1b_suffixes_size = 6;

    Tensor<string,1> step1b_suffixes(step1b_suffixes_size);

    step1b_suffixes.setValues({"eedly", "ingly", "edly", "eed", "ing", "ed"});

    const Index step2_suffixes_size = 25;

    Tensor<string,1> step2_suffixes(step2_suffixes_size);

    step2_suffixes.setValues({"ization", "ational", "fulness", "ousness", "iveness", "tional", "biliti", "lessli", "entli", "ation", "alism",
                              "aliti", "ousli", "iviti", "fulli", "enci", "anci", "abli", "izer", "ator", "alli", "bli", "ogi", "li"});

    const Index step3_suffixes_size = 9;

    Tensor<string,1> step3_suffixes(step3_suffixes_size);

    step3_suffixes.setValues({"ational", "tional", "alize", "icate", "iciti", "ative", "ical", "ness", "ful"});

    const Index step4_suffixes_size = 18;

    Tensor<string,1> step4_suffixes(step4_suffixes_size);

    step4_suffixes.setValues({"ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti", "ous",
                              "ive", "ize", "ion", "al", "er", "ic"});

    Tensor<string,2> special_words(40,2);

    special_words(0,0) = "skis";        special_words(0,1) = "ski";
    special_words(1,0) = "skies";       special_words(1,1) = "sky";
    special_words(2,0) = "dying";       special_words(2,1) = "die";
    special_words(3,0) = "lying";       special_words(3,1) = "lie";
    special_words(4,0) = "tying";       special_words(4,1) = "tie";
    special_words(5,0) = "idly";        special_words(5,1) = "idl";
    special_words(6,0) = "gently";      special_words(6,1) = "gentl";
    special_words(7,0) = "ugly";        special_words(7,1) = "ugli";
    special_words(8,0) = "early";       special_words(8,1) = "earli";
    special_words(9,0) = "only";        special_words(9,1) = "onli";
    special_words(10,0) = "singly";     special_words(10,1) = "singl";
    special_words(11,0) = "sky";        special_words(11,1) = "sky";
    special_words(12,0) = "news";       special_words(12,1) = "news";
    special_words(13,0) = "howe";       special_words(13,1) = "howe";
    special_words(14,0) = "atlas";      special_words(14,1) = "atlas";
    special_words(15,0) = "cosmos";     special_words(15,1) = "cosmos";
    special_words(16,0) = "bias";       special_words(16,1) = "bias";
    special_words(17,0) = "andes";      special_words(17,1) = "andes";
    special_words(18,0) = "inning";     special_words(18,1) = "inning";
    special_words(19,0) = "innings";    special_words(19,1) = "inning";
    special_words(20,0) = "outing";     special_words(20,1) = "outing";
    special_words(21,0) = "outings";    special_words(21,1) = "outing";
    special_words(22,0) = "canning";    special_words(22,1) = "canning";
    special_words(23,0) = "cannings";   special_words(23,1) = "canning";
    special_words(24,0) = "herring";    special_words(24,1) = "herring";
    special_words(25,0) = "herrings";   special_words(25,1) = "herring";
    special_words(26,0) = "earring";    special_words(26,1) = "earring";
    special_words(27,0) = "earrings";   special_words(27,1) = "earring";
    special_words(28,0) = "proceed";    special_words(28,1) = "proceed";
    special_words(29,0) = "proceeds";   special_words(29,1) = "proceed";
    special_words(30,0) = "proceeded";  special_words(30,1) = "proceed";
    special_words(31,0) = "proceeding"; special_words(31,1) = "proceed";
    special_words(32,0) = "exceed";     special_words(32,1) = "exceed";
    special_words(33,0) = "exceeds";    special_words(33,1) = "exceed";
    special_words(34,0) = "exceeded";   special_words(34,1) = "exceed";
    special_words(35,0) = "exceeding";  special_words(35,1) = "exceed";
    special_words(36,0) = "succeed";    special_words(36,1) = "succeed";
    special_words(37,0) = "succeeds";   special_words(37,1) = "succeed";
    special_words(38,0) = "succeeded";  special_words(38,1) = "succeed";
    special_words(39,0) = "succeeding"; special_words(39,1) = "succeed";

#pragma omp parallel for
    for(Index i = 0; i < documents_number; i++)
    {
        Tensor<string,1> current_document = tokens(i);

        replace_substring(current_document, "’", "'");
        replace_substring(current_document, "‘", "'");
        replace_substring(current_document, "‛", "'");

        for(Index j = 0; j < current_document.size(); j++)
        {
            string current_word = current_document(j);

            trim(current_word);

            if( contains(special_words.chip(0,1),current_word))
            {
                auto it = find(special_words.data(), special_words.data() + special_words.size(), current_word);

                Index word_index = it - special_words.data();

                current_document(j) = special_words(word_index,1);

                break;
            }

            if(starts_with(current_word,"'"))
            {
                current_word = current_word.substr(1);
            }

            if(starts_with(current_word, "y"))
            {
                current_word = "Y" + current_word.substr(1);
            }

            for(size_t k = 1; k < current_word.size(); k++)
            {
                if(contains(vowels, string(1,current_word[k-1]) ) && current_word[k] == 'y')
                {
                    current_word[k] = 'Y';
                }
            }

            Tensor<string,1> r1_r2(2);

            r1_r2 = get_r1_r2(current_word, vowels);

            bool step1a_vowel_found = false;
            bool step1b_vowel_found = false;

            // Step 0

            for(Index l = 0; l < step0_suffixes_size; l++)
            {
                const string current_suffix = step0_suffixes(l);

                if(ends_with(current_word,current_suffix))
                {
                    current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                    r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                    r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());
                    break;
                }
            }

            // Step 1a

            for(size_t l = 0; l < step1a_suffixes_size; l++)
            {
                const string current_suffix = step1a_suffixes[l];

                if(ends_with(current_word, current_suffix))
                {
                    if(current_suffix == "sses")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "ied" || current_suffix == "ies")
                    {
                        if(current_word.length() - current_suffix.length() > 1)
                        {
                            current_word = current_word.substr(0,current_word.length()-2);
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                        }
                        else
                        {
                            current_word = current_word.substr(0,current_word.length()-1);
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                        }
                    }
                    else if(current_suffix == "s")
                    {
                        for(size_t l = 0; l < current_word.length() - 2; l++)
                        {
                            if(contains(vowels, string(1,current_word[l])))
                            {
                                step1a_vowel_found = true;
                                break;
                            }
                        }

                        if(step1a_vowel_found)
                        {
                            current_word = current_word.substr(0,current_word.length()-1);
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                        }
                    }

                    break;
                }
            }

            // Step 1b

            for(Index k = 0; k < step1b_suffixes_size; k++)
            {
                const string current_suffix = step1b_suffixes[k];

                if(ends_with(current_word, current_suffix))
                {
                    if(current_suffix == "eed" || current_suffix == "eedly")
                    {
                        if(ends_with(r1_r2[0], current_suffix))
                        {
                            current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ee";

                            if(r1_r2[0].length() >= current_suffix.length())
                            {
                                r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ee";
                            }
                            else
                            {
                                r1_r2[0] = "";
                            }

                            if(r1_r2[1].length() >= current_suffix.length())
                            {
                                r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ee";
                            }
                            else
                            {
                                r1_r2[1] = "";
                            }
                        }
                    }
                    else
                    {
                        for(size_t l = 0; l <(current_word.length() - current_suffix.length()); l++)
                        {
                            if(contains(vowels,string(1,current_word[l])))
                            {
                                step1b_vowel_found = true;
                                break;
                            }
                        }

                        if(step1b_vowel_found)
                        {
                            current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());

                            if(ends_with(current_word, "at") || ends_with(current_word, "bl") || ends_with(current_word, "iz"))
                            {
                                current_word = current_word + "e";
                                r1_r2[0] = r1_r2[0] + "e";

                                if(current_word.length() > 5 || r1_r2[0].length() >= 3)
                                {
                                    r1_r2[1] = r1_r2[1] + "e";
                                }
                            }
                            else if(ends_with(current_word, double_consonants))
                            {
                                current_word = current_word.substr(0,current_word.length()-1);
                                r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                                r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                            }
                            else if((r1_r2[0] == "" && current_word.length() >= 3 && !contains(vowels,string(1,current_word[current_word.length()-1])) &&
                                     !(current_word[current_word.length()-1] == 'w' || current_word[current_word.length()-1] == 'x' || current_word[current_word.length()-1] == 'Y') &&
                                     contains(vowels,string(1,current_word[current_word.length()-2])) && !contains(vowels,string(1,current_word[current_word.length()-3]))) ||
                                    (r1_r2[0] == "" && current_word.length() == 2 && contains(vowels,string(1,current_word[0])) && contains(vowels, string(1,current_word[1]))))
                            {
                                current_word = current_word + "e";

                                if(r1_r2[0].length() > 0)
                                {
                                    r1_r2[0] = r1_r2[0] + "e";
                                }

                                if(r1_r2[1].length() > 0)
                                {
                                    r1_r2[1] = r1_r2[1] + "e";
                                }
                            }
                        }
                    }

                    break;
                }
            }

            // Step 1c

            if(current_word.length() > 2 &&(current_word[current_word.length()-1] == 'y' || current_word[current_word.length()-1] == 'Y') &&
                    !contains(vowels, string(1,current_word[current_word.length()-2])))
            {
                current_word = current_word.substr(0,current_word.length()-1) + "i";

                if(r1_r2[0].length() >= 1)
                {
                    r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1) + "i";
                }
                else
                {
                    r1_r2[0] = "";
                }

                if(r1_r2[1].length() >= 1)
                {
                    r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1) + "i";
                }
                else
                {
                    r1_r2[1] = "";
                }
            }

            // Step 2

            for(Index l = 0; l < step2_suffixes_size; l++)
            {
                const string current_suffix = step2_suffixes[l];

                if(ends_with(current_word,current_suffix) && ends_with(r1_r2[0],current_suffix))
                {
                    if(current_suffix == "tional")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "enci" || current_suffix == "anci" || current_suffix == "abli")
                    {
                        current_word = current_word.substr(0,current_word.length()-1) + "e";

                        if(r1_r2[0].length() >= 1)
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1) + "e";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= 1)
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1) + "e";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "entli")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "izer" || current_suffix == "ization")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ize";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ize";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ize";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "ational" || current_suffix == "ation" || current_suffix == "ator")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ate";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[1] = "e";
                        }
                    }
                    else if(current_suffix == "alism" || current_suffix == "aliti" || current_suffix == "alli")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "al";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "al";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "al";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "fulness")
                    {
                        current_word = current_word.substr(0,current_word.length()-4);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-4);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-4);
                    }
                    else if(current_suffix == "ousli" || current_suffix == "ousness")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ous";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ous";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ous";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "iveness" || current_suffix == "iviti")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ive";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ive";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ive";
                        }
                        else
                        {
                            r1_r2[1] = "e";
                        }
                    }
                    else if(current_suffix == "biliti" || current_suffix == "bli")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ble";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ble";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ble";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "ogi" && current_word[current_word.length()-4] == 'l')
                    {
                        current_word = current_word.substr(0,current_word.length()-1);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-1);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-1);
                    }
                    else if(current_suffix == "fulli" || current_suffix == "lessli")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "li" && contains(li_ending, string(1,current_word[current_word.length()-4])))
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }

                    break;
                }
            }

            // Step 3

            for(Index l = 0; l < step3_suffixes_size; l++)
            {
                const string current_suffix = step3_suffixes[l];

                if(ends_with(current_word,current_suffix) && ends_with(r1_r2[0],current_suffix))
                {
                    if(current_suffix == "tional")
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }
                    else if(current_suffix == "ational")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ate";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ate";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "alize")
                    {
                        current_word = current_word.substr(0,current_word.length()-3);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-3);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-3);
                    }
                    else if(current_suffix == "icate" || current_suffix == "iciti" || current_suffix == "ical")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length()) + "ic";

                        if(r1_r2[0].length() >= current_suffix.length())
                        {
                            r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length()) + "ic";
                        }
                        else
                        {
                            r1_r2[0] = "";
                        }

                        if(r1_r2[1].length() >= current_suffix.length())
                        {
                            r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length()) + "ic";
                        }
                        else
                        {
                            r1_r2[1] = "";
                        }
                    }
                    else if(current_suffix == "ful" || current_suffix == "ness")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());
                    }
                    else if(current_suffix == "ative" && ends_with(r1_r2[1],current_suffix))
                    {
                        current_word = current_word.substr(0,current_word.length()-5);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-5);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-5);
                    }

                    break;
                }
            }

            // Step 4

            for(Index l = 0; l < step4_suffixes_size; l++)
            {
                const string current_suffix = step4_suffixes[l];

                if(ends_with(current_word,current_suffix) && ends_with(r1_r2[1],current_suffix))
                {
                    if(current_suffix == "ion" &&(current_word[current_word.length()-4] == 's' || current_word[current_word.length()-4] == 't'))
                    {
                        current_word = current_word.substr(0,current_word.length()-3);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-3);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-3);
                    }
                    else
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());
                    }

                    break;
                }
            }

            // Step 5

            if(r1_r2[1][r1_r2[1].length()-1] == 'l' && current_word[current_word.length()-2] == 'l')
            {
                current_word = current_word.substr(0,current_word.length()-1);
            }
            else if(r1_r2[1][r1_r2[1].length()-1] == 'e')
            {
                current_word = current_word.substr(0,current_word.length()-1);
            }
            else if(r1_r2[0][r1_r2[0].length()-1] == 'e')
            {
                if(current_word.length() >= 4 &&(contains(vowels, string(1,current_word[current_word.length()-2])) ||
                                                 (current_word[current_word.length()-2] == 'w' || current_word[current_word.length()-2] == 'x' ||
                                                  current_word[current_word.length()-2] == 'Y') || !contains(vowels, string(1,current_word[current_word.length()-3])) ||
                                                 contains(vowels, string(1,current_word[current_word.length()-4]))))
                {
                    current_word = current_word.substr(0,current_word.length()-1);
                }
            }

            replace(current_word,"Y","y");
            current_document(j) = current_word;
        }
        new_tokenized_documents(i) = current_document;
    }

    return new_tokenized_documents;
}



/// Reduces inflected(or sometimes derived) words to their word stem, base or root form(spanish language).

Tensor<Tensor<string>> TextAnalytics::apply_spanish_stemmer(const Tensor<Tensor<string>>& tokens) const
{
    const size_t documents_number = tokens.size();

    Tensor<Tensor<string>> new_tokenized_documents(documents_number);

    // Set vowels and suffixes

    string vowels[] = {"a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ü"};

    const Tensor<string> vowels(Tensor<string>(vowels, vowels + sizeof(vowels) / sizeof(string) ));

    string step0_suffixes[] = {"selas", "selos", "sela", "selo", "las", "les", "los", "nos", "me", "se", "la", "le", "lo"};

    const Tensor<string> step0_suffixes(Tensor<string>(step0_suffixes, step0_suffixes + sizeof(step0_suffixes) / sizeof(string) ));

    string step1_suffixes[] = {"amientos", "imientos", "amiento", "imiento", "aciones", "uciones", "adoras", "adores",
                                       "ancias", "logías", "encias", "amente", "idades", "anzas", "ismos", "ables", "ibles",
                                       "istas", "adora", "acion", "ación", "antes", "ancia", "logía", "ución", "ucion", "encia",
                                       "mente", "anza", "icos", "icas", "ion", "ismo", "able", "ible", "ista", "osos", "osas",
                                       "ador", "ante", "idad", "ivas", "ivos", "ico", "ica", "oso", "osa", "iva", "ivo", "ud"};

    const Tensor<string> step1_suffixes(Tensor<string>(step1_suffixes, step1_suffixes + sizeof(step1_suffixes) / sizeof(string) ));

    string step2a_suffixes[] = {"yeron", "yendo", "yamos", "yais", "yan",
                                        "yen", "yas", "yes", "ya", "ye", "yo",
                                        "yó"};

    const Tensor<string> step2a_suffixes(Tensor<string>(step2a_suffixes, step2a_suffixes + sizeof(step2a_suffixes) / sizeof(string) ));

    string step2b_suffixes[] = {"aríamos", "eríamos", "iríamos", "iéramos", "iésemos", "aríais",
                                        "aremos", "eríais", "eremos", "iríais", "iremos", "ierais", "ieseis",
                                        "asteis", "isteis", "ábamos", "áramos", "ásemos", "arían",
                                        "arías", "aréis", "erían", "erías", "eréis", "irían",
                                        "irías", "iréis", "ieran", "iesen", "ieron", "iendo", "ieras",
                                        "ieses", "abais", "arais", "aseis", "éamos", "arán", "arás",
                                        "aría", "erán", "erás", "ería", "irán", "irás",
                                        "iría", "iera", "iese", "aste", "iste", "aban", "aran", "asen", "aron", "ando",
                                        "abas", "adas", "idas", "aras", "ases", "íais", "ados", "idos", "amos", "imos",
                                        "emos", "ará", "aré", "erá", "eré", "irá", "iré", "aba",
                                        "ada", "ida", "ara", "ase", "ían", "ado", "ido", "ías", "áis",
                                        "éis", "ía", "ad", "ed", "id", "an", "ió", "ar", "er", "ir", "as",
                                        "ís", "en", "es"};

    const Tensor<string> step2b_suffixes(Tensor<string>(step2b_suffixes, step2b_suffixes + sizeof(step2b_suffixes) / sizeof(string) ));

    string step3_suffixes[] = {"os", "a", "e", "o", "á", "é", "í", "ó"};

    const Tensor<string> step3_suffixes(Tensor<string>(step3_suffixes, step3_suffixes + sizeof(step3_suffixes) / sizeof(string) ));

    const size_t step0_suffixes_size = step0_suffixes.size();
    const size_t step1_suffixes_size = step1_suffixes.size();
    const size_t step2a_suffixes_size = step2a_suffixes.size();
    const size_t step2b_suffixes_size = step2b_suffixes.size();
    const size_t step3_suffixes_size = step3_suffixes.size();

    for(size_t i = 0; i < documents_number; i++)
    {
        const Tensor<string> current_document_tokens = tokens[i];
        const size_t current_document_tokens_number = current_document_tokens.size();

        new_tokenized_documents[i] = current_document_tokens;

        for(size_t j = 0; j < current_document_tokens_number; j++)
        {
            string current_word = new_tokenized_documents[i][j];

            Tensor<string> r1_r2 = get_r1_r2(current_word, vowels);
            string rv = get_rv(current_word, vowels);

            // STEP 0: attached pronoun

            for(size_t k = 0; k < step0_suffixes_size; k++)
            {
                const string current_suffix = step0_suffixes[k];
                const size_t current_suffix_length = current_suffix.length();

                if(!(ends_with(current_word,current_suffix) && ends_with(rv, current_suffix)))

                    continue;



                const string before_suffix_rv = rv.substr(0,rv.length()-current_suffix_length);
                const string before_suffix_word = current_word.substr(0,current_word.length()-current_suffix_length);

                Tensor<string> presuffix(10);

                presuffix[0] = "ando"; presuffix[1] = "ándo"; presuffix[2] = "ar"; presuffix[3] = "ár";
                presuffix[4] = "er"; presuffix[5] = "ér"; presuffix[6] = "iendo"; presuffix[7] = "iéndo";
                presuffix[4] = "ir"; presuffix[5] = "ír";

                if((ends_with(before_suffix_rv,presuffix)) ||
                  (ends_with(before_suffix_rv,"yendo") && ends_with(before_suffix_word, "uyendo")))
                {
                    current_word = replace_accented(before_suffix_word);
                    rv = replace_accented(before_suffix_rv);
                    r1_r2[0] = replace_accented(r1_r2[0].substr(0,r1_r2[0].length()-current_suffix_length));
                    r1_r2[1] = replace_accented(r1_r2[1].substr(0,r1_r2[1].length()-current_suffix_length));
                }

                break;
            }

            // STEP 1: standard suffix removal

            bool step1_success = false;

            for(size_t k = 0; k < step1_suffixes_size; k++)
            {
                const string current_suffix = step1_suffixes[k];
                const size_t current_suffix_length = current_suffix.length();

                if(!ends_with(current_word, current_suffix))
                {
                    continue;
                }

                if(current_suffix == "amente" && ends_with(r1_r2[0], current_suffix))
                {
                    step1_success = true;

                    current_word = current_word.substr(0,current_word.length()-6);
                    r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-6);
                    rv = rv.substr(0,rv.length()-6);

                    if(ends_with(r1_r2[1],"iv"))
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                        rv = rv.substr(0,rv.length()-2);

                        if(ends_with(r1_r2[1],"at"))
                        {
                            current_word = current_word.substr(0,current_word.length()-2);
                            rv = rv.substr(0,rv.length()-2);
                        }
                    }
                    else if(ends_with(r1_r2[1], "os") || ends_with(r1_r2[1], "ic") || ends_with(r1_r2[1], "ad"))
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        rv = rv.substr(0,rv.length()-2);
                    }
                }
                else if(ends_with(r1_r2[1], current_suffix))
                {
                    step1_success = true;

                    if(current_suffix == "adora" || current_suffix == "ador" || current_suffix == "ación" || current_suffix == "adoras" ||
                       current_suffix == "adores" || current_suffix == "aciones" || current_suffix == "ante" || current_suffix == "antes" ||
                       current_suffix == "ancia" || current_suffix == "ancias")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix_length);
                        rv = rv.substr(0,rv.length()-current_suffix_length);

                        if(ends_with(r1_r2[1], "ic"))
                        {
                            current_word = current_word.substr(0,current_word.length()-2);
                            rv = rv.substr(0,rv.length()-2);
                        }
                    }
                    else if(current_suffix == "logía" || current_suffix == "logías")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length) + "log";
                        rv = rv.substr(0,rv.length()-current_suffix_length) + "log";
                    }
                    else if(current_suffix == "ución" || current_suffix == "uciones")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length) + "u";
                        rv = rv.substr(0,rv.length()-current_suffix_length) + "u";
                    }
                    else if(current_suffix == "encia" || current_suffix == "encias")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length) + "ente";
                        rv = rv.substr(0,rv.length()-current_suffix_length) + "ente";
                    }
                    else if(current_suffix == "mente")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix_length);
                        rv = rv.substr(0,rv.length()-current_suffix_length);

                        if(ends_with(r1_r2[1], "ante") || ends_with(r1_r2[1], "able") || ends_with(r1_r2[1], "ible"))
                        {
                            current_word = current_word.substr(0,current_word.length()-4);
                            rv = rv.substr(0,rv.length()-4);
                        }
                    }
                    else if(current_suffix == "idad" || current_suffix == "idades")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix_length);
                        rv = rv.substr(0,rv.length()-current_suffix_length);

                        if(ends_with(r1_r2[1],"abil"))
                        {
                            current_word = current_word.substr(0,current_word.length()-4);
                            rv = rv.substr(0,rv.length()-4);
                        }
                        else if(ends_with(r1_r2[1],"ic"))
                        {
                            current_word = current_word.substr(0,current_word.length()-2);
                            rv = rv.substr(0,rv.length()-2);
                        }
                        else if(ends_with(r1_r2[1],"iv"))
                        {
                            current_word = current_word.substr(0,current_word.length()-2);
                            rv = rv.substr(0,rv.length()-2);
                        }
                    }
                    else if(current_suffix == "ivo" || current_suffix == "iva" || current_suffix == "ivos" || current_suffix == "ivas")
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix_length);
                        rv = rv.substr(0,rv.length()-current_suffix_length);

                        if(ends_with(r1_r2[1], "at"))
                        {
                            current_word = current_word.substr(0,current_word.length()-2);
                            rv = rv.substr(0,rv.length()-2);
                        }
                    }
                    else
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length);
                        rv = rv.substr(0,rv.length()-current_suffix_length);
                    }
                }

                break;
            }

            if(!step1_success)
            {
                // STEP 2a: verb suffixes beginning 'y'

                for(size_t k = 0; k < step2a_suffixes_size; k++)
                {
                    const string current_suffix = step2a_suffixes[k];
                    const size_t current_suffix_length = current_suffix.length();

                    if(ends_with(rv,current_suffix) && current_word[current_word.length() - current_suffix_length - 1] == 'u')
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length);
                        rv = rv.substr(0,rv.length()-current_suffix_length);

                        break;
                    }
                }

                // STEP 2b: other verb suffixes

                for(size_t k = 0; k < step2b_suffixes_size; k++)
                {
                    const string current_suffix = step2b_suffixes[k];
                    const size_t current_suffix_length = current_suffix.length();

                    if(ends_with(rv, current_suffix))
                    {
                        current_word = current_word.substr(0,current_word.length()-current_suffix_length);
                        rv = rv.substr(0,rv.length()-current_suffix_length);

                        if(current_suffix == "en" || current_suffix == "es" || current_suffix == "éis" || current_suffix == "emos")
                        {
                            if(ends_with(current_word, "gu"))
                            {
                                current_word = current_word.substr(0,current_word.length()-1);
                            }

                            if(ends_with(rv,"gu"))
                            {
                                rv = rv.substr(0,rv.length()-1);
                            }
                        }

                        break;
                    }
                }
            }

            // STEP 3: residual suffix

            for(size_t k = 0; k < step3_suffixes_size; k++)
            {
                const string current_suffix = step3_suffixes[k];
                const size_t current_suffix_length = current_suffix.length();

                if(ends_with(rv, current_suffix))
                {
                    current_word = current_word.substr(0,current_word.length()-current_suffix_length);

                    if(current_suffix == "e" || current_suffix == "é")
                    {
                        rv = rv.substr(0,rv.length()-current_suffix_length);

                        if(ends_with(current_word, "gu") && ends_with(rv,"u"))
                        {
                            current_word = current_word.substr(0,current_word.length()-1);
                        }
                    }

                    break;
                }
            }

            new_tokenized_documents[i][j] = replace_accented(current_word);
        }
    }

    return new_tokenized_documents;
}


/// Delete the numbers of the documents.

void TextAnalytics::delete_numbers(Tensor<Tensor<string,1>,1>& documents) const
{
    const Index documents_number = documents.size();

#pragma omp parallel for
    for(Index i = 0; i < documents_number; i++)
    {
        Tensor<string, 1> document = documents(i);

        const Index document_size = document.size();

        for(Index j = 0; j < document_size; j++)
        {
            Tensor<string,1> tokens = get_tokens(document(j));

            string result;

            for(Index k = 0; k < tokens.size(); k++)
            {
                if(!is_numeric_string(tokens(k)) )
                {
                    result += tokens(k) + " ";
                }
            }

            document(j) = result;
        }

        documents(i) = document;
    }
}


/// Remove emails from documents.

void TextAnalytics::delete_emails(Tensor<Tensor<string,1>,1>& documents) const
{
    const Index documents_number = documents.size();

#pragma omp parallel for
    for(Index i = 0; i < documents_number; i++)
    {
        Tensor<string, 1> document = documents(i);

        for(Index j = 0; j < document.size(); j++)
        {
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
        }

        documents(i) = document;
    }
}


/// Remove the accents of the vowels in the documents.

void TextAnalytics::replace_accented(Tensor<Tensor<string,1>,1>& documents) const
{
    const Index documents_size = documents.size();

    for(Index i = 0; i < documents_size; i++)
    {
        const Index document_size = documents(i).size();

        for(Index j = 0; j < document_size; j++)
        {
            replace_accented(documents(i)(j));
        }
    }
}


/// Remove the accents of the vowels of a word.

void TextAnalytics::replace_accented(string& word) const
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


Tensor<string,1> TextAnalytics::get_r1_r2(const string& word, const Tensor<string,1>& vowels) const
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


string TextAnalytics::get_rv(const string& word, const Tensor<string,1>& vowels) const
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


/// Returns a string with all the text of a file
/// @param path Path of the file to be read

string TextAnalytics::read_txt_file(const string& path) const
{
    if(path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
            << "void load_documents() method.\n"
            << "Data file name is empty.\n";

        throw runtime_error(buffer.str());
    }

    ifstream file(path.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
            << "void load_documents() method.\n"
            << "Cannot open data file: " << path << "\n";

        throw runtime_error(buffer.str());
    }

    string result="", line;

    while(file.good())
    {
        getline(file, line);
        trim(line);
        erase(line, '"');

        if(line.empty()) continue;

        result += line;

        if(file.peek() == EOF) break;
    }

    return result;
}


/// Create a word bag that contains all the unique words of the documents,
/// their frequencies and their percentages in descending order

WordBag TextAnalytics::calculate_word_bag(const Tensor<Tensor<string,1>,1>& tokens) const
{
    const Tensor<string, 1> total = join(tokens);

    const Tensor<Index, 1> count = count_unique(total);

    const Tensor<Index, 1> descending_rank = calculate_rank_greater(count.cast<type>());

    const Tensor<string,1> words = sort_by_rank(get_unique_elements(total), descending_rank);

    const Tensor<Index,1> frequencies = sort_by_rank(count, descending_rank);

    const Tensor<Index,0> total_frequencies = frequencies.sum();

    const Tensor<double,1> percentages = ( 100/double(total_frequencies(0)) * frequencies.cast<double>()  );

    WordBag word_bag;
    word_bag.words = words;
    word_bag.frequencies = frequencies;
    word_bag.percentages = percentages;

    return word_bag;
}



/// Create a word bag that contains the unique words that appear a minimum number
/// of times in the documents, their frequencies and their percentages in descending order.
/// @param minimum_frequency Minimum frequency that words must have.

WordBag TextAnalytics::calculate_word_bag_minimum_frequency(const Tensor<Tensor<string,1>,1>& tokens,
                                                                           const Index& minimum_frequency) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    Tensor<string,1> words = word_bag.words;
    Tensor<Index,1> frequencies = word_bag.frequencies;
    Tensor<double,1> percentages = word_bag.percentages;

    const Tensor<Index,1> indices = get_indices_less_than(frequencies, minimum_frequency);

    delete_indices(words, indices);
    delete_indices(frequencies, indices);
    delete_indices(percentages, indices);

    word_bag.words = words;
    word_bag.frequencies = frequencies;
    word_bag.percentages = percentages;

    return word_bag;
}


/// Create a word bag that contains the unique words that appear a minimum percentage
/// in the documents, their frequencies and their percentages in descending order.
/// @param minimum_percentage Minimum percentage of occurrence that words must have.

WordBag TextAnalytics::calculate_word_bag_minimum_percentage(const Tensor<Tensor<string,1>,1>& tokens,
                                                                            const double& minimum_percentage) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    Tensor<string,1> words = word_bag.words;
    Tensor<Index,1> frequencies = word_bag.frequencies;
    Tensor<double,1> percentages = word_bag.percentages;

    const Tensor<Index,1> indices = get_indices_less_than(percentages, minimum_percentage);

    delete_indices(words, indices);
    delete_indices(frequencies, indices);
    delete_indices(percentages, indices);

    word_bag.words = words;
    word_bag.frequencies = frequencies;
    word_bag.percentages = percentages;

    return word_bag;
}


/// Create a word bag that contains the unique words that appear a minimum ratio
/// of frequency in the documents, their frequencies and their percentages in descending order.
/// @param minimum_ratio Minimum ratio of frequency that words must have.

WordBag TextAnalytics::calculate_word_bag_minimum_ratio(const Tensor<Tensor<string,1>,1>& tokens,
                                                                       const double& minimum_ratio) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    Tensor<string,1> words = word_bag.words;
    Tensor<Index,1> frequencies = word_bag.frequencies;
    Tensor<double,1> percentages = word_bag.percentages;

    const Tensor<Index,0> frequencies_sum = frequencies.sum();

    const Tensor<double,1> ratios = frequencies.cast<double>()/double(frequencies_sum(0));

    const Tensor<Index, 1> indices = get_indices_less_than(ratios, minimum_ratio);

    delete_indices(words, indices);
    delete_indices(frequencies, indices);
    delete_indices(percentages, indices);

    word_bag.words = words;
    word_bag.frequencies = frequencies;
    word_bag.percentages = percentages;

    return word_bag;
}


/// Create a word bag that contains the unique most frequent words whose sum
/// of frequencies is less than the specified number, their frequencies
/// and their percentages in descending order.
/// @param total_frequency Maximum cumulative frequency that words must have.

WordBag TextAnalytics::calculate_word_bag_total_frequency(const Tensor<Tensor<string,1>,1>& tokens,
                                                                         const Index& total_frequency) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    const Tensor<string,1> words = word_bag.words;
    const Tensor<Index, 1> frequencies = word_bag.frequencies;

    Tensor<Index, 1> cumulative_frequencies = frequencies.cumsum(0);

    Index i;

    for( i = 0; i < frequencies.size(); i++)
    {
        if(cumulative_frequencies(i) >= total_frequency)
            break;
    }

    word_bag.words = get_first(words, i);
    word_bag.frequencies = get_first(frequencies, i);

    return word_bag;
}


/// Create a word bag that contains a maximum number of the unique most
/// frequent words, their frequencies and their percentages in descending order.
/// @param maximum_size Maximum size of words Tensor.

WordBag TextAnalytics::calculate_word_bag_maximum_size(const Tensor<Tensor<string,1>,1>& tokens,
                                                                      const Index& maximum_size) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    const Tensor<string, 1> words = word_bag.words;
    const Tensor<Index ,1> frequencies = word_bag.frequencies;

    word_bag.words = get_first(words, maximum_size);
    word_bag.frequencies = get_first(frequencies, maximum_size);

    return word_bag;
}


/// Returns weights.

Index TextAnalytics::calculate_weight(const Tensor<string, 1>& document_words, const WordBag& word_bag) const
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

Tensor<Tensor<string,1>,1> TextAnalytics::preprocess(const Tensor<string,1>& documents) const
{
    Tensor<string,1> documents_copy(documents);

    to_lower(documents_copy);

    delete_punctuation(documents_copy);

    delete_non_printable_chars(documents_copy);

    delete_extra_spaces(documents_copy);

    aux_remove_non_printable_chars(documents_copy);

    Tensor<Tensor<string,1>,1> tokenized_documents = tokenize(documents_copy);

    delete_stop_words(tokenized_documents);

    delete_short_words(tokenized_documents, short_words_length);

    delete_long_words(tokenized_documents, long_words_length);

    replace_accented(tokenized_documents);

    delete_emails(tokenized_documents);

    tokenized_documents = apply_stemmer(tokenized_documents);

    delete_numbers(tokenized_documents);

    delete_blanks(tokenized_documents);

    return tokenized_documents;
}

Tensor<Tensor<string,1>,1> TextAnalytics::preprocess_language_model(const Tensor<string,1>& documents) const
{
    Tensor<string,1> documents_copy(documents);

    to_lower(documents_copy);

    split_punctuation(documents_copy);

    delete_non_printable_chars(documents_copy);

    delete_extra_spaces(documents_copy);

    aux_remove_non_printable_chars(documents_copy);

    Tensor<Tensor<string,1>,1> tokenized_documents = tokenize(documents_copy);

    delete_emails(tokenized_documents);

    delete_blanks(tokenized_documents);

    return tokenized_documents;
}


/// Sets the words that will be removed from the documents.

void TextAnalytics::set_english_stop_words()
{
    stop_words.resize(242);

    stop_words.setValues({"i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "u", "your", "yours", "yourself", "yourselves", "he",
                          "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                          "what", "which", "who", "whom", "this", "that", "these", "those", "im", "am", "m", "is", "are", "was", "were", "be", "been", "being",
                          "have", "has", "s", "ve", "re", "ll", "t", "had", "having", "do", "does", "did", "doing", "would", "d", "shall", "should", "could",
                          "ought", "i'm", "you're", "he's", "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd",
                          "she'd", "we'd", "they'd", "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't",
                          "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't",
                          "let's", "that's", "who's", "what's", "here's", "there's", "when's", "where's", "why's", "how's", "daren't", "needn't", "oughtn't",
                          "mightn't", "shes", "its", "were", "theyre", "ive", "youve", "weve", "theyve", "id", "youd", "hed", "shed", "wed", "theyd",
                          "ill", "youll", "hell", "shell", "well", "theyll", "isnt", "arent", "wasnt", "werent", "hasnt", "havent", "hadnt",
                          "doesnt", "dont", "didnt", "wont", "wouldnt", "shant", "shouldnt", "cant", "cannot", "couldnt", "mustnt", "lets",
                          "thats", "whos", "whats", "heres", "theres", "whens", "wheres", "whys", "hows", "darent", "neednt", "oughtnt",
                          "mightnt", "a", "an", "the", "and", "n", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                          "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
                          "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
                          "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"});
}



void TextAnalytics::set_spanish_stop_words()
{
    stop_words.resize(327);

    stop_words.setValues({"de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al",
                          "es", "lo", "como", "más", "mas", "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "si", "sí", "porque", "esta", "son",
                          "entre", "está", "cuando", "muy", "aún", "aunque", "sin", "sobre", "ser", "tiene", "también", "me", "hasta", "hay", "donde", "han", "quien",
                          "están", "desde", "todo", "nos", "durante", "todos", "uno", "les", "ni", "contra", "otros", "fueron", "ese", "eso", "había",
                          "ante", "ellos", "e", "esto", "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto", "esa",
                          "estos", "mucho", "quienes", "nada", "muchos", "cual", "sea", "poco", "ella", "estar", "haber", "estas", "estaba", "estamos",
                          "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas", "nosotras", "vosotros", "vosotras", "os",
                          "mío", "mía", "míos", "mías", "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros",
                          "nuestras", "vuestro", "vuestra", "vuestros", "vuestras", "esos", "esas", "estoy", "estás", "está", "estamos", "estáis", "están",
                          "esté", "estés", "estemos", "estéis", "estén", "estaré", "estarás", "estará", "estaremos", "estaréis", "estarán", "estaría",
                          "estarías", "estaríamos", "estaríais", "estarían", "estaba", "estabas", "estábamos", "estabais", "estaban", "estuve", "estuviste",
                          "estuvo", "estuvimos", "estuvisteis", "estuvieron", "estuviera", "estuvieras", "estuviéramos", "estuvierais", "estuvieran", "estuviese",
                          "estuvieses", "estuviésemos", "estuvieseis", "estuviesen", "estando", "estado", "estada", "estados", "estadas", "estad", "he",
                          "has", "ha", "hemos", "habéis", "han", "haya", "hayas", "hayamos", "hayáis", "hayan", "habré", "habrás", "habrá", "habremos",
                          "habréis", "habrán", "habría", "habrías", "habríamos", "habríais", "habrían", "había", "habías", "habíamos", "habíais", "habían",
                          "hube", "hubiste", "hubo", "hubimos", "hubisteis", "hubieron", "hubiera", "hubieras", "hubiéramos", "hubierais", "hubieran",
                          "hubiese", "hubieses", "hubiésemos", "hubieseis", "hubiesen", "habiendo", "habido", "habida", "habidos", "habidas", "soy", "eres",
                          "es", "somos", "sois", "son", "sea", "seas", "seamos", "seáis", "sean", "seré", "serás", "será", "seremos", "seréis", "serán",
                          "sería", "serías", "seríamos", "seríais", "serían", "era", "eras", "éramos", "erais", "eran", "fui", "fuiste", "fue", "fuimos",
                          "fuisteis", "fueron", "fuera", "fueras", "fuéramos", "fuerais", "fueran", "fuese", "fueses", "fuésemos", "fueseis", "fuesen", "siendo",
                          "sido", "tengo", "tienes", "tiene", "tenemos", "tenéis", "tienen", "tenga", "tengas", "tengamos", "tengáis", "tengan", "tendré",
                          "tendrás", "tendrá", "tendremos", "tendréis", "tendrán", "tendría", "tendrías", "tendríamos", "tendríais", "tendrían", "tenía",
                          "tenías", "teníamos", "teníais", "tenían", "tuve", "tuviste", "tuvo", "tuvimos", "tuvisteis", "tuvieron", "tuviera", "tuvieras",
                          "tuviéramos", "tuvierais", "tuvieran", "tuviese", "tuvieses", "tuviésemos", "tuvieseis", "tuviesen", "teniendo", "tenido", "tenida",
                          "tenidos", "tenidas", "tened"});
}



/// Clear stop words object.

void TextAnalytics::clear_stop_words()
{
    stop_words.resize(0);
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

Tensor<Index, 1> TextAnalytics::get_sentences_number(const Tensor<string, 1>& documents) const
{
    const Index documents_number = documents.size();

    Tensor<Index, 1> sentences_number(documents_number);

    for(Index i = 0; i < documents_number; i++)
    {
        sentences_number(i) = count_tokens(documents(i), '.');
    }

    return sentences_number;
}


/// Returns a Tensor with the percentage of presence in the documents with respect to all.
/// @param words_name Tensor of words from which you want to know the percentage of presence.

Tensor<double, 1> TextAnalytics::get_words_presence_percentage(const Tensor<Tensor<string, 1>, 1>& tokens, const Tensor<string, 1>& words_name) const
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
Tensor<string, 2> TextAnalytics::calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>& tokens,
                                                                   const Index& minimum_frequency,
                                                                   const Index& combinations_length) const
{
    const Tensor<string, 1> words = join(tokens);

    const TextAnalytics::WordBag top_word_bag = calculate_word_bag_minimum_frequency(tokens, minimum_frequency);
    const Tensor<string, 1> words_name = top_word_bag.words;

    if(words_name.size() == 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "Tensor<string, 2> TextAnalytics::calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>& tokens,"
                  "const Index& minimum_frequency,"
                  "const Index& combinations_length) const method."
               << "Words number must be greater than 1.\n";

        throw runtime_error(buffer.str());
    }

    if(combinations_length < 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "Tensor<string, 2> TextAnalytics::calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>& tokens,"
                  "const Index& minimum_frequency,"
                  "const Index& combinations_length) const method."
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

    const Tensor<string, 1> combinated_words_frequency = to_string_tensor( ( count_unique( combinated_words ) ) );

    Tensor<string, 2> combinated_words_frequency_matrix(combinated_words_frequency.size(),2);

    combinated_words_frequency_matrix.chip(0,1) = get_unique_elements(combinated_words),"Combinated words");
    combinated_words_frequency_matrix.chip(1,0) = combinated_words_frequency,"Frequency");

    combinated_words_frequency_matrix = combinated_words_frequency_matrix.sort_descending_strings(1);

//    return(combinated_words_frequency_matrix);

    return Tensor<string,2>();
}


/// Returns the correlations of words that appear a minimum percentage of times
/// with the targets in descending order.
/// @param minimum_percentage Minimum percentage of frequency that the word must have.

Tensor<string, 2> TextAnalytics::top_words_correlations(const Tensor<Tensor<string, 1>, 1>& tokens,
                                                     const double& minimum_percentage,
                                                     const Tensor<Index, 1>& targets) const
{
    const TextAnalytics::WordBag top_word_bag = calculate_word_bag_minimum_percentage(tokens, minimum_percentage);
    const Tensor<string> words_name = top_word_bag.words;

    if(words_name.size() == 0)
    {
        cout << "There are no words with such high percentage of appearance" << endl;
    }

    Tensor<string> new_documents(tokens.size());

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
}
*/


void load_documents(const string& path)
{
/*
    const Index original_size = documents.size();

    if(path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "void load_documents() method.\n"
               << "Data file name is empty.\n";

        throw runtime_error(buffer.str());
    }

    ifstream file(path.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "void load_documents() method.\n"
               << "Cannot open data file: " << path << "\n";

        throw runtime_error(buffer.str());
    }

    Tensor<Tensor<string,1>, 1> documents_copy(documents);

    documents.resize(original_size + 1);

    Tensor<Tensor<string,1>, 1> targets_copy(targets);

    targets.resize(original_size + 1);

    for(Index i = 0; i < original_size; i++)
    {
        documents(i) = documents_copy(i);
        targets(i) = targets_copy(i);
    }

    Index lines_count = 0;
    Index lines_number = 0;

    string line;

    while(file.good())
    {
        getline(file, line);
        trim(line);
        erase(line, '"');

        if(line.empty()) continue;

        lines_number++;

        if(file.peek() == EOF) break;
    }

    file.close();

    Tensor<string, 1> document(lines_number);
    Tensor<string, 1> document_target(lines_number);

    ifstream file2(path.c_str());

    Index tokens_number = 0;

    string delimiter = "";

    while(file2.good())
    {
        getline(file2, line);

        if(line.empty()) continue;

        if(line[0]=='"')
        {
            replace(line,"\"\"", "\"");
            line = "\""+line;
            delimiter = "\"\"";
        }

        if( line.find("\"" + separator) != string::npos) replace(line,"\"" + separator, "\"\"" + separator);

        tokens_number = count_tokens(line,delimiter + separator);
        Tensor<string,1> tokens = get_tokens(line, delimiter + separator);

        if(tokens_number == 1)
        {
            if(tokens(0).find(delimiter,0) == 0) document(lines_count) += tokens(0).substr(delimiter.length(), tokens(0).size());
            else document(lines_count) += " " + tokens(0);
        }
        else
        {
            if(tokens_number > 2)
            {
                ostringstream buffer;

                buffer << "OpenNN Exception: TextAnalytics class.\n"
                       << "void load_documents() method.\n"
                       << "Found more than one separator in line: " << line << "\n";

                throw runtime_error(buffer.str());
            }
            if(tokens(0).empty() && tokens(1).empty())  continue;

            document(lines_count) += " " + tokens(0);
            document_target(lines_count) += tokens(1);
            delimiter = "";
            lines_count++;

        }

        if(file2.peek() == EOF) break;
    }

    Tensor<string,1> document_copy(lines_count);
    Tensor<string,1> document_target_copy(lines_count);

    copy(/*execution::par,
        document.data(),
        document.data() + lines_count,
        document_copy.data());

    copy(/*execution::par,
        document_target.data(),
        document_target.data() + lines_count,
        document_target_copy.data());

    documents(original_size) = document_copy;
    targets(original_size) = document_target_copy;

    file2.close();*/
}


/// Generates a text output based on the neural network and some input letters given by the user.
/// @param text_generation_alphabet TextGenerationAlphabet object used for the text generation model
/// @param input_string Input string given by the user
/// @param max_length Maximum length of the returned string
/// @param one_word Boolean, if true returns just one word, if false returns a phrase
/*
string TextAnalytics::calculate_text_outputs(TextGenerationAlphabet& text_generation_alphabet, const string& input_string, const Index& max_length, const bool& one_word)
{
    string result = one_word ? generate_word(text_generation_alphabet, input_string, max_length) : generate_phrase(text_generation_alphabet, input_string, max_length);

    return result;
}


/// @todo TEXT GENERATION

string TextAnalytics::generate_word(TextGenerationAlphabet& text_generation_alphabet, const string& first_letters, const Index& length)
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

string TextAnalytics::generate_phrase(TextGenerationAlphabet& text_generation_alphabet, const string& first_letters, const Index& length)
{
/*
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


string TextGenerationAlphabet::get_text() const
{
    return text;
}


Tensor<type, 2> TextGenerationAlphabet::get_data_tensor() const
{
    return data_tensor;
}


Tensor<string, 1> TextGenerationAlphabet::get_alphabet() const
{
    return alphabet;
}


Index TextGenerationAlphabet::get_alphabet_length() const
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


void TextGenerationAlphabet::print() const
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

    copy(/*execution::par,
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

    ta.replace_accented(text);

    transform(text.begin(), text.end(), text.begin(), ::tolower); // To lower
}


Index TextGenerationAlphabet::get_alphabet_index(const char& ch) const
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


Tensor<type, 1> TextGenerationAlphabet::one_hot_encode(const string &ch) const
{
    Tensor<type, 1> result(alphabet.size());

    result.setZero();

    const int word_index = get_alphabet_index(ch[0]);

    result(word_index) = type(1);

    return result;
}


Tensor<type, 2> TextGenerationAlphabet::multiple_one_hot_encode(const string &phrase) const
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


string TextGenerationAlphabet::one_hot_decode(const Tensor<type, 1>& tensor) const
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


string TextGenerationAlphabet::multiple_one_hot_decode(const Tensor<type, 2>& tensor) const
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


Tensor<type, 2> TextGenerationAlphabet::str_to_input(const string &input_string) const
{
    Tensor<type, 2> input_data = multiple_one_hot_encode(input_string);

    Tensor<type, 2> flatten_input_data(1, input_data.size());

    copy(/*execution::par,
        input_data.data(),
        input_data.data() + input_data.size(),
        flatten_input_data.data());

    return flatten_input_data;
}


string TextGenerationAlphabet::output_to_str(const Tensor<type, 2>&flatten_output_data) const
{
    const Index alphabet_length = get_alphabet_length();

    const Index tensor_size = Index(flatten_output_data.size()/alphabet_length);

    Tensor<type, 2> output_data(tensor_size, alphabet_length);

    copy(/*execution::par,
        flatten_output_data.data(),
        flatten_output_data.data() + tensor_size, output_data.data());

    return multiple_one_hot_decode(output_data);
}

}
*/


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
