//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "opennn_strings.h"
#include "data_set.h"

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

    std::string::size_type pos = 0;

    while ( s.find(sep, pos) != std::string::npos )
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

    while ((pos = str.find(sep,pos)) != string::npos)
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
        catch(const invalid_argument&)
        {
            type_vector(i) = static_cast<type>(nan(""));
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
        catch(const invalid_argument&)
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
};



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
};



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
    const string format_15  = "(0[1-9]|[1-2][0-9]|3[0-1])[.|/](0[1-9]|1[0-2])[.|/](20[0-9]{2}|[2-9][0-9]{3})\\s([0-1][0-9]|2[0-3])[:]([0-5][0-9])[:]([0-5][0-9])[.][0-9]{6}";

    const regex regular_expression(format_1 + "|" + format_2 + "|" + format_3 + "|" + format_4 + "|" + format_5 + "|" + format_6 + "|" + format_7 + "|" + format_8
                                   + "|" + format_9 + "|" + format_10 + "|" + format_11 +"|" + format_12  + "|" + format_13 + "|" + format_14 +"|" + format_15);

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
        if(ends_with(word,endings[i]))
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
    const string format_15  = "(0[1-9]|[1-2][0-9]|3[0-1])[.|/](0[1-9]|1[0-2])[.|/](20[0-9]{2}|[2-9][0-9]{3})\\s([0-1][0-9]|2[0-3])[:]([0-5][0-9])[:]([0-5][0-9])[.][0-9]{6}";

    const regex regular_expression(format_1 + "|" + format_2 + "|" + format_3 + "|" + format_4 + "|" + format_5 + "|" + format_6 + "|" + format_7 + "|" + format_8
                                   + "|" + format_9 + "|" + format_10 + "|" + format_11 +"|" + format_12  + "|" + format_13 + "|" + format_14 + "|" + format_15);

    regex_search(date, matchs, regular_expression);

    if(matchs[1] != "") // yyyy/mm/dd hh:mm:ss
    {
        if(stoi(matchs[1].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw invalid_argument(buffer.str());
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

            throw invalid_argument(buffer.str());
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

            throw invalid_argument(buffer.str());
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

            throw invalid_argument(buffer.str());
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

            throw invalid_argument(buffer.str());
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

            throw invalid_argument(buffer.str());
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

            throw invalid_argument(buffer.str());
        }
        else
        {
            regex_search(date, month, months);

            Index month_number = 0;

            if(!month.empty())
            {
                for(Index i = 1; i < 13; i++)
                {
                    if(month[static_cast<size_t>(i)] != "") month_number = i;
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

            throw invalid_argument(buffer.str());
        }
        else
        {
            regex_search(date, month, months);

            Index month_number = 0;
            if(!month.empty())
            {
                for(Index i =1 ; i<13  ; i++)
                {
                    if(month[static_cast<size_t>(i)] != "") month_number = i;
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

            throw invalid_argument(buffer.str());
        }
        else
        {
            regex_search(date, month, months);

            Index month_number = 0;
            if(!month.empty())
            {
                for(Index i =1 ; i<13  ; i++)
                {
                    if(month[static_cast<size_t>(i)] != "") month_number = i;
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

            throw invalid_argument(buffer.str());
        }
        else
        {
            regex_search(date,month,months);

            Index month_number = 0;

            if(!month.empty())
            {
                for(Index i =1 ; i<13  ; i++)
                {
                    if(month[static_cast<size_t>(i)] != "") month_number = i;
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

            throw invalid_argument(buffer.str());
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
    else if(matchs[59] != "") // dd/mm/yyyy hh:mm:ss.ssssss
    {
        if(stoi(matchs[60].str()) < 1970)
        {
            ostringstream buffer;

            buffer << "OpenNN Exception: DataSet Class.\n"
                   << "time_t date_to_timestamp(const string&) method.\n"
                   << "Cannot convert dates below 1970.\n";

            throw invalid_argument(buffer.str());
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
    else if(is_numeric_string(date)){
    }
    else
    {
        ostringstream buffer;
        buffer << "OpenNN Exception: DataSet Class.\n"
               << "time_t date_to_timestamp(const string&) method.\n"
               << "Date format (" << date << ") is not implemented.\n";
        throw logic_error(buffer.str());
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

void replace_all_appearances(std::string& s, std::string const& toReplace, std::string const& replaceWith) {
    std::string buf;

    std::size_t pos = 0;
    std::size_t prevPos;

    // Reserves rough estimate of final size of string.
    buf.reserve(s.size());

    while (true) {

        prevPos =    pos;
        pos = s.find(toReplace, pos);

        if (pos == std::string::npos)
            break;

        buf.append(s, prevPos, pos - prevPos);
        if (buf.back() == '_')
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

        if (s[0] == '$')
            out=s;

        replace_all_appearances(s, "if","i_f");
        replace_all_appearances(s, "new","ne_w");
        replace_all_appearances(s, "for","fo_r");
        replace_all_appearances(s, "try","tr_y");
        replace_all_appearances(s, "var","va_r");
        replace_all_appearances(s, "int","in_t");
        replace_all_appearances(s, "case","ca_se");
        replace_all_appearances(s, "rise","ri_se");
        replace_all_appearances(s, "super","sup_er");
        replace_all_appearances(s, "catch","cat_ch");
        replace_all_appearances(s, "float","flo_at");
        replace_all_appearances(s, "while","whi_le");
        replace_all_appearances(s, "break","bre_ak");
        replace_all_appearances(s, "import","im_port");
        replace_all_appearances(s, "return","ret_urn");
        replace_all_appearances(s, "switch","sw_itch");
        replace_all_appearances(s, "string","str_ing");
        replace_all_appearances(s, "double","dou_ble");
        replace_all_appearances(s, "default","def_ault");
        replace_all_appearances(s, "function","func_tion");
        replace_all_appearances(s, "class","cla_ss");
        replace_all_appearances(s, "max","ma_x");
        replace_all_appearances(s, "min","mi_n");

        for (char& c: s)
        {
            if (c=='1'){ out+="_one_";   continue;}
            if (c=='2'){ out+="_two_";   continue;}
            if (c=='3'){ out+="_three_"; continue;}
            if (c=='4'){ out+="_four_";  continue;}
            if (c=='5'){ out+="_five_";  continue;}
            if (c=='6'){ out+="_six_";   continue;}
            if (c=='7'){ out+="_seven_"; continue;}
            if (c=='8'){ out+="_eight_"; continue;}
            if (c=='9'){ out+="_nine_";  continue;}
            if (c=='0'){ out+="_zero_";  continue;}

            if (c=='/'){ out+="_div_";   continue;}
            if (c=='*'){ out+="_mul_";   continue;}
            if (c=='+'){ out+="_sum_";   continue;}
            if (c=='-'){ out+="_res_";   continue;}
            if (c=='='){ out+="_equ_";   continue;}
            if (c=='!'){ out+="_not_";   continue;}

            if (c=='&'){ out+="_amprsn_"; continue;}
            if (c=='?'){ out+="_ntrgtn_"; continue;}
            if (c=='<'){ out+="_lower_" ; continue;}
            if (c=='>'){ out+="_higher_"; continue;}

            if (isalnum(c)!=0){ out += c; continue;}
            if (isalnum(c)==0){ out+='_'; continue;}
        }


        return out;
}


vector<string> get_words_in_a_string(string str)
{
    vector<string> output;
    string word = "";

    for (auto x : str)
    {
        if (isalnum(x))
        {
            word = word + x;
        }else if (x=='_')
        {
            word = word + x;
        }
        else
        //if (x == ' ')
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

    for(i=0; i<slen; i++)
    {
        for(j=0; j<wordlen; j++)
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

    replace_first_and_last_char_with_missing_label(str, ';', "NA");
    replace_first_and_last_char_with_missing_label(str, ',', "NA");

    replace_double_char_with_label(str, ";", "NA");
    replace_double_char_with_label(str, ",", "NA");

    replac_substring_within_quotes(str, ",", "");
    replac_substring_within_quotes(str, ";", "");
}


void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &missing_label)
{
    if (!str.empty())
    {
        if (str[0] == target_char)
        {
            string new_string = missing_label + target_char;
            str.replace(0, 1, new_string);
        }

        if (str[str.length() - 1] == target_char)
        {
            string new_string = target_char + missing_label;
            str.replace(str.length() - 1, 1, new_string);
        }
    }
}


void replace_double_char_with_label(string &str, const string &target_char, const string &missing_label)
{
    string target_pattern = target_char + target_char;
    string new_pattern = target_char + missing_label + target_char;

    size_t pos = 0;
    while ((pos = str.find(target_pattern, pos)) != string::npos)
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

    while (std::regex_search(prefix, match, r))
    {
        string match_str = match.str();
        string replaced_str = match_str;
        size_t pos = 0;
        while ((pos = replaced_str.find(target, pos)) != string::npos)
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

void remove_non_printable_chars( std::string& wstr)
{
    // get the ctype facet for wchar_t (Unicode code points in pactice)
    typedef std::ctype< wchar_t > ctype ;
    const ctype& ct = std::use_facet<ctype>( std::locale() ) ;

    // remove non printable Unicode characters
    wstr.erase( std::remove_if( wstr.begin(), wstr.end(),
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
        str.erase(std::remove_if(str.begin(), str.end(), isNotAlnum), str.end());
}

bool find_string_in_tensor(Tensor<string, 1>& t, string val)
{
    for (Index i = 0; i < t.dimension(0); ++i)
    {
        string elem = t(i);

        if (elem == val)
        {
            return true;
        }
    }
        return false;
}

string get_word_from_token(string& token)
{
    string word = "";
    for (char& c : token)
    {
        if ( c!=' ' && c!='=' )
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

Tensor<string, 1> push_back_string (Tensor<string, 1>& tens, const string& str)
{
    Tensor<string, 1> aux_tensor(tens.dimension(0)+1);

    for (int i = 0; i < tens.dimension(0); i++)
    {
        aux_tensor(i) = tens(i);
    }

    aux_tensor(tens.dimension(0)) = str;

    return aux_tensor;
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

    if (programming_languaje == "javascript") { option = 1; }
    else if (programming_languaje == "php")   { option = 2; }
    else if (programming_languaje == "python"){ option = 3; }
    else if (programming_languaje == "c")     { option = 4; }

    int dimension = outputs.dimension(0);

    while (getline(ss, token, '\n'))
    {
        if (token.size() > 1 && token.back() == '{'){ break; }
        if (token.size() > 1 && token.back() != ';'){ token += ';'; }
        tokens = push_back_string(tokens, token);
    }

    for (int i = 0; i < tokens.dimension(0); i++)
    {
        string s = tokens(i);
        string word = "";

        for (char& c : s)
        {
            if ( c!=' ' && c!='=' ){ word += c; }
            else { break; }
        }

        if (word.size() > 1)
        {
            found_tokens = push_back_string(found_tokens, word);
        }
    }

    new_variable = found_tokens[found_tokens.size()-1];
    old_variable = outputs[dimension-1];

    if (new_variable != old_variable)
    {
        int j = found_tokens.size();

        for (int i = dimension; i --> 0;)
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
                    out = push_back_string(out, out_string);
                break;

                //Php
                case 2:
                    out_string = "$";
                    out_string += old_variable;
                    out_string += " = ";
                    out_string += "$";
                    out_string += new_variable;
                    out_string += ";";
                    out = push_back_string(out, out_string);
                break;

                //Python
                case 3:
                    out_string = old_variable;
                    out_string += " = ";
                    out_string += new_variable;
                    out = push_back_string(out, out_string);
                break;

                //C
                case 4:
                    out_string = "double ";
                    out_string = old_variable;
                    out_string += " = ";
                    out_string += new_variable;
                    out_string += ";";
                    out = push_back_string(out, out_string);
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

    for (int i = 0; i < inputs_names.dimension(0); i++)
    {
        if (inputs_names[i].empty())
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

    for (int i = 0; i < outputs_names.dimension(0); i++)
    {
        if (outputs_names[i].empty())
        {
            outputs(i) = "output_" + to_string(i);
        }
        else
        {
            output_name_aux = outputs_names[i];
            outputs(i) = replace_non_allowed_programming_expressions(output_name_aux);
        }
    }

    buffer_out = push_back_string(buffer_out, buffer.str());

    output(0) = inputs;
    output(1) = outputs;
    output(2) = buffer_out;

    return output;
}

string round_to_precision_string(type x, const int& precision)
{    
    type factor = pow(10,precision);
    type rounded_value = (round(factor*x))/factor;
    stringstream ss;
    ss << fixed << setprecision(precision) << rounded_value;
    string result = ss.str();
    return result;
}

Tensor<string,2> round_to_precision_string_matrix(Tensor<type,2> matrix, const int& precision)
{
    Tensor<string,2> matrix_rounded(matrix.dimension(0), matrix.dimension(1));

    type factor = pow(10,precision);

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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
