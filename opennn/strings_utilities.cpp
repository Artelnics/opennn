//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   S T R I N G S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "strings_utilities.h"
#include "tensors.h"

namespace opennn
{

void fill_tokens(const string& text, const string& separator, vector<string>& tokens)
{
    fill(tokens.begin(), tokens.end(), "");

    // Skip delimiters at beginning

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
            tokens[index++].clear();
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
        tokens_number--;

    if(position == text.length())
        tokens_number--;

    return tokens_number + 1;
}


vector<string> get_tokens(const string& text, const string& separator)
{
    const Index tokens_number = count_tokens(text, separator);

    vector<string> tokens(tokens_number);

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

    return tokens;
}


Tensor<type, 1> to_type_vector(const string& text, const string& separator)
{
    const vector<string> tokens = get_tokens(text, separator);

    const Index tokens_size = tokens.size();

    Tensor<type, 1> type_vector(tokens_size);

    for(Index i = 0; i < tokens_size; i++)
    {
        try
        {
            type_vector(i) = type(stof(tokens[i]));
        }
        catch(const exception&)
        {
            type_vector(i) = type(nan(""));
        }
    }

    return type_vector;
}


Tensor<Index, 1> to_index_vector(const string& text, const string& separator)
{
    const vector<string> tokens = get_tokens(text, separator);

    const Index tokens_size = tokens.size();

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


vector<string> get_unique(const vector<string>& tokens)
{
    string result;

    for(size_t i = 0; i < tokens.size(); i++)
        if(!contains_substring(result, " " + tokens[i] + " "))
            result += tokens[i] + " ";

    return get_tokens(result, " ");
}


Tensor<Index, 1> count_unique(const vector<string>& tokens)
{
    const vector<string> unique_elements = get_unique(tokens);

    const Index unique_size = unique_elements.size();

    Tensor<Index, 1> unique_count(unique_size);

    for(Index i = 0; i < unique_size; i++)
        unique_count(i) = Index(count(tokens.data(), tokens.data() + tokens.size(), unique_elements[i]));

    return unique_count;
}


bool is_numeric_string(const string& text)
{
    try
    {
        size_t index;
        [[maybe_unused]] double value = std::stod(text, &index);

        return (index == text.size() || (text.find('%') != string::npos && index + 1 == text.size()));
    }
    catch (const std::exception&)
    {
        return false;
    }
}


bool is_date_time_string(const string& text)
{
    if(is_numeric_string(text))
        return false;

    const string year = "(19[0-9][0-9]|20[0-9][0-9])";
    const string month = "(0[1-9]|1[0-2])";
    const string day = "(0[1-9]|[12][0-9]|3[01])";
    const string hour = "([01]?[0-9]|2[0-3])";
    const string minute = "([0-5][0-9])";
    const string second = "([0-5][0-9])";
    const string delimiter = "[-|/|.|\\s]";
    const string am_pm = "[AP]M";
    const string full_month = "([Jj]an(?:uary)?|[Ff]eb(?:ruary)?|[Mm]ar(?:ch)?|[Aa]pr(?:il)?|[Mm]ay|[Jj]un(?:e)?|[Jj]ul(?:y)|[Aa]ug(?:ust)?|[Ss]ep(?:tember)?|[Oo]ct(?:ober)?|[Nn]ov(?:ember)?|[Dd]ec(?:ember)?)";

    // Combine the formats into a single regex
    const regex regular_expression(
        year + delimiter + month + delimiter + day + "(\\s+" + hour + ":" + minute + "(:" + second + ")?)?|"
        + day + delimiter + month + delimiter + year + "(\\s+" + hour + ":" + minute + "(:" + second + ")?(\\s+" + am_pm + ")?)?|"
        + year + delimiter + full_month + delimiter + day + "(\\s+" + hour + ":" + minute + "(:" + second + ")?)?|"
        + full_month + "\\s+" + day + "[,\\.]?\\s+" + year + "(\\s+" + hour + ":" + minute + ")?|"
        + day + delimiter + month + delimiter + year + "\\s+" + hour + ":" + minute + ":" + second + "\\.\\d{6}|"
        + year + delimiter + month + delimiter + day + "\\s+" + hour + ":" + minute + ":" + second + "\\.\\d{6}|"
        + "^\\d{1,2}/\\d{1,2}/\\d{4}$|"
        + "^" + hour + ":" + minute + ":" + second + "$");

    return regex_match(text, regular_expression);
}


bool is_email(const string& word)
{
    const regex pattern("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+");

    return regex_match(word, pattern);
}


time_t date_to_timestamp(const string& date, const Index& gmt)
{
    struct tm time_structure = {};
    regex date_patterns[] =
    {
        // yyyy-mm-dd hh:mm:ss.sss
        regex(R"((\d{4})[-/.](\d{2})[-/.](\d{2}) (\d{2}):(\d{2}):(\d{2})\.(\d+))"),

        // yyyy/mm/dd hh:mm:ss
        regex(R"((\d{4})[-/.](\d{2})[-/.](\d{2}) (\d{2}):(\d{2}):(\d{2}))"),

        // yyyy/mm/dd hh:mm
        regex(R"((\d{4})[-/.](\d{2})[-/.](\d{2}) (\d{2}):(\d{2}))"),

        // yyyy/mm/dd
        regex(R"((\d{4})[-/.](\d{2})[-/.](\d{2}))"),

        // dd/mm/yyyy hh:mm:ss
        regex(R"((\d{2})[-/.](\d{2})[-/.](\d{4}) (\d{2}):(\d{2}):(\d{2}))"),

        // dd/mm/yyyy
        regex(R"((\d{2})[-/.](\d{2})[-/.](\d{4}))"),

        // yyyy/mm
        regex(R"((\d{4})[-/.](\d{2}))"),

        // hh:mm:ss
        regex(R"((\d{2}):(\d{2}):(\d{2}))")
    };

    for (const auto& pattern : date_patterns)
    {
        smatch matches;

        if(regex_match(date, matches, pattern))
        {
            if (matches.size() > 1)
            {
                if (matches[1].matched) time_structure.tm_year = stoi(matches[1].str()) - 1900;
                if (matches[2].matched) time_structure.tm_mon = stoi(matches[2].str()) - 1;
                if (matches[3].matched) time_structure.tm_mday = stoi(matches[3].str());
                if (matches[4].matched) time_structure.tm_hour = stoi(matches[4].str()) - gmt;
                if (matches[5].matched) time_structure.tm_min = stoi(matches[5].str());
                if (matches[6].matched) time_structure.tm_sec = stoi(matches[6].str());

                return mktime(&time_structure);
            }
        }
    }

    throw runtime_error("Date format (" + date + ") is not implemented.");
}


bool contains_substring(const string& text, const string& sub_string)
{
    return text.find(sub_string) != string::npos;
}


void replace_all_word_appearances(string& text, const string& to_replace, const string& replace_with)
{
    string buffer;
    size_t position = 0;
    size_t previous_position;
    const string underscore = "_";

    // Reserve a rough estimate of the override size of the chain

    buffer.reserve(text.size());

    while(true)
    {
        previous_position = position;
        position = text.find(to_replace, position);

        if(position == string::npos) break;

        // Verify that there are no letters before or after to_replace

        if((previous_position == 0 || !isalpha(text[previous_position - 1]))
        && (position + to_replace.size() == text.size() || !isalpha(text[position + to_replace.size()])))
        {
            // Verify that there are no underscores before or after to_replace

            if((previous_position == 0 || text[previous_position - 1] != '_')
            && (position + to_replace.size() == text.size() || text[position + to_replace.size()] != '_'))
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


void replace_all_appearances(string& text, string const& to_replace, string const& replace_with)
{
    string buffer;
    size_t position = 0;
    size_t previous_position;

    // Reserves rough estimate of override size of string

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


void trim(string& text)
{
    // Prefixing spaces

    text.erase(0, text.find_first_not_of(" \t\n\r\f\v\b"));

    // Surfixing spaces

    text.erase(text.find_last_not_of(" \t\n\r\f\v\b") + 1);

    // Special character and string modifications

    replace_first_and_last_char_with_missing_label(text, ';', "NA", "");
    replace_first_and_last_char_with_missing_label(text, ',', "NA", "");

    replace_double_char_with_label(text, ";", "NA");
    replace_double_char_with_label(text, ",", "NA");

    replace_substring_within_quotes(text, ",", "");
    replace_substring_within_quotes(text, ";", "");
}


void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &first_missing_label, const string &last_missing_label)
{
    if (str.empty()) return;
    
    if(str[0] == target_char)
    {
        const string new_string = first_missing_label + target_char;
        str.replace(0, 1, new_string);
    }

    if(str[str.length() - 1] == target_char)
    {
        const string new_string = target_char + last_missing_label;
        str.replace(str.length() - 1, 1, new_string);
    }    
}


void replace_double_char_with_label(string &str, const string &target_char, const string &missing_label)
{
    const string target_pattern = target_char + target_char;
    const string new_pattern = target_char + missing_label + target_char;

    size_t position = 0;

    while(position = str.find(target_pattern, position) != string::npos)
    {
        str.replace(position, target_pattern.length(), new_pattern);
        position += new_pattern.length();
    }
}


void replace_substring_within_quotes(string &str, const string &target, const string &replacement)
{
/*
    regex r("\"([^\"]*)\"");

    string result;
    smatch match;

    string prefix = str;

    string::const_iterator search_start(str.begin());


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
*/

    regex r("\"([^\"]*)\"");
    string result;
    string::const_iterator search_start(str.begin());
    smatch match;

    while (regex_search(search_start, str.cend(), match, r))
    {
        result += string(search_start, match[0].first); // Append text before match
        string quoted_content = match[1].str(); // Extract quoted content

        size_t position = 0;
        while ((position = quoted_content.find(target, position)) != string::npos)
        {
            quoted_content.replace(position, target.length(), replacement);
            position += replacement.length();
        }

        result += "\"" + quoted_content + "\""; // Append updated quoted content
        search_start = match[0].second; // Move search start past the current match
    }

    result += string(search_start, str.cend()); // Append remaining text
    str = result;
}


void erase(string& text, const char& character)
{
    text.erase(remove(text.begin(), text.end(), character), text.end());
}


string get_trimmed(const string& text)
{
//    string output(text);

    //prefixing spaces

//    output.erase(0, output.find_first_not_of(' '));
//    output.erase(0, output.find_first_not_of('\t'));
//    output.erase(0, output.find_first_not_of('\n'));
//    output.erase(0, output.find_first_not_of('\r'));
//    output.erase(0, output.find_first_not_of('\f'));
//    output.erase(0, output.find_first_not_of('\v'));

    //surfixing spaces

//    output.erase(output.find_last_not_of(' ') + 1);
//    output.erase(output.find_last_not_of('\t') + 1);
//    output.erase(output.find_last_not_of('\n') + 1);
//    output.erase(output.find_last_not_of('\r') + 1);
//    output.erase(output.find_last_not_of('\f') + 1);
//    output.erase(output.find_last_not_of('\v') + 1);
//    output.erase(output.find_last_not_of('\b') + 1);

 //   return output;

    auto start = find_if_not(text.begin(), text.end(), ::isspace);

    auto end = find_if_not(text.rbegin(), text.rend(), ::isspace).base();

    return (start < end) ? string(start, end) : string();
}


bool has_numbers(const vector<string>& string_list)
{
    for(size_t i = 0; i < string_list.size(); i++)
        if(is_numeric_string(string_list[i])) 
            return true;

    return false;
}


bool is_not_numeric(const vector<string>& string_list)
{
    for(size_t i = 0; i < string_list.size(); i++)
        if(is_numeric_string(string_list[i])) 
            return false;

    return true;
}


void delete_non_printable_chars(string& text)
{
    typedef ctype<wchar_t> ctype;

    const ctype& ct = use_facet<ctype>(locale());

    text.erase(remove_if(text.begin(),
                         text.end(),
                         [&ct](wchar_t ch) {return !ct.is(ctype::print, ch);}),
                         text.end());
}


void replace_substring(vector<string>& data, const string& find_what, const string& replace_with)
{
    const Index size = data.size();

    for(Index i = 0; i < size; i++)
    {
        size_t position = 0;

        while((position = data[i].find(find_what, position)) != string::npos)
        {
            data[i].replace(position, find_what.length(), replace_with);

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


bool is_not_alnum(char &c)
{
    return (c < ' ' || c > '~');
}


bool contains(vector<string>& v, const string& str)
{
    return find(v.begin(), v.end(), str) != v.end();
}


string get_first_word(string& line)
{
    string word;

    for(char& c : line)
        if(c != ' ' && c != '=')
            word += c;
        else
            break;

    return word;
}


string round_to_precision_string(type x, const int& precision)
{
    const type factor = type(pow(10, precision));

    const type rounded_value = (round(factor*x))/factor;

    stringstream buffer;
    buffer << fixed << setprecision(precision) << rounded_value;

    return buffer.str();
}


Tensor<string,2> round_to_precision_string_matrix(Tensor<type,2> matrix, const int& precision)
{
    Tensor<string,2> matrix_rounded(matrix.dimension(0), matrix.dimension(1));

    const type factor = type(pow(10, precision));

    for(int i = 0; i< matrix_rounded.dimension(0); i++)
    {
        for(int j = 0; j < matrix_rounded.dimension(1); j++)
        {
            const type rounded_value = (round(factor*matrix(i, j)))/factor;

            stringstream buffer;
            buffer << fixed << setprecision(precision) << rounded_value;

            matrix_rounded(i, j) = buffer.str();
        }
    }

    return matrix_rounded;
}


vector<string> sort_string_vector(vector<string>& string_vector)
{
    auto compare_string_length = [](const string& a, const string& b)
    {
        return a.length() > b.length();
    };
    
    sort(string_vector.begin(), string_vector.end(), compare_string_length);

    return string_vector;
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


void replace_substring_in_string (vector<string>& tokens, string& expression, const string& keyword)
{
    string::size_type previous_pos = 0;

    for(int i = 0; i < tokens.size(); i++)
    {
        const string found_token = tokens[i];
        const string to_replace(found_token);
        const string newword = keyword + " " + found_token;

        string::size_type position = 0;

        while((position = expression.find(to_replace, position)) != string::npos)
        {
            if(position > previous_pos)
            {
                expression.replace(position, to_replace.length(), newword);
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
        if(i < position)
            cout << "=";
        else if(i == position)
            cout << ">";
        else cout << " ";
    
    cout << "] " << int(progress * 100.0) << " %\r";

    cout.flush();
}


Index count_tokens(const vector<vector<string>>& tokens)
{
    const Index documents_number = tokens.size();

    Index count = 0;

    for(Index i = 0; i < documents_number; i++)
        count += tokens[i].size();

    return count;
}


vector<string> tokens_list(const vector<vector<string>>& document_tokens)
{
    const Index total_tokens_number = count_tokens(document_tokens);

    vector<string> total_tokens(total_tokens_number);

    Index position = 0;

    for (const auto& tokens : document_tokens)
        for (const auto& token : tokens)
            total_tokens[position++] = token;

    return total_tokens;
}


void to_lower(string& text)
{
    transform(text.begin(), text.end(), text.begin(), ::tolower);
}


void to_lower(vector<string>& documents)
{
    #pragma omp parallel for

    for(size_t i = 0; i < documents.size(); i++)
        to_lower(documents[i]);
}


void to_lower(vector<vector<string>>& text)
{
    #pragma omp parallel for

    for(size_t i = 0; i < text.size(); i++)
        to_lower(text[i]);
}


vector<vector<string>> get_tokens(const vector<string>& documents, const string& separator)
{
    const Index documents_number = documents.size();

    vector<vector<string>> tokens(documents_number);

    for(Index i = 0; i < documents_number-1; i++)
        tokens[i] = get_tokens(documents[i], separator);

    return tokens;
}


vector<pair<string, Index>> count_words(const vector<string>& words)
{
    unordered_map<string, Index> count;

    for(size_t i = 0; i < words.size(); i++)
        count[words[i]]++;

    vector<pair<string, Index>> word_counts(count.begin(), count.end());

    sort(word_counts.begin(), word_counts.end(), [](const auto& a, const auto& b)
    {
        return (a.second != b.second) 
            ? a.second > b.second 
            : a.first < b.first;
    });

    return word_counts;
}


void delete_extra_spaces(vector<string>& documents)
{
    vector<string> new_documents(documents);

    for(size_t i = 0; i < documents.size(); i++)
    {
        const string::iterator new_end = unique(new_documents[i].begin(), new_documents[i].end(),
            [](char lhs, char rhs) { return(lhs == rhs) && (lhs == ' '); });

        new_documents[i].erase(new_end, new_documents[i].end());
    }

    documents = new_documents;
}


void delete_non_printable_chars(vector<string>& documents)
{
    for(size_t i = 0; i < documents.size(); i++)
        delete_non_printable_chars(documents[i]);
}


void split_punctuation(vector<string>& documents)
{
    const vector<pair<string, string>> punctuations = {
        {"�", " � "}, {"\"", " \" "}, {".", " . "}, {"!", " ! "}, {"#", " # "},
        {"$", " $ "}, {"~", " ~ "}, {"%", " % "}, {"&", " & "}, {"/", " / "},
        {"(", " ( "}, {")", " ) "}, {"\\", " \\ "}, {"=", " = "}, {"?", " ? "},
        {"}", " } "}, {"^", " ^ "}, {"`", " ` "}, {"[", " [ "}, {"]", " ] "},
        {"*", " * "}, {"+", " + "}, {",", " , "}, {";", " ; "}, {":", " : "},
        {"-", " - "}, {">", " > "}, {"<", " < "}, {"|", " | "}, {"–", " – "},
        {"Ø", " Ø "}, {"º", " º "}, {"°", " ° "}, {"'", " ' "}, {"ç", " ç "},
        {"✓", " ✓ "}, {"@", " @ "}, {"€", " € "}, {"¬", " ¬ "}, {"•", " • "},
        {"·", " · "}, {"”", " ” "}, {"“", " “ "}, {"´", " ´ "}, {"§", " § "},
        {"_", " _ "}
    };

    for (const auto& [symbol, replacement] : punctuations)
        replace_substring(documents, symbol, replacement);
}


void delete_non_alphanumeric(vector<string>& documents)
{
    vector<string> new_documents(documents);

    for(size_t i = 0; i < documents.size(); i++)
        new_documents[i].erase(remove_if(new_documents[i].begin(), new_documents[i].end(), is_not_alnum), new_documents[i].end());

    documents = new_documents;
}


void delete_emails(vector<vector<string>>& documents)
{
    const Index documents_number = documents.size();

    #pragma omp parallel for

    for(Index i = 0; i < documents_number; i++)
    {
        const vector<string> document = documents[i];

        for(size_t j = 0; j < document.size(); j++)
        {
            /*
            vector<string> tokens = get_tokens(document(j));

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

        documents[i] = document;
    }
}


void print_tokens(const vector<vector<string>>& tokens)
{
    for(size_t i = 0; i < tokens.size(); i++)
    {
        for(size_t j = 0; j < tokens[i].size(); j++)
            cout << tokens[i][j] << " - ";

        cout << endl;
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
