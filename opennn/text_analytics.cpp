/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   T E X T   A N A L Y S I S   C L A S S                                                                      */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "text_analytics.h"

namespace OpenNN
{
// DEFAULT CONSTRUCTOR

TextAnalytics::TextAnalytics()
{
    set_english_stop_words();
}

// DESTRUCTOR

TextAnalytics::~TextAnalytics()
{
    //    documents.clear();
    //    tokenized_documents.clear();
    stop_words.clear();
}

// GET METHODS

TextAnalytics::Language TextAnalytics::get_language() const
{
    return lang;
}

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


Vector<string> TextAnalytics::get_stop_words() const
{
    return stop_words;
}

// SET METHODS

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
        clear_stop_words();
    }
}

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
        clear_stop_words();
    }
}

void TextAnalytics::set_stop_words(const Vector<string>& new_stop_words)
{
    stop_words = new_stop_words;
}


// Preprocess methods

/// Deletes consecutive extra spaces in documents

Vector<string> TextAnalytics::delete_extra_spaces(const Vector<string>& documents) const
{
    Vector<string> new_documents(documents);

    new_documents.replace_substring("\t", " ");
    new_documents.replace_substring("\n", " ");

    for(size_t i = 0; i < documents.size(); i++)
    {
        string::iterator new_end = unique(new_documents[i].begin(), new_documents[i].end(),
                                          [](char lhs, char rhs){ return(lhs == rhs) &&(lhs == ' '); });

        new_documents[i].erase(new_end, new_documents[i].end());
    }

    return new_documents;
}

/// Deletes punctuation in documents.

Vector<string> TextAnalytics::delete_punctuation(const Vector<string>& documents) const
{
    Vector<string> new_documents(documents);

    new_documents.replace_substring("!", " ");
    new_documents.replace_substring("#", " ");
    new_documents.replace_substring("$", " ");
    new_documents.replace_substring("~", " ");
    new_documents.replace_substring("%", " ");
    new_documents.replace_substring("&", " ");
    new_documents.replace_substring("¬", " ");
    new_documents.replace_substring("/", " ");
    new_documents.replace_substring("\\", " ");
    new_documents.replace_substring("(", " ");
    new_documents.replace_substring(")", " ");
    new_documents.replace_substring("=", " ");
    new_documents.replace_substring("?", " ");
    new_documents.replace_substring("¿", " ");
    new_documents.replace_substring("¡", " ");
    new_documents.replace_substring("}", " ");
    new_documents.replace_substring("^", " ");
    new_documents.replace_substring("`", " ");
    new_documents.replace_substring("[", " ");
    new_documents.replace_substring("]", " ");
    new_documents.replace_substring("*", " ");
    new_documents.replace_substring("]", " ");
    new_documents.replace_substring("+", " ");
    new_documents.replace_substring("´", " ");
    new_documents.replace_substring("{", " ");
    new_documents.replace_substring(",", " ");
    new_documents.replace_substring(";", " ");
    new_documents.replace_substring(":", " ");
    new_documents.replace_substring("-", " ");
    new_documents.replace_substring("“", " ");
    new_documents.replace_substring("”", " ");
    new_documents.replace_substring("\"", " ");
    new_documents.replace_substring(">", " ");
    new_documents.replace_substring("<", " ");
    new_documents.replace_substring("|", " ");

    // Replace two spaces by one

    for(size_t i = 0; i < documents.size(); i++)
    {
        string::iterator new_end = unique(new_documents[i].begin(), new_documents[i].end(),
                                          [](char lhs, char rhs){ return(lhs == rhs) &&(lhs == ' '); });

        new_documents[i].erase(new_end, new_documents[i].end());
    }

    return new_documents.trimmed();
}

/// Transforms all the letters of the documents into lower case.

Vector<string> TextAnalytics::to_lower(const Vector<string>& documents) const
{
    Vector<string> new_documents(documents);

    const size_t documents_number = documents.size();

    for(size_t i = 0; i < documents_number; i++)
    {
        transform(new_documents[i].begin(), new_documents[i].end(), new_documents[i].begin(), ::tolower);
    }

    return new_documents.trimmed();
}

/// Split documents into words vectors. Each word is equivalent to a token.

Vector< Vector<string> > TextAnalytics::tokenize(const Vector<string>& documents) const
{
    const size_t documents_number = documents.size();

    Vector< Vector<string> > new_tokenized_documents(documents_number);

    string empty_string;

    for(size_t i = 0; i < documents_number; i++)
    {
        new_tokenized_documents[i] = split_string(documents[i], ' ');

        new_tokenized_documents[i] = new_tokenized_documents[i].trimmed();

        new_tokenized_documents[i] = new_tokenized_documents[i].filter_not_equal_to("");

        if(new_tokenized_documents[i].contains(empty_string))
        {
            cout << "Empty string" << endl;
            system("pause");
        }
    }

    return new_tokenized_documents;
}

/// Join the words vectors into strings documents

Vector<string> TextAnalytics::detokenize(const Vector< Vector<string> >& tokens) const
{
    const size_t documents_number = tokens.size();

    Vector<string> new_documents(documents_number);

    for(size_t i = 0; i < documents_number; i++)
    {
        new_documents[i] = tokens[i].to_text(' ');
    }

    return new_documents;
}

/// Delete the words we want from the documents
/// @param delete_words Vector of words we want to delete

Vector< Vector<string> > TextAnalytics::delete_words(const Vector< Vector<string> >& tokens, const Vector<string>& delete_words) const
{
    const size_t documents_number = tokens.size();

    Vector< Vector<string> > new_documents(documents_number);

    for(size_t i = 0; i < documents_number; i++)
    {
        for(size_t j = 0; j < tokens[i].size(); j++)
        {
            if(!delete_words.contains(tokens[i][j]))
            {
                new_documents[i].push_back(tokens[i][j]);
            }
        }
//        new_documents[i] = documents[i].filter_not_equal_to(delete_words);
    }

    return new_documents;
}


Vector< Vector<string> > TextAnalytics::delete_stop_words(const Vector< Vector<string> >& tokens) const
{
    return delete_words(tokens, stop_words);
}

/// Delete short words from the documents
/// @param minimum_length Minimum length of the words that new documents must have(including herself)

Vector< Vector<string> > TextAnalytics::delete_short_words(const Vector< Vector<string> >& tokens, const size_t& minimum_length) const
{
    Vector< Vector<string> > new_documents(tokens);

    const size_t documents_number = tokens.size();

    for(size_t i = 0; i < documents_number; i++)
    {
        const size_t words_number = tokens[i].size();

        Vector<size_t> indices;

        for(size_t j = 0; j < words_number; j++)
        {
            const string word = tokens[i][j];

            if(word.length() <= minimum_length)
            {
                indices.push_back(j);
            }
        }

        new_documents[i] = new_documents[i].delete_indices(indices);
    }

    return new_documents;
}

/// Delete short words from the documents
/// @param maximum_length Maximum length of the words new documents must have(including herself)

Vector< Vector<string> > TextAnalytics::delete_long_words(const Vector< Vector<string> >& tokens, const size_t& maximum_length) const
{
    Vector< Vector<string> > new_documents(tokens);

    const size_t documents_number = tokens.size();

    for(size_t i = 0; i < documents_number; i++)
    {
        const size_t words_number = tokens[i].size();

        Vector<size_t> indices;

        for(size_t j = 0; j < words_number; j++)
        {
            const string word = tokens[i][j];

            if(word.length() >= maximum_length)
            {
                indices.push_back(j);
            }
        }

        new_documents[i] = new_documents[i].delete_indices(indices);
    }

    return new_documents;
}

/// Reduces inflected(or sometimes derived) words to their word stem, base or root form.

Vector< Vector<string> > TextAnalytics::apply_stemmer(const Vector< Vector<string> >& tokens) const
{
    if(lang == ENG)
    {
        return apply_english_stemmer(tokens);
    }
    else if(lang == SPA)
    {
        return apply_spanish_stemmer(tokens);
    }

    return tokens;
}

/// Reduces inflected(or sometimes derived) words to their word stem, base or root form(english language).

Vector< Vector<string> > TextAnalytics::apply_english_stemmer(const Vector< Vector<string> >& tokens) const /// @todo
{
    const size_t documents_number = tokens.size();

    Vector< Vector<string> > new_tokenized_documents(documents_number);

    // Set vowels and suffixes

    string vowels_pointer[] = {"a", "e", "i", "o", "u", "y"};

    const Vector<string> vowels(vector<string>(vowels_pointer, vowels_pointer + sizeof(vowels_pointer) / sizeof(string) ));

    string double_consonants_pointer[] = {"bb", "dd", "ff", "gg", "mm", "nn", "pp", "rr", "tt"};

    const Vector<string> double_consonants(vector<string>(double_consonants_pointer, double_consonants_pointer + sizeof(double_consonants_pointer) / sizeof(string) ));

    string li_ending_pointer[] = {"c", "d", "e", "g", "h", "k", "m", "n", "r", "t"};

    const Vector<string> li_ending(vector<string>(li_ending_pointer, li_ending_pointer + sizeof(li_ending_pointer) / sizeof(string) ));

    string step0_suffixes_pointer[] = {"'s'", "'s", "'"};

    const Vector<string> step0_suffixes(vector<string>(step0_suffixes_pointer, step0_suffixes_pointer + sizeof(step0_suffixes_pointer) / sizeof(string) ));

    string step1a_suffixes_pointer[] = {"sses", "ied", "ies", "us", "ss", "s"};

    const Vector<string> step1a_suffixes(vector<string>(step1a_suffixes_pointer, step1a_suffixes_pointer + sizeof(step1a_suffixes_pointer) / sizeof(string) ));

    string step1b_suffixes_pointer[] = {"eedly", "ingly", "edly", "eed", "ing", "ed"};

    const Vector<string> step1b_suffixes(vector<string>(step1b_suffixes_pointer, step1b_suffixes_pointer + sizeof(step1b_suffixes_pointer) / sizeof(string) ));

    string step2_suffixes_pointer[] = {"ization", "ational", "fulness", "ousness", "iveness", "tional", "biliti", "lessli", "entli", "ation", "alism",
                                       "aliti", "ousli", "iviti", "fulli", "enci", "anci", "abli", "izer", "ator", "alli", "bli", "ogi", "li"};

    const Vector<string> step2_suffixes(vector<string>(step2_suffixes_pointer, step2_suffixes_pointer + sizeof(step2_suffixes_pointer) / sizeof(string) ));

    string step3_suffixes_pointer[] = {"ational", "tional", "alize", "icate", "iciti", "ative", "ical", "ness", "ful"};

    const Vector<string> step3_suffixes(vector<string>(step3_suffixes_pointer, step3_suffixes_pointer + sizeof(step3_suffixes_pointer) / sizeof(string) ));

    string step4_suffixes_pointer[] = {"ement", "ance", "ence", "able", "ible", "ment", "ant", "ent", "ism", "ate", "iti", "ous",
                                       "ive", "ize", "ion", "al", "er", "ic"};

    const Vector<string> step4_suffixes(vector<string>(step4_suffixes_pointer, step4_suffixes_pointer + sizeof(step4_suffixes_pointer) / sizeof(string) ));

    Matrix<string> special_words(40,2);

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

    const size_t step0_suffixes_size = step0_suffixes.size();
    const size_t step1a_suffixes_size = step1a_suffixes.size();
    const size_t step1b_suffixes_size = step1b_suffixes.size();
    const size_t step2_suffixes_size = step2_suffixes.size();
    const size_t step3_suffixes_size = step3_suffixes.size();
    const size_t step4_suffixes_size = step4_suffixes.size();

    for(size_t i = 0; i < documents_number; i++)
    {
        Vector<string> current_document_tokens = tokens[i];
        const size_t current_document_tokens_number = current_document_tokens.size();

        current_document_tokens.replace_substring("’", "'");
        current_document_tokens.replace_substring("‘", "'");
        current_document_tokens.replace_substring("‛", "'");

        new_tokenized_documents[i] = current_document_tokens;

        for(size_t j = 0; j < current_document_tokens_number; j++)
        {
            string current_word = new_tokenized_documents[i][j];

            if(special_words.get_column(0).contains(current_word))
            {
                const size_t word_index = special_words.get_column(0).get_first_index(current_word);

                new_tokenized_documents[i][j] = special_words(word_index,1);

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
                if(vowels.contains(string(1,current_word[k-1])) && current_word[k] == 'y')
                {
                    current_word[k] = 'Y';
                }
            }

            Vector<string> r1_r2(2, "");

            if(starts_with(current_word,"gener") || starts_with(current_word,"commun") || starts_with(current_word,"arsen"))
            {
                if(starts_with(current_word,"gener") || starts_with(current_word,"arsen"))
                {
                    r1_r2[0] = current_word.substr(5);
                }
                else
                {
                    r1_r2[0] = current_word.substr(6);
                }

                for(size_t k = 1; k < r1_r2[0].size(); k++)
                {
                    if(!vowels.contains(string(1,r1_r2[0][k])) && vowels.contains(string(1,r1_r2[0][k-1])))
                    {
                        r1_r2[1] = r1_r2[0].substr(i+1);
                        break;
                    }
                }
            }
            else
            {
                r1_r2 = get_r1_r2(current_word, vowels);
            }

            bool step1a_vowel_found = false;
            bool step1b_vowel_found = false;

            // Step 0

            for(size_t k = 0; k < step0_suffixes_size; k++)
            {
                const string current_suffix = step0_suffixes[k];

                if(ends_with(current_word,current_suffix))
                {
                    current_word = current_word.substr(0,current_word.length()-current_suffix.length());
                    r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-current_suffix.length());
                    r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-current_suffix.length());

                    break;
                }
            }

            // Step 1a

            for(size_t k = 0; k < step1a_suffixes_size; k++)
            {
                const string current_suffix = step1a_suffixes[k];

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
                        if((current_word.length() - current_suffix.length()) > 1)
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
                            if(vowels.contains(string(1,current_word[l])))
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

            for(size_t k = 0; k < step1b_suffixes_size; k++)
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
                            if(vowels.contains(string(1,current_word[l])))
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
                            else if((r1_r2[0] == "" && current_word.length() >= 3 && !vowels.contains(string(1,current_word[current_word.length()-1])) &&
                                     !(current_word[current_word.length()-1] == 'w' || current_word[current_word.length()-1] == 'x' || current_word[current_word.length()-1] == 'Y') &&
                                     vowels.contains(string(1,current_word[current_word.length()-2])) && !vowels.contains(string(1,current_word[current_word.length()-3]))) ||
                                   (r1_r2[0] == "" && current_word.length() == 2 && vowels.contains(string(1,current_word[0])) && vowels.contains(string(1,current_word[1]))))
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
               !vowels.contains(string(1,current_word[current_word.length()-2])))
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

            for(size_t k = 0; k < step2_suffixes_size; k++)
            {
                const string current_suffix = step2_suffixes[k];

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
                    else if(current_suffix == "li" && li_ending.contains(string(1,current_word[current_word.length()-4])))
                    {
                        current_word = current_word.substr(0,current_word.length()-2);
                        r1_r2[0] = r1_r2[0].substr(0,r1_r2[0].length()-2);
                        r1_r2[1] = r1_r2[1].substr(0,r1_r2[1].length()-2);
                    }

                    break;
                }
            }

            // Step 3

            for(size_t k = 0; k < step3_suffixes_size; k++)
            {
                const string current_suffix = step3_suffixes[k];

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

            for(size_t k = 0; k < step4_suffixes_size; k++)
            {
                const string current_suffix = step4_suffixes[k];

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
                if(current_word.length() >= 4 &&(vowels.contains(string(1,current_word[current_word.length()-2])) ||
                                                 (current_word[current_word.length()-2] == 'w' || current_word[current_word.length()-2] == 'x' ||
                                                   current_word[current_word.length()-2] == 'Y') || !vowels.contains(string(1,current_word[current_word.length()-3])) ||
                                                   vowels.contains(string(1,current_word[current_word.length()-4]))))
                {
                    current_word = current_word.substr(0,current_word.length()-1);
                }
            }

            replace_substring<string>(current_word,"Y","y");

            new_tokenized_documents[i][j] = current_word;
        }
    }

    return new_tokenized_documents;
}

/// Reduces inflected(or sometimes derived) words to their word stem, base or root form(spanish language).

Vector< Vector<string> > TextAnalytics::apply_spanish_stemmer(const Vector< Vector<string> >& tokens) const
{
    const size_t documents_number = tokens.size();

    Vector< Vector<string> > new_tokenized_documents(documents_number);

    // Set vowels and suffixes

    string vowels_pointer[] = {"a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ü"};

    const Vector<string> vowels(vector<string>(vowels_pointer, vowels_pointer + sizeof(vowels_pointer) / sizeof(string) ));

    string step0_suffixes_pointer[] = {"selas", "selos", "sela", "selo", "las", "les", "los", "nos", "me", "se", "la", "le", "lo"};

    const Vector<string> step0_suffixes(vector<string>(step0_suffixes_pointer, step0_suffixes_pointer + sizeof(step0_suffixes_pointer) / sizeof(string) ));

    string step1_suffixes_pointer[] = {"amientos", "imientos", "amiento", "imiento", "aciones", "uciones", "adoras", "adores",
                                       "ancias", "logías", "encias", "amente", "idades", "anzas", "ismos", "ables", "ibles",
                                       "istas", "adora", "acion", "ación", "antes", "ancia", "logía", "ución", "ucion", "encia",
                                       "mente", "anza", "icos", "icas", "ion", "ismo", "able", "ible", "ista", "osos", "osas",
                                       "ador", "ante", "idad", "ivas", "ivos", "ico", "ica", "oso", "osa", "iva", "ivo", "ud"};

    const Vector<string> step1_suffixes(vector<string>(step1_suffixes_pointer, step1_suffixes_pointer + sizeof(step1_suffixes_pointer) / sizeof(string) ));

    string step2a_suffixes_pointer[] = {"yeron", "yendo", "yamos", "yais", "yan",
                                        "yen", "yas", "yes", "ya", "ye", "yo",
                                        "yó"};

    const Vector<string> step2a_suffixes(vector<string>(step2a_suffixes_pointer, step2a_suffixes_pointer + sizeof(step2a_suffixes_pointer) / sizeof(string) ));

    string step2b_suffixes_pointer[] = {"aríamos", "eríamos", "iríamos", "iéramos", "iésemos", "aríais",
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

    const Vector<string> step2b_suffixes(vector<string>(step2b_suffixes_pointer, step2b_suffixes_pointer + sizeof(step2b_suffixes_pointer) / sizeof(string) ));

    string step3_suffixes_pointer[] = {"os", "a", "e", "o", "á", "é", "í", "ó"};

    const Vector<string> step3_suffixes(vector<string>(step3_suffixes_pointer, step3_suffixes_pointer + sizeof(step3_suffixes_pointer) / sizeof(string) ));

    const size_t step0_suffixes_size = step0_suffixes.size();
    const size_t step1_suffixes_size = step1_suffixes.size();
    const size_t step2a_suffixes_size = step2a_suffixes.size();
    const size_t step2b_suffixes_size = step2b_suffixes.size();
    const size_t step3_suffixes_size = step3_suffixes.size();

    for(size_t i = 0; i < documents_number; i++)
    {
        const Vector<string> current_document_tokens = tokens[i];
        const size_t current_document_tokens_number = current_document_tokens.size();

        new_tokenized_documents[i] = current_document_tokens;

        for(size_t j = 0; j < current_document_tokens_number; j++)
        {
            string current_word = new_tokenized_documents[i][j];

            Vector<string> r1_r2 = get_r1_r2(current_word, vowels);
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

                Vector<string> presuffix(10);

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

Vector< Vector<string> > TextAnalytics::delete_numbers(const Vector< Vector<string> >& tokens) const
{
    const size_t documents_number = tokens.size();

    Vector< Vector<string> > new_words(documents_number);

    for(size_t i = 0; i < documents_number; i++)
    {
        const size_t tokens_number = tokens[i].size();

        Vector<size_t> valid_indices;

        for(size_t j = 0; j < tokens_number; j++)
        {
            if(!contains_number(tokens[i][j]))
            {
                valid_indices.push_back(j);
            }
        }

        new_words[i] = tokens[i].get_subvector(valid_indices);
    }

    return new_words;
}

/// Remove emails from documents.

Vector< Vector<string> > TextAnalytics::delete_emails(const Vector< Vector<string> >& tokens) const
{
    const size_t documents_number = tokens.size();

    Vector< Vector<string> > new_words(documents_number);

    for(size_t i = 0; i < documents_number; i++)
    {
        const size_t tokens_number = tokens[i].size();

        Vector<size_t> valid_indices;

        for(size_t j = 0; j < tokens_number; j++)
        {
            if(!is_email(tokens[i][j]))
            {
                valid_indices.push_back(j);
            }
        }

        new_words[i] = tokens[i].get_subvector(valid_indices);
    }

    return new_words;
}

/// Remove the accents of the vowels in the documents.

Vector< Vector<string> > TextAnalytics::replace_accented(const Vector< Vector<string> >& tokens) const
{
    const size_t documents_number = tokens.size();

    Vector< Vector<string> > new_words(documents_number);

    for(size_t i = 0; i < documents_number; i++)
    {
        const size_t tokens_number = tokens[i].size();

        new_words[i].set(tokens_number);

        for(size_t j = 0; j < tokens_number; j++)
        {
            new_words[i][j] = replace_accented(tokens[i][j]);
        }
    }

    return new_words;
}



Vector<string> TextAnalytics::get_r1_r2(const string& word, const Vector<string>& vowels) const
{
    const size_t word_length = word.length();

    string r1 = "";

    for(size_t i = 1; i < word_length; i++)
    {
        if(!vowels.contains(word.substr(i,1)) && vowels.contains(word.substr(i-1,1)))
        {
            r1 = word.substr(i+1);
            break;
        }
    }

    const size_t r1_length = r1.length();

    string r2 = "";

    for(size_t i = 1; i < r1_length; i++)
    {
        if(!vowels.contains(r1.substr(i,1)) && vowels.contains(r1.substr(i-1,1)))
        {
            r2 = r1.substr(i+1);
            break;
        }
    }

    Vector<string> r1_r2(2);

    r1_r2[0] = r1;
    r1_r2[1] = r2;

    return r1_r2;
}

string TextAnalytics::get_rv(const string& word, const Vector<string>& vowels) const
{
    string rv = "";

    const size_t word_lenght = word.length();

    if(word_lenght >= 2)
    {
        if(!vowels.contains(word.substr(1,1)))
        {
            for(size_t i = 2; i < word_lenght; i++)
            {
                if(vowels.contains(word.substr(i,1)))
                {
                    rv = word.substr(i+1);
                    break;
                }
            }
        }
        else if(vowels.contains(word.substr(0,1)) && vowels.contains(word.substr(1,1)))
        {
            for(size_t i = 2; i < word_lenght; i++)
            {
                if(!vowels.contains(word.substr(i,1)))
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

/// Remove the accents of the vowels of a word.

string TextAnalytics::replace_accented(const string& word) const
{
    string word_copy(word);

    replace_substring<string>(word_copy, "á", "a");
    replace_substring<string>(word_copy, "é", "e");
    replace_substring<string>(word_copy, "í", "i");
    replace_substring<string>(word_copy, "ó", "o");
    replace_substring<string>(word_copy, "ú", "u");

    return word_copy;
}

/// Calculate the total number of tokens in the documents.

size_t TextAnalytics::count(const Vector< Vector<string> >& tokens) const
{
    const size_t documents_number = tokens.size();

    size_t total_size = 0;

    for(size_t i = 0; i < documents_number; i++)
    {
        total_size += tokens[i].size();
    }

    return total_size;
}

/// Returns a vector with all the words as elements keeping the order.

Vector<string> TextAnalytics::join(const Vector< Vector<string> >& tokens) const
{
    const size_t total_size = count(tokens);

    Vector<string> total(total_size);

    const size_t documents_number = tokens.size();

    size_t index = 0;

    for(size_t i = 0; i < documents_number; i++)
    {
        total.tuck_in(index, tokens[i]);

        index += tokens[i].size();
    }

    return total;
}

/// Create a word bag that contains all the unique words of the documents,
/// their frequencies and their percentages in descending order

TextAnalytics::WordBag TextAnalytics::calculate_word_bag(const Vector< Vector<string> >& tokens) const
{
    const Vector<string> total = join(tokens);

    const Vector<size_t> count = total.count_unique();

    const Vector<size_t> rank = count.sort_descending_indices();

    const Vector<string> words = total.get_unique_elements().sort_rank(rank);
    const Vector<size_t> frequencies = count.sort_rank(rank);
    const Vector<double> percentages = frequencies.to_double_vector()*(100.0/frequencies.to_double_vector().calculate_sum());

    WordBag word_bag;
    word_bag.words = words;
    word_bag.frequencies = frequencies;
    word_bag.percentages = percentages;

    return word_bag;
}

/// Create a word bag that contains the unique words that appear a minimum number
/// of times in the documents, their frequencies and their percentages in descending order.
/// @param minimum_frequency Minimum frequency that words must have.

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_minimum_frequency(const Vector< Vector<string> >& tokens,
                                                         const size_t& minimum_frequency) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    const Vector<string> words = word_bag.words;
    const Vector<size_t> frequencies = word_bag.frequencies;
    const Vector<double> percentages = word_bag.percentages;

    const Vector<size_t> indices = frequencies.calculate_less_than_indices(minimum_frequency);

    word_bag.words = words.delete_indices(indices);
    word_bag.frequencies = frequencies.delete_indices(indices);
    word_bag.percentages = percentages.delete_indices(indices);

    return word_bag;
}

/// Create a word bag that contains the unique words that appear a minimum percentage
/// in the documents, their frequencies and their percentages in descending order.
/// @param minimum_percentage Minimum percentage of occurrence that words must have.

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_minimum_percentage(const Vector< Vector<string> >& tokens,
                                                         const double& minimum_percentage) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    const Vector<string> words = word_bag.words;
    const Vector<size_t> frequencies = word_bag.frequencies;
    const Vector<double> percentages = word_bag.percentages;

    const Vector<size_t> indices = percentages.calculate_less_than_indices(minimum_percentage);

    word_bag.words = words.delete_indices(indices);
    word_bag.frequencies = frequencies.delete_indices(indices);
    word_bag.percentages = percentages.delete_indices(indices);

    return word_bag;
}

/// Create a word bag that contains the unique words that appear a minimum ratio
/// of frequency in the documents, their frequencies and their percentages in descending order.
/// @param minimum_ratio Minimum ratio of frequency that words must have.

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_minimum_ratio(const Vector< Vector<string> >& tokens,
                                                         const double& minimum_ratio) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    const Vector<string> words = word_bag.words;
    const Vector<size_t> frequencies = word_bag.frequencies;
    const Vector<double> percentages = word_bag.percentages;

    const size_t frequencies_sum = frequencies.calculate_sum();

    const Vector<double> ratios = frequencies.to_double_vector()/(double)frequencies_sum;

    const Vector<size_t> indices = ratios.calculate_less_than_indices(minimum_ratio);

    word_bag.words = words.delete_indices(indices);
    word_bag.frequencies = frequencies.delete_indices(indices);
    word_bag.percentages = percentages.delete_indices(indices);

    return word_bag;
}

/// Create a word bag that contains the unique most frequent words whose sum
/// of frequencies is less than the specified number, their frequencies
/// and their percentages in descending order.
/// @param total_frequency Maximum cumulative frequency that words must have.

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_total_frequency(const Vector< Vector<string> >& tokens,
                                                         const size_t& total_frequency) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    const Vector<string> words = word_bag.words;
    const Vector<size_t> frequencies = word_bag.frequencies;

    const size_t index = frequencies.calculate_cumulative().calculate_cumulative_index(total_frequency);

    word_bag.words = words.get_first(index);
    word_bag.frequencies = frequencies.get_first(index);

    return word_bag;
}

/// Create a word bag that contains a maximum number of the unique most
/// frequent words, their frequencies and their percentages in descending order.
/// @param maximum_size Maximum size of words vector.

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_maximum_size(const Vector< Vector<string> >& tokens,
                                                                      const size_t& maximum_size) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    const Vector<string> words = word_bag.words;
    const Vector<size_t> frequencies = word_bag.frequencies;

    word_bag.words = words.get_first(maximum_size);
    word_bag.frequencies = frequencies.get_first(maximum_size);

    return word_bag;
}


size_t TextAnalytics::calculate_weight(const Vector<string>& document_words, const TextAnalytics::WordBag& word_bag) const
{
    size_t weight = 0;

    const Vector<string> bag_words = word_bag.words;

    const Vector<size_t> bag_frequencies = word_bag.frequencies;

    for(size_t i = 0; i < document_words.size(); i++)
    {
        for(size_t j = 0; j < word_bag.size(); j++)
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

Vector< Vector<string> > TextAnalytics::preprocess(const Vector<string>& documents) const
{
    Vector<string> documents_copy;

    documents_copy = delete_extra_spaces(documents);

    documents_copy = delete_punctuation(documents_copy);

    Vector< Vector<string> > tokenized_documents = tokenize(documents_copy);

    tokenized_documents = delete_emails(tokenized_documents);

    documents_copy = detokenize(tokenized_documents);

    documents_copy = to_lower(documents_copy);

    documents_copy.replace_substring("_", " ");
    documents_copy.replace_substring(".", " ");

    documents_copy = delete_extra_spaces(documents_copy);

    tokenized_documents = tokenize(documents_copy);

    tokenized_documents = delete_stop_words(tokenized_documents);

    tokenized_documents = delete_short_words(tokenized_documents, 3);

    tokenized_documents = delete_long_words(tokenized_documents);

    tokenized_documents = delete_numbers(tokenized_documents);

    tokenized_documents = delete_emails(tokenized_documents);

//    tokenized_documents = apply_stemmer(tokenized_documents);

//    tokenized_documents = replace_accented(tokenized_documents);

    return tokenized_documents;
}

/// Sets the words that will be removed from the documents.

void TextAnalytics::set_english_stop_words()
{
    string stop_words_pointer[] = {"i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
                                   "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
                                   "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have",
                                   "has", "had", "having", "do", "does", "did", "doing", "would", "shall", "should", "could", "ought", "i'm", "you're", "he's",
                                   "she's", "it's", "we're", "they're", "i've", "you've", "we've", "they've", "i'd", "you'd", "he'd", "she'd", "we'd", "they'd",
                                   "i'll", "you'll", "he'll", "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
                                   "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't", "can't", "cannot", "couldn't", "mustn't", "let's",
                                   "that's", "who's", "what's", "here's", "there's", "when's", "where's", "why's", "how's", "daren't ", "needn't", "oughtn't",
                                   "mightn't", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
                                   "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
                                   "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
                                   "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"};

    stop_words.set(vector<string>(stop_words_pointer, stop_words_pointer + sizeof(stop_words_pointer) / sizeof(string) ));
}

void TextAnalytics::set_spanish_stop_words()
{
    string stop_words_pointer[] = {"de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para", "con", "no", "una", "su", "al",
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
                                   "tenidos", "tenidas", "tened"};

    stop_words.set(vector<string>(stop_words_pointer, stop_words_pointer + sizeof(stop_words_pointer) / sizeof(string) ));
}

void TextAnalytics::clear_stop_words()
{
    stop_words.clear();
}


bool TextAnalytics::is_number(const string& str) const
{
    return strspn( str.c_str(), "-.0123456789" ) == str.size();
}

bool TextAnalytics::contains_number(const string& word) const
{
    return(find_if(word.begin(), word.end(), ::isdigit) != word.end());
}

bool TextAnalytics::is_email(const string& word) const
{
    // define a regular expression
    const regex pattern("(\\w+)(\\.|_)?(\\w*)@(\\w+)(\\.(\\w+))+");

    // try to match the string with the regular expression
    return regex_match(word, pattern);
}

bool TextAnalytics::starts_with(const string& word, const string& ending) const
{
    if(ending.length() > word.length() || ending.length() == 0)
    {
        return false;
    }

    return(word.substr(0,ending.length()) == ending);
}

bool TextAnalytics::ends_with(const string& word, const string& ending) const
{
    if(ending.length() > word.length())
    {
        return false;
    }

    return(word.substr(word.length() - ending.length()) == ending);
}

bool TextAnalytics::ends_with(const string& word, const Vector<string>& endings) const
{
    const size_t endings_size = endings.size();

    for(size_t i = 0; i < endings_size; i++)
    {
        if(ends_with(word,endings[i]))
        {
            return true;
        }
    }

    return false;
}

/// Returns a vector with the number of words that each document contains.

Vector<size_t> TextAnalytics::get_words_number(const Vector< Vector<string> >& tokens) const
{
    const size_t documents_number = tokens.size();

    Vector<size_t> words_number(documents_number);

    for(size_t i = 0; i < documents_number; i++)
    {
        words_number[i] = tokens[i].size();
    }

    return words_number;
}

/// Returns a vector with the number of sentences that each document contains.

Vector<size_t> TextAnalytics::get_sentences_number(const Vector<string>& documents) const
{
    const size_t documents_number = documents.size();

    Vector<size_t> sentences_number(documents_number);

    Vector< Vector<string> > documents_sentences(documents_number);

    string empty_string;

    for(size_t i = 0; i < documents_number; i++)
    {
        documents_sentences[i] = split_string(documents[i], '.');

        documents_sentences[i] = documents_sentences[i].trimmed();

        documents_sentences[i] = documents_sentences[i].filter_not_equal_to("");

        sentences_number[i] = documents_sentences[i].size();

        if(documents_sentences[i].contains(empty_string))
        {
            cout << "Empty string" << endl;
            system("pause");
        }
    }

    return sentences_number;
}

/// Returns a vector with the percentage of presence in the documents with respect to all.
/// @param words_name Vector of words from which you want to know the percentage of presence.

Vector<double> TextAnalytics::get_words_presence_percentage(const Vector< Vector<string> >& tokens, const Vector<string>& words_name) const
{
    Vector<double> word_presence_percentage(words_name.size());

    for(size_t i = 0; i < words_name.size(); i++)
    {
        size_t sum = 0;

        for(size_t j = 0; j < tokens.size(); j++)
        {
            if(tokens[j].contains(words_name[i]))
            {
                sum = sum + 1;
            }
        }

        word_presence_percentage[i] = static_cast<double>(sum)*((double)100.0/tokens.size());
    }


    return word_presence_percentage;
}

/// This function calculates the frequency of sets of consecutive words in all documents.
/// @param minimum_frequency Minimum frequency that a word must have to obtain its combinations.
/// @param combinations_length Words number of the combinations from 2.

Matrix<string> TextAnalytics::calculate_combinated_words_frequency(const Vector< Vector<string> >& tokens,
                                                                    const size_t& minimum_frequency,
                                                                    const size_t& combinations_length) const
{
    const TextAnalytics text_analytics;

    const Vector<string> words = text_analytics.join(tokens);

    const TextAnalytics::WordBag top_word_bag = text_analytics.calculate_word_bag_minimum_frequency(tokens, minimum_frequency);
    const Vector<string> words_name = top_word_bag.words;

    if(words_name.size() == 0)
    {
        cout << "There are no words with such frequency of appearance" << endl;
        exit(0);
    }

    if(combinations_length < 2)
    {
        cout << "Length of combination not valid, must be greater than 1" << endl;
        exit(0);
    }

    size_t combinated_words_size = 0;

    for(size_t i = 0; i < words_name.size(); i++)
    {
        for(size_t j = 0; j < words.size()-1; j++)
        {
            if(words_name[i] == words[j])
            {
                combinated_words_size++;
            }
        }
    }

    Vector<string> combinated_words(combinated_words_size);

    size_t index = 0;

    for(size_t i = 0; i < words_name.size(); i++)
    {
        for(size_t j = 0; j < words.size()-1; j++)
        {
            if(words_name[i] == words[j])
            {
                string word = words[j];

                for(size_t k = 1; k < combinations_length; k++)
                {
                    word += " " + words[j+k];
                }

                combinated_words[index] = word;

                index++;
            }
        }
    }

    const Vector<string> combinated_words_frequency = combinated_words.count_unique().to_string_vector();

    Matrix<string> combinated_words_frequency_matrix(combinated_words_frequency.size(),2);

    combinated_words_frequency_matrix.set_column(0,combinated_words.get_unique_elements(),"Combinated words");
    combinated_words_frequency_matrix.set_column(1,combinated_words_frequency,"Frequency");

    combinated_words_frequency_matrix = combinated_words_frequency_matrix.sort_descending_strings(1);

    return(combinated_words_frequency_matrix);
}


/// Returns the correlations of words that appear a minimum percentage of times
/// with the targets in descending order.
/// @param minimum_percentage Minimum percentage of frequency that the word must have.

Matrix<string> TextAnalytics::top_words_correlations(const Vector< Vector<string> >& tokens,
                                                     const double& minimum_percentage,
                                                     const Vector<size_t>& targets) const
{
    const TextAnalytics::WordBag top_word_bag = calculate_word_bag_minimum_percentage(tokens, minimum_percentage);
    const Vector<string> words_name = top_word_bag.words;

    if(words_name.size() == 0)
    {
        cout << "There are no words with such high percentage of appearance" << endl;
    }

    Vector<string> new_documents(tokens.size());

    for(size_t i = 0; i < tokens.size(); i++)
    {
      new_documents[i] = tokens[i].vector_to_string(';');
    }

    const Matrix<double> top_words_binary_matrix;// = new_documents.get_unique_binary_matrix(';', words_name).to_double_matrix();

    Vector<double> correlations(top_words_binary_matrix.get_columns_number());

    for(size_t i = 0; i < top_words_binary_matrix.get_columns_number(); i++)
    {
        correlations[i] = CorrelationAnalysis::calculate_linear_correlation(top_words_binary_matrix.get_column(i), targets.to_double_vector());
    }

    Matrix<string> top_words_correlations(correlations.size(),2);

    top_words_correlations.set_column(0,top_words_binary_matrix.get_header(),"Words");

    top_words_correlations.set_column(1,correlations.to_string_vector(),"Correlations");

    top_words_correlations = top_words_correlations.sort_descending_strings(1);

    return(top_words_correlations);
}

/// Create a binary matrix of the documents with the targets.

Matrix<double> TextAnalytics::calculate_data_set(const Vector<string>& documents,
                                  const Vector<string>& targets,
                                  const TextAnalytics::WordBag& word_bag) const
{
    const Vector< Vector<string> > document_words = preprocess(documents);

    const size_t documents_number = documents.size();

    const size_t word_bag_size = word_bag.size();

    const Vector<string> word_bag_words = word_bag.words;

    const Vector<size_t> words_number = get_words_number(document_words);

    Matrix<double> data_set(documents_number, word_bag_size, 0.0);

    data_set.set_header(word_bag_words);

    for(size_t i = 0; i < documents_number; i++)
    {
        for(size_t j = 0; j < words_number[i]; j++)
        {
            for(size_t k = 0; k < word_bag_size; k++)
            {
                if(document_words[i][j] == word_bag_words[k])
                {
                    data_set(i,k) += 1.0;

                    break;
                }
            }
        }
    }

    data_set = data_set.append_column(words_number.to_double_vector(), "words_number")
                       .append_column(targets.to_bool_vector().to_double_vector(), "targets");

    return data_set;
}

// BINARIZE METHODS

Vector<double> TextAnalytics::get_binary_vector(const Vector<string>& elements_to_binarize, const Vector<string>& unique_items) const
{
    const size_t unique_items_number = unique_items.size();

    Vector<double> binary_vector(unique_items_number);

    for(size_t i = 0; i < unique_items_number; i++)
    {
        if(elements_to_binarize.contains(unique_items[i]))
        {
            binary_vector[i] = 1.0;
        }
        else
        {
            binary_vector[i] = 0.0;
        }
    }

    return(binary_vector);
}


Matrix<double> TextAnalytics::get_binary_matrix(const Vector<string>& vector_to_binarize, const char& separator) const
{
    const size_t this_size = vector_to_binarize.size();

    const Vector<string> unique_mixes = vector_to_binarize.get_unique_elements();

    Vector< Vector<string> > items(unique_mixes.size());

    Vector<string> unique_items;

    for(size_t i = 0; i < unique_mixes.size(); i++)
    {
        items[i] = unique_mixes.split_element(i, separator);

        unique_items = unique_items.assemble(items[i]).get_unique_elements();
    }

    const size_t unique_items_number = unique_items.size();

    Matrix<double> binary_matrix(this_size, unique_items_number, 0.0);

    binary_matrix.set_header(unique_items);

    Vector<string> elements;

    Vector<double> binary_items(unique_items_number);

    for(size_t i = 0; i < this_size; i++)
    {
        elements = vector_to_binarize.split_element(i, separator);

        binary_items = get_binary_vector(elements,unique_items);

        binary_matrix.set_row(i, binary_items);
    }

    return(binary_matrix);
}


/// Returns a binary matrix indicating the elements of the columns.

Matrix<double> TextAnalytics::get_unique_binary_matrix(const Vector<string>& vector_to_binarize, const char& separator, const Vector<string>& unique_items) const
{
    const size_t this_size = vector_to_binarize.size();

    const size_t unique_items_number = unique_items.size();

    Matrix<double> binary_matrix(this_size, unique_items_number,0.0);

    binary_matrix.set_header(unique_items.to_string_vector());

    Vector<string> elements;

    Vector<double> binary_items(unique_items_number);

    for(size_t i = 0; i < this_size; i++)
    {
        elements = vector_to_binarize.split_element(i, separator);

        binary_items = get_binary_vector(elements,unique_items);

        binary_matrix.set_row(i, binary_items);
    }

    return(binary_matrix);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
