//   OpenNN: Open Neural Networks Library
//   www.opennn.net

//   T E X T   A N A L Y S I S   C L A S S                                 

//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com



// OpenNN includes

#include "text_analytics.h"

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
};


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

        throw invalid_argument(buffer.str());
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
    replace_substring(documents, "\\", " ");
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
    replace_substring(documents, "<","  ");
    replace_substring(documents, "|"," ");
    replace_substring(documents, "–"," ");
    replace_substring(documents, "Ø"," ");
    replace_substring(documents, "º", " ");
    replace_substring(documents, "°", " ");
    replace_substring(documents, "'", " ");
    replace_substring(documents, "ç", " ");
    replace_substring(documents, "✓", " ");
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


/// Transforms all the letters of the documents into lower case.

void TextAnalytics::to_lower(Tensor<string,1>& documents) const
{
    const size_t documents_number = documents.size();

    for(size_t i = 0; i < documents_number; i++)
    {
        transform(documents[i].begin(), documents[i].end(), documents[i].begin(), ::tolower);
    }

}


void TextAnalytics::aux_remove_non_printable_chars(Tensor<string, 1> &documents) const
{
    Tensor<string, 1> new_documents(documents);


    for(Index i = 0; i < documents.size(); i++)
    {

        new_documents[i].erase(std::remove_if(new_documents[i].begin(), new_documents[i].end(), isNotAlnum), new_documents[i].end());

    }

    documents = new_documents;
}


/// Split documents into words Tensors. Each word is equivalent to a token.
/// @param documents String tensor we will split

Tensor<Tensor<string,1>,1> TextAnalytics::tokenize(const Tensor<string,1>& documents) const
{
    const Index documents_number = documents.size();

    Tensor<Tensor<string,1>,1> new_tokenized_documents(documents_number);

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
};


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
                if( static_cast<Index>(tokens(k).length()) >= minimum_length )
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
                if( static_cast<Index>(tokens(k).length()) <= maximum_length )
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

    const Index empty_number = count_empty_values(vector);

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
};


void TextAnalytics::delete_blanks(Tensor<Tensor<string, 1>, 1>& tokens) const
{
    const Index documents_size = tokens.size();

    for(Index i = 0; i < documents_size; i++)
    {
        delete_blanks(tokens(i));
    }
};


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

/*

/// Reduces inflected(or sometimes derived) words to their word stem, base or root form(spanish language).

Tensor<Tensor<string>> TextAnalytics::apply_spanish_stemmer(const Tensor<Tensor<string>>& tokens) const
{
    const size_t documents_number = tokens.size();

    Tensor<Tensor<string>> new_tokenized_documents(documents_number);

    // Set vowels and suffixes

    string vowels_pointer[] = {"a", "e", "i", "o", "u", "á", "é", "í", "ó", "ú", "ü"};

    const Tensor<string> vowels(Tensor<string>(vowels_pointer, vowels_pointer + sizeof(vowels_pointer) / sizeof(string) ));

    string step0_suffixes_pointer[] = {"selas", "selos", "sela", "selo", "las", "les", "los", "nos", "me", "se", "la", "le", "lo"};

    const Tensor<string> step0_suffixes(Tensor<string>(step0_suffixes_pointer, step0_suffixes_pointer + sizeof(step0_suffixes_pointer) / sizeof(string) ));

    string step1_suffixes_pointer[] = {"amientos", "imientos", "amiento", "imiento", "aciones", "uciones", "adoras", "adores",
                                       "ancias", "logías", "encias", "amente", "idades", "anzas", "ismos", "ables", "ibles",
                                       "istas", "adora", "acion", "ación", "antes", "ancia", "logía", "ución", "ucion", "encia",
                                       "mente", "anza", "icos", "icas", "ion", "ismo", "able", "ible", "ista", "osos", "osas",
                                       "ador", "ante", "idad", "ivas", "ivos", "ico", "ica", "oso", "osa", "iva", "ivo", "ud"};

    const Tensor<string> step1_suffixes(Tensor<string>(step1_suffixes_pointer, step1_suffixes_pointer + sizeof(step1_suffixes_pointer) / sizeof(string) ));

    string step2a_suffixes_pointer[] = {"yeron", "yendo", "yamos", "yais", "yan",
                                        "yen", "yas", "yes", "ya", "ye", "yo",
                                        "yó"};

    const Tensor<string> step2a_suffixes(Tensor<string>(step2a_suffixes_pointer, step2a_suffixes_pointer + sizeof(step2a_suffixes_pointer) / sizeof(string) ));

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

    const Tensor<string> step2b_suffixes(Tensor<string>(step2b_suffixes_pointer, step2b_suffixes_pointer + sizeof(step2b_suffixes_pointer) / sizeof(string) ));

    string step3_suffixes_pointer[] = {"os", "a", "e", "o", "á", "é", "í", "ó"};

    const Tensor<string> step3_suffixes(Tensor<string>(step3_suffixes_pointer, step3_suffixes_pointer + sizeof(step3_suffixes_pointer) / sizeof(string) ));

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
*/

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

        const Index document_size = document.size();

        for(Index j = 0; j < document_size; j++)
        {
            Tensor<string,1> tokens = get_tokens(document(j));

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


/// Calculate the total number of tokens in the documents.

Index TextAnalytics::count(const Tensor<Tensor<string,1>,1>& documents) const
{
    const Index documents_number = documents.dimension(0);

    Index total_size = 0;

    for(Index i = 0; i < documents_number; i++)
    {
        for(Index j = 0; j < documents(i).dimension(0); j++)
        {
            total_size += count_tokens(documents(i)(j));
        }
    }

    return total_size;
}


/// Returns a Tensor with all the words as elements keeping the order.

Tensor<string,1> TextAnalytics::join(const Tensor<Tensor<string,1>,1>& documents) const
{
    const type words_number = count(documents);

    Tensor<string,1> words_list(words_number);

    Index current_tokens = 0;

    for(Index i = 0; i < documents.dimension(0); i++)
    {
        for(Index j = 0; j < documents(i).dimension(0); j++)
        {
            Tensor<string, 1> tokens = get_tokens(documents(i)(j));

            std::copy(tokens.data(), tokens.data() + tokens.size(), words_list.data() + current_tokens);

            current_tokens += tokens.size();
        }
    }

    return words_list;
}


/// Returns a string with all the text of a file
/// @param path Path of the file to be read

string TextAnalytics::read_txt_file(const string& path) const
{
    if (path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
            << "void load_documents() method.\n"
            << "Data file name is empty.\n";

        throw invalid_argument(buffer.str());
    }

    std::ifstream file(path.c_str());

    if (!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
            << "void load_documents() method.\n"
            << "Cannot open data file: " << path << "\n";

        throw invalid_argument(buffer.str());
    }

    string result="", line;

    while (file.good())
    {
        getline(file, line);
        trim(line);
        erase(line, '"');

        if (line.empty()) continue;

        result += line;

        if (file.peek() == EOF) break;
    }

    return result;
}


/// Create a word bag that contains all the unique words of the documents,
/// their frequencies and their percentages in descending order

TextAnalytics::WordBag TextAnalytics::calculate_word_bag(const Tensor<Tensor<string,1>,1>& tokens) const
{   
    const Tensor<string, 1> total = join(tokens);

    const Tensor<Index, 1> count = count_unique(total);

    const Tensor<Index, 1> descending_rank = calculate_rank_greater(count.cast<type>());

    const Tensor<string,1> words = sort_by_rank(get_unique_elements(total),descending_rank);

    const Tensor<Index,1> frequencies = sort_by_rank(count, descending_rank);

    const Tensor<Index,0> total_frequencies = frequencies.sum();

    const Tensor<double,1> percentages = ( 100/static_cast<double>(total_frequencies(0)) * frequencies.cast<double>()  );

    WordBag word_bag;
    word_bag.words = words;
    word_bag.frequencies = frequencies;
    word_bag.percentages = percentages;

    return word_bag;
}



/// Create a word bag that contains the unique words that appear a minimum number
/// of times in the documents, their frequencies and their percentages in descending order.
/// @param minimum_frequency Minimum frequency that words must have.

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_minimum_frequency(const Tensor<Tensor<string,1>,1>& tokens,
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

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_minimum_percentage(const Tensor<Tensor<string,1>,1>& tokens,
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

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_minimum_ratio(const Tensor<Tensor<string,1>,1>& tokens,
                                                                       const double& minimum_ratio) const
{
    WordBag word_bag = calculate_word_bag(tokens);

    Tensor<string,1> words = word_bag.words;
    Tensor<Index,1> frequencies = word_bag.frequencies;
    Tensor<double,1> percentages = word_bag.percentages;

    const Tensor<Index,0> frequencies_sum = frequencies.sum();

    const Tensor<double,1> ratios = frequencies.cast<double>()/static_cast<double>(frequencies_sum(0));

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

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_total_frequency(const Tensor<Tensor<string,1>,1>& tokens,
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

TextAnalytics::WordBag TextAnalytics::calculate_word_bag_maximum_size(const Tensor<Tensor<string,1>,1>& tokens,
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

Index TextAnalytics::calculate_weight(const Tensor<string, 1>& document_words, const TextAnalytics::WordBag& word_bag) const
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


/// Sets the words that will be removed from the documents.

void TextAnalytics::set_english_stop_words()
{
    stop_words.resize(180);

    stop_words.setValues({"i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he",
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

        word_presence_percentage(i) = static_cast<double>(sum)*(static_cast<double>(100.0/tokens.size()));
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

        throw invalid_argument(buffer.str());
    }

    if(combinations_length < 2)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "Tensor<string, 2> TextAnalytics::calculate_combinated_words_frequency(const Tensor<Tensor<string, 1>, 1>& tokens,"
                  "const Index& minimum_frequency,"
                  "const Index& combinations_length) const method."
               << "Length of combinations not valid, must be greater than 1";

        throw invalid_argument(buffer.str());
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

/*
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

    Tensor<double> correlations(top_words_binary_matrix.get_columns_number());

    for(size_t i = 0; i < top_words_binary_matrix.get_columns_number(); i++)
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

///@todo change loop to copy, doesnt work propperly with Tensor<Tensor<>>

void TextAnalytics::load_documents(const string& path)
{    
    const Index original_size = documents.size();

    if(path.empty())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "void load_documents() method.\n"
               << "Data file name is empty.\n";

        throw invalid_argument(buffer.str());
    }

    std::ifstream file(path.c_str());

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextAnalytics class.\n"
               << "void load_documents() method.\n"
               << "Cannot open data file: " << path << "\n";

        throw invalid_argument(buffer.str());
    }

    Tensor<Tensor<string,1>, 1> documents_copy(documents);

    documents.resize(original_size + 1);

    Tensor<Tensor<string,1>, 1> targets_copy(targets);

    targets.resize(original_size + 1);

    for(Index i = 0; i < original_size; i++)
    {
        documents(i) = documents_copy(i);
        targets(i) = targets_copy(i);
    };

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

    std::ifstream file2(path.c_str());

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

                throw invalid_argument(buffer.str());
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

    copy(document.data(), document.data() + lines_count, document_copy.data());
    copy(document_target.data(), document_target.data() + lines_count, document_target_copy.data());

    documents(original_size) = document_copy;
    targets(original_size) = document_target_copy;

    file2.close();
}


/// @todo Explain.

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
    cout << "Alphabet characters number: " << get_alphabet_length() << endl;

    cout << "Alphabet characters:\n" << alphabet << endl;

    cout << "Data tensor:\n" << data_tensor << endl;
}


void TextGenerationAlphabet::create_alphabet()
{
    string text_copy = text;

    sort(text_copy.begin(), text_copy.end());

    auto ip = std::unique(text_copy.begin(), text_copy.end());

    text_copy.resize(std::distance(text_copy.begin(), ip));

    alphabet.resize(text_copy.length());

    std::copy(text_copy.begin(), text_copy.end(), alphabet.data());
}


void TextGenerationAlphabet::encode_alphabet()
{
    const Index rows_number = text.length();

    const Index columns_number = alphabet.size();

    data_tensor.resize(rows_number, columns_number);
    data_tensor.setZero();

#pragma omp parallel for
    for (Index i = 0; i < text.length(); i++)
    {
        const int word_index = get_alphabet_index(text[i]);
        data_tensor(i, word_index) = 1;
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

    if (it != alphabet_end)
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

    result(word_index) = 1;

    return result;
}


Tensor<type, 2> TextGenerationAlphabet::multiple_one_hot_encode(const string &phrase) const
{
    const Index phrase_length = phrase.length();

    const Index alphabet_length = get_alphabet_length();

    Tensor<type, 2> result(alphabet_length, phrase_length);

    result.setZero();

    for(Index i = 0; i < phrase_length; i++)
    {
        const Index index = get_alphabet_index(phrase[i]);

        result(index, i) = 1;
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

        throw invalid_argument(buffer.str());
    }

    auto index = max_element(tensor.data(), tensor.data() + tensor.size()) - tensor.data();

    return alphabet(index);
}


string TextGenerationAlphabet::multiple_one_hot_decode(const Tensor<type, 2>& tensor) const
{
    const Index length = alphabet.size();

    if(tensor.dimension(0) != length)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: TextGenerationAlphabet class.\n"
               << "string one_hot_decode(Tensor<type, 1>& tensor).\n"
               << "Tensor length must be equal to alphabet length.";

        throw invalid_argument(buffer.str());
    }

    string result = "";

    for(Index i = 0; i < tensor.dimension(1); i++)
    {
        auto index = max_element(tensor.data() + i*tensor.dimension(0), tensor.data() + (i+1)*tensor.dimension(0)) - (tensor.data() + i*tensor.dimension(0));

        result += alphabet(index);

    }

    return result;
}


}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
