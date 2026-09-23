// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/dataset/text_dataset.h"
#include "opennn/dataset/field_parsing.h"
#include "opennn/core/log.h"
#include "opennn/core/string_utilities.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace opennn
{
namespace
{

const EnumMap<TextDataset::Task> task_names{
    {TextDataset::Task::Classification, "Classification"},
    {TextDataset::Task::SequenceToSequence, "SequenceToSequence"},
    {TextDataset::Task::NextToken, "NextToken"}};
const EnumMap<TextDataset::InputLayout> layout_names{
    {TextDataset::InputLayout::Tokens, "Tokens"},
    {TextDataset::InputLayout::TokensAndMask, "TokensAndMask"}};
const vector<string> sequence_reserved{"[PAD]", "[UNK]", "[START]", "[END]"};
const vector<string> corpus_reserved{"[PAD]", "[UNK]"};

Index maximum_length(const vector<vector<string>>& documents)
{
    const auto found = ranges::max_element(documents, {}, [](const auto& row) { return row.size(); });
    return found == documents.end() ? 0 : Index(found->size());
}

vector<string> make_column_markers(const vector<string>& columns)
{
    vector<string> markers;
    unordered_set<string> used(sequence_reserved.begin(), sequence_reserved.end());
    for (const string& column : columns)
    {
        string name;
        for (const char c : trim_view(column))
            if (c != '[' && c != ']') name += isspace(static_cast<unsigned char>(c)) ? '_' : c;
        if (name.empty()) name = "variable";
        string marker = "[" + name + "]";
        for (Index suffix = 2; used.contains(marker); ++suffix)
            marker = format("[{}_{}]", name, suffix);
        used.insert(marker);
        markers.push_back(std::move(marker));
    }
    return markers;
}

vector<Index> share_budget(span<const Index> lengths, Index budget)
{
    vector<Index> taken(lengths.size(), 0);
    while (budget > 0)
    {
        const Index open = Index(ranges::count_if(views::iota(size_t(0), lengths.size()),
            [&](size_t k) { return taken[k] < lengths[k]; }));
        if (open == 0) break;
        const Index share = max<Index>(1, budget / open);
        for (size_t k = 0; k < lengths.size() && budget > 0; ++k)
        {
            const Index added = min({share, lengths[k] - taken[k], budget});
            taken[k] += added;
            budget -= added;
        }
    }
    return taken;
}

Index framed_multi_column_length(span<const Index> lengths)
{
    return accumulate(lengths.begin(), lengths.end(), Index(2 + ssize(lengths)));
}

void fit_corpus_tokenizer(TokenizerOperator& tokenizer, const vector<string_view>& tokens,
                          Index maximum_vocabulary_size, Index minimum_token_frequency)
{
    unordered_map<string_view, size_t> counts;
#ifdef _OPENMP
    if (tokens.size() >= 10000)
    {
        vector<unordered_map<string_view, size_t>> local_counts(static_cast<size_t>(omp_get_max_threads()));
        #pragma omp parallel
        {
            auto& local = local_counts[size_t(omp_get_thread_num())];
            #pragma omp for schedule(static)
            for (Index i = 0; i < ssize(tokens); ++i) ++local[tokens[size_t(i)]];
        }
        for (const auto& local : local_counts)
            for (const auto& [token, count] : local) counts[token] += count;
    }
    else
#endif
        for (string_view token : tokens) ++counts[token];
    tokenizer.set_vocabulary(make_vocabulary(counts, tokenizer.get_reserved_tokens(),
                                              maximum_vocabulary_size, minimum_token_frequency));
}

}

TextDataset::TextDataset() : TextDataset(Options{}) {}

TextDataset::TextDataset(Options new_options) : options(new_options)
{
    tokenizer = make_unique<WordLevelTokenizer>(
        options.task == Task::NextToken ? corpus_reserved : sequence_reserved);
    if (options.task == Task::SequenceToSequence)
        target_tokenizer = make_unique<WordLevelTokenizer>();
    tokenizer_identity = format("{:016x}", tokenizer->fingerprint());
    if (target_tokenizer) target_tokenizer_identity = format("{:016x}", target_tokenizer->fingerprint());
    separator = options.task == Task::NextToken ? Separator::Space : Separator::Tab;
    storage_mode = options.input_layout == InputLayout::TokensAndMask
        ? StorageMode::Matrix : StorageMode::BinaryFile;
}

VariableRole TextDataset::token_role() const noexcept
{
    return options.input_layout == InputLayout::TokensAndMask
        ? VariableRole::Decoder : VariableRole::Input;
}

const TokenizerOperator* TextDataset::get_tokenizer(VariableRole role) const noexcept
{
    if (role == token_role() || (role == VariableRole::Target && options.task == Task::NextToken))
        return tokenizer.get();
    if (options.task == Task::SequenceToSequence && is_one_of(role, VariableRole::Decoder, VariableRole::Target))
        return target_tokenizer.get();
    return nullptr;
}

const vector<string>& TextDataset::get_vocabulary(VariableRole role) const noexcept
{
    if (role == VariableRole::Target && options.task == Task::Classification) return labels;
    const TokenizerOperator* selected = get_tokenizer(role);
    static const vector<string> empty;
    return selected ? selected->get_vocabulary() : empty;
}

vector<string> TextDataset::get_text_column_markers() const
{
    return text_columns.size() > 1 ? make_column_markers(text_columns) : vector<string>{};
}

void TextDataset::prepare_tokenizer()
{
    const vector<string> markers = get_text_column_markers();
    if (fixed_vocabulary || tokenizer->get_kind() != "WordLevel")
    {
        for (const string& marker : markers)
            throw_if(tokenizer->token_to_id(marker) == tokenizer->get_unk_id(),
                     "TextDataset: the tokenizer has no token for the text column marker {}.", marker);
        return;
    }
    const vector<string>& current = tokenizer->get_reserved_tokens();
    const bool had_markers = current.size() > sequence_reserved.size()
        && equal(sequence_reserved.begin(), sequence_reserved.end(), current.begin());
    if (markers.empty() && !had_markers) return;
    vector<string> reserved = sequence_reserved;
    reserved.insert(reserved.end(), markers.begin(), markers.end());
    if (current != reserved) tokenizer = make_unique<WordLevelTokenizer>(std::move(reserved));
}

vector<Index> TextDataset::encode_input(span<const string> tokens, span<const Index> lengths, Index length) const
{
    if (lengths.size() <= 1)
        return tokenizer->encode_sequence(vector<string>(tokens.begin(), tokens.end()), length);
    const vector<string> markers = get_text_column_markers();
    throw_if(markers.size() != lengths.size(), "TextDataset: expected {} texts, got {}.",
             markers.size(), lengths.size());
    const vector<Index> taken = share_budget(lengths, max<Index>(0, length - 2 - ssize(lengths)));
    vector<string> framed;
    framed.reserve(size_t(accumulate(taken.begin(), taken.end(), ssize(lengths))));
    Index offset = 0;
    for (size_t k = 0; k < lengths.size(); ++k)
    {
        framed.push_back(markers[k]);
        framed.insert(framed.end(), tokens.begin() + offset, tokens.begin() + offset + taken[k]);
        offset += lengths[k];
    }
    return tokenizer->encode_sequence(framed, length);
}

vector<Index> TextDataset::encode_text(span<const string> texts) const
{
    throw_if(options.task != Task::Classification || options.input_layout != InputLayout::Tokens,
             "TextDataset: encode_text supports token classification datasets only.");
    const size_t columns = max<size_t>(1, text_columns.size());
    throw_if(texts.size() != columns, "TextDataset: expected {} texts, got {}.", columns, texts.size());
    vector<string> tokens;
    vector<Index> lengths;
    for (const string& text : texts)
    {
        vector<string> column = tokenizer->tokenize(text);
        lengths.push_back(ssize(column));
        tokens.insert(tokens.end(), make_move_iterator(column.begin()), make_move_iterator(column.end()));
    }
    return encode_input(tokens, lengths, get_sequence_length(VariableRole::Input));
}

void TextDataset::set_tokenizer(unique_ptr<TokenizerOperator> replacement, VariableRole role)
{
    throw_if(!replacement, "TextDataset: tokenizer must not be null.");
    const bool primary = role == token_role()
        || (options.task == Task::NextToken && role == VariableRole::Target);
    throw_if(!primary && !(options.task == Task::SequenceToSequence
                          && is_one_of(role, VariableRole::Decoder, VariableRole::Target)),
             "TextDataset: this role does not contain tokens.");
    invalidate_data();
    cache_reader.close();
    data.resize(0, 0);
    sample_roles.clear();
    if (primary)
    {
        fixed_vocabulary = replacement->get_vocabulary_size() > 0;
        tokenizer_identity = format("{:016x}", replacement->fingerprint());
        tokenizer = std::move(replacement);
    }
    else
    {
        fixed_target_vocabulary = replacement->get_vocabulary_size() > 0;
        target_tokenizer_identity = format("{:016x}", replacement->fingerprint());
        target_tokenizer = std::move(replacement);
    }
}

void TextDataset::set_storage_mode(StorageMode new_mode)
{
    throw_if(new_mode == StorageMode::Matrix && get_samples_number() > 0 && data.size() == 0,
             "TextDataset: select Matrix storage before reading the corpus.");
    Dataset::set_storage_mode(new_mode);
}

void TextDataset::configure(Index samples, Index input_length, Index target_length)
{
    throw_if(samples < 0 || input_length <= 0 || target_length <= 0,
             "TextDataset: invalid sample count or sequence dimensions.");
    segments = {{token_role(), input_length, 0}};
    record_tokens = detail::checked_index_add(input_length, target_length, "TextDataset record");
    if (options.input_layout == InputLayout::TokensAndMask)
    {
        segments.push_back({VariableRole::Input, input_length, input_length});
        record_tokens = detail::checked_index_add(record_tokens, input_length, "TextDataset mask record");
    }
    else if (options.task == Task::SequenceToSequence)
    {
        const auto framing = target_tokenizer->encode_sequence(vector<string>{}, 2);
        throw_if(framing.size() != 2, "TextDataset: decoder tokenizer must provide start and end tokens.");
        segments.push_back({VariableRole::Decoder, target_length, input_length, framing.front()});
    }
    const Index target_offset = options.task == Task::NextToken ? 1 : record_tokens - target_length;
    if (options.task == Task::NextToken)
        record_tokens = detail::checked_index_add(input_length, 1, "TextDataset next-token record");
    segments.push_back({VariableRole::Target, target_length, target_offset});

    variables.clear();
    input_shape.clear();
    decoder_shape.clear();
    for (const Segment& segment : segments)
    {
        Variable variable;
        variable.role = segment.role;
        variable.name = segment.role == VariableRole::Target ? "target_sequence"
                      : segment.role == VariableRole::Decoder ? "decoder_sequence" : "input_sequence";
        variable.type = VariableType::Numeric;
        variable.features = segment.length;
        variable.scaler = ScalerMethod::None;
        variable.categories = get_vocabulary(segment.role);
        if (segment.role == VariableRole::Target && options.task == Task::Classification)
            variable.type = labels.size() == 2 ? VariableType::Binary : VariableType::Categorical;
        variables.push_back(std::move(variable));
        if (segment.role == VariableRole::Input) input_shape = {segment.length};
        else if (segment.role == VariableRole::Decoder) decoder_shape = {segment.length};
        else target_shape = {segment.length};
    }
    sample_roles.assign(size_t(samples), SampleRole::Training);
}

void TextDataset::read_txt(const filesystem::path& path)
{
    invalidate_data();
    cache_reader.close();
    data.resize(0, 0);
    data_path = path;
    throw_if(options.sequence_length < 0
             || (options.task == Task::NextToken && options.sequence_length == 0),
             "TextDataset: NextToken requires a positive sequence length; row-task limits cannot be negative.");
    throw_if(options.input_layout == InputLayout::TokensAndMask
             && (options.task != Task::Classification || options.sequence_length <= 1),
             "TextDataset: masked classification requires a sequence length of at least two.");
    if (options.input_layout == InputLayout::TokensAndMask)
    {
        throw_if(tokenizer->get_kind() != "WordPiece" || !fixed_vocabulary,
                 "TextDataset: masked classification requires a loaded WordPiece tokenizer.");
        throw_if(tokenizer->token_to_id("[CLS]") == tokenizer->get_unk_id()
                 || tokenizer->token_to_id("[SEP]") == tokenizer->get_unk_id()
                 || tokenizer->token_to_id("[PAD]") != 0,
                 "TextDataset: masked tokenization requires [CLS], [SEP], and [PAD] at id zero.");
    }
    const filesystem::path parent = cache_directory.empty()
        ? filesystem::path(data_path.string() + ".cache")
        : cache_directory / (data_path.filename().string() + ".cache");
    cache_path = parent / format("text_v5_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.bin",
        task_names.to_string(options.task), layout_names.to_string(options.input_layout),
        options.sequence_length, options.maximum_vocabulary_size, options.minimum_token_frequency,
        get_separator_name(), has_header, has_sample_ids,
        tokenizer_identity, target_tokenizer_identity);
    const filesystem::path metadata_path = cache_path.string() + ".json";
    if (uses_cache() && is_file_current(cache_path, {data_path})
        && is_file_current(metadata_path, {data_path}) && load_cache(metadata_path))
    {
        split_samples_random();
        return;
    }
    logging::info() << "Reading text dataset...\n";
    if (options.task == Task::NextToken) read_corpus();
    else read_rows();
    if (uses_cache()) save_cache(metadata_path);
    split_samples_random();
}

namespace
{

struct RecordLayout
{
    size_t first = 0;
    size_t id_fields = 0;
    size_t text_fields = 0;
    size_t fields_number = 0;
    char separator = '\t';
    bool has_quotes = false;
    vector<string> text_columns;
};

RecordLayout scan_record_layout(const CsvReader::Result& source, char separator,
                                bool has_header, bool has_ids)
{
    RecordLayout layout;
    layout.separator = separator;
    layout.has_quotes = source.has_quotes;
    layout.first = has_header && !source.lines.empty() ? 1 : 0;
    layout.id_fields = has_ids ? 1 : 0;
    layout.fields_number = 2;

    const Index rows = Index(source.lines.size() - layout.first);

    vector<string_view> header;
    string header_scratch;
    if (layout.first)
        get_token_views_maybe_quoted(source.lines[0], separator, source.has_quotes, header_scratch, header);

    if (!header.empty())
        layout.fields_number = header.size();
    else if (rows > 0)
    {
        vector<size_t> counts(size_t(rows), 0);
        #pragma omp parallel if(rows >= 256)
        {
            string scratch;
            vector<string_view> fields;
            #pragma omp for schedule(static)
            for (Index row = 0; row < rows; ++row)
            {
                get_token_views_maybe_quoted(source.lines[layout.first + size_t(row)], separator,
                                             source.has_quotes, scratch, fields);
                counts[size_t(row)] = fields.size();
            }
        }
        map<size_t, Index> frequencies;
        for (const size_t count : counts) ++frequencies[count];
        layout.fields_number = ranges::max_element(frequencies, {}, &pair<const size_t, Index>::second)->first;
    }

    if (layout.fields_number < layout.id_fields + 2) return layout;

    layout.text_fields = layout.fields_number - layout.id_fields - 1;
    layout.text_columns.resize(layout.text_fields);
    for (size_t k = 0; k < layout.text_fields; ++k)
    {
        const size_t column = layout.id_fields + k;
        const string_view name = column < header.size() ? trim_view(header[column]) : string_view{};
        layout.text_columns[k] = name.empty() ? format("variable_{}", column + 1) : string(name);
    }
    return layout;
}

bool split_record(string_view line, const RecordLayout& layout, string& scratch, vector<string_view>& fields)
{
    get_token_views_maybe_quoted(line, layout.separator, layout.has_quotes, scratch, fields);
    return fields.size() == layout.fields_number;
}

RecordLayout masked_record_layout(const CsvReader::Result& source, bool has_header)
{
    RecordLayout layout;
    layout.first = has_header && !source.lines.empty() ? 1 : 0;
    layout.text_fields = 1;
    layout.fields_number = 2;
    layout.text_columns = {"variable_1"};
    return layout;
}

}

void TextDataset::load_documents(Documents& documents) const
{
    const CsvReader::Result source = CsvReader().read(data_path);
    const bool masked = options.input_layout == InputLayout::TokensAndMask;
    const string delimiter = get_separator_string();
    const char separator_char = delimiter.empty() ? '\t' : delimiter[0];

    const RecordLayout layout = masked
        ? masked_record_layout(source, has_header)
        : scan_record_layout(source, separator_char, has_header, has_sample_ids);

    throw_if(layout.text_fields == 0, "TextDataset: each line needs {}text and target fields.",
             layout.id_fields ? "an identifier, " : "");
    throw_if(layout.text_fields > 1 && options.task != Task::Classification,
             "TextDataset: several text columns are only supported for classification.");

    const size_t first = layout.first;
    const size_t id_fields = layout.id_fields;
    const size_t text_fields = layout.text_fields;
    const Index rows = Index(source.lines.size() - first);

    documents.text_columns = layout.text_columns;

    documents.input.assign(size_t(rows), {});
    documents.target.assign(size_t(rows), {});
    documents.input_lengths.assign(text_fields > 1 ? size_t(rows) : 0, {});
    documents.ids.assign(id_fields ? size_t(rows) : 0, {});
    vector<uint8_t> valid(size_t(rows), 1);
    #pragma omp parallel if(rows >= 256)
    {
        string scratch;
        vector<string_view> fields;
        #pragma omp for schedule(static)
        for (Index row = 0; row < rows; ++row)
        {
            const string_view line = source.lines[first + size_t(row)];
            if (masked)
            {
                const size_t tab = line.rfind('\t');
                if (tab == string_view::npos) { valid[size_t(row)] = 0; continue; }
                fields = {trim_view(line.substr(0, tab)), trim_view(line.substr(tab + 1))};
                if (fields[0].empty() || fields[1].empty()) { valid[size_t(row)] = 0; continue; }
            }
            else if (!split_record(line, layout, scratch, fields))
            {
                valid[size_t(row)] = 0;
                continue;
            }
            if (id_fields) documents.ids[size_t(row)] = string(trim_view(fields[0]));
            vector<string>& input = documents.input[size_t(row)];
            if (text_fields == 1)
                input = tokenizer->tokenize(fields[id_fields]);
            else
                for (size_t k = 0; k < text_fields; ++k)
                {
                    vector<string> column = tokenizer->tokenize(fields[id_fields + k]);
                    documents.input_lengths[size_t(row)].push_back(ssize(column));
                    input.insert(input.end(), make_move_iterator(column.begin()), make_move_iterator(column.end()));
                }
            const string_view target = fields.back();
            if (options.task == Task::Classification)
                documents.target[size_t(row)] = {masked ? string(target) : ascii_lowercase(trim_view(target))};
            else
                documents.target[size_t(row)] = target_tokenizer->tokenize(target);
        }
    }
    const auto invalid = ranges::find(valid, uint8_t(0));
    throw_if(!masked && invalid != valid.end(), "Line {} must contain exactly {} fields: {}{} and target.",
             first + size_t(distance(valid.begin(), invalid)) + 1, layout.fields_number,
             id_fields ? "identifier, " : "", text_fields == 1 ? "input" : format("{} text inputs", text_fields));
    size_t retained = 0;
    for (size_t row = 0; row < valid.size(); ++row)
        if (valid[row])
        {
            if (retained != row)
            {
                documents.input[retained] = std::move(documents.input[row]);
                documents.target[retained] = std::move(documents.target[row]);
                if (!documents.input_lengths.empty())
                    documents.input_lengths[retained] = std::move(documents.input_lengths[row]);
                if (!documents.ids.empty()) documents.ids[retained] = std::move(documents.ids[row]);
            }
            ++retained;
        }
    documents.input.resize(retained);
    documents.target.resize(retained);
    if (!documents.input_lengths.empty()) documents.input_lengths.resize(retained);
    if (!documents.ids.empty()) documents.ids.resize(retained);
}

void TextDataset::for_each_record(const function<bool(Index, const Record&)>& visit) const
{
    if (data_path.empty() || !filesystem::exists(data_path)) return;

    const CsvReader::Result source = CsvReader().read(data_path);
    const string delimiter = get_separator_string();
    const char separator_char = delimiter.empty() ? '\t' : delimiter[0];
    const RecordLayout layout = scan_record_layout(source, separator_char, has_header, has_sample_ids);

    if (layout.text_fields == 0) return;

    string scratch;
    vector<string_view> fields;
    Record record;

    for (size_t line = layout.first; line < source.lines.size(); ++line)
    {
        record.id.clear();
        record.texts.clear();
        record.target.clear();

        if (split_record(source.lines[line], layout, scratch, fields))
        {
            if (layout.id_fields) record.id = string(trim_view(fields[0]));
            for (size_t k = 0; k < layout.text_fields; ++k)
                record.texts.emplace_back(trim_view(fields[layout.id_fields + k]));
            record.target = string(trim_view(fields.back()));
        }
        else
            record.texts.emplace_back(trim_view(source.lines[line]));

        if (!visit(Index(line - layout.first), record)) return;
    }
}

optional<TextDataset::Record> TextDataset::split_line(string_view line, bool with_id) const
{
    const size_t text_fields = max<size_t>(1, text_columns.size());
    const size_t id_fields = with_id && has_sample_ids ? 1 : 0;

    Record record;

    if (text_fields == 1 && id_fields == 0)
    {
        record.texts.emplace_back(trim_view(line));
        return record;
    }

    const string delimiter = get_separator_string();

    string scratch;
    vector<string_view> fields;
    get_token_views_maybe_quoted(line, delimiter.empty() ? '\t' : delimiter[0],
                                 line.find('"') != string_view::npos, scratch, fields);

    if (fields.size() != id_fields + text_fields) return nullopt;

    if (id_fields) record.id = string(trim_view(fields[0]));
    for (size_t k = 0; k < text_fields; ++k)
        record.texts.emplace_back(trim_view(fields[id_fields + k]));

    return record;
}

vector<TextDataset::Record> TextDataset::read_records() const
{
    vector<Record> records;
    for_each_record([&](Index, const Record& record)
    {
        records.push_back(record);
        return true;
    });
    return records;
}

void TextDataset::build_labels(const vector<vector<string>>& targets)
{
    unordered_map<string_view, size_t> counts;
    for (const auto& target : targets)
    {
        throw_if(target.empty() || target[0].empty(), "TextDataset: empty target label.");
        ++counts[target[0]];
    }
    labels.clear();
    if (options.input_layout == InputLayout::TokensAndMask)
    {
        for (const auto& [label, count] : counts) labels.emplace_back(label);
        ranges::sort(labels);
    }
    else
    {
        labels = make_vocabulary(counts, sequence_reserved, options.maximum_vocabulary_size,
                                 options.minimum_token_frequency);
        labels.erase(labels.begin(), labels.begin() + Index(sequence_reserved.size()));
        if (labels.size() == 2
            && (contains(positive_words, labels[0]) || contains(negative_words, labels[1])))
            swap(labels[0], labels[1]);
    }
    throw_if(labels.empty(), "TextDataset: no target classes found.");
}

void TextDataset::read_rows()
{
    Documents documents;
    load_documents(documents);
    const vector<vector<string>>& input_documents = documents.input;
    const vector<vector<string>>& target_documents = documents.target;
    throw_if(input_documents.empty(), "TextDataset: no text rows found.");
    text_columns = std::move(documents.text_columns);
    sample_ids = std::move(documents.ids);
    prepare_tokenizer();
    if (!fixed_vocabulary)
        tokenizer->build_vocabulary(input_documents, options.maximum_vocabulary_size, options.minimum_token_frequency);
    const vector<vector<Index>>& input_lengths = documents.input_lengths;
    Index input_length = 0;
    if (input_lengths.empty())
        input_length = detail::checked_index_add(maximum_length(input_documents), 2, "TextDataset input length");
    else
        for (const vector<Index>& lengths : input_lengths)
            input_length = max(input_length, framed_multi_column_length(lengths));
    const Index minimum_length = input_lengths.empty() ? 2 : 2 + 2 * ssize(text_columns);
    throw_if(!input_lengths.empty() && options.sequence_length > 0 && options.sequence_length < minimum_length,
             "TextDataset: a sequence length of at least {} is needed for {} text columns.",
             minimum_length, text_columns.size());
    if (options.sequence_length > 0)
        input_length = options.input_layout == InputLayout::TokensAndMask
            ? options.sequence_length : min(input_length, options.sequence_length);

    Index target_length = 0;
    unordered_map<string_view, Index> label_indices;
    if (options.task == Task::Classification)
    {
        build_labels(target_documents);
        target_length = labels.size() == 2 ? 1 : Index(labels.size());
        for (Index i = 0; i < ssize(labels); ++i) label_indices.emplace(labels[size_t(i)], i);
    }
    else
    {
        labels.clear();
        if (!fixed_target_vocabulary)
            target_tokenizer->build_vocabulary(target_documents, options.maximum_vocabulary_size, options.minimum_token_frequency);
        target_length = detail::checked_index_add(maximum_length(target_documents), 1, "TextDataset target length");
    }
    const Index samples = ssize(input_documents);
    configure(samples, input_length, target_length);
    vector<vector<Index>> inputs(static_cast<size_t>(samples));
    vector<vector<Index>> targets(static_cast<size_t>(samples));
    #pragma omp parallel for
    for (Index sample = 0; sample < samples; ++sample)
        inputs[size_t(sample)] = input_lengths.empty()
            ? tokenizer->encode_sequence(input_documents[size_t(sample)], input_length)
            : encode_input(input_documents[size_t(sample)], input_lengths[size_t(sample)], input_length);

    if (options.task == Task::SequenceToSequence)
    {
        #pragma omp parallel for
        for (Index sample = 0; sample < samples; ++sample)
        {
            auto framed = target_tokenizer->encode_sequence(target_documents[size_t(sample)],
                detail::checked_index_add(target_length, 1, "TextDataset framed target"));
            targets[size_t(sample)].assign(framed.begin() + 1, framed.end());
        }
    }
    else
    {
        for (Index sample = 0; sample < samples; ++sample)
        {
            const string& label = target_documents[size_t(sample)][0];
            const auto found = label_indices.find(label);
            throw_if(found == label_indices.end(), "TextDataset: unknown target label {}.", label);
            targets[size_t(sample)].assign(size_t(target_length), 0);
            if (labels.size() != 2)
                targets[size_t(sample)][size_t(found->second)] = 1;
            else if (options.input_layout == InputLayout::TokensAndMask)
                targets[size_t(sample)][0] = contains(positive_words, label)
                    || (!contains(negative_words, label) && found->second == 1) ? 1 : 0;
            else
                targets[size_t(sample)][0] = found->second;
        }
    }
    write_records(samples, [&](Index sample, span<int32_t> record)
    {
        const auto& input = inputs[size_t(sample)];
        ranges::transform(input, record.begin(), [](Index token) { return int32_t(token); });
        const auto& target = targets[size_t(sample)];
        ranges::transform(target, record.begin() + (record_tokens - target_length),
                           [](Index token) { return int32_t(token); });
        if (options.input_layout == InputLayout::TokensAndMask)
            fill_n(record.begin() + input_length, input.size(), int32_t(1));
    });
}

void TextDataset::read_corpus()
{
    string corpus = read_text_file(data_path);
    vector<Index> ids;
    if (fixed_vocabulary)
        ids = tokenizer->encode(corpus);
    else
    {
        ascii_lowercase_in_place(corpus);
        const vector<string_view> tokens = tokenize_views(corpus);
        fit_corpus_tokenizer(*tokenizer, tokens, options.maximum_vocabulary_size, options.minimum_token_frequency);
        ids.resize(tokens.size());
        #pragma omp parallel for
        for (Index i = 0; i < ssize(tokens); ++i)
            ids[size_t(i)] = tokenizer->token_to_id(tokens[size_t(i)]);
    }
    const Index block_size = detail::checked_index_add(options.sequence_length, 1, "TextDataset corpus block");
    const Index samples = ssize(ids) / block_size;
    throw_if(samples == 0, "TextDataset: corpus has {} tokens; at least {} are required.", ids.size(), block_size);
    labels.clear();
    text_columns.clear();
    sample_ids.clear();
    configure(samples, options.sequence_length, options.sequence_length);
    write_records(samples, [&](Index sample, span<int32_t> record)
    {
        for (Index i = 0; i < block_size; ++i)
            record[size_t(i)] = int32_t(ids[size_t(sample * block_size + i)]);
    });
}

bool TextDataset::uses_cache() const noexcept
{
    return storage_mode != StorageMode::Matrix || options.input_layout == InputLayout::TokensAndMask;
}

void TextDataset::store_matrix_record(Index sample, span<const int32_t> record)
{
    Index column = 0;
    for (const Segment& segment : segments)
    {
        const Index shift = segment.prefix >= 0 ? 1 : 0;
        if (shift) data(sample, column) = float(segment.prefix);
        for (Index i = shift; i < segment.length; ++i)
            data(sample, column + i) = float(record[size_t(segment.offset + i - shift)]);
        column += segment.length;
    }
}

void TextDataset::write_records(Index samples, const function<void(Index, span<int32_t>)>& fill_record)
{
    FileWriter writer;
    if (storage_mode == StorageMode::Matrix)
        data.resize(samples, get_features_number());
    if (uses_cache())
    {
        filesystem::create_directories(cache_path.parent_path());
        writer.open(cache_path.string() + ".tmp");
    }
    vector<int32_t> record(size_t(record_tokens), 0);
    for (Index sample = 0; sample < samples; ++sample)
    {
        ranges::fill(record, 0);
        fill_record(sample, record);
        if (uses_cache()) writer.write(span(record));
        if (storage_mode == StorageMode::Matrix) store_matrix_record(sample, record);
    }
    if (uses_cache())
    {
        writer.finish_with_rename(cache_path);
        cache_reader.open(cache_path);
    }
}

void TextDataset::fill_sequences(VariableRole role, const vector<Index>& samples,
                                const vector<Index>& features, float* destination,
                                ColumnContiguity contiguity) const
{
    const auto found = ranges::find(segments, role, &Segment::role);
    if (found == segments.end())
    {
        throw_if(!features.empty(), "TextDataset: this role has no features.");
        return;
    }
    const Segment& segment = *found;
    const Index width = Index(features.size());
    const Index values = detail::checked_index_multiply(Index(samples.size()), width, "TextDataset batch");
    throw_if(values > 0 && !destination, "TextDataset: output buffer is null.");
    const span<float> output(destination, size_t(values));
    Index first_feature = 0;
    for (auto current = segments.begin(); current != found; ++current)
        first_feature += current->length;
    for (Index feature : features)
        throw_if(feature < first_feature || feature - first_feature >= segment.length,
                 "TextDataset: feature does not belong to the selected role.");
    if (width == 0) return;
    for (Index sample : samples)
        throw_if(sample < 0 || sample >= get_samples_number(), "TextDataset: sample index is out of range.");
    if (storage_mode == StorageMode::Matrix)
        return fill_tensor_data(data, samples, features, output, contiguity);
    const Index shift = segment.prefix >= 0 ? 1 : 0;
    if (width == segment.length && features.front() == first_feature && is_contiguous(features))
    {
        if (shift)
            for (Index i = 0; i < ssize(samples); ++i)
                output[size_t(i * width)] = float(segment.prefix);
        read_int32_batch(cache_reader, samples, get_samples_number(), uint64_t(record_tokens),
                         segment.offset, segment.length - shift, output, width, shift, "TextDataset");
        return;
    }
    for (Index column = 0; column < width; ++column)
    {
        const Index local = features[size_t(column)] - first_feature;
        if (shift && local == 0)
            for (Index row = 0; row < ssize(samples); ++row)
                output[size_t(row * width + column)] = float(segment.prefix);
        else
            read_int32_batch(cache_reader, samples, get_samples_number(), uint64_t(record_tokens),
                segment.offset + local - shift, 1, output, width, column, "TextDataset feature selection");
    }
}

void TextDataset::fill_inputs(const vector<Index>& samples, const vector<Index>& features,
                             float* output, FillMode, ColumnContiguity contiguity) const
{
    fill_sequences(VariableRole::Input, samples, features, output, contiguity);
}

void TextDataset::fill_targets(const vector<Index>& samples, const vector<Index>& features,
                              float* output, FillMode, ColumnContiguity contiguity) const
{
    fill_sequences(VariableRole::Target, samples, features, output, contiguity);
}

void TextDataset::fill_decoder(const vector<Index>& samples, const vector<Index>& features,
                              float* output, FillMode, ColumnContiguity contiguity) const
{
    fill_sequences(VariableRole::Decoder, samples, features, output, contiguity);
}

VectorI TextDataset::calculate_target_distribution() const
{
    if (options.task != Task::Classification) return {};
    VectorI distribution = VectorI::Zero(Index(labels.size()));
    vector<float> target(size_t(get_sequence_length(VariableRole::Target)));
    const vector<Index> features = get_feature_indices(VariableRole::Target);
    for (Index sample = 0; sample < get_samples_number(); ++sample)
    {
        fill_targets({sample}, features, target.data(), FillMode::Inference);
        if (target.size() == 1 && labels.size() == 2)
            ++distribution(target[0] < 1.0f ? 0 : 1);
        else
            for (Index i = 0; i < ssize(target); ++i)
                if (target[size_t(i)] == 1.0f) { ++distribution(i); break; }
    }
    return distribution;
}

void TextDataset::write_configuration(JsonWriter& writer) const
{
    write_json(writer, {
        {"Task", task_names.to_string(options.task)},
        {"InputLayout", layout_names.to_string(options.input_layout)},
        {"SequenceLength", options.sequence_length},
        {"MaximumVocabularySize", options.maximum_vocabulary_size},
        {"MinimumTokenFrequency", options.minimum_token_frequency},
        {"InputSequenceLength", get_sequence_length(token_role())},
        {"TargetSequenceLength", get_sequence_length(VariableRole::Target)},
        {"Labels", json_array(labels)},
        {"TextColumns", json_array(text_columns)}
    });
    const auto write_tokenizer = [&](const char* name, const TokenizerOperator* value, bool fixed, const string& identity)
    {
        if (!value) return;
        writer.open_element(name);
        write_json(writer, {{"Kind", value->get_kind()}, {"FixedVocabulary", fixed}, {"Identity", identity}});
        value->to_JSON(writer);
        writer.close_element();
    };
    write_tokenizer("InputTokenizer", tokenizer.get(), fixed_vocabulary, tokenizer_identity);
    write_tokenizer("TargetTokenizer", target_tokenizer.get(), fixed_target_vocabulary, target_tokenizer_identity);
}

void TextDataset::read_configuration(const Json* root)
{
    options.task = task_names.from_string(read_json_string(root, "Task"));
    options.input_layout = layout_names.from_string(read_json_string(root, "InputLayout"));
    options.sequence_length = read_json_index(root, "SequenceLength");
    options.maximum_vocabulary_size = read_json_index(root, "MaximumVocabularySize");
    options.minimum_token_frequency = read_json_index(root, "MinimumTokenFrequency");
    labels = read_json_strings(root, "Labels");
    text_columns = root->has("TextColumns") ? read_json_strings(root, "TextColumns") : vector<string>{};
    const auto read_tokenizer = [&](const char* name, unique_ptr<TokenizerOperator>& value, bool& fixed, string& identity)
    {
        value.reset();
        fixed = false;
        identity.clear();
        if (const Json* element = root->find(name))
        {
            value = make_tokenizer_operator(read_json_string(element, "Kind"));
            value->from_JSON(element);
            fixed = read_json_bool(element, "FixedVocabulary");
            identity = read_json_string(element, "Identity");
        }
    };
    read_tokenizer("InputTokenizer", tokenizer, fixed_vocabulary, tokenizer_identity);
    read_tokenizer("TargetTokenizer", target_tokenizer, fixed_target_vocabulary, target_tokenizer_identity);
    throw_if(!tokenizer || (options.task == Task::SequenceToSequence && !target_tokenizer),
             "TextDataset: serialized tokenizer configuration is incomplete.");
}

bool TextDataset::load_cache(const filesystem::path& metadata_path)
{
    try
    {
        const JsonDocument document = load_json_file(metadata_path);
        const Json* root = get_json_root(document, "TextCache");
        if (read_json_index(root, "Version") != 2
            || read_json_string(root, "Key") != cache_path.filename().string()) return false;
        TextDataset cached;
        cached.read_configuration(root);
        if (cached.options.task != options.task || cached.options.input_layout != options.input_layout
            || cached.options.sequence_length != options.sequence_length
            || cached.options.maximum_vocabulary_size != options.maximum_vocabulary_size
            || cached.options.minimum_token_frequency != options.minimum_token_frequency
            || cached.fixed_vocabulary != fixed_vocabulary
            || cached.fixed_target_vocabulary != fixed_target_vocabulary
            || cached.tokenizer_identity != tokenizer_identity
            || cached.target_tokenizer_identity != target_tokenizer_identity
            || (fixed_vocabulary && cached.tokenizer->fingerprint() != tokenizer->fingerprint())
            || (fixed_target_vocabulary && cached.target_tokenizer->fingerprint() != target_tokenizer->fingerprint()))
            return false;
        const Index samples = read_json_index(root, "SamplesNumber");
        if (samples <= 0
            || (has_sample_ids && ssize(read_json_strings(root, "SampleIds")) != samples)) return false;
        cached.configure(samples, read_json_index(root, "InputSequenceLength"),
                          read_json_index(root, "TargetSequenceLength"));
        const Index bytes = detail::checked_index_multiply(
            detail::checked_index_multiply(samples, cached.record_tokens, "TextDataset cache"),
            Index(sizeof(int32_t)), "TextDataset cache bytes");
        error_code error;
        if (filesystem::file_size(cache_path, error) != uintmax_t(bytes) || error) return false;
        tokenizer = std::move(cached.tokenizer);
        target_tokenizer = std::move(cached.target_tokenizer);
        labels = std::move(cached.labels);
        text_columns = std::move(cached.text_columns);
        sample_ids = read_json_strings(root, "SampleIds");
        configure(samples, cached.get_sequence_length(cached.token_role()),
                   cached.get_sequence_length(VariableRole::Target));
        cache_reader.open(cache_path);
        if (storage_mode == StorageMode::Matrix)
        {
            data.resize(samples, get_features_number());
            vector<int32_t> record(size_t(record_tokens), 0);
            for (Index sample = 0; sample < samples; ++sample)
            {
                cache_reader.read_at(span(record), uint64_t(sample) * uint64_t(record_tokens) * sizeof(int32_t));
                store_matrix_record(sample, record);
            }
        }
        return true;
    }
    catch (const runtime_error&)
    {
        return false;
    }
}

void TextDataset::save_cache(const filesystem::path& metadata_path) const
{
    JsonWriter metadata;
    metadata.open_element("TextCache");
    write_json(metadata, {{"Version", 2}, {"Key", cache_path.filename().string()},
                          {"SamplesNumber", get_samples_number()}, {"SampleIds", json_array(sample_ids)}});
    write_configuration(metadata);
    metadata.close_element();
    FileWriter writer;
    writer.open(metadata_path.string() + ".tmp");
    const string contents = metadata.c_str();
    writer.write(span(contents));
    writer.finish_with_rename(metadata_path);
}

void TextDataset::to_JSON(JsonWriter& writer) const
{
    write_json_header(writer, {
        {"FileType", "txt"}, {"Path", data_path.string()},
        {"Separator", get_separator_name()}, {"HasHeader", has_header},
        {"HasSamplesId", has_sample_ids}, {"Codification", get_codification_string()},
        {"StorageMode", get_storage_mode_string()}
    });
    preview_data_to_JSON(writer);
    write_configuration(writer);
    write_json_footer(writer);
}

void TextDataset::from_JSON(const JsonDocument& document)
{
    invalidate_data();
    cache_reader.close();
    data.resize(0, 0);
    sample_roles.clear();
    const Json* root = get_json_root(document, "Dataset");
    read_configuration(root);
    const Json* source = require_json_field(root, "DataSource");
    data_path = read_json_string(source, "Path");
    set_separator_name(read_json_string(source, "Separator"));
    set_codification(read_json_string(source, "Codification"));
    set_storage_mode(read_json_string(source, "StorageMode", "BinaryFile"));
    has_header = read_json_bool(source, "HasHeader");
    has_sample_ids = read_json_bool(source, "HasSamplesId");
    display = read_json_bool(root, "Display");
    if (!data_path.empty() && filesystem::exists(data_path))
    {
        const uint64_t saved_input = tokenizer->fingerprint();
        const uint64_t saved_target = target_tokenizer ? target_tokenizer->fingerprint() : 0;
        const vector<string> saved_labels = labels;
        read_txt(data_path);
        throw_if(tokenizer->fingerprint() != saved_input
                 || (target_tokenizer ? target_tokenizer->fingerprint() : 0) != saved_target
                 || labels != saved_labels,
                 "TextDataset: source token or label mapping differs from the saved dataset.");
        const Index samples = get_samples_number();
        const Index features = get_features_number();
        read_json_blocks(root);
        throw_if(get_samples_number() != samples || get_features_number() != features,
                 "TextDataset: saved metadata dimensions do not match the source corpus.");
    }
    else
    {
        if (read_json_index(root, "InputSequenceLength") > 0)
            configure(0, read_json_index(root, "InputSequenceLength"), read_json_index(root, "TargetSequenceLength"));
        Json metadata = *root;
        erase_if(metadata.as_object(), [](const auto& field) { return field.first == "Samples"; });
        read_json_blocks(&metadata);
    }
}

}
