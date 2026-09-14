#include "tests/pch.h"
#include "opennn/dataset/text_dataset.h"
#include "opennn/dataset/dataset.h"
#include "tests/numerical_derivatives.h"
#include <cmath>
#include "opennn/models/models.h"
#include "opennn/network/network.h"
#include "opennn/network/forward_propagation.h"
#include "opennn/training/loss.h"
#include "opennn/core/configuration.h"
#include "opennn/evaluation/evaluation.h"
#include "opennn/dataset/batch.h"

using namespace opennn;

namespace
{
    string temp_language_file(const string& name, const string& content)
    {
        const string file_path = (filesystem::temp_directory_path() / name).string();

        ofstream outfile(file_path);
        if (!outfile.is_open())
            throw runtime_error("Failed to open temporary language file for writing: " + file_path);

        outfile << content;
        outfile.close();

        return file_path;
    }

    void remove_language_file(const string& file_path)
    {
        error_code error;
        filesystem::remove(file_path, error);
        filesystem::remove_all(file_path + ".cache", error);
    }

    const string sentiment_content =
        "great phone excellent product\tGood\n"
        "terrible battery awful screen\tBad\n"
        "amazing camera fantastic value\tGood\n"
        "broken charger useless device\tBad\n";
}

TEST(TextDatasetClassification, DefaultConstructorIsEmpty)
{
    TextDataset dataset;

    EXPECT_EQ(dataset.get_samples_number(), 0);
    EXPECT_EQ(dataset.get_vocabulary_size(), 0);
    EXPECT_EQ(dataset.get_vocabulary_size(VariableRole::Target), 0);
    EXPECT_EQ(dataset.get_sequence_length(), 0);
    EXPECT_EQ(dataset.get_sequence_length(VariableRole::Target), 0);
}

TEST(TextDatasetClassification, ClassificationLabelsRemainAtomicAndInputLengthIsCapped)
{
    const string file_path = temp_language_file("opennn_atomic_labels.txt",
        "one two three four five\tSci_Tech\n"
        "six seven eight nine ten\tWorld News\n");
    TextDataset dataset(TextDataset::Options{.sequence_length = 3});
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);
    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));
    EXPECT_EQ(dataset.get_vocabulary_size(VariableRole::Target), 2);
    const auto& vocabulary = dataset.get_vocabulary(VariableRole::Target);
    EXPECT_NE(find(vocabulary.begin(), vocabulary.end(), "sci_tech"), vocabulary.end());
    EXPECT_NE(find(vocabulary.begin(), vocabulary.end(), "world news"), vocabulary.end());
    EXPECT_EQ(dataset.get_sequence_length(VariableRole::Target), 1);
    EXPECT_LE(dataset.get_sequence_length(), 3);
    remove_language_file(file_path);
}

TEST(TextDatasetClassification, TokenizerFramingTokenIds)
{
    const string path = temp_language_file("opennn_text_framing.txt", sentiment_content);
    TextDataset dataset;
    dataset.read_txt(path);
    const TokenizerOperator* tokenizer = dataset.get_tokenizer();
    ASSERT_NE(tokenizer, nullptr);
    EXPECT_EQ(tokenizer->token_to_id("[PAD]"), 0);
    EXPECT_EQ(tokenizer->token_to_id("[UNK]"), 1);
    EXPECT_EQ(tokenizer->token_to_id("[START]"), 2);
    EXPECT_EQ(tokenizer->token_to_id("[END]"), 3);
    remove_language_file(path);
}

TEST(TextDatasetClassification, ReadTxtBuildsVocabularyAndSequences)
{
    const string file_path = temp_language_file("opennn_language_sentiment.txt", sentiment_content);

    TextDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));

    EXPECT_EQ(dataset.get_samples_number(), 4);
    EXPECT_GT(dataset.get_vocabulary_size(), Index(4));
    EXPECT_EQ(dataset.get_vocabulary_size(VariableRole::Target), 2);
    EXPECT_GT(dataset.get_sequence_length(), 0);
    EXPECT_EQ(dataset.get_sequence_length(VariableRole::Target), 1);

    dataset.set_sample_role(0, SampleRole::None);
    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));
    EXPECT_EQ(dataset.get_samples_number(SampleRole::None), 0);

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, ReadTxtInputVocabularyContainsReservedTokens)
{
    const string file_path = temp_language_file("opennn_language_reserved.txt", sentiment_content);

    TextDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));

    const vector<string>& input_vocabulary = dataset.get_vocabulary();

    ASSERT_GE(ssize(input_vocabulary), 4);

    EXPECT_EQ(input_vocabulary[0], "[PAD]");
    EXPECT_EQ(input_vocabulary[1], "[UNK]");
    EXPECT_EQ(input_vocabulary[2], "[START]");
    EXPECT_EQ(input_vocabulary[3], "[END]");

    EXPECT_EQ(ssize(input_vocabulary), dataset.get_vocabulary_size());

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, ReadTxtMatrixShapesAndStartToken)
{
    const string file_path = temp_language_file("opennn_language_matrix.txt", sentiment_content);

    TextDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));

    const Index input_sequence_length = dataset.get_sequence_length();
    const Index target_sequence_length = dataset.get_sequence_length(VariableRole::Target);

    const Shape input_shape = dataset.get_shape("Input");
    const Shape target_shape = dataset.get_shape("Target");

    ASSERT_EQ(input_shape.get_rank(), 1);
    EXPECT_EQ(input_shape[0], input_sequence_length);

    ASSERT_EQ(target_shape.get_rank(), 1);
    EXPECT_EQ(target_shape[0], target_sequence_length);

    const MatrixR& data = dataset.get_data();
    ASSERT_EQ(data.rows(), 4);
    ASSERT_EQ(data.cols(), input_sequence_length + target_sequence_length);

    for (Index i = 0; i < data.rows(); ++i)
        EXPECT_FLOAT_EQ(data(i, 0), TokenizerOperator::START_INDEX);

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, ReadTxtTargetDistribution)
{
    const string file_path = temp_language_file("opennn_language_distribution.txt", sentiment_content);

    TextDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));

    const VectorI distribution = dataset.calculate_target_distribution();

    ASSERT_EQ(distribution.size(), 2);
    EXPECT_EQ(distribution(0), 2);
    EXPECT_EQ(distribution(1), 2);

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, ExplicitReadAndCacheRestoreVocabulary)
{
    const string file_path = temp_language_file("opennn_language_ctor.txt", sentiment_content);

    TextDataset dataset;
    dataset.read_txt(file_path);

    EXPECT_EQ(dataset.get_samples_number(), 4);
    EXPECT_GT(dataset.get_vocabulary_size(), Index(4));
    EXPECT_EQ(dataset.get_vocabulary_size(VariableRole::Target), 2);

    TextDataset cached_dataset;
    cached_dataset.read_txt(file_path);
    EXPECT_EQ(cached_dataset.get_vocabulary(), dataset.get_vocabulary());
    EXPECT_EQ(cached_dataset.get_vocabulary(VariableRole::Target), dataset.get_vocabulary(VariableRole::Target));
    EXPECT_EQ(cached_dataset.get_samples_number(), dataset.get_samples_number());

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, BinaryStorageDoesNotPretendDeviceResidency)
{
    const string file_path = temp_language_file(
        "opennn_language_residency.txt", sentiment_content);

    TextDataset dataset;
    dataset.read_txt(file_path);
    ASSERT_EQ(dataset.get_storage_mode(), Dataset::StorageMode::BinaryFile);
    ASSERT_EQ(dataset.get_data().size(), 0);

    dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
    EXPECT_TRUE(dataset.requests_device_residency());
    EXPECT_FALSE(dataset.uses_device_residency());

    dataset.enable_device_residency();
    EXPECT_FALSE(dataset.is_device_resident());
    EXPECT_FALSE(dataset.uses_device_residency());

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, MatrixStorageAfterReadingIsRefused)
{
    const string file_path = temp_language_file(
        "opennn_language_late_matrix.txt", sentiment_content);

    TextDataset dataset;
    dataset.read_txt(file_path);
    ASSERT_EQ(dataset.get_storage_mode(), Dataset::StorageMode::BinaryFile);
    ASSERT_EQ(dataset.get_data().size(), 0);

    // The Matrix fill paths index the data matrix, which read_txt only fills
    // when the mode was already Matrix. Switching now used to segfault on the
    // first batch rather than say anything.
    EXPECT_THROW(dataset.set_storage_mode(Dataset::StorageMode::Matrix),
                 runtime_error);
    EXPECT_EQ(dataset.get_storage_mode(), Dataset::StorageMode::BinaryFile);

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, CsvReaderPreservesQuotedSeparators)
{
    const string file_path = temp_language_file(
        "opennn_language_quoted.txt",
        "\"hello\tworld\"\tGood\n"
        "\"goodbye\tworld\"\tBad\n");

    TextDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));
    EXPECT_EQ(dataset.get_samples_number(), 2);

    const vector<string>& input_vocabulary = dataset.get_vocabulary();
    EXPECT_NE(ranges::find(input_vocabulary, "hello"), input_vocabulary.end());
    EXPECT_NE(ranges::find(input_vocabulary, "world"), input_vocabulary.end());

    remove_language_file(file_path);
}

TEST(TextDatasetClassification, CsvReaderAcceptsUnbalancedQuotes)
{
    const string file_path = temp_language_file(
        "opennn_language_stray_quote.txt",
        "looks great and is strong.\"\tGood\n"
        "she said \"whoa\tBad\n"
        "\"hello\tworld\"\tGood\n");

    TextDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt(dataset.get_data_path()));
    EXPECT_EQ(dataset.get_samples_number(), 3);

    const vector<string>& target_vocabulary = dataset.get_vocabulary(VariableRole::Target);
    EXPECT_NE(ranges::find(target_vocabulary, "good"), target_vocabulary.end());
    EXPECT_NE(ranges::find(target_vocabulary, "bad"), target_vocabulary.end());

    remove_language_file(file_path);
}




namespace
{

filesystem::path write_text_generation_file(string_view name, string_view contents)
{
    const filesystem::path path = filesystem::temp_directory_path() / name;
    error_code error;
    filesystem::remove_all(path.string() + ".cache", error);

    ofstream file(path, ios::binary);
    file << contents;
    return path;
}

void remove_text_generation_files(const filesystem::path& path)
{
    error_code error;
    filesystem::remove(path, error);
    filesystem::remove_all(path.string() + ".cache", error);
}

}

TEST(TextDatasetNextToken, OwnsVocabularyThroughTokenizer)
{
    const filesystem::path path = write_text_generation_file(
        "opennn_text_generation_vocabulary.txt",
        "alpha beta alpha gamma beta delta alpha beta gamma delta");

    TextDataset dataset(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = 3});
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_data_path(path);
    dataset.set_display(false);
    dataset.read_txt(dataset.get_data_path());

    const vector<string> expected_vocabulary =
        {"[PAD]", "[UNK]", "alpha", "beta", "delta", "gamma"};

    EXPECT_EQ(dataset.get_vocabulary(), expected_vocabulary);
    EXPECT_EQ(dataset.get_tokenizer()->get_vocabulary(), expected_vocabulary);

    const Index sequence_length = dataset.get_sequence_length();
    const MatrixR& data = dataset.get_data();

    for (Index sample = 0; sample < data.rows(); ++sample)
        for (Index token = 0; token + 1 < sequence_length; ++token)
            EXPECT_FLOAT_EQ(data(sample, sequence_length + token),
                            data(sample, token + 1));

    dataset.set_sample_role(0, SampleRole::None);
    dataset.read_txt(dataset.get_data_path());
    EXPECT_EQ(dataset.get_samples_number(SampleRole::None), 0);

    remove_text_generation_files(path);
}

TEST(TextDatasetNextToken, MatrixAndBinaryStorageProduceEqualBatches)
{
    const filesystem::path path = write_text_generation_file(
        "opennn_text_generation_storage.txt",
        "zero one two three four five six seven eight nine ten eleven");

    constexpr Index sequence_length = 3;

    TextDataset matrix_dataset(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = sequence_length});
    matrix_dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    matrix_dataset.set_data_path(path);
    matrix_dataset.set_display(false);
    matrix_dataset.read_txt(matrix_dataset.get_data_path());

    TextDataset binary_dataset(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = sequence_length});
    binary_dataset.read_txt(path);
    binary_dataset.set_display(false);

    const vector<Index> all_samples = [&]
    {
        vector<Index> indices(size_t(binary_dataset.get_samples_number()));
        iota(indices.begin(), indices.end(), Index(0));
        return indices;
    }();

    vector<float> matrix_inputs(size_t(all_samples.size() * sequence_length));
    vector<float> matrix_targets(size_t(all_samples.size() * sequence_length));
    vector<float> binary_inputs(size_t(all_samples.size() * sequence_length));
    vector<float> binary_targets(size_t(all_samples.size() * sequence_length));

    matrix_dataset.fill_inputs(all_samples,
                               matrix_dataset.get_feature_indices("Input"),
                               matrix_inputs.data(),
                               FillMode::Inference);
    matrix_dataset.fill_targets(all_samples,
                                matrix_dataset.get_feature_indices("Target"),
                                matrix_targets.data(),
                                FillMode::Inference);
    binary_dataset.fill_inputs(all_samples,
                               binary_dataset.get_feature_indices("Input"),
                               binary_inputs.data(),
                               FillMode::Inference);
    binary_dataset.fill_targets(all_samples,
                                binary_dataset.get_feature_indices("Target"),
                                binary_targets.data(),
                                FillMode::Inference);

    EXPECT_EQ(binary_inputs, matrix_inputs);
    EXPECT_EQ(binary_targets, matrix_targets);

    TextDataset cached_dataset(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = sequence_length});
    cached_dataset.read_txt(path);
    EXPECT_EQ(cached_dataset.get_vocabulary(), binary_dataset.get_vocabulary());
    EXPECT_EQ(cached_dataset.get_samples_number(), binary_dataset.get_samples_number());

    remove_text_generation_files(path);
}

TEST(TextDatasetNextToken, UsesLoadedBytePairTokenizer)
{
    const filesystem::path directory =
        filesystem::temp_directory_path() / "opennn_text_generation_bpe";
    filesystem::create_directories(directory);

    const filesystem::path text_path = directory / "corpus.txt";
    const filesystem::path vocabulary_path = directory / "vocab.json";
    const filesystem::path merges_path = directory / "merges.txt";

    {
        ofstream file(text_path);
        file << "abababababab";
    }
    {
        ofstream file(vocabulary_path);
        file << R"({"a":0,"b":1,"ab":2})";
    }
    {
        ofstream file(merges_path);
        file << "#version: 0.2\na b\n";
    }

    TextDataset dataset(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = 2});
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_tokenizer(
        make_unique<BytePairTokenizer>(vocabulary_path, merges_path));
    dataset.set_data_path(text_path);
    dataset.set_display(false);
    dataset.read_txt(dataset.get_data_path());

    EXPECT_EQ(dataset.get_vocabulary_size(), 4);
    EXPECT_EQ(dataset.get_vocabulary()[3], "ab");
    EXPECT_GT(dataset.get_samples_number(), 0);

    JsonWriter saved;
    dataset.to_JSON(saved);
    JsonDocument document;
    document.set_root(Json::parse(saved.c_str()));
    TextDataset loaded;
    loaded.from_JSON(document);
    ASSERT_NE(loaded.get_tokenizer(), nullptr);
    EXPECT_EQ(loaded.get_tokenizer()->get_kind(), "BytePair");
    EXPECT_EQ(loaded.get_tokenizer()->encode("abababab"), dataset.get_tokenizer()->encode("abababab"));
    EXPECT_TRUE(loaded.get_data().isApprox(dataset.get_data()));

    error_code error;
    filesystem::remove_all(directory, error);
}

TEST(TextDatasetNextToken, CacheIdentityIncludesSequenceConfiguration)
{
    const filesystem::path path = write_text_generation_file(
        "opennn_text_generation_cache.txt",
        "a b c d e f g h i j k l m n o p q r s t");

    TextDataset short_sequences(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = 2});
    short_sequences.read_txt(path);
    TextDataset long_sequences(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = 3});
    long_sequences.read_txt(path);

    const filesystem::path cache_directory = path.string() + ".cache";
    Index cache_files = 0;
    for (const filesystem::directory_entry& entry :
         filesystem::directory_iterator(cache_directory))
        if (entry.is_regular_file()) ++cache_files;

    EXPECT_GE(cache_files, 2);

    remove_text_generation_files(path);
}

TEST(TextDatasetNextToken, CachePreservesUnfittedTokenizerConfiguration)
{
    const filesystem::path path = write_text_generation_file(
        "opennn_text_custom_reserved.txt", "a b c d e f g h i");
    const TextDataset::Options options{.task = TextDataset::Task::NextToken, .sequence_length = 2};
    TextDataset defaults(options);
    defaults.read_txt(path);

    const vector<string> reserved{"[PAD]", "[UNK]", "[EXTRA]"};
    TextDataset custom(options);
    custom.set_tokenizer(make_unique<WordLevelTokenizer>(reserved));
    custom.read_txt(path);
    ASSERT_GE(custom.get_vocabulary_size(), 3);
    EXPECT_EQ(custom.get_vocabulary()[2], "[EXTRA]");
    EXPECT_EQ(custom.get_tokenizer()->get_reserved_tokens(), reserved);
    const vector<string> vocabulary = custom.get_vocabulary();
    custom.read_txt(path);
    EXPECT_EQ(custom.get_vocabulary(), vocabulary);

    TextDataset cached(options);
    cached.set_tokenizer(make_unique<WordLevelTokenizer>(reserved));
    cached.read_txt(path);
    EXPECT_EQ(cached.get_vocabulary(), vocabulary);
    EXPECT_EQ(cached.get_tokenizer()->get_reserved_tokens(), reserved);
    EXPECT_EQ(cached.get_tokenizer()->encode("a b c"), custom.get_tokenizer()->encode("a b c"));
    remove_text_generation_files(path);
}

namespace
{
    void load_wordpiece_corpus(TextDataset& dataset, const string& text_path, const string& vocabulary_path)
    {
        auto tokenizer = make_unique<WordPieceTokenizer>();
        tokenizer->load_vocabulary(vocabulary_path);
        dataset.set_tokenizer(std::move(tokenizer), VariableRole::Decoder);
        dataset.read_txt(text_path);
    }

    string write_lines(const string& name, const vector<string>& lines)
    {
        const string path = (filesystem::temp_directory_path() / name).string();
        ofstream file(path);
        for (const string& line : lines)
            file << line << "\n";
        file.close();
        return path;
    }

    const vector<string> bert_vocabulary = {
        "[PAD]", "[UNK]", "[CLS]", "[SEP]",
        "good", "movie", "bad", "film", "great", "terrible"
    };

    const vector<string> labelled_text = {
        "good movie\tpositive",
        "bad film\tnegative",
        "great movie\tpositive",
        "terrible film\tnegative",
        "good film\tpositive",
        "bad movie\tnegative",
        "great film\tpositive",
        "terrible movie\tnegative",
        "good great\tpositive",
        "bad terrible\tnegative"
    };

    void clean_up(const string& vocab_path, const string& text_path, Index seq)
    {
        error_code error;
        filesystem::remove(vocab_path, error);
        filesystem::remove(text_path, error);
        filesystem::remove(text_path + ".bert_v2_" + to_string(seq) + ".csv", error);
        filesystem::remove(text_path + ".bert_v3_" + to_string(seq) + ".bin", error);
        filesystem::remove_all(text_path + ".cache", error);
    }
}

TEST(TextDatasetMasked, TokenizesAndWiresRoles)
{
    const string vocab_path = write_lines("opennn_bertds_vocab.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text.txt",  labelled_text);

    const Index seq = 8;
    TextDataset dataset(TextDataset::Options{.input_layout = TextDataset::InputLayout::TokensAndMask, .sequence_length = seq});
    load_wordpiece_corpus(dataset, text_path, vocab_path);

    EXPECT_EQ(dataset.get_sequence_length(), seq);
    EXPECT_EQ(dataset.get_samples_number(), Index(labelled_text.size()));

    EXPECT_EQ(dataset.get_features_number("Decoder"), seq);
    EXPECT_EQ(dataset.get_features_number("Input"), seq);
    EXPECT_GE(dataset.get_features_number("Target"), 1);
    EXPECT_EQ(dataset.get_training_tokenizer(VariableRole::Input), nullptr);
    ASSERT_NE(dataset.get_training_tokenizer(VariableRole::Decoder), nullptr);
    EXPECT_EQ(dataset.get_training_tokenizer(VariableRole::Decoder)->get_kind(), "WordPiece");

    const MatrixR& data = dataset.get_data();
    EXPECT_FLOAT_EQ(data(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(data(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(data(0, 2), 5.0f);
    EXPECT_FLOAT_EQ(data(0, 3), 3.0f);
    EXPECT_FLOAT_EQ(data(0, 4), 0.0f);
    EXPECT_FLOAT_EQ(data(0, seq), 1.0f);

    TextDataset cached_dataset(TextDataset::Options{.input_layout = TextDataset::InputLayout::TokensAndMask, .sequence_length = seq});
    load_wordpiece_corpus(cached_dataset, text_path, vocab_path);
    EXPECT_TRUE(cached_dataset.get_data().isApprox(data));

    clean_up(vocab_path, text_path, seq);
}

TEST(TextDatasetMasked, FeedsBertClassifierForward)
{
    const string vocab_path = write_lines("opennn_bertds_vocab2.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text2.txt",  labelled_text);

    const Index seq = 8;
    TextDataset dataset(TextDataset::Options{.input_layout = TextDataset::InputLayout::TokensAndMask, .sequence_length = seq});
    load_wordpiece_corpus(dataset, text_path, vocab_path);

    const Index batch  = dataset.get_samples_number();
    const Index labels = dataset.get_features_number("Target");
    const MatrixR& data = dataset.get_data();

    vector<float> input_ids(size_t(batch * seq));
    vector<float> token_type(size_t(batch * seq));
    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < seq; ++s)
        {
            input_ids[size_t(b * seq + s)]  = data(b, s);
            token_type[size_t(b * seq + s)] = data(b, seq + s);
        }

    BertForSequenceClassification model(seq, Index(bert_vocabulary.size()),
                                                   8,           2,                  16,
                                                   1, labels);
    model.set_parameters_random();

    ForwardPropagation forward_propagation(batch, &model);
    vector<TensorView> inputs = {
        TensorView(input_ids.data(),  {batch, seq}),
        TensorView(token_type.data(), {batch, seq})
    };
    model.forward_propagate(inputs, forward_propagation, ForwardPropagationMode::Inference);

    const TensorView output = forward_propagation.get_outputs();
    ASSERT_EQ(output.get_shape().get_rank(), 2);
    EXPECT_EQ(output.get_shape()[0], batch);
    EXPECT_EQ(output.get_shape()[1], labels);

    const float* values = output.as<float>();
    for (Index i = 0; i < output.size(); ++i)
        EXPECT_TRUE(isfinite(values[i])) << "non-finite output at " << i;

    clean_up(vocab_path, text_path, seq);
}

TEST(TextDatasetMasked, BertClassifierGradientOnCpu)
{
    const string vocab_path = write_lines("opennn_bertds_vocab3.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text3.txt",  labelled_text);

    const Index seq = 8;
    TextDataset dataset(TextDataset::Options{.input_layout = TextDataset::InputLayout::TokensAndMask, .sequence_length = seq});
    load_wordpiece_corpus(dataset, text_path, vocab_path);

    const Index labels = dataset.get_features_number("Target");

    BertForSequenceClassification model(seq, Index(bert_vocabulary.size()),
                                                   8,           2,                  16,
                                                   1, labels);
    model.set_parameters_random();

    Loss loss(&model, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    const VectorR gradient = calculate_gradient(loss);
    ASSERT_GT(gradient.size(), 0);
    EXPECT_TRUE(gradient.allFinite());

    clean_up(vocab_path, text_path, seq);
}

TEST(TextDatasetMasked, EvaluationSupportsMultipleInputs)
{
    const string vocab_path = write_lines("opennn_bertds_vocab4.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text4.txt", labelled_text);

    const Index seq = 8;
    TextDataset dataset(TextDataset::Options{.input_layout = TextDataset::InputLayout::TokensAndMask, .sequence_length = seq});
    load_wordpiece_corpus(dataset, text_path, vocab_path);
    dataset.set_sample_roles("Testing");

    const Index labels = dataset.get_features_number("Target");
    BertForSequenceClassification model(
        seq, Index(bert_vocabulary.size()), 8, 2, 16, 1, labels);
    model.set_parameters_random();

    Evaluation evaluation(&model, &dataset);
    evaluation.set_batch_size(3);

    const MatrixI confusion = evaluation.calculate_confusion();

    EXPECT_EQ(confusion.rows(), 3);
    EXPECT_EQ(confusion.cols(), 3);
    EXPECT_EQ(confusion.bottomRightCorner(1, 1)(0, 0), dataset.get_samples_number());

    clean_up(vocab_path, text_path, seq);
}

TEST(TextDatasetSequenceToSequence, ShiftedDecoderAndFeatureSelectionsMatchMatrix)
{
    const string path = temp_language_file("opennn_text_pairs.txt",
        "hello world\tbonjour\nhello\tbonne nuit\n");
    const TextDataset::Options options{.task = TextDataset::Task::SequenceToSequence};
    TextDataset matrix(options), binary(options);
    matrix.set_storage_mode(Dataset::StorageMode::Matrix);
    matrix.read_txt(path);
    binary.read_txt(path);
    ASSERT_EQ(matrix.get_sequence_length(), 4);
    ASSERT_EQ(matrix.get_sequence_length(VariableRole::Target), 3);
    EXPECT_EQ(matrix.get_tokenizer(VariableRole::Decoder), matrix.get_tokenizer(VariableRole::Target));
    EXPECT_NE(matrix.get_tokenizer(), matrix.get_tokenizer(VariableRole::Target));
    const MatrixR& data = matrix.get_data();
    EXPECT_FLOAT_EQ(data(0, 4), TokenizerOperator::START_INDEX);
    EXPECT_FLOAT_EQ(data(0, 5), data(0, 7));
    EXPECT_FLOAT_EQ(data(0, 6), data(0, 8));
    EXPECT_FLOAT_EQ(data(0, 8), TokenizerOperator::END_INDEX);
    EXPECT_FLOAT_EQ(data(0, 9), 0.0f);

    const vector<Index> samples{1, 0};
    for (VariableRole role : {VariableRole::Input, VariableRole::Decoder, VariableRole::Target})
    {
        const vector<Index> all_features = matrix.get_feature_indices(role);
        for (const vector<Index>& features : {all_features, vector<Index>{all_features.back(), all_features.front()}})
        {
            vector<float> actual(samples.size() * features.size() + 1, -99.0f);
            const auto fill = [&](const TextDataset& dataset, float* values)
            {
                if (role == VariableRole::Input) dataset.fill_inputs(samples, features, values, FillMode::Inference);
                else if (role == VariableRole::Decoder) dataset.fill_decoder(samples, features, values, FillMode::Inference);
                else dataset.fill_targets(samples, features, values, FillMode::Inference);
            };
            vector<float> expected(samples.size() * features.size());
            fill(matrix, expected.data());
            fill(binary, actual.data());
            EXPECT_EQ(actual.back(), -99.0f);
            actual.pop_back();
            EXPECT_EQ(actual, expected);
        }
    }
    for (const TextDataset* dataset : {&matrix, &binary})
    {
        float sentinel = -99.0f;
        EXPECT_THROW(dataset->fill_inputs({0}, {999}, &sentinel, FillMode::Inference), std::exception);
        EXPECT_THROW(dataset->fill_inputs({-1}, {0}, &sentinel, FillMode::Inference), std::exception);
        EXPECT_THROW(dataset->fill_inputs({2}, {0}, &sentinel, FillMode::Inference), std::exception);
        EXPECT_FLOAT_EQ(sentinel, -99.0f);
    }
    remove_language_file(path);
}

TEST(TextDatasetClassification, JsonRestoresSplitAndVariableMetadata)
{
    const string path = temp_language_file("opennn_text_roundtrip.txt", sentiment_content);
    for (const auto storage : {Dataset::StorageMode::Matrix, Dataset::StorageMode::BinaryFile})
    {
        TextDataset dataset;
        dataset.set_storage_mode(storage);
        dataset.read_txt(path);
        dataset.set_sample_roles("Training");
        dataset.set_sample_role(0, SampleRole::Testing);
        dataset.set_sample_role(1, SampleRole::None);
        dataset.set_sample_role(2, SampleRole::Validation);
        dataset.set_variable_names({"review", "sentiment"});
        JsonWriter saved;
        dataset.to_JSON(saved);
        JsonDocument document;
        document.set_root(Json::parse(saved.c_str()));
        TextDataset restored;
        restored.from_JSON(document);
        EXPECT_EQ(restored.get_variable_names(), dataset.get_variable_names());
        EXPECT_EQ(restored.get_vocabulary(), dataset.get_vocabulary());
        EXPECT_EQ(restored.get_vocabulary(VariableRole::Target), dataset.get_vocabulary(VariableRole::Target));
        for (SampleRole role : {SampleRole::Training, SampleRole::Validation, SampleRole::Testing, SampleRole::None})
            EXPECT_EQ(restored.get_sample_indices(role), dataset.get_sample_indices(role));
        if (storage == Dataset::StorageMode::Matrix)
            EXPECT_TRUE(restored.get_data().isApprox(dataset.get_data()));

        document.get_root()["Dataset"]["DataSource"]["Path"] = path + ".missing";
        TextDataset deployment;
        deployment.from_JSON(document);
        EXPECT_EQ(deployment.get_samples_number(), 0);
        EXPECT_TRUE(deployment.get_data().size() == 0);
        EXPECT_EQ(deployment.get_variable_names(), dataset.get_variable_names());
        EXPECT_EQ(deployment.get_vocabulary(), dataset.get_vocabulary());
    }
    remove_language_file(path);
}

TEST(TextDatasetClassification, JsonRejectsChangedClassMappings)
{
    const string name = "opennn_text_changed_labels.txt";
    const string path = temp_language_file(name,
        "same\ta\nsame\ta\nsame\ta\nsame\tb\nsame\tb\nsame\tc\n");
    TextDataset original;
    original.set_storage_mode(Dataset::StorageMode::Matrix);
    original.read_txt(path);
    ASSERT_EQ(original.get_vocabulary(VariableRole::Target), (vector<string>{"a", "b", "c"}));
    JsonWriter saved;
    original.to_JSON(saved);
    JsonDocument document;
    document.set_root(Json::parse(saved.c_str()));

    temp_language_file(name, "same\tc\nsame\tc\nsame\tc\nsame\tb\nsame\tb\nsame\ta\n");
    TextDataset restored;
    EXPECT_THROW(restored.from_JSON(document), std::exception);
    remove_language_file(path);
}

namespace
{
void check_resident_text_matrix(TextDataset& dataset)
{
    const vector<Index> samples{1, 0};
    const MatrixR expected = dataset.get_data();
    for (const Type precision : {Type::FP32, Type::BF16})
    {
        if (precision == Type::BF16 && device::cuda_compute_capability() < 80) continue;
        SCOPED_TRACE(precision == Type::FP32 ? "FP32" : "BF16");
        Configuration::instance().set(Device::CUDA, precision);
        dataset.enable_device_residency();
        ASSERT_TRUE(dataset.is_device_resident());
        Batch batch(Index(samples.size()), &dataset, Configuration::instance().resolve());
        batch.fill(samples, dataset.get_feature_selection(), FillMode::Inference);
        EXPECT_EQ(batch.device_gather.has_value(), dataset.get_shape(VariableRole::Decoder).empty());
        batch.upload_to_device_batch_async(batch, device::get_transfer_stream());
        batch.wait_h2d_on_compute_stream();
        const DeviceStream stream = device::get_compute_stream();
        for (VariableRole role : {VariableRole::Input, VariableRole::Decoder, VariableRole::Target})
        {
            const vector<Index> features = dataset.get_feature_indices(role);
            if (features.empty()) continue;
            const BatchSlot& slot = role == VariableRole::Input ? batch.input
                                  : role == VariableRole::Decoder ? batch.decoder : batch.target;
            vector<float> actual(size_t(slot.shape.size()));
            copy_device_to_host_float(slot.buffer.data(), slot.type, slot.shape.size(), actual.data(), stream);
            device::synchronize(stream);
            for (size_t row = 0; row < samples.size(); ++row)
                for (size_t column = 0; column < features.size(); ++column)
                    EXPECT_FLOAT_EQ(actual[row * features.size() + column], expected(samples[row], features[column]));
        }
    }
}
}

TEST(TextDataset, CudaResidentMatricesPreserveTokenAndMaskLayouts)
{
    if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device.";
    const ScopeExit reset_configuration([] { Configuration::instance().set(Device::CPU, Type::FP32); });
    const filesystem::path corpus = write_text_generation_file("opennn_resident_text.txt",
        "zero one two three four five six seven eight nine ten eleven");
    TextDataset next_token(TextDataset::Options{.task = TextDataset::Task::NextToken, .sequence_length = 3});
    next_token.set_storage_mode(Dataset::StorageMode::Matrix);
    next_token.read_txt(corpus);
    ASSERT_EQ(next_token.get_data().cols(), 6);
    check_resident_text_matrix(next_token);

    const string vocabulary = write_lines("opennn_resident_wordpiece_vocab.txt", bert_vocabulary);
    const string text = write_lines("opennn_resident_wordpiece_text.txt", labelled_text);
    TextDataset masked(TextDataset::Options{.input_layout = TextDataset::InputLayout::TokensAndMask, .sequence_length = 8});
    load_wordpiece_corpus(masked, text, vocabulary);
    check_resident_text_matrix(masked);
    remove_text_generation_files(corpus);
    clean_up(vocabulary, text, 8);
}
