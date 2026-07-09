#include "pch.h"

#include "opennn/language_dataset.h"
#include "opennn/dataset.h"

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


TEST(LanguageDataset, DefaultConstructorIsEmpty)
{
    LanguageDataset dataset;

    EXPECT_EQ(dataset.get_samples_number(), 0);
    EXPECT_EQ(dataset.get_input_vocabulary_size(), 0);
    EXPECT_EQ(dataset.get_target_vocabulary_size(), 0);
    EXPECT_EQ(dataset.get_maximum_input_sequence_length(), 0);
    EXPECT_EQ(dataset.get_maximum_target_sequence_length(), 0);
}


TEST(LanguageDataset, ReservedTokenConstants)
{
    EXPECT_EQ(LanguageDataset::PAD_TOKEN, "[PAD]");
    EXPECT_EQ(LanguageDataset::UNK_TOKEN, "[UNK]");
    EXPECT_EQ(LanguageDataset::START_TOKEN, "[START]");
    EXPECT_EQ(LanguageDataset::END_TOKEN, "[END]");

    ASSERT_EQ(LanguageDataset::reserved_tokens.size(), 4);
    EXPECT_EQ(LanguageDataset::reserved_tokens[0], "[PAD]");
    EXPECT_EQ(LanguageDataset::reserved_tokens[1], "[UNK]");
    EXPECT_EQ(LanguageDataset::reserved_tokens[2], "[START]");
    EXPECT_EQ(LanguageDataset::reserved_tokens[3], "[END]");

    EXPECT_FLOAT_EQ(LanguageDataset::UNK_INDEX, 1.0f);
    EXPECT_FLOAT_EQ(LanguageDataset::START_INDEX, 2.0f);
    EXPECT_FLOAT_EQ(LanguageDataset::END_INDEX, 3.0f);
}


TEST(LanguageDataset, SetVocabularyRoundTrip)
{
    LanguageDataset dataset;

    const vector<string> input_vocabulary = { "[PAD]", "[UNK]", "[START]", "[END]", "hello", "world" };
    const vector<string> target_vocabulary = { "[PAD]", "[UNK]", "[START]", "[END]", "yes", "no" };

    dataset.set_input_vocabulary(input_vocabulary);
    dataset.set_target_vocabulary(target_vocabulary);

    EXPECT_EQ(dataset.get_input_vocabulary_size(), 6);
    EXPECT_EQ(dataset.get_target_vocabulary_size(), 6);

    EXPECT_EQ(dataset.get_target_vocabulary(), target_vocabulary);

    const unordered_map<string, Index>& input_map = dataset.get_input_vocabulary_map();

    ASSERT_EQ(input_map.size(), 6);
    EXPECT_EQ(input_map.at("[PAD]"), 0);
    EXPECT_EQ(input_map.at("[UNK]"), 1);
    EXPECT_EQ(input_map.at("[START]"), 2);
    EXPECT_EQ(input_map.at("[END]"), 3);
    EXPECT_EQ(input_map.at("hello"), 4);
    EXPECT_EQ(input_map.at("world"), 5);
}


TEST(LanguageDataset, ReadTxtBuildsVocabularyAndSequences)
{
    const string file_path = temp_language_file("opennn_language_sentiment.txt", sentiment_content);

    LanguageDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt());

    EXPECT_EQ(dataset.get_samples_number(), 4);
    EXPECT_GT(dataset.get_input_vocabulary_size(), Index(LanguageDataset::reserved_tokens.size()));
    EXPECT_EQ(dataset.get_target_vocabulary_size(), 6);
    EXPECT_GT(dataset.get_maximum_input_sequence_length(), 0);
    EXPECT_EQ(dataset.get_maximum_target_sequence_length(), 1);

    remove_language_file(file_path);
}


TEST(LanguageDataset, ReadTxtInputVocabularyMapContainsReservedTokens)
{
    const string file_path = temp_language_file("opennn_language_reserved.txt", sentiment_content);

    LanguageDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt());

    const unordered_map<string, Index>& input_map = dataset.get_input_vocabulary_map();

    ASSERT_TRUE(input_map.contains("[PAD]"));
    ASSERT_TRUE(input_map.contains("[UNK]"));
    ASSERT_TRUE(input_map.contains("[START]"));
    ASSERT_TRUE(input_map.contains("[END]"));

    EXPECT_EQ(input_map.at("[PAD]"), 0);
    EXPECT_EQ(input_map.at("[UNK]"), 1);
    EXPECT_EQ(input_map.at("[START]"), 2);
    EXPECT_EQ(input_map.at("[END]"), 3);

    EXPECT_EQ(Index(input_map.size()), dataset.get_input_vocabulary_size());

    remove_language_file(file_path);
}


TEST(LanguageDataset, ReadTxtMatrixShapesAndStartToken)
{
    const string file_path = temp_language_file("opennn_language_matrix.txt", sentiment_content);

    LanguageDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt());

    const Index input_sequence_length = dataset.get_maximum_input_sequence_length();
    const Index target_sequence_length = dataset.get_maximum_target_sequence_length();

    const Shape input_shape = dataset.get_shape("Input");
    const Shape target_shape = dataset.get_shape("Target");

    ASSERT_EQ(input_shape.rank, 1);
    EXPECT_EQ(input_shape[0], input_sequence_length);

    ASSERT_EQ(target_shape.rank, 1);
    EXPECT_EQ(target_shape[0], target_sequence_length);

    const MatrixR& data = dataset.get_data();
    ASSERT_EQ(data.rows(), 4);
    ASSERT_EQ(data.cols(), input_sequence_length + target_sequence_length);

    for (Index i = 0; i < data.rows(); ++i)
        EXPECT_FLOAT_EQ(data(i, 0), LanguageDataset::START_INDEX);

    remove_language_file(file_path);
}


TEST(LanguageDataset, ReadTxtTargetDistribution)
{
    const string file_path = temp_language_file("opennn_language_distribution.txt", sentiment_content);

    LanguageDataset dataset;
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_separator(Dataset::Separator::Tab);
    dataset.set_has_header(false);
    dataset.set_display(false);
    dataset.set_data_path(file_path);

    ASSERT_NO_THROW(dataset.read_txt());

    const VectorI distribution = dataset.calculate_target_distribution();

    ASSERT_EQ(distribution.size(), 2);
    EXPECT_EQ(distribution(0), 2);
    EXPECT_EQ(distribution(1), 2);

    remove_language_file(file_path);
}


TEST(LanguageDataset, ConstructorWithPathReadsFile)
{
    const string file_path = temp_language_file("opennn_language_ctor.txt", sentiment_content);

    LanguageDataset dataset(file_path);

    EXPECT_EQ(dataset.get_samples_number(), 4);
    EXPECT_GT(dataset.get_input_vocabulary_size(), Index(LanguageDataset::reserved_tokens.size()));
    EXPECT_EQ(dataset.get_target_vocabulary_size(), 6);

    remove_language_file(file_path);
}
