#include "pch.h"

#include "opennn/text_dataset.h"

using namespace opennn;

namespace
{
string write_text_corpus(const string& name, const string& content)
{
    const string path = (filesystem::temp_directory_path() / name).string();
    ofstream file(path);
    file << content;
    return path;
}

void remove_text_corpus(const string& path)
{
    error_code error;
    filesystem::remove(path, error);
    filesystem::remove_all(path + ".cache", error);
}

string write_vocabulary(const string& name, const vector<string>& tokens)
{
    const string path = (filesystem::temp_directory_path() / name).string();
    ofstream file(path);
    for (const string& token : tokens) file << token << '\n';
    return path;
}
}

TEST(TextDataset, CausalCorpusBuildsShiftedBlocks)
{
    const string path = write_text_corpus(
        "opennn_text_dataset_causal.txt",
        "alpha beta gamma delta epsilon zeta eta theta");

    TextDataset dataset("", 3);
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_data_path(path);

    ASSERT_NO_THROW(dataset.read_txt());

    EXPECT_EQ(dataset.get_task(), TextDataset::Task::CausalLanguageModel);
    EXPECT_EQ(dataset.get_sequence_length(), 3);
    EXPECT_EQ(dataset.get_samples_number(), 2);

    const MatrixR& data = dataset.get_data();
    ASSERT_EQ(data.rows(), 2);
    ASSERT_EQ(data.cols(), 6);

    EXPECT_FLOAT_EQ(data(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(data(0, 1), 3.0f);
    EXPECT_FLOAT_EQ(data(0, 2), 7.0f);
    EXPECT_FLOAT_EQ(data(0, 3), 3.0f);
    EXPECT_FLOAT_EQ(data(0, 4), 7.0f);
    EXPECT_FLOAT_EQ(data(0, 5), 4.0f);

    remove_text_corpus(path);
}

TEST(TextDataset, CausalCorpusFactoryPreservesLegacyContract)
{
    const string path = write_text_corpus(
        "opennn_text_dataset_factory.txt",
        "alpha beta gamma delta epsilon zeta eta theta");

    unique_ptr<TextDataset> dataset = TextDataset::from_causal_corpus(path, 3);

    EXPECT_EQ(dataset->get_sequence_length(), 3);
    EXPECT_EQ(dataset->get_samples_number(), 2);
    EXPECT_EQ(dataset->get_shape("Input")[0], 3);
    EXPECT_EQ(dataset->get_shape("Target")[0], 3);
    EXPECT_EQ(dataset->get_vocabulary().front(), "[PAD]");
    EXPECT_EQ(dataset->get_vocabulary()[1], "[UNK]");

    remove_text_corpus(path);
}

TEST(TextDataset, ClassificationFactoryUsesUnifiedType)
{
    const string path = write_text_corpus(
        "opennn_text_dataset_classification.txt",
        "excellent phone\tpositive\n"
        "broken phone\tnegative\n"
        "excellent camera\tpositive\n"
        "broken camera\tnegative\n");

    unique_ptr<TextDataset> dataset = TextDataset::from_classification(path);

    EXPECT_EQ(dataset->get_task(), TextDataset::Task::Classification);
    EXPECT_EQ(dataset->get_samples_number(), 4);
    EXPECT_EQ(dataset->get_shape("Decoder").rank, 0);
    EXPECT_EQ(dataset->get_shape("Target")[0], 1);
    EXPECT_EQ(dataset->get_target_vocabulary().size(), 6);

    remove_text_corpus(path);
}

TEST(TextDataset, SequenceToSequenceFactoryUsesUnifiedType)
{
    const string path = write_text_corpus(
        "opennn_text_dataset_seq2seq.txt",
        "hello world\thola mundo\n"
        "good morning\tbuenos dias\n");

    unique_ptr<TextDataset> dataset = TextDataset::from_sequence_to_sequence(path);

    EXPECT_EQ(dataset->get_task(), TextDataset::Task::SequenceToSequence);
    EXPECT_EQ(dataset->get_samples_number(), 2);
    EXPECT_EQ(dataset->get_shape("Input").rank, 1);
    EXPECT_EQ(dataset->get_shape("Decoder").rank, 1);
    EXPECT_EQ(dataset->get_shape("Target").rank, 1);

    remove_text_corpus(path);
}

TEST(TextDataset, BertFactoryBuildsIdsMasksAndLabelsWithoutCsv)
{
    const string text_path = write_text_corpus(
        "opennn_text_dataset_bert.txt",
        "good movie\tpositive\n"
        "bad movie\tnegative\n");
    const string vocabulary_path = write_vocabulary(
        "opennn_text_dataset_bert_vocab.txt",
        {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "good", "bad", "movie"});

    unique_ptr<TextDataset> dataset =
        TextDataset::from_bert_classification(text_path, vocabulary_path, 6);

    EXPECT_EQ(dataset->get_task(), TextDataset::Task::BertClassification);
    EXPECT_EQ(dataset->get_samples_number(), 2);
    EXPECT_EQ(dataset->get_shape("Decoder")[0], 6);
    EXPECT_EQ(dataset->get_shape("Input")[0], 6);
    EXPECT_EQ(dataset->get_shape("Target")[0], 1);
    EXPECT_EQ(dataset->get_input_vocabulary().size(), 7);
    EXPECT_EQ(dataset->get_target_vocabulary().size(), 2);
    EXPECT_FALSE(filesystem::exists(text_path + ".bert6.csv"));

    remove_text_corpus(text_path);
    error_code error;
    filesystem::remove(vocabulary_path, error);
}
