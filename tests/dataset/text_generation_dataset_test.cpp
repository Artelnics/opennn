#include "tests/pch.h"

#include "opennn/dataset/text_generation_dataset.h"

using namespace opennn;

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

TEST(TextGenerationDataset, OwnsVocabularyThroughTokenizer)
{
    const filesystem::path path = write_text_generation_file(
        "opennn_text_generation_vocabulary.txt",
        "alpha beta alpha gamma beta delta alpha beta gamma delta");

    TextGenerationDataset dataset("", 3);
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_data_path(path);
    dataset.set_display(false);
    dataset.read_txt();

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

    remove_text_generation_files(path);
}

TEST(TextGenerationDataset, MatrixAndBinaryStorageProduceEqualBatches)
{
    const filesystem::path path = write_text_generation_file(
        "opennn_text_generation_storage.txt",
        "zero one two three four five six seven eight nine ten eleven");

    constexpr Index sequence_length = 3;

    TextGenerationDataset matrix_dataset("", sequence_length);
    matrix_dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    matrix_dataset.set_data_path(path);
    matrix_dataset.set_display(false);
    matrix_dataset.read_txt();

    TextGenerationDataset binary_dataset(path, sequence_length);
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

    TextGenerationDataset cached_dataset(path, sequence_length);
    EXPECT_EQ(cached_dataset.get_vocabulary(), binary_dataset.get_vocabulary());
    EXPECT_EQ(cached_dataset.get_samples_number(), binary_dataset.get_samples_number());

    remove_text_generation_files(path);
}

TEST(TextGenerationDataset, UsesLoadedBytePairTokenizer)
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

    TextGenerationDataset dataset("", 2);
    dataset.set_storage_mode(Dataset::StorageMode::Matrix);
    dataset.set_tokenizer(
        make_unique<BytePairTokenizer>(vocabulary_path, merges_path));
    dataset.set_data_path(text_path);
    dataset.set_display(false);
    dataset.read_txt();

    EXPECT_EQ(dataset.get_vocabulary_size(), 4);
    EXPECT_EQ(dataset.get_vocabulary()[3], "ab");
    EXPECT_GT(dataset.get_samples_number(), 0);

    error_code error;
    filesystem::remove_all(directory, error);
}

TEST(TextGenerationDataset, CacheIdentityIncludesSequenceConfiguration)
{
    const filesystem::path path = write_text_generation_file(
        "opennn_text_generation_cache.txt",
        "a b c d e f g h i j k l m n o p q r s t");

    TextGenerationDataset short_sequences(path, 2);
    TextGenerationDataset long_sequences(path, 3);

    const filesystem::path cache_directory = path.string() + ".cache";
    Index cache_files = 0;
    for (const filesystem::directory_entry& entry :
         filesystem::directory_iterator(cache_directory))
        if (entry.is_regular_file()) ++cache_files;

    EXPECT_GE(cache_files, 2);

    remove_text_generation_files(path);
}
