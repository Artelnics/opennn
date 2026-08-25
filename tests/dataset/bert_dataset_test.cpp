#include "tests/pch.h"
#include "tests/numerical_derivatives.h"

#include <cmath>

#include "opennn/dataset/bert_dataset.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/training_strategy/loss.h"
#include "opennn/core/configuration.h"
#include "opennn/testing_analysis/testing_analysis.h"

using namespace opennn;

namespace
{
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
    }
}

TEST(BertDatasetTest, TokenizesAndWiresRoles)
{
    const string vocab_path = write_lines("opennn_bertds_vocab.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text.txt",  labelled_text);

    const Index seq = 8;
    BertDataset dataset(text_path, vocab_path, seq);

    EXPECT_EQ(dataset.get_sequence_length(), seq);
    EXPECT_EQ(dataset.get_samples_number(), Index(labelled_text.size()));

    EXPECT_EQ(dataset.get_features_number("Decoder"), seq);
    EXPECT_EQ(dataset.get_features_number("Input"), seq);
    EXPECT_GE(dataset.get_features_number("Target"), 1);

    const MatrixR& data = dataset.get_data();
    EXPECT_FLOAT_EQ(data(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(data(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(data(0, 2), 5.0f);
    EXPECT_FLOAT_EQ(data(0, 3), 3.0f);
    EXPECT_FLOAT_EQ(data(0, 4), 0.0f);
    EXPECT_FLOAT_EQ(data(0, seq), 1.0f);

    BertDataset cached_dataset(text_path, vocab_path, seq);
    EXPECT_TRUE(cached_dataset.get_data().isApprox(data));

    clean_up(vocab_path, text_path, seq);
}

TEST(BertDatasetTest, FeedsBertClassifierForward)
{
    const string vocab_path = write_lines("opennn_bertds_vocab2.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text2.txt",  labelled_text);

    const Index seq = 8;
    BertDataset dataset(text_path, vocab_path, seq);

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

TEST(BertDatasetTest, BertClassifierGradientOnCpu)
{
    const string vocab_path = write_lines("opennn_bertds_vocab3.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text3.txt",  labelled_text);

    const Index seq = 8;
    BertDataset dataset(text_path, vocab_path, seq);

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

TEST(BertDatasetTest, TestingAnalysisSupportsMultipleInputs)
{
    const string vocab_path = write_lines("opennn_bertds_vocab4.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text4.txt", labelled_text);

    const Index seq = 8;
    BertDataset dataset(text_path, vocab_path, seq);
    dataset.set_sample_roles("Testing");

    const Index labels = dataset.get_features_number("Target");
    BertForSequenceClassification model(
        seq, Index(bert_vocabulary.size()), 8, 2, 16, 1, labels);
    model.set_parameters_random();

    TestingAnalysis testing_analysis(&model, &dataset);
    testing_analysis.set_batch_size(3);

    const MatrixI confusion = testing_analysis.calculate_confusion();

    EXPECT_EQ(confusion.rows(), 3);
    EXPECT_EQ(confusion.cols(), 3);
    EXPECT_EQ(confusion.bottomRightCorner(1, 1)(0, 0), dataset.get_samples_number());

    clean_up(vocab_path, text_path, seq);
}
