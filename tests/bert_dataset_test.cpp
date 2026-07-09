#include "pch.h"
#include "numerical_derivatives.h"

#include <cmath>

#include "opennn/bert_dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/neural_network.h"
#include "opennn/forward_propagation.h"
#include "opennn/loss.h"
#include "opennn/configuration.h"

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
        filesystem::remove(text_path + ".bert" + to_string(seq) + ".csv", error);
    }
}


TEST(BertDatasetTest, TokenizesAndWiresRoles)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const string vocab_path = write_lines("opennn_bertds_vocab.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text.txt",  labelled_text);

    const Index seq = 8;
    BertDataset dataset(text_path, vocab_path, seq);

    EXPECT_EQ(dataset.get_sequence_length(), seq);
    EXPECT_EQ(dataset.get_samples_number(), Index(labelled_text.size()));

    EXPECT_EQ(dataset.get_features_number("Decoder"), seq);   // input_ids  -> word_embeddings
    EXPECT_EQ(dataset.get_features_number("Input"), seq);     // token_type -> token_type_embeddings
    EXPECT_GE(dataset.get_features_number("Target"), 1);

    const MatrixR& data = dataset.get_data();
    EXPECT_FLOAT_EQ(data(0, 0), 2.0f);           // [CLS]
    EXPECT_FLOAT_EQ(data(0, 1), 4.0f);           // "good"
    EXPECT_FLOAT_EQ(data(0, 2), 5.0f);           // "movie"
    EXPECT_FLOAT_EQ(data(0, 3), 3.0f);           // [SEP]
    EXPECT_FLOAT_EQ(data(0, 4), 0.0f);           // [PAD]
    EXPECT_FLOAT_EQ(data(0, seq), 1.0f);         // tt_0 = segment 0 (+1)

    clean_up(vocab_path, text_path, seq);
    Configuration::instance().set();
}


TEST(BertDatasetTest, FeedsBertClassifierForward)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const string vocab_path = write_lines("opennn_bertds_vocab2.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text2.txt",  labelled_text);

    const Index seq = 8;
    BertDataset dataset(text_path, vocab_path, seq);

    const Index batch  = dataset.get_samples_number();
    const Index labels = dataset.get_features_number("Target");
    const MatrixR& data = dataset.get_data();

    std::vector<float> input_ids(size_t(batch * seq));
    std::vector<float> token_type(size_t(batch * seq));
    for (Index b = 0; b < batch; ++b)
        for (Index s = 0; s < seq; ++s)
        {
            input_ids[size_t(b * seq + s)]  = data(b, s);          // cols 0..seq-1
            token_type[size_t(b * seq + s)] = data(b, seq + s);    // cols seq..2seq-1
        }

    BertForSequenceClassification model(seq, Index(bert_vocabulary.size()),
                                        /*hidden*/ 8, /*heads*/ 2, /*intermediate*/ 16,
                                        /*layers*/ 1, labels);
    model.set_parameters_random();

    ForwardPropagation forward_propagation(batch, &model);
    std::vector<TensorView> inputs = {
        TensorView(input_ids.data(),  {batch, seq}),
        TensorView(token_type.data(), {batch, seq})
    };
    model.forward_propagate(inputs, forward_propagation, false);

    const TensorView output = forward_propagation.get_outputs();
    ASSERT_EQ(output.shape.rank, 2);
    EXPECT_EQ(output.shape[0], batch);
    EXPECT_EQ(output.shape[1], labels);

    const float* values = output.as<float>();
    for (Index i = 0; i < output.size(); ++i)
        EXPECT_TRUE(std::isfinite(values[i])) << "non-finite output at " << i;

    clean_up(vocab_path, text_path, seq);
    Configuration::instance().set();
}


TEST(BertDatasetTest, BertClassifierGradientOnCpu)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const string vocab_path = write_lines("opennn_bertds_vocab3.txt", bert_vocabulary);
    const string text_path  = write_lines("opennn_bertds_text3.txt",  labelled_text);

    const Index seq = 8;
    BertDataset dataset(text_path, vocab_path, seq);

    const Index labels = dataset.get_features_number("Target");

    BertForSequenceClassification model(seq, Index(bert_vocabulary.size()),
                                        /*hidden*/ 8, /*heads*/ 2, /*intermediate*/ 16,
                                        /*layers*/ 1, labels);
    model.set_parameters_random();

    Loss loss(&model, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    const VectorR gradient = calculate_gradient(loss);
    ASSERT_GT(gradient.size(), 0);
    EXPECT_TRUE(gradient.allFinite());

    clean_up(vocab_path, text_path, seq);
    Configuration::instance().set();
}
