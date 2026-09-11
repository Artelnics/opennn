#pragma once

#include "opennn/dataset/dataset.h"
#include "opennn/core/json.h"
#include <cstring>
#include <fstream>
#include <numeric>

namespace quality
{
using namespace opennn;

inline Shape shape(const Json& dimensions, bool omit_batch = false)
{
    vector<Index> values;
    const auto& array = dimensions.as_array();
    for (size_t i = omit_batch ? 1 : 0; i < array.size(); ++i)
        values.push_back(array[i].as_long());
    return Shape(values.begin(), values.end());
}

// Read each batch from shared prepared files rather than decode data differently
// in the two libraries. Each worker owns its stream. Large image data stays on disk.
class Dataset final : public opennn::Dataset
{
public:
    Dataset(const filesystem::path& root, const Json& specification)
        : root(root), specification(specification)
    {
        input_shape = shape(specification.at("x").at("shape"), true);
        target_shape = shape(specification.at("y").at("shape"), true);
        if (specification.has("decoder"))
            decoder_shape = shape(specification.at("decoder").at("shape"), true);
        sample_roles.assign(specification.at("x").at("shape").as_array()[0].as_long(), SampleRole::Training);
        variables.emplace_back("input", "Input", VariableType::Numeric, "None");
        variables.emplace_back("target", "Target", VariableType::Numeric, "None");
        if (!decoder_shape.empty())
            variables.emplace_back("decoder", "Decoder", VariableType::Numeric, "None");
    }

    void from_JSON(const JsonDocument&) override { throw runtime_error("Use the quality data manifest"); }
    bool supports_bf16_inputs() const override { return decoder_shape.empty(); }
    FeatureScaling prepare_training_scaling(VariableRole, const FeatureScaling& requested, Index) override
    { return requested; } // prepared tensors; retain ResNet's fixed /255 scaling

    vector<float> read(const string& field, const vector<Index>& indices) const
    {
        const auto& spec = specification.at(field);
        const Index width = shape(spec.at("shape"), true).size();
        const bool bytes = spec.at("dtype").as_string() == "uint8";
        vector<float> values(indices.size() * size_t(width));
        ifstream stream(root / spec.at("file").as_string(), ios::binary);
        if (!stream) throw runtime_error("Cannot open quality tensor");
        vector<unsigned char> temporary(bytes ? size_t(width) : 0);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            stream.seekg(indices[i] * width * (bytes ? 1 : sizeof(float)));
            if (bytes)
            {
                stream.read(reinterpret_cast<char*>(temporary.data()), width);
                for (Index j = 0; j < width; ++j) values[i * width + j] = temporary[j];
            }
            else stream.read(reinterpret_cast<char*>(values.data() + i * width), width * sizeof(float));
            if (!stream) throw runtime_error("Truncated quality tensor");
        }
        return values;
    }

    void fill_inputs(const vector<Index>& indices, const vector<Index>&, float* output,
                     FillMode, ColumnContiguity) const override { fill("x", indices, output); }
    void fill_targets(const vector<Index>& indices, const vector<Index>&, float* output,
                      FillMode, ColumnContiguity) const override { fill("y", indices, output); }
    void fill_decoder(const vector<Index>& indices, const vector<Index>&, float* output,
                      FillMode, ColumnContiguity) const override { fill("decoder", indices, output); }

private:
    void fill(const string& field, const vector<Index>& indices, float* output) const
    {
        const auto values = read(field, indices);
        std::copy(values.begin(), values.end(), output);
    }
    filesystem::path root;
    Json specification;
};
}
