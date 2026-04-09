//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B A T C H   S T R U C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "batch.h"

namespace opennn
{

void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices,
                 bool augment)
{
    dataset->fill_inputs(sample_indices, input_indices, input.data());

    if(augment)
        dataset->augment_inputs(input.data(), sample_indices.size());

    if(!decoder_shape.empty())
        dataset->fill_decoder(sample_indices, decoder_indices, decoder.data());

    dataset->fill_targets(sample_indices, target_indices, target.data());
}

Batch::Batch(const Index new_samples_number, const Dataset* new_dataset)
{
    set(new_samples_number, new_dataset);
}

void Batch::set(const Index new_samples_number, const Dataset* new_dataset)
{
    if(!new_dataset) return;

    samples_number = new_samples_number;

    dataset = new_dataset;

    // Input

    const Shape& dataset_input_shape = dataset->get_shape("Input");

    if(!dataset_input_shape.empty())
    {
        input_shape = Shape({samples_number}).append(dataset_input_shape);
        input.resize(input_shape.size());
    }

    // Target

    const Shape& dataset_target_shape = dataset->get_shape("Target");

    if(!dataset_target_shape.empty())
    {
        target_shape = Shape({samples_number}).append(dataset_target_shape);
        target.resize(target_shape.size());
    }

    // Decoder

    const Shape& dataset_decoder_shape = dataset->get_shape("Decoder");

    if(!dataset_decoder_shape.empty())
    {
        decoder_shape = Shape({samples_number}).append(dataset_decoder_shape);
        decoder.resize(decoder_shape.size());
    }
}

Index Batch::get_samples_number() const
{
    return samples_number;
}

void Batch::print() const
{
    cout << "Batch" << endl
         << "Inputs:" << endl
         << "Input shape:" << input_shape << endl;

    if (input_shape.rank == 4)
        cout << TensorMap4(const_cast<type*>(input.data()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2],
                           input_shape[3]);
    else if (input_shape.rank == 3)
        cout << TensorMap3(const_cast<type*>(input.data()),
                           input_shape[0],
                           input_shape[1],
                           input_shape[2]);
    else if (input_shape.rank == 2)
        cout << MatrixMap(const_cast<type*>(input.data()),
                          input_shape[0],
                          input_shape[1]);

    cout << endl;

    if(!decoder_shape.empty())
    {
        cout << "Decoder:" << endl
             << "Decoder shape:" << decoder_shape << endl;
    }

    cout << "Targets:" << endl
         << "Target shape:" << target_shape << endl;

    cout << MatrixMap(const_cast<type*>(target.data()),
                      target_shape[0],
                      target_shape[1]) << endl;
}

bool Batch::is_empty() const
{
    return input.size() == 0;
}

vector<TensorView> Batch::get_inputs() const
{
    vector<TensorView> input_views = {{const_cast<type*>(input.data()), input_shape}};

    if(!decoder_shape.empty())
        input_views.insert(input_views.begin(), {const_cast<type*>(decoder.data()), decoder_shape});

    return input_views;
}

TensorView Batch::get_targets() const
{
    return {const_cast<type*>(target.data()), target_shape};
}

#ifdef CUDA

// @todo BatchCuda to be replaced by unified Batch with Memory

#endif

}
