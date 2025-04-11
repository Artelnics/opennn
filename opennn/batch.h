#ifndef BATCH_H
#define BATCH_H

#include "data_set.h"

namespace opennn
{

struct Batch
{
    Batch(const Index& = 0, DataSet* = nullptr);

    vector<pair<type*, dimensions>> get_input_pairs() const;
    pair<type*, dimensions> get_target_pair() const;

    Index get_samples_number() const;

    void set(const Index& = 0, DataSet* = nullptr);

    void fill(const vector<Index>&, 
              const vector<Index>&, 
              const vector<Index>&, 
              const vector<Index>& = vector<Index>());

    Tensor<type, 2> perform_augmentation(const Tensor<type, 2>&);

    void print() const;

    bool is_empty() const;

    Index samples_number = 0;

    DataSet* data_set = nullptr;

    dimensions input_dimensions;
    Tensor<type, 1> input_tensor;

    dimensions decoder_dimensions;
    Tensor<type, 1> decoder_tensor;

    dimensions target_dimensions;
    Tensor<type, 1> target_tensor;

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;
};

#ifdef OPENNN_CUDA_test

    struct BatchCuda
    {
        BatchCuda(const Index& = 0, DataSet* = nullptr);

        vector<pair<type*, dimensions>> get_input_pairs_device() const;
        pair<type*, dimensions> get_target_pair_device() const;

        Index get_samples_number() const;

        Tensor<type, 2> get_inputs_device() const;
        Tensor<type, 2> get_decoder_device() const;
        Tensor<type, 2> get_targets_device() const;

        void set(const Index&, DataSet*);

        void copy_device();

        void free();

        void fill(const vector<Index>&,
                  const vector<Index>&,
                  const vector<Index>&,
                  const vector<Index> & = vector<Index>());

        void print() const;

        bool is_empty() const;

        Index samples_number = 0;

        DataSet* data_set = nullptr;

        dimensions input_dimensions;
        dimensions decoder_dimensions;
        dimensions target_dimensions;

        float* inputs_host = nullptr;
        float* decoder_host = nullptr;
        float* targets_host = nullptr;

        float* inputs_device = nullptr;
        float* decoder_device = nullptr;
        float* targets_device = nullptr;
    };

#endif

}
#endif
