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

#ifdef OPENNN_CUDA

    struct BatchCuda
    {
        explicit BatchCuda();
        BatchCuda(const Index&, DataSet*);
        virtual ~BatchCuda();

        void set(const Index&, DataSet*);

        void copy_device();

        Tensor<type, 2> get_inputs_device() const;

        Tensor<pair<type*, dimensions>, 1> get_inputs_pair_device() const;

        Tensor<type, 2> get_targets_device() const;

        void allocate();

        void free();

        void fill(const Tensor<Index, 1>&,
            const Tensor<Index, 1>&,
            const Tensor<Index, 1>&,
            const Tensor<Index, 1>&);

        Index get_batch_samples_number() const;

        void print();

        DataSet* data_set = nullptr;
        Index batch_size = 0;

        dimensions input_dimensions;
        dimensions target_dimensions;
        dimensions context_dimensions;

        float* inputs_host = nullptr;
        float* targets_host = nullptr;
        float* context_host = nullptr;

        float* inputs_device = nullptr;
        float* targets_device = nullptr;
        float* context_device = nullptr;

        bool has_context = false;
    };

#endif

}
#endif
