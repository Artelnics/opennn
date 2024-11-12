#ifndef DATASETBATCH_H
#define DATASETBATCH_H

#include "data_set.h"

// Cuda includes
#include "tensors.h"
#include "image_data_set.h"
#include "images.h"
#include "language_data_set.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct Batch
{
    Batch(const Index& = 0, DataSet* = nullptr);

    vector<pair<type*, dimensions>> get_input_pairs() const;

    pair<type*, dimensions> get_targets_pair() const;

    Index get_batch_samples_number() const;

    void set(const Index& = 0, DataSet* = nullptr);

    void fill(const vector<Index>&, 
              const vector<Index>&, 
              const vector<Index>&, 
              const vector<Index>& = vector<Index>());

    Tensor<type, 2> perform_augmentation(const Tensor<type, 2>&);

    void print() const;

    bool is_empty() const;

    bool has_context() const;

    Index batch_size = 0;

    DataSet* data_set = nullptr;

    dimensions input_dimensions;

    Tensor<type, 1> input_tensor;

    dimensions targets_dimensions;

    Tensor<type, 1> target_tensor;

    dimensions context_dimensions;

    Tensor<type, 1> context_tensor;

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;
};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/batch_cuda.h"
#endif

}
#endif
