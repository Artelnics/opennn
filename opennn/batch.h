#ifndef DATASETBATCH_H
#define DATASETBATCH_H

#include "data_set.h"
#include "batch.h"
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
    Batch() {}

    Batch(const Index&, DataSet*);

    virtual ~Batch();

    Tensor<pair<type*, dimensions>, 1> get_inputs_pair() const;

    pair<type*, dimensions> get_targets_pair() const;

    Index get_batch_samples_number() const;

    void set(const Index&, DataSet*);

    void fill(const Tensor<Index, 1>&, 
              const Tensor<Index, 1>&, 
              const Tensor<Index, 1>&, 
              const Tensor<Index, 1>& = Tensor<Index, 1>());

    void perform_augmentation() const;

    void print() const;

    Index batch_size = 0;

    DataSet* data_set = nullptr;

    dimensions input_dimensions;

    Tensor<type, 1> inputs_tensor;

    type* input_data = nullptr;

    dimensions targets_dimensions;

    Tensor<type, 1> targets_tensor;

    type* targets_data = nullptr;

    dimensions context_dimensions;

    Tensor<type, 1> context_tensor;

    type* context_data = nullptr;

    bool has_context = false;

    ThreadPool* thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/batch_cuda.h"
#endif

}
#endif
