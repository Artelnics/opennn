#ifndef DATASETBATCH_H
#define DATASETBATCH_H

#include <string>
#include "dynamic_tensor.h"
#include "data_set.h"

using namespace std;
using namespace Eigen;

namespace opennn
{

struct DataSetBatch
{
    /// Default constructor.

    DataSetBatch() {}

    DataSetBatch(const Index&, DataSet*);

    /// Destructor.

    virtual ~DataSetBatch()
    {
    }

    Index get_batch_samples_number() const;

    void set(const Index&, DataSet*);
/*
    void set_inputs(const Tensor<DynamicTensor<type>, 1>& new_inputs)
    {
        inputs = new_inputs;
    }

    void set_inputs(const DynamicTensor<type>& new_inputs)
    {
        inputs.resize(1);
        inputs(0) = new_inputs;
    }

    void set_inputs(Tensor<type, 2>& new_inputs)
    {
        inputs.resize(1);
        inputs(0) = DynamicTensor<type>(new_inputs);
    }

    void set_inputs(Tensor<type, 4>& new_inputs)
    {
        inputs.resize(1);
        inputs(0) = DynamicTensor<type>(new_inputs);
    }
*/
    void fill(const Tensor<Index, 1>&, const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    void perform_augmentation();

    void print() const;

    Index batch_size = 0;

    DataSet* data_set_pointer = nullptr;

    Tensor<DynamicTensor<type>, 1> inputs;

    DynamicTensor<type> targets;
};



}
#endif
