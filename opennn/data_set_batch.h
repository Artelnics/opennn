#ifndef DATASETBATCH_H
#define DATASETBATCH_H

#include <string>

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


    pair<type*, dimensions> get_inputs() const
    {
        pair<type*, dimensions> inputs;

        inputs.first = inputs_data;
        inputs.second = inputs_dimensions;

        return inputs;
    }

    Index get_batch_samples_number() const;

    void set(const Index&, DataSet*);

    void fill(const Tensor<Index, 1>&, const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    void perform_augmentation();

    void print() const;

    Index batch_size = 0;

    DataSet* data_set_pointer = nullptr;

    dimensions inputs_dimensions;

    Tensor<type, 1> inputs_tensor;

    type* inputs_data = nullptr;

    Tensor<type, 2> targets;
};

}
#endif
