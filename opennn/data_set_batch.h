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

    virtual ~DataSetBatch();

    pair<type *, dimensions> get_inputs_pair() const;

    pair<type *, dimensions> get_targets_pair() const;

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

    dimensions targets_dimensions;

    Tensor<type, 1> targets_tensor;

    type* targets_data = nullptr;
};

}
#endif
