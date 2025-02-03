#ifndef YOLO_ERROR_H
#define YOLO_ERROR_H

#include "loss_index.h"
#include "data_set.h"


namespace opennn
{

class YoloError : public LossIndex
{

public:
    // Constructors

    explicit YoloError();

    explicit YoloError(NeuralNetwork*, DataSet*);

    // Error

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const final;

    // Gradient

    void calculate_output_delta(const Batch&,
                                ForwardPropagation&,
                                BackPropagation&) const final;

    type calculate_intersection_over_union_deltas(const Tensor<type, 1>&, const Tensor<type, 1>&, type&, type&, type&, type&) const;


    string get_loss_method() const final;
    string get_error_type_text() const final;

    // Serialization

    virtual void from_XML(const tinyxml2::XMLDocument&);

    void to_XML(tinyxml2::XMLPrinter&) const final;

};


}

#endif // YOLO_ERROR_H
