#ifndef YOLO_ERROR_H
#define YOLO_ERROR_H

#include "loss_index.h"
#include "data_set.h"


namespace opennn
{

class YoloError : public LossIndex
{
// protected:
//     struct YOLO_anchor_box
//     {
//         type width;
//         type height;

//         YOLO_anchor_box() : width(0), height(0){}
//         YOLO_anchor_box(type w, type h) : width(w), height(h){}
//     };

public:
    // Constructors

    explicit YoloError();

    explicit YoloError(NeuralNetwork*, DataSet*);

    // Error

    void calculate_error(const Batch&,
                         const ForwardPropagation&,
                         BackPropagation&) const final;

    void calculate_binary_error(const Batch&,
                                const ForwardPropagation&,
                                BackPropagation&) const;

    void calculate_multiple_error(const Batch&,
                                  const ForwardPropagation&,
                                  BackPropagation&) const;

    // Gradient

    void calculate_output_delta(const Batch&,
                                ForwardPropagation&,
                                BackPropagation&) const final;

    void calculate_binary_output_delta(const Batch&,
                                       ForwardPropagation&,
                                       BackPropagation&) const;

    void calculate_multiple_output_delta(const Batch&,
                                         ForwardPropagation&,
                                         BackPropagation&) const;

    string get_loss_method() const final;
    string get_error_type_text() const final;

    // type loss_function(const Tensor<type, 2>&, const vector<YOLO_anchor_box>&, Tensor<type, 3>&);

    // Serialization

    virtual void from_XML(const tinyxml2::XMLDocument&);

    void to_XML(tinyxml2::XMLPrinter&) const final;

    // type loss_function(const Tensor<type, 2>& labels, const vector<YOLO_anchor_box> anchors, Tensor<type, 3> network_output);

};


}

#endif // YOLO_ERROR_H
