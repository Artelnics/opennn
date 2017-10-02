/******************************************************************************/
/*                                                                            */
/*   OpenNN: Open Neural Networks Library                                     */
/*   www.opennn.net                                                           */
/*                                                                            */
/*   Russell Standish                                                         */
/******************************************************************************/

#ifndef MULTIOBJECTIVE_H
#define MULTIOBJECTIVE_H
#include "loss_index.h"
#include <vector>

namespace OpenNN
{
  class MultiObjective: public LossIndex
  {
    std::vector<double> loss;
  public:
    void 	start_loss_calculation(const OpenNN::Matrix<double>&) override;
    double calculate_loss_for_individual (size_t i) const override
    {return loss[i];}
    /// number of objectives
    virtual size_t num_objectives() const=0;
    /// evaluate the objectives for individual desribed by \a parameters
    /// @return vector of length num_objectives()
    virtual std::vector<double> objectives(OpenNN::Vector<double>& parameters)=0;
  };
}

#endif
