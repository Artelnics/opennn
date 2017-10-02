/******************************************************************************/
/*                                                                            */
/*   OpenNN: Open Neural Networks Library                                     */
/*   www.opennn.net                                                           */
/*                                                                            */
/*   Russell Standish                                                         */
/******************************************************************************/

#include "multiObjective.h"
using namespace OpenNN;
using namespace std;

void MultiObjective::start_loss_calculation(const Matrix<double> &population)
{
  vector<vector<double>> obj;
  obj.reserve(population.get_rows_number());
  vector<double> reg;
  for (size_t i=0; i<population.get_rows_number(); ++i)
    {
      auto individual=population.arrange_row(i);
      obj.push_back(objectives(individual));
      reg.push_back(calculate_regularization(individual));
    }

  // compute maximum and minimum objective functions, to allow
  // discrimination between individuals with the same dominance
  vector<double> maxObj(num_objectives()),
    minObj(num_objectives(),numeric_limits<double>::max());
  for (auto& i: obj)
    for (size_t j=0; j<i.size() && j<num_objectives(); ++j)
      {
        maxObj[j]=max(maxObj[j], i[j]);
        minObj[j]=min(minObj[j], i[j]);
      }

  // compute dominance for Pareto multiobjective optimisation 
  loss.clear();
  loss.resize(population.get_rows_number());
  for (auto& i: obj)
    for (size_t j=0; j<obj.size(); ++j)
      {
        bool dominates=true;
        for (size_t k=0; k<num_objectives(); ++k)
          dominates &= i[k] < obj[j][k];
        loss[j] += dominates * num_objectives();
      }

  // invert range outside of the loop
  vector<double> invRange;
  for (size_t i=0; i<num_objectives(); ++i)
    if (maxObj[i]=minObj[i])
      invRange.push_back(0);
    else
      invRange.push_back(1/(maxObj[i]-minObj[i]));
  
  // now mixin individual objectives to discriminate between
  // individuals of the same dominance, and add the regularisation term
  for (size_t j=0; j<obj.size(); ++j)
    {
      for (size_t k=0; k<num_objectives(); ++k)
        loss[j]+=invRange[k]*(obj[j][k]-minObj[k]);
      loss[j]+=reg[j];
    }
}
