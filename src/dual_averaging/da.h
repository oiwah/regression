#ifndef REGRESSION_DUAL_AVERAGING_DA_H_
#define REGRESSION_DUAL_AVERAGING_DA_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace regression {
namespace dual_averaging {
class DualAveraging {
 public:
  explicit DualAveraging(double gamma = 1.0);
  ~DualAveraging() {};

  void Train(const datum& datum, bool primal = true);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  double Test(const feature_vector& fv) const;
  double GetFeatureWeight(size_t feature_id) const;

 private:
  void CalcWeight(const feature_vector& fv);
  void CalcWeightAll();

  double CalcScore(const feature_vector& fv) const;

  void Update(const datum& datum,
              double score);

  weight_vector weight_;
  weight_vector subgradient_sum_;

  size_t dataN_;
  double gamma_;
};

} //namespace dual_averaging
} //namespace regression

#endif //REGRESSION_DUAL_AVERAGING_DA_H_
