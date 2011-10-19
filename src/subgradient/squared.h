#ifndef REGRESSION_SUBGRADIENT_SQUARED_H_
#define REGRESSION_SUBGRADIENT_SQUARED_H_

#include <iostream>
#include <vector>

#include <tool/calc.h>

namespace regression {
namespace subgradient {
class SubgradientSquared {
 public:
  explicit SubgradientSquared(double eta = 1.0);
  ~SubgradientSquared() {};

  void Train(const datum& datum);
  void Train(const std::vector<datum>& data,
             const size_t iteration = 1);
  double Test(const feature_vector& fv) const;
  double GetFeatureWeight(size_t feature_id) const;

 private:
  double CalcScore(const feature_vector& fv) const;

  void Update(const datum& datum,
              double score);

  weight_vector weight_;
  size_t dataN_;
  double eta_;
};

} //namespace subgradient
} //namespace regression

#endif //REGRESSION_SUBGRADIENT_SQUARED_H_
