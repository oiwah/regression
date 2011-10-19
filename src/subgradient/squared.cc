#include <subgradient/squared.h>

#include <cmath>
#include <algorithm>

namespace regression {
namespace subgradient {
SubgradientSquared::SubgradientSquared(double eta) : dataN_(0), eta_(eta) {
  weight_vector().swap(weight_);
}

void SubgradientSquared::Train(const datum& datum) {
  ++dataN_;
  double score = CalcScore(datum.fv);

  Update(datum, score);
}

void SubgradientSquared::Train(const std::vector<datum>& data,
                               const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (std::vector<datum>::const_iterator it = data.begin();
         it != data.end();
         ++it) {
      Train(*it);
    }
  }
}

double SubgradientSquared::Test(const feature_vector& fv) const {
  return CalcScore(fv);
}

double SubgradientSquared::CalcScore(const feature_vector& fv) const {
  return InnerProduct(fv, weight_);
}

void SubgradientSquared::Update(const datum& datum,
                                double score) {
  double step_distance = eta_ / std::sqrt(dataN_);
  double loss = 2.0 * (datum.output - score);

  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (weight_.size() <= it->first)
      weight_.resize(it->first + 1, 1.0);
    weight_[it->first] += step_distance * loss * it->second;
  }
}

double SubgradientSquared::GetFeatureWeight(size_t feature_id) const {
  return ReturnFeatureWeight(feature_id, weight_);
}

} //namespace subgradient
} //namespace regression
