#include <dual_averaging/da.h>

#include <cmath>
#include <algorithm>

namespace regression {
namespace dual_averaging {
DualAveraging::DualAveraging(double gamma) : dataN_(0), gamma_(gamma) {
  weight_vector().swap(weight_);
  weight_vector().swap(subgradient_sum_);
}

void DualAveraging::Train(const datum& datum,
                          bool primal) {
  CalcWeight(datum.fv);
  ++dataN_;

  double score = CalcScore(datum.fv);
  Update(datum, score);

  if (primal)
    CalcWeightAll();
}

void DualAveraging::Train(const std::vector<datum>& data,
                          const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (std::vector<datum>::const_iterator it = data.begin();
         it != data.end();
         ++it) {
      Train(*it, false);
    }
  }
  CalcWeightAll();
}

double DualAveraging::Test(const feature_vector& fv) const {
  return CalcScore(fv);
}

void DualAveraging::CalcWeight(const feature_vector& fv) {
  if (dataN_ == 0) return;
  double scalar = - sqrt(dataN_) / gamma_;

  for (feature_vector::const_iterator fv_it = fv.begin();
       fv_it != fv.end();
       ++fv_it) {
    size_t index = fv_it->first;

    if (subgradient_sum_.size() <= index)
        subgradient_sum_.resize(index, 0.0);
    if (weight_.size() <= index)
        weight_.resize(index + 1, 0.0);

    weight_[index] = scalar * subgradient_sum_[index];
  }
}

void DualAveraging::CalcWeightAll() {
  double scalar = - sqrt(dataN_) / gamma_;

  if (weight_.size() < subgradient_sum_.size())
      weight_.resize(subgradient_sum_.size(), 0.0);
  for (size_t feature_id = 0; feature_id < subgradient_sum_.size(); ++feature_id) {
    weight_[feature_id] = scalar * subgradient_sum_[feature_id];
  }
}

double DualAveraging::CalcScore(const feature_vector& fv) const {
  return InnerProduct(fv, weight_);
}

void DualAveraging::Update(const datum& datum,
                           double score) {
  double loss = 2.0 * (datum.output - score);

  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (subgradient_sum_.size() <= it->first)
      subgradient_sum_.resize(it->first + 1, 0.0);
    subgradient_sum_[it->first] -= loss * it->second;
  }
}

double DualAveraging::GetFeatureWeight(size_t feature_id) const {
  return ReturnFeatureWeight(feature_id, weight_);
}

} //namespace dual_averaging
} //namespace regression
