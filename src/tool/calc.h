#ifndef REGRESSION_TOOL_CALC_H_
#define REGRESSION_TOOL_CALC_H_

#include <tool/feature.h>
#include <tool/weight.h>

namespace regression {
inline double InnerProduct(const feature_vector& fv,
                           const weight_vector& wv) {
  double score = 0.0;
  for (feature_vector::const_iterator it = fv.begin();
       it != fv.end();
       ++it) {
    if (wv.size() <= it->first) continue;
    score += wv[it->first] * it->second;
  }
  return score;
}

inline double ReturnFeatureWeight(size_t feature_id,
                                  const weight_vector& wv) {
  if (feature_id >= wv.size()) {
    return 0.0;
  }
  return wv[feature_id];
}

} //namespace regression

#endif //REGRESSION_TOOL_CALC_H_
