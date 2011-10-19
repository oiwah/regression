#ifndef REGRESSION_TOOL_FEATURE_H_
#define REGRESSION_TOOL_FEATURE_H_

#include <cfloat>
#include <unordered_map>

namespace regression {
typedef std::vector<std::pair<size_t, double> > feature_vector;

struct datum {
  double output;
  feature_vector fv;
};

} //namespace regression

#endif //REGRESSION_TOOL_FEATURE_H_
