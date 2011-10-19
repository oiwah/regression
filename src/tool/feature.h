#ifndef REGRESSION_TOOL_FEATURE_H_
#define REGRESSION_TOOL_FEATURE_H_

#include <cfloat>
#include <unordered_map>

namespace regression {
typedef std::vector<std::pair<size_t, double> > feature_vector;
typedef std::unordered_map<std::string, size_t> feature2id;

struct datum {
  double output;
  feature_vector fv;
};

} //namespace regression

#endif //REGRESSION_TOOL_FEATURE_H_
