#include <fstream>
#include <sstream>

#include <tool/feature.h>

namespace regression {
bool ParseFile(const char* file_path,
               std::vector<regression::datum>* data) {
  std::vector<regression::datum>(0).swap(*data);

  std::ifstream ifs(file_path);
  if (!ifs) {
    std::cerr << "cannot open " << file_path << std::endl;
    return false;
  }

  size_t lineN = 0;
  for (std::string line; getline(ifs, line); ++lineN) {
    datum datum;
    std::istringstream iss(line);

    double output = 0.0;
    if (!(iss >> output)) {
      std::cerr << "parse error: you must set output value in line " << lineN << std::endl;
      return false;
    }

    datum.output = output;

    size_t id = 0;
    char comma = 0;
    double value = 0.0;
    while (iss >> id >> comma >> value) {
      datum.fv.push_back(std::make_pair(id, value));
    }
    data->push_back(datum);
  }

  return true;
}

template <class T>
int Run (T& regression,
         const char* regression_name,
         const std::vector<regression::datum>& train,
         const std::vector<regression::datum>& test) {
  std::cout << regression_name << std::endl;
  regression.Train(train, 100);

  double score = 0.0;
  for (size_t i = 0; i < test.size(); ++i) {
    double predict = regression.Test(test[i].fv);
    std::cout << predict << " : " << test[i].output << std::endl;
    score += (predict - test[i].output) * (predict - test[i].output);
  }

  std::cout << "squared loss : " << score << std::endl;
  std::cout << std::endl;
  return 0;
}
} //namespace
