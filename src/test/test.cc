#include <subgradient/squared.h>
#include <dual_averaging/da.h>

#include <test/test.h>

namespace {
} //namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " [training file] [test file]" << std::endl;
    return -1;
  }

  std::vector<regression::datum> train;
  if (!ParseFile(argv[1], &train))
    return -1;

  std::vector<regression::datum> test;
  if (!ParseFile(argv[2], &test))
    return -1;

  regression::subgradient::SubgradientSquared sgh(0.01);
  if (regression::Run(sgh, "SubgradientSquared", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  regression::dual_averaging::DualAveraging da(1000.0);
  if (regression::Run(da, "DualAveraging", train, test) == -1) {
    std::cerr << "error occurring!" << std::endl;
  }

  return 0;
}
