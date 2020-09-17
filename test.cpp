#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "draftbot_optimization.h"
#include "algorithms/shared/matrix_types.h"
#include "algorithms/shared/parameters.h"
#include "algorithms/cma_es.cpp"

void save_variables(const Variables&, const std::string&) {}

std::array<std::array<double, 4>, POPULATION_SIZE> run_simulations(const std::vector<Variables>&, const std::vector<Pick>&, float, const std::shared_ptr<const Constants>&) {
    return {};
}

int main(int argc, char* argv[]) {
    std::random_device rd;
    size_t seed = rd();
    const float temperature = 1.f;
    const size_t num_generations = 2;
    const std::shared_ptr<const Constants> constants = std::make_shared<Constants>();
    const std::shared_ptr<const Variables> variables = std::make_shared<Variables>();
    std::vector<Pick> picks(4000);
    const Variables output = optimize_variables(temperature, picks, num_generations, constants, variables, seed);
}