//
// Created by Devon Richards on 9/6/2020.
//
#ifndef DRAFTBOTOPTIMIZATION_UTIL_H
#define DRAFTBOTOPTIMIZATION_UTIL_H
#include <random>

#include "../../draftbot_optimization.h"

Variables mutate_variables(Variables& variables, std::mt19937_64& gen);
#endif //DRAFTBOTOPTIMIZATION_UTIL_H