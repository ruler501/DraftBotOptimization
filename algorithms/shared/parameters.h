//
// Created by Devon Richards on 9/7/2020.
//
#ifndef DRAFTBOTOPTIMIZATION_PARAMETERS_H
#define DRAFTBOTOPTIMIZATION_PARAMETERS_H
#include "../../draftbot_optimization.h"

// Genetic Algorithm Hyperparameters
constexpr float WEIGHT_VOLATILITY = (MAX_WEIGHT - MIN_WEIGHT) / 20;
constexpr float CLIP_VOLATILITY = 1 / 20.f;
constexpr float RATING_VOLATILITY = MAX_SCORE / 40;
constexpr float MULTIPLIER_VOLATILITY = 1 / 20.f;
constexpr size_t WEIGHT_INV_PROB_TO_CHANGE = 30;
constexpr size_t CLIP_INV_PROB_TO_CHANGE = 20;
constexpr size_t RATING_INV_PROB_TO_CHANGE = 5;
constexpr size_t MULTIPLIER_INV_PROB_TO_CHANGE = 20;
constexpr size_t NUM_INITIAL_MUTATIONS = 20;

// Cross-Entropy Method Hyperparameters
constexpr float INITIAL_WEIGHT_MEAN = (MAX_WEIGHT - MIN_WEIGHT) / 2 + MIN_WEIGHT;
constexpr float INITIAL_WEIGHT_STDDEV = (MAX_WEIGHT - MIN_WEIGHT) / 4;
constexpr float INITIAL_RATING_MEAN = MAX_SCORE / 2;
constexpr float INITIAL_RATING_STDDEV = MAX_SCORE / 4;
constexpr float INITIAL_UNIT_MEAN = 0.5f;
constexpr float INITIAL_UNIT_STDDEV = 0.25f;

// Differential Evolution Hyperparameters
constexpr float CROSSOVER_RATE = 0.9f;
constexpr float DIFFERENTIAL_VOLATILITY = 0.8f;

// CMA-ES Hyperparameters
constexpr float LEARNING_RATE = 1.f;
constexpr float INITIAL_MEAN = 0;
constexpr float INITIAL_STD_DEV = MAX_SCORE * 0.3f;
constexpr float ALPHA_COVARIANCE = 200.f;
#endif //DRAFTBOTOPTIMIZATION_PARAMETERS_H
